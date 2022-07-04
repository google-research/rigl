# coding=utf-8
# Copyright 2022 RigL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module implements some common and new sparse training algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
from rigl import sparse_optimizers_base as sparse_opt_base
from rigl import sparse_utils


from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import training_util


class PruningGetterTf1Mixin:
  """Tf1 model_pruning library specific variable retrieval."""

  def get_weights(self):
    return pruning.get_weights()

  def get_masks(self):
    return pruning.get_masks()

  def get_masked_weights(self):
    return pruning.get_masked_weights()


class SparseSETOptimizer(PruningGetterTf1Mixin,
                         sparse_opt_base.SparseSETOptimizerBase):
  pass


class SparseRigLOptimizer(PruningGetterTf1Mixin,
                          sparse_opt_base.SparseRigLOptimizerBase):
  pass


class SparseStaticOptimizer(SparseSETOptimizer):
  """Sparse optimizer that re-initializes weak connections during training.

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    momentum: float, for the exponentialy moving average.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               name='SparseStaticOptimizer',
               stateless_seed_offset=0):
    super(SparseStaticOptimizer, self).__init__(
        optimizer,
        begin_step,
        end_step,
        frequency,
        drop_fraction=drop_fraction,
        drop_fraction_anneal=drop_fraction_anneal,
        grow_init=grow_init,
        use_locking=use_locking,
        name=name,
        stateless_seed_offset=stateless_seed_offset)

  def generic_mask_update(self, mask, weights, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    masked_weights = mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=hash(weights.name + 'drop'))
    # Revive n_prune many connections using momentum.
    score_grow = mask
    return self._get_update_op(
        score_drop, score_grow, mask, weights, reinit_when_same=True)


class SparseMomentumOptimizer(SparseSETOptimizer):
  """Sparse optimizer that grows connections with the expected gradients.

  A simplified implementation of Momentum based sparse optimizer. No
  redistribution of sparsity.
  Original implementation:
  https://github.com/TimDettmers/sparse_learning/blob/master/mnist_cifar/main.py

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    momentum: float, for the exponentialy moving average.
    use_tpu: bool, if true the masked_gradients are aggregated.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               momentum=0.9,
               use_tpu=False,
               name='SparseMomentumOptimizer',
               stateless_seed_offset=0):
    super(SparseMomentumOptimizer, self).__init__(
        optimizer,
        begin_step,
        end_step,
        frequency,
        drop_fraction=drop_fraction,
        drop_fraction_anneal=drop_fraction_anneal,
        grow_init=grow_init,
        use_locking=use_locking,
        name='SparseMomentumOptimizer',
        stateless_seed_offset=stateless_seed_offset)
    self._ema_grads = moving_averages.ExponentialMovingAverage(decay=momentum)
    self._use_tpu = use_tpu

  def set_masked_grads(self, grads, weights):
    if self._use_tpu:
      grads = [tpu_ops.cross_replica_sum(g) for g in grads]
    self._masked_grads = grads
    # Using names since better to hash.
    self._weight2masked_grads = {w.name: m for w, m in zip(weights, grads)}

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    grads_and_vars = self._optimizer.compute_gradients(loss, **kwargs)
    # Need to update the EMA of the masked_weights. This is a bit hacky and
    # might not work as expected if the gradients are not applied after every
    # calculation. However, it should be fine if only .minimize() call is used.
    masked_grads_vars = self._optimizer.compute_gradients(
        loss, var_list=self.get_masked_weights())
    masked_grads = [g for g, _ in masked_grads_vars]
    self.set_masked_grads(masked_grads, self.get_weights())
    return grads_and_vars

  def _before_apply_gradients(self, grads_and_vars):
    """Updates momentum before updating the weights with gradient."""
    return self._ema_grads.apply(self._masked_grads)

  def generic_mask_update(self, mask, weights, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    casted_mask = math_ops.cast(mask, dtypes.float32)
    masked_weights = casted_mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=hash(weights.name + 'drop'))
    # Revive n_prune many connections using momentum.
    masked_grad = self._weight2masked_grads[weights.name]
    score_grow = math_ops.abs(self._ema_grads.average(masked_grad))
    return self._get_update_op(score_drop, score_grow, mask, weights)


class SparseSnipOptimizer(tf_optimizer.Optimizer):
  """Implementation of dynamic sparsity optimizers.

  Implementation of Snip
  https://arxiv.org/abs/1810.02340

  Attributes:
    optimizer: tf.train.Optimizer
    default_sparsity: float, between 0 and 1.
    mask_init_method: str, used to determine mask initializations.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    use_locking: bool, passed to the super.
    use_tpu: bool, if true the masked_gradients are aggregated.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               default_sparsity,
               mask_init_method,
               custom_sparsity_map=None,
               use_locking=False,
               use_tpu=False,
               name='SparseSnipOptimizer'):
    super(SparseSnipOptimizer, self).__init__(use_locking, name)
    if not custom_sparsity_map:
      custom_sparsity_map = {}
    self._optimizer = optimizer
    self._use_tpu = use_tpu
    self._default_sparsity = default_sparsity
    self._mask_init_method = mask_init_method
    self._custom_sparsity_map = custom_sparsity_map
    self.is_snipped = variable_scope.get_variable(
        'is_snipped', initializer=lambda: False, trainable=False)

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    return self._optimizer.compute_gradients(loss, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """

    def apply_gradient_op():
      return self._optimizer.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)

    maybe_reduce = lambda x: x
    if self._use_tpu:
      maybe_reduce = tpu_ops.cross_replica_sum
    grads_and_vars_dict = {
        re.findall('(.+)/weights:0', var.name)[0]: (maybe_reduce(grad), var)
        for grad, var in grads_and_vars
        if var.name.endswith('weights:0')
    }

    def snip_fn(mask, sparsity, dtype):
      """Creates a random sparse mask with deterministic sparsity.

      Args:
        mask: tf.Tensor, used to obtain correct corresponding gradient.
        sparsity: float, between 0 and 1.
        dtype: tf.dtype, type of the return value.

      Returns:
        tf.Tensor
      """
      del dtype
      var_name = sparse_utils.mask_extract_name_fn(mask.name)
      g, v = grads_and_vars_dict[var_name]
      score_drop = math_ops.abs(g * v)
      n_total = np.prod(score_drop.shape.as_list())
      n_prune = sparse_utils.get_n_zeros(n_total, sparsity)
      n_keep = n_total - n_prune

      # Sort the entire array since the k needs to be constant for TPU.
      _, sorted_indices = nn_ops.top_k(
          array_ops.reshape(score_drop, [-1]), k=n_total)
      sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
      # We will have zeros after having `n_keep` many ones.
      new_values = array_ops.where(
          math_ops.range(n_total) < n_keep,
          array_ops.ones_like(sorted_indices, dtype=mask.dtype),
          array_ops.zeros_like(sorted_indices, dtype=mask.dtype))
      new_mask = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                      new_values.shape)
      return array_ops.reshape(new_mask, mask.shape)

    def snip_op():
      all_masks = pruning.get_masks()
      assigner = sparse_utils.get_mask_init_fn(
          all_masks,
          self._mask_init_method,
          self._default_sparsity,
          self._custom_sparsity_map,
          mask_fn=snip_fn)
      with ops.control_dependencies([assigner]):
        assign_op = state_ops.assign(
            self.is_snipped, True, name='assign_true_after_snipped')
      return assign_op

    maybe_snip_op = control_flow_ops.cond(
        math_ops.logical_and(
            math_ops.equal(global_step, 0),
            math_ops.logical_not(self.is_snipped)), snip_op, apply_gradient_op)

    return maybe_snip_op


class SparseDNWOptimizer(tf_optimizer.Optimizer):
  """Implementation of DNW optimizer.

  Implementation of DNW.
  See https://arxiv.org/pdf/1906.00586.pdf
  This optimizer ensures the mask is updated at every iteration, according to
  the current set of weights. It uses dense gradient to update weights.

  Attributes:
    optimizer: tf.train.Optimizer
    default_sparsity: float, between 0 and 1.
    mask_init_method: str, used to determine mask initializations.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    use_tpu: bool, if true the masked_gradients are aggregated.
    use_locking: bool, passed to the super.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               default_sparsity,
               mask_init_method,
               custom_sparsity_map=None,
               use_tpu=False,
               use_locking=False,
               name='SparseDNWOptimizer'):
    super(SparseDNWOptimizer, self).__init__(use_locking, name)
    self._optimizer = optimizer
    self._use_tpu = use_tpu
    self._default_sparsity = default_sparsity
    self._mask_init_method = mask_init_method
    self._custom_sparsity_map = custom_sparsity_map

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    # Replace masked variables with masked_weights so that the gradient is dense
    # and not masked
    if var_list is None:
      var_list = (
          variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    var_list = self.replace_with_masked_weights(var_list)
    grads_and_vars = self._optimizer.compute_gradients(
        loss, var_list=var_list, **kwargs)
    return self.replace_masked_weights(grads_and_vars)

  def replace_with_masked_weights(self, var_list):
    """Replaces masked variables with masked weights."""
    weight2masked_weights = {
        w.name: mw
        for w, mw in zip(self.get_weights(), self.get_masked_weights())
    }
    updated_var_list = [weight2masked_weights.get(w.name, w) for w in var_list]
    return updated_var_list

  def replace_masked_weights(self, grads_and_vars):
    """Replaces masked weight tensords with weight variables."""
    masked_weights2weight = {
        mw.name: w
        for w, mw in zip(self.get_weights(), self.get_masked_weights())
    }
    updated_grads_and_vars = [
        (g, masked_weights2weight.get(w.name, w)) for g, w in grads_and_vars
    ]
    return updated_grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    optimizer_update = self._optimizer.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)
    vars_dict = {
        re.findall('(.+)/weights:0', var.name)[0]: var
        for var in self.get_weights()
    }

    def dnw_fn(mask, sparsity, dtype):
      """Creates a mask with smallest magnitudes with deterministic sparsity.

      Args:
        mask: tf.Tensor, used to obtain correct corresponding gradient.
        sparsity: float, between 0 and 1.
        dtype: tf.dtype, type of the return value.

      Returns:
        tf.Tensor
      """
      del dtype
      var_name = sparse_utils.mask_extract_name_fn(mask.name)
      v = vars_dict[var_name]
      score_drop = math_ops.abs(v)
      n_total = np.prod(score_drop.shape.as_list())
      n_prune = sparse_utils.get_n_zeros(n_total, sparsity)
      n_keep = n_total - n_prune

      # Sort the entire array since the k needs to be constant for TPU.
      _, sorted_indices = nn_ops.top_k(
          array_ops.reshape(score_drop, [-1]), k=n_total)
      sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
      # We will have zeros after having `n_keep` many ones.
      new_values = array_ops.where(
          math_ops.range(n_total) < n_keep,
          array_ops.ones_like(sorted_indices, dtype=mask.dtype),
          array_ops.zeros_like(sorted_indices, dtype=mask.dtype))
      new_mask = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                      new_values.shape)
      return array_ops.reshape(new_mask, mask.shape)

    with ops.control_dependencies([optimizer_update]):
      all_masks = self.get_masks()
      mask_update_op = sparse_utils.get_mask_init_fn(
          all_masks,
          self._mask_init_method,
          self._default_sparsity,
          self._custom_sparsity_map,
          mask_fn=dnw_fn)

    return mask_update_op

  def get_weights(self):
    return pruning.get_weights()

  def get_masks(self):
    return pruning.get_masks()

  def get_masked_weights(self):
    return pruning.get_masked_weights()
