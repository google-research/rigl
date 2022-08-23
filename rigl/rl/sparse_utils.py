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

"""Defines pruning and sparse training utilities."""

import functools
import re

import gin
from rigl import sparse_optimizers_base as sparse_opt_base
from rigl import sparse_utils
from rigl.rigl_tf2 import init_utils
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


PRUNING_WRAPPER = pruning_wrapper.PruneLowMagnitude
PRUNED_LAYER_TYPES = (tf.keras.layers.Conv2D, tf.keras.layers.Dense)


def get_total_params(model):
  """Obtains total active parameters of a given network."""
  all_layers = get_all_layers(model)
  total_count = 0.
  nparams_dict = {}
  for layer in all_layers:
    n_param = 0.
    if isinstance(layer, PRUNING_WRAPPER):
      mask = layer.pruning_vars[0][1]
      n_param += tf.reduce_sum(mask)
      n_param += tf.size(layer.weights[1], out_type=tf.float32)
    else:
      for w in layer.weights:
        n_param += tf.size(w, out_type=tf.float32)
    nparams_dict[layer.name] = n_param
    total_count += n_param
  return total_count, nparams_dict


@gin.configurable(denylist=['layer_dict'])
def get_pruning_sparsities(
    layer_dict,
    mask_init_method='erdos_renyi_kernel',
    target_sparsity=0.9,
    erk_power_scale=1.,
    custom_sparsity_map=None):
  """Creates name/sparsity dict using the name/shapes dict (layer_dict)."""
  if target_sparsity == 0:
    return {k: 0 for k in layer_dict.keys()}

  if custom_sparsity_map is None:
    custom_sparsity_map = {}
  extract_name_fn = lambda x: re.findall('(.+):0', x)[0]
  dummy_masks_dict = {k: tf.ones(v) for k, v in layer_dict.items()}
  reverse_dict = {v.name: k
                  for k, v in dummy_masks_dict.items()}

  sparsity_dict = sparse_utils.get_sparsities(
      list(dummy_masks_dict.values()),
      mask_init_method,
      target_sparsity,
      custom_sparsity_map,
      extract_name_fn=extract_name_fn,
      erk_power_scale=erk_power_scale)
  renamed_sparsity_dict = {reverse_dict[k]: float(v)
                           for k, v in sparsity_dict.items()}
  return renamed_sparsity_dict


@gin.configurable('pruning')
def get_pruning_params(mode,
                       initial_sparsity=0.0,
                       final_sparsity=0.95,
                       begin_step=30000,
                       end_step=100000,
                       frequency=1000):
  """Gets pruning hyper-parameters."""
  p_params = {}
  if mode == 'prune':
    p_params['pruning_schedule'] = pruning_schedule.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step,
        frequency=frequency)
  elif mode in ('rigl', 'static', 'set'):
    # For sparse training methods we don't use the pruning library to update the
    # masks. Therefore we need to disable it. Following `pruning` flags serve
    # that purpose.
    # 1B. High begin_step, so it never starts.
    p_params['pruning_schedule'] = pruning_schedule.ConstantSparsity(
        target_sparsity=0, begin_step=1000000000)
  else:
    raise ValueError('Mode: %s, is not valid' % mode)
  return p_params


def maybe_prune_layer(layer, params, filter_fn=None):
  if filter_fn is None:
    filter_fn = lambda l: isinstance(l, PRUNED_LAYER_TYPES)
  if filter_fn(layer):
    return PRUNING_WRAPPER(layer, **params)
  return layer


def get_wrap_fn(mode):
  """Creates a function that wraps a given layer conditionally.

  Args:
    mode: str, If 'dense' no modification done. Otherwise the layer is pruned.

  Returns:
    function that accepts layer and returns a possibly wrapped one.
  """
  if mode == 'dense':
    # Do not wrap the layer.
    wrap_fn = lambda x: x
  else:
    wrap_fn = functools.partial(
        maybe_prune_layer, params=get_pruning_params(mode))
  return wrap_fn


def update_prune_step(model, step):
  """Updates the pruning steps of each pruning layer."""
  assign_ops = []
  for layer in get_all_pruning_layers(model):
    # Assign iteration count to the layer pruning_step.
    # pruning wrapper requires step to be >0.
    assign_op = tf1.assign(layer.pruning_step, tf.maximum(step, 1))
    assign_ops.append(assign_op)
  return tf.group(assign_ops)


def update_prune_masks(model):
  """Updates the masks if it is an update iteration."""
  update_ops = [op for op in model.updates
                if 'prune_low_magnitude' in op.name]
  return tf.group(update_ops)


def get_all_layers(model, filter_fn=lambda _: True):
  """Gets all layers of a model and layers of a layer if it is a keras.Model."""
  all_layers = []
  for l in model.layers:
    if hasattr(l, 'layers'):
      all_layers.extend(get_all_layers(l, filter_fn=filter_fn))
    elif filter_fn(l):
      all_layers.append(l)
  return all_layers


def get_all_variables_and_masks(model):
  """Gets all trainable variables (+their masks) of a model."""
  all_layers = get_all_layers(model)
  all_variables = []
  for l in all_layers:
    all_variables.extend(l.trainable_variables)
    if isinstance(l, PRUNING_WRAPPER):
      all_variables.append(l.pruning_vars[0][1])  # Adding mask.
  return all_variables


def get_all_pruning_layers(model):
  """Gets all pruned layers of a model and layers of a layer if keras.Model."""
  return get_all_layers(
      model, filter_fn=lambda l: isinstance(l, PRUNING_WRAPPER))


def log_sparsities(model):
  for layer in get_all_pruning_layers(model):
    for _, mask, threshold in layer.pruning_vars:
      scalar_name = f'sparsity/{mask.name}'
      sparsity = 1 - tf.reduce_mean(mask)
      if len(mask.shape) == 2:
        reshaped_mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)
        tf1.summary.image(f'img/{mask.name}', reshaped_mask)
      tf1.summary.scalar(scalar_name, sparsity)
      tf1.summary.scalar(f'threshold/{threshold.name}', threshold)


class SparseOptTf2Mixin:
  """Tf2 model_optimization pruning library specific variable retrieval."""

  def compute_gradients(self, *args, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    return self._optimizer.compute_gradients(*args, **kwargs)

  def set_model(self, model):
    self.model = model

  def get_weights(self):
    all_weights = [
        layer.pruning_vars[0][0] for layer in get_all_pruning_layers(self.model)
    ]
    return all_weights

  def get_masks(self):
    all_masks = [
        layer.pruning_vars[0][1] for layer in get_all_pruning_layers(self.model)
    ]
    return all_masks

  def get_masked_weights(self):
    all_masked_weights = [
        w * m for w, m in zip(self.get_weights(), self.get_masks())
    ]
    return all_masked_weights


@gin.configurable()
class UpdatedSETOptimizer(SparseOptTf2Mixin,
                          sparse_opt_base.SparseSETOptimizerBase):

  def _before_apply_gradients(self, grads_and_vars):
    return tf1.no_op()


@gin.configurable()
class UpdatedRigLOptimizer(SparseOptTf2Mixin,
                           sparse_opt_base.SparseRigLOptimizerBase):

  def _before_apply_gradients(self, grads_and_vars):
    """Updates momentum before updating the weights with gradient."""
    self._weight2masked_grads = {w.name: g for g, w in grads_and_vars}
    return tf1.no_op()


@gin.configurable()
def init_masks(model,
               mask_init_method='random',
               sparsity=0.9,
               erk_power_scale=1.,
               custom_sparsity_map=None,
               fixed_sparse_init=False):
  """Inits the masks randomly according to the given sparsity."""
  if sparsity == 0:
    return None

  if custom_sparsity_map is None:
    custom_sparsity_map = {}
  all_masks = [
      layer.pruning_vars[0][1] for layer in get_all_pruning_layers(model)
  ]

  assigner = sparse_utils.get_mask_init_fn(
      all_masks,
      mask_init_method,
      sparsity,
      custom_sparsity_map,
      erk_power_scale=erk_power_scale)
  if fixed_sparse_init:
    all_weights = [
        layer.pruning_vars[0][0] for layer in get_all_pruning_layers(model)
    ]
    with tf.control_dependencies([assigner]):
      assign_ops = []
      for param, mask in zip(all_weights, all_masks):
        new_init = init_utils.unit_scaled_init_tf1(mask)
        assign_ops.append(tf1.assign(param, new_init))
      assigner = tf.group(assign_ops)
  return assigner
