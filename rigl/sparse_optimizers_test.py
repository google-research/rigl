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

"""Tests for the sparse_optimizers file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl import flags
from absl.testing import parameterized
import numpy as np
from rigl import sparse_optimizers
from rigl import sparse_utils
import tensorflow.compat.v1 as tf  # tf

from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers


FLAGS = flags.FLAGS


class SparseSETOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_graph(self, n_inp, n_out, drop_frac, start_iter=1, end_iter=4,
                   freq_iter=2):
    """Setups a trivial training procedure for sparse training."""
    tf.reset_default_graph()
    optim = tf.train.GradientDescentOptimizer(0.1)
    sparse_optim = sparse_optimizers.SparseSETOptimizer(
        optim, start_iter, end_iter, freq_iter, drop_fraction=drop_frac)
    x = tf.random.uniform((1, n_inp))
    y = layers.masked_fully_connected(x, n_out, activation_fn=None)
    global_step = tf.train.get_or_create_global_step()
    weight = pruning.get_weights()[0]
    # There is one masked layer to be trained.
    mask = pruning.get_masks()[0]
    # Around half of the values of the mask is set to zero with `mask_update`.
    mask_update = tf.assign(
        mask,
        tf.constant(
            np.random.choice([0, 1], size=(n_inp, n_out), p=[1./2, 1./2]),
            dtype=tf.float32))
    loss = tf.reduce_mean(y)
    global_step = tf.train.get_or_create_global_step()
    train_op = sparse_optim.minimize(loss, global_step)

    # Init
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run([mask_update])

    return sess, train_op, mask, weight, global_step

  @parameterized.parameters((15, 25, 0.5), (15, 25, 0.2), (3, 5, 0.2))
  def testMaskNonUpdateIterations(self, n_inp, n_out, drop_frac):
    """Training a layer for 5 iterations and see whether mask is kept intact.

    The mask should be updated only in iterations 1 and 3 (since start_iter=1,
    end_iter=4, freq_iter=2).

    Args:
      n_inp: int, number of input channels.
      n_out: int, number of output channels
      drop_frac: float, passed to the sparse optimizer.
    """
    sess, train_op, mask, _, _ = self._setup_graph(
        n_inp, n_out, drop_frac, start_iter=1, end_iter=4, freq_iter=2)
    expected_updates = [1, 3]
    # Running 5 times to make sure the mask is not updated after end_iter.
    for i in range(1, 6):
      c_mask, = sess.run([mask])
      sess.run([train_op])
      c_mask2, = sess.run([mask])
      if i not in expected_updates:
        self.assertAllEqual(c_mask, c_mask2)

  @parameterized.parameters((15, 25, 0.5), (15, 25, 0.7), (30, 10, 0.9))
  def testUpdateIterations(self, n_inp, n_out, drop_frac):
    """Checking whether the mask is updated during correct iterations.

    The mask should be updated only in iterations 1 and 3 (since start_iter=1,
    end_iter=4, freq_iter=2). Number of 1's in the mask should be equal.

    Args:
      n_inp: int, number of input channels.
      n_out: int, number of output channels
      drop_frac: float, passed to the sparse optimizer.
    """
    sess, train_op, mask, _, _ = self._setup_graph(
        n_inp, n_out, drop_frac, start_iter=1, end_iter=4, freq_iter=2)
    expected_updates = [1, 3]
    # Running 4 times since last update is at 3.
    for i in range(1, 5):
      c_mask, = sess.run([mask])
      sess.run([train_op])
      c_mask2, = sess.run([mask])
      if i in expected_updates:
        # Number of ones (connections) should be same.
        self.assertEqual(c_mask.sum(), c_mask2.sum())
        # Assert there is some change in the mask.
        self.assertNotAllClose(c_mask, c_mask2)

  @parameterized.parameters((3, 7, 2), (1, 5, 3), (0, 4, 1))
  def testNoDrop(self, start_iter, end_iter, freq_iter):
    """Checks when the drop fraction is 0, no update is made.

    The mask should be updated only in iterations 1 and 3 (since start_iter=1,
    end_iter=4, freq_iter=2). Number of 1's in the mask should be equal.

    Args:
      start_iter: int, start iteration for sparse training.
      end_iter: int, final iteration for sparse training.
      freq_iter: int, mask update frequency.
    """
    # Setting drop_fraction to 0; so there is nothing dropped, nothing changed.
    sess, train_op, mask, _, _ = self._setup_graph(
        3, 5, 0, start_iter=start_iter, end_iter=end_iter, freq_iter=freq_iter)
    for _ in range(end_iter+2):
      c_mask, = sess.run([mask])
      sess.run([train_op])
      c_mask2, = sess.run([mask])
      self.assertAllEqual(c_mask, c_mask2)

  def testNewConnectionZeroInit(self):
    """Checks whether the new connections are initialized correctly to zeros.
    """
    end_iter = 4
    sess, train_op, mask, weight, _ = self._setup_graph(
        n_inp=3, n_out=5, drop_frac=0.5, start_iter=0, end_iter=end_iter,
        freq_iter=1)
    # Let's iterate until the mask updates are done.
    for _ in range(end_iter + 1):
      mask_tensor, = sess.run([mask])
      sess.run([train_op])
      new_mask_tensor, new_weight_tensor = sess.run([mask, weight])
      # Let's sum the values of the new connections
      new_weights = new_weight_tensor[np.logical_and(mask_tensor == 0,
                                                     new_mask_tensor == 1)]
      self.assertTrue(np.all(new_weights == 0))

  @parameterized.parameters(itertools.product(
      ((3, 7, 2), (5, 3), (1,)), ('zeros', 'random_normal', 'random_uniform')))
  def testShapeOfGetGrowTensor(self, shape, init_type):
    """Checks whether the new tensor is created with correct shape."""
    optim = tf.train.GradientDescentOptimizer(0.1)
    sparse_optim = sparse_optimizers.SparseSETOptimizer(optim, 0, 0, 1,
                                                        use_stateless=False)
    weights = tf.random_uniform(shape)
    grow_tensor = sparse_optim.get_grow_tensor(weights, init_type)
    self.assertAllEqual(weights.shape, grow_tensor.shape)

  @parameterized.parameters(itertools.product(
      (tf.float32, tf.float64),
      ('zeros', 'random_normal', 'random_uniform')))
  def testDtypeOfGetGrowTensor(self, dtype, init_type):
    """Checks whether the new tensor is created with correct data type."""
    optim = tf.train.GradientDescentOptimizer(0.1)
    sparse_optim = sparse_optimizers.SparseSETOptimizer(optim, 0, 0, 1,
                                                        use_stateless=False)
    weights = tf.random_uniform((3, 4), dtype=dtype, maxval=5)
    grow_tensor = sparse_optim.get_grow_tensor(weights, init_type)
    self.assertEqual(grow_tensor.dtype, weights.dtype)

  @parameterized.parameters('ones', 'zero', None, 0)
  def testValueErrorOfGetGrowTensor(self, method):
    """Checks whether the new tensor is created with correct shape and type."""
    optim = tf.train.GradientDescentOptimizer(0.1)
    sparse_optim = sparse_optimizers.SparseSETOptimizer(optim, 0, 0, 1,
                                                        use_stateless=False)
    weights = tf.random_uniform((3, 4))
    with self.assertRaises(ValueError):
      sparse_optim.get_grow_tensor(weights, method)


class SparseStaticOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_graph(self, n_inp, n_out, drop_frac, start_iter=1, end_iter=4,
                   freq_iter=2):
    """Setups a trivial training procedure for sparse training."""
    tf.reset_default_graph()
    optim = tf.train.GradientDescentOptimizer(0.1)
    sparse_optim = sparse_optimizers.SparseStaticOptimizer(
        optim, start_iter, end_iter, freq_iter, drop_fraction=drop_frac)
    x = tf.random.uniform((1, n_inp))
    y = layers.masked_fully_connected(x, n_out, activation_fn=None)
    global_step = tf.train.get_or_create_global_step()
    weight = pruning.get_weights()[0]
    # There is one masked layer to be trained.
    mask = pruning.get_masks()[0]
    # Around half of the values of the mask is set to zero with `mask_update`.
    mask_update = tf.assign(
        mask,
        tf.constant(
            np.random.choice([0, 1], size=(n_inp, n_out), p=[1./2, 1./2]),
            dtype=tf.float32))
    loss = tf.reduce_mean(y)
    global_step = tf.train.get_or_create_global_step()
    train_op = sparse_optim.minimize(loss, global_step)

    # Init
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run([mask_update])

    return sess, train_op, mask, weight, global_step

  @parameterized.parameters((15, 25, 0.5), (15, 25, 0.2), (3, 5, 0.2))
  def testMaskStatic(self, n_inp, n_out, drop_frac):
    """Training a layer for 5 iterations and see whether mask is kept intact.

    The mask should be updated only in iterations 1 and 3 (since start_iter=1,
    end_iter=4, freq_iter=2).

    Args:
      n_inp: int, number of input channels.
      n_out: int, number of output channels
      drop_frac: float, passed to the sparse optimizer.
    """
    sess, train_op, mask, _, _ = self._setup_graph(
        n_inp, n_out, drop_frac, start_iter=1, end_iter=4, freq_iter=2)
    # Running 5 times to make sure the mask is not updated after end_iter.
    for _ in range(5):
      c_mask, = sess.run([mask])
      sess.run([train_op])
      c_mask2, = sess.run([mask])
      self.assertAllEqual(c_mask, c_mask2)


class SparseMomentumOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_graph(self, n_inp, n_out, drop_frac, start_iter=1, end_iter=4,
                   freq_iter=2, momentum=0.5):
    """Setups a trivial training procedure for sparse training."""
    tf.reset_default_graph()
    optim = tf.train.GradientDescentOptimizer(0.1)
    sparse_optim = sparse_optimizers.SparseMomentumOptimizer(
        optim, start_iter, end_iter, freq_iter, drop_fraction=drop_frac,
        momentum=momentum)
    x = tf.ones((1, n_inp))
    y = layers.masked_fully_connected(x, n_out, activation_fn=None)
    # Multiplying the output with range of constants to have constant but
    # different gradients at the masked weights.
    y = y * tf.reshape(tf.cast(tf.range(tf.size(y)), dtype=y.dtype), y.shape)
    loss = tf.reduce_sum(y)
    global_step = tf.train.get_or_create_global_step()
    train_op = sparse_optim.minimize(loss, global_step)
    weight = pruning.get_weights()[0]
    masked_grad = sparse_optim._weight2masked_grads[weight.name]
    masked_grad_ema = sparse_optim._ema_grads.average(masked_grad)
    # Init
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess, train_op, masked_grad_ema

  @parameterized.parameters((3, 4, 0.5), (5, 2, 0.), (2, 5, 1.))
  def testMomentumUpdate(self, n_inp, n_out, momentum):
    """Checking whether momentum applied correctly."""
    sess, train_op, masked_grad_ema = self._setup_graph(
        n_inp, n_out, 0.5, start_iter=1, end_iter=4, freq_iter=2,
        momentum=momentum)

    # Running 6 times to make sure the momeuntum is always updated.
    current_momentum = np.zeros((n_inp, n_out))
    for _ in range(6):
      ema_masked_grad, = sess.run([masked_grad_ema])
      self.assertAllEqual(ema_masked_grad, current_momentum)
      sess.run([train_op])
      # This is since we multiply the output values with range(n_out)
      # Note the broadcast from n_out vector to (n_inp, n_out) matrix.
      current_momentum = (current_momentum * momentum +
                          (1 - momentum) * np.arange(n_out))

      ema_masked_grad, = sess.run([masked_grad_ema])
      self.assertAllEqual(ema_masked_grad, current_momentum)


class SparseRigLOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_graph(self, n_inp, n_out, drop_frac, start_iter=1, end_iter=4,
                   freq_iter=2):
    """Setups a trivial training procedure for sparse training."""
    tf.reset_default_graph()
    optim = tf.train.GradientDescentOptimizer(1e-3)
    global_step = tf.train.get_or_create_global_step()
    sparse_optim = sparse_optimizers.SparseRigLOptimizer(
        optim, start_iter, end_iter, freq_iter, drop_fraction=drop_frac)
    x = tf.ones((1, n_inp))
    y = layers.masked_fully_connected(x, n_out, activation_fn=None)
    # Multiplying the output with range of constants to have constant but
    # different gradients at the masked weights. We also multiply the loss with
    # global_step to increase the gradient linearly with time.
    scale_vector = (
        tf.reshape(tf.cast(tf.range(tf.size(y)), dtype=y.dtype), y.shape) *
        tf.cast(global_step, dtype=y.dtype))
    y = y * scale_vector
    loss = tf.reduce_sum(y)
    global_step = tf.train.get_or_create_global_step()
    train_op = sparse_optim.minimize(loss, global_step)
    weight = pruning.get_weights()[0]
    expected_gradient = tf.broadcast_to(scale_vector, weight.shape)
    masked_grad = sparse_optim._weight2masked_grads[weight.name]

    # Init
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess, train_op, masked_grad, expected_gradient

  @parameterized.parameters((3, 4), (5, 2), (2, 5))
  def testMaskedGradientCalculation(self, n_inp, n_out):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, train_op, masked_grad, expected_gradient = self._setup_graph(
        n_inp, n_out, 0., start_iter=0, end_iter=3, freq_iter=1)
    # Since we only update the mask every 2 iterations, we will iterate 6 times.

    for i in range(6):
      is_mask_update = i % 2 == 0
      if is_mask_update:
        expected_gradient_tensor, = sess.run([expected_gradient])
        _, masked_grad_tensor = sess.run([train_op, masked_grad])
        self.assertAllEqual(masked_grad_tensor,
                            expected_gradient_tensor)
      else:
        sess.run([train_op])

  @parameterized.parameters(
      (3, 7, 2, [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]),
      (1, 5, 3, [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]),
      (0, 4, 1, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]))
  def testApplyGradients(self, start_iter, end_iter, freq_iter, is_incremented):
    """Checking  apply_gradient is called in non mask update iterations."""
    sess, train_op, _, _ = self._setup_graph(
        3, 5, .5, start_iter=start_iter, end_iter=end_iter, freq_iter=freq_iter)
    global_step = tf.train.get_or_create_global_step()
    # Since we only update the mask every 2 iterations, we will iterate 6 times.
    for one_if_incremented in is_incremented:
      before, = sess.run([global_step])
      sess.run([train_op])
      after, = sess.run([global_step])
      if one_if_incremented == 1:
        self.assertEqual(before + 1, after)
      else:
        # Mask update step.
        self.assertEqual(before, after)


class SparseSnipOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_graph(self, default_sparsity, mask_init_method,
                   custom_sparsity_map, n_inp=3, n_out=5):
    """Setups a trivial training procedure for sparse training."""
    tf.reset_default_graph()
    optim = tf.train.GradientDescentOptimizer(1e-3)
    sparse_optim = sparse_optimizers.SparseSnipOptimizer(
        optim, default_sparsity, mask_init_method,
        custom_sparsity_map=custom_sparsity_map)

    inp_values = np.arange(1, n_inp+1)
    scale_vector_values = np.random.uniform(size=(n_out,)) - 0.5
    # The gradient is the outer product of input and the output gradients.
    # Since the loss is sample sum the output gradient is equal to the scale
    # vector.
    expected_grads = np.outer(inp_values, scale_vector_values)

    x = tf.reshape(tf.constant(inp_values, dtype=tf.float32), (1, n_inp))
    y = layers.masked_fully_connected(x, n_out, activation_fn=None)
    scale_vector = tf.constant(scale_vector_values, dtype=tf.float32)

    y = y * scale_vector
    loss = tf.reduce_sum(y)

    global_step = tf.train.get_or_create_global_step()
    train_op = sparse_optim.minimize(loss, global_step)

    # Init
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    mask = pruning.get_masks()[0]
    weights = pruning.get_weights()[0]
    return sess, train_op, expected_grads, sparse_optim, mask, weights

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testSnipSparsity(self, n_inp, n_out, default_sparsity):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, train_op, _, _, mask, _ = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    _ = sess.run([train_op])
    snipped_mask, = sess.run([mask])
    n_ones = np.sum(snipped_mask)
    n_zeros = snipped_mask.size - n_ones
    n_zeros_expected = sparse_utils.get_n_zeros(snipped_mask.size,
                                                default_sparsity)
    self.assertEqual(n_zeros, n_zeros_expected)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testGradientUsed(self, n_inp, n_out, default_sparsity):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, train_op, expected_grads, _, mask, weights = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    # Calculate sensitivity scores.
    weights, = sess.run([weights])
    expected_scores = np.abs(expected_grads*weights)
    _ = sess.run([train_op])
    snipped_mask, = sess.run([mask])
    kept_connection_scores = expected_scores[snipped_mask == 1]
    min_score_kept = np.min(kept_connection_scores)

    snipped_connection_scores = expected_scores[snipped_mask == 0]
    max_score_snipped = np.max(snipped_connection_scores)
    self.assertLessEqual(max_score_snipped, min_score_kept)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testInitialMaskIsDense(self, n_inp, n_out, default_sparsity):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, _, _, _, mask, _ = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    mask_start, = sess.run([mask])
    self.assertEqual(np.sum(mask_start), mask_start.size)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testAfterSnipTraining(self, n_inp, n_out, default_sparsity):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, train_op, _, sparse_optim, mask, _ = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    global_step = tf.train.get_or_create_global_step()
    is_snip_iter = sess.run([train_op])
    self.assertTrue(is_snip_iter)
    # On other iterations mask should stay same. Let's do 3 more iterations.
    for i in range(3):
      mask_before, c_iter = sess.run([mask, global_step])
      self.assertEqual(i, c_iter)
      is_snip_iter, is_snipped = sess.run([train_op, sparse_optim.is_snipped])
      self.assertTrue(is_snipped)
      self.assertFalse(is_snip_iter)
      mask_after, = sess.run([mask])
      self.assertAllEqual(mask_after, mask_before)


class SparseDNWOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_graph(self,
                   default_sparsity,
                   mask_init_method,
                   custom_sparsity_map,
                   n_inp=3,
                   n_out=5):
    """Setups a trivial training procedure for sparse training."""
    tf.reset_default_graph()
    optim = tf.train.GradientDescentOptimizer(1e-3)
    sparse_optim = sparse_optimizers.SparseDNWOptimizer(
        optim,
        default_sparsity,
        mask_init_method,
        custom_sparsity_map=custom_sparsity_map)

    inp_values = np.arange(1, n_inp + 1)
    scale_vector_values = np.random.uniform(size=(n_out,)) - 0.5
    # The gradient is the outer product of input and the output gradients.
    # Since the loss is sample sum the output gradient is equal to the scale
    # vector.
    expected_grads = np.outer(inp_values, scale_vector_values)

    x = tf.reshape(tf.constant(inp_values, dtype=tf.float32), (1, n_inp))
    y = layers.masked_fully_connected(x, n_out, activation_fn=None)
    scale_vector = tf.constant(scale_vector_values, dtype=tf.float32)

    y = y * scale_vector
    loss = tf.reduce_sum(y)

    global_step = tf.train.get_or_create_global_step()
    grads_and_vars = sparse_optim.compute_gradients(loss)
    train_op = sparse_optim.apply_gradients(
        grads_and_vars, global_step=global_step)
    # Init
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    mask = pruning.get_masks()[0]
    weights = pruning.get_weights()[0]
    return (sess, train_op, (expected_grads, grads_and_vars), mask, weights)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testDNWSparsity(self, n_inp, n_out, default_sparsity):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, train_op, _, mask, _ = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    _ = sess.run([train_op])
    dnw_mask, = sess.run([mask])
    n_ones = np.sum(dnw_mask)
    n_zeros = dnw_mask.size - n_ones
    n_zeros_expected = sparse_utils.get_n_zeros(dnw_mask.size, default_sparsity)
    self.assertEqual(n_zeros, n_zeros_expected)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testWeightsUsed(self, n_inp, n_out, default_sparsity):
    """Checking whether masked_grad is calculated after apply_gradients."""
    # No drop since we don't want to change the mask but check whether the grad
    # is calculated after the gradient step.
    sess, train_op, _, mask, weights = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    # Calculate sensitivity scores.
    weights, = sess.run([weights])
    expected_scores = np.abs(weights)
    _ = sess.run([train_op])
    dnw_mask, = sess.run([mask])
    kept_connection_scores = expected_scores[dnw_mask == 1]
    min_score_kept = np.min(kept_connection_scores)

    dnw_mask_connection_scores = expected_scores[dnw_mask == 0]
    max_score_removed = np.max(dnw_mask_connection_scores)
    self.assertLessEqual(max_score_removed, min_score_kept)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testGradientIsDense(self, n_inp, n_out, default_sparsity):
    """Checking whether calculated gradients are dense."""
    sess, _, grad_info, _, _ = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    expected_grad, grads_and_vars = grad_info
    grad, = sess.run([grads_and_vars[0][0]])
    self.assertAllClose(expected_grad, grad)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testDNWUpdates(self, n_inp, n_out, default_sparsity):
    """Checking whether mask is updated correctly."""
    sess, train_op, _, mask, weights = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    # On all iterations mask should have least magnitude connections.
    for _ in range(5):
      sess.run([train_op])
      mask_after, weights_after = sess.run([mask, weights])

      kept_connection_magnitudes = np.abs(weights_after[mask_after == 1])
      min_score_kept = np.min(kept_connection_magnitudes)

      removed_connection_magnitudes = np.abs(weights_after[mask_after == 0])
      max_score_removed = np.max(removed_connection_magnitudes)
      self.assertLessEqual(max_score_removed, min_score_kept)

  @parameterized.parameters((3, 4, 0.5), (5, 3, 0.8), (8, 5, 0.8))
  def testSparsityAfterDNWUpdates(self, n_inp, n_out, default_sparsity):
    """Checking whether mask is updated correctly."""
    sess, train_op, _, mask, _ = self._setup_graph(
        default_sparsity, 'random', {}, n_inp=n_inp, n_out=n_out)
    # On all iterations mask should have least magnitude connections.
    for _ in range(5):
      sess.run([train_op])
      dnw_mask, = sess.run([mask])
      n_ones = np.sum(dnw_mask)
      n_zeros = dnw_mask.size - n_ones
      n_zeros_expected = sparse_utils.get_n_zeros(dnw_mask.size,
                                                  default_sparsity)
      self.assertEqual(n_zeros, n_zeros_expected)


if __name__ == '__main__':
  tf.test.main()
