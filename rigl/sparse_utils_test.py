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

"""Tests for the data_helper input pipeline and the training process.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from rigl import sparse_utils
import tensorflow.compat.v1 as tf


class GetMaskRandomTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_session(self):
    """Resets the graph and returns a fresh session."""
    tf.reset_default_graph()
    sess = tf.Session()
    return sess

  @parameterized.parameters(((30, 40), 0.5), ((1, 2, 1, 4), 0.8), ((3,), 0.1))
  def testMaskConnectionDeterminism(self, shape, sparsity):
    sess = self._setup_session()
    mask = tf.ones(shape)
    mask1 = sparse_utils.get_mask_random(mask, sparsity, tf.int32)
    mask2 = sparse_utils.get_mask_random(mask, sparsity, tf.int32)
    mask1_array, = sess.run([mask1])
    mask2_array, = sess.run([mask2])
    self.assertEqual(np.sum(mask1_array), np.sum(mask2_array))

  @parameterized.parameters(((30, 4), 0.5, 60), ((1, 2, 1, 4), 0.8, 2),
                            ((30,), 0.1, 27))
  def testMaskFraction(self, shape, sparsity, expected_ones):
    sess = self._setup_session()
    mask = tf.ones(shape)
    mask1 = sparse_utils.get_mask_random(mask, sparsity, tf.int32)
    mask1_array, = sess.run([mask1])

    self.assertEqual(np.sum(mask1_array), expected_ones)

  @parameterized.parameters(tf.int32, tf.float32, tf.int64, tf.float64)
  def testMaskDtype(self, dtype):
    _ = self._setup_session()
    mask = tf.ones((3, 2))
    mask1 = sparse_utils.get_mask_random(mask, 0.5, dtype)
    self.assertEqual(mask1.dtype, dtype)


class GetSparsitiesTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_session(self):
    """Resets the graph and returns a fresh session."""
    tf.reset_default_graph()
    sess = tf.Session()
    return sess

  @parameterized.parameters(0., 0.4, 0.9)
  def testSparsityDictRandom(self, default_sparsity):
    _ = self._setup_session()
    all_masks = [tf.get_variable(shape=(2, 3), name='var1/mask'),
                 tf.get_variable(shape=(2, 3), name='var2/mask'),
                 tf.get_variable(shape=(1, 1, 3), name='var3/mask')]
    custom_sparsity = {'var1': 0.8}
    sparsities = sparse_utils.get_sparsities(
        all_masks, 'random', default_sparsity, custom_sparsity)
    self.assertEqual(sparsities[all_masks[0].name], 0.8)
    self.assertEqual(sparsities[all_masks[1].name], default_sparsity)
    self.assertEqual(sparsities[all_masks[2].name], default_sparsity)

  @parameterized.parameters(0.1, 0.4, 0.9)
  def testSparsityDictErdosRenyiCustom(self, default_sparsity):
    _ = self._setup_session()
    all_masks = [tf.get_variable(shape=(2, 4), name='var1/mask'),
                 tf.get_variable(shape=(2, 3), name='var2/mask'),
                 tf.get_variable(shape=(1, 1, 3), name='var3/mask')]
    custom_sparsity = {'var3': 0.8}
    sparsities = sparse_utils.get_sparsities(
        all_masks, 'erdos_renyi', default_sparsity, custom_sparsity)
    self.assertEqual(sparsities[all_masks[2].name], 0.8)

  @parameterized.parameters(0.1, 0.4, 0.9)
  def testSparsityDictErdosRenyiError(self, default_sparsity):
    _ = self._setup_session()
    all_masks = [tf.get_variable(shape=(2, 4), name='var1/mask'),
                 tf.get_variable(shape=(2, 3), name='var2/mask'),
                 tf.get_variable(shape=(1, 1, 3), name='var3/mask')]
    custom_sparsity = {'var3': 0.8}
    sparsities = sparse_utils.get_sparsities(
        all_masks, 'erdos_renyi', default_sparsity, custom_sparsity)
    self.assertEqual(sparsities[all_masks[2].name], 0.8)

  @parameterized.parameters(((2, 3), (2, 3), 0.5),
                            ((1, 1, 2, 3), (1, 1, 2, 3), 0.3),
                            ((8, 6), (4, 3), 0.7),
                            ((80, 4), (20, 20), 0.8),
                            ((2, 6), (2, 3), 0.8))
  def testSparsityDictErdosRenyiSparsitiesScale(
      self, shape1, shape2, default_sparsity):
    _ = self._setup_session()
    all_masks = [tf.get_variable(shape=shape1, name='var1/mask'),
                 tf.get_variable(shape=shape2, name='var2/mask')]
    custom_sparsity = {}
    sparsities = sparse_utils.get_sparsities(
        all_masks, 'erdos_renyi', default_sparsity, custom_sparsity)
    sparsity1 = sparsities[all_masks[0].name]
    size1 = np.prod(shape1)
    sparsity2 = sparsities[all_masks[1].name]
    size2 = np.prod(shape2)
    # Ensure that total number of connections are similar.
    expected_zeros_uniform = (
        sparse_utils.get_n_zeros(size1, default_sparsity) +
        sparse_utils.get_n_zeros(size2, default_sparsity))
    # Ensure that total number of connections are similar.
    expected_zeros_current = (
        sparse_utils.get_n_zeros(size1, sparsity1) +
        sparse_utils.get_n_zeros(size2, sparsity2))
    # Due to rounding we can have some difference. This is expected but should
    # be less than number of rounding operations we make.
    diff = abs(expected_zeros_uniform - expected_zeros_current)
    tolerance = 2
    self.assertLessEqual(diff, tolerance)

    # Ensure that ErdosRenyi proportions are preserved.
    factor1 = (shape1[-1] + shape1[-2]) / float(shape1[-1] * shape1[-2])
    factor2 = (shape2[-1] + shape2[-2]) / float(shape2[-1] * shape2[-2])
    self.assertAlmostEqual((1 - sparsity1) / factor1,
                           (1 - sparsity2) / factor2)


if __name__ == '__main__':
  tf.test.main()
