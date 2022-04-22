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

"""Tests for weight_symmetry.models.mnist_fc."""
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.models import mnist_fc
from rigl.experimental.jax.utils import utils

PARAM_COUNT_PARAM: Sequence[str] = ('kernel',)


class MNISTFCTest(parameterized.TestCase):
  """Tests the MNISTFC model."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._num_classes = 10
    self._batch_size = 2
    self._input_len = 28*28*1
    self._input_shape = ((self._batch_size, self._input_len), jnp.float32)
    self._input = jnp.zeros((self._batch_size, self._input_len), jnp.float32)
    self._param_count = 1e7

  def test_output_shapes(self):
    """Tests the output shape from the model."""
    with flax.deprecated.nn.stateful() as initial_state:
      _, initial_params = mnist_fc.MNISTFC.init_by_shape(
          self._rng, (self._input_shape,), num_classes=self._num_classes)
      model = flax.deprecated.nn.Model(mnist_fc.MNISTFC, initial_params)

    with flax.deprecated.nn.stateful(initial_state, mutable=False):
      logits = model(self._input, num_classes=self._num_classes, train=False)

    self.assertTupleEqual(logits.shape, (self._batch_size, self._num_classes))

  def test_invalid_masks_depth(self):
    """Tests a model with an invalid mask."""
    invalid_masks = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros((self._batch_size, 5 * 5 * 16))
        }
    }

    with self.assertRaisesRegex(
        ValueError, 'Mask is invalid for model.'):
      mnist_fc.MNISTFC.init_by_shape(
          self._rng,
          (self._input_shape,),
          num_classes=self._num_classes,
          masks=invalid_masks)

  def _create_model(self, features):
    """Convenience fn to create a FLAX model ."""
    _, initial_params = mnist_fc.MNISTFC.init_by_shape(
        self._rng,
        (self._input_shape,),
        num_classes=self._num_classes,
        features=features)
    return flax.deprecated.nn.Model(mnist_fc.MNISTFC, initial_params)

  @parameterized.parameters(*range(1, 6))
  def test_feature_dim_for_param_depth(self, depth):
    """Tests feature_dim_for_param with multiple depths."""
    features = mnist_fc.feature_dim_for_param(self._input_len,
                                              self._param_count, depth)
    model = self._create_model(features)
    total_size = utils.count_param(model, PARAM_COUNT_PARAM)

    with self.subTest(name='FeatureDimLen'):
      self.assertLen(features, depth)

    with self.subTest(name='FeatureDimParamCount'):
      self.assertBetween(total_size, self._param_count * 0.95,
                         self._param_count * 1.05)

if __name__ == '__main__':
  absltest.main()
