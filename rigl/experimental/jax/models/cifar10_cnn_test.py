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

"""Tests for weight_symmetry.models.cifar10_cnn."""
from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.models import cifar10_cnn


class CIFAR10CNNTest(absltest.TestCase):
  """Tests the CIFAR10CNN model."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._num_classes = 10
    self._batch_size = 2
    self._input_shape = ((self._batch_size, 32, 32, 3), jnp.float32)
    self._input = jnp.zeros(*self._input_shape)

  def test_output_shapes(self):
    """Tests the output shapes of the model."""
    with flax.deprecated.nn.stateful() as initial_state:
      _, initial_params = cifar10_cnn.CIFAR10CNN.init_by_shape(
          self._rng, (self._input_shape,), num_classes=self._num_classes)
      model = flax.deprecated.nn.Model(cifar10_cnn.CIFAR10CNN, initial_params)

    with flax.deprecated.nn.stateful(initial_state, mutable=False):
      logits = model(self._input, num_classes=self._num_classes, train=False)

    self.assertTupleEqual(logits.shape, (self._batch_size, self._num_classes))

  def test_invalid_spatial_dimensions(self):
    """Tests model with an invalid spatial dimension parameters."""
    with self.assertRaisesRegex(ValueError, 'Input spatial size, '):
      cifar10_cnn.CIFAR10CNN.init_by_shape(
          self._rng, (self._input_shape,),
          num_classes=self._num_classes,
          filters=20 * (32,))

  def test_invalid_masks_depth(self):
    """Tests model mask with the incorrect depth for the given model."""
    invalid_masks = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros((self._batch_size, 3, 3, 32))
        }
    }

    with self.assertRaisesRegex(
        ValueError, 'Mask is invalid for model.'):
      cifar10_cnn.CIFAR10CNN.init_by_shape(
          self._rng, (self._input_shape,),
          num_classes=self._num_classes,
          masks=invalid_masks)

if __name__ == '__main__':
  absltest.main()
