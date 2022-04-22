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

"""Tests for weight_symmetry.models.mnist_cnn."""
from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.models import mnist_cnn


class MNISTCNNTest(absltest.TestCase):
  """Tests the MNISTCNN model."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._num_classes = 10
    self._batch_size = 2
    self._input_shape = ((self._batch_size, 28, 28, 1), jnp.float32)
    self._input = jnp.zeros(*self._input_shape)

  def test_output_shapes(self):
    """Tests the output shapes of the model."""
    with flax.deprecated.nn.stateful() as initial_state:
      _, initial_params = mnist_cnn.MNISTCNN.init_by_shape(
          self._rng, (self._input_shape,), num_classes=self._num_classes)
      model = flax.deprecated.nn.Model(mnist_cnn.MNISTCNN, initial_params)

    with flax.deprecated.nn.stateful(initial_state, mutable=False):
      logits = model(self._input, num_classes=self._num_classes, train=False)

    self.assertTupleEqual(logits.shape, (self._batch_size, self._num_classes))

  def test_invalid_depth(self):
    """Tests model mask with the incorrect depth for the given model."""
    with self.assertRaisesRegex(ValueError, 'Input spatial size, '):
      mnist_cnn.MNISTCNN.init_by_shape(
          self._rng, (self._input_shape,),
          num_classes=self._num_classes,
          filters=10 * (32,))

if __name__ == '__main__':
  absltest.main()
