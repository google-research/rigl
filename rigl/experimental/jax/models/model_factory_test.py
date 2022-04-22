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

"""Tests for weight_symmetry.models.model_factory."""
from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.models import model_factory


class ModelCommonTest(parameterized.TestCase):
  """Tests the model factory."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._input_shape = ((1, 28, 28, 1), jnp.float32)
    self._num_classes = 10

  def _create_model(self, model_name):
    return model_factory.create_model(
        model_name,
        self._rng, (self._input_shape,),
        num_classes=self._num_classes)

  @parameterized.parameters(*model_factory.MODELS.keys())
  def test_model_supported(self, model_name):
    """Tests supported models."""
    model, state = self._create_model(model_name)

    with self.subTest(name='test_model_supported_model_instance'):
      self.assertIsInstance(model, flax.deprecated.nn.Model)

    with self.subTest(name='test_model_supported_collection_instance'):
      self.assertIsInstance(state, flax.deprecated.nn.Collection)

  def test_model_unsupported(self):
    """Tests unsupported models."""
    with self.assertRaisesRegex(ValueError, 'No such model: unsupported'):
      self._create_model('unsupported')


if __name__ == '__main__':
  absltest.main()
