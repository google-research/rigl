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
from typing import Mapping, Optional

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import mask_factory
from rigl.experimental.jax.pruning import masked


class MaskedDense(flax.deprecated.nn.Module):
  """Single-layer Dense Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self,
            inputs,
            mask = None):
    inputs = inputs.reshape(inputs.shape[0], -1)

    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_0'] if mask else None)


class MaskFactoryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._input_shape = ((1, 28, 28, 1), jnp.float32)
    self._num_classes = 10
    self._sparsity = 0.9

    _, initial_params = MaskedDense.init_by_shape(self._rng,
                                                  (self._input_shape,))
    # Use the same initialization for both masked/unmasked models.
    self._model = flax.deprecated.nn.Model(MaskedDense, initial_params)

  def _create_mask(self, mask_type):
    return mask_factory.create_mask(
        mask_type, self._model,
        self._rng, self._sparsity)

  @parameterized.parameters(*mask_factory.MASK_TYPES.keys())
  def test_mask_supported(self, mask_type):
    """Tests supported mask types."""
    mask = self._create_mask(mask_type)

    with self.subTest(name='test_mask_type'):
      self.assertIsInstance(mask, dict)

  def test_mask_unsupported(self):
    """Tests unsupported mask types."""
    with self.assertRaisesRegex(ValueError,
                                'Unknown mask type: unsupported'):
      self._create_mask('unsupported')


if __name__ == '__main__':
  absltest.main()
