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

"""Tests for weight_symmetry.pruning.pruning."""
from typing import Mapping, Optional, Sequence

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import masked
from rigl.experimental.jax.pruning import pruning


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


class MaskedTwoLayerDense(flax.deprecated.nn.Module):
  """Two-layer Dense Masked Network."""

  NUM_FEATURES: Sequence[int] = (32, 64)

  def apply(self,
            inputs,
            mask = None):
    inputs = inputs.reshape(inputs.shape[0], -1)

    inputs = masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[0],
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_0'] if mask else None)
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[1],
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_1'] if mask else None)


class MaskedConv(flax.deprecated.nn.Module):
  """Single-layer Conv Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self,
            inputs,
            mask = None):
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        kernel_size=(3, 3),
        wrapped_module=flax.deprecated.nn.Conv,
        mask=mask['MaskedModule_0'] if mask is not None else None)


class MaskedTwoLayerConv(flax.deprecated.nn.Module):
  """Two-layer Conv Masked Network."""

  NUM_FEATURES: Sequence[int] = (16, 32)

  def apply(self,
            inputs,
            mask = None):
    inputs = masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[0],
        kernel_size=(5, 5),
        wrapped_module=flax.deprecated.nn.Conv,
        mask=mask['MaskedModule_0'] if mask is not None else None)
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[1],
        kernel_size=(3, 3),
        wrapped_module=flax.deprecated.nn.Conv,
        mask=mask['MaskedModule_1'] if mask is not None else None)


class PruningTest(absltest.TestCase):
  """Tests the flax layer pruning module."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._batch_size = 2
    self._input_shape = ((self._batch_size, 28, 28, 1), jnp.float32)
    self._input = jnp.ones(*self._input_shape)

    _, initial_params = MaskedDense.init_by_shape(self._rng,
                                                  (self._input_shape,))
    self._masked_model = flax.deprecated.nn.Model(MaskedDense, initial_params)

    _, initial_params = MaskedTwoLayerDense.init_by_shape(
        self._rng, (self._input_shape,))
    self._masked_model_twolayer = flax.deprecated.nn.Model(
        MaskedTwoLayerDense, initial_params)

    _, initial_params = MaskedConv.init_by_shape(self._rng,
                                                 (self._input_shape,))
    self._masked_conv_model = flax.deprecated.nn.Model(MaskedConv,
                                                       initial_params)

    _, initial_params = MaskedTwoLayerConv.init_by_shape(
        self._rng, (self._input_shape,))
    self._masked_conv_model_twolayer = flax.deprecated.nn.Model(
        MaskedTwoLayerConv, initial_params)

  def test_prune_single_layer_dense_no_mask(self):
    """Tests pruning of single dense layer without an existing mask."""
    pruned_mask = pruning.prune(self._masked_model, 0.5)
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.5, places=3)

  def test_prune_single_layer_local_pruning(self):
    """Test pruning of model with a single layer, and local pruning schedule."""
    pruned_mask = pruning.prune(self._masked_model, {
        'MaskedModule_0': 0.5,
    })
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.5, places=3)

  def test_prune_single_layer_dense_with_mask(self):
    """Tests pruning of single dense layer with an existing mask."""
    pruned_mask = pruning.prune(
        self._masked_model,
        0.5,
        mask=masked.shuffled_mask(self._masked_model, self._rng, 0.95))
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.95, places=3)

  def test_prune_two_layers_dense_no_mask(self):
    """Tests pruning of model with two dense layers without an existing mask."""
    pruned_mask = pruning.prune(self._masked_model_twolayer, 0.5)
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_layer1_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_layer2_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_1']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.5, places=3)

  def test_prune_two_layer_local_pruning_rate(self):
    """Test pruning of model with two layers, and a local pruning schedule."""
    pruned_mask = pruning.prune(self._masked_model_twolayer, {
        'MaskedModule_1': 0.5,
    })
    mask_layer_0_sparsity = masked.mask_sparsity(pruned_mask['MaskedModule_0'])
    mask_layer_1_sparsity = masked.mask_sparsity(pruned_mask['MaskedModule_1'])

    with self.subTest(name='test_mask_layer1_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_layer2_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_1']['kernel'])

    with self.subTest(name='test_mask_layer_0_sparsity'):
      self.assertEqual(mask_layer_0_sparsity, 0.)

    with self.subTest(name='test_mask_layer_1_sparsity'):
      self.assertAlmostEqual(mask_layer_1_sparsity, 0.5, places=3)

  def test_prune_one_layer_conv_no_mask(self):
    """Tests pruning of model with one conv. layer without an existing mask."""
    pruned_mask = pruning.prune(self._masked_conv_model, 0.5)
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.5, places=1)

  def test_prune_one_layer_conv_with_mask(self):
    """Tests pruning of model with one conv. layer with an existing mask."""
    pruned_mask = pruning.prune(
        self._masked_conv_model,
        0.5,
        mask=masked.shuffled_mask(self._masked_model, self._rng, 0.95))
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.95, places=3)

  def test_prune_two_layer_conv_no_mask(self):
    """Tests pruning of model with two conv. layer without an existing mask."""
    pruned_mask = pruning.prune(self._masked_conv_model_twolayer, 0.5)
    mask_sparsity = masked.mask_sparsity(pruned_mask)

    with self.subTest(name='test_mask_layer1_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_0']['kernel'])

    with self.subTest(name='test_mask_layer2_param_not_none'):
      self.assertNotEmpty(pruned_mask['MaskedModule_1']['kernel'])

    with self.subTest(name='test_mask_sparsity'):
      self.assertAlmostEqual(mask_sparsity, 0.5, places=3)

if __name__ == '__main__':
  absltest.main()
