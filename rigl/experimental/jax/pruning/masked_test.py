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

"""Tests for weight_symmetry.pruning.masked."""
from typing import Mapping, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np

from rigl.experimental.jax.pruning import masked


class Dense(flax.deprecated.nn.Module):
  """Single-layer Dense Non-Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self, inputs):
    inputs = inputs.reshape(inputs.shape[0], -1)
    return flax.deprecated.nn.Dense(inputs, features=self.NUM_FEATURES)


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


class DenseTwoLayer(flax.deprecated.nn.Module):
  """Two-layer Dense Non-Masked Network."""

  NUM_FEATURES: Sequence[int] = (32, 64)

  def apply(self, inputs):
    inputs = inputs.reshape(inputs.shape[0], -1)
    inputs = flax.deprecated.nn.Dense(inputs, features=self.NUM_FEATURES[0])
    return flax.deprecated.nn.Dense(inputs, features=self.NUM_FEATURES[1])


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

  NUM_FEATURES: int = 16

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


class MaskedThreeLayerConvDense(flax.deprecated.nn.Module):
  """Three-layer Conv Masked Network with Dense layer."""

  NUM_FEATURES: Sequence[int] = (16, 32, 64)

  def apply(self,
            inputs,
            mask = None):
    inputs = masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[0],
        kernel_size=(5, 5),
        wrapped_module=flax.deprecated.nn.Conv,
        mask=mask['MaskedModule_0'] if mask is not None else None)
    inputs = masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[1],
        kernel_size=(3, 3),
        wrapped_module=flax.deprecated.nn.Conv,
        mask=mask['MaskedModule_1'] if mask is not None else None)
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[2],
        kernel_size=inputs.shape[1:-1],
        wrapped_module=flax.deprecated.nn.Conv,
        mask=mask['MaskedModule_2'] if mask is not None else None)


class MaskedTwoLayerMixedConvDense(flax.deprecated.nn.Module):
  """Two-layer Mixed Conv/Dense Masked Network."""

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
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_1'] if mask is not None else None)


class MaskedTest(parameterized.TestCase):
  """Tests the flax layer mask."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._batch_size = 2
    self._input_dimensions = (28, 28, 1)
    self._input_shape = ((self._batch_size,) + self._input_dimensions,
                         jnp.float32)
    self._input = jnp.ones(*self._input_shape)

    _, initial_params = Dense.init_by_shape(self._rng, (self._input_shape,))
    self._unmasked_model = flax.deprecated.nn.Model(Dense, initial_params)
    self._unmasked_output = self._unmasked_model(self._input)

    # Use the same initialization for both masked/unmasked models.
    masked_initial_params = {
        'MaskedModule_0': {
            'unmasked': initial_params['Dense_0']
        }
    }
    self._masked_model = flax.deprecated.nn.Model(MaskedDense,
                                                  masked_initial_params)

    _, initial_params = DenseTwoLayer.init_by_shape(self._rng,
                                                    (self._input_shape,))
    self._unmasked_model_twolayer = flax.deprecated.nn.Model(
        DenseTwoLayer, initial_params)
    self._unmasked_output_twolayer = self._unmasked_model_twolayer(self._input)

    # Use the same initialization for both masked/unmasked models.
    masked_initial_params = {
        'MaskedModule_0': {
            'unmasked': initial_params['Dense_0']
        },
        'MaskedModule_1': {
            'unmasked': initial_params['Dense_1']
        },
    }
    _, initial_params = MaskedTwoLayerDense.init_by_shape(
        self._rng, (self._input_shape,))
    self._masked_model_twolayer = flax.deprecated.nn.Model(
        MaskedTwoLayerDense, masked_initial_params)

    _, initial_params = MaskedConv.init_by_shape(self._rng,
                                                 (self._input_shape,))
    self._masked_conv_model = flax.deprecated.nn.Model(MaskedConv,
                                                       initial_params)

    _, initial_params = MaskedTwoLayerConv.init_by_shape(
        self._rng, (self._input_shape,))
    self._masked_conv_model_twolayer = flax.deprecated.nn.Model(
        MaskedTwoLayerConv, initial_params)

    _, initial_params = MaskedTwoLayerMixedConvDense.init_by_shape(
        self._rng, (self._input_shape,))
    self._masked_mixed_model_twolayer = flax.deprecated.nn.Model(
        MaskedTwoLayerMixedConvDense, initial_params)

    _, initial_params = MaskedThreeLayerConvDense.init_by_shape(
        self._rng, (self._input_shape,))
    self._masked_conv_fc_model_threelayer = flax.deprecated.nn.Model(
        MaskedThreeLayerConvDense, initial_params)

  def test_fully_masked_layer(self):
    """Tests masked module with full-sparsity mask."""
    full_mask = masked.simple_mask(self._masked_model, jnp.zeros, ['kernel'])

    masked_output = self._masked_model(self._input, mask=full_mask)

    with self.subTest(name='fully_masked_dense_values'):
      self.assertTrue((masked_output == 0).all())

    with self.subTest(name='fully_masked_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_no_mask_masked_layer(self):
    """Tests masked module with no mask."""
    masked_output = self._masked_model(self._input, mask=None)

    with self.subTest(name='no_mask_masked_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='no_mask_masked_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_empty_mask_masked_layer(self):
    """Tests masked module with an empty (not sparse) mask."""
    empty_mask = masked.simple_mask(self._masked_model, jnp.ones, ['kernel'])

    masked_output = self._masked_model(self._input, mask=empty_mask)

    with self.subTest(name='empty_mask_masked_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='empty_mask_masked_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_invalid_mask(self):
    """Tests using an invalid mask."""
    invalid_mask = {
        'MaskedModule_0': {
            'not_kernel':
                jnp.ones(self._unmasked_model.params['Dense_0']['kernel'].shape)
        }
    }

    with self.assertRaisesRegex(ValueError, 'Mask is invalid for model.'):
      self._masked_model(self._input, mask=invalid_mask)

  def test_shuffled_mask_invalid_model(self):
    """Tests shuffled mask with model containing no masked layers."""
    with self.assertRaisesRegex(
        ValueError, 'Model does not support masking, i.e. no layers are '
        'wrapped by a MaskedModule.'):
      masked.shuffled_mask(self._unmasked_model, self._rng, 0.5)

  def test_shuffled_mask_invalid_sparsity(self):
    """Tests shuffled mask with invalid sparsity."""

    with self.subTest(name='sparsity_too_small'):
      with self.assertRaisesRegex(
          ValueError, r'Given sparsity, -0.5, is not in range \[0, 1\]'):
        masked.shuffled_mask(self._masked_model, self._rng, -0.5)

    with self.subTest(name='sparsity_too_large'):
      with self.assertRaisesRegex(
          ValueError, r'Given sparsity, 1.5, is not in range \[0, 1\]'):
        masked.shuffled_mask(self._masked_model, self._rng, 1.5)

  def test_shuffled_mask_sparsity_full(self):
    """Tests shuffled mask generation, for 100% sparsity."""
    mask = masked.shuffled_mask(self._masked_model, self._rng, 1.0)

    with self.subTest(name='shuffled_full_mask'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_full_mask_values'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 0).all())

    with self.subTest(name='shuffled_full_mask_not_masked_values'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='shuffled_full_mask_dense_values'):
      self.assertTrue((masked_output == 0).all())

    with self.subTest(name='shuffled_full_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_shuffled_mask_sparsity_empty(self):
    """Tests shuffled mask generation, for 0% sparsity."""
    mask = masked.shuffled_mask(self._masked_model, self._rng, 0.0)

    with self.subTest(name='shuffled_empty_mask'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_empty_mask_values'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 1).all())

    with self.subTest(name='shuffled_empty_mask_not_masked_values'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='shuffled_empty_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='shuffled_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_shuffled_mask_sparsity_half_full(self):
    """Tests shuffled mask generation, for a half-full mask."""
    mask = masked.shuffled_mask(self._masked_model, self._rng, 0.5)
    param_len = self._masked_model.params['MaskedModule_0']['unmasked'][
        'kernel'].size

    with self.subTest(name='shuffled_mask_values'):
      self.assertEqual(
          jnp.sum(mask['MaskedModule_0']['kernel']), param_len // 2)

  def test_shuffled_mask_sparsity_full_twolayer(self):
    """Tests shuffled mask generation for two layers, and 100% sparsity."""
    mask = masked.shuffled_mask(self._masked_model_twolayer, self._rng, 1.0)

    with self.subTest(name='shuffled_full_mask_layer1'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_full_mask_values_layer1'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 0).all())

    with self.subTest(name='shuffled_full_mask_not_masked_values_layer1'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    with self.subTest(name='shuffled_full_mask_layer2'):
      self.assertIn('MaskedModule_1', mask)

    with self.subTest(name='shuffled_full_mask_values_layer2'):
      self.assertTrue((mask['MaskedModule_1']['kernel'] == 0).all())

    with self.subTest(name='shuffled_full_mask_not_masked_values_layer1'):
      self.assertIsNone(mask['MaskedModule_1']['bias'])

    masked_output = self._masked_model_twolayer(self._input, mask=mask)

    with self.subTest(name='shuffled_full_mask_dense_values'):
      self.assertTrue((masked_output == 0).all())

    with self.subTest(name='shuffled_full_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape,
                               self._unmasked_output_twolayer.shape)

  def test_shuffled_mask_sparsity_empty_twolayer(self):
    """Tests shuffled mask generation for two layers, for 0% sparsity."""
    mask = masked.shuffled_mask(self._masked_model_twolayer, self._rng, 0.0)

    with self.subTest(name='shuffled_empty_mask_layer1'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_empty_mask_values_layer1'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 1).all())

    with self.subTest(name='shuffled_empty_mask_layer2'):
      self.assertIn('MaskedModule_1', mask)

    with self.subTest(name='shuffled_empty_mask_values_layer2'):
      self.assertTrue((mask['MaskedModule_1']['kernel'] == 1).all())

    masked_output = self._masked_model_twolayer(self._input, mask=mask)

    with self.subTest(name='shuffled_empty_dense_values'):
      self.assertTrue(
          jnp.isclose(masked_output, self._unmasked_output_twolayer).all())

    with self.subTest(name='shuffled_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape,
                               self._unmasked_output_twolayer.shape)

  def test_random_invalid_model(self):
    """Tests random mask with model containing no masked layers."""
    with self.assertRaisesRegex(
        ValueError, 'Model does not support masking, i.e. no layers are '
        'wrapped by a MaskedModule.'):
      masked.random_mask(self._unmasked_model, self._rng, 0.5)

  def test_random_invalid_sparsity(self):
    """Tests random mask with invalid sparsity."""

    with self.subTest(name='random_sparsity_too_small'):
      with self.assertRaisesRegex(
          ValueError, r'Given sparsity, -0.5, is not in range \[0, 1\]'):
        masked.random_mask(self._masked_model, self._rng, -0.5)

    with self.subTest(name='random_sparsity_too_large'):
      with self.assertRaisesRegex(
          ValueError, r'Given sparsity, 1.5, is not in range \[0, 1\]'):
        masked.random_mask(self._masked_model, self._rng, 1.5)

  def test_random_mask_sparsity_full(self):
    """Tests random mask generation, for 100% sparsity."""
    mask = masked.random_mask(self._masked_model, self._rng, 1.)

    with self.subTest(name='random_full_mask_values'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 0).all())

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='random_full_mask_dense_values'):
      self.assertTrue((masked_output.all() == 0).all())

    with self.subTest(name='random_full_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_random_mask_sparsity_empty(self):
    """Tests random mask generation, for 0% sparsity."""
    mask = masked.random_mask(self._masked_model, self._rng, 0.)

    with self.subTest(name='random_empty_mask_values'):
      self.assertEqual(
          jnp.sum(mask['MaskedModule_0']['kernel']),
          mask['MaskedModule_0']['kernel'].size)

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='random_empty_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='random_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_random_mask_sparsity_half_full(self):
    """Tests random mask generation, for a half-full mask."""
    mask = masked.random_mask(self._masked_model, self._rng, 0.5)
    param_len = self._masked_model.params['MaskedModule_0']['unmasked'][
        'kernel'].size
    half_full = param_len / 2

    with self.subTest(name='random_mask_values'):
      self.assertBetween(
          jnp.sum(mask['MaskedModule_0']['kernel']), 0.66 * half_full,
          1.33 * half_full)

  def test_simple_mask_one_layer(self):
    """Tests generation of a simple mask."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(self._masked_model.params['MaskedModule_0']
                          ['unmasked']['kernel'].shape),
            'bias':
                None,
        }
    }

    gen_mask = masked.simple_mask(self._masked_model, jnp.zeros, ['kernel'])

    result, _ = jax.tree_flatten(
        jax.tree_util.tree_map(lambda x, *xs: (x == xs[0]).all(), mask,
                               gen_mask))

    self.assertTrue(all(result))

  def test_simple_mask_two_layer(self):
    """Tests generation of a simple mask."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(self._masked_model_twolayer.params['MaskedModule_0']
                          ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
        'MaskedModule_1': {
            'kernel':
                jnp.zeros(self._masked_model_twolayer.params['MaskedModule_1']
                          ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
    }

    gen_mask = masked.simple_mask(self._masked_model_twolayer, jnp.zeros,
                                  ['kernel'])

    result, _ = jax.tree_flatten(
        jax.tree_util.tree_map(lambda x, *xs: (x == xs[0]).all(), mask,
                               gen_mask))

    self.assertTrue(all(result))

  def test_shuffled_mask_neuron_mask_sparsity_empty(self):
    """Tests shuffled neuron mask generation, for 0% sparsity."""
    mask = masked.shuffled_neuron_mask(self._masked_model, self._rng, 0.0)

    with self.subTest(name='shuffled_neuron_empty_mask'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_neuron_empty_mask_values'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 1).all())

    with self.subTest(name='shuffled_neuron_empty_mask_not_masked_values'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='shuffled_neuron_empty_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='shuffled_neuron_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_shuffled_mask_neuron_mask_sparsity_half_full(self):
    """Tests shuffled mask generation, for a half-full mask."""
    mask = masked.shuffled_neuron_mask(self._masked_model, self._rng, 0.5)
    param_len = len(
        self._masked_model.params['MaskedModule_0']['unmasked']['kernel'][:, 0])
    mask_sum = jnp.sum(mask['MaskedModule_0']['kernel'][:, 0])

    with self.subTest(name='shuffled_mask_values'):
      # Check that single neuron has the correct sparsity.
      self.assertEqual(mask_sum, param_len // 2)

    with self.subTest(name='shuffled_mask_rows_different'):
      # Check that two rows are different.
      self.assertFalse(
          jnp.isclose(mask['MaskedModule_0']['kernel'][:, 0],
                      mask['MaskedModule_0']['kernel'][:, 1]).all())

  def test_symmetric_mask_sparsity_empty(self):
    """Tests symmetric mask generation, for 0% sparsity."""
    mask = masked.symmetric_mask(self._masked_model, self._rng, 0.0)

    with self.subTest(name='shuffled_neuron_empty_mask'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='symmetric_empty_mask_values'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 1).all())

    with self.subTest(name='symmetric_empty_mask_not_masked_values'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='symmetric_empty_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='symmetric_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_symmetric_mask_sparsity_half_full(self):
    """Tests shuffled mask generation, for a half-full mask."""
    mask = masked.symmetric_mask(self._masked_model, self._rng, 0.5)
    param_len = len(
        self._masked_model.params['MaskedModule_0']['unmasked']['kernel'][:, 0])
    mask_sum = jnp.sum(mask['MaskedModule_0']['kernel'][:, 0])

    with self.subTest(name='symmetric_mask_values'):
      # Check that single neuron has the correct sparsity.
      self.assertEqual(mask_sum, param_len // 2)

    with self.subTest(name='symmetric_mask_rows_different'):
      # Check that two rows are same.
      self.assertTrue(
          jnp.isclose(mask['MaskedModule_0']['kernel'][:, 0],
                      mask['MaskedModule_0']['kernel'][:, 1]).all())

  def test_propagate_masks_ablated_neurons_one_layer(self):
    """Tests mask propagation on a single layer model."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jax.random.normal(
                    self._rng,
                    self._masked_model_twolayer.params['MaskedModule_0']
                    ['unmasked']['kernel'].shape,
                    dtype=jnp.float32),
            'bias':
                None,
        },
    }

    refined_mask = masked.propagate_masks(mask)

    # Since this is a single layer, should not affect mask at all.
    self.assertTrue((mask['MaskedModule_0']['kernel'] ==
                     refined_mask['MaskedModule_0']['kernel']).all())

  def test_propagate_masks_ablated_neurons_two_layers(self):
    """Tests mask propagation on a two-layer model."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(self._masked_model_twolayer.params['MaskedModule_0']
                          ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
        'MaskedModule_1': {
            'kernel':
                jnp.ones(self._masked_model_twolayer.params['MaskedModule_1']
                         ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
    }

    refined_mask = masked.propagate_masks(mask)

    with self.subTest(name='layer_1'):
      self.assertTrue((refined_mask['MaskedModule_0']['kernel'] == 0).all())

    # Since layer 1 is all zero, layer 2 is also effectively zero.
    with self.subTest(name='layer_2'):
      self.assertTrue((refined_mask['MaskedModule_1']['kernel'] == 0).all())

  def test_propagate_masks_ablated_neurons_two_layers_nonmasked(self):
    """Tests mask propagation where previous layer is not masked."""
    mask = {
        'Dense_0': {
            'kernel': None,
            'bias': None,
        },
        'MaskedModule_1': {
            'kernel':
                jax.random.normal(
                    self._rng,
                    self._masked_model_twolayer.params['MaskedModule_1']
                    ['unmasked']['kernel'].shape,
                    dtype=jnp.float32),
            'bias':
                None,
        },
    }

    refined_mask = masked.propagate_masks(mask)

    with self.subTest(name='layer_1'):
      self.assertIsNone(refined_mask['Dense_0']['kernel'])

    # Since layer 1 is all zero, layer 2 is also effectively zero.
    with self.subTest(name='layer_2'):
      # Since this is a single masked layer, should not affect mask at all.
      self.assertTrue((mask['MaskedModule_1']['kernel'] ==
                       refined_mask['MaskedModule_1']['kernel']).all())

  def test_propagate_masks_ablated_neurons_one_conv_layer(self):
    """Tests mask propagation on a single layer model."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jax.random.normal(
                    self._rng,
                    self._masked_conv_model.params['MaskedModule_0']['unmasked']
                    ['kernel'].shape,
                    dtype=jnp.float32),
            'bias':
                None,
        },
    }

    refined_mask = masked.propagate_masks(mask)

    # Since this is a single layer, should not affect mask at all.
    self.assertTrue((mask['MaskedModule_0']['kernel'] ==
                     refined_mask['MaskedModule_0']['kernel']).all())

  def test_propagate_masks_ablated_neurons_two_conv_layers(self):
    """Tests mask propagation on a two-layer convolutional model."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(
                    self._masked_conv_model_twolayer.params['MaskedModule_0']
                    ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
        'MaskedModule_1': {
            'kernel':
                jnp.ones(
                    self._masked_conv_model_twolayer.params['MaskedModule_1']
                    ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
    }

    refined_mask = masked.propagate_masks(mask)

    with self.subTest(name='layer_1'):
      self.assertTrue((refined_mask['MaskedModule_0']['kernel'] == 0).all())

    # Since layer 1 is all zero, layer 2 is also effectively zero.
    with self.subTest(name='layer_2'):
      self.assertTrue((refined_mask['MaskedModule_1']['kernel'] == 0).all())

  def test_propagate_masks_ablated_neurons_three_conv_fc_layers(self):
    """Tests mask propagation on a two-layer convolutional model with dense."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(self._masked_conv_fc_model_threelayer
                          .params['MaskedModule_0']['unmasked']['kernel'].shape
                         ),
            'bias':
                None,
        },
        'MaskedModule_1': {
            'kernel':
                jnp.ones(self._masked_conv_fc_model_threelayer
                         .params['MaskedModule_1']['unmasked']['kernel'].shape),
            'bias':
                None,
        },
        'MaskedModule_2': {
            'kernel':
                jnp.ones(self._masked_conv_fc_model_threelayer
                         .params['MaskedModule_2']['unmasked']['kernel'].shape),
            'bias':
                None,
        },
    }

    refined_mask = masked.propagate_masks(mask)

    with self.subTest(name='layer_1'):
      self.assertTrue((refined_mask['MaskedModule_0']['kernel'] == 0).all())

    # Since layer 1 is all zero, layer 2 is also effectively zero.
    with self.subTest(name='layer_2'):
      self.assertTrue((refined_mask['MaskedModule_1']['kernel'] == 0).all())

    # Since layer 2 is all zero, layer 3 is also effectively zero.
    with self.subTest(name='layer_3'):
      self.assertTrue((refined_mask['MaskedModule_2']['kernel'] == 0).all())

  def test_propagate_masks_ablated_neurons_mixed_conv_dense_layers(self):
    """Tests mask propagation on a two-layer convolutional/dense model."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(
                    self._masked_mixed_model_twolayer.params['MaskedModule_0']
                    ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
        'MaskedModule_1': {
            'kernel':
                jnp.ones(
                    self._masked_mixed_model_twolayer.params['MaskedModule_1']
                    ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
    }

    with self.assertRaisesRegex(
        ValueError, 'propagate_masks requires knowledge of the spatial '
        'dimensions of the previous layer. Use a functionally equivalent '
        'conv. layer in place of a dense layer in a model with a mixed '
        'conv/dense setting.'):
      masked.propagate_masks(mask)

  def test_mask_layer_sparsity_zero_mask(self):
    """Tests mask calculation with a zeroed mask."""
    zero_mask = masked.simple_mask(self._masked_model, jnp.ones, ['kernel'])

    self.assertEqual(
        masked.mask_layer_sparsity(zero_mask['MaskedModule_0']), 0.)

  def test_mask_layer_sparsity_half_mask(self):
    """Tests mask calculation with a half-filled mask."""
    half_mask = masked.shuffled_mask(self._masked_model, self._rng, 0.5)

    self.assertAlmostEqual(
        masked.mask_layer_sparsity(half_mask['MaskedModule_0']), 0.5)

  def test_mask_layer_sparsity_ones_mask(self):
    """Tests mask calculation with a mask full of ones."""
    one_mask = masked.simple_mask(self._masked_model, jnp.zeros, ['kernel'])

    self.assertEqual(
        masked.mask_layer_sparsity(one_mask['MaskedModule_0']), 1.)

  def test_mask_sparsity_zero_mask(self):
    """Tests mask calculation with a zeroed mask."""
    zero_mask = masked.simple_mask(self._masked_model, jnp.ones, ['kernel'])

    self.assertEqual(masked.mask_sparsity(zero_mask), 0.)

  def test_mask_sparsity_ones_mask(self):
    """Tests mask calculation with a mask full of ones."""
    one_mask = masked.simple_mask(self._masked_model, jnp.zeros, ['kernel'])

    self.assertEqual(masked.mask_sparsity(one_mask), 1.)

  def test_mask_sparsity_mixed_mask(self):
    """Tests mask calculation with a mask different sparsity masked layers."""
    mask = {
        'MaskedModule_0': {
            'kernel':
                jnp.zeros(
                    self._masked_conv_model_twolayer.params['MaskedModule_0']
                    ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
        'MaskedModule_1': {
            'kernel':
                jnp.ones(
                    self._masked_conv_model_twolayer.params['MaskedModule_1']
                    ['unmasked']['kernel'].shape),
            'bias':
                None,
        },
    }

    mask_sparsity = masked.mask_sparsity(mask)
    true_sparsity = self._masked_conv_model_twolayer.params['MaskedModule_1'][
        'unmasked']['kernel'].size / (
            self._masked_conv_model_twolayer.params['MaskedModule_0']
            ['unmasked']['kernel'].size + self._masked_conv_model_twolayer
            .params['MaskedModule_1']['unmasked']['kernel'].size)

    self.assertAlmostEqual(mask_sparsity, 1.0 - true_sparsity)

  @parameterized.parameters(
      # Simple masked 1-layer model.
      (1,),
      # Simple masked 2-layer model.
      (2,),
      # Simple masked 10-layer model.
      (10,),
  )
  def test_generate_model_masks_depth_only(self, depth):
    mask = masked.generate_model_masks(depth)
    with self.subTest(name='test_model_mask_length'):
      self.assertLen(mask, depth)

    for i in range(depth):
      with self.subTest(name=f'test_model_mask_value_layer_{i}'):
        self.assertIsNone(mask[f'MaskedModule_{i}'])

  @parameterized.parameters(
      # Simple masked 1-layer model, no masked indices.
      (1, []),
      # Simple masked 2-layer model, second layer masked.
      (2, (1,)),
      # Simple masked 10-layer model, 4 layers masked.
      (10, (1, 2, 3, 9)),
  )
  def test_generate_model_masks_indices(self, depth, indices):
    mask = masked.generate_model_masks(depth, None, indices)

    with self.subTest(name='test_model_mask_length'):
      self.assertLen(mask, len(indices))

    for i in indices:
      with self.subTest(name=f'test_model_mask_value_layer_{i}'):
        self.assertIsNone(mask[f'MaskedModule_{i}'])

  @parameterized.parameters(
      # Existing 1-layer mask.
      (1, {'MaskedModule_0': np.ones(1)}, None),
      (2, {'MaskedModule_0': np.ones(1),
           'MaskedModule_1': np.ones(1)}, None),
      # Existing 2-layer mask, only using one due to mask indices.
      (2, {'MaskedModule_0': np.ones(1),
           'MaskedModule_1': np.ones(1),}, (1,)),
  )
  def test_generate_model_masks_existing_mask(self, depth, existing_mask,
                                              indices):
    mask = masked.generate_model_masks(depth, existing_mask, indices)

    # Need to differentiate from empty sequence by explicitly checking is None.
    if indices is None:
      indices = range(depth)

    with self.subTest(name='test_model_mask_length'):
      self.assertLen(mask, len(indices))

    for i in indices:
      with self.subTest(name=f'test_model_mask_value_layer_{i}'):
        self.assertIsNotNone(mask[f'MaskedModule_{i}'])

    # Ensure existing mask layers that aren't in indices aren't in output.
    for i in range(depth):
      if i not in indices:
        with self.subTest(
            name=f'test_model_mask_only_allowed_indices_layer_{i}'):
          self.assertNotIn(f'MaskedModule_{i}', mask)

  def test_generate_model_masks_invalid_depth_zero(self):
    with self.assertRaisesWithLiteralMatch(ValueError,
                                           'Invalid model depth: 0'):
      masked.generate_model_masks(0)

  def test_generate_model_masks_invalid_index_toohigh(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Invalid indices for given depth (2): (1, 2)'):
      masked.generate_model_masks(2, None, (1, 2))

  def test_generate_model_masks_invalid_index_negative(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Invalid indices for given depth (2): (-1, 2)'):
      masked.generate_model_masks(2, None, (-1, 2))

  def test_shuffled_neuron_no_input_ablation_mask_invalid_model(self):
    """Tests shuffled mask with model containing no masked layers."""
    with self.assertRaisesRegex(
        ValueError, 'Model does not support masking, i.e. no layers are '
        'wrapped by a MaskedModule.'):
      masked.shuffled_neuron_no_input_ablation_mask(self._unmasked_model,
                                                    self._rng, 0.5)

  def test_shuffled_neuron_no_input_ablation_mask_invalid_sparsity(self):
    """Tests shuffled mask with invalid sparsity."""

    with self.subTest(name='sparsity_too_small'):
      with self.assertRaisesRegex(
          ValueError, r'Given sparsity, -0.5, is not in range \[0, 1\]'):
        masked.shuffled_neuron_no_input_ablation_mask(self._masked_model,
                                                      self._rng, -0.5)

    with self.subTest(name='sparsity_too_large'):
      with self.assertRaisesRegex(
          ValueError, r'Given sparsity, 1.5, is not in range \[0, 1\]'):
        masked.shuffled_neuron_no_input_ablation_mask(self._masked_model,
                                                      self._rng, 1.5)

  def test_shuffled_neuron_no_input_ablation_mask_sparsity_full(self):
    """Tests shuffled mask generation, for 100% sparsity."""
    mask = masked.shuffled_neuron_no_input_ablation_mask(self._masked_model,
                                                         self._rng, 1.0)

    with self.subTest(name='shuffled_full_mask'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_full_mask_values'):
      self.assertEqual(jnp.count_nonzero(mask['MaskedModule_0']['kernel']),
                       jnp.prod(jnp.array(self._input_dimensions)))

    with self.subTest(name='shuffled_full_no_input_ablation'):
      # Check no row (neurons are columns) is completely ablated.
      self.assertTrue((jnp.count_nonzero(
          mask['MaskedModule_0']['kernel'], axis=0) != 0).all())

    with self.subTest(name='shuffled_full_mask_not_masked_values'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='shuffled_full_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_shuffled_neuron_no_input_ablation_mask_sparsity_empty(self):
    """Tests shuffled mask generation, for 0% sparsity."""
    mask = masked.shuffled_neuron_no_input_ablation_mask(self._masked_model,
                                                         self._rng, 0.0)

    with self.subTest(name='shuffled_empty_mask'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_empty_mask_values'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 1).all())

    with self.subTest(name='shuffled_empty_mask_not_masked_values'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    masked_output = self._masked_model(self._input, mask=mask)

    with self.subTest(name='shuffled_empty_dense_values'):
      self.assertTrue(jnp.isclose(masked_output, self._unmasked_output).all())

    with self.subTest(name='shuffled_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape, self._unmasked_output.shape)

  def test_shuffled_neuron_no_input_ablation_mask_sparsity_half_full(self):
    """Tests shuffled mask generation, for a half-full mask."""
    mask = masked.shuffled_neuron_no_input_ablation_mask(self._masked_model,
                                                         self._rng, 0.5)
    param_shape = self._masked_model.params['MaskedModule_0']['unmasked'][
        'kernel'].shape

    with self.subTest(name='shuffled_mask_values'):
      self.assertEqual(
          jnp.sum(mask['MaskedModule_0']['kernel']),
          param_shape[0]//2 * param_shape[1])

    with self.subTest(name='shuffled_half_no_input_ablation'):
      # Check no row (neurons are columns) is completely ablated.
      self.assertTrue((jnp.count_nonzero(
          mask['MaskedModule_0']['kernel'], axis=0) != 0).all())

  def test_shuffled_neuron_no_input_ablation_mask_sparsity_quarter_full(self):
    """Tests shuffled mask generation, for a half-full mask."""
    mask = masked.shuffled_neuron_no_input_ablation_mask(self._masked_model,
                                                         self._rng, 0.25)
    param_shape = self._masked_model.params['MaskedModule_0']['unmasked'][
        'kernel'].shape

    with self.subTest(name='shuffled_mask_values'):
      self.assertEqual(
          jnp.sum(mask['MaskedModule_0']['kernel']),
          0.75 * param_shape[0] * param_shape[1])

    with self.subTest(name='shuffled_half_no_input_ablation'):
      # Check no row (neurons are columns) is completely ablated.
      self.assertTrue((jnp.count_nonzero(
          mask['MaskedModule_0']['kernel'], axis=0) != 0).all())

  def test_shuffled_neuron_no_input_ablation_mask_sparsity_full_twolayer(self):
    """Tests shuffled mask generation for two layers, and 100% sparsity."""
    mask = masked.shuffled_neuron_no_input_ablation_mask(
        self._masked_model_twolayer, self._rng, 1.0)

    with self.subTest(name='shuffled_full_mask_layer1'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_full_mask_values_layer1'):
      self.assertEqual(jnp.count_nonzero(mask['MaskedModule_0']['kernel']),
                       jnp.prod(jnp.array(self._input_dimensions)))

    with self.subTest(name='shuffled_full_mask_not_masked_values_layer1'):
      self.assertIsNone(mask['MaskedModule_0']['bias'])

    with self.subTest(name='shuffled_full_no_input_ablation_layer1'):
      # Check no row (neurons are columns) is completely ablated.
      self.assertTrue((jnp.count_nonzero(
          mask['MaskedModule_0']['kernel'], axis=0) != 0).all())

    with self.subTest(name='shuffled_full_mask_layer2'):
      self.assertIn('MaskedModule_1', mask)

    with self.subTest(name='shuffled_full_mask_values_layer2'):
      self.assertEqual(jnp.count_nonzero(mask['MaskedModule_1']['kernel']),
                       jnp.prod(MaskedTwoLayerDense.NUM_FEATURES[0]))

    with self.subTest(name='shuffled_full_mask_not_masked_values_layer2'):
      self.assertIsNone(mask['MaskedModule_1']['bias'])

    with self.subTest(name='shuffled_full_no_input_ablation_layer2'):
      # Note: check no *inputs* are ablated, and inputs < num_neurons.
      self.assertEqual(
          jnp.sum(jnp.count_nonzero(mask['MaskedModule_1']['kernel'], axis=0)),
          MaskedTwoLayerDense.NUM_FEATURES[0])

    masked_output = self._masked_model_twolayer(self._input, mask=mask)

    with self.subTest(name='shuffled_full_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape,
                               self._unmasked_output_twolayer.shape)

  def test_shuffled_neuron_no_input_ablation_mask_sparsity_empty_twolayer(self):
    """Tests shuffled mask generation for two layers, for 0% sparsity."""
    mask = masked.shuffled_neuron_no_input_ablation_mask(
        self._masked_model_twolayer, self._rng, 0.0)

    with self.subTest(name='shuffled_empty_mask_layer1'):
      self.assertIn('MaskedModule_0', mask)

    with self.subTest(name='shuffled_empty_mask_values_layer1'):
      self.assertTrue((mask['MaskedModule_0']['kernel'] == 1).all())

    with self.subTest(name='shuffled_empty_mask_layer2'):
      self.assertIn('MaskedModule_1', mask)

    with self.subTest(name='shuffled_empty_mask_values_layer2'):
      self.assertTrue((mask['MaskedModule_1']['kernel'] == 1).all())

    masked_output = self._masked_model_twolayer(self._input, mask=mask)

    with self.subTest(name='shuffled_empty_dense_values'):
      self.assertTrue(
          jnp.isclose(masked_output, self._unmasked_output_twolayer).all())

    with self.subTest(name='shuffled_empty_mask_dense_shape'):
      self.assertSequenceEqual(masked_output.shape,
                               self._unmasked_output_twolayer.shape)


if __name__ == '__main__':
  absltest.main()
