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

"""Tests for weight_symmetry.pruning.symmetry."""
import functools
import math
import operator
from typing import Mapping, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np

from rigl.experimental.jax.pruning import masked
from rigl.experimental.jax.pruning import symmetry


class MaskedDense(flax.deprecated.nn.Module):
  """Single-layer Dense Masked Network.

  Attributes:
    NUM_FEATURES: The number of neurons in the single dense layer.
  """

  NUM_FEATURES: int = 16

  def apply(self,
            inputs,
            mask = None):
    inputs = inputs.reshape(inputs.shape[0], -1)
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_0'] if mask is not None else None)


class MaskedConv(flax.deprecated.nn.Module):
  """Single-layer Conv Masked Network.

  Attributes:
    NUM_FEATURES: The number of filters in the single conv layer.
  """

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


class MaskedTwoLayerDense(flax.deprecated.nn.Module):
  """Two-layer Dense Masked Network.

  Attributes:
    NUM_FEATURES: The number of neurons in the dense layers.
  """

  NUM_FEATURES: Sequence[int] = (16, 32)

  def apply(self,
            inputs,
            mask = None):
    inputs = inputs.reshape(inputs.shape[0], -1)
    inputs = masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[0],
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_0'] if mask is not None else None)
    inputs = flax.deprecated.nn.relu(inputs)
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES[1],
        wrapped_module=flax.deprecated.nn.Dense,
        mask=mask['MaskedModule_1'] if mask is not None else None)


class SymmetryTest(parameterized.TestCase):
  """Tests symmetry analysis methods."""

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._batch_size = 2
    self._input_shape = ((self._batch_size, 2, 2, 1), jnp.float32)
    self._flat_input_shape = ((self._batch_size, 2 * 2 * 1), jnp.float32)

    _, initial_params = MaskedDense.init_by_shape(self._rng,
                                                  (self._flat_input_shape,))
    self._masked_model = flax.deprecated.nn.Model(MaskedDense, initial_params)

    _, initial_params = MaskedConv.init_by_shape(self._rng,
                                                 (self._input_shape,))
    self._masked_conv_model = flax.deprecated.nn.Model(MaskedConv,
                                                       initial_params)

    _, initial_params = MaskedTwoLayerDense.init_by_shape(
        self._rng, (self._flat_input_shape,))
    self._masked_two_layer_model = flax.deprecated.nn.Model(
        MaskedTwoLayerDense, initial_params)

  def test_count_permutations_layer_mask_full(self):
    """Tests count of weight permutations in a full mask."""
    mask_layer = {
        'kernel':
            jnp.ones(self._masked_model.params['MaskedModule_0']['unmasked']
                     ['kernel'].shape),
    }

    stats = symmetry.count_permutations_mask_layer(mask_layer)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 1)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'],
                       math.factorial(MaskedDense.NUM_FEATURES))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedDense.NUM_FEATURES)

  def test_count_permutations_layer_mask_empty(self):
    """Tests count of weight permutations in an empty mask."""
    mask_layer = {
        'kernel':
            jnp.zeros(self._masked_model.params['MaskedModule_0']['unmasked']
                      ['kernel'].shape),
    }

    stats = symmetry.count_permutations_mask_layer(mask_layer)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 0)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 0)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], MaskedDense.NUM_FEATURES)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedDense.NUM_FEATURES)

  def test_count_permutations_conv_layer_mask_full(self):
    """Tests count of weight permutations in a full mask for a conv. layer."""
    mask_layer = {
        'kernel':
            jnp.ones(self._masked_conv_model.params['MaskedModule_0']
                     ['unmasked']['kernel'].shape),
    }

    stats = symmetry.count_permutations_mask_layer(mask_layer)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 1)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'],
                       math.factorial(MaskedConv.NUM_FEATURES))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_conv_layer_mask_empty(self):
    """Tests count of weight permutations in an empty mask for a conv. layer."""
    mask_layer = {
        'kernel':
            jnp.zeros(self._masked_conv_model.params['MaskedModule_0']
                      ['unmasked']['kernel'].shape),
    }

    stats = symmetry.count_permutations_mask_layer(mask_layer)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 0)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 0)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], MaskedConv.NUM_FEATURES)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_layer_mask_known_perm(self):
    """Tests count of weight permutations in a mask with known # permutations."""
    param_shape = self._masked_model.params['MaskedModule_0']['unmasked'][
        'kernel'].shape

    # Create two unique random mask rows.
    row_type_one = jax.random.bernoulli(
        self._rng, p=0.3, shape=(param_shape[0],)).astype(jnp.int32)
    row_type_two = jax.random.bernoulli(
        self._rng, p=0.9, shape=(param_shape[0],)).astype(jnp.int32)

    # Create mask by repeating the two unique rows.
    repeat_one = param_shape[-1] // 3
    repeat_two = param_shape[-1] - repeat_one
    mask_layer = {'kernel': jnp.concatenate(
        (jnp.repeat(row_type_one[:, jnp.newaxis], repeat_one, axis=-1),
         jnp.repeat(row_type_two[:, jnp.newaxis], repeat_two, axis=-1)),
        axis=-1)}

    stats = symmetry.count_permutations_mask_layer(mask_layer)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 2)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'],
                       math.factorial(repeat_one) * math.factorial(repeat_two))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], param_shape[-1])

  def test_count_permutations_layer_mask_known_perm_zeros(self):
    """Tests count of weight permutations in a mask with zeroed neurons."""
    param_shape = self._masked_model.params['MaskedModule_0']['unmasked'][
        'kernel'].shape

    # Create two unique random mask rows.
    row_type_one = jax.random.bernoulli(
        self._rng, p=0.3, shape=(param_shape[0],)).astype(jnp.int32)
    row_type_two = jnp.zeros(shape=(param_shape[0],), dtype=jnp.int32)

    # Create mask by repeating the two unique rows.
    repeat_one = param_shape[-1] // 3
    repeat_two = param_shape[-1] - repeat_one
    mask_layer = {'kernel': jnp.concatenate(
        (jnp.repeat(row_type_one[:, jnp.newaxis], repeat_one, axis=-1),
         jnp.repeat(row_type_two[:, jnp.newaxis], repeat_two, axis=-1)),
        axis=-1)}

    stats = symmetry.count_permutations_mask_layer(mask_layer)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 1)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], math.factorial(repeat_one))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], repeat_two)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], param_shape[-1])

  def test_count_permutations_shuffled_full_mask(self):
    """Tests count of weight permutations on a generated full mask."""
    mask = masked.shuffled_mask(self._masked_model, rng=self._rng, sparsity=1)

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 0)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 0)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], MaskedConv.NUM_FEATURES)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_shuffled_empty_mask(self):
    """Tests count of weight permutations on a generated empty mask."""
    mask = masked.shuffled_mask(self._masked_model, rng=self._rng, sparsity=0)

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 1)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'],
                       math.factorial(MaskedConv.NUM_FEATURES))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_mask_layer_twolayer_known_symmetric(self):
    """Tests count of permutations in a known mask with 2 permutations."""
    mask = {
        'MaskedModule_0': {
            'kernel': jnp.array(((1, 0), (1, 0), (0, 1))).T,
        },
        'MaskedModule_1': {
            'kernel': jnp.array(((1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 0))).T,
        },
    }

    stats = symmetry.count_permutations_mask_layer(mask['MaskedModule_0'],
                                                   mask['MaskedModule_1'])

    with self.subTest(name='count_permutations_unique'):
      self.assertEqual(stats['unique_neurons'], 2)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 2)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'],
                       mask['MaskedModule_0']['kernel'].shape[-1])

  # Note: Can't pass jnp.array here since global, InitGoogle() not called yet.
  @parameterized.parameters(
      # Tests mask with 1 permutation only if both layers are considered.
      ({
          'MaskedModule_0': {
              'kernel': np.array(((1, 0), (1, 0), (0, 1))).T,
          },
          'MaskedModule_1': {
              'kernel':
                  np.array(((1, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0))).T,
          },
      }, 3, 1, 0, 3),
      # Tests mask zero count with an ablated neuron in first layer.
      ({
          'MaskedModule_0': {
              'kernel': np.array(((1, 0), (1, 0), (0, 0))).T,
          },
          'MaskedModule_1': {
              'kernel':
                  np.array(((1, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0))).T,
          },
      }, 2, 1, 1, 3),
      # Tests mask zero count with first layer completely ablated.
      ({
          'MaskedModule_0': {
              'kernel': np.array(((0, 0), (0, 0), (0, 0))).T,
          },
          'MaskedModule_1': {
              'kernel':
                  np.array(((1, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0))).T,
          },
      }, 0, 0, 3, 3),
      # Tests mask zero count with second layer completely ablated.
      ({
          'MaskedModule_0': {
              'kernel': np.array(((1, 0), (1, 0), (0, 1))).T,
          },
          'MaskedModule_1': {
              'kernel':
                  np.array(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))).T,
          },
      }, 0, 0, 3, 3),
      # """Tests layer 1 permutation matrix mask, having only 1 permutation."""
      ({
          'MaskedModule_0': {
              'kernel': np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))).T,
          },
          'MaskedModule_1': {
              'kernel':
                  np.array(((1, 1, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1))).T,
          },
      }, 3, 1, 0, 3),
      )
  def test_count_permutations_mask_layer_twolayer(self, mask, unique,
                                                  permutations, zeroed, total):
    """Test mask permutations if both layers are considered."""
    stats = symmetry.count_permutations_mask_layer(mask['MaskedModule_0'],
                                                   mask['MaskedModule_1'])

    with self.subTest(name='count_permutations_unique'):
      self.assertEqual(stats['unique_neurons'], unique)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], permutations)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], zeroed)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], total)

  def test_count_permutations_mask_full(self):
    """Tests count of weight permutations in a full mask."""
    mask = masked.simple_mask(self._masked_model, jnp.ones, ['kernel'])

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 1)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'],
                       math.factorial(MaskedDense.NUM_FEATURES))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_mask_bn_layer_full(self):
    """Tests count of permutations on a mask for model with non-masked layers."""
    mask = masked.simple_mask(self._masked_model, jnp.ones, ['kernel'])

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 1)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'],
                       math.factorial(MaskedDense.NUM_FEATURES))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_mask_empty(self):
    """Tests count of weight permutations in an empty mask."""
    mask = masked.simple_mask(self._masked_model, jnp.zeros, ['kernel'])

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 0)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 0)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], MaskedConv.NUM_FEATURES)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'], MaskedConv.NUM_FEATURES)

  def test_count_permutations_mask_twolayer_full(self):
    """Tests count of weight permutations in a full mask for 2 layers."""
    mask = masked.simple_mask(self._masked_two_layer_model, jnp.ones,
                              ['kernel'])

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 2)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(
          stats['permutations'],
          functools.reduce(
              operator.mul,
              [math.factorial(x) for x in MaskedTwoLayerDense.NUM_FEATURES]))

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 0)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'],
                       sum(MaskedTwoLayerDense.NUM_FEATURES))

  def test_count_permutations_mask_twolayers_empty(self):
    """Tests count of weight permutations in an empty mask for 2 layers."""
    mask = masked.simple_mask(self._masked_two_layer_model, jnp.zeros,
                              ['kernel'])

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 0)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 0)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'],
                       sum(MaskedTwoLayerDense.NUM_FEATURES))

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(stats['total_neurons'],
                       sum(MaskedTwoLayerDense.NUM_FEATURES))

  def test_count_permutations_mask_twolayer_known_symmetric(self):
    """Tests count of permutations in a known mask with 4 permutations."""
    mask = {
        'MaskedModule_0': {
            'kernel': jnp.array(((1, 0), (1, 0), (0, 1))).T
        },
        'MaskedModule_1': {
            'kernel': jnp.array(((1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 0))).T
        }
    }

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_full_mask_unique'):
      self.assertEqual(stats['unique_neurons'], 4)

    with self.subTest(name='count_permutations_full_mask_permutations'):
      self.assertEqual(stats['permutations'], 4)

    with self.subTest(name='count_permutations_full_mask_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 1)

    with self.subTest(name='Count_permutations_full_mask_total'):
      self.assertEqual(
          stats['total_neurons'], mask['MaskedModule_0']['kernel'].shape[-1] +
          mask['MaskedModule_1']['kernel'].shape[-1])

  def test_count_permutations_mask_twolayer_known_non_symmetric(self):
    """Tests mask with 1 permutation only if both layers are considered."""
    mask = {
        'MaskedModule_0': {
            'kernel': jnp.array(((1, 0), (1, 0), (0, 1))).T
        },
        'MaskedModule_1': {
            'kernel': jnp.array(((1, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0))).T
        }
    }

    stats = symmetry.count_permutations_mask(mask)

    with self.subTest(name='count_permutations_unique'):
      self.assertEqual(stats['unique_neurons'], 6)

    with self.subTest(name='count_permutations_permutations'):
      self.assertEqual(stats['permutations'], 1)

    with self.subTest(name='count_permutations_zeroed'):
      self.assertEqual(stats['zeroed_neurons'], 1)

    with self.subTest(name='count_permutations_total'):
      self.assertEqual(
          stats['total_neurons'], mask['MaskedModule_0']['kernel'].shape[-1] +
          mask['MaskedModule_1']['kernel'].shape[-1])

  def test_get_mask_stats_keys_values(self):
    """Tests the returned dict has the required keys, and value types/ranges."""
    mask = {
        'MaskedModule_0': {
            'kernel': jnp.array(((1, 0), (1, 0), (0, 1))).T
        },
        'MaskedModule_1': {
            'kernel': jnp.array(((1, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0))).T
        }
    }

    mask_stats = symmetry.get_mask_stats(mask)

    with self.subTest(name='sparsity_exists'):
      self.assertIn('sparsity', mask_stats)

    with self.subTest(name='sparsity_value'):
      self.assertBetween(mask_stats['sparsity'], 0.0, 1.0)

    with self.subTest(name='permutation_num_digits_exists'):
      self.assertIn('permutation_num_digits', mask_stats)

    with self.subTest(name='permutation_num_digits_value'):
      self.assertGreaterEqual(mask_stats['permutation_num_digits'], 0.0)

    with self.subTest(name='permutation_log10_exists'):
      self.assertIn('permutation_log10', mask_stats)

    with self.subTest(name='permutation_log10_value'):
      self.assertGreaterEqual(mask_stats['permutation_log10'], 0.0)

    with self.subTest(name='unique_neurons_exists'):
      self.assertIn('unique_neurons', mask_stats)

    with self.subTest(name='unique_neurons_value'):
      self.assertEqual(mask_stats['unique_neurons'], 6)

    with self.subTest(name='permutations_exists'):
      self.assertIn('permutations', mask_stats)

    with self.subTest(name='permutations_value'):
      self.assertEqual(mask_stats['permutations'], 1)

    with self.subTest(name='zeroed_neurons_exists'):
      self.assertIn('zeroed_neurons', mask_stats)

    with self.subTest(name='zeroed_neurons_value'):
      self.assertEqual(mask_stats['zeroed_neurons'], 1)

    with self.subTest(name='total_neurons_exists'):
      self.assertIn('total_neurons', mask_stats)

    with self.subTest(name='total_neurons_value'):
      self.assertEqual(mask_stats['total_neurons'],
                       mask['MaskedModule_0']['kernel'].shape[-1] +
                       mask['MaskedModule_1']['kernel'].shape[-1])

if __name__ == '__main__':
  absltest.main()
