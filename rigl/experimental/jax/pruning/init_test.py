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

"""Tests for weight_symmetry.pruning.init."""
from typing import Any, Mapping, Optional

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import init
from rigl.experimental.jax.pruning import masked


class MaskedDense(flax.deprecated.nn.Module):
  """Single-layer Dense Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self,
            inputs,
            mask = None):
    inputs = inputs.reshape(inputs.shape[0], -1)

    layer_mask = mask['MaskedModule_0'] if mask else None
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        wrapped_module=flax.deprecated.nn.Dense,
        mask=layer_mask,
        kernel_init=flax.deprecated.nn.initializers.kaiming_normal())


class MaskedDenseSparseInit(flax.deprecated.nn.Module):
  """Single-layer Dense Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self,
            inputs,
            *args,
            mask = None,
            **kwargs):
    inputs = inputs.reshape(inputs.shape[0], -1)

    layer_mask = mask['MaskedModule_0'] if mask else None
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        wrapped_module=flax.deprecated.nn.Dense,
        mask=layer_mask,
        kernel_init=init.kaiming_sparse_normal(
            layer_mask['kernel'] if layer_mask is not None else None),
        **kwargs)


class MaskedCNN(flax.deprecated.nn.Module):
  """Single-layer CNN Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self,
            inputs,
            mask = None):

    layer_mask = mask['MaskedModule_0'] if mask else None
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        wrapped_module=flax.deprecated.nn.Conv,
        kernel_size=(3, 3),
        mask=layer_mask,
        kernel_init=flax.deprecated.nn.initializers.kaiming_normal())


class MaskedCNNSparseInit(flax.deprecated.nn.Module):
  """Single-layer CNN Masked Network."""

  NUM_FEATURES: int = 32

  def apply(self,
            inputs,
            *args,
            mask = None,
            **kwargs):

    layer_mask = mask['MaskedModule_0'] if mask else None
    return masked.MaskedModule(
        inputs,
        features=self.NUM_FEATURES,
        wrapped_module=flax.deprecated.nn.Conv,
        kernel_size=(3, 3),
        mask=layer_mask,
        kernel_init=init.kaiming_sparse_normal(
            layer_mask['kernel'] if layer_mask is not None else None),
        **kwargs)


class InitTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = jax.random.PRNGKey(42)
    self._batch_size = 2
    self._input_shape = ((self._batch_size, 28, 28, 1), jnp.float32)
    self._input = jnp.ones(*self._input_shape)

  def test_init_kaiming_sparse_normal_output(self):
    """Tests the output shape/type of kaiming normal sparse initialization."""
    input_array = jnp.ones((64, 16), jnp.float32)
    mask = jax.random.bernoulli(self._rng, shape=(64, 16))

    base_init = flax.deprecated.nn.initializers.kaiming_normal()(
        self._rng, input_array.shape, input_array.dtype)
    sparse_init = init.kaiming_sparse_normal(mask)(self._rng, input_array.shape,
                                                   input_array.dtype)

    with self.subTest(name='test_sparse_init_output_shape'):
      self.assertSequenceEqual(sparse_init.shape, base_init.shape)

    with self.subTest(name='test_sparse_init_output_dtype'):
      self.assertEqual(sparse_init.dtype, base_init.dtype)

    with self.subTest(name='test_sparse_init_output_notallzero'):
      self.assertTrue((sparse_init != 0).any())

  def test_dense_no_mask(self):
    """Checks that in the special case of no mask, init is same as base_init."""
    _, initial_params = MaskedDense.init_by_shape(self._rng,
                                                  (self._input_shape,))
    self._unmasked_model = flax.deprecated.nn.Model(MaskedDense, initial_params)

    _, initial_params = MaskedDenseSparseInit.init_by_shape(
        jax.random.PRNGKey(42), (self._input_shape,), mask=None)
    self._masked_model_sparse_init = flax.deprecated.nn.Model(
        MaskedDenseSparseInit, initial_params)

    self.assertTrue(
        jnp.isclose(
            self._masked_model_sparse_init.params['MaskedModule_0']['unmasked']
            ['kernel'], self._unmasked_model.params['MaskedModule_0']
            ['unmasked']['kernel']).all())

  def test_dense_sparse_init_kaiming(self):
    """Checks kaiming normal sparse initialization for dense layer."""
    _, initial_params = MaskedDense.init_by_shape(self._rng,
                                                  (self._input_shape,))
    self._unmasked_model = flax.deprecated.nn.Model(MaskedDense, initial_params)

    mask = masked.simple_mask(self._unmasked_model, jnp.ones,
                              masked.WEIGHT_PARAM_NAMES)

    _, initial_params = MaskedDenseSparseInit.init_by_shape(
        jax.random.PRNGKey(42), (self._input_shape,), mask=mask)
    self._masked_model_sparse_init = flax.deprecated.nn.Model(
        MaskedDenseSparseInit, initial_params)

    mean_init = jnp.mean(
        self._unmasked_model.params['MaskedModule_0']['unmasked']['kernel'])

    stddev_init = jnp.std(
        self._unmasked_model.params['MaskedModule_0']['unmasked']['kernel'])

    mean_sparse_init = jnp.mean(
        self._masked_model_sparse_init.params['MaskedModule_0']['unmasked']
        ['kernel'])

    stddev_sparse_init = jnp.std(
        self._masked_model_sparse_init.params['MaskedModule_0']['unmasked']
        ['kernel'])

    with self.subTest(name='test_cnn_sparse_init_mean'):
      self.assertBetween(mean_sparse_init, mean_init - 2 * stddev_init,
                         mean_init + 2 * stddev_init)

    with self.subTest(name='test_cnn_sparse_init_stddev'):
      self.assertBetween(stddev_sparse_init, 0.5 * stddev_init,
                         1.5 * stddev_init)

  def test_cnn_sparse_init_kaiming(self):
    """Checks kaiming normal sparse initialization for convolutional layer."""
    _, initial_params = MaskedCNN.init_by_shape(self._rng, (self._input_shape,))
    self._unmasked_model = flax.deprecated.nn.Model(MaskedCNN, initial_params)

    mask = masked.simple_mask(self._unmasked_model, jnp.ones,
                              masked.WEIGHT_PARAM_NAMES)

    _, initial_params = MaskedCNNSparseInit.init_by_shape(
        jax.random.PRNGKey(42), (self._input_shape,), mask=mask)
    self._masked_model_sparse_init = flax.deprecated.nn.Model(
        MaskedCNNSparseInit, initial_params)

    mean_init = jnp.mean(
        self._unmasked_model.params['MaskedModule_0']['unmasked']['kernel'])

    stddev_init = jnp.std(
        self._unmasked_model.params['MaskedModule_0']['unmasked']['kernel'])

    mean_sparse_init = jnp.mean(
        self._masked_model_sparse_init.params['MaskedModule_0']['unmasked']
        ['kernel'])

    stddev_sparse_init = jnp.std(
        self._masked_model_sparse_init.params['MaskedModule_0']['unmasked']
        ['kernel'])

    with self.subTest(name='test_cnn_sparse_init_mean'):
      self.assertBetween(mean_sparse_init, mean_init - 2 * stddev_init,
                         mean_init + 2 * stddev_init)

    with self.subTest(name='test_cnn_sparse_init_stddev'):
      self.assertBetween(stddev_sparse_init, 0.5 * stddev_init,
                         1.5 * stddev_init)


if __name__ == '__main__':
  absltest.main()
