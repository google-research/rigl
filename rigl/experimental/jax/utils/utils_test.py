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

"""Tests for weight_symmetry.nn.nn_functions."""
import functools
import json
import operator
import tempfile
from typing import Optional, Sequence, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np

from rigl.experimental.jax.training import training
from rigl.experimental.jax.utils import utils


class TwoLayerDense(flax.deprecated.nn.Module):
  """Two-layer Dense Network."""

  NUM_FEATURES: Sequence[int] = (32, 64)

  def apply(self, inputs):
    # If inputs are in image dimensions, flatten image.
    inputs = inputs.reshape(inputs.shape[0], -1)

    inputs = flax.deprecated.nn.Dense(inputs, features=self.NUM_FEATURES[0])
    return flax.deprecated.nn.Dense(inputs, features=self.NUM_FEATURES[1])


class UtilsTest(parameterized.TestCase):
  """Test functions for NN convenience functions."""

  def setUp(self):
    """Common setup for test cases."""
    super().setUp()
    self._batch_size = 2
    self._num_classes = 10
    self._true_logit = 0.5
    self._input_shape = ((self._batch_size, 28, 28, 1), jnp.float32)
    self._input = jnp.ones(*self._input_shape)

    self._rng = jax.random.PRNGKey(42)
    _, initial_params = TwoLayerDense.init_by_shape(self._rng,
                                                    (self._input_shape,))
    self._model = flax.deprecated.nn.Model(TwoLayerDense, initial_params)
    _, initial_params = TwoLayerDense.init_by_shape(self._rng,
                                                    (self._input_shape,))
    self._model_diff_init = flax.deprecated.nn.Model(TwoLayerDense,
                                                     initial_params)

  def _create_logits_labels(self, correct):
    """Creates a set of logits/labels resulting from correct classification.

    Args:
      correct: If true, creates labels for a correct classifiction, otherwise
        creates labels for an incorrect classification.
    Returns:
      A tuple of logits, labels.
    """
    logits = np.full((self._batch_size, self._num_classes),
                     (1.0 - self._true_logit) / self._num_classes,
                     dtype=np.float32)

    # Diagonal over batch will be true.
    for i in range(self._batch_size):
      logits[i, i % self._num_classes] = self._true_logit

    labels = np.zeros(self._batch_size, dtype=jnp.int32)

    # Diagonal over batch will be true.
    for i in range(self._batch_size):
      labels[i] = (i if correct else i + 1) % self._num_classes

    return jnp.array(logits), jnp.array(labels)

  def test_compute_metrics_correct(self):
    """Tests output when logit outputs indicate correct classification."""
    logits, labels_correct = self._create_logits_labels(True)
    logits = training._shard_batch(logits)
    labels_correct = training._shard_batch(labels_correct)

    p_compute_metrics = jax.pmap(utils.compute_metrics, axis_name='batch')
    metrics = p_compute_metrics(logits, labels_correct)
    loss = metrics['loss']
    accuracy = metrics['accuracy']

    with self.subTest(name='loss_type'):
      self.assertIsInstance(loss, jnp.ndarray)

    with self.subTest(name='loss_len'):
      self.assertEqual(loss.size, 1)

    with self.subTest(name='loss_values'):
      self.assertGreaterEqual(loss.all(), 0)

    with self.subTest(name='accuracy_type'):
      self.assertIsInstance(accuracy, jnp.ndarray)

    with self.subTest(name='accuracy_Len'):
      self.assertEqual(accuracy.size, 1)

    with self.subTest(name='accuracy_values'):
      self.assertAlmostEqual(accuracy.all(), 1.0)

  def test_compute_metrics_incorrect(self):
    """Tests output when logit outputs indicate incorrect classification."""
    logits, labels_incorrect = self._create_logits_labels(False)
    logits = training._shard_batch(logits)
    labels_incorrect = training._shard_batch(labels_incorrect)

    p_compute_metrics = jax.pmap(utils.compute_metrics, axis_name='batch')
    metrics = p_compute_metrics(logits, labels_incorrect)
    loss = metrics['loss']
    accuracy = metrics['accuracy']

    with self.subTest(name='loss_type'):
      self.assertIsInstance(loss, jnp.ndarray)

    with self.subTest(name='loss_len'):
      self.assertEqual(loss.size, 1)

    with self.subTest(name='loss_values'):
      self.assertGreaterEqual(loss.all(), 0)

    with self.subTest(name='accuracy_type'):
      self.assertIsInstance(accuracy, jnp.ndarray)

    with self.subTest(name='accuracy_len'):
      self.assertEqual(accuracy.size, 1)

    with self.subTest(name='accuracy_values'):
      self.assertAlmostEqual(accuracy.all(), 0.0)

  def test_compute_metrics_equal_logits(self):
    """Tests output when the logit outputs are equal for all classes."""
    logits, labels_correct = self._create_logits_labels(True)
    logits = training._shard_batch(logits)
    labels_correct = training._shard_batch(labels_correct)

    p_compute_metrics = jax.pmap(utils.compute_metrics, axis_name='batch')
    metrics = p_compute_metrics(logits, labels_correct)
    loss = metrics['loss']
    accuracy = metrics['accuracy']

    with self.subTest(name='loss_type'):
      self.assertIsInstance(loss, jnp.ndarray)

    with self.subTest(name='loss_len'):
      self.assertEqual(loss.size, 1)

    with self.subTest(name='loss_values'):
      self.assertGreaterEqual(loss.all(), 0)

    with self.subTest(name='accuracy_type'):
      self.assertIsInstance(accuracy, jnp.ndarray)

    with self.subTest(name='accuracy_len'):
      self.assertEqual(accuracy.size, 1)

    with self.subTest(name='accuracy_values'):
      self.assertAlmostEqual(accuracy.all(), 1.0)

  def test_dump_dict_json(self):
    """Tests JSON dumping function."""
    data_dict = {
        'np_float': np.dtype('float32').type(1.0),
        'jnp_float': jnp.dtype('float32').type(1.0),
        'np_int': np.dtype('int32').type(1),
        'jnp_int': jnp.dtype('int32').type(1),
        'np_array': np.array(1.0, dtype=np.float32),
        'jnp_array': jnp.array(1.0, dtype=jnp.float32),
    }
    converted_dict = {
        key: utils._np_converter(value) for key, value in data_dict.items()
    }
    json_path = tempfile.NamedTemporaryFile()
    utils.dump_dict_json(data_dict, json_path.name)

    with open(json_path.name, 'r') as input_file:
      loaded_dict = json.load(input_file)
    self.assertDictEqual(loaded_dict, converted_dict)

  def test_count_param_two_layer_dense(self):
    """Tests model parameter counting on small FC model."""
    count = utils.count_param(self._model, ('kernel',))

    self.assertEqual(
        count,
        self._input.size / self._batch_size * TwoLayerDense.NUM_FEATURES[0] +
        TwoLayerDense.NUM_FEATURES[0] * TwoLayerDense.NUM_FEATURES[1])

  def test_count_invalid_param(self):
    """Tests model parameter counting for a non-existent parameter name."""
    count = utils.count_param(self._model, ('not_kernel',))

    self.assertEqual(count, 0)

  def test_model_param_as_array(self):
    """Tests method for returning single parameter vector for model."""
    param_array = utils.param_as_array(self._model.params)

    with self.subTest(name='test_param_is_vector'):
      self.assertLen(param_array.shape, 1)

    param_sizes = [param.size for param in jax.tree_leaves(self._model.params)]
    model_size = functools.reduce(operator.add, param_sizes)

    with self.subTest(name='test_param_size'):
      self.assertEqual(param_array.size, model_size)

  def test_cosine_similarity_random(self):
    """Tests cosine similarity for two random weight matrices."""
    a = jax.random.normal(self._rng, (3, 4))
    b = jax.random.normal(self._rng, (3, 4))

    cosine_similarity = utils.cosine_similarity(a, b)

    with self.subTest(name='test_cosine_distance_range'):
      self.assertBetween(cosine_similarity, 0., 1.)

  def test_cosine_similarity_same(self):
    """Tests cosine similarity for the same weight matrix."""
    a = jax.random.normal(self._rng, (3, 4))

    cosine_similarity = utils.cosine_similarity(a, a)

    with self.subTest(name='test_cosine_distance_range'):
      self.assertAlmostEqual(cosine_similarity, 1., places=5)

  def test_cosine_similarity_same_model(self):
    """Tests cosine similarity for the same model."""
    cosine_dist = utils.cosine_similarity_model(self._model, self._model)

    self.assertAlmostEqual(cosine_dist, 1., places=5)

  def test_vector_difference_norm_diff_model(self):
    """Tests vector difference norm for different models."""
    vector_diff_norm = utils.vector_difference_norm_model(
        self._model, self._model_diff_init)

    self.assertGreaterEqual(vector_diff_norm, 0.)

  def test_vector_difference_norm_same_model(self):
    """Tests vector difference norm for the same model."""
    vector_diff_norm = utils.vector_difference_norm_model(
        self._model, self._model)

    self.assertAlmostEqual(vector_diff_norm, 0., places=5)

  T = TypeVar('T')
  @parameterized.parameters(

      # Tests pairwise longest iterator convenience function with list.
      ((1, 2, 3, 4), ((1, 2), (2, 3), (3, 4), (4, None))),
      # Tests pairwise longest iterator with empty input iterator.
      (iter(()), ()),
      # Tests pairwise longest iterator with single element iterator.
      ((1,), ((1, None),))
  )
  def test_pairwise_longest_list_iterator(
      self, input_sequence,
      output_sequence):
    """Tests pairwise longest iterator with list iterators."""
    output = list(utils.pairwise_longest(iter(input_sequence)))

    self.assertSequenceEqual(output, output_sequence)


if __name__ == '__main__':
  absltest.main()
