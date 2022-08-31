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

"""Tests for weight_symmetry.training.training."""
import functools
import math

from absl.testing import absltest
import flax
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
import jax.numpy as jnp

from rigl.experimental.jax.datasets import dataset_factory
from rigl.experimental.jax.models import model_factory
from rigl.experimental.jax.training import training


class TrainingTest(absltest.TestCase):
  """Tests functions for training loop and training convenience functions."""

  def setUp(self):
    super().setUp()

    self._batch_size = 128  # Note: Tests are run on GPU/TPU.
    self._batch_size_test = 128
    self._shuffle_buffer_size = 1024
    self._rng = jax.random.PRNGKey(42)
    self._input_shape = ((self._batch_size, 28, 28, 1), jnp.float32)
    self._num_classes = 10
    self._num_epochs = 1

    self._learning_rate_fn = lambda _: 0.01
    self._weight_decay = 0.0001
    self._momentum = 0.9
    self._rng = jax.random.PRNGKey(42)

    self._min_loss = jnp.finfo(float).eps
    self._max_loss = 2.0 * math.log(self._num_classes)

    self._dataset_name = 'MNIST'
    self._model_name = 'MNIST_CNN'

    self._summarywriter = tensorboard.SummaryWriter('/tmp/')

    self._dataset = dataset_factory.create_dataset(
        self._dataset_name,
        self._batch_size,
        self._batch_size_test,
        shuffle_buffer_size=self._shuffle_buffer_size)

    self._model, self._state = model_factory.create_model(
        self._model_name,
        self._rng, (self._input_shape,),
        num_classes=self._num_classes)

    self._optimizer = flax.optim.Momentum(  # pytype: disable=module-attr
        learning_rate=self._learning_rate_fn(0),
        beta=self._momentum,
        weight_decay=self._weight_decay)

  def test_train_one_step(self):
    """Tests training loop over one step."""
    iterator = self._dataset.get_train()
    batch = next(iterator)

    state = jax_utils.replicate(self._state)
    optimizer = jax_utils.replicate(self._optimizer.create(self._model))

    self._rng, step_key = jax.random.split(self._rng)
    batch = training._shard_batch(batch)
    sharded_keys = common_utils.shard_prng_key(step_key)

    p_train_step = jax.pmap(
        functools.partial(
            training.train_step, learning_rate_fn=self._learning_rate_fn),
        axis_name='batch')
    _, _, loss, gradient_norm = p_train_step(optimizer, batch, sharded_keys,
                                             state)

    loss = jnp.mean(loss)
    gradient_norm = jax_utils.unreplicate(gradient_norm)

    with self.subTest(name='test_loss_range'):
      self.assertBetween(loss, self._min_loss, self._max_loss)

    with self.subTest(name='test_gradient_norm'):
      self.assertGreaterEqual(gradient_norm, 0)

  def test_train_one_epoch(self):
    """Tests training loop over one epoch."""
    trainer = training.Trainer(self._optimizer, self._model, self._state,
                               self._dataset)

    with self.subTest(name='trainer_instantiation'):
      self.assertIsInstance(trainer, training.Trainer)

    best_model, best_metrics = trainer.train(self._num_epochs)

    with self.subTest(name='best_model_type'):
      self.assertIsInstance(best_model, flax.deprecated.nn.Model)

    with self.subTest(name='train_accuracy'):
      self.assertBetween(best_metrics['train_accuracy'], 0., 1.)

    with self.subTest(name='train_avg_loss'):
      self.assertBetween(best_metrics['train_avg_loss'], self._min_loss,
                         self._max_loss)

    with self.subTest(name='step'):
      self.assertGreater(best_metrics['step'], 0)

    with self.subTest(name='cosine_distance'):
      self.assertBetween(best_metrics['cosine_distance'], 0., 1.)

    with self.subTest(name='cumulative_gradient_norm'):
      self.assertGreater(best_metrics['cumulative_gradient_norm'], 0)

    with self.subTest(name='test_accuracy'):
      self.assertBetween(best_metrics['test_accuracy'], 0., 1.)

    with self.subTest(name='test_avg_loss'):
      self.assertBetween(best_metrics['test_avg_loss'], self._min_loss,
                         self._max_loss)

  def test_train_one_epoch_tensorboard(self):
    """Tests training loop over one epoch, with tensorboard."""

    trainer = training.Trainer(
        self._optimizer,
        self._model,
        self._state,
        self._dataset,
        summary_writer=self._summarywriter)

    with self.subTest(name='TrainerInstantiation'):
      self.assertIsInstance(trainer, training.Trainer)

    best_model, best_metrics = trainer.train(self._num_epochs)
    with self.subTest(name='best_model_type'):
      self.assertIsInstance(best_model, flax.deprecated.nn.Model)

    with self.subTest(name='train_accuracy'):
      self.assertBetween(best_metrics['train_accuracy'], 0., 1.)

    with self.subTest(name='train_avg_loss'):
      self.assertBetween(best_metrics['train_avg_loss'], self._min_loss,
                         self._max_loss)

    with self.subTest(name='step'):
      self.assertGreater(best_metrics['step'], 0)

    with self.subTest(name='cosine_distance'):
      self.assertBetween(best_metrics['cosine_distance'], 0., 1.)

    with self.subTest(name='cumulative_gradient_norm'):
      self.assertGreater(best_metrics['cumulative_gradient_norm'], 0)

    with self.subTest(name='test_accuracy'):
      self.assertBetween(best_metrics['test_accuracy'], 0., 1.)

    with self.subTest(name='test_avg_loss'):
      self.assertBetween(best_metrics['test_avg_loss'], self._min_loss,
                         self._max_loss)

  def test_train_one_epoch_pruning_global_schedule(self):
    """Tests training loop over one epoch with global pruning rate schedule."""
    trainer = training.Trainer(self._optimizer, self._model, self._state,
                               self._dataset)

    with self.subTest(name='trainer_instantiation'):
      self.assertIsInstance(trainer, training.Trainer)

    best_model, best_metrics = trainer.train(self._num_epochs,
                                             pruning_rate_fn=lambda _: 0.5)

    with self.subTest(name='best_model_type'):
      self.assertIsInstance(best_model, flax.deprecated.nn.Model)

    with self.subTest(name='train_accuracy'):
      self.assertBetween(best_metrics['train_accuracy'], 0., 1.)

    with self.subTest(name='train_avg_loss'):
      self.assertBetween(best_metrics['train_avg_loss'], self._min_loss,
                         self._max_loss)

    with self.subTest(name='step'):
      self.assertGreater(best_metrics['step'], 0)

    with self.subTest(name='cosine_distance'):
      self.assertBetween(best_metrics['cosine_distance'], 0., 1.)

    with self.subTest(name='cumulative_gradient_norm'):
      self.assertGreater(best_metrics['cumulative_gradient_norm'], 0.)

    with self.subTest(name='test_accuracy'):
      self.assertBetween(best_metrics['test_accuracy'], 0., 1.)

    with self.subTest(name='test_avg_loss'):
      self.assertBetween(best_metrics['test_avg_loss'], self._min_loss,
                         self._max_loss)

  def test_train_one_epoch_pruning_local_schedule(self):
    """Tests training loop over one epoch with local pruning rate schedule."""
    trainer = training.Trainer(self._optimizer, self._model, self._state,
                               self._dataset)

    with self.subTest(name='trainer_instantiation'):
      self.assertIsInstance(trainer, training.Trainer)

    best_model, best_metrics = trainer.train(
        self._num_epochs, pruning_rate_fn={'MaskedModule_0': lambda _: 0.5})

    with self.subTest(name='best_model_type'):
      self.assertIsInstance(best_model, flax.deprecated.nn.Model)

    with self.subTest(name='train_accuracy'):
      self.assertBetween(best_metrics['train_accuracy'], 0., 1.)

    with self.subTest(name='train_avg_loss'):
      self.assertBetween(best_metrics['train_avg_loss'], self._min_loss,
                         self._max_loss)

    with self.subTest(name='step'):
      self.assertGreater(best_metrics['step'], 0)

    with self.subTest(name='cosine_distance'):
      self.assertBetween(best_metrics['cosine_distance'], 0., 1.)

    with self.subTest(name='cumulative_gradient_norm'):
      self.assertGreater(best_metrics['cumulative_gradient_norm'], 0.)

    with self.subTest(name='test_accuracy'):
      self.assertBetween(best_metrics['test_accuracy'], 0., 1.)

    with self.subTest(name='test_avg_loss'):
      self.assertBetween(best_metrics['test_avg_loss'], self._min_loss,
                         self._max_loss)

  def test_eval_batch(self):
    """Tests model per-batch evaluation function."""
    state = jax_utils.replicate(self._state)
    optimizer = jax_utils.replicate(self._optimizer.create(self._model))

    iterator = self._dataset.get_test()
    batch = next(iterator)
    batch = training._shard_batch(batch)

    metrics = jax.pmap(training._eval_step, axis_name='batch')(
        optimizer.target, state, batch)

    loss = jnp.mean(metrics['loss'])
    accuracy = jnp.mean(metrics['accuracy'])

    with self.subTest(name='test_eval_batch_loss'):
      self.assertBetween(loss, self._min_loss, self._max_loss)

    with self.subTest(name='test_eval_batch_accuracy'):
      self.assertBetween(accuracy, 0., 1.)

  def test_eval(self):
    """Tests model evaluation function."""
    state = jax_utils.replicate(self._state)
    optimizer = jax_utils.replicate(self._optimizer.create(self._model))

    metrics = training.eval_model(optimizer.target, state,
                                  self._dataset.get_test())

    loss = metrics['loss']
    accuracy = metrics['accuracy']

    with self.subTest(name='test_eval_loss'):
      self.assertBetween(loss, 0., 2.0*math.log(self._num_classes))

    with self.subTest(name='test_eval_accuracy'):
      self.assertBetween(accuracy, 0., 1.)

if __name__ == '__main__':
  absltest.main()
