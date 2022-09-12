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

"""Common training code.

This module contains utility functions for training NN.

Attributes:
  LABELKEY: The key used to retrieve a label from the batch dictionary.
  DATAKEY: The key used to retrieve an input image from the batch dictionary.
  PruningRateFnType: Typing alias for a valid pruning rate function.
"""
from collections import abc
import functools
import time
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

from absl import logging
import flax
from flax import jax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp
from rigl.experimental.jax.datasets import dataset_base
from rigl.experimental.jax.models import model_factory
from rigl.experimental.jax.pruning import masked
from rigl.experimental.jax.pruning import pruning
from rigl.experimental.jax.pruning import symmetry
from rigl.experimental.jax.utils import utils
import tensorflow.compat.v2 as tf

LABELKEY = dataset_base.ImageDataset.LABELKEY
DATAKEY = dataset_base.ImageDataset.DATAKEY

PruningRateFnType = Union[Mapping[str, Callable[[int], float]], Callable[[int],
                                                                         float]]


def _shard_batch(xs):
  """Shards a batch for a pmap, based on the number of devices."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def train_step(
    optimizer: flax.optim.Optimizer, batch: Mapping[str, jnp.array],  # pytype: disable=module-attr
    rng: Callable[[int], jnp.array], state: flax.deprecated.nn.Collection,
    learning_rate_fn: Callable[[int], float]
) -> Tuple[flax.optim.Optimizer, flax.deprecated.nn.Collection, float, float]:  # pytype: disable=module-attr
  """Performs training for one minibatch.

  Args:
    optimizer: Optimizer to use.
    batch: Minibatch to train with.
    rng: Random number generator, i.e. jax.random.PRNGKey, to use for model
      training, e.g. dropout.
    state: Model state.
    learning_rate_fn: A function that returns the learning rate given the step.

  Returns:
    A tuple consisting of the new optimizer, new state, mini-batch loss, and
    gradient norm.
  """

  def loss_fn(
      model: flax.deprecated.nn.Model
  ) -> Tuple[float, Tuple[flax.deprecated.nn.Collection, jnp.array]]:
    """Evaluates the loss function.

    Args:
      model: The model with which to evaluate the loss.

    Returns:
      Tuple of the loss for the mini-batch, and model state.
    """
    with flax.deprecated.nn.stateful(state) as new_state:
      with flax.deprecated.nn.stochastic(rng):
        logits = model(batch[DATAKEY])
    loss = utils.cross_entropy_loss(logits, batch[LABELKEY])
    return loss, new_state

  lr = learning_rate_fn(optimizer.state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, new_state), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')

  new_opt = optimizer.apply_gradient(grad, learning_rate=lr)

  grad_norm = jnp.linalg.norm(utils.param_as_array(grad))

  return new_opt, new_state, loss, grad_norm


class Trainer:
  """Training class with the state and methods for training a neural network.

  Attributes:
    optimizer: Optimizer used for training, None if training hasn't begun.
    state: Model state used for training.
  """

  def __init__(
      self,
      optimizer_def: flax.optim.OptimizerDef,  # pytype: disable=module-attr
      initial_model: flax.deprecated.nn.Model,
      initial_state: flax.deprecated.nn.Collection,
      dataset: jnp.array,
      rng: Callable[[int], jnp.array] = None,
      summary_writer: Optional[tf.summary.SummaryWriter] = None,
  ):
    """Creates a Trainer object.

    Args:
      optimizer_def: The flax optimizer def (i.e. not instantiated with a model
        using .create) to use for training.
      initial_model: The initial model to train.
      initial_state: The initial state of the model.
      dataset: The training dataset.
      rng: Random number generator, i.e. jax.random.PRNGKey, to use for model
        training, e.g. dropout.
      summary_writer: An optional tensorboard summary writer for logging
    self._rng = rng

    if self._rng is None:
      self._rng = jax.random.PRNGKey(42)

  def _update_optimizer(self, model: flax.deprecated.nn.Model):
    """Updates the optimizer based on the given model."""
    self.optimizer = jax_utils.replicate(
        self._optimizer_def.create(model))

  def train(
      self,
      num_epochs: int,
      lr_fn: Optional[Callable[[int], float]] = None,
      pruning_rate_fn: Optional[PruningRateFnType] = None,
      update_iter: int = 100,
      update_epoch: int = 10
  ) -> Tuple[flax.deprecated.nn.Model, Mapping[str, Union[int, float, Mapping[
      str, float]]]]:
    """Trains the model over the given number of epochs.

    Args:
      num_epochs: The total number of epochs to train over.
      lr_fn: The learning rate function, takes the current iteration/step as an
        argument and returns the current learning rate, uses constant learning
        rate if no function is provided.
      pruning_rate_fn: The pruning rate function, takes the current epoch as an
        argument and returns the current pruning rate, no further pruning is
        performed during training if not set. Can be a dictionary, containing
        the pruning rate schedule functions for each layer, or a single function
        for all layers.
      update_iter: Period of iterations in which to log/update per-batch
        metrics.
      update_epoch: Period of epochs in which to log/update full training/test
        metrics.

    Returns:
      Tuple consisting of the best model found during training, and metrics.

    Raises:
      ValueError: If the batch size of the data set is not evenly divisible by
                  number of devices, or the model batch size is not the training
                  data batch size/number of jax devices.
    """
    best_test_acc = 0
    best_train_loss = None
    best_iter = None

    if lr_fn is None:
      lr_fn = lambda _: self.optimizer.optimizer_def.hyper_params.learning_rate

    host_count = jax.host_count()
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    logging.info('JAX hosts %d, devices: %d, local devices: %d', host_count,
                 device_count, local_device_count)

    # TODO Implement multi-host training.
    if host_count > 1:
      raise NotImplementedError('Multi-host training is not supported yet, '
                                'see b/155550457.')

    if self._dataset.batch_size % device_count > 0:
      raise ValueError(
          'Train batch size ({}) must be divisible by number of local devices '
          '({})'.format(self._dataset.batch_size, local_device_count))

    if self._dataset.batch_size_test % device_count > 0:
      raise ValueError(
          'Test batch size ({}) must be divisible by number of local devices '
          '({})'.format(self._dataset.batch_size_test, local_device_count))

    # Required to use state and optimizer with jax.pmap.
    state = jax_utils.replicate(self.state)
    self._update_optimizer(self._initial_model)

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=lr_fn),
        axis_name='batch')

    # Function to sync the batch statistics across replicas.
    p_synchronized_batch_stats = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')

    p_cosine_similarity = functools.partial(utils.cosine_similarity_model,
                                            self._initial_model)
    p_vector_difference_norm = functools.partial(
        utils.vector_difference_norm_model, self._initial_model)

    pruning_rate = None
    mask = None

    cumulative_grad_norm = 0

    start_time = time.time()

    # Main training loop.
    for epoch in range(num_epochs):
      if epoch % update_epoch == 0 or epoch == num_epochs - 1:
        epoch_start_time = time.time()

      # If we get different schedules for different layers.
      if isinstance(pruning_rate_fn, abc.Mapping):
        next_pruning_rate = {
            layer: layer_fn(epoch)
            for layer, layer_fn in pruning_rate_fn.items()
        }
      elif pruning_rate_fn:
        next_pruning_rate = pruning_rate_fn(epoch)

      # If pruning rate has changed/is first epoch, we need to update mask.
      # Note: pruning_rate could be zero, so must explicitly check it's None.
      if pruning_rate_fn and (pruning_rate is None or
                              pruning_rate != next_pruning_rate):

        pruning_rate = next_pruning_rate

        logging.info('[%d] Pruning Rate: %s', epoch, str(pruning_rate))

        # Unreplicate optimizer/current model, and mask.
        self.optimizer = jax_utils.unreplicate(self.optimizer)
        mask = jax_utils.unreplicate(mask) if mask else None

        # Performs pruning to get updated mask.
        mask = pruning.prune(self.optimizer.target, pruning_rate, mask=mask)

        logging.info('[%d] Mask Sparsity: %0.3f', epoch,
                     masked.mask_sparsity(mask))

        for layer, layer_mask in sorted(mask.items()):
          if layer_mask:
            logging.info('[%d] Layer: %s, Mask Sparsity: %0.3f', epoch, layer,
                         masked.mask_layer_sparsity(layer_mask))

        if jax.host_id() == 0:
          mask_stats = symmetry.get_mask_stats(mask)
          logging.info('Mask stats: %s', str(mask_stats))


          if self._summary_writer:
            for label, value in mask_stats.items():
              try:
                self._summary_writer.scalar(f'mask_{epoch}/{label}', value, 0)
              # Needed when permutations (long int) can't be cast to float32.
              except (OverflowError, ValueError):
                self._summary_writer.text(f'mask_{epoch}/{label}', str(value),
                                          0)
                logging.error(
                    'Could not write mask_%d/%s to tensorflow summary as float32'
                    ', writing as string instead.', epoch, label)

        # Creates a new optimizer, based on a new model with new mask.
        self._update_optimizer(
            model_factory.update_model(self.optimizer.target, masks=mask))

      # Begins epoch.
      for batch in self._dataset.get_train():
        # Note: Because of replicate, step has # device identical vals.
        step = jax_utils.unreplicate(self.optimizer.state.step)

        if step % update_iter == 0:
          batch_start_time = time.time()

        # These are required for pmap call.
        self._rng, step_key = jax.random.split(self._rng)
        batch = _shard_batch(batch)
        sharded_keys = common_utils.shard_prng_key(step_key)

        (self.optimizer, state, opt_loss,
         grad_norm) = p_train_step(self.optimizer, batch, sharded_keys, state)

        if state.state:
          state = p_synchronized_batch_stats(state)

        grad_norm = jax_utils.unreplicate(grad_norm)

        cumulative_grad_norm += grad_norm

        # Per-iteration status/metrics update.
        if jax.host_id() == 0 and step % update_iter == 0:
          batch_time = time.time() - batch_start_time

          if self._summary_writer is not None:
            self._summary_writer.scalar('training/train_batch_loss',
                                        jnp.mean(opt_loss),
                                        step)
            self._summary_writer.scalar('training/gradient_norm', grad_norm,
                                        step)
          logging.info('[epoch %d] %d, loss %0.5f, lr %0.3f, %0.3f sec', epoch,
                       step, jnp.mean(opt_loss), lr_fn(step), batch_time)

      # Per-epoch status/metrics update.
      if (jax.host_id() == 0 and
          (epoch % update_epoch == 0 or epoch == num_epochs - 1)):
        epoch_time = time.time() - epoch_start_time

        cosine_distance = p_cosine_similarity(
            jax_utils.unreplicate(self.optimizer.target))
        vector_difference_norm = p_vector_difference_norm(
            jax_utils.unreplicate(self.optimizer.target))

        train_metrics = eval_model(self.optimizer.target, state,
                                   self._dataset.get_train())
        test_metrics = eval_model(self.optimizer.target, state,
                                  self._dataset.get_test())

        train_loss = train_metrics['loss']
        train_acc = train_metrics['accuracy']

        test_loss = test_metrics['loss']
        test_acc = test_metrics['accuracy']

        if jax.host_id() == 0:
          metrics = {
              'wallclock_time':
                  float(epoch_time),
              'train_accuracy':
                  float(train_acc),
              'train_avg_loss':
                  float(train_loss),
              'test_accuracy':
                  float(test_acc),
              'test_avg_loss':
                  float(test_loss),
              'lr':
                  float(lr_fn(step)),
              'cosine_distance':
                  float(cosine_distance),
              'cumulative_gradient_norm':
                  float(cumulative_grad_norm),
              'vector_difference_norm':
                  float(vector_difference_norm),
          }


          if self._summary_writer is not None:
            for label, value in metrics.items():
              self._summary_writer.scalar('training/{}'.format(label), value,
                                          step)

        if test_acc >= best_test_acc:
          best_model = self.optimizer.target

          best_test_acc = test_acc
          best_test_metrics = {
              'train_avg_loss': float(train_loss),
              'train_accuracy': float(train_acc),
              'test_avg_loss': float(test_loss),
              'test_accuracy': float(test_acc),
              'step': int(step),
              'cosine_distance': float(cosine_distance),
              'cumulative_gradient_norm': float(cumulative_grad_norm),
              'vector_difference_norm': float(vector_difference_norm),
          }
          best_iter = step

        if best_train_loss is None or train_loss <= best_train_loss:
          best_train_loss = train_loss
          best_train_metrics = {
              'train_avg_loss': float(train_loss),
              'train_accuracy': float(train_acc),
              'test_avg_loss': float(test_loss),
              'test_accuracy': float(test_acc),
              'step': int(step),
              'cosine_distance': float(cosine_distance),
              'cumulative_gradient_norm': float(cumulative_grad_norm),
              'vector_difference_norm': float(vector_difference_norm),
          }

        log_format_str = (
            '[epoch %d] train avg. loss %0.4f, train acc. %0.4f, test avg. '
            'loss %0.4f, test acc. %0.4f, %0.4f sec, cosine sim.: %0.3f, cum. '
            'grad. norm: %0.3f, vector diff: %0.3f')
        log_vars = [
            epoch, train_loss, train_acc, test_loss, test_acc, epoch_time,
            float(cosine_distance),
            float(cumulative_grad_norm),
            float(vector_difference_norm)
        ]
        logging.info(log_format_str, *log_vars)
      # End epoch.


    training_time = time.time() - start_time
    logging.info('Training finished, Total wallclock time: %0.2f sec',
                 training_time)

    if jax.host_id() == 0 and self._summary_writer is not None:
      for label, value in best_test_metrics.items():
        self._summary_writer.scalar('best_test_acc/{}'.format(label), value,
                                    best_iter)
    logging.info('Best Test Accuracy: iteration %d, test acc. %0.5f',
                 best_test_metrics['step'], best_test_acc)

    if jax.host_id() == 0 and self._summary_writer is not None:
      for label, value in best_test_metrics.items():
        self._summary_writer.scalar(
            'best_train_loss/{}'.format(label),
            value,
            step=best_train_metrics['step'])
    logging.info('Best Train Loss: iteration %d, test loss. %0.5f',
                 best_train_metrics['step'], best_train_loss)

    return (best_model, best_test_metrics)


def _eval_step(model: flax.deprecated.nn.Model,
               state: flax.deprecated.nn.Collection,
               batch: Mapping[str, jnp.array]) -> Dict[str, jnp.array]:
  """Evaluates a mini-batch of data.

  Args:
    model: The model to use to evaluate.
    state: Model state containing state for stateful flax.deprecated.nn
      functions, such as batch normalization.
    batch: Mini-batch of data to evaluate on.

  Returns:
    Dictionary consisting of the mini-batch the loss and accuracy.
  """
  state = jax.lax.pmean(state, 'batch')
  with flax.deprecated.nn.stateful(state, mutable=False):
    logits = model(batch[DATAKEY], train=False)
  metrics = utils.compute_metrics(logits, batch[LABELKEY])
  return metrics


def eval_model(model: flax.deprecated.nn.Model,
               state: flax.deprecated.nn.Collection,
               eval_dataset: jnp.array) -> Dict[str, float]:
  """Evaluates the given model using the given dataset.

  Args:
    model: The model the evaluate.
    state: Model state containing state for stateful flax.deprecated.nn
      functions, such as batch normalization.
    eval_dataset: Dataset to evaluate the model over.

  Returns:
  Dictionary containing the average loss and accuracy of the model on the given
  dataset.
  """
  p_eval_step = jax.pmap(_eval_step, axis_name='batch')

  batch_sizes = []
  metrics = []
  for batch in eval_dataset:
    batch_size = len(batch[LABELKEY])

    # These are required for pmap call.
    batch = _shard_batch(batch)
    batch_metrics = p_eval_step(model, state, batch)

    batch_sizes.append(batch_size)
    metrics.append(batch_metrics)

  # Note: use weighted mean, since we do mean of means with potentially
  # different batch sizes otherwise.
  batch_sizes = jnp.array(batch_sizes)
  weights = batch_sizes / jnp.sum(batch_sizes)
  eval_metrics = common_utils.get_metrics(metrics)
  return jax.tree_map(lambda x: (weights * x).sum(), eval_metrics)
