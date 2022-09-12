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

"""Weight Symmetry: Iteratively Prune Model during Training.

Command for training and pruning an MNIST fully-connected model for 10 epochs
with a fixed pruning rate of 0.95:

prune --xm_runlocal --dataset=MNIST --model=MNIST_FC --epochs=10
--pruning_rate=0.95

Command for training and pruning an MNIST fully-connected model for 10
epochs, with pruning rates 0.3, 0.6 and 0.95 at epochs 2, 5, and 8 respectively
for all layers:

prune --xm_runlocal --dataset=MNIST --model=MNIST_FC --epochs=10
--pruning_schedule='[(2, 0.3), (5, 0.6), (8, 0.95)]'

Command for doing the same, but performing pruning only on the second layer:

prune --xm_runlocal --dataset=MNIST --model=MNIST_FC --epochs=10
--pruning_schedule="{'1': [(2, 0.3), (5, 0.6), (8, 0.95)]}"
"""
import ast
from collections import abc
import functools
from os import path
from typing import List
import uuid

from absl import app
from absl import flags
from absl import logging
import flax
from flax.metrics import tensorboard
from flax.training import lr_schedule
import jax
import jax.numpy as jnp
from rigl.experimental.jax.datasets import dataset_factory
from rigl.experimental.jax.models import model_factory
from rigl.experimental.jax.training import training
from rigl.experimental.jax.utils import utils

  experiment_dir = path.join(FLAGS.experiment_dir, str(work_unit_id))

  logging.info('Saving experimental results to %s', experiment_dir)

  host_count = jax.host_count()
  local_device_count = jax.local_device_count()
  logging.info('Device count: %d, host count: %d, local device count: %d',
               jax.device_count(), host_count, local_device_count)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(experiment_dir)

  dataset = dataset_factory.create_dataset(
      FLAGS.dataset,
      FLAGS.batch_size,
      FLAGS.batch_size_test,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size)

  logging.info('Training %s on the %s dataset...', FLAGS.model, FLAGS.dataset)

  rng = jax.random.PRNGKey(FLAGS.random_seed)

  input_shape = (1,) + dataset.shape
  base_model, _ = model_factory.create_model(
      FLAGS.model,
      rng, ((input_shape, jnp.float32),),
      num_classes=dataset.num_classes)

  initial_model, initial_state = model_factory.create_model(
      FLAGS.model,
      rng, ((input_shape, jnp.float32),),
      num_classes=dataset.num_classes,
      masked_layer_indices=FLAGS.masked_layer_indices)

  if FLAGS.optimizer == 'Adam':
    optimizer = flax.optim.Adam(
        learning_rate=FLAGS.lr, weight_decay=FLAGS.weight_decay)
  elif FLAGS.optimizer == 'Momentum':
    optimizer = flax.optim.Momentum(
        learning_rate=FLAGS.lr,
        beta=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        nesterov=False)

  steps_per_epoch = dataset.get_train_len() // FLAGS.batch_size

  if FLAGS.lr_schedule == LR_SCHEDULE_CONSTANT:
    lr_fn = lr_schedule.create_constant_learning_rate_schedule(
        FLAGS.lr, steps_per_epoch)
  elif FLAGS.lr_schedule == LR_SCHEDULE_STEPPED:
    lr_schedule_steps = ast.literal_eval(FLAGS.lr_schedule_steps)
    lr_fn = lr_schedule.create_stepped_learning_rate_schedule(
        FLAGS.lr, steps_per_epoch, lr_schedule_steps)
  elif FLAGS.lr_schedule == LR_SCHEDULE_COSINE:
    lr_fn = lr_schedule.create_cosine_learning_rate_schedule(
        FLAGS.lr, steps_per_epoch, FLAGS.epochs)
  else:
    raise ValueError(f'Unknown LR schedule type {FLAGS.lr_schedule}')

  # Reuses the FLAX learning rate schedule framework for pruning rate schedule.
  pruning_fn_p = functools.partial(
      lr_schedule.create_stepped_learning_rate_schedule, FLAGS.pruning_rate,
      steps_per_epoch)
  if FLAGS.pruning_schedule:
    pruning_schedule = ast.literal_eval(FLAGS.pruning_schedule)
    if isinstance(pruning_schedule, abc.Mapping):
      pruning_rate_fn = {
          f'MaskedModule_{layer_num}': pruning_fn_p(schedule)
          for layer_num, schedule in pruning_schedule.items()
      }
    else:
      pruning_rate_fn = pruning_fn_p(pruning_schedule)
  else:
    pruning_rate_fn = lr_schedule.create_constant_learning_rate_schedule(
        FLAGS.pruning_rate, steps_per_epoch)

  if jax.host_id() == 0:
    trainer = training.Trainer(
        optimizer,
        initial_model,
        initial_state,
        dataset,
        rng,
        summary_writer=summary_writer,
    )
  else:
    trainer = training.Trainer(
        optimizer, initial_model, initial_state, dataset, rng)

  _, best_metrics = trainer.train(
      FLAGS.epochs,
      lr_fn=lr_fn,
      pruning_rate_fn=pruning_rate_fn,
      update_iter=FLAGS.update_iterations,
      update_epoch=FLAGS.update_epoch,
  )

  logging.info('Best metrics: %s', str(best_metrics))

  if jax.host_id() == 0:
    if FLAGS.dump_json:
      utils.dump_dict_json(best_metrics,
                           path.join(experiment_dir, 'best_metrics.json'))

    for label, value in best_metrics.items():
      summary_writer.scalar(f'best/{label}', value,
                            FLAGS.epochs * steps_per_epoch)
    summary_writer.close()


def main(argv: List[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_training()

if __name__ == '__main__':
  app.run(main)
