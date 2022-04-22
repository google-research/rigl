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

"""Weight Symmetry: Train Model.

Trains a model from scratch, saving the relevant early weight snapshots.
"""
import ast
from os import path
from typing import List, Sequence
import uuid

from absl import app
from absl import flags
from absl import logging
import flax
from flax.metrics import tensorboard
from flax.training import lr_schedule
import jax
import jax.numpy as np
from rigl.experimental.jax.datasets import dataset_factory
from rigl.experimental.jax.models import model_factory
from rigl.experimental.jax.training import training


FLAGS = flags.FLAGS

MODEL_LIST: Sequence[str] = tuple(model_factory.MODELS.keys())
DATASET_LIST: Sequence[str] = tuple(dataset_factory.DATASETS.keys())

flags.DEFINE_enum('model', MODEL_LIST[0], MODEL_LIST,
                  'Model to train.')
flags.DEFINE_enum('dataset', DATASET_LIST[0], DATASET_LIST,
                  'Dataset to train on.')
flags.DEFINE_enum('optimizer', 'Adam', ['Momentum', 'Adam'],
                  'Optimizer to use.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.', short_name='lr')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay penalty.',
                   short_name='wd')
flags.DEFINE_float('momentum', 0.9, 'Momentum weighting.')
flags.DEFINE_string(
    'lr_schedule', default='stepped',
    help=('Learning rate schedule type; constant, stepped or cosine.'))
flags.DEFINE_string(
    'lr_schedule_steps', default='[[50, 0.01], [70, 0.001], [90, 0.0001]]',
    help=('Learning rate schedule steps as a Python list; '
          '[[step1_epoch, step1_lr_scale], '
          '[step2_epoch, step2_lr_scale], ...]'))
flags.DEFINE_integer(
    'batch_size', 128, 'Training minibatch size.', lower_bound=1)
flags.DEFINE_integer(
    'batch_size_test',
    128,
    'Test minibatch size.',
    lower_bound=1)
flags.DEFINE_integer(
    'epochs', 100, 'Number of epochs to train over.', lower_bound=1)
flags.DEFINE_integer('random_seed', 42, 'Random seed.')
flags.DEFINE_integer('shuffle_buffer_size', 1024,
                     'Dataset shuffle buffer size.')
flags.DEFINE_string(
    'experiment_dir', '/tmp/experiments',
    'Path to store experiment output in, i.e. models, snapshots.')
flags.DEFINE_integer(
    'update_iterations',
    1000,
    'Epoch interval after which to evaluate test error.',
    lower_bound=1)
flags.DEFINE_integer(
    'update_epoch', 10, 'Epoch interval after which to evaluate test error.',
    lower_bound=1)


def run_training():
  """Trains a model."""
  print('Logging to {}'.format(FLAGS.log_dir))
  work_unit_id = uuid.uuid4()
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
  model, initial_state = model_factory.create_model(
      FLAGS.model,
      rng, ((input_shape, np.float32),),
      num_classes=dataset.num_classes)

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

  if FLAGS.lr_schedule == 'constant':
    lr_fn = lr_schedule.create_constant_learning_rate_schedule(
        FLAGS.lr, steps_per_epoch)
  elif FLAGS.lr_schedule == 'stepped':
    lr_schedule_steps = ast.literal_eval(FLAGS.lr_schedule_steps)
    lr_fn = lr_schedule.create_stepped_learning_rate_schedule(
        FLAGS.lr, steps_per_epoch, lr_schedule_steps)
  elif FLAGS.lr_schedule == 'cosine':
    lr_fn = lr_schedule.create_cosine_learning_rate_schedule(
        FLAGS.lr, steps_per_epoch, FLAGS.epochs)
  else:
    raise ValueError('Unknown LR schedule type {}'.format(FLAGS.lr_schedule))

  if jax.host_id() == 0:
    trainer = training.Trainer(
        optimizer,
        model,
        initial_state,
        dataset,
        rng,
        summary_writer=summary_writer,
    )
  else:
    trainer = training.Trainer(optimizer, model, initial_state, dataset, rng)

  _, best_metrics = trainer.train(
      FLAGS.epochs,
      lr_fn=lr_fn,
      update_iter=FLAGS.update_iterations,
      update_epoch=FLAGS.update_epoch)

  logging.info('Best metrics: %s', str(best_metrics))

  if jax.host_id() == 0:
    for label, value in best_metrics.items():
      summary_writer.scalar('best/{}'.format(label), value,
                            FLAGS.epochs * steps_per_epoch)
    summary_writer.close()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_training()

if __name__ == '__main__':
  app.run(main)
