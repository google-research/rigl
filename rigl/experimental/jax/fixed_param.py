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

"""Weight Symmetry: Train models with fixed param, but diff. depth and width."""
import ast
import functools
import operator
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
import jax.numpy as jnp
from rigl.experimental.jax.datasets import dataset_factory
from rigl.experimental.jax.models import mnist_fc
from rigl.experimental.jax.models import model_factory
from rigl.experimental.jax.pruning import masked
from rigl.experimental.jax.pruning import symmetry
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

  logging.info('Training %s on the %s dataset...', MODEL, FLAGS.dataset)

  rng = jax.random.PRNGKey(FLAGS.random_seed)

  input_shape = (1,) + dataset.shape

  input_len = functools.reduce(operator.mul, dataset.shape)

  features = mnist_fc.feature_dim_for_param(
      input_len,
      FLAGS.param_count,
      FLAGS.depth)

  logging.info('Model Configuration: %s', str(features))

  base_model, _ = model_factory.create_model(
      MODEL,
      rng, ((input_shape, jnp.float32),),
      num_classes=dataset.num_classes,
      features=features)

  model_param_count = utils.count_param(base_model, ('kernel',))

  logging.info(
      'Model Config: param.: %d, depth: %d. max width: %d, min width: %d',
      model_param_count, len(features), max(features), min(features))

  logging.info('Generating random mask based on model')

  # Re-initialize the RNG to maintain same training pattern (as in prune code).
  mask_rng = jax.random.PRNGKey(FLAGS.random_seed)
  mask = masked.shuffled_mask(
      base_model,
      rng=mask_rng,
      sparsity=FLAGS.mask_sparsity)

  if jax.host_id() == 0:
    mask_stats = symmetry.get_mask_stats(mask)
    logging.info('Mask stats: %s', str(mask_stats))


    for label, value in mask_stats.items():
      try:
        summary_writer.scalar(f'mask/{label}', value, 0)
      # This is needed because permutations (long int) can't be cast to float32.
      except (OverflowError, ValueError):
        summary_writer.text(f'mask/{label}', str(value), 0)
        logging.error('Could not write mask/%s to tensorflow summary as float32'
                      ', writing as string instead.', label)

    if FLAGS.dump_json:
      mask_stats['permutations'] = str(mask_stats['permutations'])
      utils.dump_dict_json(
          mask_stats, path.join(experiment_dir, 'mask_stats.json'))

    if FLAGS.dump_json:
      mask_stats['permutations'] = str(mask_stats['permutations'])
      utils.dump_dict_json(mask_stats,
                           path.join(experiment_dir, 'mask_stats.json'))

    model_stats = {
        'depth': len(features),
        'max_width': max(features),
        'min_width': min(features),
    }
    model_stats.update(
        {'feature_{}'.format(i): value for i, value in enumerate(features)})


    if FLAGS.dump_json:
      utils.dump_dict_json(model_stats,
                           path.join(experiment_dir, 'model_stats.json'))

  model, initial_state = model_factory.create_model(
      'MNIST_FC',
      rng, ((input_shape, jnp.float32),),
      num_classes=dataset.num_classes,
      features=features, masks=mask)

  if FLAGS.opt == 'Adam':
    optimizer = flax.optim.Adam(
        learning_rate=FLAGS.lr, weight_decay=FLAGS.weight_decay)
  elif FLAGS.opt == 'Momentum':
    optimizer = flax.optim.Momentum(
        learning_rate=FLAGS.lr,
        beta=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        nesterov=False)
  else:
    raise ValueError('Unknown Optimizer: {}'.format(FLAGS.opt))

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
      update_epoch=FLAGS.update_epoch,
  )

  logging.info('Best metrics: %s', str(best_metrics))

  if jax.host_id() == 0:
    if FLAGS.dump_json:
      utils.dump_dict_json(best_metrics,
                           path.join(experiment_dir, 'best_metrics.json'))

    for label, value in best_metrics.items():
      summary_writer.scalar('best/{}'.format(label), value,
                            FLAGS.epochs * steps_per_epoch)
    summary_writer.close()


def main(argv: List[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_training()

if __name__ == '__main__':
  app.run(main)
