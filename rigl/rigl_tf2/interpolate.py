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

r"""Script for interpolating between checkpoints.
"""

import os

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
from rigl.rigl_tf2 import utils
import tensorflow.compat.v2 as tf

from pyglib import timer
FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', '/tmp/sparse_spectrum/interpolation',
                    'Directory to save experiment in.')
flags.DEFINE_string('ckpt_start', '/tmp/sparse_spectrum/cp-0001.ckpt',
                    'Directory to save experiment in.')
flags.DEFINE_string('ckpt_end', '/tmp/sparse_spectrum/cp-0041.ckpt',
                    'Directory to save experiment in.')
flags.DEFINE_string(
    'preload_gin_config', '', 'If non-empty reads a gin file '
    'before parsing gin_config and bindings. This is useful,'
    'when you want to start from a configuration of another '
    'run. Values are then overwritten by additional configs '
    'and bindings provided.')
flags.DEFINE_bool('use_tpu', True, 'Whether to run on TPU or not.')
flags.DEFINE_bool('eval_on_train', True, 'Whether to evaluate on training set.')
flags.DEFINE_integer('load_mask_from', 0, '0 means start checkpoint, 1 means '
                     'end checkpoint. -1 means no mask loaded.')
flags.DEFINE_enum('mode', 'train_eval', ('train_eval', 'hessian'),
                  'Whether to run on TPU or not.')
flags.DEFINE_string(
    'tpu_job_name', 'tpu_worker',
    'Name of the TPU worker job. This is required when having '
    'multiple TPU worker jobs.')
flags.DEFINE_string('master', None, 'TPU worker.')
flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')


def test_model(model, d_test, batch_size=1000):
  """Tests the model and calculates cross entropy loss and accuracy."""
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  for x, y in d_test.batch(batch_size):
    predictions = model(x, training=False)
    batch_loss = loss_object(y, predictions)
    test_loss.update_state(batch_loss)
    test_accuracy.update_state(y, predictions)
  logging.info('Test loss: %f', test_loss.result().numpy())
  logging.info('Test accuracy: %f', test_accuracy.result().numpy())
  return test_loss.result().numpy(), test_accuracy.result().numpy()


@gin.configurable(
    'interpolate',
    denylist=['model_start', 'model_end', 'model_inter', 'd_set'])
def interpolate(model_start, model_end, model_inter, d_set,
                i_start=-0.2, i_end=1.2, n_interpolation=29):
  """Interpolates between 2 sparse networks linearly and evaluates."""
  interpolation_coefs = np.linspace(i_start, i_end, n_interpolation)
  all_scores = {}
  for i_coef in interpolation_coefs:
    logging.info('Interpolating with: %f', i_coef)
    for var_start, var_end, var_inter in zip(model_start.trainable_variables,
                                             model_end.trainable_variables,
                                             model_inter.trainable_variables):
      new_value = (1 - i_coef) * var_start + i_coef * var_end
      var_inter.assign(new_value)
    scores = test_model(model_inter, d_set)
    all_scores[i_coef] = scores
  return all_scores


def main(unused_argv):
  init_timer = timer.Timer()
  init_timer.Start()
  if FLAGS.preload_gin_config:
    # Load default values from the original experiment, always the first one.
    with gin.unlock_config():
      gin.parse_config_file(FLAGS.preload_gin_config, skip_unknown=True)
    logging.info('Operative Gin configurations loaded from: %s',
                 FLAGS.preload_gin_config)
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)

  data_train, data_test, info = utils.get_dataset()
  input_shape = info.features['image'].shape
  num_classes = info.features['label'].num_classes
  logging.info('Input Shape: %s', input_shape)
  logging.info('train samples: %s', info.splits['train'].num_examples)
  logging.info('test samples: %s', info.splits['test'].num_examples)
  data_eval = data_train if FLAGS.eval_on_train else data_test
  pruning_params = utils.get_pruning_params(mode='constant')
  mask_load_dict = {-1: None, 0: FLAGS.ckpt_start, 1: FLAGS.ckpt_end}
  mask_path = mask_load_dict[FLAGS.load_mask_from]
  # Currently we interpolate only on the same sparse space.
  model_start = utils.get_network(
      pruning_params,
      input_shape,
      num_classes,
      mask_init_path=mask_path,
      weight_init_path=FLAGS.ckpt_start)
  model_start.summary()
  model_end = utils.get_network(
      pruning_params,
      input_shape,
      num_classes,
      mask_init_path=mask_path,
      weight_init_path=FLAGS.ckpt_end)
  model_end.summary()

  # Create a third network for interpolation.
  model_inter = utils.get_network(
      pruning_params,
      input_shape,
      num_classes,
      mask_init_path=mask_path,
      weight_init_path=FLAGS.ckpt_end)
  logging.info('Performance at init (model_start:')
  test_model(model_start, data_eval)
  logging.info('Performance at init (model_end:')
  test_model(model_end, data_eval)
  all_results = interpolate(model_start=model_start, model_end=model_end,
                            model_inter=model_inter, d_set=data_eval)

  tf.io.gfile.makedirs(FLAGS.logdir)
  results_path = os.path.join(FLAGS.logdir, 'all_results')
  with tf.io.gfile.GFile(results_path, 'wb') as f:
    np.save(f, all_results)
  logging.info('Total runtime: %.3f s', init_timer.GetDuration())
  logconfigfile_path = os.path.join(FLAGS.logdir, 'operative_config.gin')
  with tf.io.gfile.GFile(logconfigfile_path, 'w') as f:
    f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
