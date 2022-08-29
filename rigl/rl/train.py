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

r"""The entry point for training a sparse DQN agent."""

import os

from absl import app
from absl import flags
import gin
from rigl.rl import run_experiment
import tensorflow as tf



flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_atari_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


def create_sparsetrain_runner(base_dir):
  assert base_dir is not None
  return run_experiment.SparseTrainRunner(base_dir)


def main(unused_argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

  runner = create_sparsetrain_runner(FLAGS.base_dir)
  runner.run_experiment()

  logconfigfile_path = os.path.join(FLAGS.base_dir, 'operative_config.gin')
  with tf.io.gfile.GFile(logconfigfile_path, 'w') as f:
    f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
