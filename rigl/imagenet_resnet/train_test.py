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

r"""Tests for the data_helper input pipeline and the training process.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.testing.parameterized as parameterized

from rigl.imagenet_resnet.imagenet_train_eval import resnet_model_fn_w_pruning
from rigl.imagenet_resnet.imagenet_train_eval import set_lr_schedule
import tensorflow.compat.v1 as tf  # tf
from official.resnet import imagenet_input
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

FLAGS = flags.FLAGS


class DataInputTest(tf.test.TestCase, parameterized.TestCase):

  def _retrieve_data(self, is_training, data_dir):

    dataset = imagenet_input.ImageNetInput(
        is_training=is_training,
        data_dir=data_dir,
        transpose_input=False,
        num_parallel_calls=8,
        use_bfloat16=False)

    return dataset

  @parameterized.parameters('snip', 'set', 'rigl', 'scratch')
  def testTrainingPipeline(self, training_method):
    output_directory = '/tmp/'

    g = tf.Graph()
    with g.as_default():

      dataset = self._retrieve_data(is_training=False, data_dir=False)

      FLAGS.transpose_input = False
      FLAGS.use_tpu = False
      FLAGS.mode = 'train'
      FLAGS.mask_init_method = 'random'
      FLAGS.precision = 'float32'
      FLAGS.train_steps = 1
      FLAGS.train_batch_size = 1
      FLAGS.eval_batch_size = 1
      FLAGS.steps_per_eval = 1
      FLAGS.model_architecture = 'resnet'

      params = {}
      params['output_dir'] = output_directory
      params['training_method'] = training_method
      params['use_tpu'] = False
      set_lr_schedule()

      run_config = tpu_config.RunConfig(
          master=None,
          model_dir=None,
          save_checkpoints_steps=1,
          tpu_config=tpu_config.TPUConfig(iterations_per_loop=1, num_shards=1))

      classifier = tpu_estimator.TPUEstimator(
          use_tpu=False,
          model_fn=resnet_model_fn_w_pruning,
          params=params,
          config=run_config,
          train_batch_size=1,
          eval_batch_size=1)

      classifier.train(input_fn=dataset.input_fn, max_steps=1)


if __name__ == '__main__':
  tf.test.main()
