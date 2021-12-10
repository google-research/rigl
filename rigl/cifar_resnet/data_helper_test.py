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

import os
from absl import flags
from absl import logging
import absl.testing.parameterized as parameterized
from rigl.cifar_resnet import resnet_train_eval
from rigl.cifar_resnet.data_helper import input_fn
import tensorflow.compat.v1 as tf
from tensorflow.contrib.model_pruning.python import pruning

FLAGS = flags.FLAGS

BATCH_SIZE = 1
NUM_IMAGES = 1
JITTER_MULTIPLIER = 2


class DataHelperTest(tf.test.TestCase, parameterized.TestCase):

  def get_next(self):
    data_directory = FLAGS.data_directory
    # we pass the updated eval and train string to the params dictionary.
    params = {
        'mode': 'test',
        'data_split': 'eval',
        'batch_size': BATCH_SIZE,
        'data_directory': data_directory
    }

    test_inputs, test_labels = input_fn(params)

    return test_inputs, test_labels

  def testInputPipeline(self):

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      test_inputs, test_labels = self.get_next()

      with self.test_session() as sess:
        test_images_out, test_labels_out = sess.run([test_inputs, test_labels])
        self.assertAllEqual(test_images_out.shape, [BATCH_SIZE, 32, 32, 3])
        self.assertAllEqual(test_labels_out.shape, [BATCH_SIZE])

  @parameterized.parameters(
      {
          'training_method': 'baseline',
      },
      {
          'training_method': 'threshold',
      },
      {
          'training_method': 'rigl',
      },
  )
  def testTrainingStep(self, training_method):

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

      images, labels = self.get_next()

      global_step, _, _, logits = resnet_train_eval.build_model(
          mode='train',
          images=images,
          labels=labels,
          training_method=training_method,
          num_classes=FLAGS.num_classes,
          depth=FLAGS.resnet_depth,
          width=FLAGS.resnet_width)

      tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

      total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

      learning_rate = 0.1

      opt = tf.train.MomentumOptimizer(
          learning_rate, momentum=FLAGS.momentum, use_nesterov=True)

      if training_method in ['threshold']:
        # Create a pruning object using the pruning hyperparameters
        pruning_obj = pruning.Pruning()

        logging.info('starting mask update op')
        mask_update_op = pruning_obj.conditional_mask_update_op()

      # Create the training op
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = opt.minimize(total_loss, global_step)

      init_op = tf.global_variables_initializer()

      with self.test_session() as sess:
        # test that we can train successfully for 1 step
        sess.run(init_op)
        for _ in range(1):
          sess.run(train_op)
          if training_method in ['threshold']:
            sess.run(mask_update_op)


if __name__ == '__main__':
  tf.test.main()
