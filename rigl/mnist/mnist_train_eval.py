# coding=utf-8
# Copyright 2019 RigL Authors.
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

r"""A configurable, multi-layer fully connected network trained on MNIST.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import flags

import numpy as np
from rigl import sparse_optimizers
from rigl import sparse_utils
import tensorflow as tf

from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data


flags.DEFINE_string('mnist', '/tmp/data', 'Location of the MNIST ' 'dataset.')

## optimizer hyperparameters
flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch')
flags.DEFINE_float('learning_rate', .2, 'Initial learning rate.')
flags.DEFINE_float('momentum', .9, 'Momentum.')
flags.DEFINE_boolean('use_nesterov', True, 'Use nesterov momentum.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to run.')
flags.DEFINE_integer('lr_drop_epoch', 75, 'The epoch to start dropping lr.')
flags.DEFINE_string('optimizer', 'momentum',
                    'Optimizer to use. momentum or adam')
flags.DEFINE_float('l2_scale', 1e-4, 'l2 loss scale')
flags.DEFINE_string('network_type', 'fc',
                    'Type of the network. See below for available options.')
flags.DEFINE_enum(
    'training_method', 'baseline',
    ('scratch', 'set', 'baseline', 'momentum', 'rigl', 'static', 'snip',
     'prune'),
    'Method used for training sparse network. `scratch` means initial mask is '
    'kept during training. `set` is for sparse evalutionary training and '
    '`baseline` is for dense baseline.')
flags.DEFINE_float('drop_fraction', 0.3,
                   'When changing mask dynamically, this fraction decides how '
                   'much of the ')
flags.DEFINE_string('drop_fraction_anneal', 'cosine',
                    'If not empty the drop fraction is annealed during sparse'
                    ' training. One of the following: `constant`, `cosine` or '
                    '`exponential_(\\d*\\.?\\d*)$`. For example: '
                    '`exponential_3`, `exponential_.3`, `exponential_0.3`. '
                    'The number after `exponential` defines the exponent.')
flags.DEFINE_string('grow_init', 'zeros',
                    'Passed to the SparseInitializer, one of: zeros, '
                    'initial_value, random_normal, random_uniform.')
flags.DEFINE_float('s_momentum', 0.9,
                   'Momentum values for exponential moving average of '
                   'gradients. Used when training_method="momentum".')
flags.DEFINE_string('input_mask_path', '',
                    'Momentum values for exponential moving average of ')
flags.DEFINE_float('sparsity_scale', 0.9, 'Relative sparsity of second layer.')
flags.DEFINE_float('rigl_acc_scale', 0.,
                   'Used to scale initial accumulated gradients for new '
                   'connections.')
flags.DEFINE_integer('maskupdate_begin_step', 0, 'Step to begin mask updates.')
flags.DEFINE_integer('maskupdate_end_step', 50000, 'Step to end mask updates.')
flags.DEFINE_integer('maskupdate_frequency', 100,
                     'Step interval between mask updates.')
flags.DEFINE_integer('mask_record_frequency', 0,
                     'Step interval between mask updates.')
flags.DEFINE_string(
    'mask_init_method',
    default='random',
    help='If not empty string and mask is not loaded from a checkpoint, '
    'indicates the method used for mask initialization. One of the following: '
    '`random`, `erdos_renyi`.')
flags.DEFINE_integer('prune_begin_step', 2000, 'step to begin pruning')
flags.DEFINE_integer('prune_end_step', 30000, 'step to end pruning')
flags.DEFINE_float('end_sparsity', .98, 'desired sparsity of final model.')
flags.DEFINE_integer('pruning_frequency', 500, 'how often to prune.')
flags.DEFINE_float('threshold_decay', 0, 'threshold_decay for pruning.')
flags.DEFINE_string('save_path', '', 'Where to save the model.')
flags.DEFINE_boolean('save_model', True, 'Whether to save model or not.')
flags.DEFINE_integer('seed', default=0, help=('Sets the random seed.'))

FLAGS = flags.FLAGS


# momentum = 0.9
# lr = 0.2
# batch = 100
# decay = 1e-4
def mnist_network_fc(input_batch, reuse=False, model_pruning=False):
  """Define a basic FC network."""
  regularizer = contrib_layers.l2_regularizer(scale=FLAGS.l2_scale)
  if model_pruning:
    y = layers.masked_fully_connected(
        inputs=input_batch[0],
        num_outputs=300,
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        reuse=reuse,
        scope='layer1')
    y1 = layers.masked_fully_connected(
        inputs=y,
        num_outputs=100,
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        reuse=reuse,
        scope='layer2')
    logits = layers.masked_fully_connected(
        inputs=y1, num_outputs=10, reuse=reuse, activation_fn=None,
        weights_regularizer=regularizer, scope='layer3')
  else:
    y = tf.layers.dense(
        inputs=input_batch[0],
        units=300,
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        reuse=reuse,
        name='layer1')
    y1 = tf.layers.dense(
        inputs=y,
        units=100,
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        reuse=reuse,
        name='layer2')
    logits = tf.layers.dense(inputs=y1, units=10, reuse=reuse,
                             kernel_regularizer=regularizer, name='layer3')

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=input_batch[1], logits=logits)

  cross_entropy += tf.losses.get_regularization_loss()

  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(input_batch[1], predictions), tf.float32))

  return cross_entropy, accuracy




def get_compressed_fc(masks):
  """Given the masks of a sparse network returns the compact network."""
  # Dead input pixels.
  inds = np.sum(masks[0], axis=1) != 0
  masks[0] = masks[0][inds]
  compressed_masks = []
  for i in range(len(masks)):
    w = masks[i]
    # Find neurons that doesn't have any incoming edges.
    do_w = np.sum(w, axis=0) != 0
    if i < (len(masks) - 1):
      # Find neurons that doesn't have any outgoing edges.
      di_wnext = np.sum(masks[i+1], axis=1) != 0
      # Kept neurons should have at least one incoming and one outgoing edges.
      do_w = np.logical_and(do_w, di_wnext)
    compressed_w = w[:, do_w]
    compressed_masks.append(compressed_w)
    if i < (len(masks) - 1):
      # Remove incoming edges from removed neurons.
      masks[i+1] = masks[i+1][do_w]
  sparsities = [np.sum(m == 0) / float(np.size(m)) for m in compressed_masks]
  sizes = [compressed_masks[0].shape[0]]
  for m in compressed_masks:
    sizes.append(m.shape[1])
  return sparsities, sizes


def main(unused_args):
  tf.set_random_seed(FLAGS.seed)
  tf.get_variable_scope().set_use_resource(True)
  np.random.seed(FLAGS.seed)

  # Load the MNIST data and set up an iterator.
  mnist_data = input_data.read_data_sets(
      FLAGS.mnist, one_hot=False, validation_size=0)
  train_images = mnist_data.train.images
  test_images = mnist_data.test.images
  if FLAGS.input_mask_path:
    reader = tf.train.load_checkpoint(FLAGS.input_mask_path)
    input_mask = reader.get_tensor('layer1/mask')
    indices = np.sum(input_mask, axis=1) != 0
    train_images = train_images[:, indices]
    test_images = test_images[:, indices]
  dataset = tf.data.Dataset.from_tensor_slices(
      (train_images, mnist_data.train.labels.astype(np.int32)))
  num_batches = mnist_data.train.images.shape[0] // FLAGS.batch_size
  dataset = dataset.shuffle(buffer_size=mnist_data.train.images.shape[0])
  batched_dataset = dataset.repeat(FLAGS.num_epochs).batch(FLAGS.batch_size)
  iterator = batched_dataset.make_one_shot_iterator()

  test_dataset = tf.data.Dataset.from_tensor_slices(
      (test_images, mnist_data.test.labels.astype(np.int32)))
  num_test_images = mnist_data.test.images.shape[0]
  test_dataset = test_dataset.repeat(FLAGS.num_epochs).batch(num_test_images)
  test_iterator = test_dataset.make_one_shot_iterator()

  # Set up loss function.
  use_model_pruning = FLAGS.training_method != 'baseline'

  if FLAGS.network_type == 'fc':
    cross_entropy_train, _ = mnist_network_fc(
        iterator.get_next(), model_pruning=use_model_pruning)
    cross_entropy_test, accuracy_test = mnist_network_fc(
        test_iterator.get_next(), reuse=True, model_pruning=use_model_pruning)
  else:
    raise RuntimeError(FLAGS.network + ' is an unknown network type.')

  # Remove extra added ones. Current implementation adds the variables twice
  # to the collection. Improve this hacky thing.
  # TODO test the following with the convnet or any other network.
  if use_model_pruning:
    for k in ('masks', 'masked_weights', 'thresholds', 'kernel'):
      # del tf.get_collection_ref(k)[2]
      # del tf.get_collection_ref(k)[2]
      collection = tf.get_collection_ref(k)
      del collection[len(collection)//2:]
      print(tf.get_collection_ref(k))

  # Set up optimizer and update ops.
  global_step = tf.train.get_or_create_global_step()
  batch_per_epoch = mnist_data.train.images.shape[0] // FLAGS.batch_size

  if FLAGS.optimizer != 'adam':
    if not use_model_pruning:
      boundaries = [int(round(s * batch_per_epoch)) for s in [60, 70, 80]]
    else:
      boundaries = [int(round(s * batch_per_epoch)) for s
                    in [FLAGS.lr_drop_epoch, FLAGS.lr_drop_epoch + 20]]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries,
        values=[FLAGS.learning_rate / (3. ** i)
                for i in range(len(boundaries) + 1)])
  else:
    learning_rate = FLAGS.learning_rate

  if FLAGS.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
  elif FLAGS.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum,
                                     use_nesterov=FLAGS.use_nesterov)
  elif FLAGS.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise RuntimeError(FLAGS.optimizer + ' is unknown optimizer type')
  custom_sparsities = {
      'layer2': FLAGS.end_sparsity * FLAGS.sparsity_scale,
      'layer3': FLAGS.end_sparsity * 0
  }

  if FLAGS.training_method == 'set':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseSETOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)
  elif FLAGS.training_method == 'static':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseStaticOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)
  elif FLAGS.training_method == 'momentum':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseMomentumOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, momentum=FLAGS.s_momentum,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        grow_init=FLAGS.grow_init,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal, use_tpu=False)
  elif FLAGS.training_method == 'rigl':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseRigLOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency,
        drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal,
        initial_acc_scale=FLAGS.rigl_acc_scale, use_tpu=False)
  elif FLAGS.training_method == 'snip':
    opt = sparse_optimizers.SparseSnipOptimizer(
        opt,
        mask_init_method=FLAGS.mask_init_method,
        default_sparsity=FLAGS.end_sparsity,
        custom_sparsity_map=custom_sparsities,
        use_tpu=False)
  elif FLAGS.training_method in ('scratch', 'baseline', 'prune'):
    pass
  else:
    raise ValueError('Unsupported pruning method: %s' % FLAGS.training_method)

  train_op = opt.minimize(cross_entropy_train, global_step=global_step)


  if FLAGS.training_method == 'prune':
    hparams_string = ('begin_pruning_step={0},sparsity_function_begin_step={0},'
                      'end_pruning_step={1},sparsity_function_end_step={1},'
                      'target_sparsity={2},pruning_frequency={3},'
                      'threshold_decay={4}'.format(
                          FLAGS.prune_begin_step, FLAGS.prune_end_step,
                          FLAGS.end_sparsity, FLAGS.pruning_frequency,
                          FLAGS.threshold_decay))
    pruning_hparams = pruning.get_pruning_hparams().parse(hparams_string)
    pruning_hparams.set_hparam('weight_sparsity_map',
                               ['{0}:{1}'.format(k, v) for k, v
                                in custom_sparsities.items()])
    print(pruning_hparams)
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)
    with tf.control_dependencies([train_op]):
      train_op = pruning_obj.conditional_mask_update_op()
  weight_sparsity_levels = pruning.get_weight_sparsity()
  global_sparsity = sparse_utils.calculate_sparsity(pruning.get_masks())
  tf.summary.scalar('test_accuracy', accuracy_test)
  tf.summary.scalar('global_sparsity', global_sparsity)
  for k, v in zip(pruning.get_masks(), weight_sparsity_levels):
    tf.summary.scalar('sparsity/%s' % k.name, v)
  if FLAGS.training_method in ('prune', 'snip', 'baseline'):
    mask_init_op = tf.no_op()
    tf.logging.info('No mask is set, starting dense.')
  else:
    all_masks = pruning.get_masks()
    mask_init_op = sparse_utils.get_mask_init_fn(
        all_masks, FLAGS.mask_init_method, FLAGS.end_sparsity,
        custom_sparsities)

  if FLAGS.save_model:
    saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()
  hyper_params_string = '_'.join([FLAGS.network_type, str(FLAGS.batch_size),
                                  str(FLAGS.learning_rate),
                                  str(FLAGS.momentum),
                                  FLAGS.optimizer,
                                  str(FLAGS.l2_scale),
                                  FLAGS.training_method,
                                  str(FLAGS.prune_begin_step),
                                  str(FLAGS.prune_end_step),
                                  str(FLAGS.end_sparsity),
                                  str(FLAGS.pruning_frequency),
                                  str(FLAGS.seed)])
  tf.io.gfile.makedirs(FLAGS.save_path)
  filename = os.path.join(FLAGS.save_path, hyper_params_string + '.txt')
  merged_summary_op = tf.summary.merge_all()

  # Run session.
  if not use_model_pruning:
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.save_path,
                                             graph=tf.get_default_graph())
      print('Epoch', 'Epoch time', 'Test loss', 'Test accuracy')
      sess.run([init_op])
      tic = time.time()
      with tf.io.gfile.GFile(filename, 'w') as outputfile:
        for i in range(FLAGS.num_epochs * num_batches):
          sess.run([train_op])

          if (i % num_batches) == (-1 % num_batches):
            epoch_time = time.time() - tic
            loss, accuracy, summary = sess.run([cross_entropy_test,
                                                accuracy_test,
                                                merged_summary_op])
            # Write logs at every test iteration.
            summary_writer.add_summary(summary, i)
            log_str = '%d, %.4f, %.4f, %.4f' % (
                i // num_batches, epoch_time, loss, accuracy)
            print(log_str)
            print(log_str, file=outputfile)
            tic = time.time()
      if FLAGS.save_model:
        saver.save(sess, os.path.join(FLAGS.save_path, 'model.ckpt'))
  else:
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.save_path,
                                             graph=tf.get_default_graph())
      if FLAGS.network_type == 'fc':
        log_str = ','.join(['Epoch', 'Iteration', 'Test loss', 'Test accuracy',
                            'G_Sparsity', 'Sparsity Layer 0',
                            'Sparsity Layer 1'])
      else:
        log_str = ','.join(['Epoch', 'Iteration', 'Test loss', 'Test accuracy',
                            'G_Sparsity', 'Sparsity Layer 0',
                            'Sparsity Layer 1'])
      sess.run(init_op)
      sess.run(mask_init_op)
      tic = time.time()
      mask_records = {}
      with tf.io.gfile.GFile(filename, 'w') as outputfile:
        print(log_str)
        print(log_str, file=outputfile)
        for i in range(FLAGS.num_epochs * num_batches):
          if (FLAGS.mask_record_frequency > 0 and
              i % FLAGS.mask_record_frequency == 0):
            mask_vals = sess.run(pruning.get_masks())
            # Cast into bool to save space.
            mask_records[i] = [a.astype(np.bool) for a in mask_vals]
          sess.run([train_op])
          weight_sparsity, global_sparsity_val = sess.run(
              [weight_sparsity_levels, global_sparsity])
          if (i % num_batches) == (-1 % num_batches):
            epoch_time = time.time() - tic
            loss, accuracy, summary = sess.run([cross_entropy_test,
                                                accuracy_test,
                                                merged_summary_op])
            # Write logs at every test iteration.
            summary_writer.add_summary(summary, i)
            log_str = '%d, %d, %.4f, %.4f, %.4f, %.4f, %.4f' % (
                i // num_batches, i, loss, accuracy, global_sparsity_val,
                weight_sparsity[0], weight_sparsity[1])
            print(log_str)
            print(log_str, file=outputfile)
            mask_vals = sess.run(pruning.get_masks())
            if FLAGS.network_type == 'fc':
              sparsities, sizes = get_compressed_fc(mask_vals)
              print('[COMPRESSED SPARSITIES/SHAPE]: %s %s' % (sparsities,
                                                              sizes))
              print('[COMPRESSED SPARSITIES/SHAPE]: %s %s' % (sparsities,
                                                              sizes),
                    file=outputfile)
            tic = time.time()
      if FLAGS.save_model:
        saver.save(sess, os.path.join(FLAGS.save_path, 'model.ckpt'))
      if mask_records:
        np.save(os.path.join(FLAGS.save_path, 'mask_records'), mask_records)


if __name__ == '__main__':
  tf.app.run()
