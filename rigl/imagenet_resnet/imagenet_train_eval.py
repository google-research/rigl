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

r"""This script trains a ResNet model that implements various pruning methods.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags
from absl import logging
from rigl import sparse_optimizers
from rigl import sparse_utils
from rigl.imagenet_resnet import mobilenetv1_model
from rigl.imagenet_resnet import mobilenetv2_model
from rigl.imagenet_resnet import resnet_model
from rigl.imagenet_resnet import utils
from rigl.imagenet_resnet import vgg
from official.resnet import imagenet_input
from tensorflow.contrib import estimator as contrib_estimator
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.training.python.training import evaluation
from tensorflow_estimator.python.estimator import estimator

DST_METHODS = [
    'set',
    'momentum',
    'rigl',
    'static'
]

ALL_METHODS = tuple(['scratch', 'baseline', 'snip', 'dnw'] + DST_METHODS)
NO_MASK_INIT_METHODS = ('snip', 'dnw', 'baseline')

flags.DEFINE_string(
    'precision',
    default='float32',
    help=('Precision to use; one of: {bfloat16, float32}'))
flags.DEFINE_integer('num_workers', 1, 'Number of training workers.')
flags.DEFINE_float(
    'base_learning_rate',
    default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum',
    default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))
flags.DEFINE_integer('ps_task', 0,
                     'Task id of the replica running the training.')
flags.DEFINE_float(
    'weight_decay',
    default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))
flags.DEFINE_string('master', '', 'Master job.')
flags.DEFINE_string('tpu_job_name', None, 'For complicated TensorFlowFlock')
flags.DEFINE_integer(
    'steps_per_checkpoint',
    default=1000,
    help=('Controls how often checkpoints are generated. More steps per '
          'checkpoint = higher utilization of TPU and generally higher '
          'steps/sec'))
flags.DEFINE_integer(
    'keep_checkpoint_max', default=0, help=('Number of checkpoints to hold.'))
flags.DEFINE_integer(
    'seed', default=0, help=('Sets the random seed.'))
flags.DEFINE_string(
    'data_directory', None, 'The location of the sstable used for training.')
flags.DEFINE_string('eval_once_ckpt_prefix', '',
                    'File name of the eval chekpoint used for evaluation.')
flags.DEFINE_string(
    'data_format',
    default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))
flags.DEFINE_bool(
    'transpose_input',
    default=False,
    help='Use TPU double transpose optimization')
flags.DEFINE_bool(
    'log_mask_imgs_each_iteration',
    default=False,
    help='Use to log few masks as images. Be careful when using. This is'
    ' very likely to slow down your training and create huge logs.')
flags.DEFINE_string(
    'mask_init_method',
    default='',
    help='If not empty string and mask is not loaded from a checkpoint, '
    'indicates the method used for mask initialization. One of the following: '
    '`random`, `erdos_renyi`.')
flags.DEFINE_integer(
    'resnet_depth',
    default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))
flags.DEFINE_float('label_smoothing', 0.1,
                   'Relax confidence in the labels by (1-label_smoothing).')
flags.DEFINE_float(
    'erk_power_scale', 1.0,
    'Softens the ERK distribituion. Value 0 means uniform.'
    '1 means regular ERK.')
flags.DEFINE_integer(
    'train_steps',
    default=2,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))
flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')
flags.DEFINE_integer(
    'eval_batch_size', default=1000, help='Batch size for evaluation.')
flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')
flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')
flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')
flags.DEFINE_integer(
    'steps_per_eval',
    default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))
flags.DEFINE_bool(
    'use_tpu',
    default=False,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))
flags.DEFINE_integer(
    'iterations_per_loop',
    default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))
flags.DEFINE_integer(
    'num_parallel_calls',
    default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))
flags.DEFINE_integer(
    'num_cores',
    default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))
flags.DEFINE_string('output_dir', '/tmp/imagenet/',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_bool('use_folder_stub', True,
                  'If True the output_dir is extended with some parameters.')
flags.DEFINE_bool('use_batch_statistics', False,
                  'If True the forward pass is made in training mode. ')
flags.DEFINE_bool('eval_on_train', False,
                  'If True the evaluation is made on training set.')
flags.DEFINE_enum(
    'mode', 'train', ('train_and_eval', 'train', 'eval', 'eval_once'),
    'One of {"train_and_eval", "train", "eval"}.')
flags.DEFINE_integer('export_model_freq', 2502,
                     'The rate at which estimator exports the model.')

flags.DEFINE_enum(
    'training_method', 'scratch', ALL_METHODS,
    'Method used for training sparse network. `scratch` means initial mask is '
    'kept during training. `set` is for sparse evalutionary training and '
    '`baseline` is for dense baseline.')
flags.DEFINE_enum(
    'init_method', 'baseline', ('baseline', 'sparse'),
    'Method for initialization.  If sparse and training_method=scratch, then '
    'use initializers that take into account starting sparsity.')
# flags.DEFINE_enum(
#     'mask_init_method', 'baseline', ('default'),
#     'Method for initializating masks. If not default, end_sparsities are used'
#     ' to define the layer wise random sparse connectivity.')

flags.DEFINE_bool(
    'is_warm_up',
    default=True,
    help=('Boolean for whether to scale weight of regularizer.'))

flags.DEFINE_float(
    'width', -1., 'Multiplier for the number of channels in each layer')
# first and last layer are somewhat special.  First layer has almost no
# parameters, but 3% of the total flops.  Last layer has only .05% of the total
# flops but 10% of the total parameters.  Depending on whether the goal is max
# compression or max acceleration, pruning goals will be different.
flags.DEFINE_bool('use_adam', False,
                  'Whether to use Adam or not')
flags.DEFINE_bool('use_sgdr', False,
                  'Whether to use SGDR for learning rate schedule.')
flags.DEFINE_float('sgdr_decay_step', 5, 'Initial cycle length for SGDR.')
flags.DEFINE_float('sgdr_t_mul', 1.5, 'Cycle length multiplier for SGDR')
flags.DEFINE_float('sgdr_m_mul', .5,
                   'Learning rate drop at each restart cycle.')
flags.DEFINE_float('end_sparsity', 0.9,
                   'Target sparsity desired by end of training.')
flags.DEFINE_float('drop_fraction', 0.3,
                   'When changing mask dynamically, this fraction decides how '
                   'much of the ')
flags.DEFINE_string('drop_fraction_anneal', 'constant',
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
flags.DEFINE_float('rigl_acc_scale', 0.,
                   'Used to scale initial accumulated gradients for new '
                   'connections.')
flags.DEFINE_integer('maskupdate_begin_step', 0, 'Step to begin pruning at.')
flags.DEFINE_integer('maskupdate_end_step', 25000, 'Step to end pruning at.')
flags.DEFINE_integer('maskupdate_frequency', 100,
                     'Step interval between pruning.')
flags.DEFINE_float(
    'first_layer_sparsity', 0.,
    'Sparsity to use for the first layer. Overrides default end_sparsity '
    'if greater than 0. If -1, default sparsity is applied. If 0, layer is not'
    'pruned or masked.')
flags.DEFINE_float(
    'last_layer_sparsity', -1,
    'Sparsity to use for the last layer. Overrides default end_sparsity '
    'if greater than 0. If -1, default sparsity is applied. If 0, layer is not'
    'pruned or masked.')
flags.DEFINE_string(
    'load_mask_dir', '',
    'Directory of a trained model from which to load only the mask')
flags.DEFINE_string(
    'initial_value_checkpoint', '',
    'Directory of a model from which to load only the parameters')
flags.DEFINE_string(
    'model_architecture', 'resnet',
    'Which architecture to use. Options: resnet, mobilenet_v1, mobilenet_v2.'
    'vgg_16, vgg_a, vgg_19.')
flags.DEFINE_float('expansion_factor', 6.,
                   'how much to expand filters before depthwise conv')
flags.DEFINE_float('training_steps_multiplier', 1.0,
                   'Training schedule is shortened or extended with the '
                   'multiplier, if it is not 1.')
flags.DEFINE_integer('block_width', 1, 'width of block')
flags.DEFINE_integer('block_height', 1, 'height of block')
FLAGS = flags.FLAGS
LR_SCHEDULE = []
PARAM_SUFFIXES = ('gamma', 'beta', 'weights', 'biases')
MASK_SUFFIX = 'mask'


# Learning rate schedule (multiplier, epoch to start) tuples
def set_lr_schedule():
  """Sets the learning schedule: LR_SCHEDULE for the training."""
  global LR_SCHEDULE
  if FLAGS.model_architecture == 'mobilenet_v2' or FLAGS.model_architecture == 'mobilenet_v1':
    LR_SCHEDULE = [(1.0, 8), (0.1, 40), (0.01, 75), (0.001, 95), (.0003, 120)]
  elif (FLAGS.model_architecture == 'resnet' or
        FLAGS.model_architecture.startswith('vgg')):
    LR_SCHEDULE = [(1.0, 0), (0.1, 30), (0.01, 70), (0.001, 90), (.0001, 120)]
  else:
    raise ValueError('Unknown architecture ' + FLAGS.model_architecture)
  if FLAGS.training_steps_multiplier != 1.0:
    multiplier = FLAGS.training_steps_multiplier
    LR_SCHEDULE = [(x, y * multiplier) for x, y in LR_SCHEDULE]
    FLAGS.train_steps = int(FLAGS.train_steps * multiplier)
    FLAGS.maskupdate_begin_step = int(FLAGS.maskupdate_begin_step * multiplier)
    FLAGS.maskupdate_end_step = int(FLAGS.maskupdate_end_step * multiplier)
    tf.logging.info(
        'Training schedule is updated with multiplier: %.2f' % multiplier)
  tf.logging.info('LR schedule: %s' % LR_SCHEDULE)
  tf.logging.info('Training Steps: %d' % FLAGS.train_steps)
# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

CUSTOM_SPARSITY_MAP = {}


def set_custom_sparsity_map():
  if FLAGS.first_layer_sparsity > 0.:
    CUSTOM_SPARSITY_MAP[
        'resnet_model/initial_conv'] = FLAGS.first_layer_sparsity
  if FLAGS.last_layer_sparsity > 0.:
    CUSTOM_SPARSITY_MAP[
        'resnet_model/final_dense'] = FLAGS.last_layer_sparsity


def lr_schedule(current_epoch):
  """Computes learning rate schedule."""
  scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
  if FLAGS.use_sgdr:
    decay_rate = tf.train.cosine_decay_restarts(
        scaled_lr, current_epoch, FLAGS.sgdr_decay_step,
        t_mul=FLAGS.sgdr_t_mul, m_mul=FLAGS.sgdr_m_mul)
  else:
    decay_rate = (
        scaled_lr * LR_SCHEDULE[0][0] * current_epoch / LR_SCHEDULE[0][1])
    for mult, start_epoch in LR_SCHEDULE:
      decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                            scaled_lr * mult)
  return decay_rate


def train_function(training_method, loss, cross_loss, reg_loss, output_dir,
                   use_tpu):
  """Training script for resnet model.

  Args:
   training_method: string indicating pruning method used to compress model.
   loss: tensor float32 of the cross entropy + regularization losses.
   cross_loss: tensor, only cross entropy loss, passed for logging.
   reg_loss: tensor, only regularization loss, passed for logging.
   output_dir: string tensor indicating the directory to save summaries.
   use_tpu: boolean indicating whether to run script on a tpu.

  Returns:
    host_call: summary tensors to be computed at each training step.
    train_op: the optimization term.
  """

  global_step = tf.train.get_global_step()

  steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
  current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
  learning_rate = lr_schedule(current_epoch)
  if FLAGS.use_adam:
    # We don't use step decrease for the learning rate.
    learning_rate = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  else:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=True)

  if use_tpu:
    # use CrossShardOptimizer when using TPU.
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

  if training_method == 'set':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseSETOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal,
        stateless_seed_offset=FLAGS.seed)
  elif training_method == 'static':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseStaticOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal,
        stateless_seed_offset=FLAGS.seed)
  elif training_method == 'momentum':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseMomentumOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, momentum=FLAGS.s_momentum,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        grow_init=FLAGS.grow_init, stateless_seed_offset=FLAGS.seed,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal, use_tpu=use_tpu)
  elif training_method == 'rigl':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseRigLOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency,
        drop_fraction=FLAGS.drop_fraction, stateless_seed_offset=FLAGS.seed,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal,
        initial_acc_scale=FLAGS.rigl_acc_scale, use_tpu=use_tpu)

  elif training_method == 'snip':
    optimizer = sparse_optimizers.SparseSnipOptimizer(
        optimizer, mask_init_method=FLAGS.mask_init_method,
        custom_sparsity_map=CUSTOM_SPARSITY_MAP,
        default_sparsity=FLAGS.end_sparsity, use_tpu=use_tpu)
  elif training_method == 'dnw':
    optimizer = sparse_optimizers.SparseDNWOptimizer(
        optimizer,
        mask_init_method=FLAGS.mask_init_method,
        custom_sparsity_map=CUSTOM_SPARSITY_MAP,
        default_sparsity=FLAGS.end_sparsity,
        use_tpu=use_tpu)
  elif training_method in ('scratch', 'baseline'):
    pass
  else:
    raise ValueError('Unsupported pruning method: %s' % FLAGS.training_method)
  # UPDATE_OPS needs to be added as a dependency due to batch norm
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops), tf.name_scope('train'):
    grads_and_vars = optimizer.compute_gradients(loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          'No gradients provided for any variable, check your graph for ops'
          ' that do not support gradients, between variables %s and loss %s.' %
          ([str(v) for _, v in grads_and_vars], loss))

    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
  metrics = {
      'global_step': tf.train.get_or_create_global_step(),
      'loss': loss,
      'cross_loss': cross_loss,
      'reg_loss': reg_loss,
      'learning_rate': learning_rate,
      'current_epoch': current_epoch,
  }

  # Logging drop_fraction if dynamic sparse training.
  is_dst_method = training_method in DST_METHODS
  if is_dst_method:
    metrics['drop_fraction'] = optimizer.drop_fraction

  def flatten_list_of_vars(var_list):
    flat_vars = [tf.reshape(v, [-1]) for v in var_list]
    return tf.concat(flat_vars, axis=-1)

  if use_tpu:
    reduced_grads = [tf.tpu.cross_replica_sum(g) for g, _ in grads_and_vars]
  else:
    reduced_grads = [g for g, _ in grads_and_vars]
  metrics['grad_norm'] = tf.norm(flatten_list_of_vars(reduced_grads))
  metrics['var_norm'] = tf.norm(
      flatten_list_of_vars([v for _, v in grads_and_vars]))
  # Let's log some statistics from a single parameter-mask couple.
  # This is useful for debugging.
  test_var = pruning.get_weights()[0]
  test_var_mask = pruning.get_masks()[0]
  metrics.update({
      'fw_nz_weight': tf.count_nonzero(test_var),
      'fw_nz_mask': tf.count_nonzero(test_var_mask),
      'fw_l1_weight': tf.reduce_sum(tf.abs(test_var))
  })

  masks = pruning.get_masks()
  global_sparsity = sparse_utils.calculate_sparsity(masks)
  metrics['global_sparsity'] = global_sparsity
  metrics.update(
      utils.mask_summaries(masks, with_img=FLAGS.log_mask_imgs_each_iteration))

  host_call = (functools.partial(utils.host_call_fn, output_dir),
               utils.format_tensors(metrics))

  return host_call, train_op


def resnet_model_fn_w_pruning(features, labels, mode, params):
  """The model_fn for ResNet-50 with pruning.

  Args:
    features: A float32 batch of images.
    labels: A int32 batch of labels.
    mode: Specifies whether training or evaluation.
    params: Dictionary of parameters passed to the model.

  Returns:
    A TPUEstimatorSpec for the model
  """

  width = 1. if FLAGS.width <= 0 else FLAGS.width

  if isinstance(features, dict):
    features = features['feature']

  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input  # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  if FLAGS.transpose_input and mode != tf_estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  training_method = params['training_method']
  use_tpu = params['use_tpu']

  def build_network():
    """Construct the network in the graph."""
    if FLAGS.model_architecture == 'mobilenet_v2':
      network_func = functools.partial(
          mobilenetv2_model.mobilenet_v2,
          expansion_factor=FLAGS.expansion_factor)
    elif FLAGS.model_architecture == 'mobilenet_v1':
      network_func = functools.partial(mobilenetv1_model.mobilenet_v1)
    elif FLAGS.model_architecture == 'resnet':
      prune_first_layer = FLAGS.first_layer_sparsity != 0.
      network_func = functools.partial(
          resnet_model.resnet_v1_,
          resnet_depth=FLAGS.resnet_depth,
          init_method=FLAGS.init_method,
          end_sparsity=FLAGS.end_sparsity,
          prune_first_layer=prune_first_layer)
    elif FLAGS.model_architecture.startswith('vgg'):
      network_func = functools.partial(
          vgg.vgg,
          vgg_type=FLAGS.model_architecture,
          init_method=FLAGS.init_method,
          end_sparsity=FLAGS.end_sparsity)
    else:
      raise ValueError('Unknown archiecture ' + FLAGS.archiecture)
    prune_last_layer = FLAGS.last_layer_sparsity != 0.
    network = network_func(
        num_classes=FLAGS.num_label_classes,
        # TODO remove the pruning_method option.
        pruning_method='threshold',
        width=width,
        prune_last_layer=prune_last_layer,
        data_format=FLAGS.data_format,
        weight_decay=FLAGS.weight_decay)

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)
    if FLAGS.use_batch_statistics:
      is_training = True
    return network(inputs=features, is_training=is_training)

  if FLAGS.precision == 'bfloat16':
    with contrib_tpu.bfloat16_scope():
      logits = build_network()
    logits = tf.cast(logits, tf.float32)
  elif FLAGS.precision == 'float32':
    logits = build_network()

  if mode == tf_estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf_estimator.export.PredictOutput(predictions)
        })
  output_dir = params['output_dir']
  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)

  # make sure we reuse the same label smoothing parameter is we're doing
  # scratch / lottery ticket experiments.
  label_smoothing = FLAGS.label_smoothing
  if FLAGS.training_method == 'scratch' and FLAGS.load_mask_dir:
    scratch_stripped = FLAGS.load_mask_dir.replace('/scratch', '')
    label_smoothing = float(scratch_stripped.split('/')[15])
    tf.logging.info('LABEL SMOOTHING USED: %.2f' % label_smoothing)
  cross_loss = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=label_smoothing)
  # Add regularization loss term
  reg_loss = tf.losses.get_regularization_loss()
  loss = cross_loss + reg_loss

  host_call = None
  if mode == tf_estimator.ModeKeys.TRAIN:
    host_call, train_op = train_function(training_method, loss, cross_loss,
                                         reg_loss, output_dir, use_tpu)
  else:
    train_op = None

  eval_metrics = None
  if mode == tf_estimator.ModeKeys.EVAL:

    def metric_fn(labels, logits, cross_loss, reg_loss):
      """Calculate eval metrics."""
      logging.info('In metric function')
      eval_metrics = {}
      predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      eval_metrics['top_5_eval_accuracy'] = tf.metrics.mean(in_top_5)
      eval_metrics['cross_loss'] = tf.metrics.mean(cross_loss)
      eval_metrics['reg_loss'] = tf.metrics.mean(reg_loss)
      eval_metrics['eval_accuracy'] = tf.metrics.accuracy(
          labels=labels, predictions=predictions)

      # If evaluating once lets also calculate sparsities.
      if FLAGS.mode == 'eval_once':
        sparsity_summaries = utils.mask_summaries(pruning.get_masks())
        # We call mean on a scalar to create tensor, update_op pairs.
        sparsity_summaries = {k: tf.metrics.mean(v) for k, v
                              in sparsity_summaries.items()}
        eval_metrics.update(sparsity_summaries)
      return eval_metrics

    tensors = [labels, logits,
               tf.broadcast_to(cross_loss, tf.shape(labels)),
               tf.broadcast_to(reg_loss, tf.shape(labels))]

    eval_metrics = (metric_fn, tensors)

  if (FLAGS.load_mask_dir and
      FLAGS.training_method not in NO_MASK_INIT_METHODS):

    def scaffold_fn():
      """For initialization, passed to the estimator."""
      utils.initialize_parameters_from_ckpt(FLAGS.load_mask_dir,
                                            FLAGS.output_dir, MASK_SUFFIX)
      if FLAGS.initial_value_checkpoint:
        utils.initialize_parameters_from_ckpt(FLAGS.initial_value_checkpoint,
                                              FLAGS.output_dir, PARAM_SUFFIXES)
      return tf.train.Scaffold()
  elif (FLAGS.mask_init_method and
        FLAGS.training_method not in NO_MASK_INIT_METHODS):

    def scaffold_fn():
      """For initialization, passed to the estimator."""
      if FLAGS.initial_value_checkpoint:
        utils.initialize_parameters_from_ckpt(FLAGS.initial_value_checkpoint,
                                              FLAGS.output_dir, PARAM_SUFFIXES)
      all_masks = pruning.get_masks()
      assigner = sparse_utils.get_mask_init_fn(
          all_masks,
          FLAGS.mask_init_method,
          FLAGS.end_sparsity,
          CUSTOM_SPARSITY_MAP,
          erk_power_scale=FLAGS.erk_power_scale)
      def init_fn(scaffold, session):
        """A callable for restoring variable from a checkpoint."""
        del scaffold  # Unused.
        session.run(assigner)
      return tf.train.Scaffold(init_fn=init_fn)
  else:
    assert FLAGS.training_method in NO_MASK_INIT_METHODS
    scaffold_fn = None
    tf.logging.info('No mask is set, starting dense.')

  return contrib_tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


class ExportModelHook(tf.train.SessionRunHook):
  """Train hooks called after each session run for exporting the model."""

  def __init__(self, classifier, export_dir):
    self.classifier = classifier
    self.global_step = None
    self.export_dir = export_dir
    self.last_export = 0
    self.supervised_input_receiver_fn = (
        contrib_estimator.build_raw_supervised_input_receiver_fn(
            {
                'feature':
                    tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            }, tf.placeholder(dtype=tf.int32, shape=[None])))

  def begin(self):
    self.global_step = tf.train.get_or_create_global_step()

  def after_run(self, run_context, run_values):
    # export saved model
    global_step = run_context.session.run(self.global_step)
    if global_step - self.last_export >= FLAGS.export_model_freq:
      tf.logging.info(
          'Export model for prediction (step={}) ...'.format(global_step))

      self.last_export = global_step
      contrib_estimator.export_all_saved_models(
          self.classifier, os.path.join(self.export_dir, str(global_step)), {
              tf_estimator.ModeKeys.EVAL:
                  self.supervised_input_receiver_fn,
              tf_estimator.ModeKeys.PREDICT:
                  imagenet_input.image_serving_input_fn
          })


def main(argv):
  del argv  # Unused.

  tf.enable_resource_variables()
  tf.set_random_seed(FLAGS.seed)
  set_lr_schedule()
  set_custom_sparsity_map()
  folder_stub = os.path.join(FLAGS.training_method, str(FLAGS.end_sparsity),
                             str(FLAGS.maskupdate_begin_step),
                             str(FLAGS.maskupdate_end_step),
                             str(FLAGS.maskupdate_frequency),
                             str(FLAGS.drop_fraction),
                             str(FLAGS.label_smoothing),
                             str(FLAGS.weight_decay))

  output_dir = FLAGS.output_dir
  if FLAGS.use_folder_stub:
    output_dir = os.path.join(output_dir, folder_stub)

  export_dir = os.path.join(output_dir, 'export_dir')

  # we pass the updated eval and train string to the params dictionary.
  params = {}
  params['output_dir'] = output_dir
  params['training_method'] = FLAGS.training_method
  params['use_tpu'] = FLAGS.use_tpu

  dataset_func = functools.partial(
      imagenet_input.ImageNetInput, data_dir=FLAGS.data_directory,
      transpose_input=False, num_parallel_calls=FLAGS.num_parallel_calls,
      use_bfloat16=False)
  imagenet_train, imagenet_eval = [dataset_func(is_training=is_training)
                                   for is_training in [True, False]]

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=output_dir,
      save_checkpoints_steps=FLAGS.steps_per_checkpoint,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          tpu_job_name=FLAGS.tpu_job_name))

  classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn_w_pruning,
      params=params,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  cpu_classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn_w_pruning,
      params=params,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      export_to_tpu=False,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.num_eval_images % FLAGS.eval_batch_size != 0:
    raise ValueError(
        'eval_batch_size (%d) must evenly divide num_eval_images(%d)!' %
        (FLAGS.eval_batch_size, FLAGS.num_eval_images))

  eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
  if FLAGS.mode == 'eval_once':
    ckpt_path = os.path.join(output_dir, FLAGS.eval_once_ckpt_prefix)
    dataset = imagenet_train if FLAGS.eval_on_train else imagenet_eval
    classifier.evaluate(
        input_fn=dataset.input_fn,
        steps=eval_steps,
        checkpoint_path=ckpt_path,
        name='{0}'.format(FLAGS.eval_once_ckpt_prefix))
  elif FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(output_dir):
      tf.logging.info('Starting to evaluate.')
      try:
        dataset = imagenet_train if FLAGS.eval_on_train else imagenet_eval
        classifier.evaluate(
            input_fn=dataset.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt,
            name='eval')
        # Terminate eval job when final checkpoint is reached
        global_step = int(os.path.basename(ckpt).split('-')[1])
        if global_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d' % global_step)
          break

      except tf.errors.NotFoundError:
        logging('Checkpoint no longer exists,skipping checkpoint.')

  else:
    global_step = estimator._load_global_step_from_checkpoint_dir(output_dir)
    # Session run hooks to export model for prediction
    export_hook = ExportModelHook(cpu_classifier, export_dir)
    hooks = [export_hook]

    if FLAGS.mode == 'train':
      tf.logging.info('start training...')
      classifier.train(
          input_fn=imagenet_train.input_fn,
          hooks=hooks,
          max_steps=FLAGS.train_steps)
    else:
      assert FLAGS.mode == 'train_and_eval'
      tf.logging.info('start training and eval...')
      while global_step < FLAGS.train_steps:
        next_checkpoint = min(global_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        global_step = next_checkpoint
        logging('Completed training up to step :', global_step)
        classifier.evaluate(input_fn=imagenet_eval.input_fn, steps=eval_steps)


if __name__ == '__main__':
  app.run(main)
