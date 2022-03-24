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

Implement pruning method during training:

Specify the pruning method to use using FLAGS.training_method
- To train a model with no pruning, specify FLAGS.training_method='baseline'

Specify desired end sparsity using FLAGS.end_sparsity
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from rigl import sparse_optimizers
from rigl import sparse_utils
from rigl.cifar_resnet.data_helper import input_fn
from rigl.cifar_resnet.resnet_model import WideResNetModel
from rigl.imagenet_resnet import utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import training as contrib_training
from tensorflow.contrib.model_pruning.python import pruning

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow runtime to use.')
flags.DEFINE_integer('ps_task', 0,
                     'Task id of the replica running the training.')
flags.DEFINE_integer('keep_checkpoint_max', 5,
                     'Number of checkpoints to save, set 0 for all.')
flags.DEFINE_string('pruning_hparams', '',
                    'Comma separated list of pruning-related hyperparameters')
flags.DEFINE_string('train_dir', '/tmp/cifar10/',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_string(
    'load_mask_dir', '',
    'Directory of a trained model from which to load only the mask')
flags.DEFINE_string(
    'initial_value_checkpoint', '',
    'Directory of a model from which to load only the parameters')
flags.DEFINE_integer(
    'seed', default=0, help=('Sets the random seed.'))
flags.DEFINE_float('momentum', 0.9, 'The momentum value.')
# 250 Epochs
flags.DEFINE_integer('max_steps', 97656, 'Number of steps to run.')
flags.DEFINE_float('l2', 5e-4, 'Scale factor for L2 weight decay.')
flags.DEFINE_integer('resnet_depth', 16, 'Number of core convolutional layers'
                     'in the network.')
flags.DEFINE_integer('resnet_width', 4, 'Width of the residual blocks.')
flags.DEFINE_string(
    'data_directory', '', 'data directory where cifar10 records are stored')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('dataset_size', 50000, 'Size of training dataset.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('checkpoint_steps', 5000, 'Specifies step interval for'
                     'saving model checkpoints.')
flags.DEFINE_integer(
    'summaries_steps', 300, 'Specifies interval in steps for'
    'saving model summaries.')
flags.DEFINE_bool('per_class_metrics', True, 'Whether to add per-class'
                  'performance summaries.')
flags.DEFINE_enum('mode', 'train', ('train_and_eval', 'train', 'eval'),
                  'String that specifies either inference or training')

# pruning flags
flags.DEFINE_integer('sparsity_begin_step', 20000, 'Step to begin pruning at.')
flags.DEFINE_integer('sparsity_end_step', 75000, 'Step to end pruning at.')
flags.DEFINE_integer('pruning_frequency', 1000,
                     'Step interval between pruning steps.')
flags.DEFINE_float('end_sparsity', 0.9,
                   'Target sparsity desired by end of training.')
flags.DEFINE_enum(
    'training_method', 'baseline',
    ('scratch', 'set', 'baseline', 'momentum', 'rigl', 'static', 'snip',
     'prune'),
    'Method used for training sparse network. `scratch` means initial mask is '
    'kept during training. `set` is for sparse evalutionary training and '
    '`baseline` is for dense baseline.')
flags.DEFINE_bool('prune_first_layer', False,
                  'Whether or not to apply sparsification to the first layer')
flags.DEFINE_bool('prune_last_layer', True,
                  'Whether or not to apply sparsification to the last layer')
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
flags.DEFINE_integer('maskupdate_begin_step', 0, 'Step to begin mask updates.')
flags.DEFINE_integer('maskupdate_end_step', 75000, 'Step to end mask updates.')
flags.DEFINE_integer('maskupdate_frequency', 100,
                     'Step interval between mask updates.')
flags.DEFINE_string(
    'mask_init_method',
    default='random',
    help='If not empty string and mask is not loaded from a checkpoint, '
    'indicates the method used for mask initialization. One of the following: '
    '`random`, `erdos_renyi`.')
flags.DEFINE_float('training_steps_multiplier', 1.0,
                   'Training schedule is shortened or extended with the '
                   'multiplier, if it is not 1.')

FLAGS = flags.FLAGS
PARAM_SUFFIXES = ('gamma', 'beta', 'weights', 'biases')
MASK_SUFFIX = 'mask'
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]


def create_eval_metrics(labels, logits):
  """Creates the evaluation metrics for the model."""

  eval_metrics = {}
  label_keys = CLASSES
  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  eval_metrics['eval_accuracy'] = tf.metrics.accuracy(
      labels=labels, predictions=predictions)
  if FLAGS.per_class_metrics:
    with tf.name_scope('class_level_summaries') as scope:
      for i in range(len(label_keys)):
        labels = tf.cast(labels, tf.int64)
        name = scope + '/' + label_keys[i]
        eval_metrics[('class_level_summaries/precision/' +
                      label_keys[i])] = tf.metrics.precision_at_k(
                          labels=labels,
                          predictions=logits,
                          class_id=i,
                          k=1,
                          name=name)
        eval_metrics[('class_level_summaries/recall/' +
                      label_keys[i])] = tf.metrics.recall_at_k(
                          labels=labels,
                          predictions=logits,
                          class_id=i,
                          k=1,
                          name=name)
  return eval_metrics


def train_fn(training_method, global_step, total_loss, train_dir, accuracy,
             top_5_accuracy):
  """Training script for resnet model.

  Args:
   training_method: specifies the method used to sparsify networks.
   global_step: the current step of training/eval.
   total_loss: tensor float32 of the cross entropy + regularization losses.
   train_dir: string specifying where directory where summaries are saved.
   accuracy: tensor float32 batch classification accuracy.
   top_5_accuracy: tensor float32 batch classification accuracy (top_5 classes).

  Returns:
    hooks: summary tensors to be computed at each training step.
    eval_metrics: set to None during training.
    train_op: the optimization term.
  """
  # Rougly drops at every 30k steps.
  boundaries = [30000, 60000, 90000]
  if FLAGS.training_steps_multiplier != 1.0:
    multiplier = FLAGS.training_steps_multiplier
    boundaries = [int(x * multiplier) for x in boundaries]
    tf.logging.info(
        'Learning Rate boundaries are updated with multiplier:%.2f', multiplier)

  learning_rate = tf.train.piecewise_constant(
      global_step,
      boundaries,
      values=[0.1 / (5.**i) for i in range(len(boundaries) + 1)],
      name='lr_schedule')

  optimizer = tf.train.MomentumOptimizer(
      learning_rate, momentum=FLAGS.momentum, use_nesterov=True)

  if training_method == 'set':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseSETOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)
  elif training_method == 'static':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseStaticOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)
  elif training_method == 'momentum':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseMomentumOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, momentum=FLAGS.s_momentum,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        grow_init=FLAGS.grow_init,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal, use_tpu=False)
  elif training_method == 'rigl':
    # We override the train op to also update the mask.
    optimizer = sparse_optimizers.SparseRigLOptimizer(
        optimizer, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency,
        drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal,
        initial_acc_scale=FLAGS.rigl_acc_scale, use_tpu=False)
  elif training_method == 'snip':
    optimizer = sparse_optimizers.SparseSnipOptimizer(
        optimizer, mask_init_method=FLAGS.mask_init_method,
        default_sparsity=FLAGS.end_sparsity, use_tpu=False)
  elif training_method in ('scratch', 'baseline', 'prune'):
    pass
  else:
    raise ValueError('Unsupported pruning method: %s' % FLAGS.training_method)
  # Create the training op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(total_loss, global_step)

  if training_method == 'prune':
    # construct the necessary hparams string from the FLAGS
    hparams_string = ('begin_pruning_step={0},'
                      'sparsity_function_begin_step={0},'
                      'end_pruning_step={1},'
                      'sparsity_function_end_step={1},'
                      'target_sparsity={2},'
                      'pruning_frequency={3},'
                      'threshold_decay=0,'
                      'use_tpu={4}'.format(
                          FLAGS.sparsity_begin_step,
                          FLAGS.sparsity_end_step,
                          FLAGS.end_sparsity,
                          FLAGS.pruning_frequency,
                          False,
                      ))
    # Parse pruning hyperparameters
    pruning_hparams = pruning.get_pruning_hparams().parse(hparams_string)

    # Create a pruning object using the pruning hyperparameters
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

    tf.logging.info('starting mask update op')

    # We override the train op to also update the mask.
    with tf.control_dependencies([train_op]):
      train_op = pruning_obj.conditional_mask_update_op()

  masks = pruning.get_masks()
  mask_metrics = utils.mask_summaries(masks)
  for name, tensor in mask_metrics.items():
    tf.summary.scalar(name, tensor)

  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.scalar('top_5_accuracy', top_5_accuracy)
  # Logging drop_fraction if dynamic sparse training.
  if training_method in ('set', 'momentum', 'rigl', 'static'):
    tf.summary.scalar('drop_fraction', optimizer.drop_fraction)

  summary_op = tf.summary.merge_all()
  summary_hook = tf.train.SummarySaverHook(
      save_secs=300, output_dir=train_dir, summary_op=summary_op)
  hooks = [summary_hook]
  eval_metrics = None

  return hooks, eval_metrics, train_op


def build_model(mode,
                images,
                labels,
                training_method='baseline',
                num_classes=10,
                depth=10,
                width=4):
  """Build the wide ResNet model for training or eval.

  If regularizer is specified, a regularizer term is added to the loss function.
  The regularizer term is computed using either the pre-softmax activation or an
  auxiliary network logits layer based upon activations earlier in the network
  after the first resnet block.

  Args:
    mode: String for whether training or evaluation is taking place.
    images:  A 4D float32 tensor containing the model input images.
    labels:  A int32 tensor of size (batch size, number of classes)
    containing the model labels.
    training_method: The method used to sparsify the network weights.
    num_classes: The number of distinct labels in the dataset.
    depth: Number of core convolutional layers in the network.
    width: The width of the convolurional filters in the resnet block.

  Returns:
    total_loss: A 1D float32 tensor that is the sum of cross-entropy and
      all regularization losses.
    accuracy: A 1D float32 accuracy tensor.
  Raises:
      ValueError: if depth is not the minimum amount required to build the
        model.
  """
  regularizer_term = tf.constant(FLAGS.l2, tf.float32)
  kernel_regularizer = contrib_layers.l2_regularizer(scale=regularizer_term)

  # depth should be 6n+4 where n is the desired number of resnet blocks
  # if n=2,depth=10  n=3,depth=22, n=5,depth=34 n=7,depth=46
  if (depth - 4) % 6 != 0:
    raise ValueError('Depth of ResNet specified not sufficient.')

  if mode == 'train':
    is_training = True
  else:
    is_training = False
  # 'threshold' would create layers with mask.
  pruning_method = 'baseline' if training_method == 'baseline' else 'threshold'

  model = WideResNetModel(
      is_training=is_training,
      regularizer=kernel_regularizer,
      data_format='channels_last',
      pruning_method=pruning_method,
      prune_first_layer=FLAGS.prune_first_layer,
      prune_last_layer=FLAGS.prune_last_layer)

  logits = model.build(
      images, depth=depth, width=width, num_classes=num_classes)

  global_step = tf.train.get_or_create_global_step()

  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

  in_top_5 = tf.cast(
      tf.nn.in_top_k(predictions=logits, targets=labels, k=5), tf.float32)

  top_5_accuracy = tf.cast(tf.reduce_mean(in_top_5), tf.float32)

  return global_step, accuracy, top_5_accuracy, logits


def wide_resnet_w_pruning(features, labels, mode, params):
  """The model_fn for ResNet wide with pruning.

  Args:
    features: A float32 batch of images.
    labels: A int32 batch of labels.
    mode: Specifies whether training or evaluation.
    params: Dictionary of parameters passed to the model.

  Returns:
    A EstimatorSpec for the model

  Raises:
      ValueError: if mode is not recognized as train or eval.
  """

  if isinstance(features, dict):
    features = features['feature']

  train_dir = params['train_dir']
  training_method = params['training_method']

  global_step, accuracy, top_5_accuracy, logits = build_model(
      mode=mode,
      images=features,
      labels=labels,
      training_method=training_method,
      num_classes=FLAGS.num_classes,
      depth=FLAGS.resnet_depth,
      width=FLAGS.resnet_width)

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

  with tf.name_scope('computing_cross_entropy_loss'):
    entropy_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    tf.summary.scalar('cross_entropy_loss', entropy_loss)

  with tf.name_scope('computing_total_loss'):
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

  if mode == tf_estimator.ModeKeys.TRAIN:
    hooks, eval_metrics, train_op = train_fn(training_method, global_step,
                                             total_loss, train_dir, accuracy,
                                             top_5_accuracy)
  elif mode == tf_estimator.ModeKeys.EVAL:
    hooks = None
    train_op = None
    with tf.name_scope('summaries'):
      eval_metrics = create_eval_metrics(labels, logits)
  else:
    raise ValueError('mode not recognized as training or eval.')

  # If given load parameter values.
  if FLAGS.initial_value_checkpoint:
    tf.logging.info('Loading inital values from: %s',
                    FLAGS.initial_value_checkpoint)
    utils.initialize_parameters_from_ckpt(FLAGS.initial_value_checkpoint,
                                          FLAGS.train_dir, PARAM_SUFFIXES)

  # Load or randomly initialize masks.
  if (FLAGS.load_mask_dir and
      FLAGS.training_method not in ('snip', 'baseline', 'prune')):
    # Init masks.
    tf.logging.info('Loading masks from %s', FLAGS.load_mask_dir)
    utils.initialize_parameters_from_ckpt(FLAGS.load_mask_dir, FLAGS.train_dir,
                                          MASK_SUFFIX)
    scaffold = tf.train.Scaffold()
  elif (FLAGS.mask_init_method and
        FLAGS.training_method not in ('snip', 'baseline', 'scratch', 'prune')):
    tf.logging.info('Initializing masks using method: %s',
                    FLAGS.mask_init_method)
    all_masks = pruning.get_masks()
    assigner = sparse_utils.get_mask_init_fn(
        all_masks, FLAGS.mask_init_method, FLAGS.end_sparsity, {})
    def init_fn(scaffold, session):
      """A callable for restoring variable from a checkpoint."""
      del scaffold  # Unused.
      session.run(assigner)
    scaffold = tf.train.Scaffold(init_fn=init_fn)
  else:
    assert FLAGS.training_method in ('snip', 'baseline', 'prune')
    scaffold = None
    tf.logging.info('No mask is set, starting dense.')

  return tf_estimator.EstimatorSpec(
      mode=mode,
      training_hooks=hooks,
      loss=total_loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics,
      scaffold=scaffold)


def main(argv):
  del argv  # Unused.
  tf.set_random_seed(FLAGS.seed)
  if FLAGS.training_steps_multiplier != 1.0:
    multiplier = FLAGS.training_steps_multiplier
    FLAGS.max_steps = int(FLAGS.max_steps * multiplier)
    FLAGS.maskupdate_begin_step = int(FLAGS.maskupdate_begin_step * multiplier)
    FLAGS.maskupdate_end_step = int(FLAGS.maskupdate_end_step * multiplier)
    FLAGS.sparsity_begin_step = int(FLAGS.sparsity_begin_step * multiplier)
    FLAGS.sparsity_end_step = int(FLAGS.sparsity_end_step * multiplier)
    tf.logging.info(
        'Training schedule is updated with multiplier: %.2f', multiplier)
  # configures train directories based upon hyperparameters used.
  if FLAGS.training_method == 'prune':
    folder_stub = os.path.join(FLAGS.training_method, str(FLAGS.end_sparsity),
                               str(FLAGS.sparsity_begin_step),
                               str(FLAGS.sparsity_end_step),
                               str(FLAGS.pruning_frequency))

  elif FLAGS.training_method in ('set', 'momentum', 'rigl', 'static'):
    folder_stub = os.path.join(FLAGS.training_method, str(FLAGS.end_sparsity),
                               str(FLAGS.maskupdate_begin_step),
                               str(FLAGS.maskupdate_end_step),
                               str(FLAGS.maskupdate_frequency))
  elif FLAGS.training_method in ('baseline', 'snip', 'scratch'):
    folder_stub = os.path.join(FLAGS.training_method, str(0.0), str(0.0),
                               str(0.0), str(0.0))
  else:
    raise ValueError('Training method is not known %s' % FLAGS.training_method)

  train_dir = os.path.join(FLAGS.train_dir, folder_stub)

  # we pass the updated eval and train string to the params dictionary.
  params = {}
  params['train_dir'] = train_dir
  params['data_split'] = FLAGS.mode
  params['batch_size'] = FLAGS.batch_size
  params['data_directory'] = FLAGS.data_directory
  params['mode'] = FLAGS.mode
  params['training_method'] = FLAGS.training_method

  run_config = tf_estimator.RunConfig(
      model_dir=train_dir,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      save_summary_steps=FLAGS.summaries_steps,
      save_checkpoints_steps=FLAGS.checkpoint_steps,
      log_step_count_steps=100)

  classifier = tf_estimator.Estimator(
      model_fn=wide_resnet_w_pruning,
      model_dir=train_dir,
      config=run_config,
      params=params)

  if FLAGS.mode == 'eval':
    eval_steps = 10000 // FLAGS.batch_size
    # Run evaluation when there's a new checkpoint
    for ckpt in contrib_training.checkpoints_iterator(train_dir):
      print('Starting to evaluate.')
      try:
        classifier.evaluate(
            input_fn=input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt,
            name='eval')
        # Terminate eval job when final checkpoint is reached
        global_step = int(os.path.basename(ckpt).split('-')[1])
        if global_step >= FLAGS.max_steps:
          print('Evaluation finished after training step %d' % global_step)
          break

      except tf.errors.NotFoundError:
        print('Checkpoint no longer exists,skipping checkpoint.')

  else:
    print('Starting training...')
    if FLAGS.mode == 'train':
      classifier.train(input_fn=input_fn, max_steps=FLAGS.max_steps)


if __name__ == '__main__':
  tf.app.run(main)
