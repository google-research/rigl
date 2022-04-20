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

r"""Training script for running experiments.
"""

import os
from typing import List  # Non-expensive-to-import types.

from absl import app
from absl import flags
from absl import logging
import gin
import jax
from jax.scipy.linalg import eigh
import numpy as np
from rigl.rigl_tf2 import mask_updaters
from rigl.rigl_tf2 import metainit
from rigl.rigl_tf2 import utils
import tensorflow.compat.v2 as tf
from pyglib import timer

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', '/tmp/sparse_spectrum',
                    'Directory to save experiment in.')
flags.DEFINE_string('preload_gin_config', '', 'If non-empty reads a gin file '
                    'before parsing gin_config and bindings. This is useful,'
                    'when you want to start from a configuration of another '
                    'run. Values are then overwritten by additional configs '
                    'and bindings provided.')
flags.DEFINE_bool('use_tpu', True, 'Whether to run on TPU or not.')
flags.DEFINE_enum('mode', 'train_eval', ('train_eval', 'hessian'),
                  'Whether to run on TPU or not.')
flags.DEFINE_string(
    'tpu_job_name', 'tpu_worker',
    'Name of the TPU worker job. This is required when having '
    'multiple TPU worker jobs.')
flags.DEFINE_integer('seed', default=0, help=('Sets the random seed.'))
flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')


@tf.function
def get_rows(model, variables, masks, ind_l, indices, x_batch, y_batch,
             is_dense_spectrum):
  """Calculates the rows (given by `ind_l`) of the Hessian."""
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  with tf.GradientTape(persistent=True) as tape:
    predictions = model(x_batch, training=True)
    loss = loss_object(y_batch, predictions)
    grads, = tape.gradient(loss, [variables[ind_l]])
    # Since the variables are masked before not during the forward pass,
    # gradients are dense. We need to ensure they are sparse.
    sparse_grads = grads * masks[ind_l]
    single_grad = tf.reshape(sparse_grads, [-1])
    s_grads = tf.gather(single_grad, indices)

  flattened_list = []
  hessians_slice_vars = tape.jacobian(
      s_grads, variables, experimental_use_pfor=False)
  for h, m in zip(hessians_slice_vars, masks):
    if is_dense_spectrum:
      # We apply the masks since weights are not hard constrained with sparsity.
      vals = tf.reshape(h * m, (h.shape[0], -1))
    else:
      boolean_mask = tf.broadcast_to(tf.equal(m, 1), h.shape)
      vals = tf.reshape(h[boolean_mask], (h.shape[0], -1))
    flattened_list.append(vals)

  res = tf.concat(flattened_list, 1)
  return res


def sparse_hessian_calculator(model,
                              data,
                              rows_at_once,
                              eigvals_path,
                              overwrite,
                              is_dense_spectrum=False):
  """Calculates the Hessian of the model parameters. Biases are dense."""
  # Read all data at once
  x_batch, y_batch = list(data.batch(100000))[0]

  if tf.io.gfile.exists(eigvals_path) and overwrite:
    logging.info('Deleting existing Eigvals: %s', eigvals_path)
    tf.io.gfile.rmtree(eigvals_path)
  if tf.io.gfile.exists(eigvals_path):
    with tf.io.gfile.GFile(eigvals_path, 'rb') as f:
      eigvals = np.load(f)
    logging.info('Eigvals exists, skipping :%s', eigvals_path)
    return eigvals

  # First lets create lists that indicate the valid dimension of each variable.
  # If we want to calculate sparse spectrum, then we have to omit masked
  # dimensions. Biases are dense, therefore have masks of 1's.
  masks = []
  variables = []
  layer_group_indices = []
  for l in model.layers:
    if isinstance(l, utils.PRUNING_WRAPPER):
      # TODO following the outcome of b/148083099, update following.
      # Add the weight, mask and the valid dimensions.
      weight = l.weights[0]
      variables.append(weight)

      mask = l.weights[2]
      masks.append(mask)
      logging.info(mask.shape)

      if is_dense_spectrum:
        n_params = tf.size(mask)
        layer_group_indices.append(tf.range(n_params))
      else:
        fmask = tf.reshape(mask, [-1])
        indices = tf.where(tf.equal(fmask, 1))[:, 0]
        layer_group_indices.append(indices)
      # Add the bias mask of ones and all of its dimensions.
      bias = l.weights[1]
      variables.append(bias)
      masks.append(tf.ones_like(bias))
      layer_group_indices.append(tf.range(tf.size(bias)))
    else:
      # For now we assume all parameterized layers are wrapped with
      # PruneLowMagnitude.
      assert not l.trainable_variables
  result_all = []
  init_timer = timer.Timer()
  init_timer.Start()
  n_total = 0
  logging.info('Calculating Hessian...')
  for i, inds in enumerate(layer_group_indices):
    n_split = np.ceil(tf.size(inds).numpy() / rows_at_once)
    logging.info('Nsplit: %d', n_split)
    for c_slice in np.array_split(inds.numpy(), n_split):
      res = get_rows(model, variables, masks, i, c_slice, x_batch, y_batch,
                     is_dense_spectrum)
      result_all.append(res.numpy())
      n_total += res.shape[0]
      target_n = float(res.shape[1])
    logging.info('%.3f %% ..', (n_total / target_n))
  # We convert in numpy so that it is on cpu automatically and we don't get OOM.
  c_hessian = np.concatenate(result_all, 0)
  logging.info('Total runtime for hessian: %.3f s', init_timer.GetDuration())
  init_timer.Start()
  eigens = jax.jit(eigh, backend='cpu')(c_hessian)
  eigvals = np.asarray(eigens[0])
  with tf.io.gfile.GFile(eigvals_path, 'wb') as f:
    np.save(f, eigvals)
  logging.info('EigVals saved: %s', eigvals_path)
  logging.info('Total runtime for eigvals: %.3f s', init_timer.GetDuration())
  return eigvals


@gin.configurable(denylist=['model', 'ds_train', 'logdir'])
def hessian(model,
            ds_train,
            logdir,
            ckpt_ids = gin.REQUIRED,
            overwrite = False,
            batch_size = 1000,
            rows_at_once = 10,
            is_dense_spectrum = False):
  """Loads checkpoints under a folder and calculates their hessian spectrum."""
  # Note that hessian is calculated using the same batch in different runs.
  # This is needed since if the job dies and restarted we want it to be same.
  data_hessian = ds_train.take(batch_size)
  for ckpt_id in ckpt_ids:
    # `cp-0005.ckpt.index` -> 15012
    ckpt = tf.train.Checkpoint(model=model)
    c_path = os.path.join(logdir, 'ckpt-%d' % ckpt_id)
    ckpt.restore(c_path)
    logging.info('Loaded from: %s', c_path)
    eigvals_path = c_path + '.eigvals'
    sparse_hessian_calculator(
        model=model, data=data_hessian, eigvals_path=eigvals_path,
        overwrite=overwrite, is_dense_spectrum=is_dense_spectrum,
        rows_at_once=rows_at_once)


def update_prune_step(model, step):
  for layer in model.layers:
    if isinstance(layer, utils.PRUNING_WRAPPER):
      # Assign iteration count to the layer pruning_step.
      layer.pruning_step.assign(step)


def log_sparsities(model):
  for layer in model.layers:
    if isinstance(layer, utils.PRUNING_WRAPPER):
      for _, mask, threshold in layer.pruning_vars:
        scalar_name = f'sparsity/{mask.name}'
        sparsity = 1 - tf.reduce_mean(mask)
        tf.summary.scalar(scalar_name, sparsity)
        tf.summary.scalar(f'threshold/{threshold.name}', threshold)


def cosine_distance(x, y):
  """Calculates the distance between 2 tensors of same shape."""
  normalizedx = tf.math.l2_normalize(x)
  normalizedy = tf.math.l2_normalize(y)
  return 1. - tf.reduce_sum(tf.multiply(normalizedx, normalizedy))


def flatten_list_of_vars(var_list):
  flat_vars = [tf.reshape(v, -1) for v in var_list]
  return tf.concat(flat_vars, axis=-1)


def var_to_img(tensor):
  if len(tensor.shape) <= 1:
    gray_image = tf.reshape(tensor, [1, -1])
  elif len(tensor.shape) == 2:
    gray_image = tensor
  else:
    gray_image = tf.reshape(tensor, [-1, tensor.shape[-1]])
  # (H, W) -> (1, H, W, 1)
  return tf.expand_dims(tf.expand_dims(gray_image, 0), -1)


def mask_gradients(model, gradients, variables):
  name_to_grad = {var.name: grad for grad, var in zip(gradients, variables)}
  for layer in model.layers:
    if isinstance(layer, utils.PRUNING_WRAPPER):
      for weights, mask, _ in layer.pruning_vars:
        if weights.name in name_to_grad:
          name_to_grad[weights.name] = name_to_grad[weights.name] * mask
  masked_gradients = [name_to_grad[var.name] for var in variables]
  return masked_gradients


@gin.configurable(
    'training', denylist=['model', 'ds_train', 'ds_test', 'logdir'])
def train_model(model,
                ds_train,
                ds_test,
                logdir,
                total_steps = 5000,
                batch_size = 128,
                val_batch_size = 1000,
                save_freq = 5,
                log_freq = 250,
                use_metainit = False,
                oneshot_prune_fraction = 0.,
                gradient_regularization=0):
  """Training of the CNN on MNIST."""
  logging.info('Writing training logs to %s', logdir)
  writer = tf.summary.create_file_writer(os.path.join(logdir, 'train_logs'))
  optimizer = utils.get_optimizer(total_steps)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  train_batch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_batch_accuracy')
  # Let's create 2 disjoint validation sets.
  (val_x, val_y), (val2_x, val2_y) = [
      d for d in ds_train.take(val_batch_size * 2).batch(val_batch_size)
  ]

  # We use a separate set than the one we are using in our training.
  def loss_fn(x, y):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    predictions = model(x, training=True)
    reg_loss = tf.add_n(model.losses) if model.losses else 0
    return loss_object(y, predictions) + reg_loss

  mask_updater = mask_updaters.get_mask_updater(model, optimizer, loss_fn)
  if mask_updater:
    mask_updater.set_validation_data(val2_x, val2_y)
  update_prune_step(model, 0)
  if oneshot_prune_fraction > 0:
    logging.info('Running one shot prunning at the beginning.')
    if not mask_updater:
      raise ValueError('mask_updater does not exists. Please set '
                       'mask_updater.update_alg flag for one shot pruning.')
    mask_updater.prune(oneshot_prune_fraction)
  if use_metainit:
    n_params = 0
    for layer in model.layers:
      if isinstance(layer, utils.PRUNING_WRAPPER):
        for _, mask, _ in layer.pruning_vars:
          n_params += tf.reduce_sum(mask)
    metainit.meta_init(model, loss_object, (128, 28, 28, 1), (128, 10),
                       n_params, mask_gradient_fn=mask_gradients)
  # This is used to calculate some distances, would give incorrect results when
  # we restart the training.
  initial_params = list(map(lambda a: a.numpy(), model.trainable_variables))

  # Create the checkpoint object and restore if there is a checkpoint in the
  # folder.
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=logdir, max_to_keep=None)
  if ckpt_manager.latest_checkpoint:
    logging.info('Restored from %s', ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    is_restored = True
  else:
    logging.info('Starting from scratch.')
    is_restored = False
  # Obtain global_step after loading checkpoint.
  global_step = optimizer.iterations
  tf.summary.experimental.set_step(global_step)
  trainable_vars = model.trainable_variables

  def get_gradients(x, y, log_batch_gradient=False, is_regularized=True):
    """Gets spars gradients and possibly logs some statistics."""
    is_grad_regularized = gradient_regularization != 0
    with tf.GradientTape(persistent=is_grad_regularized) as tape:
      predictions = model(x, training=True)
      batch_loss = loss_object(y, predictions)
      if is_regularized and is_grad_regularized:
        gradients = tape.gradient(batch_loss, trainable_vars)
        gradients = mask_gradients(model, gradients, trainable_vars)
        grad_vec = flatten_list_of_vars(gradients)
        batch_loss += tf.nn.l2_loss(grad_vec) * gradient_regularization
      # Regularization might have been disabled.
      reg_loss = tf.add_n(model.losses) if model.losses else 0
      if is_regularized:
        batch_loss += reg_loss
    gradients = tape.gradient(batch_loss, trainable_vars)
    # Gradients are dense, we should mask them to ensure updates are sparse;
    # So is the norm calculation.
    gradients = mask_gradients(model, gradients, trainable_vars)
    # If batch gradient log it.
    if log_batch_gradient:
      tf.summary.scalar('train_batch_loss', batch_loss)
      tf.summary.scalar('train_batch_reg_loss', reg_loss)
      train_batch_accuracy.update_state(y, predictions)
      tf.summary.scalar('train_batch_accuracy', train_batch_accuracy.result())
      train_batch_accuracy.reset_states()
    return gradients

  def log_fn():
    logging.info('Logging at iter: %d', global_step.numpy())
    log_sparsities(model)
    test_loss, test_acc = test_model(model, ds_test)
    tf.summary.scalar('test_loss', test_loss)
    tf.summary.scalar('test_acc', test_acc)
    # Log gradient norm.
    # We want to obtain/log gradients without regularization term.
    gradients = get_gradients(val_x, val_y, log_batch_gradient=False,
                              is_regularized=False)
    for var, grad in zip(trainable_vars, gradients):
      tf.summary.scalar(f'gradnorm/{var.name}', tf.norm(grad))
    # Log all gradients together
    all_norm = tf.norm(flatten_list_of_vars(gradients))
    tf.summary.scalar('.allparams/gradnorm', all_norm)
    # Log momentum values:
    for s_name in optimizer.get_slot_names():
      # Currently we only log momentum.
      if s_name not in ['momentum']:
        continue
      all_slots = [optimizer.get_slot(var, s_name) for var in trainable_vars]
      all_norm = tf.norm(flatten_list_of_vars(all_slots))
      tf.summary.scalar(f'.allparams/norm_{s_name}', all_norm)
    # Log distance to init.
    for initial_val, val in zip(initial_params, model.trainable_variables):
      tf.summary.scalar(f'dist_init_l2/{val.name}', tf.norm(initial_val - val))
      cos_distance = cosine_distance(initial_val, val)
      tf.summary.scalar(f'dist_init_cosine/{val.name}', cos_distance)
    # Mask update logs:
    if mask_updater:
      tf.summary.scalar('drop_fraction', mask_updater.last_drop_fraction)
    # Log all distances together.
    flat_initial = flatten_list_of_vars(initial_params)
    flat_current = flatten_list_of_vars(model.trainable_variables)
    tf.summary.scalar('.allparams/dist_init_l2/',
                      tf.norm(flat_initial - flat_current))
    tf.summary.scalar('.allparams/dist_init_cosine/',
                      cosine_distance(flat_initial, flat_current))
    # Log masks
    for layer in model.layers:
      if isinstance(layer, utils.PRUNING_WRAPPER):
        for _, mask, _ in layer.pruning_vars:
          tf.summary.image('mask/%s' % mask.name, var_to_img(mask))
    writer.flush()

  def save_fn(step=None):
    save_step = step if step else global_step
    saved_ckpt = ckpt_manager.save(checkpoint_number=save_step)
    logging.info('Saved checkpoint: %s', saved_ckpt)

  with writer.as_default():
    for x, y in ds_train.repeat().shuffle(
        buffer_size=60000).batch(batch_size):
      if global_step >= total_steps:
        logging.info('Total steps: %d is completed', global_step.numpy())
        save_fn()
        break
      update_prune_step(model, global_step)
      if tf.equal(global_step, 0):
        logging.info('Seed: %s First 10 Label: %s', FLAGS.seed, y[:10])
      if global_step % save_freq == 0:
        # If just loaded, don't save it again.
        if is_restored:
          is_restored = False
        else:
          save_fn()
      if global_step % log_freq == 0:
        log_fn()
      gradients = get_gradients(x, y, log_batch_gradient=True)
      tf.summary.scalar('lr', optimizer.lr(global_step))
      optimizer.apply_gradients(zip(gradients, trainable_vars))
      if mask_updater and mask_updater.is_update_iter(global_step):
        # Save the network before mask_update, we want to use negative integers
        # for this.
        save_fn(step=(-global_step + 1))
        # Gradient norm before.
        gradients = get_gradients(
            val_x, val_y, log_batch_gradient=False, is_regularized=False)
        norm_before = tf.norm(flatten_list_of_vars(gradients))
        results = mask_updater.update(global_step)
        # Save network again
        save_fn(step=-global_step)
        if results:
          for mask_name, drop_frac in results.items():
            tf.summary.scalar('drop_fraction/%s' % mask_name, drop_frac)

        # Gradient norm after mask update.
        gradients = get_gradients(
            val_x, val_y, log_batch_gradient=False, is_regularized=False)
        norm_after = tf.norm(flatten_list_of_vars(gradients))
        tf.summary.scalar('.allparams/gradnorm_mask_update_improvment',
                          norm_after - norm_before)

    logging.info('Performance after training:')
    log_fn()
  return model


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
  return test_loss.result(), test_accuracy.result()


def main(unused_argv):
  tf.random.set_seed(FLAGS.seed)
  init_timer = timer.Timer()
  init_timer.Start()

  if FLAGS.mode == 'hessian':
    # Load default values from the original experiment.
    FLAGS.preload_gin_config = os.path.join(FLAGS.logdir,
                                            'operative_config.gin')

  # Maybe preload a gin config.
  if FLAGS.preload_gin_config:
    config_path = FLAGS.preload_gin_config
    gin.parse_config_file(config_path)
    logging.info('Gin configuration pre-loaded from: %s', config_path)

  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  ds_train, ds_test, info = utils.get_dataset()
  input_shape = info.features['image'].shape
  num_classes = info.features['label'].num_classes
  logging.info('Input Shape: %s', input_shape)
  logging.info('train samples: %s', info.splits['train'].num_examples)
  logging.info('test samples: %s', info.splits['test'].num_examples)

  pruning_params = utils.get_pruning_params()
  model = utils.get_network(pruning_params, input_shape, num_classes)
  model.summary(print_fn=logging.info)
  if FLAGS.mode == 'train_eval':
    train_model(model, ds_train, ds_test, FLAGS.logdir)
  elif FLAGS.mode == 'hessian':
    test_model(model, ds_test)
    hessian(model, ds_train, FLAGS.logdir)
  logging.info('Total runtime: %.3f s', init_timer.GetDuration())

  logconfigfile_path = os.path.join(
      FLAGS.logdir,
      'hessian_' if FLAGS.mode == 'hessian' else '' + 'operative_config.gin')
  with tf.io.gfile.GFile(logconfigfile_path, 'w') as f:
    f.write('# Gin-Config:\n %s' % gin.config.operative_config_str())


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
