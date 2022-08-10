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

"""Utility functions for sparse tf agents training."""

import re
from absl import logging
import gin
from rigl import sparse_utils as sparse_utils_rigl
from rigl.rl import sparse_utils

import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

PRUNING_WRAPPER = pruning_wrapper.PruneLowMagnitude
_LAYER_TYPES_TO_WRAP = (tf.keras.layers.Dense, tf.keras.layers.Conv2D,
                        tf.keras.layers.Conv1D)


def log_total_params(networks):
  total_params = 0
  for net in networks:
    total_net_params, _ = sparse_utils.get_total_params(net)
    total_params += total_net_params
  with tf.name_scope('Params/'):
    tf.compat.v2.summary.scalar('total', total_params)


def scale_width(num_units, width):
  assert width > 0
  return int(max(1, num_units * width))


@gin.configurable
def wrap_all_layers(layers,
                    input_dim,
                    mode='constant',
                    mask_init_method='erdos_renyi_kernel',
                    initial_sparsity=0.0,
                    final_sparsity=0.9,
                    begin_step=200000,
                    end_step=600000,
                    frequency=10000):
  """Wraps a list of dense keras layers to be used by sparse training."""
  # We only need to define static masks here, we will update them through
  # mask updater later.
  new_layers = []
  if mode == 'constant':
    for layer in layers:
      schedule = pruning_schedule.ConstantSparsity(
          target_sparsity=0, begin_step=1000000000)
      new_layers.append(PRUNING_WRAPPER(layer, pruning_schedule=schedule))
  elif mode == 'prune':
    logging.info('Pruning schedule: initial sparsity: %f', initial_sparsity)
    logging.info('Pruning schedule: mask_init_method: %s', mask_init_method)
    logging.info('Pruning schedule: final sparsity: %f', final_sparsity)
    logging.info('Pruning schedule: begin step: %f', begin_step)
    logging.info('Pruning schedule: end step: %f', end_step)
    logging.info('Pruning schedule: frequency: %f', frequency)

    # Create dummy masks to get layer-wise sparsities. This is because the
    # get_sparsities function expects mask variables to calculate the
    # sparsities.
    dummy_masks_dict = {}
    layer_input_dim = input_dim
    for layer in layers:
      mask = tf.Variable(tf.ones([layer_input_dim, layer.units]),
                         trainable=False, name=f'dummymask_{layer.name}')
      layer_input_dim = layer.units
      dummy_masks_dict[layer.name] = mask

    # Get layer-wise sparsities.
    extract_name_fn = lambda x: re.findall('(.+):0', x)[0]
    reverse_dict = {v.name: k
                    for k, v in dummy_masks_dict.items()}
    sparsity_dict = sparse_utils_rigl.get_sparsities(
        list(dummy_masks_dict.values()),
        mask_init_method,
        final_sparsity,
        custom_sparsity_map={},
        extract_name_fn=extract_name_fn)
    # This dict will have {layer_name: layer_sparsity}
    renamed_sparsity_dict = {reverse_dict[k]: float(v)
                             for k, v in sparsity_dict.items()}
    # Wrap layers with possibly non-uniform pruning schedule.
    for layer in layers:
      sparsity = renamed_sparsity_dict[layer.name]
      logging.info('Layer: %s, sparsity: %f', layer.name, sparsity)
      schedule = pruning_schedule.PolynomialDecay(
          initial_sparsity=initial_sparsity,
          final_sparsity=sparsity,
          begin_step=begin_step,
          end_step=end_step,
          frequency=frequency)
      new_layers.append(PRUNING_WRAPPER(layer, pruning_schedule=schedule))

  return new_layers


@gin.configurable
def wrap_layer(layer,
               mode='constant',
               initial_sparsity=0.0,
               final_sparsity=0.9,
               begin_step=200000,
               end_step=600000,
               frequency=10000):
  """Wraps a keras layer to be used by sparse training."""
  # We only need to define static masks here, we will update them through
  # mask updater later.
  if mode == 'constant':
    schedule = pruning_schedule.ConstantSparsity(
        target_sparsity=0, begin_step=1000000000)
  elif mode == 'prune':
    logging.info('Pruning schedule: initial sparsity: %f', initial_sparsity)
    logging.info('Pruning schedule: final sparsity: %f', final_sparsity)
    logging.info('Pruning schedule: begin step: %f', begin_step)
    logging.info('Pruning schedule: end step: %f', end_step)
    logging.info('Pruning schedule: frequency: %f', frequency)
    schedule = pruning_schedule.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step,
        frequency=frequency)

  return PRUNING_WRAPPER(layer, pruning_schedule=schedule)


def is_valid_layer_to_wrap(layer):
  for layer_type in _LAYER_TYPES_TO_WRAP:
    if isinstance(layer, layer_type):
      return True

  return False


@gin.configurable
def log_sparsities(model, model_name='q_net', log_images=False):
  """Logs relevant sparsity stats to tensorboard."""
  for layer in sparse_utils.get_all_pruning_layers(model):
    for _, mask, threshold in layer.pruning_vars:
      if log_images:
        reshaped_mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)
        with tf.name_scope('Masks/'):
          tf.compat.v2.summary.image(f'{model_name}/{mask.name}', reshaped_mask)
      with tf.name_scope('Sparsity/'):
        sparsity = 1 - tf.reduce_mean(mask)
        tf.compat.v2.summary.scalar(f'{model_name}/{mask.name}', sparsity)
      with tf.name_scope('Threshold/'):
        tf.compat.v2.summary.scalar(f'{model_name}/{threshold.name}', threshold)

  total_params, nparam_dict = sparse_utils.get_total_params(model)
  with tf.name_scope('Params/'):
    tf.compat.v2.summary.scalar(f'{model_name}/total', total_params)
    for k, val in nparam_dict.items():
      tf.compat.v2.summary.scalar(f'{model_name}/' + k, val)


def update_prune_step(model, step):
  for layer in sparse_utils.get_all_pruning_layers(model):
    # Assign iteration count to the layer pruning_step.
    layer.pruning_step.assign(step)


def flatten_list_of_vars(var_list):
  flat_vars = [tf.reshape(v, [-1]) for v in var_list]
  return tf.concat(flat_vars, axis=-1)


@gin.configurable
def log_snr(tape, loss, step, variables_to_train, freq=1000):
  """Given a gradient tape and loss, it logs signal-to-noise ratio."""

  def true_fn():
    grads_per_sample = tape.jacobian(loss, variables_to_train)
    list_of_snrs = []
    for grad in grads_per_sample:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
      grad_mean = tf.math.reduce_mean(grad_values, axis=0)
      grad_std = tf.math.reduce_std(grad_values, axis=0)
      list_of_snrs.append(tf.abs(grad_mean / (grad_std + 1e-10)))

    snr_mean = tf.reduce_mean(flatten_list_of_vars(list_of_snrs))
    snr_std = tf.math.reduce_std((flatten_list_of_vars(list_of_snrs)))
    with tf.name_scope('SNR/'):
      tf.compat.v2.summary.scalar(name='mean', data=snr_mean, step=step)
      tf.compat.v2.summary.scalar(name='std', data=snr_std, step=step)

  tf.cond(step % freq == 0, true_fn, lambda: None)
