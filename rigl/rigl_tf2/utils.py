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

"""Utilities for training.
"""
import functools
from typing import Optional, Tuple

from absl import flags
from absl import logging
import gin
from rigl.rigl_tf2 import init_utils
from rigl.rigl_tf2 import networks
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

FLAGS = flags.FLAGS
PRUNING_WRAPPER = pruning_wrapper.PruneLowMagnitude
PRUNED_LAYER_TYPES = (tf.keras.layers.Conv2D, tf.keras.layers.Dense)


@gin.configurable('data')
def get_dataset():
  """Loads the dataset."""
  # the data, shuffled and split between train and test sets.
  datasets, info = tfds.load('mnist', with_info=True)
  ds_train, ds_test = datasets['train'].cache(), datasets['test'].cache()

  preprocess_fn = lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label'])
  ds_train = ds_train.map(preprocess_fn)
  ds_test = tfds.load('mnist', split='test').cache()
  ds_test = ds_test.map(preprocess_fn)
  return ds_train, ds_test, info


@gin.configurable('pruning')
def get_pruning_params(mode='prune',
                       initial_sparsity=0.0,
                       final_sparsity=0.8,
                       begin_step=2000,
                       end_step=4000,
                       frequency=200):
  """Gets pruning hyper-parameters."""
  p_params = {}
  if mode == 'prune':
    p_params['pruning_schedule'] = pruning_schedule.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step,
        frequency=frequency)
  elif mode == 'constant':
    p_params['pruning_schedule'] = pruning_schedule.ConstantSparsity(
        target_sparsity=final_sparsity, begin_step=begin_step)
  else:
    raise ValueError('Mode: %s, is not valid' % mode)
  return p_params


# Forked from tensorflow_model_optimization/python/core/sparsity/keras/prune.py
def maybe_prune_layer(layer, params, filter_fn):
  if filter_fn(layer):
    return PRUNING_WRAPPER(layer, **params)
  return layer


@gin.configurable('network')
def get_network(
    pruning_params,
    input_shape,
    num_classes,
    activation = 'relu',
    network_name = 'lenet5',
    mask_init_path = None,
    shuffle_mask = False,
    weight_init_path = None,
    weight_init_method = None,
    weight_decay = 0.,
    noise_stddev = 0.,
    pruned_layer_types = PRUNED_LAYER_TYPES):
  """Creates the network."""
  kernel_regularizer = (
      tf.keras.regularizers.l2(weight_decay) if (weight_decay > 0) else None)
  # (1) Create keras model.
  model = getattr(networks, network_name)(
      input_shape, num_classes, activation=activation,
      kernel_regularizer=kernel_regularizer)
  model.summary(print_fn=logging.info)
  # (2) Adding wrappers. i.e. sparsify if conv or dense.
  filter_fn = lambda layer: isinstance(layer, pruned_layer_types)
  clone_fn = functools.partial(maybe_prune_layer,
                               params=pruning_params,
                               filter_fn=filter_fn)
  model = tf.keras.models.clone_model(model, clone_function=clone_fn)

  # (3) Update parameters of the model as necessary.
  if mask_init_path:
    logging.info('Loading masks from: %s', mask_init_path)
    mask_init_model = tf.keras.models.clone_model(model)
    ckpt = tf.train.Checkpoint(model=mask_init_model)
    ckpt.restore(mask_init_path)
    for l_source, l_target in zip(mask_init_model.layers, model.layers):
      if isinstance(l_source, PRUNING_WRAPPER):
        # l.pruning_vars[0][1] is the mask.
        mask = l_target.pruning_vars[0][1]
        n_active = tf.reduce_sum(mask)
        n_dense = tf.cast(tf.size(mask), dtype=n_active.dtype)
        logging.info('Before: %s, %.2f', l_target.name,
                     (n_active / n_dense).numpy())
        loaded_mask = l_source.pruning_vars[0][1]
        if shuffle_mask:
          # tf shuffle shuffles along the first dim, so we need to flatten.
          loaded_mask = tf.reshape(
              tf.random.shuffle(tf.reshape(loaded_mask, -1)), loaded_mask.shape)
        mask.assign(loaded_mask)
        n_active = tf.reduce_sum(mask)
        n_dense = tf.cast(tf.size(mask), dtype=n_active.dtype)
        logging.info('After: %s, %.2f', l_target.name,
                     (n_active / n_dense).numpy())
    del mask_init_model
  if weight_init_path:
    logging.info('Loading weights from: %s', weight_init_path)
    weight_init_model = tf.keras.models.clone_model(model)
    ckpt = tf.train.Checkpoint(model=weight_init_model)
    ckpt.restore(weight_init_path)
    for l_source, l_target in zip(weight_init_model.layers, model.layers):
      for var_source, var_target in zip(l_source.trainable_variables,
                                        l_target.trainable_variables):
        var_target.assign(var_source)
        logging.info('Weight %s loaded from ckpt.', var_target.name)
    del weight_init_model
  elif weight_init_method == 'unit_scaled':
    logging.info('Using unit_scaled initialization.')
    for layer in model.layers:
      if isinstance(layer, PRUNING_WRAPPER):
        # TODO following the outcome of b/148083099, update following.
        # Add the weight, mask and the valid dimensions.
        weight = layer.weights[0]
        mask = layer.weights[2]
        new_init = init_utils.unit_scaled_init(mask)
        weight.assign(new_init)
        logging.info('Weight %s updated init.', weight.name)
  elif weight_init_method == 'layer_scaled':
    logging.info('Using layer_scaled initialization.')
    for layer in model.layers:
      if isinstance(layer, PRUNING_WRAPPER):
        # TODO following the outcome of b/148083099, update following.
        # Add the weight, mask and the valid dimensions.
        weight = layer.weights[0]
        mask = layer.weights[2]
        new_init = init_utils.layer_scaled_init(mask)
        weight.assign(new_init)
        logging.info('Weight %s updated init.', weight.name)
  if noise_stddev > 0.:
    logging.info('Adding noise to the initial point')
    for layer in model.layers:
      for var in layer.trainable_variables:
        noise = tf.random.normal(var.shape, mean=0, stddev=noise_stddev)
        var.assign_add(noise)
  # Do this call to mask the weights with existing masks if it is not done
  # already. This is needed for example when we use initial parameters to cal-
  # culate distance.
  model(tf.expand_dims(tf.ones(input_shape), 0))
  return model


@gin.configurable('optimizer', denylist=['total_steps'])
def get_optimizer(total_steps,
                  name = 'adam',
                  learning_rate = 0.001,
                  clipnorm = None,
                  clipvalue = None,
                  momentum = None):
  """Creates the optimizer according to the arguments."""
  name = name.lower()
  # We use cosine decay.
  lr_decayed_fn = tf.keras.experimental.CosineDecay(learning_rate, total_steps)
  kwargs = {}
  if clipnorm:
    # Not correct implementation, see http://b/152868229 .
    kwargs['clipnorm'] = clipnorm
  if clipvalue:
    kwargs['clipvalue'] = clipvalue
  if name == 'adam':
    return tf.keras.optimizers.Adam(lr_decayed_fn, **kwargs)
  if name == 'momentum':
    return tf.keras.optimizers.SGD(lr_decayed_fn, momentum=momentum, **kwargs)
  if name == 'sgd':
    return tf.keras.optimizers.SGD(lr_decayed_fn, **kwargs)
  if name == 'rmsprop':
    return tf.keras.optimizers.RMSprop(
        lr_decayed_fn, momentum=momentum, **kwargs)
  raise NotImplementedError(f'Optimizers {name} not implemented.')
