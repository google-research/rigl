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

"""Implements initializations for sparse layers."""
import math
import gin
import tensorflow as tf


@gin.configurable(denylist=['mask'])
def unit_scaled_init(mask, method='fanavg_uniform', scale=1.0):
  """Scales the variance of each unit with correct fan_in."""
  mode, distribution = method.strip().split('_')
  # Lets calculate all fan_ins.
  if len(mask.shape) == 4:
    mask_reduced2d = tf.reduce_sum(mask, axis=[0, 1])
  elif len(mask.shape) == 2:
    mask_reduced2d = mask
  else:
    raise ValueError(f'mask.shape: {mask.shape} must be 4 or 2 dimensional.')
  fan_ins = tf.reduce_sum(mask_reduced2d, axis=-2)
  fan_outs = tf.reduce_sum(mask_reduced2d, axis=-1)
  non_zero_indices = tf.where(mask)  # shape=(NZ, N_dim)
  # Lets sample each row with the correct fan_in.
  new_vals = []
  # Following iterates over each output channel.
  for index in non_zero_indices:
    # Get fan_in and out of neurons that the non_zero connection connects.
    fan_in = fan_ins[index[-1]]
    fan_out = fan_outs[index[-2]]
    # Following code is modified from `tensorflow/python/ops/init_ops_v2.py`.
    if mode == 'fanin':
      current_scale = scale / max(1., fan_in)
    elif mode == 'fanout':
      current_scale = scale / max(1., fan_out)
    elif mode == 'fanavg':
      current_scale = scale / max(1., (fan_in + fan_out) / 2.)
    else:
      raise ValueError(f'mode: {mode} must can be fanin, fanout, fanavg.')
    if distribution == 'normal':
      stddev = math.sqrt(current_scale)
      new_val = tf.random.normal((1,), 0.0, stddev, mask.dtype)
    elif distribution == 'uniform':
      limit = math.sqrt(3.0 * current_scale)
      new_val = tf.random.uniform((1,), -limit, limit, mask.dtype)
    else:
      raise ValueError(f'mode: {mode} must can be fanin, fanout, fanavg.')
    new_vals.append(new_val)
  new_vals = tf.concat(new_vals, axis=-1)
  new_weights = tf.scatter_nd(
      indices=non_zero_indices,
      updates=new_vals,
      shape=mask.shape)
  return new_weights


@gin.configurable(denylist=['mask'])
def layer_scaled_init(mask, method='fanavg_uniform', scale=1.0):
  """Scales the variance of each unit with correct fan_in."""
  mode, distribution = method.strip().split('_')
  init_factory = tf.keras.initializers.VarianceScaling(
      mode=mode.replace('fan', 'fan_'), scale=scale, distribution=distribution)
  dense_init = init_factory(shape=mask.shape, dtype=mask.dtype)
  fraction_nnz = tf.reduce_sum(mask) / tf.size(mask, out_type=mask.dtype)
  new_weights = dense_init / tf.math.sqrt(fraction_nnz)
  return new_weights


def unit_scaled_init_tf1(mask,
                         method='fanavg_uniform',
                         scale=1.0,
                         dtype=tf.float32):
  """Scales the variance of each unit with correct fan_in."""
  mode, distribution = method.strip().split('_')
  # Lets calculate all fan_ins.
  if len(mask.shape) == 4:
    mask_reduced2d = tf.reduce_sum(mask, axis=[0, 1])
  elif len(mask.shape) == 2:
    mask_reduced2d = mask
  else:
    raise ValueError(f'mask.shape: {mask.shape} must be 4 or 2 dimensional.')
  fan_ins = tf.reduce_sum(mask_reduced2d, axis=-2)
  fan_outs = tf.reduce_sum(mask_reduced2d, axis=-1)
  non_zero_indices = tf.where(mask)  # shape=(NZ, N_dim)

  # Lets sample each row with the correct fan_in.
  def new_val_fn(index):
    # Get fan_in and out of neurons that the non_zero connection connects.
    fan_in = fan_ins[index[-1]]
    fan_out = fan_outs[index[-2]]
    # Following code is modified from `tensorflow/python/ops/init_ops_v2.py`.
    if mode == 'fanin':
      current_scale = scale / tf.math.maximum(1., fan_in)
    elif mode == 'fanout':
      current_scale = scale / tf.math.maximum(1., fan_out)
    elif mode == 'fanavg':
      current_scale = scale / tf.math.maximum(1., (fan_in + fan_out) / 2.)
    else:
      raise ValueError(f'mode: {mode} must can be fanin, fanout, fanavg.')
    if distribution == 'normal':
      stddev = tf.math.sqrt(current_scale)
      new_val = tf.random.normal((1,), 0.0, stddev, dtype)
    elif distribution == 'uniform':
      limit = tf.math.sqrt(3.0 * current_scale)
      new_val = tf.random.uniform((1,), -limit, limit, dtype)
    else:
      raise ValueError(f'mode: {mode} must can be fanin, fanout, fanavg.')
    return new_val

  # Following iterates over each output channel.
  new_vals = tf.squeeze(tf.map_fn(new_val_fn, non_zero_indices, dtype=dtype))
  new_weights = tf.scatter_nd(
      indices=non_zero_indices, updates=new_vals, shape=mask.shape)
  return new_weights
