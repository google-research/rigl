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

"""This module has networks used in experiments.
"""
from typing import Optional, Tuple  # Non-expensive-to-import types.
import gin

import tensorflow.compat.v2 as tf


@gin.configurable(allowlist=['hidden_sizes', 'use_batch_norm'])
def lenet5(input_shape,
           num_classes,
           activation,
           kernel_regularizer,
           use_batch_norm = False,
           hidden_sizes = (6, 16, 120, 84)):
  """Lenet5 implementation."""
  network = tf.keras.Sequential()
  kwargs = {
      'activation': activation,
      'kernel_regularizer': kernel_regularizer,
  }
  def maybe_add_batchnorm():
    if use_batch_norm:
      network.add(tf.keras.layers.BatchNormalization())
  network.add(tf.keras.layers.Conv2D(
      hidden_sizes[0], 5, input_shape=input_shape, **kwargs))
  network.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
  maybe_add_batchnorm()
  network.add(tf.keras.layers.Conv2D(hidden_sizes[1], 5, **kwargs))
  network.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
  maybe_add_batchnorm()
  network.add(tf.keras.layers.Flatten())
  network.add(tf.keras.layers.Dense(hidden_sizes[2], **kwargs))
  maybe_add_batchnorm()
  network.add(tf.keras.layers.Dense(hidden_sizes[3], **kwargs))
  maybe_add_batchnorm()
  kwargs['activation'] = None
  network.add(tf.keras.layers.Dense(num_classes, **kwargs))
  return network


@gin.configurable(allowlist=['hidden_sizes', 'use_batch_norm'])
def mlp(input_shape,
        num_classes,
        activation,
        kernel_regularizer,
        use_batch_norm = False,
        hidden_sizes = (300, 100)):
  """Lenet5 implementation."""
  network = tf.keras.Sequential()
  kwargs = {
      'activation': activation,
      'kernel_regularizer': kernel_regularizer
  }
  def maybe_add_batchnorm():
    if use_batch_norm:
      network.add(tf.keras.layers.BatchNormalization())
  network.add(tf.keras.layers.Flatten(input_shape=input_shape))
  network.add(tf.keras.layers.Dense(hidden_sizes[0], **kwargs))
  maybe_add_batchnorm()
  network.add(tf.keras.layers.Dense(hidden_sizes[1], **kwargs))
  maybe_add_batchnorm()
  kwargs['activation'] = None
  network.add(tf.keras.layers.Dense(num_classes, **kwargs))
  return network
