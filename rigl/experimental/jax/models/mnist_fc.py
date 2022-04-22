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

"""MNIST Fully-Connected Neural Network.

A fully-connected model for the MNIST dataset, consists of a number of
dense layers (determined by length of features parameter).
"""
import math
from typing import Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
import flax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import init
from rigl.experimental.jax.pruning import masked


def feature_dim_for_param(input_len,
                          param_count,
                          depth,
                          depth_mult = 2.):
  """Calculates feature dimensions for a fixed parameter count and depth.

  This is calculated for the specific case of a fully-connected neural
  network, where each layer consists of l * a**i neurons, where a is a
  multiplier for each layer.

  Assume,
    x is the input size,
    a is the depth multiplier,
    l is the initial layer width,
    d is the depth.

  The total number of parameters, n, is then given by,
  $$n = x*l + l^2 * sum_{i=2}^d a^{2i-3})$$.

  Args:
    input_len: Input size.
    param_count: Number of parameters model should maintain.
    depth: Depth of the model.
    depth_mult: The layer width multiplier w.r.t. depth.

  Returns:
    The feature specification for a fully-connected model, as a tuple of layer
    widths.

  Raises:
    ValueError: If the given number of parameters is too low for the given
    depth and input size.
  """
  # Calculate the initial width for the first layer.
  if depth == 1:
    initial_width = param_count / input_len
  else:
    # l = ((x^2 + 4cn)^{1/2} - x)/(2c) where c = sum_{i=2}^d a^{2i-3}.
    depth_sum = sum(depth_mult**(2 * i - 3) for i in range(2, depth + 1))
    initial_width = (math.sqrt(input_len**2 + 4 * depth_sum * param_count) -
                     input_len) / (2 * depth_sum)

  if initial_width < 1:
    raise ValueError(
        'Expected parameter count too low for given depth and input size.')

  return tuple(int(int(initial_width) * depth_mult**i) for i in range(depth))


class MNISTFC(flax.deprecated.nn.Module):
  """MNIST Fully-Connected Neural Network."""

  def apply(self,
            inputs,
            num_classes,
            features = (32, 32),
            train=True,
            init_fn = flax.deprecated.nn.initializers.kaiming_normal,
            activation_fn = flax.deprecated.nn.relu,
            masks = None,
            masked_layer_indices = None,
            dropout_rate = 0.):
    """Applies fully-connected neural network to the inputs.

    Args:
      inputs: Input data with dimensions (batch, features), if features has more
        than one dimension, it is flattened.
      num_classes: Number of classes in the dataset.
      features: Number of neurons in each layer, and number of layers (given by
        length of sequence) + one layer for softmax.
      train: If model is being evaluated in training mode or not.
      init_fn: Initialization function used for dense layers.
      activation_fn: Activation function to be used for dense layers.
      masks: Masks of the layers in this model, in the same form as module
        params, or None.
      masked_layer_indices: The layer indices of layers in model to be masked.
      dropout_rate: Dropout rate, if 0 then dropout is not used (default).

    Returns:
      A tensor of shape (batch, num_classes), containing the logit output.
    """
    batch_norm = flax.deprecated.nn.BatchNorm.partial(
        use_running_average=not train, momentum=0.99, epsilon=1e-5)

    depth = 1 + len(features)
    masks = masked.generate_model_masks(depth, masks,
                                        masked_layer_indices)

    # If inputs are in image dimensions, flatten image.
    inputs = inputs.reshape(inputs.shape[0], -1)

    for i, feature_num in enumerate(features):
      if f'MaskedModule_{i}' in masks:
        logging.info('Layer %d is masked in model', i)
        mask = masks[f'MaskedModule_{i}']
        inputs = masked.masked(flax.deprecated.nn.Dense, mask)(
            inputs,
            features=feature_num,
            kernel_init=init.sparse_init(
                init_fn(), mask['kernel'] if mask is not None else None))
      else:
        inputs = flax.deprecated.nn.Dense(
            inputs, features=feature_num, kernel_init=init_fn())
      inputs = batch_norm(inputs, name=f'bn_conv_{i}')
      inputs = activation_fn(inputs)
      if dropout_rate > 0.0:
        inputs = flax.deprecated.nn.dropout(
            inputs, dropout_rate, deterministic=not train)

    inputs = flax.deprecated.nn.Dense(
        inputs,
        features=num_classes,
        kernel_init=flax.deprecated.nn.initializers.xavier_normal())

    return flax.deprecated.nn.log_softmax(inputs)
