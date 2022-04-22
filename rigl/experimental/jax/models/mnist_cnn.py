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

"""MNIST CNN.

A small CNN for the MNIST dataset, consists of a number of convolutional layers
(determined by length of filters parameter), followed by a fully-connected
layer.
"""
from typing import Callable, Mapping, Optional, Sequence

from absl import logging
import flax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import init
from rigl.experimental.jax.pruning import masked


class MNISTCNN(flax.deprecated.nn.Module):
  """Small MNIST CNN."""

  def apply(self,
            inputs,
            num_classes,
            filter_shape = (5, 5),
            filters = (16, 32),
            dense_size = 64,
            train=True,
            init_fn = flax.deprecated.nn.initializers.kaiming_normal,
            activation_fn = flax.deprecated.nn.relu,
            masks = None,
            masked_layer_indices = None):
    """Applies a convolution to the inputs.

    Args:
      inputs: Input data with dimensions (batch, spatial_dims..., features).
      num_classes: Number of classes in the dataset.
      filter_shape: Shape of the convolutional filters.
      filters: Number of filters in each convolutional layer, and number of conv
        layers (given by length of sequence).
      dense_size: Number of filters in each convolutional layer, and number of
        conv layers (given by length of sequence).
      train: If model is being evaluated in training mode or not.
      init_fn: Initialization function used for convolutional layers.
      activation_fn: Activation function to be used for convolutional layers.
      masks: Masks of the layers in this model, in the same form as
             module params, or None.
      masked_layer_indices: The layer indices of layers in model to be masked.

    Returns:
      A tensor of shape (batch, num_classes), containing the logit output.
    Raises:
      ValueError if the number of pooling layers is too many for the given input
        size.
    """
    # Note: First dim is batch, last dim is channels, other dims are "spatial".
    if not all([(dim >= 2**len(filters)) for dim in inputs.shape[1:-2]]):
      raise ValueError(
          'Input spatial size, {}, does not allow {} pooling layers.'.format(
              str(inputs.shape[1:-2]), len(filters))
          )

    depth = 2 + len(filters)
    masks = masked.generate_model_masks(depth, masks,
                                        masked_layer_indices)

    batch_norm = flax.deprecated.nn.BatchNorm.partial(
        use_running_average=not train, momentum=0.99, epsilon=1e-5)

    for i, filter_num in enumerate(filters):
      if f'MaskedModule_{i}' in masks:
        logging.info('Layer %d is masked in model', i)
        mask = masks[f'MaskedModule_{i}']
        inputs = masked.masked(flax.deprecated.nn.Conv, mask)(
            inputs,
            features=filter_num,
            kernel_size=filter_shape,
            kernel_init=init.sparse_init(
                init_fn(), mask['kernel'] if mask is not None else None))
      else:
        inputs = flax.deprecated.nn.Conv(
            inputs,
            features=filter_num,
            kernel_size=filter_shape,
            kernel_init=init_fn())
      inputs = batch_norm(inputs, name='bn_conv_{}'.format(i))
      inputs = activation_fn(inputs)

      if i < len(filters) - 1:
        inputs = flax.deprecated.nn.max_pool(
            inputs, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    # Global average pool at end of convolutional layers.
    inputs = flax.deprecated.nn.avg_pool(
        inputs, window_shape=inputs.shape[1:-1], padding='VALID')

    # This is effectively a Dense layer, but we cast it as a convolution layer
    # to allow us to easily propagate masks, avoiding b/156135283.
    if f'MaskedModule_{depth - 2}' in masks:
      mask_dense_1 = masks[f'MaskedModule_{depth - 2}']
      inputs = masked.masked(flax.deprecated.nn.Conv, mask_dense_1)(
          inputs,
          features=dense_size,
          kernel_size=inputs.shape[1:-1],
          kernel_init=init.sparse_init(
              init_fn(),
              mask_dense_1['kernel'] if mask_dense_1 is not None else None))
    else:
      inputs = flax.deprecated.nn.Conv(
          inputs,
          features=dense_size,
          kernel_size=inputs.shape[1:-1],
          kernel_init=init_fn())
    inputs = batch_norm(inputs, name='bn_dense_1')
    inputs = activation_fn(inputs)

    inputs = flax.deprecated.nn.Dense(
        inputs,
        features=num_classes,
        kernel_init=flax.deprecated.nn.initializers.xavier_normal())
    inputs = batch_norm(inputs, name='bn_dense_2')
    inputs = jnp.squeeze(inputs)
    return flax.deprecated.nn.log_softmax(inputs)
