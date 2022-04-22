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

"""CIFAR10 CNN.

A small CNN for the CIFAR10 dataset, consists of a number of convolutional
layers (determined by length of filters parameter), followed by a
fully-connected layer.
"""
from typing import Callable, Mapping, Optional, Sequence

from absl import logging
import flax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import init
from rigl.experimental.jax.pruning import masked


class CIFAR10CNN(flax.deprecated.nn.Module):
  """Small CIFAR10 CNN."""

  def apply(self,
            inputs,
            num_classes,
            filter_shape = (3, 3),
            filters = (32, 32, 64, 64, 128, 128),
            init_fn=flax.deprecated.nn.initializers.kaiming_normal,
            train=True,
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
      init_fn: Initialization function used for convolutional layers.
      train: If model is being evaluated in training mode or not.
      activation_fn: Activation function to be used for convolutional layers.
      masks: Masks of the layers in this model, in the same form as
         module params, or None.
      masked_layer_indices: The layer indices of layers in model to be masked.

    Returns:
      A tensor of shape (batch, num_classes), containing the logit output.

    Raises:
      ValueError if the number of pooling layers is too many for the given input
        size, or if the provided mask is not of the correct depth for the model.
    """
    # Note: First dim is batch, last dim is channels, other dims are "spatial".
    if not all([(dim >= 2**(len(filters)//2)) for dim in inputs.shape[1:-2]]):
      raise ValueError(
          'Input spatial size, {}, does not allow {} pooling layers.'.format(
              str(inputs.shape[1:-2]), len(filters))
          )

    depth = 1 + len(filters)
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

      if i % 2 == 1:
        inputs = flax.deprecated.nn.max_pool(
            inputs, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    # Global average pooling if we have spatial dimensions left.
    inputs = flax.deprecated.nn.avg_pool(
        inputs, window_shape=(inputs.shape[1:-1]), padding='VALID')
    inputs = inputs.reshape((inputs.shape[0], -1))

    # This is effectively a Dense layer, but we cast it as a convolution layer
    # to allow us to easily propagate masks, avoiding b/156135283.
    inputs = flax.deprecated.nn.Conv(
        inputs,
        features=num_classes,
        kernel_size=inputs.shape[1:-1],
        kernel_init=flax.deprecated.nn.initializers.xavier_normal())
    inputs = batch_norm(inputs, name='bn_dense_1')
    inputs = jnp.squeeze(inputs)
    return flax.deprecated.nn.log_softmax(inputs)
