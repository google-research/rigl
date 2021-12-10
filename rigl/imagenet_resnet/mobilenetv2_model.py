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

"""Straightforward MobileNet v2 for inputs of size 224x224."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags
from rigl.imagenet_resnet import resnet_model
from rigl.imagenet_resnet.pruning_layers import sparse_conv2d
from rigl.imagenet_resnet.pruning_layers import sparse_fully_connected
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

FLAGS = flags.FLAGS


def _make_divisible(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def depthwise_conv2d_fixed_padding(inputs,
                                   kernel_size,
                                   stride,
                                   data_format='channels_first',
                                   name=None):
  """Depthwise Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs:  Input tensor, float32 or bfloat16 of size [batch, channels, height,
      width].
    kernel_size: Int designating size of kernel to be used in the convolution.
    stride: Int specifying the stride. If stride >1, the input is downsampled.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor of size [batch, filters, height_out, width_out]

  Raises:
    ValueError: If the data_format provided is not a valid string.
  """
  if stride > 1:
    inputs = resnet_model.fixed_padding(
        inputs, kernel_size, data_format=data_format)
  padding = 'SAME' if stride == 1 else 'VALID'

  if data_format == 'channels_last':
    data_format_channels = 'NHWC'
  elif data_format == 'channels_first':
    data_format_channels = 'NCHW'
  else:
    raise ValueError('Not a valid channel string:', data_format)

  return contrib_layers.separable_conv2d(
      inputs=inputs,
      num_outputs=None,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      data_format=data_format_channels,
      activation_fn=None,
      weights_regularizer=None,
      biases_initializer=None,
      biases_regularizer=None,
      scope=name)


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         pruning_method='baseline',
                         data_format='channels_first',
                         weight_decay=0.,
                         name=None):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs:  Input tensor, float32 or bfloat16 of size [batch, channels, height,
      width].
    filters: Int specifying number of filters for the first two convolutions.
    kernel_size: Int designating size of kernel to be used in the convolution.
    strides: Int specifying the stride. If stride >1, the input is downsampled.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    weight_decay: Weight for the l2 regularization loss.
    name: String that specifies name for model layer.

  Returns:
    The output activation tensor of size [batch, filters, height_out, width_out]

  Raises:
    ValueError: If the data_format provided is not a valid string.
  """
  if strides > 1:
    inputs = resnet_model.fixed_padding(
        inputs, kernel_size, data_format=data_format)
    padding = 'VALID'
  else:
    padding = 'SAME'

  kernel_initializer = tf.variance_scaling_initializer()

  kernel_regularizer = contrib_layers.l2_regularizer(weight_decay)
  return sparse_conv2d(
      x=inputs,
      units=filters,
      activation=None,
      kernel_size=[kernel_size, kernel_size],
      use_bias=False,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_initializer=None,
      biases_regularizer=None,
      sparsity_technique=pruning_method,
      normalizer_fn=None,
      strides=[strides, strides],
      padding=padding,
      data_format=data_format,
      name=name)


def inverted_res_block_(inputs,
                        filters,
                        is_training,
                        stride,
                        width=1.,
                        expansion_factor=6.,
                        block_id=0,
                        pruning_method='baseline',
                        data_format='channels_first',
                        weight_decay=0.,):
  """Standard building block for mobilenetv2 networks.

  Args:
    inputs:  Input tensor, float32 or bfloat16 of size [batch, channels, height,
      width].
    filters: Int specifying number of filters for the first two convolutions.
    is_training: Boolean specifying whether the model is training.
    stride: Int specifying the stride. If stride >1, the input is downsampled.
    width: multiplier for channel dimensions
    expansion_factor: How much to increase the filters before the depthwise
      conv.
    block_id: which block this is
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    data_format: String that specifies either "channels_first" for [batch,
      channels, height,width] or "channels_last" for [batch, height, width,
      channels].
    weight_decay: Weight for the l2 regularization loss.

  Returns:
    The output activation tensor.
  """

  # 1x1 expanded conv, followed by separable_conv_2d followed by
  # contracting 1x1 conv.

  shortcut = inputs

  if data_format == 'channels_first':
    prev_depth = inputs.get_shape().as_list()[1]
  elif data_format == 'channels_last':
    prev_depth = inputs.get_shape().as_list()[3]
  else:
    raise ValueError('Unknown data_format ' + data_format)

  # Expand
  multiplier = expansion_factor if block_id > 0 else 1
  # skip the expansion if this is the first block
  if block_id:
    end_point = 'expand_1x1_%s' % block_id
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=int(multiplier * prev_depth),
        kernel_size=1,
        strides=1,
        pruning_method=pruning_method,
        data_format=data_format,
        weight_decay=weight_decay,
        name=end_point)
    inputs = resnet_model.batch_norm_relu(
        inputs, is_training, relu=True, data_format=data_format)

  end_point = 'depthwise_nxn_%s' % block_id
  # Depthwise
  depthwise_out = depthwise_conv2d_fixed_padding(
      inputs=inputs,
      kernel_size=3,
      stride=stride,
      data_format=data_format,
      name=end_point)

  depthwise_out = resnet_model.batch_norm_relu(
      depthwise_out, is_training, relu=True, data_format=data_format)

  # Contraction
  end_point = 'contraction_1x1_%s' % block_id
  divisible_by = 8
  if block_id == 0:
    divisible_by = 1
  out_filters = _make_divisible(int(width * filters), divisor=divisible_by)

  contraction_out = conv2d_fixed_padding(
      inputs=depthwise_out,
      filters=out_filters,
      kernel_size=1,
      strides=1,
      pruning_method=pruning_method,
      data_format=data_format,
      weight_decay=weight_decay,
      name=end_point)
  contraction_out = resnet_model.batch_norm_relu(
      contraction_out, is_training, relu=False, data_format=data_format)

  output = contraction_out
  if prev_depth == out_filters and stride == 1:
    output += shortcut
  return output


def mobilenet_v2_generator(num_classes=1000,
                           pruning_method='baseline',
                           width=1.,
                           expansion_factor=6.,
                           prune_last_layer=False,
                           data_format='channels_first',
                           weight_decay=0.,
                           name=None):
  """Generator for mobilenet v2 models.

  Args:
    num_classes: Int number of possible classes for image classification.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    width: Float that scales the number of filters in each layer.
    expansion_factor: How much to expand the input filters for the depthwise
      conv.
    prune_last_layer: Whether or not to prune the last layer.
    data_format: String either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    weight_decay: Weight for the l2 regularization loss.
    name: String that specifies name for model layer.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """

  def model(inputs, is_training):
    """Creation of the model graph."""
    with tf.variable_scope(name, 'resnet_model'):
      inputs = resnet_model.fixed_padding(
          inputs, kernel_size=3, data_format=data_format)
      padding = 'VALID'

      kernel_initializer = tf.variance_scaling_initializer()
      kernel_regularizer = contrib_layers.l2_regularizer(weight_decay)

      inputs = tf.layers.conv2d(
          inputs=inputs,
          filters=_make_divisible(32 * width),
          kernel_size=3,
          strides=2,
          padding=padding,
          use_bias=False,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          data_format=data_format,
          name='initial_conv')

      inputs = tf.identity(inputs, 'initial_conv')
      inputs = resnet_model.batch_norm_relu(
          inputs, is_training, data_format=data_format)

      inverted_res_block = functools.partial(
          inverted_res_block_,
          is_training=is_training,
          width=width,
          expansion_factor=expansion_factor,
          pruning_method=pruning_method,
          data_format=data_format,
          weight_decay=weight_decay)

      inputs = inverted_res_block(inputs, filters=16, stride=1, block_id=0)

      inputs = inverted_res_block(inputs, filters=24, stride=2, block_id=1)
      inputs = inverted_res_block(inputs, filters=24, stride=1, block_id=2)

      inputs = inverted_res_block(inputs, filters=32, stride=2, block_id=3)
      inputs = inverted_res_block(inputs, filters=32, stride=1, block_id=4)
      inputs = inverted_res_block(inputs, filters=32, stride=1, block_id=5)

      inputs = inverted_res_block(inputs, filters=64, stride=2, block_id=6)
      inputs = inverted_res_block(inputs, filters=64, stride=1, block_id=7)
      inputs = inverted_res_block(inputs, filters=64, stride=1, block_id=8)
      inputs = inverted_res_block(inputs, filters=64, stride=1, block_id=9)

      inputs = inverted_res_block(inputs, filters=96, stride=1, block_id=10)
      inputs = inverted_res_block(inputs, filters=96, stride=1, block_id=11)
      inputs = inverted_res_block(inputs, filters=96, stride=1, block_id=12)

      inputs = inverted_res_block(inputs, filters=160, stride=2, block_id=13)
      inputs = inverted_res_block(inputs, filters=160, stride=1, block_id=14)
      inputs = inverted_res_block(inputs, filters=160, stride=1, block_id=15)

      inputs = inverted_res_block(inputs, filters=320, stride=1, block_id=16)

      last_block_filters = max(1280, _make_divisible(1280 * width, 8))

      inputs = conv2d_fixed_padding(
          inputs=inputs,
          filters=last_block_filters,
          kernel_size=1,
          strides=1,
          pruning_method=pruning_method,
          data_format=data_format,
          weight_decay=weight_decay,
          name='final_1x1_conv')

      inputs = resnet_model.batch_norm_relu(
          inputs, is_training, data_format=data_format)

      if data_format == 'channels_last':
        pool_size = (inputs.shape[1], inputs.shape[2])
      elif data_format == 'channels_first':
        pool_size = (inputs.shape[2], inputs.shape[3])

      inputs = tf.layers.average_pooling2d(
          inputs=inputs,
          pool_size=pool_size,
          strides=1,
          padding='VALID',
          data_format=data_format,
          name='final_avg_pool')
      inputs = tf.identity(inputs, 'final_avg_pool')
      inputs = tf.reshape(inputs, [-1, last_block_filters])

      kernel_initializer = tf.variance_scaling_initializer()

      kernel_regularizer = contrib_layers.l2_regularizer(weight_decay)
      if prune_last_layer:
        inputs = sparse_fully_connected(
            x=inputs,
            units=num_classes,
            sparsity_technique=pruning_method
            if prune_last_layer else 'baseline',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='final_dense')
      else:
        inputs = tf.layers.dense(
            inputs=inputs,
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='final_dense')

      inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def mobilenet_v2(num_classes,
                 pruning_method='baseline',
                 width=1.,
                 expansion_factor=6.,
                 prune_last_layer=True,
                 data_format='channels_first',
                 weight_decay=0.,):
  """Returns the mobilenet_V2 model for a given size and number of output classes.

  Args:
    num_classes: Int number of possible classes for image classification.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    width: Float multiplier of the number of filters in each layer.
    expansion_factor: How much to increase the number of filters before the
      depthwise conv.
    prune_last_layer: Whether or not to prune the last layer.
    data_format: String specifying either "channels_first" for `[batch,
      channels, height, width]` or "channels_last for `[batch, height, width,
      channels]`.
    weight_decay: Weight for the l2 regularization loss.

  Raises:
    ValueError: If the resnet_depth int is not in the model_params dictionary.
  """
  return mobilenet_v2_generator(
      num_classes, pruning_method, width, expansion_factor, prune_last_layer,
      data_format, weight_decay)
