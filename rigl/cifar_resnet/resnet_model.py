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

r"""Model implementation of wide resnet model.

Implements masking layer if pruning method is selected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rigl.imagenet_resnet.pruning_layers import sparse_conv2d
from rigl.imagenet_resnet.pruning_layers import sparse_fully_connected
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
_BN_EPS = 1e-5
_BN_MOMENTUM = 0.9


class WideResNetModel(object):
  """Implements WideResNet model."""

  def __init__(self,
               is_training,
               regularizer=None,
               data_format='channels_last',
               pruning_method='baseline',
               droprate=0.3,
               prune_first_layer=True,
               prune_last_layer=True):
    """WideResnet as described in https://arxiv.org/pdf/1605.07146.pdf.

    Args:
      is_training: Boolean, True during model training,
        false for evaluation/inference.
      regularizer: A regularization function (mapping variables to
        regularization losses), or None.
      data_format: A string that indicates whether the channels are the second
        or last index in the matrix. 'channels_first' or 'channels_last'.
      pruning_method: str, 'threshold' or 'baseline'.
      droprate: float, dropout rate to apply activations.
      prune_first_layer: bool, if True first layer is pruned.
      prune_last_layer: bool, if True last layer is pruned.
    """
    self._training = is_training
    self._regularizer = regularizer
    self._data_format = data_format
    self._pruning_method = pruning_method
    self._droprate = droprate
    self._prune_first_layer = prune_first_layer
    self._prune_last_layer = prune_last_layer
    if data_format == 'channels_last':
      self._channel_axis = -1
    elif data_format == 'channels_first':
      self._channel_axis = 1

  def build(self, inputs, depth, width, num_classes, name=None):
    """Model architecture to train the model.

    The configuration of the resnet blocks requires that depth should be
    6n+4 where n is the number of resnet blocks desired.

    Args:
      inputs: A 4D float tensor containing the model inputs.
      depth: Number of convolutional layers in the network.
      width: Size of the convolutional filters in the residual blocks.
      num_classes: Positive integer number of possible classes.
      name: Optional string, the name of the resulting op in the TF graph.

    Returns:
      A 2D float logits tensor of shape (batch_size, num_classes).
    Raises:
      ValueError: if depth is not the minimum amount required to build the
        model.
    """

    if (depth - 4) % 6 != 0:
      raise ValueError('Depth of ResNet specified not sufficient.')

    resnet_blocks = (depth - 4) // 6
    with tf.variable_scope(name, 'resnet_model'):

      first_layer_technique = self._pruning_method
      if not self._prune_first_layer:
        first_layer_technique = 'baseline'
      net = self._conv(
          inputs,
          'conv_1',
          output_size=16,
          sparsity_technique=first_layer_technique)
      net = self._residual_block(
          net, 'conv_2', 16 * width, subsample=False, blocks=resnet_blocks)

      net = self._residual_block(
          net, 'conv_3', 32 * width, subsample=True, blocks=resnet_blocks)
      net = self._residual_block(
          net, 'conv_4', 64 * width, subsample=True, blocks=resnet_blocks)

      # Put the final BN, relu before the max pooling.
      with tf.name_scope('Pooling'):
        net = self._batch_norm(net)
        net = tf.nn.relu(net)
        net = tf.layers.average_pooling2d(
            net, pool_size=8, strides=1, data_format=self._data_format)

      net = contrib_layers.flatten(net)
      last_layer_technique = self._pruning_method
      if not self._prune_last_layer:
        last_layer_technique = 'baseline'
      net = self._dense(
          net, num_classes, 'logits', sparsity_technique=last_layer_technique)
    return net

  def _batch_norm(self, net, name=None):
    """Adds batchnorm to the model.

    Input gradients cannot be computed with fused batch norm; causes recursive
    loop of tf.gradient call. If regularizer is specified, fused batchnorm must
    be set to False (default setting).

    Args:
      net: Pre-batch norm tensor activations.
      name: Specified name for batch normalization layer.

    Returns:
      batch norm layer: Activations from the batch normalization layer.
    """
    return tf.layers.batch_normalization(
        inputs=net,
        fused=False,
        training=self._training,
        axis=self._channel_axis,
        momentum=_BN_MOMENTUM,
        epsilon=_BN_EPS,
        name=name)

  def _dense(self, net, num_units, name=None, sparsity_technique='baseline'):
    return sparse_fully_connected(
        x=net,
        units=num_units,
        sparsity_technique=sparsity_technique,
        kernel_regularizer=self._regularizer,
        name=name)

  def _conv(self,
            net,
            name,
            output_size,
            strides=(1, 1),
            padding='SAME',
            sparsity_technique='baseline'):
    """returns conv layer."""
    return sparse_conv2d(
        x=net,
        units=output_size,
        activation=None,
        kernel_size=[3, 3],
        use_bias=False,
        kernel_initializer=None,
        kernel_regularizer=self._regularizer,
        bias_initializer=None,
        biases_regularizer=None,
        sparsity_technique=sparsity_technique,
        normalizer_fn=None,
        strides=strides,
        padding=padding,
        data_format=self._data_format,
        name=name)

  def _residual_block(self, net, name, output_size, subsample, blocks):
    """Adds a residual block to the model."""
    with tf.name_scope(name):
      for n in range(blocks):
        with tf.name_scope('res_%d' % n):
          # when subsample is true + first block a larger stride is used.
          if subsample and n == 0:
            strides = [2, 2]
          else:
            strides = [1, 1]

          # Create the skip connection
          skip = net
          end_point = 'skip_%s' % name
          net = self._batch_norm(net)
          net = tf.nn.relu(net)
          if net.get_shape()[3].value != output_size:
            skip = sparse_conv2d(
                x=net,
                units=output_size,
                activation=None,
                kernel_size=[1, 1],
                use_bias=False,
                kernel_initializer=None,
                kernel_regularizer=self._regularizer,
                bias_initializer=None,
                biases_regularizer=None,
                sparsity_technique=self._pruning_method,
                normalizer_fn=None,
                strides=strides,
                padding='VALID',
                data_format=self._data_format,
                name=end_point)

          # Create residual
          net = self._conv(
              net,
              '%s_%d_1' % (name, n),
              output_size,
              strides,
              sparsity_technique=self._pruning_method)
          net = self._batch_norm(net)
          net = tf.nn.relu(net)
          net = tf.keras.layers.Dropout(self._droprate)(net, self._training)
          net = self._conv(
              net,
              '%s_%d_2' % (name, n),
              output_size,
              sparsity_technique=self._pruning_method)

          # Combine the residual and the skip connection
          net += skip
    return net
