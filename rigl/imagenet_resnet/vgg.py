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

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_net(inputs,scope='vgg_19')
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

from rigl.imagenet_resnet import resnet_model
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers

network_cfg = {
    'vgg_a': [1, 1, 2, 2, 2],
    'vgg_16': [2, 2, 3, 3, 3],
    'vgg_19': [2, 2, 4, 4, 4],
}


def vgg_net(inputs,
            num_classes=1000,
            spatial_squeeze=True,
            name='vgg_a',
            global_pool=True,
            pruning_method='baseline',
            init_method='baseline',
            data_format='channels_last',
            width=1.,
            prune_last_layer=True,
            end_sparsity=0.,
            weight_decay=0.):
  """Oxford Net VGG.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    name: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    init_method: ('baseline', 'sparse', 'random_zeros') Whether to use standard
      initialization or initialization that takes into the existing sparsity of
      the layer. 'sparse' only makes sense when combined with
      pruning_method == 'scratch'. 'random_zeros' set random weights to zero
      using end_sparsoty parameter and used with 'baseline' method.
    data_format: String specifying either "channels_first" for `[batch,
      channels, height, width]` or "channels_last for `[batch, height, width,
      channels]`.
    width: Float multiplier of the number of filters in each layer.
    prune_last_layer: Whether or not to prune the last layer.
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations. For
      backwards compatibility, some Tensors appear multiple times in the dict.
  """
  net_cfg = network_cfg[name]
  sparse_conv2d = functools.partial(
      resnet_model.conv2d_fixed_padding,
      pruning_method=pruning_method,
      init_method=init_method,
      data_format=data_format,
      init_scale=2.0,  # Heinit
      end_sparsity=end_sparsity,
      weight_decay=weight_decay)

  def new_sparse_conv2d(*args, **kwargs):
    kwargs['name'] = kwargs['scope']
    del kwargs['scope']
    activation_fn = 'relu'
    if 'activation_fn' in kwargs:
      activation_fn = kwargs['activation_fn']
      del kwargs['activation_fn']
    out = sparse_conv2d(*args, **kwargs)
    if activation_fn == 'relu':
      out = tf.nn.relu(out)
    return out

  with tf.variable_scope(name, name, values=[inputs]):
    net = layers.repeat(
        inputs,
        net_cfg[0],
        new_sparse_conv2d,
        int(64 * width),
        3,
        strides=1,
        scope='conv1')
    net = layers.max_pool2d(net, [2, 2], scope='pool1')
    net = layers.repeat(
        net,
        net_cfg[1],
        new_sparse_conv2d,
        int(128 * width),
        3,
        strides=1,
        scope='conv2')
    net = layers.max_pool2d(net, [2, 2], scope='pool2')
    net = layers.repeat(
        net,
        net_cfg[2],
        new_sparse_conv2d,
        int(256 * width),
        3,
        strides=1,
        scope='conv3')
    net = layers.max_pool2d(net, [2, 2], scope='pool3')
    net = layers.repeat(
        net,
        net_cfg[3],
        new_sparse_conv2d,
        int(512 * width),
        3,
        strides=1,
        scope='conv4')
    net = layers.max_pool2d(net, [2, 2], scope='pool4')
    net = layers.repeat(
        net,
        net_cfg[4],
        new_sparse_conv2d,
        int(512 * width),
        3,
        strides=1,
        scope='conv5')

    # # Use conv2d instead of fully_connected layers.
    # net = new_sparse_conv2d(net, 512, [7, 7], strides=1, scope='fc6')
    # # net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
    # #                      scope='dropout6')
    # net = new_sparse_conv2d(net, 512, [1, 1], strides=1, scope='fc7')
    if global_pool:
      net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
    if num_classes:
      # net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
      #                      scope='dropout7')
      if prune_last_layer:
        net = new_sparse_conv2d(
            net, num_classes, 1, activation_fn=None, strides=1, scope='fc8')
      else:
        net = layers.conv2d(
            net, num_classes, [1, 1], activation_fn=None, scope='fc8')
    if spatial_squeeze:
      net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    return net


def vgg(vgg_type,
        num_classes,
        pruning_method='baseline',
        init_method='baseline',
        width=1.,
        prune_last_layer=True,
        data_format='channels_last',
        end_sparsity=0.,
        weight_decay=0.):
  """Returns the ResNet model for a given size and number of output classes.

  Args:
    vgg_type: Int number of blocks in the architecture.
    num_classes: Int number of possible classes for image classification.
    pruning_method: String that specifies the pruning method used to identify
      which weights to remove.
    init_method: ('baseline', 'sparse', 'random_zeros') Whether to use standard
      initialization or initialization that takes into the existing sparsity of
      the layer. 'sparse' only makes sense when combined with pruning_method ==
      'scratch'. 'random_zeros' set random weights to zero using end_sparsoty
      parameter and used with 'baseline' method.
    width: Float multiplier of the number of filters in each layer.
    prune_last_layer: Whether or not to prune the last layer.
    data_format: String specifying either "channels_first" for `[batch,
      channels, height, width]` or "channels_last for `[batch, height, width,
      channels]`.
    end_sparsity: Desired sparsity at the end of training. Necessary to
      initialize an already sparse network.
    weight_decay: Weight for the l2 regularization loss.

  Raises:
    ValueError: If the resnet_depth int is not in the model_params dictionary.
  """

  def model_fn(inputs, is_training):
    del is_training
    return vgg_net(
        inputs,
        num_classes,
        name=vgg_type,
        pruning_method=pruning_method,
        init_method=init_method,
        data_format=data_format,
        width=width,
        prune_last_layer=prune_last_layer,
        end_sparsity=end_sparsity,
        weight_decay=weight_decay)

  return model_fn
