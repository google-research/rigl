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

"""Tensorflow layers with parameters for implementing pruning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.python.ops import init_ops


def get_model_variables(getter,
                        name,
                        shape=None,
                        dtype=None,
                        initializer=None,
                        regularizer=None,
                        trainable=True,
                        collections=None,
                        caching_device=None,
                        partitioner=None,
                        rename=None,
                        use_resource=None,
                        **_):
  """This ensure variables are retrieved in a consistent way for core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      collections=collections,
      trainable=trainable,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=getter,
      use_resource=use_resource)


def variable_getter(rename=None):
  """Ensures scope is respected and consistently used."""

  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return get_model_variables(getter, *args, **kwargs)

  return layer_variable_getter


def sparse_conv2d(x,
                  units,
                  kernel_size,
                  activation=None,
                  use_bias=False,
                  kernel_initializer=None,
                  kernel_regularizer=None,
                  bias_initializer=None,
                  biases_regularizer=None,
                  sparsity_technique='baseline',
                  normalizer_fn=None,
                  strides=(1, 1),
                  padding='SAME',
                  data_format='channels_last',
                  name=None):
  """Function that constructs conv2d with any desired pruning method.

  Args:
    x: Input, float32 tensor.
    units: Int representing size of output tensor.
    kernel_size: The size of the convolutional window, int of list of ints.
    activation: If None, a linear activation is used.
    use_bias: Boolean specifying whether bias vector should be used.
    kernel_initializer: Initializer for the convolution weights.
    kernel_regularizer: Regularization method for the convolution weights.
    bias_initializer: Initalizer of the bias vector.
    biases_regularizer: Optional regularizer for the bias vector.
    sparsity_technique: Method used to introduce sparsity.
      ['threshold', 'baseline']
    normalizer_fn: function used to transform the output activations.
    strides: stride length of convolution, a single int is expected.
    padding: May be populated as 'VALID' or 'SAME'.
    data_format: Either 'channels_last', 'channels_first'.
    name: String speciying name scope of layer in network.

  Returns:
    Output: activations.

  Raises:
    ValueError: If the rank of the input is not greater than 2.
  """

  if data_format == 'channels_last':
    data_format_channels = 'NHWC'
  elif data_format == 'channels_first':
    data_format_channels = 'NCHW'
  else:
    raise ValueError('Not a valid channel string:', data_format)

  layer_variable_getter = variable_getter({
      'bias': 'biases',
      'kernel': 'weights',
  })
  input_rank = x.get_shape().ndims
  if input_rank != 4:
    raise ValueError('Rank not supported {}'.format(input_rank))

  with tf.variable_scope(
      name, 'Conv', [x], custom_getter=layer_variable_getter) as sc:

    input_shape = x.get_shape().as_list()
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `Convolution` '
                       'should be defined. Found `None`.')

    pruning_methods = ['threshold']

    if sparsity_technique in pruning_methods:
      return layers.masked_conv2d(
          inputs=x,
          num_outputs=units,
          kernel_size=kernel_size[0],
          stride=strides[0],
          padding=padding,
          data_format=data_format_channels,
          rate=1,
          activation_fn=activation,
          weights_initializer=kernel_initializer,
          weights_regularizer=kernel_regularizer,
          normalizer_fn=normalizer_fn,
          normalizer_params=None,
          biases_initializer=bias_initializer,
          biases_regularizer=biases_regularizer,
          outputs_collections=None,
          trainable=True,
          scope=sc)
    elif sparsity_technique == 'baseline':
      return tf.layers.conv2d(
          inputs=x,
          filters=units,
          kernel_size=kernel_size,
          strides=strides,
          padding=padding,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          data_format=data_format,
          name=name)
    else:
      raise ValueError(
          'Unsupported sparsity technique {}'.format(sparsity_technique))


def sparse_fully_connected(x,
                           units,
                           activation=None,
                           use_bias=True,
                           kernel_initializer=None,
                           kernel_regularizer=None,
                           bias_initializer=init_ops.zeros_initializer(),
                           biases_regularizer=None,
                           sparsity_technique='baseline',
                           name=None):
  """Constructs sparse_fully_connected with any desired pruning method.

  Args:
    x: Input, float32 tensor.
    units: Int representing size of output tensor.
    activation: If None, a linear activation is used.
    use_bias: Boolean specifying whether bias vector should be used.
    kernel_initializer: Initializer for the convolution weights.
    kernel_regularizer: Regularization method for the convolution weights.
    bias_initializer: Initalizer of the bias vector.
    biases_regularizer: Optional regularizer for the bias vector.
    sparsity_technique: Method used to introduce sparsity. ['baseline',
      'threshold']
    name: String speciying name scope of layer in network.

  Returns:
    Output: activations.

  Raises:
    ValueError: If the rank of the input is not greater than 2.
  """

  layer_variable_getter = variable_getter({
      'bias': 'biases',
      'kernel': 'weights',
  })

  with tf.variable_scope(
      name, 'Dense', [x], custom_getter=layer_variable_getter) as sc:

    input_shape = x.get_shape().as_list()
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')

    pruning_methods = ['threshold']

    if sparsity_technique in pruning_methods:
      return layers.masked_fully_connected(
          inputs=x,
          num_outputs=units,
          activation_fn=activation,
          weights_initializer=kernel_initializer,
          weights_regularizer=kernel_regularizer,
          biases_initializer=bias_initializer,
          biases_regularizer=biases_regularizer,
          outputs_collections=None,
          trainable=True,
          scope=sc)

    elif sparsity_technique == 'baseline':
      return tf.layers.dense(
          inputs=x,
          units=units,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_initializer=bias_initializer,
          bias_regularizer=biases_regularizer,
          name=name)
    else:
      raise ValueError(
          'Unsupported sparsity technique {}'.format(sparsity_technique))
