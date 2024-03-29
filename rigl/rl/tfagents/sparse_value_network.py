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

"""Sample Keras Value Network.

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params
  Dense -> 1         # Value output
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from rigl.rl.tfagents import sparse_encoding_network
from rigl.rl.tfagents import tf_sparse_utils
import tensorflow as tf

from tf_agents.networks import network


@gin.configurable
class ValueNetwork(network.Network):
  """Feed Forward value network. Reduces to 1 value output per batch item."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               weight_decay=0.0,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='ValueNetwork',
               is_sparse=False,
               sparse_output_layer=False,
               width=1.0):
    """Creates an instance of `ValueNetwork`.

    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent`, if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      weight_decay: L2 weight decay regularization parameter.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.
      is_sparse: Whether the network is sparse.
      sparse_output_layer: Whether the output layer should be sparse. Only
        applied when is_sparse=True.
      width: Scaling factor to apply to the layers.

    Raises:
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.
    """
    super(ValueNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._is_sparse = is_sparse
    self._sparse_output_layer = sparse_output_layer
    self._width = width

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    self._encoder = sparse_encoding_network.EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        weight_decay_params=[weight_decay] * len(fc_layer_params),
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype,
        width=self._width)

    self._postprocessing_layers = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03),
        kernel_regularizer=tf.keras.regularizers.L2(weight_decay))

    if is_sparse:
      layers_to_wrap = [l for l in self._encoder._postprocessing_layers
                        if tf_sparse_utils.is_valid_layer_to_wrap(l)]
      input_dim = input_tensor_spec.shape[0]
      if sparse_output_layer:
        layers_to_wrap.append(self._postprocessing_layers)
        wrapped_layers = tf_sparse_utils.wrap_all_layers(
            layers_to_wrap, input_dim)
        self._postprocessing_layers = wrapped_layers[-1]
        wrapped_layers = wrapped_layers[:-1]
      else:
        wrapped_layers = tf_sparse_utils.wrap_all_layers(
            layers_to_wrap, input_dim)
      # We need to recreate the original layer list after wrapping the layers.
      new_layer_list = []
      i = 0
      for unwrapped_layer in self._encoder._postprocessing_layers:
        if tf_sparse_utils.is_valid_layer_to_wrap(unwrapped_layer):
          new_layer_list.append(wrapped_layers[i])
          i += 1
        else:
          new_layer_list.append(unwrapped_layer)
      self._encoder._postprocessing_layers = new_layer_list

  def call(self, observation, step_type=None, network_state=(), training=False):
    state, network_state = self._encoder(
        observation, step_type=step_type, network_state=network_state,
        training=training)
    value = self._postprocessing_layers(state, training=training)
    return tf.squeeze(value, -1), network_state
