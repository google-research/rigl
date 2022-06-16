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

r"""Sparse Discrete Sequential Actor Network for PPO."""

import functools
import sys
import numpy as np
from rigl.rl.tfagents import tf_sparse_utils

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.networks import sequential
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec


def tanh_and_scale_to_spec(inputs, spec):
  """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
  mean = (spec.maximum + spec.minimum) / 2.0
  magnitude = spec.maximum - spec.minimum

  return mean + (magnitude * tf.tanh(inputs)) / 2.0


class PPODiscreteActorNetwork():
  """Contains the actor network structure."""

  def __init__(self, seed_stream_class=tfp.util.SeedStream,
               is_sparse=False,
               sparse_output_layer=False,
               weight_decay=0,
               width=1.0):
    if is_sparse:
      raise ValueError('This functionality is not enabled. wrap_all_layers,'
                       'functionality needs to be implemented')
    self.seed_stream_class = seed_stream_class
    # Sparse params.
    self._is_sparse = is_sparse
    self._sparse_output_layer = sparse_output_layer
    self._width = width
    self._weight_decay = weight_decay

  def create_sequential_actor_net(self,
                                  fc_layer_units,
                                  action_tensor_spec,
                                  logits_init_output_factor=0.1,
                                  seed=None):
    """Helper method for creating the actor network."""

    self._seed_stream = self.seed_stream_class(
        seed=seed, salt='tf_agents_sequential_layers')
    # action_tensor_spec is a BoundedArraySpec which is an array with defined
    # bounds. Maximum and minimum are arrays with the same shape as the
    # main array.
    unique_num_actions = np.unique(action_tensor_spec.maximum -
                                   action_tensor_spec.minimum + 1)
    if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
      raise ValueError('Bounds on discrete actions must be the same for all '
                       'dimensions and have at least 1 action. Projection '
                       'Network requires num_actions to be equal across '
                       'action dimensions. Implement a more general '
                       'categorical projection if you need more flexibility.')

    output_shape = action_tensor_spec.shape.concatenate(
        [int(unique_num_actions)])

    def _get_seed():
      seed = self._seed_stream()
      if seed is not None:
        seed = seed % sys.maxsize
      return seed

    def create_dist(logits):
      input_param_spec = {
          'logits': tensor_spec.TensorSpec(
              shape=(1,) + output_shape, dtype=tf.float32)
      }
      dist_spec = distribution_spec.DistributionSpec(
          tfp.distributions.Categorical,
          input_param_spec,
          sample_spec=action_tensor_spec,
          dtype=action_tensor_spec.dtype)
      logits = tf.reshape(logits, [-1] + output_shape.as_list())
      return dist_spec.build_distribution(logits=logits)

    def dense_layer(num_units):
      dense = functools.partial(
          tf.keras.layers.Dense,
          activation=tf.nn.tanh,
          kernel_initializer=tf.keras.initializers.Orthogonal(seed=_get_seed()),
          kernel_regularizer=tf.keras.regularizers.L2(self._weight_decay))
      layer = dense(tf_sparse_utils.scale_width(num_units, self._width))
      if self._is_sparse:
        return tf_sparse_utils.wrap_layer(layer)
      else:
        return layer

    output_layer = tf.keras.layers.Dense(
        output_shape.num_elements(),
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=logits_init_output_factor, seed=_get_seed()),
        kernel_regularizer=tf.keras.regularizers.L2(self._weight_decay),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits',
        dtype=tf.float32)
    if self._is_sparse and self._sparse_output_layer:
      output_layer = tf_sparse_utils.wrap_layer(output_layer)

    return sequential.Sequential(
        [dense_layer(num_units) for num_units in fc_layer_units] +
        [output_layer] +
        [tf.keras.layers.Lambda(create_dist)])
