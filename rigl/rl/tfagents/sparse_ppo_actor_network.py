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

r"""Sequential Actor Network for PPO."""
import sys

import numpy as np
from rigl.rl.tfagents import tf_sparse_utils
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.keras_layers import bias_layer

from tf_agents.networks import nest_map
from tf_agents.networks import sequential


def tanh_and_scale_to_spec(inputs, spec):
  """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
  means = (spec.maximum + spec.minimum) / 2.0
  magnitudes = (spec.maximum - spec.minimum) / 2.0

  return means + magnitudes * tf.tanh(inputs)


class PPOActorNetwork():
  """Contains the actor network structure."""

  def __init__(self,
               seed_stream_class=tfp.util.SeedStream,
               is_sparse=False,
               sparse_output_layer=False,
               weight_decay=0.0,
               width=1.0):
    self.seed_stream_class = seed_stream_class
    self._is_sparse = is_sparse
    self._sparse_output_layer = sparse_output_layer
    self._weight_decay = weight_decay
    self._width = width

  def create_sequential_actor_net(self,
                                  fc_layer_units,
                                  action_tensor_spec,
                                  input_dim,
                                  seed=None):
    """Helper method for creating the actor network."""
    self._seed_stream = self.seed_stream_class(
        seed=seed, salt='tf_agents_sequential_layers')

    def _get_seed():
      seed = self._seed_stream()
      if seed is not None:
        seed = seed % sys.maxsize
      return seed

    def create_dist(loc_and_scale):
      loc = loc_and_scale['loc']
      loc = tanh_and_scale_to_spec(loc, action_tensor_spec)

      scale = loc_and_scale['scale']
      scale = tf.math.softplus(scale)

      return tfp.distributions.MultivariateNormalDiag(
          loc=loc, scale_diag=scale, validate_args=True)

    def means_layers():
      layer = tf.keras.layers.Dense(
          action_tensor_spec.shape.num_elements(),
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=0.1, seed=_get_seed()),
          kernel_regularizer=tf.keras.regularizers.L2(self._weight_decay),
          name='means_projection_layer')

      return layer

    def std_layers():
      std_bias_initializer_value = np.log(np.exp(0.35) - 1)
      return bias_layer.BiasLayer(
          bias_initializer=tf.constant_initializer(
              value=std_bias_initializer_value))

    def no_op_layers():
      return tf.keras.layers.Lambda(lambda x: x)

    def dense_layer(num_units):
      layer = tf.keras.layers.Dense(
          tf_sparse_utils.scale_width(num_units, self._width),
          activation=tf.nn.tanh,
          kernel_initializer=tf.keras.initializers.Orthogonal(seed=_get_seed()),
          kernel_regularizer=tf.keras.regularizers.L2(self._weight_decay),
          )
      return layer

    all_layers = [dense_layer(n) for n in fc_layer_units]
    all_layers.append(means_layers())
    if self._is_sparse:
      if self._sparse_output_layer:
        all_layers = tf_sparse_utils.wrap_all_layers(all_layers, input_dim)
      else:
        new_layers = tf_sparse_utils.wrap_all_layers(all_layers[:-1], input_dim)
        all_layers = new_layers + all_layers[-1:]

    return sequential.Sequential(
        all_layers +
        [tf.keras.layers.Lambda(
            lambda x: {'loc': x, 'scale': tf.zeros_like(x)})] +
        [nest_map.NestMap({
            'loc': no_op_layers(),
            'scale': std_layers(),
        })] +
        # Create the output distribution from the mean and standard deviation.
        [tf.keras.layers.Lambda(create_dist)])
