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

"""Project inputs to a tanh-squashed MultivariateNormalDiag distribution.

This network reproduces Soft Actor-Critic refererence implementation in:
https://github.com/rail-berkeley/softlearning/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Optional, Text

import gin
import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.typing import types


@gin.configurable
class SparseTanhNormalProjectionNetwork(
    tanh_normal_projection_network.TanhNormalProjectionNetwork):
  """Generates a tanh-squashed MultivariateNormalDiag distribution.

  Note: Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               sample_spec,
               activation_fn = None,
               std_transform = tf.exp,
               name = 'SparseTanhNormalProjectionNetwork',
               weight_decay=0.0):
    """Creates an instance of SparseTanhNormalProjectionNetwork.

    Args:
      sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape and
        dtypes of samples pulled from the output distribution.
      activation_fn: Activation function to use in dense layer.
      std_transform: Transformation function to apply to the stddevs.
      name: A string representing name of the network.
      weight_decay: Weight decay for L2 regularization.
    """
    super(SparseTanhNormalProjectionNetwork, self).__init__(
        sample_spec=sample_spec,
        activation_fn=activation_fn,
        std_transform=std_transform,
        name=name)

    # We reinitialize the projection layer with L2 regularization and also
    # optionally sparsify it.
    self._projection_layer = tf.keras.layers.Dense(
        sample_spec.shape.num_elements() * 2,
        activation=activation_fn,
        kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        name='projection_layer')
