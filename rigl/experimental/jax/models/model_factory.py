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

"""Factory for neural network models.

Attributes:
  MODELS: A list of the models that can be created.
"""
from typing import Any, Callable, Mapping, Sequence, Tuple, Type

import flax
import jax.numpy as jnp

from rigl.experimental.jax.models import cifar10_cnn
from rigl.experimental.jax.models import mnist_cnn
from rigl.experimental.jax.models import mnist_fc

MODELS: Mapping[str, Type[flax.deprecated.nn.Model]] = {
    'MNIST_CNN': mnist_cnn.MNISTCNN,
    'MNIST_FC': mnist_fc.MNISTFC,
    'CIFAR10_CNN': cifar10_cnn.CIFAR10CNN,
}


def create_model(
    name, rng,
    input_specs, **kwargs
):
  """Creates a Model.

  Args:
      name: the name of the model to instantiate.
      rng : the random number generator to use for init.
      input_specs: an iterable of (shape, dtype) pairs specifying the inputs.
      **kwargs: list of model specific keyword arguments.

  Returns:
      A tuple of FLAX model (flax.deprecated.nn.Model), and initial model state.

  Raises:
      ValueError if a model with the given name does not exist.
  """
  if name not in MODELS:
    raise ValueError('No such model: {}'.format(name))

  with flax.deprecated.nn.stateful() as init_state:
    with flax.deprecated.nn.stochastic(rng):
      model_class = MODELS[name].partial(**kwargs)
      _, params = model_class.init_by_shape(rng, input_specs)

  return flax.deprecated.nn.Model(model_class, params), init_state


def update_model(model,
                 **kwargs):
  """Updates a model to use different model arguments, but same parameters.

  Args:
      model: The model to update.
      **kwargs: List of model specific keyword arguments.

  Returns:
      A FLAX model.
  """
  return flax.deprecated.nn.Model(model.module.partial(**kwargs), model.params)
