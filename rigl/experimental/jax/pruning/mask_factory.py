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

"""Pruning mask factory.

Attributes:
  MaskFnType: A type alias for functions to create sparse masks.
  MASK_TYPES: Masks types that can be created.
"""
from typing import Any, Callable, Mapping

import flax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import masked

# A function to create a mask, takes as arguments: a flax model, JAX PRNG Key,
# sparsity level as a float in [0, 1].
MaskFnType = Callable[
    [flax.deprecated.nn.Model, Callable[[int],
                                        jnp.array], float], masked.MaskType]

MASK_TYPES: Mapping[str, MaskFnType] = {
    'random':
        masked.shuffled_mask,
    'per_neuron':
        masked.shuffled_neuron_mask,
    'per_neuron_no_input_ablation':
        masked.shuffled_neuron_no_input_ablation_mask,
    'symmetric':
        masked.symmetric_mask,
}


def create_mask(mask_type, base_model,
                rng, sparsity,
                **kwargs):
  """Creates a Mask of the given type.

  Args:
      mask_type: the name of the type of mask to instantiate.
      base_model: the model to create a mask for.
      rng : the random number generator to use for init.
      sparsity: the mask sparsity.
      **kwargs: list of model specific keyword arguments.

  Returns:
      A mask for a FLAX model.

  Raises:
      ValueError if a model with the given name does not exist.
  """
  if mask_type not in MASK_TYPES:
    raise ValueError(f'Unknown mask type: {mask_type}')

  return MASK_TYPES[mask_type](base_model, rng, sparsity, **kwargs)
