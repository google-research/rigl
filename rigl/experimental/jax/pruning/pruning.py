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

"""Functions for pruning FLAX masked models."""
from collections import abc
from typing import Any, Callable, Mapping, Optional, Union

import flax
import jax.numpy as jnp

from rigl.experimental.jax.pruning import masked


def weight_magnitude(weights):
  """Creates weight magnitude-based saliencies, given a weight matrix."""
  return jnp.absolute(weights)


def prune(
    model,
    pruning_rate,
    saliency_fn = weight_magnitude,
    mask = None,
    compare_fn = jnp.greater):
  """Returns a mask for a model where the params in each layer are pruned using a saliency function.

  Args:
    model: The model to create a pruning mask for.
    pruning_rate: The fraction of lowest magnitude saliency weights that are
      pruned. If a float, the same rate is used for all layers, otherwise if it
      is a mapping, it must contain a rate for all masked layers in the model.
    saliency_fn: A function that returns a float number used to rank
      the importance of individual weights in the layer.
    mask: If the model has an existing mask, the mask will be applied before
      pruning the model.
    compare_fn: A pairwise operator to compare saliency with threshold, and
      return True if the saliency indicates the value should not be masked.

  Returns:
    A pruned mask for the given model.
  """
  if not mask:
    mask = masked.simple_mask(model, jnp.ones, masked.WEIGHT_PARAM_NAMES)

  if not isinstance(pruning_rate, abc.Mapping):
    pruning_rate_dict = {}
    for param_name, _ in masked.iterate_mask(mask):
      # Get the layer name from the parameter's full name/path.
      layer_name = param_name.split('/')[-2]
      pruning_rate_dict[layer_name] = pruning_rate
    pruning_rate = pruning_rate_dict

  for param_path, param_mask in masked.iterate_mask(mask):
    split_param_path = param_path.split('/')
    layer_name = split_param_path[-2]
    param_name = split_param_path[-1]

    # If we don't have a pruning rate for the given layer, don't mask it.
    if layer_name in pruning_rate and mask[layer_name][param_name] is not None:
      param_value = model.params[layer_name][
          masked.MaskedModule.UNMASKED][param_name]

      # Here any existing mask is first applied to weight matrix.
      # Note: need to check explicitly is not None for np array.
      if param_mask is not None:
        saliencies = saliency_fn(param_mask * param_value)
      else:
        saliencies = saliency_fn(param_value)

      # TODO: Use partition here (partial sort) instead of sort,
      # since it's O(N), not O(N log N), however JAX doesn't support it.
      sorted_param = jnp.sort(jnp.abs(saliencies.flatten()))

      # Figure out the weight magnitude threshold.
      threshold_index = jnp.round(pruning_rate[layer_name] *
                                  sorted_param.size).astype(jnp.int32)
      threshold = sorted_param[threshold_index]

      mask[layer_name][param_name] = jnp.array(
          compare_fn(saliencies, threshold), dtype=jnp.int32)

  return mask
