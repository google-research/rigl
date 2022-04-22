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

"""Code for analyzing symmetries in NN."""

import functools
import math
import operator
from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np

from rigl.experimental.jax.pruning import masked
from rigl.experimental.jax.utils import utils


def count_permutations_mask_layer(
    mask_layer,
    next_mask_layer = None,
    parameter_key = 'kernel'):
  """Calculates the number of permutations for a layer, given binary masks.

  Args:
   mask_layer: The binary weight mask of a dense/conv layer, where last
     dimension is number of neurons/filters.
   next_mask_layer: The binary weight mask of the following a dense/conv layer,
     or None if this is the last layer.
   parameter_key: The name of the parameter to count the permutations of in each
     layer.

  Returns:
   A dictionary with stats on the permutation structure of a mask, including
   the number of symmetric permutations of the mask, number of unique mask
   columns, count of the zeroed out (structurally pruned) neurons, and total
   number of neurons/filters.
  """
  # Have to check 'is None' since mask_layer[parameter_key] is jnp.array.
  if not mask_layer or parameter_key not in mask_layer or mask_layer[
      parameter_key] is None:
    return {
        'permutations': 1,
        'zeroed_neurons': 0,
        'total_neurons': 0,
        'unique_neurons': 0,
    }

  mask = mask_layer[parameter_key]

  num_neurons = mask.shape[-1]

  # Initialize with stats for an empty mask.
  mask_stats = {
      'permutations': 0,
      'zeroed_neurons': num_neurons,
      'total_neurons': num_neurons,
      'unique_neurons': 0,
  }

  # Re-shape masks as 1D, in case they are 2D (e.g. convolutional).
  connection_mask = mask.reshape(-1, num_neurons)

  # Only consider non-zero columns (in JAX neurons/filters are last index).
  non_zero_neurons = ~jnp.all(connection_mask == 0, axis=0)

  # Count only zeroed neurons in the current layer.
  zeroed_count = num_neurons - jnp.count_nonzero(non_zero_neurons)

  # Special case where all neurons in current layer are ablated.
  if zeroed_count == num_neurons:
    return mask_stats

  # Have to check is None since next_mask_layer[parameter_key] is jnp.array.
  if next_mask_layer and parameter_key in next_mask_layer and next_mask_layer[
      parameter_key] is not None:
    next_mask = next_mask_layer[parameter_key]

    # Re-shape masks as 1D, in case they are 2D (e.g. convolutional).
    next_connection_mask = next_mask.T.reshape(-1, num_neurons)

    # Update with neurons that are non-zero in outgoing connections too.
    non_zero_neurons &= ~jnp.all(next_connection_mask == 0, axis=0)

    # Remove rows corresponding to neurons that are ablated.
    next_connection_mask = next_connection_mask[:, non_zero_neurons]

    connection_mask = connection_mask[:, non_zero_neurons]

    # Combine the outgoing and incoming masks in one vector per-neuron.
    connection_mask = jnp.concatenate(
        (connection_mask, next_connection_mask), axis=0)

  else:
    connection_mask = connection_mask[:, non_zero_neurons]

  # Effectively no connections between these two layers.
  if not connection_mask.size:
    return mask_stats

  # Note: np.unique not implemented in JAX numpy yet.
  _, unique_counts = np.unique(connection_mask, axis=-1, return_counts=True)

  # Convert from device array.
  mask_stats['zeroed_neurons'] = int(zeroed_count)

  mask_stats['permutations'] = functools.reduce(
      operator.mul, (np.math.factorial(t) for t in unique_counts))
  mask_stats['unique_neurons'] = len(unique_counts)

  return mask_stats


def count_permutations_mask(mask):
  """Calculates the number of permutations for a given model mask.

  Args:
    mask: Model masks to check, similar to Model.params.

  Returns:
   A dictionary with stats on the permutation structure of a mask, including
   the number of symmetric permutations of the mask, number of unique mask
   columns, count of the zeroed out (structurally pruned) neurons, and total
   number of neurons/filters.
  """
  sum_keys = ('total_neurons', 'unique_neurons', 'zeroed_neurons')
  product_keys = ('permutations',)

  # Count permutation stats for each pairwise set of layers.
  # Note: I tried doing this with more_itertools.pairwise/itertools.chain, but
  # there is a type conflict in passing iterators of different types to
  # itertools.chain.
  counts = [
      count_permutations_mask_layer(layer, next_layer)
      for layer, next_layer in utils.pairwise_longest(mask.values())
  ]

  sum_stats = {}
  for key in sum_keys:
    sum_stats[key] = functools.reduce(operator.add, (z[key] for z in counts))

  product_stats = {}
  for key in product_keys:
    product_stats[key] = functools.reduce(operator.mul,
                                          (z[key] for z in counts))

  return {**sum_stats, **product_stats}


def get_mask_stats(mask):
  """Calculates an array of mask statistics.

  Args:
    mask: A model mask to calculate the statistics of.

  Returns:
    A dictionary, containing a set of mask statistics.
  """
  mask_stats = count_permutations_mask(mask)
  mask_stats.update({
      'sparsity': masked.mask_sparsity(mask),
      'permutation_num_digits': len(str(mask_stats['permutations'])),
      'permutation_log10': math.log10(mask_stats['permutations'] + 1),
  })

  return mask_stats
