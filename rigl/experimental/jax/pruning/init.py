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

"""Tools for initialization of masked models."""
import functools
from typing import Callable, Sequence, Optional

import flax
import jax
import jax.numpy as jnp


def sparse_init(
    base_init,
    mask,
    dtype=jnp.float32):
  """Weight initializer with correct fan in/fan out for a masked model.

  The weight initializer uses any dense initializer to correctly initialize a
  masked weight matrix by calling the given initialization method with the
  correct fan in/fan out for every neuron in the layer. If the mask is None, it
  reverts to the original initialization method.

  Args:
    base_init: The base (dense) initialization method to use.
    mask: The layer's mask, or None.
    dtype: The weight array jnp.dtype.

  Returns:
    An initialization method that is mask aware for the given layer and mask.
  """
  def init(rng, shape, dtype=dtype):
    if mask is None:
      return base_init(rng, shape, dtype)

    # Find the ablated neurons in the mask, to determine correct fan_out.
    neuron_weight_count = jnp.sum(
        jnp.reshape(mask, (-1, mask.shape[-1])), axis=0)
    non_zero_neurons = jnp.sum(neuron_weight_count != 0)

    # Special case of completely ablated weight matrix/layer.
    if jnp.sum(non_zero_neurons) == 0:
      print('Empty weight mask!')
      return jnp.zeros(shape, dtype)

    # Neurons have different fan_in w/mask, build up initialization per-unit.
    init_cols = []
    rng, *split_rngs = jax.random.split(rng, mask.shape[-1] + 1)
    for i in range(mask.shape[-1]):
      # Special case of ablated neuron.
      if neuron_weight_count[i] == 0:
        init_cols.append(jnp.zeros(shape[:-1] + (1,), dtype))
        continue

      # Fake shape of weight matrix with correct fan_in, and fan_out.
      sparse_shape = (int(neuron_weight_count[i]), int(non_zero_neurons))

      # Use only the first column of init from initializer, since faked fan_out.
      init = base_init(split_rngs[i], sparse_shape, dtype)[Ellipsis, 0]

      # Expand out to full sparse array.
      expanded_init = jnp.zeros(
          mask[Ellipsis, i].shape,
          dtype).flatten().at[jnp.where(mask[Ellipsis, i].flatten() == 1)].set(init)
      expanded_init = jnp.reshape(expanded_init, mask[Ellipsis, i].shape)
      init_cols.append(expanded_init[Ellipsis, jnp.newaxis])

    return jnp.concatenate(init_cols, axis=-1)

  return init


xavier_sparse_normal = glorot_sparse_normal = functools.partial(
    sparse_init, flax.deprecated.nn.initializers.xavier_normal())
kaiming_sparse_normal = he_sparse_normal = functools.partial(
    sparse_init, flax.deprecated.nn.initializers.kaiming_normal())
