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

"""Masked wrapped for FLAX modules.

Attributes:
  WEIGHT_PARAM_NAMES: The name of the weight parameters to use.
  MaskType: Model mask type for static type checking.
  MaskLayerType: Mask layer type for static type checking.
  MutableMaskType: Mutable model mask type for static type checking.
  MutableMaskLayerType: Mutable mask layer type for static type checking.
"""
import functools
import operator
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Type

from absl import logging
import flax
import jax
import jax.numpy as jnp
import jax.ops

# Model weight param names, e.g. 'kernel', (as opposed batch norm param, etc).
WEIGHT_PARAM_NAMES = ('kernel',)  # Note: Bias is not typically masked.


# Mask layer type for static type checking.
MaskLayerType = Mapping[str, Optional[jnp.array]]


# Model mask type for static type checking.
MaskType = Mapping[str, Optional[MaskLayerType]]


# Mask layer type for static type checking.
MutableMaskLayerType = MutableMapping[str, Optional[jnp.array]]


# Model mask type for static type checking.
MutableMaskType = MutableMapping[str, MutableMaskLayerType]


class MaskedModule(flax.deprecated.nn.Module):
  """Generic FLAX Masking Module.

     Masks a FLAX module, given a mask for params of each layer.

     Attributes:
       UNMASKED: The key to use for the unmasked parameter dictionary.
  """

  UNMASKED = 'unmasked'

  def apply(self,
            *args,
            wrapped_module,
            mask = None,
            **kwargs):
    """Apply the wrapped module, while applying the given masks to its params.

    Args:
      *args: The positional arguments for the wrapped module.
      wrapped_module: The module class to be wrapped.
      mask: The mask nested dictionary containing masks for the wrapped module's
        params, in the same format/with the same keys as the module param dict
        (or None if not to mask).
      **kwargs: The keyword arguments for the wrapped module.

    Returns:
    The intermediate outputs specified by truncate_path.

    Raises:
    ValueError: If the given mask is not valid for the wrapped module, i.e. the
                pytrees do not match.
    """

    # Explicitly create the parameters of the wrapped module.
    def init_fn(rng, input_shape):
      del input_shape  # Unused.

      # Call init to get the params of the wrapped module.
      _, params = wrapped_module.init(rng, *args, **kwargs)
      return params

    unmasked_params = self.param(self.UNMASKED, None, init_fn)

    if mask is not None:
      try:
        masked_params = jax.tree_util.tree_map(
            lambda x, *xs: x
            if xs[0] is None else x * xs[0], unmasked_params, mask)
      except ValueError as err:
        raise ValueError('Mask is invalid for model.') from err

      # Call the wrapped module with the masked params.
      return wrapped_module.call(masked_params, *args, **kwargs)
    else:
      logging.warning('Using masked module without mask!')
      # Call the wrapped module with the unmasked params.
      return wrapped_module.call(unmasked_params, *args, **kwargs)


def masked(module, mask):
  """Convenience function for masking a FLAX module with MaskedModule."""
  return MaskedModule.partial(wrapped_module=module, mask=mask)


def generate_model_masks(
    depth,
    mask = None,
    masked_layer_indices = None):
  """Creates empty masks for this model, or initializes with existing mask.

  Args:
    depth: Number of layers in the model.
    mask: Existing model mask for layers in this model, if not given, all
      module masks are initialized to None.
    masked_layer_indices: The layer indices of layers in model to be masked, or
      all if None.

  Returns:
    A model mask, with None where no mask is given for a model layer, or that
    specific layer is indicated as not to be masked by the masked_layer_indices
    parameter.
  """
  if depth <= 0:
    raise ValueError(f'Invalid model depth: {depth}')

  if mask is None:
    mask = {f'MaskedModule_{i}': None for i in range(depth)}

  # Have to explicitly check for None to differentiate from empty array.
  if masked_layer_indices is not None:
    # Check none of the indices are outside of model's layer bounds.
    if any(i < 0 or i >= depth for i in masked_layer_indices):
      raise ValueError(
          f'Invalid indices for given depth ({depth}): {masked_layer_indices}')
    mask = {
        f'MaskedModule_{i}': mask[f'MaskedModule_{i}']
        for i in masked_layer_indices
    }

  return mask


def _filter_param(param_names,
                  invert = False):
  """Convenience function for filtering maskable parameters from paths.

  Args:
    param_names: Names of parameters we are looking for.
    invert: Inverts filter to exclude, rather than include, given parameters.

  Returns:
    A function to use with flax.deprecated.nn.optim.ModelParamTraversal for
    filtering.
  """

  def filter_fn(path, value):
    del value  # Unused.
    parameter_found = any([
        '{}/{}'.format(MaskedModule.UNMASKED, param_name) in path
        for param_name in param_names
    ])
    return not parameter_found if invert else parameter_found

  return filter_fn


def mask_map(model,
             fn):
  """Convenience function to create a mask for a model.

  Args:
    model: The Flax model, with at least one MaskedModule layer.
    fn: The function to call on each masked parameter, to create the mask for
      that parameter, takes the parameter name, and parameter value as arguments
      and returns the new parameter value.

  Returns:
    A model parameter dictionary, with all masked parameters set by the given
    function, and all other parameters set to None.

  Raises:
    ValueError: If the given model does not support masking, i.e. none of the
                layers are wrapped by a MaskedModule.
  """
  maskable = False
  for layer_key, layer in model.params.items():
    if MaskedModule.UNMASKED not in layer:
      logging.warning(
          'Layer \'%s\' does not support masking, i.e. it is not '
          'wrapped by a MaskedModule', layer_key)
    else:
      maskable = True

  if not maskable:
    raise ValueError('Model does not support masking, i.e. no layers are '
                     'wrapped by a MaskedModule.')

  # First set all non-masked params to None in copy of model pytree.
  filter_non_masked = _filter_param(WEIGHT_PARAM_NAMES, invert=True)
  nonmasked_traversal = flax.optim.ModelParamTraversal(filter_non_masked)  # pytype: disable=module-attr
  mask_model = nonmasked_traversal.update(lambda _: None, model)

  # Then find params to mask, and set to array.
  for param_name in WEIGHT_PARAM_NAMES:
    filter_masked = _filter_param(WEIGHT_PARAM_NAMES)
    mask_traversal = flax.optim.ModelParamTraversal(filter_masked)  # pytype: disable=module-attr
    mask_model = mask_traversal.update(
        functools.partial(fn, param_name), mask_model)

  mask = mask_model.params
  # Remove unneeded unmasked param for mask.
  for layer_key, layer in mask.items():
    if MaskedModule.UNMASKED in layer:
      mask[layer_key] = layer[MaskedModule.UNMASKED]

  return mask


def iterate_mask(
    mask,
    param_names = None
):
  """Iterate over the parameters in as mask.

  Args:
    mask: The model mask.
    param_names: The parameter names to iterate over in each layer, if None
      iterates over all parameters of all layers.

  Yields:
    An iterator of tuples containing the parameter path and parameter value
    in sorted order of layer parameters matching the names in param_names (or
    all parameters if None).
  """
  flat_mask = flax.traverse_util.flatten_dict(mask)
  for key, value in flat_mask.items():
    if param_names is None or key in param_names:
      path = '/' + '/'.join(key)
      yield path, value


def shuffled_mask(model, rng,
                  sparsity):
  """Returns a randomly shuffled mask with a given sparsity for all layers.

  Returns a random weight mask for a model param array, by randomly shuffling a
  mask with a fixed number of non-zero/zero entries, given by the sparsity.

  Args:
    model: Flax model that contains masked modules.
    rng: Random number generator, i.e. jax.random.PRNGKey.
    sparsity: The per-layer sparsity of the mask (i.e. % of zeroes), 1.0 will
      mask all weights, while 0 will mask none.

  Returns:
    A randomly shuffled weight mask, in the same form as flax.Module.params.

  Raises:
    ValueError: If the sparsity is beyond the bounds [0, 1], or no layers are
                maskable, i.e. is wrapped by MaskedModule.
  """
  if sparsity > 1 or sparsity < 0:
    raise ValueError(
        'Given sparsity, {}, is not in range [0, 1]'.format(sparsity))

  def create_shuffled_mask(param_name, param):
    del param_name  # Unused.
    mask = jnp.arange(param.size)
    mask = jnp.where(mask >= sparsity * param.size, jnp.ones_like(mask),
                     jnp.zeros_like(mask))
    mask = jax.random.permutation(rng, mask)
    return mask.reshape(param.shape)

  return mask_map(model, create_shuffled_mask)


def random_mask(model,
                rng,
                mean_sparsity = 0.5):
  """Returns a random weight mask for a masked model.

  Args:
    model: Flax model that contains masked modules.
    rng: Random number generator, i.e. jax.random.PRNGKey.
    mean_sparsity: The mean number of 0's in the mask, i.e. mean = (1 -
      mean_sparsity) for the Bernoulli distribution to sample from.

  Returns:
    A random weight mask, in the same form as flax.Module.params

  Raises:
    ValueError: If the sparsity is beyond the bounds [0, 1], or if a layer to
                mask is not maskable, i.e. is not wrapped by MaskedModule.
  """
  if mean_sparsity > 1 or mean_sparsity < 0:
    raise ValueError(
        'Given sparsity, {}, is not in range [0, 1]'.format(mean_sparsity))

  # Invert mean_sparsity to get mean for Bernoulli distribution.
  mean = 1. - mean_sparsity

  def create_random_mask(param_name, param):
    del param_name  # Unused.
    return jax.random.bernoulli(
        rng, p=mean,
        shape=param.shape).astype(jnp.int32)  # TPU doesn't support uint8.

  return mask_map(model, create_random_mask)


def simple_mask(model,
                init_fn,
                masked_param):
  """Creates a mask given a model and numpy initialization function.

  Args:
    model: The model to create a mask for.
    init_fn: The numpy initialization function, e.g. numpy.ones.
    masked_param: The list of parameters to mask.

  Returns:
    A mask for the model.
  """

  def create_init_fn_mask(param_name, param):
    if param_name in masked_param:
      return init_fn(param.shape)
    return None

  return mask_map(model, create_init_fn_mask)


def symmetric_mask(model,
                   rng,
                   sparsity = 0.5):
  """Generates a random weight mask that's symmetric, i.e. structurally pruned.

  Args:
    model: Flax model that contains masked modules.
    rng: Random number generator, i.e. jax.random.PRNGKey.
    sparsity: The per-layer sparsity of the mask (i.e. % of zeroes), in the
      range  [0, 1]: 1.0 will mask all weights, while 0 will mask none.

  Returns:
    A symmetric random weight mask, in the same form as flax.Module.params.
  """
  if sparsity > 1 or sparsity < 0:
    raise ValueError(f'Given sparsity, {sparsity}, is not in range [0, 1]')

  def create_neuron_symmetric_mask(param_name, param):
    del param_name  # Unused.
    neuron_length = functools.reduce(operator.mul, param.shape[:-1])
    neuron_mask = jnp.arange(neuron_length)
    neuron_mask = jnp.where(neuron_mask >= sparsity * neuron_mask.size,
                            jnp.ones_like(neuron_mask),
                            jnp.zeros_like(neuron_mask))
    neuron_mask = jax.random.shuffle(rng, neuron_mask)
    mask = jnp.repeat(neuron_mask[Ellipsis, jnp.newaxis], param.shape[-1], axis=1)
    return mask.reshape(param.shape)

  return mask_map(model, create_neuron_symmetric_mask)


class _PerNeuronShuffle:
  """This class is needed to get around the fact that JAX RNG is stateless."""

  def __init__(self, init_rng, sparsity):
    """Creates the per-neuron shuffle class, with initial RNG state.

    Args:
      init_rng: The initial random number generator state to use.
      sparsity: The per-layer sparsity of the mask (i.e. % of zeroes), 1.0 will
        mask all weights, while 0 will mask none.
    """
    self._rng = init_rng
    self._sparsity = sparsity

  def __call__(self, param_name, param):
    """Shuffles the weight matrix/mask for a given parameter, per-neuron.

    This is to be used with mask_map, and accepts the standard mask_map
    function parameters.

    Args:
      param_name: The parameter's name.
      param: The parameter's weight or mask matrix.

    Returns:
      A shuffled weight/mask matrix, with each neuron shuffled independently.
    """
    del param_name  # Unused.
    neuron_length = functools.reduce(operator.mul, param.shape[:-1])
    neuron_mask = jnp.arange(neuron_length)
    neuron_mask = jnp.where(neuron_mask >= self._sparsity * neuron_mask.size,
                            jnp.ones_like(neuron_mask),
                            jnp.zeros_like(neuron_mask))
    mask = jnp.repeat(neuron_mask[Ellipsis, jnp.newaxis], param.shape[-1], axis=1)
    self._rng, rng_input = jax.random.split(self._rng)
    mask = jax.random.shuffle(rng_input, mask, axis=0)
    return mask.reshape(param.shape)


def shuffled_neuron_mask(model,
                         rng,
                         sparsity):
  """Returns a shuffled mask with a given fixed sparsity for all neurons/layers.

  Returns a randomly shuffled weight mask for a model param array, by setting a
  fixed sparsity (i.e. number of ones/zeros) for every neuron's weight vector
  in the model, and then randomly shuffling each neuron's weight mask with a
  fixed number of non-zero/zero entries, given by the sparsity. This ensures no
  neuron is ablated for a non-zero sparsity.

  Note: This is much more complicated for convolutional layers due to the
  receptive field being different for every pixel! We only take into account
  channel-wise masks and not spatial ablations in propagation in that case.

  Args:
    model: Flax model that contains masked modules.
    rng: Random number generator, i.e. jax.random.PRNGKey.
    sparsity: The per-layer sparsity of the mask (i.e. % of zeroes), 1.0 will
      mask all weights, while 0 will mask none.

  Returns:
    A randomly shuffled weight mask, in the same form as flax.Module.params.

  Raises:
    ValueError: If the sparsity is beyond the bounds [0, 1], or no layers are
                maskable, i.e. is wrapped by MaskedModule.
  """
  if sparsity > 1 or sparsity < 0:
    raise ValueError(f'Given sparsity, {sparsity}, is not in range [0, 1]')

  return mask_map(model, _PerNeuronShuffle(rng, sparsity))


def _fill_diagonal_wrap(shape,
                        value,
                        dtype = jnp.uint8):
  """Fills the diagonal of a 2D array, while also wrapping tall arrays.

  For a matrix of dimensions (N x M),:
    if N <= M, i.e. the array is wide rectangular, the array's diagonal is
    filled, for example:

    _fill_diagonal_wrap(jnp.zeroes((2, 3), dtype=uint8), 1)
    > [[1, 0, 0],
       [0, 1, 0]]

    if N > M, i.e. the array is tall rectangular, the array's diagonal, and
    offset diagonals are filled. This differs from
    numpy.fill_diagonal(..., wrap=True), in that it does not include a single
    row gap between the diagonals, and it is not in-place but returns a copy of
    the given array. For example,

    _fill_diagonal_wrap(jnp.zeroes((3, 2), dtype=uint8), 1)
    > [[1, 0],
       [0, 1],
       [1, 0]]

  Args:
    shape: The shape of the 2D array to return with the diagonal filled.
    value: The value to fill in for the diagonal, and offset diagonals.
    dtype: The datatype of the jax numpy array to return.
  Returns:
    A copy of the given array with the main diagonal filled, and offset
    diagonals filled if the given array is tall.
  """
  if len(shape) != 2:
    raise ValueError(
        f'Expected an 2D array, however array has dimensions: {shape}')

  array = jnp.zeros(shape, dtype=dtype)
  rows, cols = shape

  def diagonal_indices(offset):  # Returns jax.ops._Indexable.
    """Returns slice of the nth diagonal of an array, where n is offset."""
    # This is an a numpy-style advanced slice of the form [start:end:step], that
    # gives you the offset (vertically) diagonal of an array. If it was the main
    # diagonal of a (flattened) square matrix of n X n it would be 0:n**2:n+1,
    # i.e. start at 0, and look at each n+1 elements, end when you get to end
    # of array. We need to look at vertically-offset diagonals as well, which is
    # handled by offset.
    return jnp.index_exp[cols * offset:cols * (offset + cols):cols + 1]

  # Fills (square) matrix diagonals with the given value, tiling over tall
  # rectangular arrays by offsetting the filled diagonals by multiples of the
  # height of the square arrays.
  diagonals = [
      array.ravel().at[diagonal_indices(offset)].set(value).reshape(array.shape)
      for offset in range(0, rows, cols)
  ]
  return functools.reduce(jnp.add, diagonals)


def _random_neuron_mask(neuron_length,
                        unmasked_count,
                        rng,
                        dtype = jnp.uint32):
  """Generates a random mask for a neuron.

  Args:
    neuron_length: The length of the neuron's weight vector.
    unmasked_count: The number of elements that should be unmasked.
    rng: A jax.random.PRNGKey random seed.
    dtype: Type of array to create.
  Returns:
    A random neuron weight vector mask.
  """
  if unmasked_count > neuron_length:
    raise ValueError('unmasked_count cannot be greater that neuron_length: '
                     f'{unmasked_count} > {neuron_length}')
  neuron_mask = jnp.concatenate(
      (jnp.ones(unmasked_count), jnp.zeros(neuron_length - unmasked_count)),
      axis=0)
  neuron_mask = jax.random.shuffle(rng, neuron_mask)
  return neuron_mask.astype(dtype)


class _PerNeuronNoInputAblationShuffle:
  """This class is needed to get around the fact that JAX RNG is stateless."""

  def __init__(self, init_rng, sparsity):
    """Creates the per-neuron shuffle class, with initial RNG state.

    Args:
      init_rng: The initial random number generator state to use.
      sparsity: The per-layer sparsity of the mask (i.e. % of zeroes), 1.0 will
        mask all weights, while 0 will mask none.
    """
    self._rng = init_rng
    self._sparsity = sparsity

  def _get_rng(self):
    """Creates a new JAX RNG, while updating RNG state."""
    self._rng, rng_input = jax.random.split(self._rng)
    return rng_input

  def __call__(self, param_name, param):
    """Shuffles the weight matrix/mask for a given parameter, per-neuron.

    This is to be used with mask_map, and accepts the standard mask_map
    function parameters.

    Args:
      param_name: The parameter's name.
      param: The parameter's weight or mask matrix.

    Returns:
      A shuffled weight/mask matrix, with each neuron shuffled independently.
    """
    del param_name  # Unused.

    incoming_connections = jnp.prod(jnp.array(param.shape[:-1]))
    num_neurons = param.shape[-1]

    # Ensure each input neuron has at least one connection unmasked.
    mask = _fill_diagonal_wrap((incoming_connections, num_neurons), 1,
                               dtype=jnp.uint8)

    # Randomly shuffle which of the neurons have these connections.
    mask = jax.random.shuffle(self._get_rng(), mask, axis=0)

    # Add extra required random connections to mask to satisfy sparsity.
    mask_cols = []
    for col in range(mask.shape[-1]):
      neuron_mask = mask[:, col]
      off_diagonal_count = max(
          round((1 - self._sparsity) * incoming_connections)
          - jnp.count_nonzero(neuron_mask), 0)

      zero_indices = jnp.flatnonzero(neuron_mask == 0)
      random_entries = _random_neuron_mask(
          len(zero_indices), off_diagonal_count, self._get_rng())

      neuron_mask = neuron_mask.at[zero_indices].set(random_entries)
      mask_cols.append(neuron_mask)

    return jnp.column_stack(mask_cols).reshape(param.shape)


def shuffled_neuron_no_input_ablation_mask(model,
                                           rng,
                                           sparsity):
  """Returns a shuffled mask with a given fixed sparsity for all neurons/layers.

  Returns a randomly shuffled weight mask for a model param array, by setting a
  fixed sparsity (i.e. number of ones/zeros) for every neuron's weight vector
  in the model, and then randomly shuffling each neuron's weight mask with a
  fixed number of non-zero/zero entries, given by the sparsity. This ensures no
  neuron is ablated for a non-zero sparsity.

  This function also ensures that no neurons in the previous layer are
  effectively ablated, by ensuring that each neuron has at least one connection.

  Note: This is much more complicated for convolutional layers due to the
  receptive field being different for every pixel! We only take into account
  channel-wise masks and not spatial ablations in propagation in that case.

  Args:
    model: Flax model that contains masked modules.
    rng: Random number generator, i.e. jax.random.PRNGKey.
    sparsity: The per-layer sparsity of the mask (i.e. % of zeroes), 1.0 will
      mask all weights, except for the minimum number required to maintain,
      connectivity with the input layer, while 0 will mask none.

  Returns:
    A randomly shuffled weight mask, in the same form as flax.Module.params.

  Raises:
    ValueError: If the sparsity is beyond the bounds [0, 1], or no layers are
                maskable, i.e. is wrapped by MaskedModule.
  """
  if sparsity > 1.0 or sparsity < 0.0:
    raise ValueError(f'Given sparsity, {sparsity}, is not in range [0, 1]')

  # First, generate a random permutation matrix, and ensure our mask has at
  # least N connections, where there are N neurons in the previous layer.
  return mask_map(model, _PerNeuronNoInputAblationShuffle(rng, sparsity))


def propagate_masks(
    mask,
    param_names = WEIGHT_PARAM_NAMES
):
  """Accounts for implicitly pruned neurons in a model's weight masks.

  When neurons are randomly ablated in one layer, they can effectively ablate
  neurons in the next layer if in effect all incoming weights of a neuron are
  zero. This method accounts for this by propagating forward mask information
  through the entire model.

  Args:
    mask: Model masks to check, in same pytree structure as Model.params.
    param_names: List of param keys in mask to count.

  Returns:
   A refined model mask with weights that are effectively ablated in the
   original mask set to zero.
  """

  flat_mask = flax.traverse_util.flatten_dict(mask)
  mask_layer_list = list(flat_mask.values())
  mask_layer_keys = list(flat_mask.keys())

  mask_layer_param_names = [layer_param[-1] for layer_param in mask_layer_keys]

  for param_name in param_names:
    # Find which of the param arrays correspond to leaf nodes with this name.
    param_indices = [
        i for i, names in enumerate(mask_layer_param_names)
        if param_name in names
    ]

    for i in range(1, len(param_indices)):
      last_weight_mask = mask_layer_list[param_indices[i - 1]]
      weight_mask = mask_layer_list[param_indices[i]]

      if last_weight_mask is None or weight_mask is None:
        continue

      last_weight_mask_reshaped = jnp.reshape(last_weight_mask,
                                              (-1, last_weight_mask.shape[-1]))

      # Neurons with any outgoing weights from previous layer.
      alive_incoming = jnp.sum(last_weight_mask_reshaped, axis=0) != 0

      # Combine effective mask of previous layer with neuron's current mask.
      if len(weight_mask.shape) > 2:
        # Convolutional layer, only consider channel-wise masks, if any spatial
        # weight is non-zero that channel is considered non-masked.
        spatial_dim = len(weight_mask.shape) - 2
        new_weight_mask = alive_incoming[:, jnp.newaxis] * jnp.amax(
            weight_mask, axis=tuple(range(spatial_dim)))
        new_weight_mask = jnp.tile(new_weight_mask,
                                   weight_mask.shape[:-2] + (1, 1))
      else:
        # Check for case of dense following convolution, i.e. spatial input into
        # dense, to prevent b/156135283. Must use convolution for these layers.
        if len(last_weight_mask.shape) > 2:
          raise ValueError(
              'propagate_masks requires knowledge of the spatial '
              'dimensions of the previous layer. Use a functionally equivalent '
              'conv. layer in place of a dense layer in a model with a mixed '
              'conv/dense setting.')
        new_weight_mask = alive_incoming[:, jnp.newaxis] * weight_mask

      mask_layer_list[param_indices[i]] = jnp.reshape(
          new_weight_mask, mask_layer_list[param_indices[i]].shape)

  return flax.traverse_util.unflatten_dict(
      dict(zip(mask_layer_keys, mask_layer_list)))


def mask_layer_sparsity(mask_layer):
  """Calculates the sparsity of a single layer's mask.

  Args:
    mask_layer: mask layer to calculate the sparsity of.

  Returns:
    The sparsity of the mask.
  """
  parameter_count = 0
  masked_count = 0

  for key in mask_layer:
    if mask_layer[key] is not None and key in WEIGHT_PARAM_NAMES:
      parameter_count += mask_layer[key].size
      masked_count += jnp.sum(mask_layer[key])

  if parameter_count == 0:
    return 0.

  return 1. - masked_count/parameter_count


def mask_sparsity(
    mask,
    param_names = None):
  """Calculates the sparsity of the given parameters over a model mask.

  Args:
    mask: Model mask to calculate sparsity over.
    param_names: List of param keys in mask to count.

  Returns:
    The overall sparsity of the mask.
  """
  if param_names is None:
    param_names = WEIGHT_PARAM_NAMES

  parameter_count = 0
  masked_count = 0

  for path, value in iterate_mask(mask):
    if value is not None and any(
        param_name in path for param_name in param_names):
      parameter_count += value.size
      masked_count += jnp.sum(value.flatten())

  if parameter_count == 0:
    return 0.

  return 1.0 - float(masked_count / parameter_count)
