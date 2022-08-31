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

"""Convenience Functions for NN training.

Misc. common functions used in training NN models.
"""
import functools
import itertools
import json
import operator
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

import flax
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np



def cross_entropy_loss(log_softmax_logits,
                       labels):
  """Returns the cross-entropy classification loss.

  Args:
    log_softmax_logits: The log of the softmax of the logits for the mini-batch,
      e.g. as output by jax.nn.log_softmax(logits).
    labels: The labels for the mini-batch.
  """
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / labels.size


def compute_metrics(logits,
                    labels):
  """Computes the classification loss and accuracy for a mini-batch.

  Args:
     logits: NN model's logit outputs for the mini-batch.
     labels: The classification labels for the mini-batch.

  Returns:
     Metrics dictionary where 'loss' the mini-batch loss and 'accuracy' is
     the classification accuracy.

  Raises:
    ValueError: If the given logits array is not of the correct shape.
  """
  if len(logits.shape) != 2:
    raise ValueError(
        'Expected an array of (BATCHSIZE, NUM_CLASSES), but got {}'.format(
            logits.shape))

  metrics = {
      'loss': cross_entropy_loss(logits, labels),
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == labels)
  }

  return jax.lax.pmean(metrics, 'batch')


def _np_converter(obj):
  """Explicitly cast Numpy types not recognized by JSON serializer."""
  if isinstance(obj, jnp.integer) or isinstance(obj, np.integer):
    return int(obj)
  elif isinstance(obj, jnp.floating) or isinstance(obj, np.floating):
    return float(obj)
  elif isinstance(obj, jnp.ndarray) or isinstance(obj, np.ndarray):
    return obj.tolist()


def dump_dict_json(data_dict, path):
  """Dumps a dictionary to a JSON file, ensuring Numpy types are cast correctly.

  Args:
    data_dict: A metrics dictionary.
    path: Path of the JSON file to save.

  Raises:
  """

  with open(path, 'w') as json_file:
    json.dump(data_dict, json_file, default=_np_converter)


def count_param(model,
                param_names):
  """Counts the number of parameters in the given model.

  Args:
    model: The model for which to count the parameters.
    param_names: The parameters in each layer which should be accounted for.

  Returns:
    The total number of parameters of the given names in the model.
  """

  param_traversal = flax.optim.ModelParamTraversal(  # pytype: disable=module-attr
      lambda path, _: any(param_name in path for param_name in param_names))

  return functools.reduce(
      operator.add, [param.size for param in param_traversal.iterate(model)], 0)


@jax.jit
def cosine_similarity(a, b):
  """Calculates the cosine similarity between two tensors of same shape."""
  a = a.flatten()
  b = b.flatten()
  return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def param_as_array(params):
  """Returns a Flax parameter pytree as a single numpy weight vector."""
  params_flat = jax.tree_util.tree_leaves(params)
  return jnp.concatenate([param.flatten() for param in params_flat])


def cosine_similarity_model(initial_model,
                            current_model):
  """Calculates the cosine similarity between two model's parameters."""
  initial_params = param_as_array(initial_model.params)
  params = param_as_array(current_model.params)

  return cosine_similarity(initial_params, params)


def vector_difference_norm_model(initial_model,
                                 current_model):
  """Calculates norm of the difference between two model's parameter vectors."""
  initial_params = param_as_array(initial_model.params)
  params = param_as_array(current_model.params)

  return jnp.linalg.norm(params - initial_params)

# Use typevar to hint that we expect unspecified types to match.
T = TypeVar('T')


def pairwise_longest(iterable):
  """Creates a meta-iterator to iterate over current/next values concurrently.

  This is different from itertools pairwise recipe in that it returns the final
  element as (final, None).

  Args:
    iterable: An Iterable of any type.
  Returns:
    An iterable which returns the current and next items in the iterable, or
    None if there is no next. For example, for an iterator over the list
    (1, 2, 3, 4), this would return an iterator as
    ((1, 2), (2, 3), (3, 4), (4, None)).
  """
  # From itertools example documentation.
  a, b = itertools.tee(iterable)
  next(b, None)
  return itertools.zip_longest(a, b)
