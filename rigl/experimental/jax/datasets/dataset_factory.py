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

"""Dataset Factory.

Dataset factory to allow us to easily use tensorflow datasets (TFDS)
with JAX/FLAX, by defining a bunch of wrappers, including preprocessing.

Attributes:
  DATASETS: A list of the datasets that can be created.
"""

from typing import Any, Mapping, Type
from rigl.experimental.jax.datasets import cifar10
from rigl.experimental.jax.datasets import dataset_base
from rigl.experimental.jax.datasets import mnist
import tensorflow.compat.v2 as tf


DATASETS: Mapping[str, Type[dataset_base.Dataset]] = {
    'MNIST': mnist.MNISTDataset,
    'CIFAR10': cifar10.CIFAR10Dataset,
}


def create_dataset(name, *args, **kwargs):
  """Creates a Tensorflow datasets (TFDS) dataset.

  Args:
      name: The TFDS name of the dataset.
      *args: Dataset arguments.
      **kwargs: Dataset keyword arguments.

  Returns:
      Dataset: An abstracted dataset object.

  Raises:
      ValueError if a dataset with the given name does not exist.
  """
  if name not in DATASETS:
    raise ValueError(f'No such dataset: {name}')
  return DATASETS[name](*args, **kwargs)
