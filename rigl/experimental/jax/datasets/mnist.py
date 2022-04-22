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

"""MNIST Dataset.

Dataset abstraction/factory to allow us to easily use tensorflow datasets (TFDS)
with JAX/FLAX, by defining a bunch of wrappers, including preprocessing.
In this case, the MNIST dataset.
"""
from typing import MutableMapping
from rigl.experimental.jax.datasets import dataset_base
import tensorflow.compat.v2 as tf


class MNISTDataset(dataset_base.ImageDataset):
  """MNIST dataset.

  Attributes:
      NAME: The Tensorflow Dataset's dataset name.
  """
  NAME: str = 'mnist'

  def __init__(self,
               batch_size,
               batch_size_test,
               shuffle_buffer_size = 1024,
               seed = 42):
    """MNIST dataset.

    Args:
        batch_size: The batch size to use for the training datasets.
        batch_size_test: The batch size to used for the test dataset.
        shuffle_buffer_size: The buffer size to use for dataset shuffling.
        seed: The random seed used to shuffle.

    Returns:
        Dataset: A dataset object.
    """
    super().__init__(MNISTDataset.NAME, batch_size, batch_size_test,
                     shuffle_buffer_size, seed)

  def preprocess(
      self, data):
    """Normalizes MNIST images: `uint8` -> `float32`.

    Args:
      data: Data sample.

    Returns:
    Data after being augmented/normalized/transformed.
    """
    data = super().preprocess(data)
    data['image'] = (tf.cast(data['image'], tf.float32) / 255.) - 0.5
    return data
