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

"""CIFAR10 Dataset.

Dataset abstraction/factory to allow us to easily use tensorflow datasets (TFDS)
with JAX/FLAX, by defining a bunch of wrappers, including preprocessing.
In this case, the CIFAR10 dataset.
"""
from typing import MutableMapping, Sequence
from rigl.experimental.jax.datasets import dataset_base
import tensorflow.compat.v2 as tf


class CIFAR10Dataset(dataset_base.ImageDataset):
  """CIFAR10 dataset.

  Attributes:
      NAME: The Tensorflow Dataset's dataset name.
  """
  NAME: str = 'cifar10'
  # Computed from the training set by taking the per-channel mean/std-dev
  # over sample, height and width axes of all training samples.
  MEAN_RGB: Sequence[float] = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
  STDDEV_RGB: Sequence[float] = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]

  def __init__(self,
               batch_size,
               batch_size_test,
               shuffle_buffer_size = 1024,
               seed = 42):
    """CIFAR10 dataset.

    Args:
        batch_size: The batch size to use for the training datasets.
        batch_size_test: The batch size used for the test dataset.
        shuffle_buffer_size: The buffer size to use for dataset shuffling.
        seed: The random seed used to shuffle.

    Returns:
        Dataset: A dataset object.

    Raises:
        ValueError: If the test dataset is not evenly divisible by the
                    test batch size.
    """
    super().__init__(CIFAR10Dataset.NAME, batch_size, batch_size_test,
                     shuffle_buffer_size, seed)
    if self.get_test_len() % batch_size_test != 0:
      raise ValueError(
          'Test data not evenly divisible by batch size: {} % {} != 0.'.format(
              self.get_test_len(), batch_size_test))

  def preprocess(
      self, data):
    """Normalizes CIFAR10 images: `uint8` -> `float32`.

    Args:
      data: Data sample.

    Returns:
    Data after being augmented/normalized/transformed.
    """
    data = super().preprocess(data)
    mean_rgb = tf.constant(self.MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
    std_rgb = tf.constant(self.STDDEV_RGB, shape=[1, 1, 3], dtype=tf.float32)

    data['image'] = (tf.cast(data['image'], tf.float32) - mean_rgb) / std_rgb
    return data
