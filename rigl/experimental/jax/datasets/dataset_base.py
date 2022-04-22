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

"""Dataset Classes.

Dataset abstraction/factory to allow us to easily use tensorflow datasets (TFDS)
with JAX/FLAX, by defining a bunch of wrappers, including preprocessing.
"""

import abc
from typing import MutableMapping, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset(metaclass=abc.ABCMeta):
  """Base class for datasets.

  Attributes:
      DATAKEY: The key used for the data component of a Tensorflow Dataset
        (TFDS) sample, e.g. 'image' for image datasets.
      LABELKEY: The key used fot the label component of a Tensorflow Dataset
        sample, i.e. 'label'.
      name: The TFDS name of the dataset.
      batch_size: The batch size to use for the training dataset.
      batch_size_test: The batch size to use for the test dataset.
      num_classes: the number of supervised classes in the dataset.
      shape: the shape of an input data array.
  """

  DATAKEY: Optional[str] = None
  LABELKEY: str = 'label'

  def __init__(self,
               name,
               batch_size,
               batch_size_test,
               shuffle_buffer_size,
               prefetch_size = 1,
               seed = None):  # pytype: disable=annotation-type-mismatch
    """Base class for datasets.

    Args:
        name: The TFDS name of the dataset.
        batch_size: The batch size to use for the training dataset.
        batch_size_test: The batch size to use for the test dataset.
        shuffle_buffer_size: The buffer size to use for dataset shuffling.
        prefetch_size: The number of mini-batches to prefetch.
        seed: The random seed used to shuffle.

    Returns:
        A Dataset object.
    """
    super().__init__()
    self.name = name
    self.batch_size = batch_size
    self.batch_size_test = batch_size_test
    self._shuffle_buffer_size = shuffle_buffer_size
    self._prefetch_size = prefetch_size

    self._train_ds, self._train_info = tfds.load(
        self.name,
        split=tfds.Split.TRAIN,
        data_dir=self._dataset_dir(),
        with_info=True)
    self._train_ds = self._train_ds.shuffle(
        self._shuffle_buffer_size,
        seed).map(self.preprocess).cache().map(self.augment).batch(
            self.batch_size, drop_remainder=True).prefetch(self._prefetch_size)

    self._test_ds, self._test_info = tfds.load(
        self.name,
        split=tfds.Split.TEST,
        data_dir=self._dataset_dir(),
        with_info=True)
    self._test_ds = self._test_ds.map(self.preprocess).cache().batch(
        self.batch_size_test).prefetch(self._prefetch_size)

    self.num_classes = self._train_info.features['label'].num_classes
    self.shape = self._train_info.features['image'].shape

  def _dataset_dir(self):
    """Returns the dataset path for the TFDS data."""
    return None

  def get_train(self):
    """Returns the training dataset."""
    return iter(tfds.as_numpy(self._train_ds))

  def get_train_len(self):
    """Returns the length of the training dataset."""
    return self._train_info.splits['train'].num_examples

  def get_test(self):
    """Returns the test dataset."""
    return iter(tfds.as_numpy(self._test_ds))

  def get_test_len(self):
    """Returns the length of the test dataset."""
    return self._test_info.splits['test'].num_examples

  def preprocess(
      self, data):
    """Preprocessing fn used by TFDS map for normalization.

    This function is for transformations that can be cached, e.g.
    normalization/whitening.

    Args:
      data: Data sample.

    Returns:
    Data after being normalized/transformed.
    """
    return data

  def augment(
      self, data):
    """Preprocessing fn used by TFDS map for augmentation at training time.

    This function is for transformations that should not be cached, e.g. random
    augmentation that should change for every sample, and are only applied at
    training time.

    Args:
      data: Data sample.

    Returns:
    Data after being augmented/transformed.
    """
    return data


class ImageDataset(Dataset):
  """Base class for image datasets."""

  DATAKEY = 'image'

  def preprocess(
      self, data):
    """Preprocessing function used by TFDS map for normalization.

    This function is for transformations that can be cached, e.g.
    normalization/whitening.

    Args:
      data: Data sample.

    Returns:
      Data after being normalized/transformed.
    """
    data = super().preprocess(data)
    # Ensure we only provide the image and label, stripping out other keys.
    return dict((key, val)
                for key, val in data.items()
                if key in [self.LABELKEY, self.DATAKEY])
