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

"""Tests for weight_symmetry.datasets.cifar10."""
from absl.testing import absltest
import numpy as np

from rigl.experimental.jax.datasets import cifar10


class CIFAR10DatasetTest(absltest.TestCase):
  """Test cases for CIFAR10 Dataset."""

  def setUp(self):
    """Common setup routines/variables for test cases."""
    super().setUp()
    self._batch_size = 16
    self._batch_size_test = 10
    self._shuffle_buffer_size = 8

    self._dataset = cifar10.CIFAR10Dataset(
        self._batch_size,
        batch_size_test=self._batch_size_test,
        shuffle_buffer_size=self._shuffle_buffer_size)

  def test_create_dataset(self):
    """Tests creation of dataset."""
    self.assertIsInstance(self._dataset, cifar10.CIFAR10Dataset)

  def test_train_image_dims_content(self):
    """Tests dimensions and contents of test data."""
    iterator = self._dataset.get_train()
    sample = next(iterator)
    image, label = sample['image'], sample['label']

    with self.subTest(name='DataShape'):
      self.assertTupleEqual(image.shape, (self._batch_size, 32, 32, 3))

    with self.subTest(name='DataType'):
      self.assertTrue(np.issubdtype(image.dtype, float))

    with self.subTest(name='DataValues'):
      # Normalized by stddev., expect nothing to fall outside 3 stddev.
      self.assertTrue((image >= -3.).all() and (image <= 3.).all())

    with self.subTest(name='LabelShape'):
      self.assertLen(label, self._batch_size)

    with self.subTest(name='LabelType'):
      self.assertTrue(np.issubdtype(label.dtype, int))

    with self.subTest(name='LabelValues'):
      self.assertTrue((label >= 0).all() and
                      (label <= self._dataset.num_classes).all())

  def test_test_image_dims_content(self):
    """Tests dimensions and contents of train data."""
    iterator = self._dataset.get_test()
    sample = next(iterator)
    image, label = sample['image'], sample['label']

    with self.subTest(name='DataShape'):
      self.assertTupleEqual(image.shape, (self._batch_size_test, 32, 32, 3))

    with self.subTest(name='DataType'):
      self.assertTrue(np.issubdtype(image.dtype, float))

    with self.subTest(name='DataValues'):
      # Normalized by stddev., expect nothing to fall outside 3 stddev.
      self.assertTrue((image >= -3.).all() and (image <= 3.).all())

    with self.subTest(name='LabelShape'):
      self.assertLen(label, self._batch_size_test)

    with self.subTest(name='LabelType'):
      self.assertTrue(np.issubdtype(label.dtype, int))

    with self.subTest(name='LabelValues'):
      self.assertTrue((label >= 0).all() and
                      (label <= self._dataset.num_classes).all())

  def test_train_data_length(self):
    """Tests length of training dataset."""
    total_count = 0
    for batch in self._dataset.get_train():
      total_count += len(batch['label'])

    self.assertEqual(total_count, self._dataset.get_train_len())

  def test_test_data_length(self):
    """Tests length of test dataset."""
    total_count = 0
    for batch in self._dataset.get_test():
      total_count += len(batch['label'])

    self.assertEqual(total_count, self._dataset.get_test_len())

  def test_dataset_nonevenly_divisible_batch_size(self):
    """Tests non-evenly divisible test batch size."""
    with self.assertRaisesRegex(
        ValueError, 'Test data not evenly divisible by batch size: .*'):
      self._dataset = cifar10.CIFAR10Dataset(
          self._batch_size, batch_size_test=101)


if __name__ == '__main__':
  absltest.main()
