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

"""Tests for weight_symmetry.datasets.mnist."""
from absl.testing import absltest
import numpy as np

from rigl.experimental.jax.datasets import mnist


class MNISTDatasetTest(absltest.TestCase):
  """Test cases for MNIST Dataset."""

  def setUp(self):
    """Common setup routines/variables for test cases."""
    super().setUp()
    self._batch_size = 16
    self._batch_size_test = 10
    self._shuffle_buffer_size = 8

    self._dataset = mnist.MNISTDataset(
        self._batch_size,
        batch_size_test=self._batch_size_test,
        shuffle_buffer_size=self._shuffle_buffer_size)

  def test_create_dataset(self):
    """Tests creation of dataset."""
    self.assertIsInstance(self._dataset, mnist.MNISTDataset)

  def test_train_image_dims_content(self):
    """Tests dimensions and contents of test data."""
    iterator = self._dataset.get_train()
    sample = next(iterator)
    image, label = sample['image'], sample['label']

    with self.subTest(name='data_shape'):
      self.assertTupleEqual(image.shape, (self._batch_size, 28, 28, 1))

    with self.subTest(name='data_values'):
      self.assertTrue((image >= -1.).all() and (image <= 1.).all())

    with self.subTest(name='data_type'):
      self.assertTrue(np.issubdtype(image.dtype, float))

    with self.subTest(name='label_shape'):
      self.assertLen(label, self._batch_size)

    with self.subTest(name='label_type'):
      self.assertTrue(np.issubdtype(label.dtype, int))

    with self.subTest(name='label_values'):
      self.assertTrue((label >= 0).all() and
                      (label <= self._dataset.num_classes).all())

  def test_test_image_dims_content(self):
    """Tests dimensions and contents of train data."""
    iterator = self._dataset.get_test()
    sample = next(iterator)
    image, label = sample['image'], sample['label']

    with self.subTest(name='data_shape'):
      self.assertTupleEqual(image.shape, (self._batch_size_test, 28, 28, 1))

    with self.subTest(name='data_type'):
      self.assertTrue(np.issubdtype(image.dtype, float))

    # TODO: Find a better approach to testing with JAX arrays.
    with self.subTest(name='data_values'):
      self.assertTrue((image >= -1.).all() and (image <= 1.).all())

    with self.subTest(name='label_shape'):
      self.assertLen(label, self._batch_size_test)

    with self.subTest(name='label_type'):
      self.assertTrue(np.issubdtype(label.dtype, int))

    with self.subTest(name='label_values'):
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

    # Check image size/content.
    self.assertEqual(total_count, self._dataset.get_test_len())

if __name__ == '__main__':
  absltest.main()
