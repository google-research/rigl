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

"""Tests for weight_symmetry.datasets.dataset_common."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from rigl.experimental.jax.datasets import dataset_base
from rigl.experimental.jax.datasets import dataset_factory


class DatasetCommonTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 32
    self._batch_size_test = 10
    self._shuffle_buffer_size = 128

  def _create_dataset(self, dataset_name):
    """Helper function for creating a dataset."""
    return dataset_factory.create_dataset(
        dataset_name,
        self._batch_size,
        self._batch_size_test,
        shuffle_buffer_size=self._shuffle_buffer_size)

  def test_dataset_supported(self):
    """Tests supported datasets."""
    for dataset_name in dataset_factory.DATASETS:
      dataset = self._create_dataset(dataset_name)

      self.assertIsInstance(dataset, dataset_base.Dataset)

  @parameterized.parameters(*dataset_factory.DATASETS.keys())
  def test_dataset_train_iterators(self, dataset_name):
    """Tests dataset's train iterator."""
    dataset = self._create_dataset(dataset_name)
    sample = next(dataset.get_train())

    with self.subTest(name='{}_sample'.format(dataset_name)):
      self.assertNotEmpty(sample)

    with self.subTest(name='{}_label_type'.format(dataset_name)):
      self.assertIsInstance(sample['label'], np.ndarray)

    with self.subTest(name='{}_label_batch_size'.format(dataset_name)):
      self.assertLen(sample['label'], self._batch_size)

    with self.subTest(name='{}_image_type'.format(dataset_name)):
      self.assertIsInstance(sample['image'], np.ndarray)

    with self.subTest(name='{}_image_shape'.format(dataset_name)):
      self.assertLen(sample['image'].shape, 4)

    with self.subTest(name='{}_image_batch_size'.format(dataset_name)):
      self.assertEqual(sample['image'].shape[0], self._batch_size)

    with self.subTest(
        name='{}_non_zero_image_dimensions'.format(dataset_name)):
      self.assertGreater(sample['image'].shape[1], 1)

  @parameterized.parameters(*dataset_factory.DATASETS.keys())
  def test_dataset_test_iterators(self, dataset_name):
    """Tests dataset's test iterator."""
    dataset = self._create_dataset(dataset_name)
    sample = next(dataset.get_test())

    with self.subTest(name='{}_sample'.format(dataset_name)):
      self.assertNotEmpty(sample)

    with self.subTest(name='{}_label_type'.format(dataset_name)):
      self.assertIsInstance(sample['label'], np.ndarray)

    with self.subTest(name='{}_label_batch_size'.format(dataset_name)):
      self.assertLen(sample['label'], self._batch_size_test)

    with self.subTest(name='{}_image_type'.format(dataset_name)):
      self.assertIsInstance(sample['image'], np.ndarray)

    with self.subTest(name='{}_image_shape'.format(dataset_name)):
      self.assertLen(sample['image'].shape, 4)

    with self.subTest(name='{}_image_batch_size'.format(dataset_name)):
      self.assertEqual(sample['image'].shape[0], self._batch_size_test)

    with self.subTest(
        name='{}_non_zero_image_dimensions'.format(dataset_name)):
      self.assertGreater(sample['image'].shape[1], 1)

  def test_dataset_unsupported(self):
    """Tests unsupported datasets."""
    with self.assertRaisesRegex(ValueError, 'No such dataset: unsupported'):
      self._create_dataset('unsupported')

if __name__ == '__main__':
  absltest.main()
