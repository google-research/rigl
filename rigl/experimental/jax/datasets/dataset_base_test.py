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

"""Tests for weight_symmetry.datasets.dataset_base."""
from absl.testing import absltest

from rigl.experimental.jax.datasets import dataset_base


class DummyDataset(dataset_base.ImageDataset):
  """A dummy implementation of the abstract dataset class.

  Attributes:
      NAME: The Tensorflow Dataset's dataset name.
  """
  NAME: str = 'mnist'

  def __init__(self,
               batch_size,
               batch_size_test,
               shuffle_buffer_size = 1024,
               seed = 42):
    """Dummy MNIST dataset.

    Args:
        batch_size: The batch size to use for the training datasets.
        batch_size_test: The batch size to used for the test dataset.
        shuffle_buffer_size: The buffer size to use for dataset shuffling.
        seed: The random seed used to shuffle.

    Returns:
        Dataset: A dataset object.
    """
    super().__init__(DummyDataset.NAME, batch_size, batch_size_test,
                     shuffle_buffer_size, seed)


class DummyDatasetTest(absltest.TestCase):
  """Test cases for dummy dataset."""

  def setUp(self):
    """Common setup routines/variables for test cases."""
    super().setUp()
    self._batch_size = 16
    self._batch_size_test = 10
    self._shuffle_buffer_size = 8
    self._dataset = DummyDataset(
        self._batch_size,
        batch_size_test=self._batch_size_test,
        shuffle_buffer_size=self._shuffle_buffer_size)

  def test_create_dataset(self):
    """Tests creation of dataset."""
    self.assertIsInstance(self._dataset, DummyDataset)

  def test_train_image_dims_content(self):
    """Tests dimensions and contents of test data."""
    iterator = iter(self._dataset.get_train())
    sample = next(iterator)
    image, label = sample['image'], sample['label']

    with self.subTest(name='data_shape'):
      self.assertTupleEqual(image.shape, (self._batch_size, 28, 28, 1))

    with self.subTest(name='data_values'):
      self.assertBetween(image.all(), 0, 256)

    with self.subTest(name='label_shape'):
      self.assertLen(label, self._batch_size)

    with self.subTest(name='label_values'):
      self.assertBetween(label.all(), 0, self._dataset.num_classes)

  def test_test_image_dims_content(self):
    """Tests dimensions and contents of train data."""
    iterator = iter(self._dataset.get_test())
    sample = next(iterator)
    image, label = sample['image'], sample['label']

    with self.subTest(name='data_shape'):
      self.assertTupleEqual(image.shape, (self._batch_size_test, 28, 28, 1))

    with self.subTest(name='data_values'):
      self.assertBetween(image.all(), 0, 256)

    with self.subTest(name='label_shape'):
      self.assertLen(label, self._batch_size_test)

    with self.subTest(name='label_values'):
      self.assertBetween(label.all(), 0, self._dataset.num_classes)

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
