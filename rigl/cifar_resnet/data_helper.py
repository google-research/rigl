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

"""Helper functions for CIFAR10 data input pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf

import tensorflow_datasets as tfds

IMG_SIZE = 32


def pad_input(x, crop_dim=4):
  """Concatenates sides of image with pixels cropped from the border of image.

  Args:
    x: Input image float32 tensor.
    crop_dim: Number of pixels to crop from the edge of the image.
      Cropped pixels are then concatenated to the original image.
  Returns:
    x: input image float32 tensor. Transformed by padding edges with cropped
      pixels.
  """
  x = tf.concat(
      [x[:crop_dim, :, :][::-1], x, x[-crop_dim:, :, :][::-1]], axis=0)
  x = tf.concat(
      [x[:, :crop_dim, :][:, ::-1], x, x[:, -crop_dim:, :][:, ::-1]], axis=1)
  return x


def preprocess_train(x, width, height):
  """Pre-processing applied to training data set.

  Args:
    x: Input image float32 tensor.
    width: int specifying intended width in pixels of image after preprocessing.
    height: int specifying intended height in pixels of image after
      preprocessing.
  Returns:
    x: transformed input with random crops, flips and reflection.
  """
  x = pad_input(x, crop_dim=4)
  x = tf.random_crop(x, [width, height, 3])
  x = tf.image.random_flip_left_right(x)
  return x


def input_fn(params):
  """Provides batches of CIFAR data.

  Args:
    params: A dictionary with a set of arguments, namely:
      * batch_size (int32), specifies data points in a batch
      * data_split (string), designates train or eval
      * data_dictionary (string), specifies directory location of input dataset

  Returns:
    images: A float32`Tensor` of size [batch_size, 32, 32, 3].
    labels: A  int32`Tensor` of size [batch_size, num_classes].
  """

  def parse_serialized_example(record):
    """Parses a CIFAR10 example."""
    image = record['image']
    label = tf.cast(record['label'], tf.int32)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if data_split == 'train':
      image = preprocess_train(image, IMG_SIZE, IMG_SIZE)
    return image, label

  data_split = params['data_split']
  batch_size = params['batch_size']
  if data_split == 'eval':
    data_split = 'test'
  dataset = tfds.load('cifar10:3.*.*', split=data_split)

  # we only repeat an example and shuffle inputs during training
  if data_split == 'train':
    dataset = dataset.repeat().shuffle(buffer_size=50000)

  # deserialize record into tensors and apply pre-processing.
  dataset = dataset.map(parse_serialized_example).prefetch(batch_size)

  # at test time, for the final batch we drop remaining examples so that no
  # example is seen twice.
  dataset = dataset.batch(batch_size)

  images_batch, labels_batch = tf.data.make_one_shot_iterator(
      dataset).get_next()

  return (tf.reshape(images_batch, [batch_size, IMG_SIZE, IMG_SIZE, 3]),
          tf.reshape(labels_batch, [batch_size]))
