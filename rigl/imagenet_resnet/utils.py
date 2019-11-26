# coding=utf-8
# Copyright 2019 RigL Authors.
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

"""Helped functions to concatenate subset of noisy images to batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import summary

IMG_SUMMARY_PREFIX = '_img_'


def format_tensors(*dicts):
  """Format metrics to be callable as tf.summary scalars on tpu's.

  Args:
    *dicts: A set of metric dictionaries, containing metric name + value tensor.

  Returns:
    A single formatted dictionary that holds all tensors.

  Raises:
   ValueError: if any tensor is not a scalar.
  """
  merged_summaries = {}
  for d in dicts:
    for metric_name, value in d.iteritems():
      shape = value.shape.as_list()
      if metric_name.startswith(IMG_SUMMARY_PREFIX):
        # If image, shape it into 2d.
        merged_summaries[metric_name] = tf.reshape(value,
                                                   (1, -1, value.shape[-1], 1))
      elif not shape:
        merged_summaries[metric_name] = tf.expand_dims(value, axis=0)
      elif shape == [1]:
        merged_summaries[metric_name] = value
      else:
        raise ValueError(
            'Metric {} has value {} that is not reconciliable'.format(
                metric_name, value))
  return merged_summaries


def host_call_fn(model_dir, **kwargs):
  """host_call function used for creating training summaries when using TPU.

  Args:
    model_dir: String indicating the output_dir to save summaries in.
    **kwargs: Set of metric names and tensor values for all desired summaries.

  Returns:
    Summary op to be passed to the host_call arg of the estimator function.
  """
  gs = kwargs.pop('global_step')[0]
  with summary.create_file_writer(model_dir).as_default():
    with summary.always_record_summaries():
      for name, tensor in kwargs.iteritems():
        if name.startswith(IMG_SUMMARY_PREFIX):
          summary.image(name.replace(IMG_SUMMARY_PREFIX, ''), tensor,
                        max_images=1)
        else:
          summary.scalar(name, tensor[0], step=gs)
      return summary.all_summary_ops()


def mask_summaries(masks, with_img=False):
  metrics = {}
  for mask in masks:
    metrics['pruning/{}/sparsity'.format(
        mask.op.name)] = tf.nn.zero_fraction(mask)
    if with_img:
      metrics[IMG_SUMMARY_PREFIX + 'mask/' + mask.op.name] = mask
  return metrics
