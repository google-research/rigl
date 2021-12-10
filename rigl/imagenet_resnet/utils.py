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

"""Helped functions to concatenate subset of noisy images to batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v2 import summary

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
    for metric_name, value in d.items():
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
    # Always record summaries.
    with summary.record_if(True):
      for name, tensor in kwargs.items():
        if name.startswith(IMG_SUMMARY_PREFIX):
          summary.image(name.replace(IMG_SUMMARY_PREFIX, ''), tensor,
                        max_images=1)
        else:
          summary.scalar(name, tensor[0], step=gs)
      # Following function is under tf:1x, so we use it.
      return tf.summary.all_v2_summary_ops()


def mask_summaries(masks, with_img=False):
  metrics = {}
  for mask in masks:
    metrics['pruning/{}/sparsity'.format(
        mask.op.name)] = tf.nn.zero_fraction(mask)
    if with_img:
      metrics[IMG_SUMMARY_PREFIX + 'mask/' + mask.op.name] = mask
  return metrics


def initialize_parameters_from_ckpt(ckpt_path, model_dir, param_suffixes):
  """Load parameters from an existing checkpoint.

  Args:
    ckpt_path: str, loads the mask variables from this checkpoint.
    model_dir: str, if checkpoint exists in this folder no-op.
    param_suffixes: list or str, suffix of parameters to be load from
      checkpoint.
  """
  already_has_ckpt = model_dir and tf.train.latest_checkpoint(
      model_dir) is not None
  if already_has_ckpt:
    tf.logging.info(
        'Training already started on this model, not loading masks from'
        'previously trained model')
    return

  reader = tf.train.NewCheckpointReader(ckpt_path)
  param_names = reader.get_variable_to_shape_map().keys()
  param_names = [x for x in param_names if x.endswith(param_suffixes)]

  variable_map = {}
  for var in tf.global_variables():
    var_name = var.name.split(':')[0]
    if var_name in param_names:
      tf.logging.info('Loading parameter variable from checkpoint: %s',
                      var_name)
      variable_map[var_name] = var
    elif var_name.endswith(param_suffixes):
      tf.logging.info(
          'Cannot find parameter variable in checkpoint, skipping: %s',
          var_name)
  tf.train.init_from_checkpoint(ckpt_path, variable_map)
