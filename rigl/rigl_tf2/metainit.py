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

"""MetaInit algorithm to dynamically initialize neural nets."""

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class ScaleSGD(tf1.train.Optimizer):
  """SGD optimizer that only trains the scales of the parameters.

  This optimizer only tunes the scales of weight matrices.
  """

  def __init__(self, learning_rate=0.1, momentum=0.9, mindim=3,
               use_locking=False, name="ScaleSGD"):
    super(ScaleSGD, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._momentum = momentum
    self._mindim = mindim

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._momentum_t = None

  def _prepare(self):
    self._lr_t = tf1.convert_to_tensor(self._lr, name="learning_rate")
    self._momentum_t = tf1.convert_to_tensor(self._momentum, name="momentum_t")

  def _create_slots(self, var_list):
    for v in var_list:
      self._get_or_make_slot_with_initializer(v,
                                              tf1.constant_initializer(0),
                                              tf1.TensorShape([]),
                                              tf1.float32,
                                              "m",
                                              self._name)

  def _resource_apply_dense(self, grad, handle):
    var = handle
    m = self.get_slot(var, "m")

    if len(var.shape) < self._mindim:
      return tf.group(*[var, m])
    lr_t = tf1.cast(self._lr_t, var.dtype.base_dtype)
    momentum_t = tf1.cast(self._momentum_t, var.dtype.base_dtype)

    scale = tf1.sqrt(tf1.reduce_sum(var ** 2))
    dscale = tf1.sign(tf1.reduce_sum(var * grad) / (scale + 1e-12))

    m_t = m.assign(momentum_t * m - lr_t * dscale)

    new_scale = scale + m_t
    var_update = tf1.assign(var, var * new_scale / (scale + 1e-12))
    return tf1.group(*[var_update, m_t])

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")


def meta_init(model, loss, x_shape, y_shape, n_params, learning_rate=0.001,
              momentum=0.9, meta_steps=1000, eps=1e-5, mask_gradient_fn=None):
  """Run MetaInit algorithm. See `https://papers.nips.cc/paper/9427-metainit-initializing-learning-by-learning-to-initialize`"""
  optimizer = ScaleSGD(learning_rate, momentum=momentum)

  for _ in range(meta_steps):
    x = np.random.normal(0, 1, x_shape)
    y = np.random.randint(0, y_shape[1], y_shape[0])

    with tf.GradientTape(persistent=True) as tape:
      batch_loss = loss(y, model(x, training=True))
      grad = tape.gradient(batch_loss, model.trainable_variables)
      if mask_gradient_fn is not None:
        grad = mask_gradient_fn(model, grad, model.trainable_variables)
      prod = tape.gradient(tf.reduce_sum([tf.reduce_sum(g**2) / 2
                                          for g in grad]),
                           model.trainable_variables)
      if mask_gradient_fn is not None:
        prod = mask_gradient_fn(model, prod, model.trainable_variables)
      meta_loss = [tf.abs(1 - ((g - p) / (g + eps * tf.stop_gradient(
          (2 * tf.cast(tf.greater_equal(g, 0), tf.float32)) - 1))))
                   for g, p in zip(grad, prod)]
      if mask_gradient_fn is not None:
        meta_loss = mask_gradient_fn(model, meta_loss,
                                     model.trainable_variables)
      meta_loss = sum([tf.reduce_sum(m) for m in meta_loss]) / n_params
    tf.summary.scalar("meta_loss", meta_loss)

    gradients = tape.gradient(meta_loss, model.trainable_variables)
    if mask_gradient_fn is not None:
      gradients = mask_gradient_fn(model, gradients, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
