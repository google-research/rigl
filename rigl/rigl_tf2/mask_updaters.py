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

"""Implements RigL."""
import gin
from rigl.rigl_tf2 import utils
import tensorflow as tf


def get_all_layers(model, filter_fn=lambda _: True):
  """Gets all layers of a model and layers of a layer if it is a keras.Model."""
  all_layers = []
  for l in model.layers:
    if hasattr(l, 'layers'):
      all_layers.extend(get_all_layers(l, filter_fn=filter_fn))
    elif filter_fn(l):
      all_layers.append(l)
  return all_layers


def is_pruned(layer):
  return isinstance(layer, utils.PRUNING_WRAPPER) and layer.trainable


class MaskUpdater(object):
  """Base class for mask update algorithms.

    Attributes:
    model: tf.keras.Model
    optimizer: tf.train.Optimizer
    use_stateless: bool, if True stateless operations are used. This is
      important for multi-worker jobs not to diverge.
    stateless_seed_offset: int, added to the seed of stateless operations.
      Use this to create randomness without divergence across workers.
  """

  def __init__(self, model, optimizer, use_stateless=True,
               stateless_seed_offset=0, loss_fn=None):
    self._model = model
    self._optimizer = optimizer
    self._use_stateless = use_stateless
    self._stateless_seed_offset = stateless_seed_offset
    self._loss_fn = loss_fn
    self.val_x = self.val_y = None

  def prune_masks(self, prune_fraction):
    """Updates a fraction of weights in each layer."""
    all_masks, all_vars = self.get_vars_and_masks()
    drop_scores = self.get_drop_scores(all_vars, all_masks)
    grow_score = None
    for mask, var, drop_score in zip(all_masks, all_vars, drop_scores):
      self.generic_mask_update(mask, var, drop_score, grow_score,
                               prune_fraction)

  def update_masks(self, drop_fraction):
    """Updates a fraction of weights in each layer."""
    all_masks, all_vars = self.get_vars_and_masks()
    drop_scores = self.get_drop_scores(all_vars, all_masks)
    grow_scores = self.get_grow_scores(all_vars, all_masks)
    for mask, var, drop_score, grow_score in zip(all_masks, all_vars,
                                                 drop_scores, grow_scores):
      self.generic_mask_update(mask, var, drop_score, grow_score, drop_fraction)

  def get_all_pruning_layers(self):
    """Returns all pruned layers from the model."""
    if hasattr(self._model, 'layers'):
      return get_all_layers(self._model, filter_fn=is_pruned)
    else:
      return [self._model] if is_pruned(self._model) else []

  def get_vars_and_masks(self):
    """Gets all masked variables and corresponding masks."""
    all_masks = []
    all_vars = []
    for layer in self.get_all_pruning_layers():
      for var, mask, _ in layer.pruning_vars:
        all_vars.append(var)
        all_masks.append(mask)
    return all_masks, all_vars

  def get_drop_scores(self, all_vars, all_masks):
    raise NotImplementedError

  def get_grow_scores(self, all_vars, all_masks):
    raise NotImplementedError

  def generic_mask_update(self, mask, var, score_drop, score_grow,
                          drop_fraction, reinit_when_same=False):
    """Prunes+grows connections, all tensors same shape."""
    n_total = tf.size(score_drop)
    n_ones = tf.cast(tf.reduce_sum(mask), dtype=tf.int32)
    n_prune = tf.cast(
        tf.cast(n_ones, dtype=tf.float32) * drop_fraction, tf.int32)
    n_keep = n_ones - n_prune

    # Sort the entire array since the k needs to be constant for TPU.
    _, sorted_indices = tf.math.top_k(
        tf.reshape(score_drop, [-1]), k=n_total)
    sorted_indices_ex = tf.expand_dims(sorted_indices, 1)
    # We will have zeros after having `n_keep` many ones.
    new_values = tf.where(
        tf.range(n_total) < n_keep,
        tf.ones_like(sorted_indices, dtype=mask.dtype),
        tf.zeros_like(sorted_indices, dtype=mask.dtype))
    mask1 = tf.scatter_nd(sorted_indices_ex, new_values,
                          new_values.shape)
    if score_grow is not None:
      # Flatten the scores.
      score_grow = tf.reshape(score_grow, [-1])
      # Set scores of the enabled connections(ones) to min(s) - 1, so that they
      # have the lowest scores.
      score_grow_lifted = tf.where(
          tf.math.equal(mask1, 1),
          tf.ones_like(mask1) * (tf.reduce_min(score_grow) - 1), score_grow)
      _, sorted_indices = tf.math.top_k(score_grow_lifted, k=n_total)
      sorted_indices_ex = tf.expand_dims(sorted_indices, 1)
      new_values = tf.where(
          tf.range(n_total) < n_prune,
          tf.ones_like(sorted_indices, dtype=mask.dtype),
          tf.zeros_like(sorted_indices, dtype=mask.dtype))
      mask2 = tf.scatter_nd(sorted_indices_ex, new_values, new_values.shape)
      # Ensure masks are disjoint.
      tf.debugging.assert_near(tf.reduce_sum(mask1 * mask2), 0.)

      # Let's set the weights of the growed connections.
      mask2_reshaped = tf.reshape(mask2, mask.shape)
      # Set the values of the new connections.
      grow_tensor = tf.zeros_like(var, dtype=var.dtype)
      if reinit_when_same:
        # If dropped and grown, we re-initialize.
        new_connections = tf.math.equal(mask2_reshaped, 1)
      else:
        new_connections = tf.math.logical_and(
            tf.math.equal(mask2_reshaped, 1), tf.math.equal(mask, 0))
      new_weights = tf.where(new_connections, grow_tensor, var)
      var.assign(new_weights)
      # Ensure there is no momentum value for new connections
      self.reset_momentum(var, new_connections)
      mask_combined = tf.reshape(mask1 + mask2, mask.shape)
    else:
      mask_combined = tf.reshape(mask1, mask.shape)
    mask.assign(mask_combined)

  def reset_momentum(self, var, new_connections):
    for s_name in self._optimizer.get_slot_names():
      # Momentum variable for example, we reset the aggregated values to zero.
      optim_var = self._optimizer.get_slot(var, s_name)
      new_values = tf.where(new_connections,
                            tf.zeros_like(optim_var), optim_var)
      optim_var.assign(new_values)

  def _random_uniform(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = tf.cast(
          tf.stack([c_seed, self._optimizer.iterations]), tf.int32)
      return tf.random.stateless_uniform(*args, **kwargs)
    else:
      return tf.random.uniform(*args, **kwargs)

  def _random_normal(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = tf.cast(
          tf.stack([c_seed, self._optimizer.iterations]), tf.int32)
      return tf.random.stateless_normal(*args, **kwargs)
    else:
      return tf.random.normal(*args, **kwargs)

  def set_validation_data(self, val_x, val_y):
    self.val_x, self.val_y = val_x, val_y

  def _get_gradients(self, all_vars):
    """Returns the gradients of the given weights using the validation data."""
    with tf.GradientTape() as tape:
      batch_loss = self._loss_fn(self.val_x, self.val_y)
    grads = tape.gradient(batch_loss, all_vars)
    if grads:
      grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    return grads


class SET(MaskUpdater):
  """Implementation of dynamic sparsity optimizers.

  Implementation of SET.
  See https://www.nature.com/articles/s41467-018-04316-3
  This optimizer wraps a regular optimizer and performs updates on the masks
  according to schedule given.
  """

  def get_drop_scores(self, all_vars, all_masks, noise_std=0):
    def score_fn(mask, var):
      score = tf.math.abs(mask*var)
      if noise_std != 0:
        score += self._random_normal(
            score.shape, stddev=noise_std, dtype=score.dtype,
            seed=(hash(var.name + 'drop')))
      return score
    return [score_fn(mask, var) for mask, var in zip(all_masks, all_vars)]

  def get_grow_scores(self, all_vars, all_masks):
    return [self._random_uniform(var.shape, seed=hash(var.name + 'grow'))
            for var in all_vars]


class RigL(MaskUpdater):
  """Implementation of dynamic sparsity optimizers.

  Implementation of RigL.
  """

  def get_drop_scores(self, all_vars, all_masks, noise_std=0):
    def score_fn(mask, var):
      score = tf.math.abs(mask*var)
      if noise_std != 0:
        score += self._random_normal(
            score.shape, stddev=noise_std, dtype=score.dtype,
            seed=(hash(var.name + 'drop')))
      return score
    return [score_fn(mask, var) for mask, var in zip(all_masks, all_vars)]

  def get_grow_scores(self, all_vars, all_masks):
    return [tf.abs(g) for g in self._get_gradients(all_vars)]


class RigLInverted(RigL):
  """Implementation of dynamic sparsity optimizers.

  Implementation of RigL.
  """

  def get_grow_scores(self, all_vars, all_masks):
    return [-tf.abs(g) for g in self._get_gradients(all_vars)]




class UpdateSchedule(object):
  """Base class for mask update algorithms.

    Attributes:
    mask_updater: MaskUpdater, to invoke.
    update_freq: int, frequency of mask updates.
    init_drop_fraction: float, initial drop fraction.
  """

  def __init__(self, mask_updater, init_drop_fraction, update_freq,
               last_update_step):
    self._mask_updater = mask_updater
    self.update_freq = update_freq
    self.last_update_step = last_update_step
    self.init_drop_fraction = tf.convert_to_tensor(init_drop_fraction)
    self.last_drop_fraction = 0

  def get_drop_fraction(self, step):
    raise NotImplementedError

  def is_update_iter(self, step):
    """Returns true if it is a valid mask update step."""
    # last_update_step < 0 means, there is no last step.
    # last_update_step = 0 means, never update.
    tf.debugging.Assert(step >= 0, [step])

    if self.last_update_step < 0:
      is_valid_step = True
    elif self.last_update_step == 0:
      is_valid_step = False
    else:
      is_valid_step = step <= self.last_update_step

    return tf.logical_and(is_valid_step, step % self.update_freq == 0)

  def update(self, step, check_update_iter=True):
    if check_update_iter:
      tf.debugging.Assert(self.is_update_iter(step), [step])
    self.last_drop_fraction = self.get_drop_fraction(step)

    def true_fn():
      self._mask_updater.update_masks(self.last_drop_fraction)

    tf.cond(self.last_drop_fraction > 0., true_fn, lambda: None)

  def prune(self, prune_fraction):
    self.last_drop_fraction = prune_fraction
    self._mask_updater.prune_masks(self.last_drop_fraction)

  def set_validation_data(self, val_x, val_y):
    self._mask_updater.set_validation_data(val_x, val_y)


class ConstantUpdateSchedule(UpdateSchedule):
  """Updates a constant fraction of connections."""

  def get_drop_fraction(self, step):
    return self.init_drop_fraction


class CosineUpdateSchedule(UpdateSchedule):
  """Updates a constant fraction of connections."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._drop_fraction_fn = tf.keras.experimental.CosineDecay(
        self.init_drop_fraction,
        self.last_update_step,
        alpha=0.0,
        name='cosine_drop_fraction')

  def get_drop_fraction(self, step):
    return self._drop_fraction_fn(step)


class ScaledLRUpdateSchedule(UpdateSchedule):
  """Scales the drop fraction with learning rate."""

  def __init__(self, mask_updater, init_drop_fraction, update_freq,
               last_update_step, optimizer):
    self._optimizer = optimizer
    self._initial_lr = self._get_lr(0)
    super(ScaledLRUpdateSchedule, self).__init__(
        mask_updater, init_drop_fraction, update_freq, last_update_step)

  def _get_lr(self, step):
    if isinstance(self._optimizer.lr, tf.Variable):
      return self._optimizer.lr.numpy()
    else:
      return self._optimizer.lr(step)

  def get_drop_fraction(self, step):
    current_lr = self._get_lr(step)
    return (self.init_drop_fraction / self._initial_lr) * current_lr




@gin.configurable(
    'mask_updater',
    allowlist=[
        'update_alg',
        'schedule_alg',
        'update_freq',
        'init_drop_fraction',
        'last_update_step',
        'use_stateless',
    ])
def get_mask_updater(
    model,
    optimizer,
    loss_fn,
    update_alg='',
    schedule_alg='lr',
    update_freq=100,
    init_drop_fraction=0.3,
    last_update_step=-1,
    use_stateless=True):
  """Retrieves the update algorithm and passes it to the schedule object."""
  if not update_alg:
    return None
  elif update_alg == 'set':
    mask_updater = SET(model, optimizer, use_stateless=use_stateless)
  elif update_alg == 'rigl':
    mask_updater = RigL(
        model, optimizer, loss_fn=loss_fn, use_stateless=use_stateless)
  elif update_alg == 'rigl_inverted':
    mask_updater = RigLInverted(
        model, optimizer, loss_fn=loss_fn, use_stateless=use_stateless)
  else:
    raise ValueError('update_alg:%s  is not valid.' % update_alg)
  if schedule_alg == 'lr':
    update_schedule = ScaledLRUpdateSchedule(
        mask_updater, init_drop_fraction, update_freq, last_update_step,
        optimizer)
  elif schedule_alg == 'cosine':
    update_schedule = CosineUpdateSchedule(
        mask_updater, init_drop_fraction, update_freq, last_update_step)
  elif schedule_alg == 'constant':
    update_schedule = ConstantUpdateSchedule(mask_updater, init_drop_fraction,
                                             update_freq, last_update_step)
  else:
    raise ValueError('schedule_alg:%s  is not valid.' % schedule_alg)
  return update_schedule
