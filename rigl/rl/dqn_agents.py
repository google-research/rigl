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

"""Variants of DQN with sparsity."""

import functools
import math
from absl import logging
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
import gin
from rigl.rl import sparse_utils
import tensorflow as tf
import tensorflow.compat.v1 as tf1


# one of ('dense', 'prune', 'rigl', 'static', 'set'). If 'dense' no modification
# done. If 'prune', the agent is pruned after training.
# If ('rigl', 'static', 'set') the corresponding sparse-to-sparse training
# algorithm is used.
LEARNER_MODES = ('dense', 'prune', 'rigl', 'static', 'set')


def flatten_list_of_vars(var_list):
  flat_vars = [tf.reshape(v, [-1]) for v in var_list]
  return tf.concat(flat_vars, axis=-1)


def _get_bn_layer_name(block_id, i):
  return f'batch_norm_{block_id},{i}'


def _get_conv_layer_name(block_id, i):
  return f'conv_{block_id},{i}'


class _Stack(tf.keras.Model):
  """Stack of pooling and convolutional blocks with residual connections.
  """

  def __init__(self,
               num_ch,
               num_blocks,
               use_max_pooling=True,
               use_batch_norm=False,
               name='stack'):
    super(_Stack, self).__init__(name=name)
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same')
    self.use_max_pooling = use_max_pooling
    self.use_batch_norm = use_batch_norm
    self.num_blocks = num_blocks
    if self.use_batch_norm:
      self._batch_norm = tf.keras.layers.BatchNormalization()
    if self.use_max_pooling:
      self._max_pool = tf.keras.layers.MaxPool2D(
          pool_size=3, padding='same', strides=2)
    for block_id in range(num_blocks):
      for i in range(2):
        name = _get_conv_layer_name(block_id, i)
        layer = tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same',
            name=f'res_{block_id}/conv2d_{i}')
        setattr(self, name, layer)
        if self.use_batch_norm:
          name = _get_bn_layer_name(block_id, i)
          setattr(self, name, tf.keras.layers.BatchNormalization())

  def call(self, conv_out, training=False):
    # Downscale.
    conv_out = self._conv(conv_out)
    if self.use_max_pooling:
      conv_out = self._max_pool(conv_out)
    if self.use_batch_norm:
      conv_out = self._batch_norm(conv_out, training=training)

    # Residual block(s).
    for block_id in range(self.num_blocks):
      block_input = conv_out
      for i in range(2):
        conv_out = tf.nn.relu(conv_out)
        conv_layer = getattr(self, _get_conv_layer_name(block_id, i))
        conv_out = conv_layer(conv_out)
        if self.use_batch_norm:
          bn_layer = getattr(self, _get_bn_layer_name(block_id, i))
          conv_out = bn_layer(conv_out, training=training)
      conv_out += block_input
    return conv_out


@gin.configurable
class ImpalaNetwork(tf.keras.Model):
  """Agent with ResNet, but without LSTM and additional inputs.

  The deep model used for DQN which follows
  "IMPALA: Scalable Distributed Deep-RL with Importance Weighted
  Actor-Learner Architectures" by Espeholt, Soyer, Munos et al.

  Original implementation by Rishabh Agarwal, with minor modifications as
  follows:
  * rename nn_scale to width to fit with the sparserl API
  * allow for non-integer widths.
  * add training mode.
  * removed the option to have multiple heads.
  * modified the call function to return a compatible type.
  * added custom logic for sparse training.
  """

  def __init__(self,
               num_actions,
               width=1.0,
               mode='dense',
               name='impala_deep_network',
               prune_allow_key='',
               use_batch_norm=False):
    super().__init__(name=name)
    self._width = width
    self._mode = mode

    def _scale_width(n):
      return int(math.ceil(n * width))

    self.num_actions = num_actions
    self.use_batch_norm = use_batch_norm
    logging.info('Using batch norm in %s: %s', name, use_batch_norm)
    stack_fn = functools.partial(_Stack, use_batch_norm=use_batch_norm)
    # Parameters and layers for _torso.
    self._stacks = [
        stack_fn(_scale_width(32), 2, name='stack1'),
        stack_fn(_scale_width(64), 2, name='stack2'),
        stack_fn(_scale_width(64), 2, name='stack3'),
    ]
    self._dense1 = tf.keras.layers.Dense(_scale_width(256))
    self._dense2 = tf.keras.layers.Dense(
        self.num_actions, name='policy_logits')

    layer_shape_dict = {
        '_dense1': (7744, 512),
        '_dense2': (512, self.num_actions),
    }
    def add_stack_shapes(name, in_width, out_width):
      # First conv
      layer_shape_dict[f'{name}/_conv'] = (3, 3, in_width, out_width)
      for i in range(2):
        for j in range(2):
          l_name = _get_conv_layer_name(i, j)
          layer_shape_dict[f'{name}/{l_name}'] = (3, 3, out_width, out_width)
    add_stack_shapes('stack0', 4, _scale_width(32))
    add_stack_shapes('stack1', _scale_width(32), _scale_width(64))
    add_stack_shapes('stack2', _scale_width(64), _scale_width(64))

    if mode != 'dense':
      custom_sparsities = sparse_utils.get_pruning_sparsities(layer_shape_dict)
      for l_name, sparsity in custom_sparsities.items():
        logging.info('pruning, layer: %s, sparsity: %.4f', l_name, sparsity)
        if l_name.startswith('stack'):
          # stack1 -> 1
          stack_id = int(l_name[len('stack')])
          c_module = self._stacks[stack_id]
          # `stack1/_conv` -> `_conv`
          l_name = l_name.split('/')[1]
        else:
          c_module = self
        if mode == 'prune':
          if prune_allow_key and (prune_allow_key not in l_name):
            sparsity = 0
            logging.info('%s not pruned since, prune_allow_key: %s', l_name,
                         prune_allow_key)
          wrapped_layer = sparse_utils.maybe_prune_layer(
              getattr(c_module, l_name),
              params=sparse_utils.get_pruning_params(
                  mode, final_sparsity=sparsity))
        else:
          wrapped_layer = sparse_utils.maybe_prune_layer(
              getattr(c_module, l_name),
              params=sparse_utils.get_pruning_params(mode))
        setattr(c_module, l_name, wrapped_layer)

  def get_features(self, state, training=True):
    x = tf.cast(state, tf.float32)
    x /= 255
    conv_out = x
    for stack in self._stacks:
      conv_out = stack(conv_out, training=training)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    out = self._dense1(conv_out)
    out = tf.nn.relu(out)
    out = self._dense2(out)
    return out

  def call(self, state, training=True):
    out = self.get_features(state, training=training)
    return atari_lib.DQNNetworkType(out)


@gin.configurable
class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, width=1, mode='dense', name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      width: float, Scales the width of the network uniformly.
      mode: str, one of LEARNER_MODES.
      name: str, used to create scope for network parameters.
    """
    super().__init__(name=name)
    self.num_actions = num_actions
    self._width = width
    self._mode = mode

    def _scale_width(n):
      return int(math.ceil(n * width))
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(
        _scale_width(32), [8, 8],
        strides=4,
        padding='same',
        activation=activation_fn,
        name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        _scale_width(64), [4, 4],
        strides=2,
        padding='same',
        activation=activation_fn,
        name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        _scale_width(64), [3, 3],
        strides=1,
        padding='same',
        activation=activation_fn,
        name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        _scale_width(512), activation=activation_fn,
        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

    layer_shape_dict = {
        'conv1': (_scale_width(32), 8, 8, 4),
        'conv2': (_scale_width(64), 4, 4, _scale_width(32)),
        'conv3': (_scale_width(64), 3, 3, _scale_width(64)),
        'dense1': (7744, _scale_width(512)),
        'dense2': (_scale_width(512), num_actions)
    }
    if mode == 'dense':
      pass
    elif mode == 'prune':
      custom_sparsities = sparse_utils.get_pruning_sparsities(layer_shape_dict)
      for l_name, sparsity in custom_sparsities.items():
        logging.info('pruning, layer: %s, sparsity: %.4f', l_name, sparsity)
        wrapped_layer = sparse_utils.maybe_prune_layer(
            getattr(self, l_name),
            params=sparse_utils.get_pruning_params(
                mode, final_sparsity=sparsity))
        setattr(self, l_name, wrapped_layer)
    else:
      # static, rigl, set.
      for l_name in layer_shape_dict:
        wrapped_layer = sparse_utils.maybe_prune_layer(
            getattr(self, l_name),
            params=sparse_utils.get_pruning_params(mode))
        setattr(self, l_name, wrapped_layer)

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    return atari_lib.DQNNetworkType(self.dense2(x))


@gin.configurable
class SparseDQNAgent(dqn_agent.DQNAgent):
  """A variant of DQN that is trained with sparse backbones."""

  def __init__(self,
               sess,
               num_actions,
               mode='dense',
               weight_decay=0.,
               summary_writer=None):
    """Initializes the agent and constructs graph components.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      mode: str, one of LEARNER_MODES.
      weight_decay: float, used to regularize online_convnet.
      summary_writer: tf.SummaryWriter, for Tensorboard.
    """
    self._weight_decay = weight_decay
    if mode in LEARNER_MODES:
      self._mode = mode
    else:
      raise ValueError(f'mode:{mode} not one of {LEARNER_MODES}')
    self._global_step = tf1.train.get_or_create_global_step()
    # update_period=1, we always update as the supervisor is fixed.
    super().__init__(
        sess, num_actions, summary_writer=summary_writer)

  def _create_network(self, name):
    network = self.network(
        self.num_actions,
        name=name + 'learner',
        mode=self._mode)
    return network

  def _set_additional_ops(self):
    if self._mode == 'dense':
      self.step_update_op = tf.no_op()
      self.mask_update_op = tf.no_op()
      self.mask_init_op = tf.no_op()
    elif self._mode in ['rigl', 'set', 'static']:
      self.step_update_op = sparse_utils.update_prune_step(
          self.online_convnet, self._global_step)
      # This ensures sparse masks are applied before each run.
      self.mask_update_op = sparse_utils.update_prune_masks(self.online_convnet)
      self.mask_init_op = sparse_utils.init_masks(self.online_convnet)
      # Wrap the optimizer.
      if self._mode == 'rigl':
        self.optimizer = sparse_utils.UpdatedRigLOptimizer(self.optimizer)
        self.optimizer.set_model(self.online_convnet)
      elif self._mode == 'set':
        self.optimizer = sparse_utils.UpdatedSETOptimizer(self.optimizer)
        self.optimizer.set_model(self.online_convnet)
    elif self._mode == 'prune':
      self.step_update_op = sparse_utils.update_prune_step(
          self.online_convnet, self._global_step)
      self.mask_update_op = sparse_utils.update_prune_masks(self.online_convnet)
      self.mask_init_op = tf.no_op()
    else:
      raise ValueError(f'Invalid mode: {self._mode}')

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(loss)
    if self.summary_writer is not None:
      tf1.summary.scalar('Losses/HuberLoss', loss)

    reg_loss = 0.
    if self._weight_decay:
      for v in self.online_convnet.trainable_variables:
        if 'bias' not in v.name:
          reg_loss += tf.nn.l2_loss(v) * self._weight_decay
      loss += reg_loss
      tf1.summary.scalar('Losses/RegLoss', reg_loss)
    tf1.summary.scalar('Losses/TotalLoss', loss)
    sparse_utils.log_sparsities(self.online_convnet)
    self._set_additional_ops()
    grads_and_vars = self.optimizer.compute_gradients(loss)
    train_op = self.optimizer.apply_gradients(
        grads_and_vars, global_step=self._global_step)
    self._create_summary_ops(grads_and_vars)
    return train_op

  def _create_summary_ops(self, grads_and_vars):
    with tf1.variable_scope('Norm'):
      all_norm = tf.norm(
          flatten_list_of_vars(self.online_convnet.trainable_variables))
      tf1.summary.scalar('online_convnet/weights_norm', all_norm)
      all_norm = tf.norm(
          flatten_list_of_vars(self.target_convnet.trainable_variables))
      tf1.summary.scalar('target_convnet/weights_norm', all_norm)
      all_grad_norm = tf.norm(
          flatten_list_of_vars([
              g for g, v in grads_and_vars
              if v in self.online_convnet.trainable_variables
          ]))
      tf1.summary.scalar('online_convnet/grad_norm', all_grad_norm)

    total_params, nparam_dict = sparse_utils.get_total_params(
        self.online_convnet)
    tf1.summary.scalar('params/total', total_params)
    for k, val in nparam_dict.items():
      tf1.summary.scalar('params/' + k, val)

    if self._mode == 'rigl':
      tf1.summary.scalar('drop_fraction', self.optimizer.drop_fraction)

  def update_prune_step(self):
    self._sess.run(self.step_update_op)

  def maybe_update_and_apply_masks(self):
    self._sess.run(self.mask_update_op)

  def maybe_init_masks(self):
    # If `dense`; no initialization.
    self._sess.run(self.mask_init_op)

  def _train_step(self):
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self.update_prune_step()
        self.maybe_update_and_apply_masks()
        self._sess.run(self._train_op)
        c_step = self._sess.run(self._global_step)
        if (self.summary_writer is not None and
            self._merged_summaries is not None and
            c_step % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, c_step)
      if self.training_steps % self.target_update_period == 0:
        # Mask weights before syncing
        self.maybe_update_and_apply_masks()
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

  def _build_sync_op(self):
    """Builds ops for assigning weights from online to target network.

    Returns:
      ops: A list of ops assigning weights from online to target network.
    """
    # Get trainable variables from online and target DQNs
    sync_qt_ops = []
    online_vars = sparse_utils.get_all_variables_and_masks(self.online_convnet)
    target_vars = sparse_utils.get_all_variables_and_masks(self.target_convnet)
    for (v_online, v_target) in zip(online_vars, target_vars):
      # Assign weights from online to target network.
      sync_qt_ops.append(v_target.assign(v_online, use_locking=True))
    return sync_qt_ops

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    Same as the `super` class expect training=True flags are passed.
    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph, training=True)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                   training=True)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)
