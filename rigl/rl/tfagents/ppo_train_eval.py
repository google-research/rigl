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

r"""Sparse training PPO using actor/learner in a gym environment.
"""

import collections
import functools
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import reverb
from rigl.rigl_tf2 import mask_updaters
from rigl.rl import sparse_utils
from rigl.rl.tfagents import sparse_ppo_actor_network
from rigl.rl.tfagents import sparse_ppo_discrete_actor_network
from rigl.rl.tfagents import sparse_value_network
from rigl.rl.tfagents import tf_sparse_utils
import tensorflow.compat.v2 as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.ppo import ppo_utils
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_mujoco
from tf_agents.metrics import py_metrics
from tf_agents.networks import network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import ppo_learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "train_eval.env_name=Acrobot-v1",'
    '      "init_masks.sparsity=0.9").')

# Env params
flags.DEFINE_bool('is_atari', False, 'Whether the env is an atari game.')
flags.DEFINE_bool('is_mujoco', False, 'Whether the env is a mujoco game.')
flags.DEFINE_bool('is_classic', False,
                  'Whether the env is a classic control game.')
flags.DEFINE_float(
    'average_last_fraction', 0.1,
    'Tells what fraction latest evaluation scores are averaged. This is used'
    ' to reduce variance.')

SparsePPOLossInfo = collections.namedtuple('SparsePPOLossInfo', (
    'policy_gradient_loss',
    'value_estimation_loss',
    'l2_regularization_loss',
    'entropy_regularization_loss',
    'kl_penalty_loss',
    'total_loss_per_sample',
))


def _normalize_advantages(advantages, axes=(0,), variance_epsilon=1e-8):
  adv_mean, adv_var = tf.nn.moments(advantages, axes=axes, keepdims=True)
  normalized_advantages = tf.nn.batch_normalization(
      advantages,
      adv_mean,
      adv_var,
      offset=None,
      scale=None,
      variance_epsilon=variance_epsilon)
  return normalized_advantages


@gin.configurable
class SparsePPOAgent(ppo_clip_agent.PPOClipAgent):
  """Wrapped PPOClipAgent that supports sparse training."""

  def __init__(self,
               *args,
               policy_l2_reg=0.0,
               value_function_l2_reg=0.0,
               shared_vars_l2_reg=0.0,
               **kwargs):
    super().__init__(*args,
                     policy_l2_reg=policy_l2_reg,
                     value_function_l2_reg=value_function_l2_reg,
                     shared_vars_l2_reg=shared_vars_l2_reg,
                     **kwargs)
    # Name scoping has been removed here so
    # debug_summaries are permenantly disabled. To restore with proper
    # scoping.
    self._debug_summaries = False
    # Pruning layer requires the pruning_step to be >1 during forward pass.
    tf_sparse_utils.update_prune_step(
        self._actor_net, self.train_step_counter + 1)
    tf_sparse_utils.update_prune_step(
        self._value_net, self.train_step_counter + 1)
    _ = sparse_utils.init_masks(self._actor_net)
    _ = sparse_utils.init_masks(self._value_net)

    # BEGIN: sparse training create mask updaters
    def loss_fn(experience_data, weights_data):
      # The following is just to fit to the existing API.
      (time_steps, actions, old_act_log_probs, returns, normalized_advantages,
       old_action_distribution_parameters, masked_weights,
       old_value_predictions) = self._process_experience_weights(
           experience_data, weights_data)
      loss_info = self.get_loss(
          time_steps,
          actions,
          old_act_log_probs,
          returns,
          normalized_advantages,
          old_action_distribution_parameters,
          masked_weights,
          self.train_step_counter,
          False,
          old_value_predictions=old_value_predictions,
          training=True)
      return loss_info.extra.total_loss_per_sample
    self._mask_updater_actor = mask_updaters.get_mask_updater(
        self._actor_net, self._optimizer, loss_fn)
    self._mask_updater_value = mask_updaters.get_mask_updater(
        self._value_net, self._optimizer, loss_fn)
    # END: sparse training create mask updaters
    logging.info('SparsePPOAgent: policy_l2_reg %.5f.', policy_l2_reg)
    logging.info('SparsePPOAgent: value_function_l2_reg %.5f.',
                 value_function_l2_reg)
    logging.info('SparsePPOAgent: shared_vars_l2_reg %.5f.', shared_vars_l2_reg)

  def _process_experience_weights(self, experience, weights):
    experience = self._as_trajectory(experience)

    if self._compute_value_and_advantage_in_train:
      processed_experience = self._preprocess(experience)
    else:
      processed_experience = experience

    # Mask trajectories that cannot be used for training.
    valid_mask = ppo_utils.make_trajectory_mask(processed_experience)
    if weights is None:
      masked_weights = valid_mask
    else:
      masked_weights = weights * valid_mask

    # Reconstruct per-timestep policy distribution from stored distribution
    #   parameters.
    old_action_distribution_parameters = processed_experience.policy_info[
        'dist_params']

    old_actions_distribution = (
        ppo_utils.distribution_from_spec(
            self._action_distribution_spec,
            old_action_distribution_parameters,
            legacy_distribution_network=isinstance(
                self._actor_net, network.DistributionNetwork)))

    # Compute log probability of actions taken during data collection, using the
    #   collect policy distribution.
    old_act_log_probs = common.log_probability(old_actions_distribution,
                                               processed_experience.action,
                                               self._action_spec)

    if self._debug_summaries and not tf.config.list_logical_devices('TPU'):
      actions_list = tf.nest.flatten(processed_experience.action)
      show_action_index = len(actions_list) != 1
      for i, single_action in enumerate(actions_list):
        action_name = ('actions_{}'.format(i)
                       if show_action_index else 'actions')
        tf.compat.v2.summary.histogram(
            name=action_name, data=single_action, step=self.train_step_counter)

    time_steps = ts.TimeStep(
        step_type=processed_experience.step_type,
        reward=processed_experience.reward,
        discount=processed_experience.discount,
        observation=processed_experience.observation)
    actions = processed_experience.action
    returns = processed_experience.policy_info['return']
    advantages = processed_experience.policy_info['advantage']

    normalized_advantages = _normalize_advantages(advantages,
                                                  variance_epsilon=1e-8)

    if self._debug_summaries and not tf.config.list_logical_devices('TPU'):
      tf.compat.v2.summary.histogram(
          name='advantages_normalized',
          data=normalized_advantages,
          step=self.train_step_counter)
    old_value_predictions = processed_experience.policy_info['value_prediction']

    return (time_steps, actions, old_act_log_probs, returns,
            normalized_advantages, old_action_distribution_parameters,
            masked_weights, old_value_predictions)

  def _train(self, experience, weights):
    tf.compat.v2.summary.experimental.set_step(self.train_step_counter)

    (time_steps, actions, old_act_log_probs, returns, normalized_advantages,
     old_action_distribution_parameters, masked_weights,
     old_value_predictions) = self._process_experience_weights(
         experience, weights)

    if self._compute_value_and_advantage_in_train:
      processed_experience = self._preprocess(experience)
    else:
      processed_experience = experience

    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    # Loss tensors across batches will be aggregated for summaries.
    policy_gradient_losses = []
    value_estimation_losses = []
    l2_regularization_losses = []
    entropy_regularization_losses = []
    kl_penalty_losses = []

    loss_info = None
    variables_to_train = list(
        object_identity.ObjectIdentitySet(self._actor_net.trainable_weights +
                                          self._value_net.trainable_weights))
    # Sort to ensure tensors on different processes end up in same order.
    variables_to_train = sorted(variables_to_train, key=lambda x: x.name)

    for _ in range(self._num_epochs):
      # Name scoping has been removed here so
      # debug_summaries are permenantly disabled. To restore with proper
      # scoping.
      debug_summaries = False

      with tf.GradientTape(persistent=True) as tape:
        loss_info = self.get_loss(
            time_steps,
            actions,
            old_act_log_probs,
            returns,
            normalized_advantages,
            old_action_distribution_parameters,
            masked_weights,
            self.train_step_counter,
            debug_summaries,
            old_value_predictions=old_value_predictions,
            training=True)

      grads = tape.gradient(loss_info.loss, variables_to_train)

      tf_sparse_utils.log_snr(tape, loss_info.extra.total_loss_per_sample,
                              self.train_step_counter, variables_to_train)

      # BEGIN sparse training mask update
      # We use the lastest set of gradients to update the masks for sparse
      # training. Note, we do this before gradient clipping.
      def _mask_update_step(mask_updater, updater_name):
        mask_updater.set_validation_data(experience, weights)
        mask_updater.update(self.train_step_counter)
        with tf.name_scope('Drop_fraction/'):
          tf.summary.scalar(
              name=f'{updater_name}',
              data=mask_updater.last_drop_fraction)

      mask_update_step_actor = functools.partial(
          _mask_update_step, self._mask_updater_actor, 'actor')
      mask_update_step_value = functools.partial(
          _mask_update_step, self._mask_updater_value, 'value')

      tf_sparse_utils.log_sparsities(self._actor_net, 'actor')
      tf_sparse_utils.log_sparsities(self._value_net, 'value')
      tf_sparse_utils.log_total_params([self._actor_net, self._value_net])
      if self._mask_updater_actor is not None:
        is_update_actor = self._mask_updater_actor.is_update_iter(
            self.train_step_counter)

        tf.cond(is_update_actor, mask_update_step_actor, lambda: None)

      if self._mask_updater_value is not None:
        is_update_value = self._mask_updater_value.is_update_iter(
            self.train_step_counter)

        tf.cond(is_update_value, mask_update_step_value, lambda: None)
      # END sparse training mask update

      if self._gradient_clipping > 0:
        grads, _ = tf.clip_by_global_norm(grads, self._gradient_clipping)

      # Tuple is used for py3, where zip is a generator producing values once.
      grads_and_vars = tuple(zip(grads, variables_to_train))

      # If summarize_gradients, create functions for summarizing both
      # gradients and variables.
      if self._summarize_grads_and_vars and debug_summaries:
        eager_utils.add_gradients_summaries(grads_and_vars,
                                            self.train_step_counter)
        eager_utils.add_variables_summaries(grads_and_vars,
                                            self.train_step_counter)

      self._optimizer.apply_gradients(grads_and_vars)
      self.train_step_counter.assign_add(1)

      policy_gradient_losses.append(loss_info.extra.policy_gradient_loss)
      value_estimation_losses.append(loss_info.extra.value_estimation_loss)
      l2_regularization_losses.append(loss_info.extra.l2_regularization_loss)
      entropy_regularization_losses.append(
          loss_info.extra.entropy_regularization_loss)
      kl_penalty_losses.append(loss_info.extra.kl_penalty_loss)

    if self._initial_adaptive_kl_beta > 0:
      # After update epochs, update adaptive kl beta, then update observation
      #   normalizer and reward normalizer.
      policy_state = self._collect_policy.get_initial_state(batch_size)
      # Compute the mean kl from previous action distribution.
      kl_divergence = self._kl_divergence(
          time_steps, old_action_distribution_parameters,
          self._collect_policy.distribution(time_steps, policy_state).action)
      self.update_adaptive_kl_beta(kl_divergence)

    if self.update_normalizers_in_train:
      self.update_observation_normalizer(time_steps.observation)
      self.update_reward_normalizer(processed_experience.reward)

    loss_info = tf.nest.map_structure(tf.identity, loss_info)

    # Make summaries for total loss averaged across all epochs.
    # The *_losses lists will have been populated by
    #   calls to self.get_loss. Assumes all the losses have same length.
    with tf.name_scope('Losses/'):
      num_epochs = len(policy_gradient_losses)
      total_policy_gradient_loss = tf.add_n(policy_gradient_losses) / num_epochs
      total_value_estimation_loss = tf.add_n(
          value_estimation_losses) / num_epochs
      total_l2_regularization_loss = tf.add_n(
          l2_regularization_losses) / num_epochs
      total_entropy_regularization_loss = tf.add_n(
          entropy_regularization_losses) / num_epochs
      total_kl_penalty_loss = tf.add_n(kl_penalty_losses) / num_epochs
      tf.compat.v2.summary.scalar(
          name='policy_gradient_loss',
          data=total_policy_gradient_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=total_value_estimation_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='l2_regularization_loss',
          data=total_l2_regularization_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='entropy_regularization_loss',
          data=total_entropy_regularization_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='kl_penalty_loss',
          data=total_kl_penalty_loss,
          step=self.train_step_counter)

      total_abs_loss = (
          tf.abs(total_policy_gradient_loss) +
          tf.abs(total_value_estimation_loss) +
          tf.abs(total_entropy_regularization_loss) +
          tf.abs(total_l2_regularization_loss) + tf.abs(total_kl_penalty_loss))

      tf.compat.v2.summary.scalar(
          name='total_abs_loss',
          data=total_abs_loss,
          step=self.train_step_counter)

    with tf.name_scope('LearningRate/'):
      learning_rate = ppo_utils.get_learning_rate(self._optimizer)
      tf.compat.v2.summary.scalar(
          name='learning_rate',
          data=learning_rate,
          step=self.train_step_counter)

    if self._summarize_grads_and_vars and not tf.config.list_logical_devices(
        'TPU'):
      with tf.name_scope('Variables/'):
        all_vars = (
            self._actor_net.trainable_weights +
            self._value_net.trainable_weights)
        for var in all_vars:
          tf.compat.v2.summary.histogram(
              name=var.name.replace(':', '_'),
              data=var,
              step=self.train_step_counter)

    return loss_info

  def get_loss(self,
               time_steps,
               actions,
               act_log_probs,
               returns,
               normalized_advantages,
               action_distribution_parameters,
               weights,
               train_step,
               debug_summaries,
               old_value_predictions = None,
               training = False):
    """Compute the loss and create optimization op for one training epoch.

    All tensors should have a single batch dimension.

    Args:
      time_steps: A minibatch of TimeStep tuples.
      actions: A minibatch of actions.
      act_log_probs: A minibatch of action probabilities (probability under the
        sampling policy).
      returns: A minibatch of per-timestep returns.
      normalized_advantages: A minibatch of normalized per-timestep advantages.
      action_distribution_parameters: Parameters of data-collecting action
        distribution. Needed for KL computation.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      train_step: A train_step variable to increment for each train step.
        Typically the global_step.
      debug_summaries: True if debug summaries should be created.
      old_value_predictions: (Optional) The saved value predictions, used for
        calculating the value estimation loss when value clipping is performed.
      training: Whether this loss is being used for training.

    Returns:
      A tf_agent.LossInfo named tuple with the total_loss and all intermediate
        losses in the extra field contained in a PPOLossInfo named tuple.
    """
    # Evaluate the current policy on timesteps.

    # batch_size from time_steps
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._collect_policy.get_initial_state(batch_size)
    # We must use _distribution because the distribution API doesn't pass down
    # the training= kwarg.
    distribution_step = self._collect_policy._distribution(
        time_steps,
        policy_state,
        training=training)
    current_policy_distribution = distribution_step.action

    # Call all loss functions and add all loss values.
    (value_estimation_loss,
     value_estimation_loss_per_sample) = self.value_estimation_loss(
         time_steps=time_steps,
         returns=returns,
         old_value_predictions=old_value_predictions,
         weights=weights,
         debug_summaries=debug_summaries,
         training=training)
    (policy_gradient_loss,
     policy_gradient_loss_per_sample) = self.policy_gradient_loss(
         time_steps,
         actions,
         tf.stop_gradient(act_log_probs),
         tf.stop_gradient(normalized_advantages),
         current_policy_distribution,
         weights,
         debug_summaries=debug_summaries)

    if (self._policy_l2_reg > 0.0 or self._value_function_l2_reg > 0.0 or
        self._shared_vars_l2_reg > 0.0):
      l2_regularization_loss = self.l2_regularization_loss(debug_summaries)
    else:
      l2_regularization_loss = tf.zeros_like(policy_gradient_loss)
    l2_regularization_loss_per_sample = tf.repeat(
        l2_regularization_loss / tf.cast(batch_size, tf.float32), batch_size)

    if self._entropy_regularization > 0.0:
      (entropy_regularization_loss, entropy_regularization_loss_per_sample
      ) = self.entropy_regularization_loss(time_steps,
                                           current_policy_distribution, weights,
                                           debug_summaries)
    else:
      entropy_regularization_loss = tf.zeros_like(policy_gradient_loss)
      entropy_regularization_loss_per_sample = tf.repeat(
          tf.constant(0, dtype=tf.float32), batch_size)

    if self._initial_adaptive_kl_beta == 0:
      kl_penalty_loss = tf.zeros_like(policy_gradient_loss)
    else:
      kl_penalty_loss = self.kl_penalty_loss(time_steps,
                                             action_distribution_parameters,
                                             current_policy_distribution,
                                             weights, debug_summaries)
    kl_penalty_loss_per_sample = tf.repeat(
        kl_penalty_loss / tf.cast(batch_size, tf.float32), batch_size)

    total_loss = (
        policy_gradient_loss + value_estimation_loss + l2_regularization_loss +
        entropy_regularization_loss + kl_penalty_loss)
    total_loss_per_sample = (
        policy_gradient_loss_per_sample + value_estimation_loss_per_sample +
        l2_regularization_loss_per_sample +
        entropy_regularization_loss_per_sample + kl_penalty_loss_per_sample)

    return tf_agent.LossInfo(
        total_loss,
        SparsePPOLossInfo(
            policy_gradient_loss=policy_gradient_loss,
            value_estimation_loss=value_estimation_loss,
            l2_regularization_loss=l2_regularization_loss,
            entropy_regularization_loss=entropy_regularization_loss,
            kl_penalty_loss=kl_penalty_loss,
            total_loss_per_sample=total_loss_per_sample
            ))

  def value_estimation_loss(self,
                            time_steps,
                            returns,
                            weights,
                            old_value_predictions = None,
                            debug_summaries = False,
                            training = False):
    """Computes the value estimation loss for actor-critic training.

    All tensors should have a single batch dimension.

    Args:
      time_steps: A batch of timesteps.
      returns: Per-timestep returns for value function to predict. (Should come
        from TD-lambda computation.)
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      old_value_predictions: (Optional) The saved value predictions from
        policy_info, required when self._value_clipping > 0.
      debug_summaries: True if debug summaries should be created.
      training: Whether this loss is going to be used for training.

    Returns:
      value_estimation_loss: A scalar value_estimation_loss loss.

    Raises:
      ValueError: If old_value_predictions was not passed in, but value clipping
        was performed.
    """
    observation = time_steps.observation
    if debug_summaries and not tf.config.list_logical_devices('TPU'):
      observation_list = tf.nest.flatten(observation)
      show_observation_index = len(observation_list) != 1
      for i, single_observation in enumerate(observation_list):
        observation_name = ('observations_{}'.format(i)
                            if show_observation_index else 'observations')
        tf.compat.v2.summary.histogram(
            name=observation_name,
            data=single_observation,
            step=self.train_step_counter)

    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    value_state = self._collect_policy.get_initial_value_state(batch_size)

    value_preds, _ = self._collect_policy.apply_value_network(
        time_steps.observation,
        time_steps.step_type,
        value_state=value_state,
        training=training)
    value_estimation_error = tf.math.squared_difference(returns, value_preds)

    if self._value_clipping > 0:
      if old_value_predictions is None:
        raise ValueError(
            'old_value_predictions is None but needed for value clipping.')
      clipped_value_preds = old_value_predictions + tf.clip_by_value(
          value_preds - old_value_predictions, -self._value_clipping,
          self._value_clipping)
      clipped_value_estimation_error = tf.math.squared_difference(
          returns, clipped_value_preds)
      value_estimation_error = tf.maximum(value_estimation_error,
                                          clipped_value_estimation_error)

    if self._aggregate_losses_across_replicas:
      value_estimation_loss = (
          common.aggregate_losses(
              per_example_loss=value_estimation_error,
              sample_weight=weights).total_loss * self._value_pred_loss_coef)
    else:
      value_estimation_loss = tf.math.reduce_mean(
          value_estimation_error * weights) * self._value_pred_loss_coef

    value_estimation_loss_per_sample = tf.reduce_mean(value_estimation_error,
                                                      axis=0)
    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='value_pred_avg',
          data=tf.reduce_mean(input_tensor=value_preds),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_actual_avg',
          data=tf.reduce_mean(input_tensor=returns),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=value_estimation_loss,
          step=self.train_step_counter)
      if not tf.config.list_logical_devices('TPU'):
        tf.compat.v2.summary.histogram(
            name='value_preds', data=value_preds, step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='value_estimation_error',
            data=value_estimation_error,
            step=self.train_step_counter)

    if self._check_numerics:
      value_estimation_loss = tf.debugging.check_numerics(
          value_estimation_loss, 'value_estimation_loss')
      value_estimation_loss_per_sample = tf.debugging.check_numerics(
          value_estimation_loss_per_sample, 'value_estimation_loss_per_sample')

    return value_estimation_loss, value_estimation_loss_per_sample

  def policy_gradient_loss(
      self,
      time_steps,
      actions,
      sample_action_log_probs,
      advantages,
      current_policy_distribution,
      weights,
      debug_summaries = False):
    """Create tensor for policy gradient loss.

    All tensors should have a single batch dimension.

    Args:
      time_steps: TimeSteps with observations for each timestep.
      actions: Tensor of actions for timesteps, aligned on index.
      sample_action_log_probs: Tensor of sample probability of each action.
      advantages: Tensor of advantage estimate for each timestep, aligned on
        index. Works better when advantage estimates are normalized.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.

    Returns:
      policy_gradient_loss: A tensor that will contain policy gradient loss for
        the on-policy experience.
    """
    nest_utils.assert_same_structure(time_steps, self.time_step_spec)
    action_log_prob = common.log_probability(current_policy_distribution,
                                             actions, self._action_spec)
    action_log_prob = tf.cast(action_log_prob, tf.float32)
    if self._log_prob_clipping > 0.0:
      action_log_prob = tf.clip_by_value(action_log_prob,
                                         -self._log_prob_clipping,
                                         self._log_prob_clipping)
    if self._check_numerics:
      action_log_prob = tf.debugging.check_numerics(action_log_prob,
                                                    'action_log_prob')

    # Prepare both clipped and unclipped importance ratios.
    importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
    importance_ratio_clipped = tf.clip_by_value(
        importance_ratio, 1 - self._importance_ratio_clipping,
        1 + self._importance_ratio_clipping)

    if self._check_numerics:
      importance_ratio = tf.debugging.check_numerics(importance_ratio,
                                                     'importance_ratio')
      if self._importance_ratio_clipping > 0.0:
        importance_ratio_clipped = tf.debugging.check_numerics(
            importance_ratio_clipped, 'importance_ratio_clipped')

    # Pessimistically choose the minimum objective value for clipped and
    #   unclipped importance ratios.
    per_timestep_objective = importance_ratio * advantages
    per_timestep_objective_clipped = importance_ratio_clipped * advantages
    per_timestep_objective_min = tf.minimum(per_timestep_objective,
                                            per_timestep_objective_clipped)

    if self._importance_ratio_clipping > 0.0:
      policy_gradient_loss = -per_timestep_objective_min
    else:
      policy_gradient_loss = -per_timestep_objective

    policy_gradient_loss_per_sample = tf.reduce_mean(policy_gradient_loss,
                                                     axis=0)

    if self._aggregate_losses_across_replicas:
      policy_gradient_loss = common.aggregate_losses(
          per_example_loss=policy_gradient_loss,
          sample_weight=weights).total_loss
    else:
      policy_gradient_loss = tf.math.reduce_mean(policy_gradient_loss * weights)

    if debug_summaries:
      if self._importance_ratio_clipping > 0.0:
        clip_fraction = tf.reduce_mean(
            input_tensor=tf.cast(
                tf.greater(
                    tf.abs(importance_ratio -
                           1.0), self._importance_ratio_clipping), tf.float32))
        tf.compat.v2.summary.scalar(
            name='clip_fraction',
            data=clip_fraction,
            step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='importance_ratio_mean',
          data=tf.reduce_mean(input_tensor=importance_ratio),
          step=self.train_step_counter)
      entropy = common.entropy(current_policy_distribution, self.action_spec)
      tf.compat.v2.summary.scalar(
          name='policy_entropy_mean',
          data=tf.reduce_mean(input_tensor=entropy),
          step=self.train_step_counter)
      if not tf.config.list_logical_devices('TPU'):
        tf.compat.v2.summary.histogram(
            name='action_log_prob',
            data=action_log_prob,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='action_log_prob_sample',
            data=sample_action_log_probs,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='importance_ratio',
            data=importance_ratio,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='importance_ratio_clipped',
            data=importance_ratio_clipped,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='per_timestep_objective',
            data=per_timestep_objective,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='per_timestep_objective_clipped',
            data=per_timestep_objective_clipped,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='per_timestep_objective_min',
            data=per_timestep_objective_min,
            step=self.train_step_counter)

        tf.compat.v2.summary.histogram(
            name='policy_entropy', data=entropy, step=self.train_step_counter)
        for i, (single_action, single_distribution) in enumerate(
            zip(
                tf.nest.flatten(self.action_spec),
                tf.nest.flatten(current_policy_distribution))):
          # Categorical distribution (used for discrete actions) doesn't have a
          # mean.
          distribution_index = '_{}'.format(i) if i > 0 else ''
          if not tensor_spec.is_discrete(single_action):
            tf.compat.v2.summary.histogram(
                name='actions_distribution_mean' + distribution_index,
                data=single_distribution.mean(),
                step=self.train_step_counter)
            tf.compat.v2.summary.histogram(
                name='actions_distribution_stddev' + distribution_index,
                data=single_distribution.stddev(),
                step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='policy_gradient_loss',
            data=policy_gradient_loss,
            step=self.train_step_counter)

    if self._check_numerics:
      policy_gradient_loss = tf.debugging.check_numerics(
          policy_gradient_loss, 'policy_gradient_loss')
      policy_gradient_loss_per_sample = tf.debugging.check_numerics(
          policy_gradient_loss_per_sample, 'policy_gradient_loss_per_sample')

    return policy_gradient_loss, policy_gradient_loss_per_sample

  def entropy_regularization_loss(
      self,
      time_steps,
      current_policy_distribution,
      weights,
      debug_summaries = False):
    """Create regularization loss tensor based on agent parameters."""
    if self._entropy_regularization > 0:
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)
      with tf.name_scope('entropy_regularization'):
        entropy = tf.cast(
            common.entropy(current_policy_distribution, self.action_spec),
            tf.float32)

        if self._aggregate_losses_across_replicas:
          entropy_reg_loss = common.aggregate_losses(
              per_example_loss=-entropy,
              sample_weight=weights).total_loss * self._entropy_regularization
        else:
          entropy_reg_loss = (
              tf.math.reduce_mean(-entropy * weights) *
              self._entropy_regularization)

        if self._check_numerics:
          entropy_reg_loss = tf.debugging.check_numerics(
              entropy_reg_loss, 'entropy_reg_loss')

        if debug_summaries and not tf.config.list_logical_devices('TPU'):
          tf.compat.v2.summary.histogram(
              name='entropy_reg_loss',
              data=entropy_reg_loss,
              step=self.train_step_counter)
    else:
      raise ValueError('This is not allowed, this is handled at loss level.')

    entropy_reg_loss_per_sample = -entropy
    if self._check_numerics:
      entropy_reg_loss_per_sample = tf.debugging.check_numerics(
          entropy_reg_loss_per_sample, 'entropy_reg_loss_per_sample')

    return entropy_reg_loss, entropy_reg_loss_per_sample


class ReverbFixedLengthSequenceObserver(reverb_utils.ReverbAddTrajectoryObserver
                                       ):
  """Reverb fixed length sequence observer.

  This is a specialized observer similar to ReverbAddTrajectoryObserver but each
  sequence contains a fixed number of steps and can span multiple episodes. This
  implementation is consistent with (Schulman, 17).

  **Note**: Counting of steps in drivers does not include boundary steps. To
  guarantee only 1 item is pushed to the replay when collecting n steps with a
  `sequence_length` of n make sure to set the `stride_length`.
  """

  def __call__(self, trajectory):
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
    """
    self._writer.append(trajectory)
    self._cached_steps += 1

    self._write_cached_steps()


@gin.configurable
def train_eval(
    root_dir,
    env_name='HalfCheetah-v2',
    # Training params
    num_iterations=1600,
    actor_fc_layers=(64, 64),
    value_fc_layers=(64, 64),
    learning_rate=3e-4,
    collect_sequence_length=2048,
    minibatch_size=64,
    num_epochs=10,
    # Agent params
    importance_ratio_clipping=0.2,
    lambda_value=0.95,
    discount_factor=0.99,
    entropy_regularization=0.,
    value_pred_loss_coef=0.5,
    use_gae=True,
    use_td_lambda_return=True,
    gradient_clipping=0.5,
    value_clipping=None,
    # Replay params
    reverb_port=None,
    replay_capacity=10000,
    # Others
    policy_save_interval=5000,
    summary_interval=1000,
    eval_interval=10000,
    eval_episodes=100,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    train_mode_actor='dense',
    train_mode_value='dense',
    sparse_output_layer=True,
    weight_decay=0.0,
    width=1.0):
  """Trains and evaluates DQN."""

  logging.info('Actor fc layer params: %s', actor_fc_layers)
  logging.info('Value fc layer params: %s', value_fc_layers)
  logging.info('Policy save interval: %s', policy_save_interval)
  logging.info('Eval interval: %s', eval_interval)
  logging.info('Environment name: %s', env_name)
  logging.info('Learning rate: %s', learning_rate)
  logging.info('Num iterations: %s', num_iterations)
  logging.info('Sparse output layer: %s', sparse_output_layer)
  logging.info('Train mode actor: %s', train_mode_actor)
  logging.info('Train mode value: %s', train_mode_value)
  logging.info('Width: %s', width)
  logging.info('Weight decay: %s', weight_decay)

  if FLAGS.is_mujoco:
    collect_env = suite_mujoco.load(env_name)
    eval_env = suite_mujoco.load(env_name)
    logging.info('Loaded Mujoco environment %s', env_name)
  elif FLAGS.is_classic:
    collect_env = suite_gym.load(env_name)
    eval_env = suite_gym.load(env_name)
    logging.info('Loaded Classic control environment %s', env_name)
  else:
    raise ValueError('Environment init for Atari not supported yet.')

  num_environments = 1

  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))
  observation_tensor_spec = tf.TensorSpec(
      dtype=tf.float32, shape=observation_tensor_spec.shape)

  train_step = train_utils.create_train_step()

  if FLAGS.is_classic:
    actor_net_constructor = sparse_ppo_discrete_actor_network.PPODiscreteActorNetwork
  else:
    actor_net_constructor = sparse_ppo_actor_network.PPOActorNetwork

  actor_net_builder = actor_net_constructor(
            is_sparse=train_mode_actor == 'sparse',
            sparse_output_layer=sparse_output_layer,
            weight_decay=0,
            width=width)
  actor_net = actor_net_builder.create_sequential_actor_net(
      actor_fc_layers, action_tensor_spec,
      input_dim=time_step_tensor_spec.observation.shape[0])

  value_net = sparse_value_network.ValueNetwork(
      observation_tensor_spec,
      fc_layer_params=value_fc_layers,
      kernel_initializer=tf.keras.initializers.Orthogonal(),
      is_sparse=train_mode_value == 'sparse',
      sparse_output_layer=sparse_output_layer,
      weight_decay=0,
      width=width)
  logging.info('Train eval: weight decay %.5f.', weight_decay)

  current_iteration = tf.Variable(0, dtype=tf.int64)
  def learning_rate_fn():
    # Linearly decay the learning rate.
    return learning_rate * (1 - current_iteration / num_iterations)

  agent = SparsePPOAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=learning_rate_fn, epsilon=1e-5),
      actor_net=actor_net,
      value_net=value_net,
      importance_ratio_clipping=importance_ratio_clipping,
      lambda_value=lambda_value,
      discount_factor=discount_factor,
      entropy_regularization=entropy_regularization,
      value_pred_loss_coef=value_pred_loss_coef,
      policy_l2_reg=weight_decay,
      value_function_l2_reg=weight_decay,
      shared_vars_l2_reg=weight_decay,
      # This is a legacy argument for the number of times we repeat the data
      # inside of the train function, incompatible with mini batch learning.
      # We set the epoch number from the replay buffer and tf.Data instead.
      num_epochs=1,
      use_gae=use_gae,
      use_td_lambda_return=use_td_lambda_return,
      gradient_clipping=gradient_clipping,
      value_clipping=value_clipping,
      compute_value_and_advantage_in_train=False,
      # Skips updating normalizers in the agent, as it's handled in the learner.
      update_normalizers_in_train=False,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step)
  agent.initialize()

  reverb_server = reverb.Server(
      [
          reverb.Table(  # Replay buffer storing experience for training.
              name='training_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_capacity,
              max_times_sampled=1,
          ),
          reverb.Table(  # Replay buffer storing experience for normalization.
              name='normalization_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_capacity,
              max_times_sampled=1,
          )
      ],
      port=reverb_port)

  # Create the replay buffer.
  reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=collect_sequence_length,
      table_name='training_table',
      server_address='localhost:{}'.format(reverb_server.port),
      # The only collected sequence is used to populate the batches.
      max_cycle_length=1,
      num_workers_per_iterator=1,
      max_samples_per_stream=1,
      rate_limiter_timeout_ms=1000)
  reverb_replay_normalization = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=collect_sequence_length,
      table_name='normalization_table',
      server_address='localhost:{}'.format(reverb_server.port),
      # The only collected sequence is used to populate the batches.
      max_cycle_length=1,
      num_workers_per_iterator=1,
      max_samples_per_stream=1,
      rate_limiter_timeout_ms=1000)

  rb_observer = ReverbFixedLengthSequenceObserver(
      reverb_replay_train.py_client, ['training_table', 'normalization_table'],
      sequence_length=collect_sequence_length,
      stride_length=collect_sequence_length)

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  collect_env_step_metric = py_metrics.EnvironmentSteps()
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={
              triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
          }),
      triggers.StepPerSecondLogTrigger(train_step, interval=summary_interval),
  ]

  def training_dataset_fn():
    return reverb_replay_train.as_dataset(
        sample_batch_size=num_environments,
        sequence_preprocess_fn=agent.preprocess_sequence)

  def normalization_dataset_fn():
    return reverb_replay_normalization.as_dataset(
        sample_batch_size=num_environments,
        sequence_preprocess_fn=agent.preprocess_sequence)

  agent_learner = ppo_learner.PPOLearner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=training_dataset_fn,
      normalization_dataset_fn=normalization_dataset_fn,
      num_samples=1,
      summary_interval=10,
      num_epochs=num_epochs,
      minibatch_size=minibatch_size,
      shuffle_buffer_size=collect_sequence_length,
      triggers=learning_triggers)

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=collect_sequence_length,
      observers=[rb_observer, collect_env_step_metric],
      metrics=actor.collect_metrics(buffer_size=10) + [collect_env_step_metric],
      reference_metrics=[collect_env_step_metric],
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
      summary_interval=summary_interval)

  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      agent.policy, use_tf_function=True)

  average_returns = []
  if eval_interval:
    logging.info('Intial evaluation.')
    eval_actor = actor.Actor(
        eval_env,
        eval_greedy_policy,
        train_step,
        metrics=actor.eval_metrics(eval_episodes),
        reference_metrics=[collect_env_step_metric],
        summary_dir=os.path.join(root_dir, 'eval'),
        episodes_per_run=eval_episodes)

    eval_actor.run_and_log()
    for metric in eval_actor.metrics:
      if isinstance(metric, py_metrics.AverageReturnMetric):
        average_returns.append(metric._buffer.mean())

  logging.info('Training on %s', env_name)
  last_eval_step = 0
  for i in range(num_iterations):
    logging.info('collect_actor.run')
    collect_actor.run()
    # Reset the reverb observer to make sure the data collected is flushed and
    # written to the RB.
    # At this point, there a small number of steps left in the cache because the
    # actor does not count a boundary step as a step, whereas it still gets
    # added to Reverb for training. We throw away those extra steps without
    # padding to align with the paper implementation which never collects them
    # in the first place.
    logging.info('rb_observer.reset')
    rb_observer.reset(write_cached_steps=False)
    logging.info('reverb_replay_normalization.size: %d',
                 reverb_replay_normalization.get_table_info().current_size)
    logging.info('reverb_replay_train.size: %d',
                 reverb_replay_train.get_table_info().current_size)
    logging.info('agent_learner.run')
    agent_learner.run()
    logging.info('reverb_replay_train.clear')
    reverb_replay_train.clear()
    logging.info('reverb_replay_normalization.clear')
    reverb_replay_normalization.clear()
    current_iteration.assign_add(1)

    # Eval only if `eval_interval` has been set. Then, eval if the current train
    # step is equal or greater than the `last_eval_step` + `eval_interval` or if
    # this is the last iteration. This logic exists because agent_learner.run()
    # does not return after every train step.
    if (eval_interval and
        (agent_learner.train_step_numpy >= eval_interval + last_eval_step
         or i == num_iterations - 1)):
      logging.info('Evaluating.')
      eval_actor.run_and_log()
      last_eval_step = agent_learner.train_step_numpy
      for metric in eval_actor.metrics:
        if isinstance(metric, py_metrics.AverageReturnMetric):
          average_returns.append(metric._buffer.mean())

  # Log last section of evaluation scores for the final metric.
  idx = int(FLAGS.average_last_fraction * len(average_returns))
  avg_return = np.mean(average_returns[-idx:])
  logging.info('Step %d, Average Return: %f', collect_env_step_metric.result(),
               avg_return)

  rb_observer.close()
  reverb_server.stop()


def main(_):
  tf.config.experimental_run_functions_eagerly(False)
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  logging.info('Gin bindings: %s', FLAGS.gin_bindings)

  train_eval(
      FLAGS.root_dir,
      reverb_port=FLAGS.reverb_port)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  multiprocessing.handle_main(functools.partial(app.run, main))
