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

r"""Train and Eval SAC.
"""

import functools
import os

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import reverb
from rigl.rigl_tf2 import mask_updaters
from rigl.rl import sparse_utils
from rigl.rl.tfagents import sparse_tanh_normal_projection_network
from rigl.rl.tfagents import tf_sparse_utils
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.keras_layers import inner_reshape
from tf_agents.metrics import py_metrics
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.utils import object_identity

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin binding parameters.')

# Env params
flags.DEFINE_bool('is_atari', False, 'Whether the env is an atari game.')
flags.DEFINE_bool('is_mujoco', False, 'Whether the env is a mujoco game.')
flags.DEFINE_bool('is_classic', False,
                  'Whether the env is a classic control game.')
flags.DEFINE_float(
    'average_last_fraction', 0.1,
    'Tells what fraction latest evaluation scores are averaged. This is used'
    ' to reduce variance.')

dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer='glorot_uniform')


def create_fc_layers(layer_units, width=1.0, weight_decay=0):
  layers = [
      dense(tf_sparse_utils.scale_width(num_units, width=width),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay))
      for num_units in layer_units
  ]
  return layers


def create_identity_layer():
  return tf.keras.layers.Lambda(lambda x: x)


def create_sequential_critic_network(obs_fc_layer_units,
                                     action_fc_layer_units,
                                     joint_fc_layer_units,
                                     input_dim,
                                     is_sparse = False,
                                     width = 1.0,
                                     weight_decay = 0.0,
                                     sparse_output_layer = True):
  """Create a sequential critic network."""
  # Split the inputs into observations and actions.
  def split_inputs(inputs):
    return {'observation': inputs[0], 'action': inputs[1]}

  # Create an observation network layers.
  obs_network_layers = (
      create_fc_layers(obs_fc_layer_units, width=width,
                       weight_decay=weight_decay)
      if obs_fc_layer_units else None)

  # Create an action network layers.
  action_network_layers = (
      create_fc_layers(action_fc_layer_units, width=width,
                       weight_decay=weight_decay)
      if action_fc_layer_units else None)

  # Create a joint network layers.
  joint_network_layers = (
      create_fc_layers(joint_fc_layer_units, width=width,
                       weight_decay=weight_decay)
      if joint_fc_layer_units else None)

  # Final layer.
  value_layer = tf.keras.layers.Dense(
      1, kernel_initializer='glorot_uniform',
      kernel_regularizer=tf.keras.regularizers.L2(weight_decay))

  layer_list = [obs_network_layers, action_network_layers,
                joint_network_layers]
  if is_sparse:
    # We need to process all-layers together to distribute sparsities for
    # pruning.
    all_layers = []
    for layers in layer_list:
      if layers is not None:
        all_layers += layers
    if sparse_output_layer:
      all_layers.append(value_layer)
      new_layers = tf_sparse_utils.wrap_all_layers(all_layers, input_dim)
      value_layer = new_layers[-1]
      new_layers = new_layers[:-1]
    else:
      new_layers = tf_sparse_utils.wrap_all_layers(all_layers, input_dim)
    # Split back the layers to their own groups
    c_index = 0
    new_layer_list = []
    for layers in layer_list:
      if layers is None:
        new_layer_list.append(None)
      else:
        new_layer_list.append(new_layers[c_index:len(layers)])
        c_index += len(layers)
    layer_list = new_layer_list
  # Convert layer_list to sequential or identity lambdas:
  module_list = [create_identity_layer() if layers is None else
                 sequential.Sequential(layers)
                 for layers in layer_list]
  obs_network, action_network, joint_network = module_list

  return sequential.Sequential([
      tf.keras.layers.Lambda(split_inputs),
      nest_map.NestMap({
          'observation': obs_network,
          'action': action_network
      }),
      nest_map.NestFlatten(),
      tf.keras.layers.Concatenate(),
      joint_network,
      value_layer,
      inner_reshape.InnerReshape(current_shape=[1], new_shape=[])
  ], name='sequential_critic')


class _TanhNormalProjectionNetworkWrapper(
    sparse_tanh_normal_projection_network.SparseTanhNormalProjectionNetwork):
  """Wrapper to pass predefined `outer_rank` to underlying projection net."""

  def __init__(self, sample_spec, predefined_outer_rank=1, weight_decay=0.0):
    super(_TanhNormalProjectionNetworkWrapper, self).__init__(
        sample_spec=sample_spec,
        weight_decay=weight_decay)
    self.predefined_outer_rank = predefined_outer_rank

  def call(self, inputs, network_state=(), **kwargs):
    kwargs['outer_rank'] = self.predefined_outer_rank
    if 'step_type' in kwargs:
      del kwargs['step_type']
    return super(_TanhNormalProjectionNetworkWrapper,
                 self).call(inputs, **kwargs)


def create_sequential_actor_network(actor_fc_layers,
                                    action_tensor_spec,
                                    input_dim,
                                    is_sparse = False,
                                    width = 1.0,
                                    weight_decay = 0.0,
                                    sparse_output_layer = True):
  """Create a sequential actor network."""
  def tile_as_nest(non_nested_output):
    return tf.nest.map_structure(lambda _: non_nested_output,
                                 action_tensor_spec)

  dense_layers = [
      dense(tf_sparse_utils.scale_width(num_units, width=width),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay))
      for num_units in actor_fc_layers
  ]
  tanh_normal_projection_network_fn = functools.partial(
      _TanhNormalProjectionNetworkWrapper,
      weight_decay=weight_decay)
  last_layer = nest_map.NestMap(
      tf.nest.map_structure(tanh_normal_projection_network_fn,
                            action_tensor_spec))
  if is_sparse:
    if sparse_output_layer:

      dense_layers.append(last_layer.layers[0]._projection_layer)
      new_layers = tf_sparse_utils.wrap_all_layers(dense_layers, input_dim)
      dense_layers = new_layers[:-1]
      last_layer.layers[0]._projection_layer = new_layers[-1]

    else:
      dense_layers = tf_sparse_utils.wrap_all_layers(dense_layers, input_dim)

  return sequential.Sequential(
      dense_layers +
      [tf.keras.layers.Lambda(tile_as_nest)] + [last_layer])


@gin.configurable
class SparseSacAgent(sac_agent.SacAgent):
  """Wrapped DqnAgent that supports sparse training."""

  def __init__(self,
               time_step_spec,
               action_spec,
               *args,
               actor_sparsity=None,
               critic_sparsity=None,
               **kwargs):
    super().__init__(time_step_spec,
                     action_spec,
                     *args,
                     **kwargs)
    # Pruning layer requires the pruning_step to be >1 during forward pass.
    tf_sparse_utils.update_prune_step(
        self._critic_network_1, self.train_step_counter + 1)
    tf_sparse_utils.update_prune_step(
        self._critic_network_2, self.train_step_counter + 1)
    tf_sparse_utils.update_prune_step(
        self._actor_network, self.train_step_counter + 1)

    if critic_sparsity is not None:
      _ = sparse_utils.init_masks(self._critic_network_1,
                                  sparsity=critic_sparsity)
      _ = sparse_utils.init_masks(self._critic_network_2,
                                  sparsity=critic_sparsity)
    else:  # Uses init_mask.sparsity value. Either the default or set via gin.
      _ = sparse_utils.init_masks(self._critic_network_1)
      _ = sparse_utils.init_masks(self._critic_network_2)

    if actor_sparsity is not None:
      _ = sparse_utils.init_masks(self._actor_network,
                                  sparsity=actor_sparsity)
    else:
      _ = sparse_utils.init_masks(self._actor_network)

    net_observation_spec = time_step_spec.observation
    critic_spec = (net_observation_spec, action_spec)
    self._target_critic_network_1 = (
        common.maybe_copy_target_network_with_checks(
            self._critic_network_1,
            None,
            input_spec=critic_spec,
            name='TargetCriticNetwork1'))
    self._target_critic_network_1 = (
        common.maybe_copy_target_network_with_checks(
            self._critic_network_2,
            None,
            input_spec=critic_spec,
            name='TargetCriticNetwork2'))

    def critic_loss_fn(experience, weights):
      # The following is just to fit to the existing API.
      transition = self._as_transition(experience)
      time_steps, policy_steps, next_time_steps = transition
      actions = policy_steps.action
      return self._critic_loss_weight * self.critic_loss(
          time_steps,
          actions,
          next_time_steps,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)

    def actor_loss_fn(experience, weights):
      # The following is just to fit to the existing API.
      transition = self._as_transition(experience)
      time_steps, _, _ = transition
      return self._actor_loss_weight*self.actor_loss(
          time_steps, weights=weights, training=True)

    # Create mask updater if doesn't exists
    self._mask_updater_critic_1 = mask_updaters.get_mask_updater(
        self._critic_network_1, self._critic_optimizer, critic_loss_fn)
    self._mask_updater_critic_2 = mask_updaters.get_mask_updater(
        self._critic_network_2, self._critic_optimizer, critic_loss_fn)
    self._mask_updater_actor = mask_updaters.get_mask_updater(
        self._actor_network, self._actor_optimizer, actor_loss_fn)

  def _train(self, experience, weights):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    tf.summary.experimental.set_step(self.train_step_counter)
    transition = self._as_transition(experience)
    time_steps, policy_steps, next_time_steps = transition
    actions = policy_steps.action

    trainable_critic_variables = list(object_identity.ObjectIdentitySet(
        self._critic_network_1.trainable_variables +
        self._critic_network_2.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self._critic_loss_weight*self.critic_loss(
          time_steps,
          actions,
          next_time_steps,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self._actor_loss_weight*self.actor_loss(
          time_steps, weights=weights, training=True)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    # BEGIN sparse training mask update
    # We use the lastest set of gradients to update the masks for sparse
    # training. Note, we do this before gradient clipping.

    # Define helper methods.
    def _mask_update_step(mask_updater, updater_name):
      mask_updater.set_validation_data(experience, weights)
      mask_updater.update(self.train_step_counter)
      with tf.name_scope('Drop_fraction/'):
        tf.summary.scalar(
            name=f'{updater_name}',
            data=mask_updater.last_drop_fraction)

    mask_update_step_critic_1 = functools.partial(_mask_update_step,
                                                  self._mask_updater_critic_1,
                                                  'critic_1')
    mask_update_step_critic_2 = functools.partial(_mask_update_step,
                                                  self._mask_updater_critic_2,
                                                  'critic_2')
    mask_update_step_actor = functools.partial(_mask_update_step,
                                               self._mask_updater_actor,
                                               'actor')

    # Log sparsities every 1000 train steps.
    def _log_sparsities():
      tf_sparse_utils.log_sparsities(self._critic_network_1, 'critic_1')
      tf_sparse_utils.log_sparsities(self._critic_network_2, 'critic_2')
      tf_sparse_utils.log_sparsities(self._actor_network, 'actor')
      tf_sparse_utils.log_total_params(
          [self._critic_network_1,
           self._critic_network_2,
           self._actor_network])
    tf.cond(self.train_step_counter % 1000 == 0, _log_sparsities, lambda: None)

    # Update critics
    if self._mask_updater_critic_1 is not None:
      is_update_critic_1 = self._mask_updater_critic_1.is_update_iter(
          self.train_step_counter)
      tf.cond(is_update_critic_1, mask_update_step_critic_1, lambda: None)

    if self._mask_updater_critic_2 is not None:
      is_update_critic_2 = self._mask_updater_critic_2.is_update_iter(
          self.train_step_counter)
      tf.cond(is_update_critic_2, mask_update_step_critic_2, lambda: None)

    # Update actor
    if self._mask_updater_actor is not None:
      is_update_actor = self._mask_updater_actor.is_update_iter(
          self.train_step_counter)
      tf.cond(is_update_actor, mask_update_step_actor, lambda: None)
    # END sparse training mask update

    alpha_variable = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert alpha_variable, 'No alpha variable to optimize.'
      tape.watch(alpha_variable)
      alpha_loss = self._alpha_loss_weight * self.alpha_loss(
          time_steps, weights=weights, training=True)
    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_grads = tape.gradient(alpha_loss, alpha_variable)
    self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = critic_loss + actor_loss + alpha_loss

    extra = sac_agent.SacLossInfo(
        critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)


@gin.configurable
def train_eval(
    root_dir,
    strategy,
    env_name='HalfCheetah-v2',
    # Training params
    initial_collect_steps=10000,
    num_iterations=1000000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Agent params
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    gamma=0.99,
    target_update_tau=0.005,
    target_update_period=1,
    reward_scale_factor=0.1,
    # Replay params
    reverb_port=None,
    replay_capacity=1000000,
    # Others
    policy_save_interval=10000,
    replay_buffer_save_interval=100000,
    eval_interval=10000,
    eval_episodes=30,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    sparse_output_layer = False,
    width = 1.0,
    train_mode_actor = 'dense',
    train_mode_value = 'dense',
    weight_decay = 0.0,
    actor_critic_sparsities_str = '',
    actor_critic_widths_str = ''):
  """Trains and evaluates SAC."""
  assert FLAGS.is_mujoco

  if actor_critic_widths_str:
    actor_critic_widths = [float(s) for s in actor_critic_widths_str.split('_')]
    width_actor = actor_critic_widths[0]
    width_value = actor_critic_widths[1]
  else:
    width_actor = width
    width_value = width

  if actor_critic_sparsities_str:
    actor_critic_sparsities = [
        float(s) for s in actor_critic_sparsities_str.split('_')
    ]
  else:
    # init_mask.sparsity value will be used. Either the default or set via gin.
    actor_critic_sparsities = [None, None]

  logging.info('Training SAC on: %s', env_name)
  logging.info('SAC params: train mode actor: %s', train_mode_actor)
  logging.info('SAC params: train mode value: %s', train_mode_value)
  logging.info('SAC params: sparse_output_layer: %s', sparse_output_layer)
  logging.info('SAC params: width: %s', width)
  logging.info('SAC params: actor_critic_widths_str: %s',
               actor_critic_widths_str)
  logging.info('SAC params: width_actor: %s', width_actor)
  logging.info('SAC params: width_value: %s', width_value)
  logging.info('SAC params: weight_decay: %s', weight_decay)
  logging.info('SAC params: actor_critic_sparsities_str %s type %s',
               actor_critic_sparsities_str, type(actor_critic_sparsities_str))
  logging.info('SAC params: actor_sparsity: %s', actor_critic_sparsities[0])
  logging.info('SAC params: critic_sparsity: %s', actor_critic_sparsities[1])

  collect_env = suite_mujoco.load(env_name)
  eval_env = suite_mujoco.load(env_name)

  _, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  actor_net = create_sequential_actor_network(
      actor_fc_layers=actor_fc_layers, action_tensor_spec=action_tensor_spec,
      input_dim=time_step_tensor_spec.observation.shape[0],
      is_sparse=(train_mode_actor == 'sparse'),
      width=width_actor,
      weight_decay=weight_decay,
      sparse_output_layer=sparse_output_layer)

  critic_input_dim = (
      action_tensor_spec.shape[0] + time_step_tensor_spec.observation.shape[0])
  critic_net = create_sequential_critic_network(
      obs_fc_layer_units=critic_obs_fc_layers,
      action_fc_layer_units=critic_action_fc_layers,
      joint_fc_layer_units=critic_joint_fc_layers,
      input_dim=critic_input_dim,
      is_sparse=(train_mode_value == 'sparse'),
      width=width_value,
      weight_decay=weight_decay,
      sparse_output_layer=sparse_output_layer)

  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = SparseSacAgent(
        time_step_spec=time_step_tensor_spec,
        action_spec=action_tensor_spec,
        actor_sparsity=actor_critic_sparsities[0],
        critic_sparsity=actor_critic_sparsities[1],
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step)
    agent.initialize()
  table_name = 'uniform_table'
  table = reverb.Table(
      table_name,
      max_size=replay_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1))

  reverb_checkpoint_dir = os.path.join(root_dir, learner.TRAIN_DIR,
                                       learner.REPLAY_BUFFER_CHECKPOINT_DIR)
  reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
      path=reverb_checkpoint_dir)
  reverb_server = reverb.Server([table],
                                port=reverb_port,
                                checkpointer=reverb_checkpointer)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name=table_name,
      local_server=reverb_server)
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client,
      table_name,
      sequence_length=2,
      stride_length=1)

  def experience_dataset_fn():
    return reverb_replay.as_dataset(
        sample_batch_size=batch_size, num_steps=2).prefetch(50)

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  env_step_metric = py_metrics.EnvironmentSteps()
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric}),
      triggers.ReverbCheckpointTrigger(
          train_step,
          interval=replay_buffer_save_interval,
          reverb_client=reverb_replay.py_client),
      triggers.StepPerSecondLogTrigger(train_step, interval=1000),
  ]

  agent_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
      strategy=strategy)

  random_policy = random_py_policy.RandomPyPolicy(
      collect_env.time_step_spec(), collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
      observers=[rb_observer, env_step_metric])

  tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_greedy_policy, use_tf_function=True)

  eval_actor = actor.Actor(
      eval_env,
      eval_greedy_policy,
      train_step,
      episodes_per_run=eval_episodes,
      metrics=actor.eval_metrics(eval_episodes),
      summary_dir=os.path.join(root_dir, 'eval'),
  )

  average_returns = []
  if eval_interval:
    logging.info('Evaluating.')
    eval_actor.run_and_log()
    for metric in eval_actor.metrics:
      if isinstance(metric, py_metrics.AverageReturnMetric):
        average_returns.append(metric._buffer.mean())

  logging.info('Training.')
  for _ in range(num_iterations):
    collect_actor.run()
    agent_learner.run(iterations=1)

    if eval_interval and agent_learner.train_step_numpy % eval_interval == 0:
      logging.info('Evaluating.')
      eval_actor.run_and_log()
      for metric in eval_actor.metrics:
        if isinstance(metric, py_metrics.AverageReturnMetric):
          average_returns.append(metric._buffer.mean())

  # Log last section of evaluation scores for the final metric.
  idx = int(FLAGS.average_last_fraction * len(average_returns))
  avg_return = np.mean(average_returns[-idx:])
  logging.info('Step %d, Average Return: %f', env_step_metric.result(),
               avg_return)

  rb_observer.close()
  reverb_server.stop()


def main(_):
  tf.config.run_functions_eagerly(False)
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  logging.info('Gin bindings: %s', FLAGS.gin_bindings)
  logging.info('# Gin-Config:\n %s', gin.config.operative_config_str())

  train_eval(
      FLAGS.root_dir,
      strategy=strategy,
      reverb_port=FLAGS.reverb_port)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
