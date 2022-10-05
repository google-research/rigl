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

r"""Sparse training DQN using actor/learner in a gym environment.
"""
import functools
import os

from typing import Tuple

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import reverb
from rigl.rigl_tf2 import mask_updaters
from rigl.rl import sparse_utils
from rigl.rl.tfagents import tf_sparse_utils
import tensorflow.compat.v2 as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_atari
from tf_agents.environments import suite_gym
from tf_agents.metrics import py_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.utils import eager_utils

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
flags.DEFINE_float(
    'average_last_fraction', 0.1,
    'Tells what fraction latest evaluation scores are averaged. This is used'
    ' to reduce variance.')



@gin.configurable
class SparseDqnAgent(dqn_agent.DqnAgent):
  """Wrapped DqnAgent that supports sparse training."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    _ = sparse_utils.init_masks(self._q_network)
    def loss_fn(experience_data, weights_data):
      # The following is just to fit to the existing API.
      loss_info = self._loss(
          experience_data,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights_data,
          training=True)
      return loss_info.extra.td_loss
    # Create mask updater if doesn't exists
    self._mask_updater = mask_updaters.get_mask_updater(
        self._q_network, self._optimizer, loss_fn)

  def _train(self, experience, weights):
    tf.compat.v2.summary.experimental.set_step(self.train_step_counter)

    tf_sparse_utils.update_prune_step(self._q_network, self._train_step_counter)
    with tf.GradientTape(persistent=True) as tape:
      loss_info = self._loss(
          experience,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)
    tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
    variables_to_train = self._q_network.trainable_weights
    non_trainable_weights = self._q_network.non_trainable_weights
    assert list(variables_to_train), "No variables in the agent's q_network."
    grads = tape.gradient(loss_info.loss, variables_to_train)

    tf_sparse_utils.log_snr(tape, loss_info.extra.td_loss,
                            self.train_step_counter, variables_to_train)

    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = list(zip(grads, variables_to_train))

    def _mask_update_step():
      # Second argument is not used.
      self._mask_updater.set_validation_data(experience, weights)
      self._mask_updater.update(self.train_step_counter)
      with tf.name_scope('/'):
        tf.summary.scalar(
            name='drop_fraction', data=self._mask_updater.last_drop_fraction)

    tf_sparse_utils.log_sparsities(self._q_network)
    if self._mask_updater is not None:
      is_update = self._mask_updater.is_update_iter(self.train_step_counter)
      tf.cond(is_update, _mask_update_step, lambda: None)

    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      grads_and_vars_with_non_trainable = (
          grads_and_vars + [(None, v) for v in non_trainable_weights])
      eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)
    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)

    self._update_target()

    return loss_info


def _scale_width(num_units, width):
  assert width > 0
  return int(max(1, num_units * width))


def build_network(
    fc_layer_params,
    num_actions,
    is_sparse,
    input_dim,
    width = 1.0,
    weight_decay = 0.0,
    sparse_output_layer = True
    ):
  """Builds a Sequential model."""

  def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'),
        kernel_regularizer=tf.keras.regularizers.L2(weight_decay),)

  # QNetwork consists of a sequence of Dense layers followed by a dense layer
  # with `num_actions` units to generate one q_value per available action as
  # its output.
  all_layers = [
      dense_layer(_scale_width(num_units, width=width)
                  ) for num_units in fc_layer_params]
  all_layers.append(
      tf.keras.layers.Dense(
          num_actions,
          activation=None,
          kernel_initializer=tf.keras.initializers.RandomUniform(
              minval=-0.03, maxval=0.03),
          bias_initializer=tf.keras.initializers.Constant(-0.2)))
  if is_sparse:
    if sparse_output_layer:
      all_layers = tf_sparse_utils.wrap_all_layers(all_layers, input_dim)
    else:
      all_layers = (tf_sparse_utils.wrap_all_layers(all_layers[:-1], input_dim)
                    + all_layers[-1:])
  return sequential.Sequential(all_layers)




@gin.configurable
def train_eval(
    root_dir,
    env_name='CartPole-v0',
    # Training params
    update_frequency=1,
    initial_collect_steps=1000,
    num_iterations=100000,
    fc_layer_params=(100,),
    # Agent params
    epsilon_greedy=0.1,
    epsilon_decay_period=250000,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    target_update_tau=1.0,
    target_update_period=100,
    reward_scale_factor=1.0,
    # Replay params
    reverb_port=None,
    replay_capacity=100000,
    # Others
    policy_save_interval=1000,
    eval_interval=1000,
    eval_episodes=10,
    weight_decay = 0.0,
    width = 1.0,
    debug_summaries=False,
    sparse_output_layer=True,
    train_mode='dense'):
  """Trains and evaluates DQN."""

  logging.info('DQN params: Fc layer params: %s', fc_layer_params)
  logging.info('DQN params: Train mode: %s', train_mode)
  logging.info('DQN params: Target update period: %s', target_update_period)
  logging.info('DQN params: Policy save interval: %s', policy_save_interval)
  logging.info('DQN params: Eval interval: %s', eval_interval)
  logging.info('DQN params: Environment name: %s', env_name)
  logging.info('DQN params: Weight decay: %s', weight_decay)
  logging.info('DQN params: Width: %s', width)
  logging.info('DQN params: Batch size: %s', batch_size)
  logging.info('DQN params: Target update period: %s', target_update_period)
  logging.info('DQN params: Learning rate: %s', learning_rate)
  logging.info('DQN params: Num iterations: %s', num_iterations)
  logging.info('DQN params: Sparse output layer: %s', sparse_output_layer)

  collect_env = suite_gym.load(env_name)
  eval_env = suite_gym.load(env_name)
  logging.info('Collect env: %s', collect_env)
  logging.info('Eval env: %s', eval_env)

  time_step_tensor_spec = tensor_spec.from_spec(collect_env.time_step_spec())
  action_tensor_spec = tensor_spec.from_spec(collect_env.action_spec())
  train_step = train_utils.create_train_step()
  num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
  observation_shape = collect_env.observation_spec().shape
  # Build network and get pruning params
  is_atari = False
  if not is_atari:
    q_net = build_network(
        fc_layer_params=fc_layer_params,
        num_actions=num_actions,
        is_sparse=(train_mode == 'sparse'),
        # observation_shape is 1-dimensional. We need this so that we can
        # calculate the dimensions of the first layer.
        input_dim=observation_shape[-1],
        width=width,
        weight_decay=weight_decay,
        sparse_output_layer=sparse_output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = common.element_wise_squared_loss
    decay_fn = epsilon_greedy

  agent = SparseDqnAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      q_network=q_net,
      epsilon_greedy=decay_fn,
      n_step_update=n_step_update,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      optimizer=optimizer,
      td_errors_loss_fn=loss,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      train_step_counter=train_step,
      debug_summaries=debug_summaries)
  table_name = 'uniform_table'
  table = reverb.Table(
      table_name,
      max_size=replay_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1))
  reverb_server = reverb.Server([table], port=reverb_port)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name=table_name,
      local_server=reverb_server)
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client, table_name,
      sequence_length=2,
      stride_length=1)

  dataset = reverb_replay.as_dataset(
      num_parallel_calls=3, sample_batch_size=batch_size,
      num_steps=2).prefetch(3)
  experience_dataset_fn = lambda: dataset

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  env_step_metric = py_metrics.EnvironmentSteps()

  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric}),
      triggers.StepPerSecondLogTrigger(train_step, interval=100),
  ]

  dqn_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
      run_optimizer_variable_init=False)

  # If we haven't trained yet make sure we collect some random samples first to
  # fill up the Replay Buffer with some experience.
  random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(),
                                                  collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy,
                                                      use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=update_frequency,
      observers=[rb_observer, env_step_metric],
      metrics=actor.collect_metrics(10),
      reference_metrics=[env_step_metric],
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
  )

  tf_greedy_policy = agent.policy
  greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_greedy_policy,
                                                     use_tf_function=True)

  eval_actor = actor.Actor(
      eval_env,
      greedy_policy,
      train_step,
      episodes_per_run=eval_episodes,
      metrics=actor.eval_metrics(eval_episodes),
      reference_metrics=[env_step_metric],
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
    dqn_learner.run(iterations=1)

    if eval_interval and dqn_learner.train_step_numpy % eval_interval == 0:
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
