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

"""Run policy evaluation as supervised learning, reloading representations."""

import sys

from absl import logging
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
import gin
import numpy as np
from rigl.rl import dqn_agents
import tensorflow.compat.v1 as tf1

# Last 10% of the training is averaged to get final reward.
AVG_REWARD_FRAC = 0.1


@gin.configurable
def create_sparse_agent(sess, num_actions, agent=None, summary_writer=None):
  """Creates a sparse agent.

  Args:
    sess: tf.Session.
    num_actions: int, number of actions.
    agent: str, type of learner/actor agent to create.
    summary_writer: tf.SummaryWriter, for Tensorboard.

  Returns:
    A learner/actor agent.
  """
  assert agent is not None
  if agent == 'dqn':
    return dqn_agents.SparseDQNAgent(
        sess, num_actions, summary_writer=summary_writer)
  else:
    raise ValueError('Unknown learner agent: {}'.format(agent))


@gin.configurable
class SparseTrainRunner(run_experiment.Runner):
  """Policy evaluation as supervised learning, from a loaded representation."""

  def __init__(self,
               base_dir,
               agent_type,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               load_env_fn=gym_lib.create_gym_environment,
               clip_rewards=True,
               atari_100k_eval=False,
               num_eval_episodes=100,
               observation_noise=None):
    """Initialize SparseTrainRunner in charge of running the experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      agent_type: str, defines the type of targets to be learned. Can be one of
        {'dqn', 'rainbow'}.
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      load_env_fn: fn, function which loads and returns an environment.
      clip_rewards: bool, whether to clip rewards in [-1, 1].
      atari_100k_eval: bool, whether we are using the eval for Atari 100K.
      num_eval_episodes: int, the number of full episodes to run during eval,
        only used if atari_100k_eval is True.
      observation_noise: float (optional), the stddev to use to add noise to the
        observations before sending to the agent.
    """
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._clip_rewards = clip_rewards
    self._atari_100k_eval = atari_100k_eval
    self._num_eval_episodes = num_eval_episodes
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf1.summary.FileWriter(self._base_dir)
    self._observation_noise = observation_noise

    self._environment = load_env_fn()

    num_actions = self._environment.action_space.n

    config = tf1.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.
    self._sess = tf1.Session('local', config=config)
    self._agent = create_sparse_agent(
        self._sess, num_actions, agent=agent_type,
        summary_writer=self._summary_writer)
    self._summary_writer.add_graph(graph=tf1.get_default_graph())
    self._sess.run(tf1.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

  def _run_one_phase_fix_episodes(self, max_episodes, statistics):
    """Run one eval phase for the Atari 100k benchmark.

    As opposed to the standard eval phase which runs for a fixed number of
    steps, this will run for a fixed number of episodes, producing less noisy
    results.

    Args:
      max_episodes: int, max number of episodes to run.
      statistics: `IterationStatistics` object which records the experimental
        results.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while num_episodes < max_episodes:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          'eval_episode_lengths': episode_length,
          'eval_episode_returns': episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Num episodes: {} '.format(num_episodes) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_eval_phase(self, statistics):
    if not self._atari_100k_eval:
      return super()._run_eval_phase(statistics)
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase_fix_episodes(
        self._num_eval_episodes, statistics)
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_step(self, action):
    """Maybe adds noise to observations."""
    observation, reward, is_terminal, _ = self._environment.step(action)
    if self._observation_noise is not None:
      observation += np.random.normal(
          scale=self._observation_noise,
          size=observation.shape).astype(observation.dtype)
    return observation, reward, is_terminal

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return
    self._agent.update_prune_step()
    self._agent.maybe_init_masks()
    all_eval_returns = []
    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      all_eval_returns.append(statistics['eval_average_return'][-1])
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
    last_n = int(self._num_iterations * AVG_REWARD_FRAC)
    avg_return = np.mean(all_eval_returns[-last_n:])
    logging.info('Step %d, Average Return: %f', iteration, avg_return)
