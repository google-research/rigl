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

"""Tests for sparse_ppo_discrete_actor_network."""

from absl import flags
from absl.testing import parameterized

from rigl.rl.tfagents import sparse_ppo_discrete_actor_network
import tensorflow as tf
from tf_agents.distributions import utils as distribution_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import test_utils

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

FLAGS = flags.FLAGS


class DeterministicSeedStream(object):
  """A fake seed stream class that always generates a deterministic seed."""

  def __init__(self, seed, salt=''):
    del salt
    self._seed = seed

  def __call__(self):
    return self._seed


class PpoActorNetworkTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(PpoActorNetworkTest, self).setUp()
    # Run in full eager mode in order to inspect the content of tensors.
    tf.config.experimental_run_functions_eagerly(True)
    self.observation_tensor_spec = tf.TensorSpec(shape=[3], dtype=tf.float32)
    self.action_tensor_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)

  def tearDown(self):
    tf.config.experimental_run_functions_eagerly(False)
    super(PpoActorNetworkTest, self).tearDown()

  def _init_network(
      self, is_sparse=False, sparse_output_layer=False,
      width=1.0, weight_decay=0):
    actor_net_lib = sparse_ppo_discrete_actor_network.PPODiscreteActorNetwork(
        is_sparse=is_sparse, sparse_output_layer=sparse_output_layer,
        width=width, weight_decay=weight_decay)
    actor_net_lib.seed_stream_class = DeterministicSeedStream
    return actor_net_lib.create_sequential_actor_net(
        fc_layer_units=(1,), action_tensor_spec=self.action_tensor_spec, seed=1)

  def test_no_mismatched_shape(self):
    actor_net = self._init_network()
    actor_output_spec = actor_net.create_variables(self.observation_tensor_spec)
    distribution_utils.assert_specs_are_compatible(
        actor_output_spec, self.action_tensor_spec,
        'actor_network output spec does not match action spec')

  @parameterized.named_parameters(
      ('dense-output-F', False, False,
       (tf.keras.layers.Dense, tf.keras.layers.Dense)),
      ('dense-output-T', False, True,
       (tf.keras.layers.Dense, tf.keras.layers.Dense)),
      ('sparse-all', True, True,
       (pruning_wrapper.PruneLowMagnitude, pruning_wrapper.PruneLowMagnitude)),
      ('sparse-outp-dense', True, False,
       (pruning_wrapper.PruneLowMagnitude, tf.keras.layers.Dense)),
      )
  def test_is_sparse(self, is_sparse, sparse_output_layer, expected_layers):
    expected_units = (1, 4)
    actor_net = self._init_network(
        is_sparse=is_sparse, sparse_output_layer=sparse_output_layer)
    for i, (expected_layer, exp_units) in enumerate(
        zip(expected_layers, expected_units)):
      layer = actor_net.layers[i]
      self.assertIsInstance(layer, expected_layer)
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        self.assertEqual(layer.layer.units, exp_units)
      else:
        self.assertEqual(layer.units, exp_units)

  def test_width_scaling(self):
    with self.subTest('dense'):
      actor_net = self._init_network(width=2.0)
      self.assertEqual(actor_net.layers[0].units, 2)
      self.assertEqual(actor_net.layers[1].units, 4)

    with self.subTest('sparse'):
      actor_net = self._init_network(
          is_sparse=True, sparse_output_layer=True, width=2.0)
      self.assertEqual(actor_net.layers[0].layer.units, 2)
      self.assertEqual(actor_net.layers[1].layer.units, 4)

  @parameterized.named_parameters(
      ('no-wd-d-d', False, False, 0),
      ('no-wd-s-d', True, False, 0),
      ('no-wd-s-s', True, True, 0),
      ('wd-d-d', False, False, 0.1),
      ('wd-s-d', True, False, 0.1),
      ('wd-s-s', True, True, 0.1))
  def test_weight_decay(self, is_sparse, sparse_output_layer,
                        expected_weight_decay):
    actor_net = self._init_network(is_sparse=is_sparse,
                                   sparse_output_layer=sparse_output_layer,
                                   weight_decay=expected_weight_decay)
    for i in range(2):
      layer = actor_net.layers[i]
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        l2_weight_decay = layer.layer.kernel_regularizer.get_config()['l2']
      else:
        l2_weight_decay = layer.kernel_regularizer.get_config()['l2']
      self.assertAlmostEqual(l2_weight_decay, expected_weight_decay)


if __name__ == '__main__':
  tf.test.main()
