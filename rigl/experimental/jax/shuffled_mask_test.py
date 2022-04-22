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

"""Tests for weight_symmetry.shuffled_mask."""
import glob
from os import path
import tempfile

from absl.testing import absltest
from absl.testing import flagsaver
from rigl.experimental.jax import shuffled_mask


class ShuffledMaskTest(absltest.TestCase):

  def test_run_fc(self):
    """Tests if the driver for shuffled training runs correctly with FC NN."""
    experiment_dir = tempfile.mkdtemp()
    eval_flags = dict(
        epochs=1,
        experiment_dir=experiment_dir,
        model='MNIST_FC',
    )

    with flagsaver.flagsaver(**eval_flags):
      shuffled_mask.main([])

    outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
    files = glob.glob(outfile)

    self.assertTrue(len(files) == 1 and path.exists(files[0]))

  def test_run_conv(self):
    """Tests if the driver for shuffled training runs correctly with CNN."""
    experiment_dir = tempfile.mkdtemp()
    eval_flags = dict(
        epochs=1,
        experiment_dir=experiment_dir,
        model='MNIST_CNN',
    )

    with flagsaver.flagsaver(**eval_flags):
      shuffled_mask.main([])

    outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
    files = glob.glob(outfile)

    self.assertTrue(len(files) == 1 and path.exists(files[0]))

  def test_run_random(self):
    """Test random mask driver with per-neuron sparsity."""
    experiment_dir = tempfile.mkdtemp()
    self._eval_flags = dict(
        epochs=1,
        experiment_dir=experiment_dir,
        mask_type='random',
    )

    with flagsaver.flagsaver(**self._eval_flags):
      shuffled_mask.main([])

    outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
    files = glob.glob(outfile)

    self.assertTrue(len(files) == 1 and path.exists(files[0]))

  def test_run_per_neuron(self):
    """Test random mask driver with per-neuron sparsity."""
    experiment_dir = tempfile.mkdtemp()
    self._eval_flags = dict(
        epochs=1,
        experiment_dir=experiment_dir,
        mask_type='per_neuron',
    )

    with flagsaver.flagsaver(**self._eval_flags):
      shuffled_mask.main([])

    outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
    files = glob.glob(outfile)

    self.assertTrue(len(files) == 1 and path.exists(files[0]))

  def test_run_symmetric(self):
    """Test random mask driver with per-neuron sparsity."""
    experiment_dir = tempfile.mkdtemp()
    self._eval_flags = dict(
        epochs=1,
        experiment_dir=experiment_dir,
        mask_type='symmetric',
    )

    with flagsaver.flagsaver(**self._eval_flags):
      shuffled_mask.main([])

    outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
    files = glob.glob(outfile)

    self.assertTrue(len(files) == 1 and path.exists(files[0]))

if __name__ == '__main__':
  absltest.main()
