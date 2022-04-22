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

"""Tests for weight_symmetry.prune."""
import glob
from os import path

from absl.testing import absltest
from absl.testing import flagsaver

from rigl.experimental.jax import prune


class PruneTest(absltest.TestCase):

  def test_prune_fixed_schedule(self):
    """Tests training/pruning driver with a fixed global sparsity."""
    experiment_dir = self.create_tempdir().full_path
    eval_flags = dict(
        epochs=1,
        pruning_rate=0.95,
        experiment_dir=experiment_dir,
    )

    with flagsaver.flagsaver(**eval_flags):
      prune.main([])

      outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
      files = glob.glob(outfile)

      self.assertTrue(len(files) == 1 and path.exists(files[0]))

  def test_prune_global_pruning_schedule(self):
    """Tests training/pruning driver with a global sparsity schedule."""
    experiment_dir = self.create_tempdir().full_path
    eval_flags = dict(
        epochs=10,
        pruning_schedule='[(5, 0.33), (7, 0.66), (9, 0.95)]',
        experiment_dir=experiment_dir,
    )

    with flagsaver.flagsaver(**eval_flags):
      prune.main([])

      outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
      files = glob.glob(outfile)

      self.assertTrue(len(files) == 1 and path.exists(files[0]))

  def test_prune_local_pruning_schedule(self):
    """Tests training/pruning driver with a single layer sparsity schedule."""
    experiment_dir = self.create_tempdir().full_path
    eval_flags = dict(
        epochs=10,
        pruning_schedule='{1:[(5, 0.33), (7, 0.66), (9, 0.95)]}',
        experiment_dir=experiment_dir,
    )

    with flagsaver.flagsaver(**eval_flags):
      prune.main([])

      outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
      files = glob.glob(outfile)

      self.assertTrue(len(files) == 1 and path.exists(files[0]))

if __name__ == '__main__':
  absltest.main()
