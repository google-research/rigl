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

"""Tests for weight_symmetry.fixed_param."""
import glob
from os import path
import tempfile

from absl.testing import absltest
from absl.testing import flagsaver

from rigl.experimental.jax import fixed_param


class FixedParamTest(absltest.TestCase):

  def test_run(self):
    """Tests if the driver for shuffled training runs correctly."""
    experiment_dir = tempfile.mkdtemp()
    eval_flags = dict(
        epochs=1,
        experiment_dir=experiment_dir,
    )

    with flagsaver.flagsaver(**eval_flags):
      fixed_param.main([])

    with self.subTest(name='tf_summary_file_exists'):
      outfile = path.join(experiment_dir, '*', 'events.out.tfevents.*')
      files = glob.glob(outfile)
      self.assertTrue(len(files) == 1 and path.exists(files[0]))
if __name__ == '__main__':
  absltest.main()
