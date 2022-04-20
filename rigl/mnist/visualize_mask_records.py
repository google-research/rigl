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

r"""Visualizes the dumped masks using matplotlib.

We count the number of outgoing edges from the input dimensions. For the first
layer input dimensions correspond to the input pixels and we can visualize it
nicely. You can control which layer is visualized by changing `layer_id` and
`new_shape`. Default is the first layer and we visualize the number of outgoing
connections from individual pixels.

python visualize_mask_records.py --records_path=/tmp/mnist/mask_records.npy

To save the results as gif:
python visualize_mask_records.py --records_path=/path/to/mask_records.npy \
--save_path=/path/to/mask.gif

Modified from:
https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

flags.DEFINE_string('records_path', '/tmp/mnist/mask_records.npy',
                    'Path to load masks records.')
flags.DEFINE_string('save_path', '', 'Path to save the animation.')
flags.DEFINE_list('new_shape', '28,28', 'Path for reshaping the units.')
flags.DEFINE_integer('interval', 100, 'Miliseconds between plot updates.')
flags.DEFINE_integer('layer_id', 0, 'of which we plot statistics during '
                     'training.')
flags.DEFINE_integer('skip_mask', 10, 'number of checkpoints to skip for '
                     'each frame.')
flags.DEFINE_integer(
    'slow_until', 50, 'Number of masks to show with slower '
    'speed. After this number of frames, we start skipping '
    'frames to make the video shorter.')
FLAGS = flags.FLAGS


def main(unused_args):
  fig, ax = plt.subplots()
  fig.set_tight_layout(True)

  # Query the figure's on-screen size and DPI. Note that when saving the figure
  # to a file, we need to provide a DPI for that separately.
  print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(),
                                                       fig.get_size_inches()))

  # Plot a scatter that persists (isn't redrawn) and the initial line.
  mask_records = np.load(FLAGS.records_path, allow_pickle=True).item()
  sorted_keys = sorted(mask_records.keys())
  new_shape = [int(a) for a in FLAGS.new_shape]
  reshape_fn = lambda mask: np.reshape(np.sum(mask, axis=1), new_shape)
  c_mask = mask_records[sorted_keys[0]][FLAGS.layer_id]
  im = plt.imshow(reshape_fn(c_mask), interpolation='none', vmin=0, vmax=30)
  fig.colorbar(im, ax=ax)

  def update(i):
    """Updates the plot."""
    save_iter = sorted_keys[i]
    label = 'timestep {0}'.format(save_iter)

    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    c_data = reshape_fn(mask_records[save_iter][FLAGS.layer_id])
    im.set_data(c_data)
    ax.set_xlabel(label)
    return [im, ax]

  # FuncAnimation will call the 'update' function for each frame; here
  # animating over 10 frames, with an interval of 200ms between frames.
  iteration = FLAGS.slow_until
  frames = (
      list(np.arange(0, iteration, 1)) +
      list(np.arange(iteration, len(sorted_keys), FLAGS.skip_mask)))

  anim = FuncAnimation(fig, update, frames=frames, interval=FLAGS.interval)
  if FLAGS.save_path:
    anim.save(FLAGS.save_path, dpi=80, writer='imagemagick')
  else:
    # plt.show() will just loop the animation forever.
    plt.show()


if __name__ == '__main__':
  tf.app.run(main)
