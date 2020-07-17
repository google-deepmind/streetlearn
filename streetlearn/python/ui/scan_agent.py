# Copyright 2018 Google LLC.
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

"""Basic panorama scanning agent for StreetLearn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import time
import numpy as np
import pygame

from streetlearn.python.environment import coin_game
from streetlearn.python.environment import default_config
from streetlearn.python.environment import streetlearn

FLAGS = flags.FLAGS
flags.DEFINE_integer("width", 400, "Observation and map width.")
flags.DEFINE_integer("height", 400, "Observation and map height.")
flags.DEFINE_integer('field_of_view', 60, 'Field of view.')
flags.DEFINE_string("dataset_path", None, "Dataset path.")
flags.DEFINE_string("list_pano_ids_yaws", None, "List of pano IDs and yaws.")
flags.DEFINE_bool("save_images", False, "Save the images?")
flags.mark_flag_as_required("dataset_path")
flags.mark_flag_as_required("list_pano_ids_yaws")


def loop(env, screen, pano_ids_yaws):
  """Main loop of the scan agent."""
  for (pano_id, yaw) in pano_ids_yaws:

    # Retrieve the observation at a specified pano ID and heading.
    logging.info('Retrieving view at pano ID %s and yaw %f', pano_id, yaw)
    observation = env.goto(pano_id, yaw)

    current_yaw = observation["yaw"]
    view_image = observation["view_image_hwc"]
    graph_image = observation["graph_image_hwc"]
    screen_buffer = np.concatenate((view_image, graph_image), axis=0)
    pygame.surfarray.blit_array(screen, screen_buffer.swapaxes(0, 1))
    pygame.display.update()

    for event in pygame.event.get():
      if (event.type == pygame.QUIT or
          (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
        return
    if FLAGS.save_images:
      filename = 'scan_agent_{}_{}.bmp'.format(pano_id, yaw)
      pygame.image.save(screen, filename)

def main(argv):
  config = {'width': FLAGS.width,
            'height': FLAGS.height,
            'field_of_view': FLAGS.field_of_view,
            'graph_width': FLAGS.width,
            'graph_height': FLAGS.height,
            'graph_zoom': 1,
            'full_graph': True,
            'proportion_of_panos_with_coins': 0.0,
            'action_spec': 'streetlearn_fast_rotate',
            'observations': ['view_image_hwc', 'graph_image_hwc', 'yaw']}
  with open(FLAGS.list_pano_ids_yaws, 'r') as f:
    lines = f.readlines()
    pano_ids_yaws = [(line.split('\t')[0], float(line.split('\t')[1]))
                     for line in lines]
  config = default_config.ApplyDefaults(config)
  game = coin_game.CoinGame(config)
  env = streetlearn.StreetLearn(FLAGS.dataset_path, config, game)
  env.reset()
  pygame.init()
  screen = pygame.display.set_mode((FLAGS.width, FLAGS.height * 2))
  loop(env, screen, pano_ids_yaws)

if __name__ == '__main__':
  app.run(main)
