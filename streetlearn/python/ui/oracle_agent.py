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

"""Basic oracle agent for StreetLearn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import time
import numpy as np
import pygame

from streetlearn.python.environment import courier_game
from streetlearn.python.environment import default_config
from streetlearn.python.environment import streetlearn

FLAGS = flags.FLAGS
flags.DEFINE_integer('width', 400, 'Observation and map width.')
flags.DEFINE_integer('height', 400, 'Observation and map height.')
flags.DEFINE_integer('field_of_view', 60, 'Field of view.')
flags.DEFINE_integer('graph_zoom', 1, 'Zoom level.')
flags.DEFINE_float('horizontal_rot', 22.5, 'Horizontal rotation step (deg).')
flags.DEFINE_string('dataset_path', None, 'Dataset path.')
flags.DEFINE_string('start_pano', '',
                     'Pano at root of partial graph (default: full graph).')
flags.DEFINE_integer('graph_depth', 200, 'Depth of the pano graph.')
flags.DEFINE_integer('frame_cap', 1000, 'Number of frames / episode.')
flags.DEFINE_string('stats_path', None, 'Statistics path.')
flags.DEFINE_float('proportion_of_panos_with_coins', 0, 'Proportion of coins.')
flags.mark_flag_as_required('dataset_path')


TOL_BEARING = 30


def interleave(array, w, h):
  """Turn a planar RGB array into an interleaved one.

  Args:
    array: An array of bytes consisting the planar RGB image.
    w: Width of the image.
    h: Height of the image.
  Returns:
    An interleaved array of bytes shape shaped (h, w, 3).
  """
  arr = array.reshape(3, w * h)
  return np.ravel((arr[0], arr[1], arr[2]),
                  order='F').reshape(h, w, 3).swapaxes(0, 1)

def loop(env, screen):
  """Main loop of the oracle agent."""
  action = np.array([0, 0, 0, 0])
  action_spec = env.action_spec()
  sum_rewards = 0
  sum_rewards_at_goal = 0
  previous_goal_id = None
  seen_pano_ids = {}
  while True:
    observation = env.observation()
    view_image = interleave(observation['view_image'],
                            FLAGS.width, FLAGS.height)
    graph_image = interleave(observation['graph_image'],
                             FLAGS.width, FLAGS.height)
    screen_buffer = np.concatenate((view_image, graph_image), axis=1)
    pygame.surfarray.blit_array(screen, screen_buffer)
    pygame.display.update()

    for event in pygame.event.get():
      if (event.type == pygame.QUIT or
          (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
        return
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_p:
          filename = time.strftime('oracle_agent_%Y%m%d_%H%M%S.bmp')
          pygame.image.save(screen, filename)

    # Take a step given the previous action and record the reward.
    _, reward, done, info = env.step(action)
    sum_rewards += reward
    if (reward > 0) and (info['current_goal_id'] is not previous_goal_id):
      sum_rewards_at_goal += reward
      seen_pano_ids = {}
    previous_goal_id = info['current_goal_id']
    if done:
      print('Episode reward: {}'.format(sum_rewards))
      if FLAGS.stats_path:
        with open(FLAGS.stats_path, 'a') as f:
          f.write(str(sum_rewards) + '\t' + str(sum_rewards_at_goal) + '\n')
      sum_rewards = 0
      sum_rewards_at_goal = 0

    # Determine the next pano and bearing to that pano.
    current_pano_id = info['current_pano_id']
    next_pano_id = info['next_pano_id']
    bearing = info['bearing_to_next_pano']
    logging.info('Current pano: %s, next pano %s at %f',
                 current_pano_id, next_pano_id, bearing)

    # Maintain the count of pano visits, in case the agent gets stuck.
    if current_pano_id in seen_pano_ids:
      seen_pano_ids[current_pano_id] += 1
    else:
      seen_pano_ids[current_pano_id] = 1

    # Bearing-based navigation.
    if bearing > TOL_BEARING:
      if bearing > TOL_BEARING + 2 * FLAGS.horizontal_rot:
        action = 3 * FLAGS.horizontal_rot * action_spec['horizontal_rotation']
      else:
        action = FLAGS.horizontal_rot * action_spec['horizontal_rotation']
    elif bearing < -TOL_BEARING:
      if bearing < -TOL_BEARING - 2 * FLAGS.horizontal_rot:
        action = -3 * FLAGS.horizontal_rot * action_spec['horizontal_rotation']
      else:
        action = -FLAGS.horizontal_rot * action_spec['horizontal_rotation']
    else:
      action = action_spec['move_forward']

      # Sometimes, two panos B and C are close to each other, which causes
      # cyclic loops: A -> C -> A -> C -> A... whereas agent wants to go A -> B.
      # There is a simple strategy to get out of that A - C loop: detect that A
      # has been visited a large number of times in the current trajectory, then
      # instead of moving forward A -> B and ending up in C, directly jump to B.
      # First, we check if the agent has spent more time in a pano than required
      # to make a full U-turn...
      if seen_pano_ids[current_pano_id] > (180.0 / FLAGS.horizontal_rot):
        # ... then we teleport to the desired location and turn randomly.
        logging.info('Teleporting from %s to %s', current_pano_id, next_pano_id)
        _ = env.goto(next_pano_id, np.random.randint(359))

def main(argv):
  config = {'width': FLAGS.width,
            'height': FLAGS.height,
            'field_of_view': FLAGS.field_of_view,
            'graph_width': FLAGS.width,
            'graph_height': FLAGS.height,
            'graph_zoom': FLAGS.graph_zoom,
            'goal_timeout': FLAGS.frame_cap,
            'frame_cap': FLAGS.frame_cap,
            'full_graph': (FLAGS.start_pano == ''),
            'start_pano': FLAGS.start_pano,
            'min_graph_depth': FLAGS.graph_depth,
            'max_graph_depth': FLAGS.graph_depth,
            'proportion_of_panos_with_coins':
                FLAGS.proportion_of_panos_with_coins,
            'action_spec': 'streetlearn_fast_rotate',
            'observations': ['view_image', 'graph_image', 'yaw', 'pitch']}
  config = default_config.ApplyDefaults(config)
  game = courier_game.CourierGame(config)
  env = streetlearn.StreetLearn(FLAGS.dataset_path, config, game)
  env.reset()
  pygame.init()
  screen = pygame.display.set_mode((FLAGS.width, FLAGS.height * 2))
  loop(env, screen)

if __name__ == '__main__':
  app.run(main)
