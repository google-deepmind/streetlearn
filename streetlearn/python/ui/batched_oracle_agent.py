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

import copy
import time
import numpy as np
import pygame

from streetlearn.python.environment import courier_game
from streetlearn.python.environment import default_config
from streetlearn.python.environment import batched_streetlearn

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('num_env_per_shared_cache', 4, 'Num env per shared cache.')
flags.DEFINE_integer('width', 168, 'Observation and map width.')
flags.DEFINE_integer('height', 168, 'Observation and map height.')
flags.DEFINE_integer('field_of_view', 60, 'Field of view.')
flags.DEFINE_integer('graph_zoom', 1, 'Zoom level.')
flags.DEFINE_boolean('graph_black_on_white', False,
                     'Show graph as black on white. False by default.')
flags.DEFINE_float('horizontal_rot', 22.5, 'Horizontal rotation step (deg).')
flags.DEFINE_string('dataset_path', None, 'Dataset path.')
flags.DEFINE_integer('max_cache_size', 30000, 'Max cache size.')
flags.DEFINE_string('start_pano', '',
                     'Pano at root of partial graph (default: full graph).')
flags.DEFINE_integer('graph_depth', 200, 'Depth of the pano graph.')
flags.DEFINE_integer('frame_cap', 1000, 'Number of frames / episode.')
flags.DEFINE_string('stats_path', None, 'Statistics path.')
flags.DEFINE_float('proportion_of_panos_with_coins', 0, 'Proportion of coins.')
flags.mark_flag_as_required('dataset_path')


TOL_BEARING = 30


def loop(env, screen):
  """Main loop of the oracle agent."""
  action = []
  sum_rewards = []
  sum_rewards_at_goal = []
  previous_goal_id = []
  seen_pano_ids = []
  for _ in range(FLAGS.batch_size):
    action.append(np.array([0, 0, 0, 0]))
    sum_rewards.append(0)
    sum_rewards_at_goal.append(0)
    previous_goal_id.append(None)
    seen_pano_ids.append({})
  action_spec = env.action_spec()
  save_video = False
  frame = 0
  horizontal_rotation = action_spec['horizontal_rotation']
  move_forward = action_spec['move_forward']
  while True:

    # Read the keyboard.
    for event in pygame.event.get():
      if (event.type == pygame.QUIT or
          (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
        return
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_p:
          filename = time.strftime('/tmp/oracle_agent_%Y%m%d_%H%M%S.bmp')
          pygame.image.save(screen, filename)
        if event.key == pygame.K_v:
          save_video = True
        if event.key == pygame.K_c:
          save_video = False
        elif event.key == pygame.K_i:
          for k in range(FLAGS.batch_size):
            action[k] += action_spec['map_zoom']
          print('zoom in')
        elif event.key == pygame.K_o:
          for k in range(FLAGS.batch_size):
            action[k] -= action_spec['map_zoom']
          print('zoom out')

    # Take a step given the previous action.
    observations, reward, done, info = env.step(action)

    # Visualise the observations.
    images = []
    for k in range(FLAGS.batch_size):
      view_image = observations['view_image_hwc'][k, ...]
      graph_image = observations['graph_image_hwc'][k, ...]
      image_k = np.concatenate((view_image, graph_image), axis=0)
      images.append(image_k)
    screen_buffer = np.concatenate(images, axis=1)
    pygame.surfarray.blit_array(screen, screen_buffer.swapaxes(0, 1))
    pygame.display.update()

    # Save a video?
    if save_video:
      filename = time.strftime('/tmp/oracle_agent_video_%Y%m%d_%H%M%S')
      filename += '_' + str(frame) + '.bmp'
      pygame.image.save(screen, filename)
    frame += 1

    # Record the reward.
    for k in range(FLAGS.batch_size):
      sum_rewards[k] += reward[k]
      if ((reward[k] > 0) and
          (info[k]['current_goal_id'] is not previous_goal_id[k])):
        sum_rewards_at_goal[k] += reward[k]
        seen_pano_ids[k] = {}
      previous_goal_id[k] = info[k]['current_goal_id']
      if done[k]:
        num_successes = info[k]['num_successes']
        spl = info[k]['spl']
        spl_without_last_goal = info[k]['spl_without_last_goal']
        print('Episode [{}] reward: {}, goals: {}, SPL: {}/{}'.format(
            k, sum_rewards[k], num_successes, spl, spl_without_last_goal))
        if FLAGS.stats_path:
          with open(FLAGS.stats_path, 'a') as f:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                k, sum_rewards[k], sum_rewards_at_goal[k], num_successes, spl,
                spl_without_last_goal))
        sum_rewards[k] = 0
        sum_rewards_at_goal[k] = 0
    # logging.info('Cache size: %d', env.cache_size())

    # Determine the next pano and bearing to that pano.
    for k in range(FLAGS.batch_size):
      current_pano_id = info[k]['current_pano_id']
      next_pano_id = info[k]['next_pano_id']
      bearing = info[k]['bearing_to_next_pano']
      logging.info('Current pano %d: %s, next pano %s at %f, cache size %d',
                   k, current_pano_id, next_pano_id, bearing, env.cache_size[k])

      # Maintain the count of pano visits, in case the agent gets stuck.
      if current_pano_id in seen_pano_ids[k]:
        seen_pano_ids[k][current_pano_id] += 1
      else:
        seen_pano_ids[k][current_pano_id] = 1

      # Bearing-based navigation.
      if bearing > TOL_BEARING:
        if bearing > TOL_BEARING + 2 * FLAGS.horizontal_rot:
          action[k] = copy.copy(3 * FLAGS.horizontal_rot * horizontal_rotation)
        else:
          action[k] = copy.copy(FLAGS.horizontal_rot * horizontal_rotation)
      elif bearing < -TOL_BEARING:
        if bearing < -TOL_BEARING - 2 * FLAGS.horizontal_rot:
          action[k] = copy.copy(-3 * FLAGS.horizontal_rot * horizontal_rotation)
        else:
          action[k] = copy.copy(-FLAGS.horizontal_rot * horizontal_rotation)
      else:
        action[k] = copy.copy(move_forward)

        # Sometimes, two panos B and C are close to each other, which causes
        # cyclic loops: A -> C -> A -> C -> A... whereas agent wants to go
        # A -> B. There is a simple strategy to get out of that A - C loop:
        # detect that A has been visited a large number of times in the current
        # trajectory, then instead of moving forward A -> B and ending up in C,
        # directly jump to B. First, we check if the agent has spent more time
        # in a pano than required to make a full U-turn...
        if seen_pano_ids[k][current_pano_id] > (180.0 / FLAGS.horizontal_rot):
          # ... then we teleport to the desired location and turn randomly.
          logging.info('Teleporting from %s to %s', current_pano_id,
                       next_pano_id)
          _ = env.goto(k, next_pano_id, np.random.randint(359))

def main(argv):
  config = {'width': FLAGS.width,
            'height': FLAGS.height,
            'field_of_view': FLAGS.field_of_view,
            'graph_width': FLAGS.width,
            'graph_height': FLAGS.height,
            'graph_zoom': FLAGS.graph_zoom,
            'graph_black_on_white': FLAGS.graph_black_on_white,
            'goal_timeout': FLAGS.frame_cap,
            'frame_cap': FLAGS.frame_cap,
            'full_graph': (FLAGS.start_pano == ''),
            'start_pano': FLAGS.start_pano,
            'min_graph_depth': FLAGS.graph_depth,
            'max_graph_depth': FLAGS.graph_depth,
            'max_cache_size': FLAGS.max_cache_size,
            'proportion_of_panos_with_coins':
                FLAGS.proportion_of_panos_with_coins,
            'action_spec': 'streetlearn_fast_rotate',
            'observations': ['view_image_hwc', 'graph_image_hwc', 'yaw',
                             'pitch']}
  config = default_config.ApplyDefaults(config)
  # Create as many configs and games as the batch size.
  games = []
  configs = []
  for k in range(FLAGS.batch_size):
    this_config = copy.copy(config)
    this_config['seed'] = k
    configs.append(this_config)
    games.append(courier_game.CourierGame(this_config))
  env = batched_streetlearn.BatchedStreetLearn(
      FLAGS.dataset_path, configs, games,
      num_env_per_shared_cache=FLAGS.num_env_per_shared_cache)
  env.reset()
  pygame.init()
  screen = pygame.display.set_mode(
      (FLAGS.width * FLAGS.batch_size, FLAGS.height * 2))
  loop(env, screen)

if __name__ == '__main__':
  app.run(main)
