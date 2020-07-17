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

import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pygame
from six.moves import range

from streetlearn.python.environment import default_config
from streetlearn.python.environment import goal_instruction_game
from streetlearn.python.environment import incremental_instruction_game
from streetlearn.python.environment import step_by_step_instruction_game
from streetlearn.python.environment import streetlearn

FLAGS = flags.FLAGS
flags.DEFINE_integer('width', 400, 'Observation and map width.')
flags.DEFINE_integer('height', 400, 'Observation and map height.')
flags.DEFINE_integer('field_of_view', 60, 'Field of view.')
flags.DEFINE_integer('graph_zoom', 1, 'Zoom level.')
flags.DEFINE_boolean('graph_black_on_white', False,
                     'Show graph as black on white. False by default.')
flags.DEFINE_integer('width_text', 300, 'Text width.')
flags.DEFINE_integer('font_size', 16, 'Font size.')
flags.DEFINE_float('horizontal_rot', 10, 'Horizontal rotation step (deg).')
flags.DEFINE_string('dataset_path', None, 'Dataset path.')
flags.DEFINE_string('instruction_file', None, 'Instruction path.')
flags.DEFINE_string(
    'game',
    'incremental_instruction_game',
    'Game name [goal_instruction_game|'
    'incremental_instruction_game|step_by_step_instruction_game]')
flags.DEFINE_float('reward_at_waypoint', 0.5, 'Reward at waypoint.')
flags.DEFINE_float('reward_at_goal', 1.0, 'Reward at waypoint.')
flags.DEFINE_integer('num_instructions', 5, 'Number of instructions.')
flags.DEFINE_integer('max_instructions', 5, 'Maximum number of instructions.')
flags.DEFINE_string('start_pano', '',
                     'Pano at root of partial graph (default: full graph).')
flags.DEFINE_integer('graph_depth', 200, 'Depth of the pano graph.')
flags.DEFINE_boolean('show_shortest_path', True,
                     'Whether to highlight the shortest path in the UI.')
flags.DEFINE_integer('frame_cap', 1000, 'Number of frames / episode.')
flags.DEFINE_string('stats_path', None, 'Statistics path.')
flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('instruction_file')

COLOR_WAYPOINT = (0, 178, 178)
COLOR_GOAL = (255, 0, 0)
COLOR_INSTRUCTION = (255, 255, 255)


def blit_instruction(screen, instruction, font, color, x_min, y, x_max):
  """Render and blit a multiline instruction onto the PyGame screen."""
  words = instruction.split()
  space_width = font.size(' ')[0]
  x = x_min
  for word in words:
    word_surface = font.render(word, True, color)
    word_width, word_height = word_surface.get_size()
    if x + word_width >= x_max:
      x = x_min
      y += word_height
    screen.blit(word_surface, (x, y))
    x += word_width + space_width

def loop(env, screen, x_max, y_max, subsampling, font):
  """Main loop of the oracle agent."""
  screen_buffer = np.zeros((x_max, y_max, 3), np.uint8)
  action = np.array([0, 0, 0, 0])
  action_spec = env.action_spec()
  sum_rewards = 0
  sum_rewards_at_goal = 0
  previous_goal_id = None
  while True:

    # Take a step given the previous action and record the reward.
    observation, reward, done, info = env.step(action)
    sum_rewards += reward
    if (reward > 0) and (info['current_goal_id'] is not previous_goal_id):
      sum_rewards_at_goal += reward
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
    bearing_info = info['bearing_to_next_pano']
    bearing = observation['ground_truth_direction']
    logging.info('Current pano: %s, next pano %s at %f (%f)',
                 current_pano_id, next_pano_id, bearing, bearing_info)
    current_step = info.get('current_step', -1)

    # Bearing-based navigation.
    if bearing > FLAGS.horizontal_rot:
      if bearing > FLAGS.horizontal_rot + 2 * FLAGS.horizontal_rot:
        action = 3 * FLAGS.horizontal_rot * action_spec['horizontal_rotation']
      else:
        action = FLAGS.horizontal_rot * action_spec['horizontal_rotation']
    elif bearing < -FLAGS.horizontal_rot:
      if bearing < -FLAGS.horizontal_rot - 2 * FLAGS.horizontal_rot:
        action = -3 * FLAGS.horizontal_rot * action_spec['horizontal_rotation']
      else:
        action = -FLAGS.horizontal_rot * action_spec['horizontal_rotation']
    else:
      action = action_spec['move_forward']

    # Draw the observations (view, graph, thumbnails, instructions).
    view_image = observation['view_image_hwc']
    graph_image = observation['graph_image_hwc']
    screen_buffer[:FLAGS.width, :FLAGS.height, :] = view_image.swapaxes(0, 1)
    screen_buffer[:FLAGS.width, FLAGS.height:(FLAGS.height*2), :] = (
        graph_image.swapaxes(0, 1))
    thumb_image = np.copy(observation['thumbnails'])
    for k in range(FLAGS.max_instructions+1):
      if k != current_step:
        thumb_image[k, :, :, :] = thumb_image[k, :, :, :] / 2
    thumb_image = thumb_image.reshape(
        FLAGS.height * (FLAGS.max_instructions + 1), FLAGS.width, 3)
    thumb_image = thumb_image.swapaxes(0, 1)
    thumb_image = thumb_image[::subsampling, ::subsampling, :]
    screen_buffer[FLAGS.width:(FLAGS.width+thumb_image.shape[0]),
                  0:thumb_image.shape[1],
                  :] = thumb_image
    pygame.surfarray.blit_array(screen, screen_buffer)
    instructions = observation['instructions'].decode('utf-8')
    instructions = instructions.split('|')
    instructions.append('[goal]')
    x_min = x_max - FLAGS.width_text + 10
    y = 10
    for k in range(len(instructions)):
      instruction = instructions[k]
      if k == current_step:
        color = COLOR_WAYPOINT
      elif k == len(instructions) - 1:
        color = COLOR_GOAL
      else:
        color = COLOR_INSTRUCTION
      blit_instruction(screen, instruction, font, color, x_min, y, x_max)
      y += int(FLAGS.height / subsampling)
    pygame.display.update()

    for event in pygame.event.get():
      if (event.type == pygame.QUIT or
          (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
        return
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_p:
          filename = time.strftime('/tmp/oracle_agent_%Y%m%d_%H%M%S.bmp')
          pygame.image.save(screen, filename)

def main(argv):
  config = {'width': FLAGS.width,
            'height': FLAGS.height,
            'field_of_view': FLAGS.field_of_view,
            'status_height': 0,
            'graph_width': FLAGS.width,
            'graph_height': FLAGS.height,
            'graph_zoom': FLAGS.graph_zoom,
            'graph_black_on_white': FLAGS.graph_black_on_white,
            'show_shortest_path': FLAGS.show_shortest_path,
            'calculate_ground_truth': True,
            'goal_timeout': FLAGS.frame_cap,
            'frame_cap': FLAGS.frame_cap,
            'full_graph': (FLAGS.start_pano == ''),
            'start_pano': FLAGS.start_pano,
            'min_graph_depth': FLAGS.graph_depth,
            'max_graph_depth': FLAGS.graph_depth,
            'reward_at_waypoint': FLAGS.reward_at_waypoint,
            'reward_at_goal': FLAGS.reward_at_goal,
            'instruction_file': FLAGS.instruction_file,
            'num_instructions': FLAGS.num_instructions,
            'max_instructions': FLAGS.max_instructions,
            'proportion_of_panos_with_coins': 0.0,
            'action_spec': 'streetlearn_fast_rotate',
            'observations': ['view_image_hwc', 'graph_image_hwc', 'yaw',
                             'thumbnails', 'instructions',
                             'ground_truth_direction']}
  # Configure game and environment.
  config = default_config.ApplyDefaults(config)
  if FLAGS.game == 'goal_instruction_game':
    game = goal_instruction_game.GoalInstructionGame(config)
  elif FLAGS.game == 'incremental_instruction_game':
    game = incremental_instruction_game.IncrementalInstructionGame(config)
  elif FLAGS.game == 'step_by_step_instruction_game':
    game = step_by_step_instruction_game.StepByStepInstructionGame(config)
  else:
    print('Unknown game: [{}]'.format(FLAGS.game))
    print('Run instruction_following_oracle_agent --help.')
    return
  env = streetlearn.StreetLearn(FLAGS.dataset_path, config, game)
  env.reset()

  # Configure pygame.
  pygame.init()
  pygame.font.init()
  subsampling = int(np.ceil((FLAGS.max_instructions + 1) / 2))
  x_max = FLAGS.width + int(FLAGS.width / subsampling) + FLAGS.width_text
  y_max = FLAGS.height * 2
  logging.info('Rendering images at %dx%d, thumbnails subsampled by %d',
               x_max, y_max, subsampling)
  screen = pygame.display.set_mode((x_max, y_max))
  font = pygame.font.SysFont('arial', FLAGS.font_size)

  loop(env, screen, x_max, y_max, subsampling, font)

if __name__ == '__main__':
  app.run(main)
