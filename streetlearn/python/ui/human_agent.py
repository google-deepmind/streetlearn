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

"""Basic human agent for StreetLearn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import time
import numpy as np
import pygame

from streetlearn.engine.python import color
from streetlearn.python.environment import coin_game
from streetlearn.python.environment import courier_game
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
flags.DEFINE_integer('width_text', 300, 'Text width.')
flags.DEFINE_integer('font_size', 30, 'Font size.')
flags.DEFINE_float('horizontal_rot', 10, 'Horizontal rotation step (deg).')
flags.DEFINE_float('vertical_rot', 10, 'Vertical rotation step (deg).')
flags.DEFINE_string('dataset_path', None, 'Dataset path.')
flags.DEFINE_string('instruction_file', None, 'Instruction path.')
flags.DEFINE_string(
    'game',
    'coin_game',
    'Game name [coin_game|courier_game|goal_instruction_game|'
    'incremental_instruction_game|step_by_step_instruction_game]')
flags.DEFINE_float('reward_at_waypoint', 0.5, 'Reward at waypoint.')
flags.DEFINE_float('reward_at_goal', 1.0, 'Reward at waypoint.')
flags.DEFINE_integer('num_instructions', 5, 'Number of instructions.')
flags.DEFINE_integer('max_instructions', 5, 'Maximum number of instructions.')
flags.DEFINE_string('start_pano', '',
                     'Pano at root of partial graph (default: full graph).')
flags.DEFINE_integer('graph_depth', 200, 'Depth of the pano graph.')
flags.DEFINE_boolean('hide_goal', False,
                     'Whether to hide the goal location on the graph.')
flags.DEFINE_boolean('show_shortest_path', True,
                     'Whether to highlight the shortest path in the UI.')
flags.DEFINE_integer('frame_cap', 1000, 'Number of frames / episode.')
flags.DEFINE_float('proportion_of_panos_with_coins', 0.0, 'Proportion of coins.')
flags.mark_flag_as_required('dataset_path')

COLOR_WAYPOINT = (0, 178, 178)
COLOR_GOAL = (255, 0, 0)
COLOR_INSTRUCTION = (255, 255, 255)


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

def loop(env, screen, x_max, y_max, subsampling=None, font=None):
  """Main loop of the human agent."""
  screen_buffer = np.zeros((x_max, y_max, 3), np.uint8)
  action = np.array([0, 0, 0, 0])
  action_spec = env.action_spec()
  sum_rewards = 0
  sum_rewards_at_goal = 0
  previous_goal_id = None
  while True:

    # Take a step through the environment and record the reward.
    observation, reward, done, info = env.step(action)
    sum_rewards += reward
    pano_id = env.current_pano_id
    if reward > 0:
      print('Collected reward of {} at {}'.format(reward, pano_id))
    if done:
      print('Episode reward: {}'.format(sum_rewards))
      sum_rewards = 0
      sum_rewards_at_goal = 0

    # Draw the observations (view, graph).
    observation = env.observation()
    view_image = interleave(observation['view_image'],
                            FLAGS.width, FLAGS.height)
    graph_image = interleave(observation['graph_image'],
                             FLAGS.width, FLAGS.height)

    if FLAGS.game == 'coin_game' or FLAGS.game == 'courier_game':
      screen_buffer = np.concatenate((view_image, graph_image), axis=1)
      pygame.surfarray.blit_array(screen, screen_buffer)
    else:
      # Draw extra observations (thumbnails, instructions).
      screen_buffer[:FLAGS.width, :FLAGS.height, :] = view_image
      screen_buffer[:FLAGS.width, FLAGS.height:(FLAGS.height*2), :] = graph_image
      thumb_image = np.copy(observation['thumbnails'])
      current_step = info.get('current_step', -1)
      for k in range(FLAGS.max_instructions+1):
        if k != current_step:
          thumb_image[k, :, :, :] = thumb_image[k, :, :, :] / 2
      thumb_image = np.swapaxes(thumb_image, 0, 1)
      thumb_image = interleave(thumb_image,
                               FLAGS.width,
                               FLAGS.height * (FLAGS.max_instructions + 1))
      thumb_image = thumb_image[::subsampling, ::subsampling, :]
      screen_buffer[FLAGS.width:(FLAGS.width+thumb_image.shape[0]),
                    0:thumb_image.shape[1],
                    :] = thumb_image
      pygame.surfarray.blit_array(screen, screen_buffer)
      instructions = observation['instructions'].split('|')
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

    action_spec = env.action_spec()
    action = np.array([0, 0, 0, 0])

    while True:
      event = pygame.event.wait()
      if event.type == pygame.QUIT:
        return
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          print(pano_id + ': exit')
          return
        if event.key == pygame.K_SPACE:
          action = action_spec['move_forward']
          print(pano_id + ': move')
        elif event.key == pygame.K_p:
          filename = time.strftime('human_agent_%Y%m%d_%H%M%S.bmp')
          pygame.image.save(screen, filename)
        elif event.key == pygame.K_i:
          action = action_spec['map_zoom']
          print(pano_id + ': zoom in')
        elif event.key == pygame.K_o:
          action = -1 * action_spec['map_zoom']
          print(pano_id + ': zoom out')
        elif event.key == pygame.K_a:
          action = -FLAGS.horizontal_rot * action_spec['horizontal_rotation']
          print(pano_id + ': rotate left')
        elif event.key == pygame.K_d:
          action = FLAGS.horizontal_rot * action_spec['horizontal_rotation']
          print(pano_id + ': rotate right')
        elif event.key == pygame.K_w:
          action = -FLAGS.vertical_rot * action_spec['vertical_rotation']
          print(pano_id + ': look up')
        elif event.key == pygame.K_s:
          action = FLAGS.vertical_rot * action_spec['vertical_rotation']
          print(pano_id + ': look down')
      elif event.type == pygame.KEYUP:
        break

def main(argv):
  config = {'width': FLAGS.width,
            'height': FLAGS.height,
            'field_of_view': FLAGS.field_of_view,
            'graph_width': FLAGS.width,
            'graph_height': FLAGS.height,
            'graph_zoom': FLAGS.graph_zoom,
            'show_shortest_path': FLAGS.show_shortest_path,
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
            'proportion_of_panos_with_coins':
                FLAGS.proportion_of_panos_with_coins,
            'observations': ['view_image', 'graph_image', 'yaw', 'thumbnails',
                             'pitch', 'instructions', 'latlng',
                             'target_latlng']}
  if FLAGS.hide_goal:
    config['color_for_goal'] = color.Color(1.0, 1.0, 1.0)
  config = default_config.ApplyDefaults(config)
  if FLAGS.game == 'coin_game':
    game = coin_game.CoinGame(config)
  elif FLAGS.game == 'courier_game':
    game = courier_game.CourierGame(config)
  elif FLAGS.game == 'goal_instruction_game':
    game = goal_instruction_game.GoalInstructionGame(config)
  elif FLAGS.game == 'incremental_instruction_game':
    game = incremental_instruction_game.IncrementalInstructionGame(config)
  elif FLAGS.game == 'step_by_step_instruction_game':
    game = step_by_step_instruction_game.StepByStepInstructionGame(config)
  else:
    print('Unknown game: [{}]'.format(FLAGS.game))
    print('Run with --help for available options.')
    return
  env = streetlearn.StreetLearn(FLAGS.dataset_path, config, game)
  env.reset()

  # Configure pygame.
  pygame.init()
  pygame.font.init()
  if FLAGS.game == 'coin_game' or FLAGS.game == 'courier_game':
    subsampling = 1
    x_max = FLAGS.width
    y_max = FLAGS.height * 2
    logging.info('Rendering images at %dx%d', x_max, y_max)
  else:
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
