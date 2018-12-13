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

"""Settings for the StreetLearn environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from streetlearn.engine.python import color
from streetlearn.python.environment import coin_game
from streetlearn.python.environment import courier_game
from streetlearn.python.environment import exploration_game


DEFAULT_CONFIG = {
    'seed': 1234,
    'width': 320,
    'height': 240,
    'graph_width': 320,
    'graph_height': 240,
    'status_height': 10,
    'field_of_view': 60,
    'min_graph_depth': 200,
    'max_graph_depth': 200,
    'max_cache_size': 1000,
    'bbox_lat_min': -90.0,
    'bbox_lat_max': 90.0,
    'bbox_lng_min': -180.0,
    'bbox_lng_max': 180.0,
    'max_reward_per_goal': 10.0,
    'min_radius_meters': 100.0,
    'max_radius_meters': 200.0,
    'goal_timeout': 1000,
    'frame_cap': 1000,
    'full_graph': True,
    'sample_graph_depth': True,
    'start_pano': '',
    'graph_zoom': 32,
    'neighbor_resolution': 8,
    'color_for_touched_pano': color.Color(1.0, 0.5, 0.5),
    'color_for_observer': color.Color(0.2, 0.2, 1.0),
    'color_for_coin': color.Color(1.0, 1.0, 0.0),
    'color_for_goal': color.Color(1.0, 0.0, 0.0),
    'color_for_shortest_path': color.Color(0.5, 0.0, 0.5),
    'observations': ['view_image', 'graph_image'],
    'reward_per_coin': 1.0,
    'proportion_of_panos_with_coins': 0.5,
    'level_name': 'coin_game',
    'action_spec': 'streetlearn_fast_rotate',
    'rotation_speed': 22.5,
}

NAME_TO_LEVEL = {
    'coin_game': coin_game.CoinGame,
    'courier_game': courier_game.CourierGame,
    'exploration_game': exploration_game.ExplorationGame,
}

def ApplyDefaults(config):
  result = copy.copy(config)
  for default_key, default_value in DEFAULT_CONFIG.items():
    if not default_key in result:
      result[default_key] = default_value
    else:
      assert type(default_value) == type(result[default_key])
  return result

def CreateLevel(name, config):
  assert name in NAME_TO_LEVEL, "Unknown level name: %r" % name
  return NAME_TO_LEVEL[name](config)
