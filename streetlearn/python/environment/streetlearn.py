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

"""The StreetLearn RL environment.

Episodes take place either in a mini-map created by performing a breadth-first
traversal of the StreetView graph starting from a starting location, or in
the entire fully-connected graph.

Observations:
{
  view_image: numpy array of dimension [3, height, width] containing the
    street imagery.
  graph_image: numpy array of dimension [3, graph_height, graph_width]
    containing the map graph image.
  metadata: learning_deepmind.datasets.street_learn.Pano proto
    without compressed_image.
  target_metadata: learning_deepmind.datasets.street_learn.Pano proto
    without compressed_image.
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import inflection
import numpy as np
import six

from streetlearn.python.environment import observations
from streetlearn.python.environment import default_config
from streetlearn.engine.python import streetlearn_engine

_MIN_ZOOM = 1
_MAX_ZOOM = 32

def _action(*entries):
  return np.array(entries, dtype=np.float)

ACTIONS = {
    'move_forward': _action(1, 0.0, 0.0, 0.0),
    'horizontal_rotation': _action(0, 1.0, 0.0, 0.0),
    'vertical_rotation': _action(0, 0.0, 1.0, 0.0),
    'map_zoom': _action(0, 0.0, 0.0, 1.0),
}

NUM_ACTIONS = 4

ACTION_SETS = {
    "streetlearn_default": lambda rotation_speed: (
        ACTIONS["move_forward"],
        ACTIONS["horizontal_rotation"] * (-rotation_speed),
        ACTIONS["horizontal_rotation"] * rotation_speed),
    "streetlearn_fast_rotate": lambda rotation_speed: (
        ACTIONS["move_forward"],
        ACTIONS["horizontal_rotation"] * (-rotation_speed),
        ACTIONS["horizontal_rotation"] * (-rotation_speed * 3),
        ACTIONS["horizontal_rotation"] * rotation_speed,
        ACTIONS["horizontal_rotation"] * rotation_speed * 3),
    "streetlearn_tilt": lambda rotation_speed: (
        ACTIONS["move_forward"],
        ACTIONS["horizontal_rotation"] * (-rotation_speed),
        ACTIONS["horizontal_rotation"] * rotation_speed,
        ACTIONS["vertical_rotation"] * rotation_speed,
        ACTIONS["vertical_rotation"] * (-rotation_speed)),
}

def get_action_set(action_spec, rotation_speed):
  """Returns the set of StreetLearn actions for the given action_spec."""

  # If action_spec is a string, it should be the name of a standard action set.
  if isinstance(action_spec, basestring):
    if action_spec not in ACTION_SETS:
      raise ValueError("Unrecognized action specification %s." % action_spec)
    else:
      return np.array(ACTION_SETS[action_spec](rotation_speed), dtype=np.float)
  raise ValueError("Action specification %s not a string." % action_spec)

class StreetLearn(object):
  """The Streetlearn environment."""

  def __init__(self, dataset_path, config, game):
    """Construct the StreetLearn environment.

    Args:
      dataset_path: filesystem path where the dataset resides.
      config: dictionary containing various config settings. Will be extended
        with defaults from default_config.DEFAULT_CONFIG.
      game: an instance of Game.
    """
    assert game, "Did not provide game."
    logging.info('dataset_path:')
    logging.info(dataset_path)
    logging.info('config:')
    logging.info(config)
    logging.info('game:')
    logging.info(game)
    self._config = default_config.ApplyDefaults(config)
    self._seed = self._config["seed"]
    self._start_pano_id = self._config["start_pano"]
    self._zoom = self._config["graph_zoom"]
    self._frame_cap = self._config["frame_cap"]
    self._field_of_view = self._config["field_of_view"]
    self._neighbor_resolution = self._config["neighbor_resolution"]
    self._sample_graph_depth = self._config["sample_graph_depth"]
    self._min_graph_depth = self._config["min_graph_depth"]
    self._max_graph_depth = self._config["max_graph_depth"]
    self._full_graph = self._config["full_graph"]
    self._color_for_observer = self._config["color_for_observer"]
    self._action_spec = self._config["action_spec"]
    self._rotation_speed = self._config["rotation_speed"]
    self._auto_reset = self._config["auto_reset"]
    self._action_set = get_action_set(self._action_spec, self._rotation_speed)
    logging.info('Action set:')
    logging.info(self._action_set)
    self._bbox_lat_min = self._config["bbox_lat_min"]
    self._bbox_lat_max = self._config["bbox_lat_max"]
    self._bbox_lng_min = self._config["bbox_lng_min"]
    self._bbox_lng_max = self._config["bbox_lng_max"]

    self._game = game
    self._current_pano_id = None
    self._episode_id = -1
    self._frame_count = 0

    self._engine = streetlearn_engine.StreetLearnEngine.Create(
        dataset_path,
        width=self._config["width"],
        height=self._config["height"],
        graph_width=self._config["graph_width"],
        graph_height=self._config["graph_height"],
        status_height=self._config["status_height"],
        field_of_view=self._field_of_view,
        min_graph_depth=self._min_graph_depth,
        max_graph_depth=self._max_graph_depth,
        max_cache_size=self._config["max_cache_size"])
    assert self._engine, "Could not initialise engine from %r." % dataset_path
    self._observations = []
    for name in self._config["observations"]:
      try:
        self._observations.append(observations.Observation.create(name, self))
      except ValueError as e:
        logging.warning(str(e))

    self._reward = 0
    self._done = False
    self._info = {}

  @property
  def config(self):
    return self._config

  @property
  def game(self):
    return self._game

  @property
  def field_of_view(self):
    return self._field_of_view

  @property
  def current_pano_id(self):
    return self._current_pano_id

  @property
  def frame_cap(self):
    return self._frame_cap

  @frame_cap.setter
  def frame_cap(self, value):
    self._frame_cap = value

  @property
  def frame_count(self):
    return self._frame_count

  @property
  def graph(self):
    return self._graph

  @property
  def engine(self):
    return self._engine

  @property
  def neighbor_resolution(self):
    return self._neighbor_resolution

  @property
  def bbox_lat_min(self):
    return self._bbox_lat_min

  @property
  def bbox_lat_max(self):
    return self._bbox_lat_max

  @property
  def bbox_lng_min(self):
    return self._bbox_lng_min

  @property
  def bbox_lng_max(self):
    return self._bbox_lng_max

  def observation_spec(self):
    """Returns the observation spec, dependent on the observation format."""
    return {observation.name: observation.observation_spec
            for observation in self._observations}

  def action_spec(self):
    """Returns the action spec."""
    return ACTIONS

  def reset(self):
    """Start a new episode."""
    self._frame_count = 0
    self._episode_id += 1
    if self._sample_graph_depth:
      max_depth = np.random.randint(self._min_graph_depth,
                                    self._max_graph_depth + 1)
      self._engine.SetGraphDepth(self._min_graph_depth, max_depth)

    self._engine.InitEpisode(self._episode_id, self._seed)

    # Build a new graph if we don't have one yet.
    if not self._current_pano_id:
      if self._full_graph:
        self._current_pano_id = self._engine.BuildEntireGraph()
      elif self._start_pano_id:
        self._current_pano_id = self._engine.BuildGraphWithRoot(
            self._start_pano_id)
      else:
        self._current_pano_id = self._engine.BuildRandomGraph()
      logging.info('Built new graph with root %s', self._current_pano_id)
    # else respawn in current graph.
    elif not self._start_pano_id:
      self._current_pano_id = np.random.choice(self._engine.GetGraph().keys())
      self._engine.SetPosition(self._current_pano_id)
      logging.info('Reusing existing graph and respawning %s',
                   self._current_pano_id)

    self._graph = self._engine.GetGraph()
    highlighted_panos = self._game.on_reset(self)
    self._engine.InitGraphRenderer(self._color_for_observer, highlighted_panos)
    self._engine.SetZoom(_MAX_ZOOM)

  def step(self, action):
    """Takes a step in the environment.

    Args:
      action: a 1d array containing a combination of actions.
    Returns:
      observation: tuple with observations for the last time step.
      reward: scalar reward at the last time step.
      done: boolean indicating the end of an episode.
      info: dictionary with additional debug information.
    """
    self._frame_count += 1
    if type(action) != np.ndarray:
      action = np.array(action, dtype=np.float)
    assert action.size == NUM_ACTIONS, "Wrong number of actions."
    move_forward = np.dot(action, ACTIONS['move_forward'])
    horizontal_rotation = np.dot(action, ACTIONS['horizontal_rotation'])
    vertical_rotation = np.dot(action, ACTIONS['vertical_rotation'])
    map_zoom = np.dot(action, ACTIONS['map_zoom'])

    if move_forward:
      self._current_pano_id = self._engine.MoveToNextPano()
    if map_zoom > 0:
      self._zoom = min(self._zoom * 2, _MAX_ZOOM)
    elif map_zoom < 0:
      self._zoom = max(self._zoom / 2, _MIN_ZOOM)

    if horizontal_rotation or vertical_rotation:
      self._engine.RotateObserver(horizontal_rotation, vertical_rotation)

    self._engine.SetZoom(self._zoom)
    self._game.on_step(self)

    # Update the reward and done flag. Because we do not know the code logic
    # inside each game, it is safer to obtain these immediately after step(),
    # and store them for subsequent calls to reward(), done() and info().
    self._reward = self._game.get_reward(self)
    self._done = (self._frame_count > self._frame_cap) or self._game.done()
    self._info = self._game.get_info(self)
    if self._auto_reset and self._done:
      self.reset()

    # Return
    return self.observation(), self.reward(), self.done(), self.info()

  def observation(self):
    """Returns the observations for the last time step."""
    return {item.name: item.observation for item in self._observations}

  def reward(self):
    """Returns the reward for the last time step."""
    return self._reward

  def done(self):
    """Return a flag indicating the end of the current episode."""
    return self._done

  def info(self):
    """Return a dictionary with environment information at the current step."""
    return self._info

  def get_metadata(self, pano_id):
    """Return the metadata corresponding to the selected pano.

    Args:
      pano_id: a string containing the ID of a pano.
    Returns:
      metadata: a protocol buffer with the pano metadata.
    """
    if hasattr(self, '_graph') and pano_id in self.graph:
      return self._engine.GetMetadata(pano_id)
    else:
      return None

  def render(self):
    """Empty function, for compatibility with OpenAI Gym."""
    pass

