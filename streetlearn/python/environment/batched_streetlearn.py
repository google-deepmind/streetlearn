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

"""The Batched StreetLearn RL environment.

Episodes take place either in a mini-map created by performing a breadth-first
traversal of the StreetView graph starting from a starting location, or in
the entire fully-connected graph. Multiple StreetLearn environments are
instantiated, sharing the same cache of panoramas, action space and observation
specs.

Observations:
{
  view_image: numpy array of dimension [batch_size, 3, height, width] containing
    the street imagery.
  graph_image: numpy array of dimension
    [batch_size, 3, graph_height, graph_width] containing the map graph images.
  view_image_hwc: numpy array of dimension [batch_size, height, width, 3]
    containing the street imagery.
  graph_image_hwc: numpy array, dimension
    [batch_size, graph_height, graph_width, 3] containing the map graph images.
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np

from streetlearn.python.environment import streetlearn


class BatchedStreetLearn(object):
  """The Batched Streetlearn environment."""

  def __init__(self, dataset_path, configs, games, num_env_per_shared_cache=1):
    """Construct the StreetLearn environment.

    Args:
      dataset_path: filesystem path where the dataset resides.
      configs: list of batch_size elements, each element being a dictionary
        containing various config settings, that will be extended with defaults
        from default_config.DEFAULT_CONFIG.
      games: list of batch_size instances of Game.
      num_env_per_shared_cache: number of environments that share the same cache
        By default equal to 1 (no cache sharing).
    """
    # Check that batch_size is a multiple of num_env_per_shared_cache and that
    # the action_spec, rotation_speed and observations are compatible between
    # all environments.
    batch_size = len(games)
    assert batch_size > 0
    assert num_env_per_shared_cache > 0
    num_env_per_shared_cache = min(num_env_per_shared_cache, batch_size)
    num_unique_node_caches = int(batch_size / num_env_per_shared_cache)
    logging.info('batch_size: %d, num_env_per_shared_cache: %d',
                 batch_size, num_env_per_shared_cache)
    logging.info('num_unique_node_caches: %d', num_unique_node_caches)
    assert (num_env_per_shared_cache * num_unique_node_caches) == batch_size
    assert len(configs) == batch_size
    for k in range(1, batch_size):
      assert configs[0]['action_spec'] == configs[k]['action_spec']
      assert configs[0]['rotation_speed'] == configs[k]['rotation_speed']
      observations = configs[k]['observations'].sort()
      assert configs[0]['observations'].sort() == observations

    # Instantiate the environments.
    self._envs = []
    k = 0
    for i in range(num_unique_node_caches):
      logging.info('Instantiating environment %d with a new node_cache', k)
      self._envs.append(streetlearn.StreetLearn(
          dataset_path, configs[k], games[k]))
      k += 1
      for j in range(1, num_env_per_shared_cache):
        logging.info('Instantiating environment %d reusing last node_cache', k)
        self._envs.append(streetlearn.StreetLearn(
            dataset_path, configs[k], games[k], self._envs[k-1].engine))
        k += 1

    # Preallocate the matrices for the batch observations.
    self._observation_batch = {}
    for item in self._envs[0]._observations:
      if item.observation_spec == [0]:
        batched_shape = [batch_size,]
      else:
        batched_shape = [batch_size,] + item.observation_spec
      batched_obs = np.zeros(batched_shape, dtype=item.observation_spec_dtypes)
      self._observation_batch[item.name] = batched_obs
    self._batch_size = batch_size

  @property
  def config(self):
    return [env.config for env in self._envs]

  @property
  def seed(self):
    return [env.seed for env in self._envs]

  @property
  def game(self):
    return [env.game for env in self._envs]

  @property
  def field_of_view(self):
    return [env.field_of_view for env in self._envs]

  @property
  def current_pano_id(self):
    return [env.current_pano_id for env in self._envs]

  @property
  def frame_cap(self):
    return [env.frame_cap for env in self._envs]

  @frame_cap.setter
  def frame_cap(self, value):
    for env in self._envs:
      env.frame_cap(value)

  @property
  def frame_count(self):
    return [env.frame_count for env in self._envs]

  def graph(self):
    """Return a list of graphs for all the environments."""
    return [env.graph for env in self._envs]

  @property
  def neighbor_resolution(self):
    return [env.neighbor_resolution for env in self._envs]

  @property
  def bbox_lat_min(self):
    return [env.bbox_lat_min for env in self._envs]

  @property
  def bbox_lat_max(self):
    return [env.bbox_lat_max for env in self._envs]

  @property
  def bbox_lng_min(self):
    return [env.bbox_lng_min for env in self._envs]

  @property
  def bbox_lng_max(self):
    return [env.bbox_lng_max for env in self._envs]

  def observation_spec(self):
    """Returns the observation spec, dependent on the observation format."""
    return {name: list(item.shape)
            for name, item in self._observation_batch.items()}

  def action_set(self):
    """Returns the set of actions, mapping integer actions to 1D arrays."""
    return self._envs[0].action_set()

  def action_spec(self):
    """Returns the action spec."""
    return self._envs[0].action_spec()

  def reset(self):
    """Start a new episode in all environments."""
    for env in self._envs:
      env.reset()

  def goto(self, env_id, pano_id, yaw):
    """Go to a specific pano and yaw in the environment.

    Args:
      env_id: an integer ID for the environment.
      pano_id: a string containing the ID of a pano.
      yaw: a float with relative yaw w.r.t. north.
    Returns:
      observation: tuple with observations.
    """
    return self._envs[env_id].goto(pano_id, yaw)

  def step(self, action):
    """Takes a step in all the environments, and returns results in all envs.

    Args:
      action: a list of 1d arrays containing a combination of actions in each
        environment.
    Returns:
      observation: tuple with batched observations for the last time step.
      reward: list of scalar rewards at the last time step.
      done: list of booleans indicating the end of an episode.
      info: list of dictionaries with additional debug information.
    """
    for action_k, env in zip(action, self._envs):
      env.step(action_k)

    # Return
    return self.observation(), self.reward(), self.done(), self.info()

  def observation(self):
    """Returns the batched observations for the last time step."""
    for k in range(self._batch_size):
      env = self._envs[k]
      for name in self._observation_batch:
        obs_k = env.observation()[name]
        if obs_k is not None:
          self._observation_batch[name][k, ...] = obs_k
    return {name:item for name, item in self._observation_batch.items()}

  def reward(self):
    """Returns the list of rewards for the last time step."""
    return [env.reward() for env in self._envs]

  def prev_reward(self):
    """Returns the list of rewards for the previous time step."""
    return [env.prev_reward() for env in self._envs]

  def prev_action(self):
    """Returns the list of actions for the previous time step."""
    return [env.prev_action() for env in self._envs]

  def done(self):
    """Return the list of flags indicating the end of the current episode."""
    return [env.done() for env in self._envs]

  def info(self):
    """Return a list of dictionaries with env. info at the current step."""
    return [env.info() for env in self._envs]

  def get_metadata(self, pano_id):
    """Return the metadata corresponding to the selected pano.

    Args:
      pano_id: a string containing the ID of a pano.
    Returns:
      metadata: a protocol buffer with the pano metadata.
    """
    return self._envs[0].get_metadata(pano_id)

  @property
  def cache_size(self):
    return [env._engine.GetNodeCacheSize() for env in self._envs]

  def render(self):
    """Empty function, for compatibility with OpenAI Gym."""
    pass
