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

"""StreetLearn level for the courier task with random goals/targets.

In this environment, the agent receives a reward for every coin it collects and
an extra reward for locating the goal pano.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import six

from streetlearn.engine.python import color
from streetlearn.python.environment import coin_game


class CourierGame(coin_game.CoinGame):
  """Coin game that gives extra reward for finding the goal pano. A courier goal
  is randomly selected from panos in the graph. On success or timeout, a new
  goal is chosen. The episode ends after a fixed episode length.
  """

  def __init__(self, config):
    """Constructor.

    This coin game gives extra reward for finding the goal pano, and resets the
    goal once the goal has been found (or on timeout). Panos can be assigned
    rewards (coins) randomly and the agent will receive the reward the first
    time they visit these panos.

    Args:
      config: config dict of various settings.
    """
    super(CourierGame, self).__init__(config)
    self._reward_current_goal = config['max_reward_per_goal']
    self._min_radius_meters = config['min_radius_meters']
    self._max_radius_meters = config['max_radius_meters']
    self._goal_timeout = config['goal_timeout']
    self._colors['goal'] = color.Color(*config['color_for_goal'])
    self._colors['shortest_path'] = color.Color(
        *config['color_for_shortest_path'])

    self._num_steps_this_goal = 0
    self._success_inverse_path_len = []
    self._min_distance_reached = np.finfo(np.float32).max
    self._initial_distance_to_goal = np.finfo(np.float32).max
    self._current_goal_id = None
    self._visited_panos = set()
    self._shortest_path = {}
    self._timed_out = False

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Selects a random pano as goal destination and resets episode statistics.

    If there are any coins, clears the set of touched panos and randomly
    generates reward-yielding coins and populates pano_id_to_color.

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    # Populate the list of panos and assign optional coins to panos.
    pano_id_to_color = super(CourierGame, self).on_reset(streetlearn)

    # Assign the goal location to one of the panos.
    self._pick_random_goal(streetlearn)

    # Resets the episode statistics.
    self._num_steps_this_goal = 0
    self._success_inverse_path_len = []
    return pano_id_to_color

  @property
  def goal_id(self):
    """Returns the ID of the goal Pano."""
    return self._current_goal_id

  def get_reward(self, streetlearn):
    """Looks at current_pano_id and collects any reward found there.

    Args:
      streetlearn: A streetlearn instance.
    Returns:
      reward: the reward from the last step.
    """
    # If we have exceeded the maximum steps to look for a goal, reset the goal.
    if self._num_steps_this_goal > self._goal_timeout:
      logging.info('seed %d, frame %d: courier target TIMEOUT (%d steps)',
                   streetlearn.seed, streetlearn.frame_count,
                   self._num_steps_this_goal)
      self._num_steps_this_goal = 0
      self._pick_random_goal(streetlearn)

    (reward, found_goal) = self._compute_reward(streetlearn)

    # If we have found the goal, set a new one and update episode statistics.
    if found_goal:
      logging.info('seed %d, frame %d: courier target FOUND (%d steps)',
                   streetlearn.seed, streetlearn.frame_count,
                   self._num_steps_this_goal)
      _, num_remaining_panos_to_goal_center = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
      self._success_inverse_path_len.append(
          self._compute_spl_current_goal(streetlearn))
      self._pick_random_goal(streetlearn)
      self._num_steps_this_goal = 0

    # Give additional reward if current pano has a coin.
    if streetlearn.current_pano_id in self._coin_pano_id_set:
      reward += self._reward_per_coin
      self._coin_pano_id_set.remove(streetlearn.current_pano_id)

    self._num_steps_this_goal += 1
    return reward

  def get_info(self, streetlearn):
    """"Returns current information about the state of the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """
    info = super(CourierGame, self).get_info(streetlearn)
    info['num_steps_this_goal'] = self._num_steps_this_goal
    info['current_goal_id'] = self._current_goal_id
    info['min_distance_reached'] = self._min_distance_reached
    info['initial_distance_to_goal'] = self._initial_distance_to_goal
    info['reward_current_goal'] = self._reward_current_goal
    num_successes = len(self._success_inverse_path_len)
    info['num_successes'] = num_successes
    info['spl'] = sum(self._success_inverse_path_len) / (num_successes + 1)
    if num_successes > 0:
      info['spl_without_last_goal'] = (
          sum(self._success_inverse_path_len) / num_successes)
    else:
      info['spl_without_last_goal'] = 0
    next_pano_id = self._panos_to_goal[streetlearn.current_pano_id]
    info['next_pano_id'] = next_pano_id
    bearing_to_next_pano = streetlearn.engine.GetPanoBearing(
        streetlearn.current_pano_id, next_pano_id) - streetlearn.engine.GetYaw()
    info['bearing_to_next_pano'] = (bearing_to_next_pano + 180) % 360 - 180
    return info

  def done(self):
    """Returns a flag indicating the end of the current episode.

    This game ends only at the end of the episode or if the goal times out.
    During a single episode, every time a goal is found, a new one is chosen,
    until the time runs out.
    """
    if self._timed_out:
      self._timed_out = False
      return True
    else:
      return False

  def _sample_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal.

    Args:
      streetlearn: The StreetLearn environment.
    """
    goals = [goal for goal in streetlearn.graph
             if ((goal != self._current_goal_id) and
                 (goal != streetlearn.current_pano_id))]
    self._current_goal_id = np.random.choice(goals)
    self._min_distance_reached = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self._current_goal_id)
    self._initial_distance_to_goal = self._min_distance_reached

  def _pick_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal.

    Args:
      streetlearn: The StreetLearn environment.
    """
    self._visited_panos.clear()
    self._sample_random_goal(streetlearn)
    logging.info('seed %d, frame %d: new goal id: %s distance: %f',
                 streetlearn.seed, streetlearn.frame_count, self.goal_id,
                 self._min_distance_reached)
    pano_data = streetlearn.engine.GetMetadata(self.goal_id).pano
    logging.info('seed %d: new goal at (%f, %f)',
                 streetlearn.seed, pano_data.coords.lat, pano_data.coords.lng)

    # Compute the extended graph and shortest path to goal to estimate
    # the reward to give to the agent.
    shortest_path, num_panos = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
    self._reward_current_goal = num_panos
    logging.info('seed %d: goal reward depends on #panos to goal: %d',
                 streetlearn.seed, self._reward_current_goal)
    # Decorate the graph.
    self._pano_id_to_color = {coin_pano_id: self._colors['coin']
                              for coin_pano_id in self._coin_pano_id_set}
    self._update_pano_id_to_color()
    for pano_id in six.iterkeys(shortest_path):
      self._pano_id_to_color[pano_id] = self._colors['shortest_path']
    self._pano_id_to_color[self.goal_id] = self._colors['goal']

  def _compute_reward(self, streetlearn):
    """Reward is a piecewise linear function of distance to the goal.

    If agent is greater than max_radius_meters from the goal, reward is 0. If
    agent is less than min_radius_reters, reward is max_reward_per_goal. Between
    min and max radius the reward is linear between 0 and max_reward_per_goal.

    Args:
      streetlearn: The StreetLearn environment.
    Returns:
      The reward for the current step.
    """
    # Do not give rewards for already visited panos.
    reward = 0
    found_goal = False
    if streetlearn.current_pano_id in self._visited_panos:
      return (reward, found_goal)

    # Mark the pano as visited and compute distance to goal.
    self._visited_panos.add(streetlearn.current_pano_id)
    distance_to_goal = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self.goal_id)
    if distance_to_goal < self._min_radius_meters:
      # Have we reached the goal?
      reward = self._reward_current_goal
      found_goal = True
      logging.info(
          'seed %d, frame %d: reached goal, distance_to_goal=%s, reward=%s',
          streetlearn.seed, streetlearn.frame_count, distance_to_goal, reward)
    else:
      if distance_to_goal < self._max_radius_meters:
        # Early reward shaping.
        if distance_to_goal < self._min_distance_reached:
          reward = (self._reward_current_goal *
                    (self._max_radius_meters - distance_to_goal) /
                    (self._max_radius_meters - self._min_radius_meters))
          self._min_distance_reached = distance_to_goal
          logging.info('seed %d, frame %d: distance_to_goal=%f, reward=%d',
                       streetlearn.seed, streetlearn.frame_count,
                       distance_to_goal, reward)

    return (reward, found_goal)

  def _compute_spl_current_goal(self, streetlearn):
    """Compute the success weighted by inverse path length for the current goal.

    We use the SPL definition from Eq. 1 in the following paper:
    Anderson et al. (2018) "On Evaluation of Embodied Navigation Agents"
    https://arxiv.org/pdf/1807.06757.pdf

    Args:
      streetlearn: The StreetLearn environment.
    Returns:
      The SPL metric for the current goal.
    """
    # Since reaching the goal is defined as being within a circle around the
    # goal pano, we subtract the panoramas within that circle from the shortest
    # path length estimate, as well as from the actual path length.
    # We add 1 to handle cases when the agent spawned within that circle.
    _, num_remaining_panos_to_goal = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
    shortest_path_len = self._reward_current_goal - num_remaining_panos_to_goal
    shortest_path_len = max(shortest_path_len, 1)
    actual_path_len = len(self._visited_panos) - num_remaining_panos_to_goal
    actual_path_len = max(actual_path_len, 1)
    return shortest_path_len / max(actual_path_len, shortest_path_len)

