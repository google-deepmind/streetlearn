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
    self._colors['goal'] = config['color_for_goal']
    self._colors['shortest_path'] = config['color_for_shortest_path']

    self._num_steps_this_goal = 0
    self._min_distance_reached = np.finfo(np.float32).max
    self._current_goal_id = None
    self._visited_panos = set()

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Selects a random pano as goal destination.

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
    self._num_steps_this_goal = 0
    pano_id_to_color[self.goal_id] = self._colors['goal']
    for pano_id in self._shortest_path:
      pano_id_to_color[pano_id] = self._colors['shortest_path']
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
      logging.info('%d Courier target TIMEOUT (%d steps)',
                   streetlearn.frame_count, self._num_steps_this_goal)
      self._num_steps_this_goal = 0
      self._pick_random_goal(streetlearn)

    reward = self._compute_reward(streetlearn)

    # If we have found the goal, set a new one.
    if reward >= self._reward_current_goal:
      logging.info('%d Courier target FOUND (%d steps)',
                   streetlearn.frame_count, self._num_steps_this_goal)
      self._num_steps_this_goal = 0
      self._pick_random_goal(streetlearn)

    # Give additional reward if current pano has a coin.
    if streetlearn.current_pano_id in self._coin_pano_id_set:
      reward += self._reward_per_coin
      self._coin_pano_id_set.remove(streetlearn.current_pano_id)
      logging.info('Num. remaining coins: %d', len(self._coin_pano_id_set))

    self._num_steps_this_goal += 1
    return reward

  def done(self):
    """Returns a flag indicating the end of the current episode.

    This game does not end when all the coins are collected.
    """
    if self._found_goal:
      self._found_goal = False
      return True
    else:
      return False

  def get_draw_graph_params(self, streetlearn):
    """Returns the color of the home node."""
    result = super(CourierGame, self).get_draw_graph_params(streetlearn)
    result['pano_id_to_color'][str(self._current_goal_id)] = self._goal_color
    return result

  def _sample_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal."""
    goals = [goal for goal in streetlearn.graph
             if ((goal != self._current_goal_id) and
                 (goal != streetlearn.current_pano_id))]
    self._current_goal_id = np.random.choice(goals)
    self._min_distance_reached = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self._current_goal_id)

  def _pick_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal.

    Args:
      streetlearn: The StreetLearn environment.
    """
    self._visited_panos.clear()
    self._sample_random_goal(streetlearn)
    logging.info('%d CourierGame: New goal chosen', streetlearn.frame_count)
    logging.info('%d New goal id: %s ', streetlearn.frame_count, self.goal_id)
    logging.info('%d Distance to goal: %f', streetlearn.frame_count,
                 self._min_distance_reached)
    # Compute the extended graph and shortest path to goal to estimate
    # the reward to give to the agent.
    shortest_path, num_panos = self._shortest_paths(
        streetlearn, self._current_goal_id, streetlearn.current_pano_id)
    self._reward_current_goal = num_panos
    logging.info('%d Reward for the current goal depends on #panos to goal: %d',
                 streetlearn.frame_count, self._reward_current_goal)
    # Decorate the graph.
    pano_id = streetlearn.current_pano_id
    self._shortest_path = [pano_id]
    while pano_id in shortest_path:
      pano_id = shortest_path[pano_id]
      self._shortest_path.append(pano_id)

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
    if streetlearn.current_pano_id in self._visited_panos:
      return reward

    # Mark the pano as visited and compute distance to goal.
    self._visited_panos.add(streetlearn.current_pano_id)
    distance_to_goal = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self.goal_id)
    if distance_to_goal < self._min_radius_meters:
      # Have we reached the goal?
      reward = self._reward_current_goal
      self._found_goal = True
      logging.info('%d Reached goal, distance_to_goal=%s, reward=%s',
                   streetlearn.frame_count, distance_to_goal, reward)
    else:
      if distance_to_goal < self._max_radius_meters:
        # Early reward shaping.
        if distance_to_goal < self._min_distance_reached:
          reward = (self._reward_current_goal *
                    (self._max_radius_meters - distance_to_goal) /
                    (self._max_radius_meters - self._min_radius_meters))
          self._min_distance_reached = distance_to_goal
          logging.info('%d Distance_to_goal=%f, reward=%d',
                       streetlearn.frame_count, distance_to_goal, reward)

    return reward
