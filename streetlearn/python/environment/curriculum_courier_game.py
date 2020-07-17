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

"""StreetLearn level for the curriculum-based courier task.

This is a version of the courier task that increases the distance to the
goal with each episode using the given annealing rate.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import time

from streetlearn.python.environment import courier_game

_SECONDS_IN_HOUR = 3600


class CurriculumCourierGame(courier_game.CourierGame):
  """Coin game that gives extra reward for finding the goal pano. A courier goal
  is randomly selected from panos in the graph according to a curriculum that
  starts with panos within a maximum distance from the current agent position,
  then anneals it with time. On success or timeout, a new goal is chosen. The
  episode ends after a fixed episode length.
  """

  def __init__(self, config):
    """Creates an instance of the RandomTaxiCurriculum level.

    This coin game gives extra reward for finding the goal pano, and resets the
    goal once the goal has been found (or on timeout). Panos can be assigned
    rewards (coins) randomly and the agent will receive the reward the first
    time they visit these panos. Goal panos are assigned within a circle whose
    radius grows in time from min_goal_distance to max_goal_distance.

    Args:
      config: config dict of various settings.
    """
    super(CurriculumCourierGame, self).__init__(config)
    self._timestamp_start = config['timestamp_start_curriculum']
    self._annealing_rate = config['annealing_rate_curriculum']
    self._hours_curriculum_part_1 = config['hours_curriculum_part_1']
    self._hours_curriculum_part_2 = config['hours_curriculum_part_2']
    self._min_goal_distance = config['min_goal_distance_curriculum']
    self._max_goal_distance = config['max_goal_distance_curriculum']
    self._allowed_goal_distance = self._min_goal_distance
    assert self._timestamp_start <= time.time()
    assert self._annealing_rate > 0
    assert self._hours_curriculum_part_1 >= 0
    assert self._hours_curriculum_part_2 > 0
    assert self._min_goal_distance < self._max_goal_distance

    logging.info(
        'Curriculum: starts at t=%d, dist <= %f in P1 (%f h)',
        self._timestamp_start, self._min_goal_distance,
        self._hours_curriculum_part_1)
    logging.info(
        'Curriculum: then %f < dist <= %f in P2 (%f h)',
        self._min_goal_distance, self._max_goal_distance,
        self._hours_curriculum_part_2)
    logging.info('Curriculum: annealing rate: %f', self._annealing_rate)

  def _update_curriculum_goal_distance(self):
    """Updates the allowed distance to the goal according to the curriculum."""
    hours_train = max(0,
                      (time.time() - self._timestamp_start) / _SECONDS_IN_HOUR)
    if hours_train <= self._hours_curriculum_part_1:
      # During part 1 of the curriculum, sample goals within a minimal distance.
      self._allowed_goal_distance = self._min_goal_distance
    else:
      # During part 2 of the curriculum, sample goals within a distance
      # that grows from a minimum value to a maximum value.
      numerator = hours_train - self._hours_curriculum_part_1
      denom = self._hours_curriculum_part_2
      time_factor = pow(min(1, max(0, numerator / denom)), self._annealing_rate)
      self._allowed_goal_distance = (
          (self._max_goal_distance - self._min_goal_distance
           ) * time_factor + self._min_goal_distance)

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
    # Update the allowed distance to the goal according to the curriculum.
    self._update_curriculum_goal_distance()
    # Populate the list of panos and assign optional coins to panos.
    # Assign the goal location to one of the panos.
    return super(CurriculumCourierGame, self).on_reset(streetlearn)

  def get_info(self, streetlearn):
    """"Returns current information about the state of the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """
    info = super(CurriculumCourierGame, self).get_info(streetlearn)
    info['allowed_goal_distance'] = self._allowed_goal_distance
    return info

  def _sample_random_goal(self, streetlearn):
    """Randomly sets a new pano for the current goal according to a curriculum.

    Args:
      streetlearn: The StreetLearn environment.
    """
    # Sample a goal among the pano ids that is within that distance.
    goals = [goal for goal in streetlearn.graph
             if ((goal != self._current_goal_id) and
                 (goal != streetlearn.current_pano_id))]
    self._initial_distance_to_goal = float('inf')
    while self._initial_distance_to_goal > self._allowed_goal_distance:
      self._current_goal_id = np.random.choice(goals)
      self._min_distance_reached = streetlearn.engine.GetPanoDistance(
          streetlearn.current_pano_id, self._current_goal_id)
      self._initial_distance_to_goal = self._min_distance_reached
      logging.info(
          'seed %d, frame %d: distance to goal: %f (max allowed: %f)',
          streetlearn.seed, streetlearn.frame_count,
          self._initial_distance_to_goal, self._allowed_goal_distance)
