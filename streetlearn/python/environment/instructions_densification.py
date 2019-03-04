# Copyright 2019 Google LLC.
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

"""StreetLearn level with reward circles around each waypoint.

This file extends the instruction-following game by adding reward densification,
giving fractional reward to agents as they approach a waypoint.  At any point in
time, the level will project a reward cone around the next waypoint given the
most recently passed waypoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from streetlearn.python.environment import instructions_curriculum


class InstructionsDensification(
    instructions_curriculum.InstructionsCurriculum):
  """StreetLang game with a cone around each waypoint reward."""

  def __init__(self, config):
    """Creates an instance of the StreetLang level.

    Args:
      config: config dict of various settings.
    """
    super(InstructionsDensification, self).__init__(config)
    self._max_reward_per_cone = config['max_reward_per_cone']
    self._cone_radius_meters = config['cone_radius_meters']
    self._min_distance_reached = float('inf')
    self._distance_to_next_waypoint = float('inf')
    # We cannot have coins in this game.
    assert config['proportion_of_panos_with_coins'] == 0

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    self._min_distance_reached = float('inf')
    self._distance_to_next_waypoint = float('inf')
    return super(InstructionsDensification, self).on_reset(streetlearn)

  def get_reward(self, streetlearn):
    """Returns the reward from the last step.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      reward: the reward from the last step.
    """
    # Calculate distance to next waypoint for _check_reward to use.
    next_waypoint_pano = self._pano_by_step[self._current_step + 1]
    self._distance_to_next_waypoint = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, next_waypoint_pano)

    # Check if pano ID is within a cone from the next waypoint.
    dense_reward = 0
    if self._distance_to_next_waypoint < min(self._cone_radius_meters,
                                             self._min_distance_reached):
      dense_reward = (
          self._max_reward_per_cone *
          (self._cone_radius_meters - self._distance_to_next_waypoint) /
          self._cone_radius_meters)
      self._min_distance_reached = self._distance_to_next_waypoint
      logging.info('distance_to_next_waypoint=%f, extra reward=%f',
                   self._distance_to_next_waypoint, dense_reward)

    # Compute the regular reward using the logic from the base class.
    prev_step = self._current_step
    reward = super(InstructionsDensification, self).get_reward(streetlearn)

    # Reset the minimum distance threshold?
    if prev_step != self._current_step:
      self._min_distance_reached = float('inf')

    return reward + dense_reward
