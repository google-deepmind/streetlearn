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

"""StreetLearn level for the single instruction-following game with curriculum.

In this environment, the agent receives a reward for every waypoint it hits
as well as a larger reward for reaching the final goal. At any point in time
the agent will receive exactly one instruction, matching the next waypoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from streetlearn.python.environment import instructions_densification


class StepByStepInstructionGame(
    instructions_densification.InstructionsDensification):
  """StreetLang game for following a single instructions at a time."""

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    super(StepByStepInstructionGame, self).on_reset(streetlearn)

    # Save instruction and thumbnail vectors into a separate holder.
    # It is sufficient to use _with_goal as this is a superset of the other.
    self._all_thumbs = self._thumbnails.copy()
    self._all_instrs = self._instructions

    # Initialise everything to the first entry
    self._thumbnails[2:, :] = 0  # Zero all but the first and second one.
    self._instructions = [self._all_instrs[0]]

    logging.info(self._all_instrs)
    logging.info(self._instructions)

    return self._pano_id_to_color

  def _check_reward(self, pano_id, streetlearn):
    """Check what reward the current pano yields, based on instructions.

    Args:
      pano_id: centroid pano id.
      streetlearn: streetlearn graph for establishing neighbours.
    Returns:
      The reward for the current step.
    """
    reward = 0

    previous_step = self._current_step
    reward = super(StepByStepInstructionGame, self)._check_reward(
        pano_id, streetlearn)

    if previous_step != self._current_step and not self._reached_goal:
      # If we changed the step, but haven't terminated the game, update instrs.
      self._thumbnails[0, :] = self._all_thumbs[self._current_step]
      self._thumbnails[1, :] = (
          self._all_thumbs[self._current_step + 1])
      self._instructions = [self._all_instrs[self._current_step]]

      # Remove epsilon from reward to avoid triggering the waypoint switchers.
      epsilon = 0.01
      reward -= epsilon
      logging.info('Switched from step %d to step %d.',
                   previous_step, self._current_step)
      logging.info(self._instructions)

    return reward

  def get_info(self, streetlearn):
    """"Returns current information about the state of the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """
    info = super(StepByStepInstructionGame, self).get_info(streetlearn)
    info['current_step'] = 0
    return info
