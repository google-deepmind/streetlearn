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

"""StreetLearn level for the goal reward instruction-following game.

In this environment, the agent receives a reward for reaching the goal of
a given set of instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from streetlearn.python.environment import instructions_curriculum


class GoalInstructionGame(instructions_curriculum.InstructionsCurriculum):
  """StreetLang game with goal reward only."""

  def __init__(self, config):
    """Creates an instance of the StreetLearn level.

    Args:
      config: config dict of various settings.
    """
    super(GoalInstructionGame, self).__init__(config)

    # Disable waypoint rewards.
    self._reward_at_waypoint = 0
