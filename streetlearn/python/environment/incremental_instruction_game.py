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
as well as a larger reward for reaching the final goal.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from streetlearn.python.environment import instructions_densification


class IncrementalInstructionGame(
    instructions_densification.InstructionsDensification):
  """StreetLang game with goal and waypoint rewards."""

  def __init__(self, config):
    """Creates an instance of the StreetLearn level.

    Args:
      config: config dict of various settings.
    """
    super(IncrementalInstructionGame, self).__init__(config)

    # Verify that waypoints receive reward.
    assert self._reward_at_waypoint > 0, "Waypoint reward should be nonzero."
