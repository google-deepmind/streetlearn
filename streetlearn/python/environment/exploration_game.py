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

"""Coin game level with early termination.

In this environment, the agent receives a reward for every coin it collects,
and the episode ends when all the coins are collected.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from streetlearn.python.environment import coin_game


class ExplorationGame(coin_game.CoinGame):

  def done(self, streetlearn):
    """Returns a flag indicating the end of the current episode.

    This game ends when all the coins are collected.

    Args:
      streetlearn: the streetlearn environment.
    Returns:
      reward: the reward from the last step.
      pcontinue: a flag indicating the end of an episode.
    """
    return not bool(self._coin_pano_id_set)
