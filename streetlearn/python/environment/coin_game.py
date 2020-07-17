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

"""Coin game level.

In this environment, the agent receives a reward for every coin it collects.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np

from streetlearn.engine.python import color
from streetlearn.python.environment import game


class CoinGame(game.Game):
  """A simple game that allows an agent to explore the environment and collect
  coins yielding rewards, randomly scattered through the environment. Can be
  extended as needed to add more complex game logic."""

  def __init__(self, config):
    """Constructor.

    Panos can be assigned rewards (coins) randomly and the agent will receive
    the reward the first time they visit these panos.

    Args:
      config: config dict of various settings.
    """
    super(CoinGame, self).__init__()

    # Create colors from the input lists.
    self._colors = {
        'coin': color.Color(*config['color_for_coin']),
        'touched': color.Color(*config['color_for_touched_pano']),
    }
    self._reward_per_coin = config['reward_per_coin']

    # List of panos (will be populated using the streetlearn object).
    self._pano_ids = None

    # Association between pano id and color.
    self._pano_id_to_color = {}

    # Panos that (can) contain coins.
    self._proportion_of_panos_with_coins = config[
        'proportion_of_panos_with_coins']
    self._touched_pano_id_set = set()
    self._coin_pano_id_set = []
    self._num_coins = 0
    logging.info('Proportion of panos with coins: %f',
                 self._proportion_of_panos_with_coins)
    logging.info('Reward per coin: %f', self._reward_per_coin)

  def on_step(self, streetlearn):
    """Gets called after StreetLearn:step(). Updates the set of touched panos.

    Args:
      streetlearn: A streetlearn instance.
    """
    self._touched_pano_id_set.add(streetlearn.current_pano_id)
    self._update_pano_id_to_color()

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Clears the set of touched panos and randomly generates reward-yielding coins
    and populates pano_id_to_color.

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    logging.info('seed %d, %d/%d coins left, cache size %d, current pano: %s',
                 streetlearn.seed, len(self._coin_pano_id_set), self._num_coins,
                 streetlearn.cache_size, streetlearn.current_pano_id)
    # Populate list of available panos.
    if not self._pano_ids:
      self._pano_ids = sorted(streetlearn.graph)

    self._touched_pano_id_set.clear()
    num_pano_ids = len(self._pano_ids)
    self._num_coins = int(self._proportion_of_panos_with_coins * num_pano_ids)
    print("Sampling {} coins in graph of {} panos.".format(
        self._num_coins, num_pano_ids))
    self._coin_pano_id_set = np.random.choice(
        self._pano_ids, self._num_coins, replace=False).tolist()
    self._pano_id_to_color = {coin_pano_id: self._colors['coin']
                              for coin_pano_id in self._coin_pano_id_set}

    return self._pano_id_to_color

  def get_reward(self, streetlearn):
    """Returns the reward from the last step.

    In this game, we give rewards when a coin is collected. Coins can be
    collected only once per episode and do not reappear.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      reward: the reward from the last step.
    """
    if streetlearn.current_pano_id in self._coin_pano_id_set:
      reward = self._reward_per_coin
      self._coin_pano_id_set.remove(streetlearn.current_pano_id)
    else:
      reward = 0
    return reward

  def get_info(self, streetlearn):
    """"Returns current information about the state of the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """
    info = {}
    info['num_coins_left'] = len(self._coin_pano_id_set)
    info['num_coins'] = self._num_coins
    info['current_pano_id'] = streetlearn.current_pano_id
    return info

  def done(self):
    """Returns a flag indicating the end of the current episode.

    This game does not end when all the coins are collected.
    """
    return False

  def highlighted_panos(self):
    """Returns the list of highlighted panos and their colors."""
    return self._pano_id_to_color

  def _update_pano_id_to_color(self):
    """Update the pano id to color table."""
    self._pano_id_to_color.update({touched_pano_id: self._colors['touched']
                                   for touched_pano_id
                                   in self._touched_pano_id_set})
