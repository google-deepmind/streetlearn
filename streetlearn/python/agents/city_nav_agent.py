# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the goal-driven StreetLearn CityNavAgent with auxiliary losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import tensorflow.compat.v1 as tf
import sonnet as snt

import streetlearn.python.agents.goal_nav_agent as goal_nav_agent
import streetlearn.python.agents.locale_pathway as locale_pathway
from tensorflow.contrib import rnn as contrib_rnn


class CityNavAgent(goal_nav_agent.GoalNavAgent):
  """Core with A2C/A3C-compatible outputs for simple visual observations."""

  def __init__(self,
               num_actions,
               observation_names,
               goal_type='target_latlng',
               heading_stop_gradient=False,
               heading_num_hiddens=256,
               heading_num_bins=16,
               xy_stop_gradient=True,
               xy_num_hiddens=256,
               xy_num_bins_lat=32,
               xy_num_bins_lng=32,
               target_xy_stop_gradient=True,
               locale_lstm_num_hiddens=256,
               dropout=0.5,
               locale_bottleneck_num_hiddens=64,
               skip_connection=True,
               policy_lstm_num_hiddens=256,
               feed_action_and_reward=True,
               max_reward=1.0,
               name="streetlearn_core"):
    """Initializes an agent core designed to be used with A3C/IMPALA.

    Supports a single visual observation tensor and goal instruction tensor and
    outputs a single, scalar discrete action with policy logits and a baseline
    value, as well as the agent heading, XY position and target XY predictions.

    Args:
      num_actions: Number of actions available.
      observation_names: String with observation names separated by semi-colon.
      goal_type: String with the name of the target observation field, can be
          `target_latlng` or `target_landmarks`.
      heading_stop_gradient: Boolean for stopping gradient between the LSTM core
          and the heading prediction MLP.
      heading_num_hiddens: Number of hiddens in the heading prediction MLP.
      heading_num_bins: Number of outputs in the heading prediction MLP.
      xy_stop_gradient: Boolean for stopping gradient between the LSTM core
          and the XY position prediction MLP.
      xy_num_hiddens: Number of hiddens in the XY position prediction MLP.
      xy_num_bins_lat: Number of lat outputs in the XY position prediction MLP.
      xy_num_bins_lng: Number of lng outputs in the XY position prediction MLP.
      target_xy_stop_gradient: Boolean for stopping gradient between the LSTM
          core and the target XY position prediction MLP.
      locale_lstm_num_hiddens: Number of hiddens in the locale pathway core.
      dropout: Dropout probabibility after the locale pathway.
      locale_bottleneck_num_hiddens: Number of hiddens in the bottleneck after
          the locale pathway.
      skip_connection: Is there a direct connection from convnet to policy LSTM?
      policy_lstm_num_hiddens: Number of hiddens in the policy LSTM core.
      feed_action_and_reward: If True, the last action (one hot) and last reward
          (scalar) will be concatenated to the torso.
      max_reward: If `feed_action_and_reward` is True, the last reward will
          be clipped to `[-max_reward, max_reward]`. If `max_reward`
          is None, no clipping will be applied. N.B., this is different from
          reward clipping during gradient descent, or reward clipping by the
          environment.
      name: Optional name for the module.
    """
    super(CityNavAgent, self).__init__(
        num_actions,
        observation_names,
        goal_type,
        heading_stop_gradient,
        heading_num_hiddens,
        heading_num_bins,
        xy_stop_gradient,
        xy_num_hiddens,
        xy_num_bins_lat,
        xy_num_bins_lng,
        target_xy_stop_gradient,
        dropout,
        lstm_num_hiddens=locale_lstm_num_hiddens,
        feed_action_and_reward=feed_action_and_reward,
        max_reward=max_reward,
        name=name)

    # Skip connection for convnet, short-circuiting the global pathway?
    self._skip_connection = skip_connection
    tf.logging.info("Convnet skip connection? " + str(self._skip_connection))

    with self._enter_variable_scope():
      # Recurrent policy LSTM core of the agent.
      tf.logging.info('LSTM core with %d hiddens', policy_lstm_num_hiddens)
      self._policy_lstm = contrib_rnn.LSTMBlockCell(
          policy_lstm_num_hiddens, name="policy_lstm")
      # Add an optional bottleneck after the global LSTM
      if locale_bottleneck_num_hiddens > 0:
        self._locale_bottleneck = snt.nets.MLP(
            output_sizes=(locale_bottleneck_num_hiddens,),
            activation=tf.nn.tanh,
            activate_final=True,
            name="locale_bottleneck")
        tf.logging.info("Auxiliary global pathway bottleneck with %d hiddens",
                        locale_bottleneck_num_hiddens)
      else:
        self._locale_bottleneck = tf.identity

  def initial_state(self, batch_size):
    """Returns an initial state with zeros, for a batch size and data type."""
    tf.logging.info("Initial state consists of the locale pathway and policy "
                    "LSTM core initial states.")
    initial_state_list = []
    initial_state_list.append(self._policy_lstm.zero_state(
        batch_size, tf.float32))
    initial_state_list.append(self._locale_pathway.initial_state(batch_size))
    return tuple(initial_state_list)

  def _core(self, core_input, core_state):
    """Assemble the recurrent core network components."""
    (conv_output, action_reward, goal) = core_input

    # Get the states
    policy_state, locale_state = core_state

    # Locale-specific pathway
    locale_input = conv_output
    locale_output, locale_state = self._locale_pathway((locale_input, goal),
                                                       locale_state)
    (lstm_output, heading_output, xy_output, target_xy_output) = locale_output

    # Policy LSTM
    policy_input = self._locale_bottleneck(lstm_output)
    if self._skip_connection:
      policy_input = tf.concat([policy_input, conv_output], axis=1)
    if self._feed_action_and_reward:
      policy_input = tf.concat([policy_input, action_reward], axis=1)
    policy_input = tf.identity(policy_input, name="policy_input")

    policy_output, policy_state = self._policy_lstm(policy_input, policy_state)

    core_output = (policy_output, heading_output, xy_output, target_xy_output)
    core_state_list = []
    core_state_list.append(policy_state)
    core_state_list.append(locale_state)
    core_state = tuple(core_state_list)
    return core_output, core_state
