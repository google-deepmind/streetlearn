# Copyright 2018 Google LLC
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

"""Importance Weighted Actor-Learner Architecture goalless navigation agent.

Note that this is a modification of code previously published by Lasse Espeholt
under an Apache license at:
https://github.com/deepmind/scalable_agent
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from six.moves import range
from six.moves import zip
import sonnet as snt
import tensorflow.compat.v1 as tf

import streetlearn.python.agents.locale_pathway as locale_pathway
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest

AgentOutput = collections.namedtuple(
    "AgentOutput", "action policy_logits baseline heading xy target_xy")


class GoalNavAgent(snt.RNNCore):
  """Agent with a simple residual convnet and LSTM."""

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
               dropout=0.5,
               lstm_num_hiddens=256,
               feed_action_and_reward=True,
               max_reward=1.0,
               name="streetlearn_core"):
    """Initializes an agent core designed to be used with A3C/IMPALA.

    Supports a single visual observation tensor and goal instruction tensor and
    outputs a single, scalar discrete action with policy logits and a baseline
    value, as well as the agent heading prediction.

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
      dropout: Dropout probabibility after the locale pathway.
      lstm_num_hiddens: Number of hiddens in the LSTM core.
      feed_action_and_reward: If True, the last action (one hot) and last reward
          (scalar) will be concatenated to the torso.
      max_reward: If `feed_action_and_reward` is True, the last reward will
          be clipped to `[-max_reward, max_reward]`. If `max_reward`
          is None, no clipping will be applied. N.B., this is different from
          reward clipping during gradient descent, or reward clipping by the
          environment.
      name: Optional name for the module.
    """
    super(GoalNavAgent, self).__init__(name='agent')

    # Policy config
    self._num_actions = num_actions
    tf.logging.info('Agent trained on %d-action policy', self._num_actions)
    # Append last reward (clipped) and last action?
    self._feed_action_and_reward = feed_action_and_reward
    self._max_reward = max_reward
    # Policy LSTM core config
    self._lstm_num_hiddens = lstm_num_hiddens
    # Extract the observation names
    observation_names = observation_names.split(';')
    self._idx_frame = observation_names.index('view_image')
    tf.logging.info('Looking for goal of type %s', goal_type)
    self._idx_goal = observation_names.index(goal_type)

    with self._enter_variable_scope():
      # Convnet
      self._convnet = snt.nets.ConvNet2D(
            output_channels=(16, 32),
            kernel_shapes=(8, 4),
            strides=(4, 2),
            paddings=[snt.VALID],
            activation=tf.nn.relu,
            activate_final=True)
      # Recurrent LSTM core of the agent.
      tf.logging.info('Locale pathway LSTM core with %d hiddens',
                      self._lstm_num_hiddens)
      self._locale_pathway = locale_pathway.LocalePathway(
          heading_stop_gradient, heading_num_hiddens, heading_num_bins,
          xy_stop_gradient, xy_num_hiddens, xy_num_bins_lat, xy_num_bins_lng,
          target_xy_stop_gradient, lstm_num_hiddens, dropout)

  def initial_state(self, batch_size):
    """Return initial state with zeros, for a given batch size and data type."""
    tf.logging.info("Initial state consists of the LSTM core initial state.")
    return self._locale_pathway.initial_state(batch_size)

  def _torso(self, input_):
    """Processing of all the visual and language inputs to the LSTM core."""

    # Extract the inputs
    last_action, env_output = input_
    last_reward, _, _, observation = env_output
    frame = observation[self._idx_frame]
    goal = observation[self._idx_goal]
    goal = tf.to_float(goal)

    # Convert to image to floats and normalise.
    frame = tf.to_float(frame)
    frame = snt.FlattenTrailingDimensions(dim_from=3)(frame)
    frame /= 255.0

    # Feed image through convnet.
    with tf.variable_scope('convnet'):
      # Convolutional layers.
      conv_out = self._convnet(frame)
      # Fully connected layer.
      conv_out = snt.BatchFlatten()(conv_out)
      conv_out = snt.Linear(256)(conv_out)
      conv_out = tf.nn.relu(conv_out)

    # Concatenate outputs of the visual and instruction pathways.
    if self._feed_action_and_reward:
      # Append clipped last reward and one hot last action.
      tf.logging.info('Append last reward clipped to: %f', self._max_reward)
      clipped_last_reward = tf.expand_dims(
          tf.clip_by_value(last_reward, -self._max_reward, self._max_reward),
          -1)
      tf.logging.info('Append last action (one-hot of %d)', self._num_actions)
      one_hot_last_action = tf.one_hot(last_action, self._num_actions)
      tf.logging.info('Append goal:')
      tf.logging.info(goal)
      action_and_reward = tf.concat([clipped_last_reward, one_hot_last_action],
                                    axis=1)
    else:
      action_and_reward = tf.constant([0], dtype=tf.float32)
    return conv_out, action_and_reward, goal

  def _core(self, core_input, core_state):
    """Assemble the recurrent core network components."""
    (conv_output, action_reward, goal) = core_input
    locale_input = tf.concat([conv_output, action_reward], axis=1)
    core_output, core_state = self._locale_pathway((locale_input, goal),
                                                   core_state)
    return core_output, core_state

  def _head(self, policy_input, heading, xy, target_xy):
    """Build the head of the agent: linear policy and value function, and pass
    the auxiliary outputs through.
    """

    # Linear policy and value function.
    policy_logits = snt.Linear(
        self._num_actions, name='policy_logits')(policy_input)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(policy_input), axis=-1)

    # Sample an action from the policy.
    new_action = tf.multinomial(
        policy_logits, num_samples=1, output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return AgentOutput(
        new_action, policy_logits, baseline, heading, xy, target_xy)

  def _build(self, input_, core_state):
    """Assemble the network components."""
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs, core_state = self.unroll(actions, env_outputs, core_state)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

  @snt.reuse_variables
  def unroll(self, actions, env_outputs, core_state):
    """Manual implementation of the network unroll."""
    _, _, done, _ = env_outputs

    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
    tf.logging.info(torso_outputs)
    conv_outputs, actions_and_rewards, goals = torso_outputs

    # Note, in this implementation we can't use CuDNN RNN to speed things up due
    # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
    # changed to implement snt.LSTMCell).
    initial_core_state = self.initial_state(tf.shape(actions)[1])
    policy_input_list = []
    heading_output_list = []
    xy_output_list = []
    target_xy_output_list = []
    for torso_output_, action_and_reward_, goal_, done_ in zip(
        tf.unstack(conv_outputs),
        tf.unstack(actions_and_rewards),
        tf.unstack(goals),
        tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = nest.map_structure(
          functools.partial(tf.where, done_), initial_core_state, core_state)
      core_output, core_state = self._core(
          (torso_output_, action_and_reward_, goal_), core_state)
      policy_input_list.append(core_output[0])
      heading_output_list.append(core_output[1])
      xy_output_list.append(core_output[2])
      target_xy_output_list.append(core_output[3])
    head_output = snt.BatchApply(self._head)(tf.stack(policy_input_list),
                                             tf.stack(heading_output_list),
                                             tf.stack(xy_output_list),
                                             tf.stack(target_xy_output_list))

    return head_output, core_state
