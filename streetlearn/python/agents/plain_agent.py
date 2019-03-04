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
import sonnet as snt
import tensorflow as tf

nest = tf.contrib.framework.nest

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class PlainAgent(snt.RNNCore):
  """Agent with a simple residual convnet and LSTM."""

  def __init__(self,
               num_actions,
               observation_names,
               lstm_num_hiddens=256,
               feed_action_and_reward=True,
               max_reward=1.0,
               name="streetlearn_core"):
    """Initializes an agent core designed to be used with A3C/IMPALA.

    Supports a single visual observation tensor and outputs a single, scalar
    discrete action with policy logits and a baseline value.

    Args:
      num_actions: Number of actions available.
      observation_names: String with observation types separated by semi-colon.
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
    super(PlainAgent, self).__init__(name='agent')

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

    with self._enter_variable_scope():
      tf.logging.info('LSTM core with %d hiddens', self._lstm_num_hiddens)
      self._core = tf.contrib.rnn.LSTMBlockCell(self._lstm_num_hiddens)

  def initial_state(self, batch_size):
    """Return initial state with zeros, for a given batch size and data type."""
    tf.logging.info("Initial state consists of the LSTM core initial state.")
    return self._core.zero_state(batch_size, tf.float32)

  def _torso(self, input_):
    """Processing of all the visual and language inputs to the LSTM core."""

    # Extract the inputs
    last_action, env_output = input_
    last_reward, _, _, observation = env_output
    if type(observation) == list:
      frame = observation[self._idx_frame]
    else:
      frame = observation

    # Convert to image to floats and normalise.
    frame = tf.to_float(frame)
    frame /= 255

    # Feed image through convnet.
    with tf.variable_scope('convnet'):
      conv_out = frame
      for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
        # Downscale.
        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
        conv_out = tf.nn.pool(
            conv_out,
            window_shape=[3, 3],
            pooling_type='MAX',
            padding='SAME',
            strides=[2, 2])
        # Residual block(s).
        for j in range(num_blocks):
          with tf.variable_scope('residual_%d_%d' % (i, j)):
            block_input = conv_out
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            conv_out += block_input
    # Fully connected layer.
    conv_out = tf.nn.relu(conv_out)
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
      core_input = tf.concat(
          [conv_out, clipped_last_reward, one_hot_last_action],
          axis=1)
    else:
      core_input = conv_out
    return core_input

  def _head(self, core_output):
    """Build the head of the agent: linear policy and value function."""
    policy_logits = snt.Linear(
        self._num_actions, name='policy_logits')(
            core_output)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.multinomial(
        policy_logits, num_samples=1, output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return AgentOutput(new_action, policy_logits, baseline)

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

    # Note, in this implementation we can't use CuDNN RNN to speed things up due
    # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
    # changed to implement snt.LSTMCell).
    initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = nest.map_structure(
          functools.partial(tf.where, d), initial_core_state, core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)

    return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state
