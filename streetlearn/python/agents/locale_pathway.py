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

"""Implements the locale-specific core for StreetLearn agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import sonnet as snt

# Outputs of the global city-specific pathway
LocalePathwayOutput = collections.namedtuple(
    "LocalePathwayOutput", ["lstm_output", "heading", "xy", "target_xy"])


class LocalePathway(snt.RNNCore):
  """City-specific locale core, operating on visual embeddings."""

  def __init__(self,
               heading_stop_gradient=False,
               heading_num_hiddens=256,
               heading_num_bins=16,
               xy_stop_gradient=True,
               xy_num_hiddens=256,
               xy_num_bins_lat=32,
               xy_num_bins_lng=32,
               target_xy_stop_gradient=True,
               lstm_num_hiddens=256,
               dropout=0.5,
               name="locale_pathway_core"):
    """Initializes a city-specific global localisation core,
    operating on visual embeddings and target positions.

    Supports a single embedding tensor and a single target position tensor,
    and outputs a single hidden state as well as auxiliary localisation outputs.
    Relies on a recurrent LSTM core.

    Args:
      aux_config: ConfigDict with additional ConfigDict for auxiliary tasks.
      name: Optional name for the module.

    Returns:
    """
    super(LocalePathway, self).__init__(name=name)

    self._heading_stop_gradient = heading_stop_gradient
    self._xy_stop_gradient = xy_stop_gradient
    self._target_xy_stop_gradient = target_xy_stop_gradient
    tf.logging.info("Stop gradient? heading:%s, XY:%s and target XY:%s",
                    str(heading_stop_gradient), str(xy_stop_gradient),
                    str(target_xy_stop_gradient))
    self._lstm_num_hiddens = lstm_num_hiddens
    tf.logging.info("Number of hiddens in locale-specific LSTM: %d",
                    lstm_num_hiddens)
    self._dropout = dropout
    tf.logging.info("Dropout after LSTM: %f", dropout)

    with self._enter_variable_scope():
      # Add an LSTM for global landmark, heading and XY prediction tasks
      tf.logging.info("Auxiliary global pathway LSTM with %d hiddens",
                      self._lstm_num_hiddens)
      assert(self._lstm_num_hiddens > 0)
      self._lstm = tf.contrib.rnn.LSTMBlockCell(self._lstm_num_hiddens,
                                                name="global_pathway_lstm")
      # Add an MLP head for absolute heading (north) bin prediction
      tf.logging.info("%d-bin absolute heading prediction with %s hiddens",
                      heading_num_bins,
                      heading_num_hiddens)
      self._heading_logits = snt.nets.MLP(
          output_sizes=(heading_num_hiddens, heading_num_bins),
          activate_final=False,
          name="heading_logits")
      # Add an MLP head for XY location bin prediction
      xy_num_bins = xy_num_bins_lat * xy_num_bins_lng
      tf.logging.info("%d-bin XY location prediction (%d lat, %d lng)",
                      xy_num_bins, xy_num_bins_lat, xy_num_bins_lng)
      tf.logging.info("with %s hiddens", xy_num_hiddens)
      self._xy_logits = snt.nets.MLP(
          output_sizes=(xy_num_hiddens, xy_num_bins),
          activate_final=False,
          name="xy_logits")
      # Add an MLP head for XY target location bin prediction
      tf.logging.info("%d-bin target XY location prediction (%d lat, %d lng)",
                      xy_num_bins, xy_num_bins_lat, xy_num_bins_lng)
      tf.logging.info("with %s hiddens", xy_num_hiddens)
      self._target_xy_logits = snt.nets.MLP(
          output_sizes=(xy_num_hiddens, xy_num_bins),
          activate_final=False,
          name="target_xy_logits")

  def _build(self, (embedding, target_position), state):
    """Connects the core into the graph.

    This core is designed to be used for embeddings coming from a convnet.

    Args:
      embedding: The result of convnet embedding.
      target_position: Representation of the target position.
      state: The current state of the global LSTM component of the core.

    Returns:
      A tuple `(action, other_output), next_state`, where:
        * `action` is the action selected by the core. An iterable containing
          a single Tensor with shape `[batch, 1]`, which is the zero-based index
          of the selected action.
        * `other_output` is a namedtuple with fields `policy_logits` (a Tensor
          of shape `[batch, num_actions]`) and `baseline` (a Tensor of shape
          `[batch]`).
        * `next_state` is the output of the LSTM component of the core.
    """

    # Add the target to global LSTM
    with tf.name_scope("targets") as scope:
      lstm_input = tf.concat([embedding, tf.cast(target_position,
                                                 dtype=tf.float32)],
                             axis=1)

    # Global pathway tasks
    with tf.name_scope("locale_pathway") as scope:
      lstm_output, next_state = self._lstm(lstm_input, state)

      # Heading decoding or prediction
      if self._heading_stop_gradient:
        input_heading = tf.stop_gradient(lstm_output,
                                         name='heading_stop_gradient')
      else:
        input_heading = lstm_output
      heading = self._heading_logits(input_heading)

      # XY decoding or prediction
      if self._xy_stop_gradient:
        input_xy = tf.stop_gradient(lstm_output,
                                    name='xy_stop_gradient')
      else:
        input_xy = lstm_output
      xy = self._xy_logits(input_xy)

      # Target XY decoding
      if self._target_xy_stop_gradient:
        input_target_xy = tf.stop_gradient(lstm_output,
                                    name='target_xy_stop_gradient')
      else:
        input_target_xy = lstm_output
      target_xy = self._target_xy_logits(input_target_xy)

      # Add dropout
      if self._dropout > 0:
        lstm_output = tf.nn.dropout(lstm_output,
                                    keep_prob=self._dropout,
                                    name="lstm_output_dropout")
      else:
        lstm_output = tf.identity(lstm_output,
                                  name="lstm_output_without_dropout")

    # Outputs
    core_output = LocalePathwayOutput(lstm_output=lstm_output,
                                      heading=heading,
                                      xy=xy,
                                      target_xy=target_xy)
    return core_output, next_state

  def initial_state(self, batch_size):
    """Returns an initial state with zeros, for a batch size and data type."""
    tf.logging.info("Initial state includes the locale LSTM")
    return self._lstm.zero_state(batch_size, tf.float32)
