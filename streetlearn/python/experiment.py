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

"""Main experiment file for the StreetLearn agent, based on an implementation of
Importance Weighted Actor-Learner Architectures.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.

Note that this derives from code previously published by Lasse Espeholt
under an Apache license at:
https://github.com/deepmind/scalable_agent
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import os
import sys
import time

import numpy as np
from six.moves import range
import sonnet as snt
import tensorflow.compat.v1 as tf

from streetlearn.python.agents import goal_nav_agent
from streetlearn.python.agents import city_nav_agent
from streetlearn.python.scalable_agent import py_process
from streetlearn.python.scalable_agent import vtrace
from streetlearn.python.environment import default_config
from streetlearn.python.environment import streetlearn
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import staging as contrib_staging

nest = contrib_framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')
flags.DEFINE_string('master', '', 'Session master.')

# Training.
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 1, 'Number of actors.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 50, 'Unroll length in agent steps.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
flags.DEFINE_float('heading_prediction_cost', 1.0,
                   'Auxiliary cost/multiplier for heading prediction.')
flags.DEFINE_float('xy_prediction_cost', 1.0,
                   'Auxiliary cost/multiplier for XY position prediction.')
flags.DEFINE_float('target_xy_prediction_cost', 1.0,
                   'Auxiliary cost/multiplier for XY target prediction.')

# Environment settings.
flags.DEFINE_string('game_name', 'curriculum_courier_game',
                    'Game name for the StreetLearn agent.')
flags.DEFINE_string('level_names', 'manhattan_lowres',
                    'Lavel name for the StreetLearn agent.')
flags.DEFINE_string('dataset_paths', None, 'Path were the levels are stored.')
flags.DEFINE_integer('width', 84, 'Width of observation.')
flags.DEFINE_integer('height', 84, 'Height of observation.')
flags.DEFINE_integer('graph_width', 84, 'Width of graph visualisation.')
flags.DEFINE_integer('graph_height', 84, 'Height of graph visualisation.')
flags.DEFINE_integer('graph_zoom', 1, 'Zoom in graph visualisation.')
flags.DEFINE_string('start_pano', '',
                     'Pano at root of partial graph (default: full graph).')
flags.DEFINE_integer('graph_depth', 200, 'Depth of the pano graph.')
flags.DEFINE_integer('frame_cap', 1000, 'Number of frames / episode.')
flags.DEFINE_string('action_set', 'streetlearn_fast_rotate',
                    'Set of actions used by the agent.')
flags.DEFINE_float('rotation_speed', 22.5,
                   'Rotation speed of the actor.')
flags.DEFINE_string('observations',
                    'view_image;graph_image;latlng;target_latlng;yaw;yaw_label;'
                    'latlng_label;target_latlng_label',
                    'Observations used by the agent.')
flags.DEFINE_float('timestamp_start_curriculum', time.time(),
                   'Timestamp at the start of the curriculum.')
flags.DEFINE_float('hours_curriculum_part_1', 0.0,
                   'Number of hours for 1st part of curriculum.')
flags.DEFINE_float('hours_curriculum_part_2', 24.0,
                   'Number of hours for 2nd part of curriculum.')
flags.DEFINE_float('min_goal_distance_curriculum', 500.0,
                   'Maximum distance to goal at beginning of curriculum.')
flags.DEFINE_float('max_goal_distance_curriculum', 3500.0,
                   'Maximum distance to goal at end of curriculum.')
flags.DEFINE_float('bbox_lat_min', 0, 'Minimum latitude.')
flags.DEFINE_float('bbox_lat_max', 100, 'Maximum latitude.')
flags.DEFINE_float('bbox_lng_min', 0, 'Minimum longitude.')
flags.DEFINE_float('bbox_lng_max', 100, 'Maximum longitude.')
flags.DEFINE_float('min_radius_meters', 100.0, 'Radius of goal area.')
flags.DEFINE_float('max_radius_meters', 200.0, 'Radius of early rewards.')
flags.DEFINE_float('proportion_of_panos_with_coins', 0, 'Proportion of coins.')

# Agent settings.
flags.DEFINE_string('agent', 'city_nav_agent', 'Agent name.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline heading')


def is_single_machine():
  return FLAGS.task == -1


StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')


class FlowEnvironment(object):
  """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.

  Note that this is a copy of the code previously published by Lasse Espeholt
  under an Apache license at:
  https://github.com/deepmind/scalable_agent
  """
  def __init__(self, env):
    """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation). `observation`
        should be the observation after the step is taken. If `done` is
        True, the observation should be the first observation in the next
        episode.
    """
    self._env = env

  def initial(self):
    """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
    with tf.name_scope('flow_environment_initial'):
      initial_reward = tf.constant(0.)
      initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0))
      initial_done = tf.constant(True)
      initial_observation = self._env.initial()

      initial_output = StepOutput(initial_reward, initial_info, initial_done,
                                  initial_observation)

      # Control dependency to make sure the next step can't be taken before the
      # initial output has been read from the environment.
      with tf.control_dependencies(nest.flatten(initial_output)):
        initial_flow = tf.constant(0, dtype=tf.int64)
      initial_state = (initial_flow, initial_info)
      return initial_output, initial_state

  def step(self, action, state):
    """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """
    with tf.name_scope('flow_environment_step'):
      flow, info = nest.map_structure(tf.convert_to_tensor, state)

      # Make sure the previous step has been executed before running the next
      # step.
      with tf.control_dependencies([flow]):
        reward, done, observation = self._env.step(action)

      with tf.control_dependencies(nest.flatten(observation)):
        new_flow = tf.add(flow, 1)

      # When done, include the reward in the output info but not in the
      # state for the next step.
      new_info = StepOutputInfo(info.episode_return + reward,
                                info.episode_step + 1)
      new_state = new_flow, nest.map_structure(
          lambda a, b: tf.where(done, a, b),
          StepOutputInfo(tf.constant(0.), tf.constant(0)), new_info)

      output = StepOutput(reward, new_info, done, observation)
      return output, new_state


class StreetLearnImpalaAdapter(streetlearn.StreetLearn):
  def __init__(self, dataset_path, config, game):
    super(StreetLearnImpalaAdapter, self).__init__(dataset_path, config, game)
    self.reset()

  def initial(self):
    """Returns the original observation."""
    super(StreetLearnImpalaAdapter, self).step([0.0, 0.0, 0.0, 0.0])
    observation = self._reshape_observation(self.observation())
    return observation

  def step(self, action):
    """Takes a step in the environment.

    Args:
      action: a 1d array containing a combination of actions.
    Returns:
      reward: float value.
      done: boolean indicator.
      observation: observation at the last step.
    """
    (observation, reward, done, _) = super(
        StreetLearnImpalaAdapter, self).step(action)
    reward = np.array(reward, dtype=np.float32)
    observation = self._reshape_observation(observation)
    return reward, done, observation

  def _reshape_observation(self, observation):
    return [
        np.transpose(np.reshape(observation['view_image'],
                                [3, FLAGS.height, FLAGS.width]),
                     axes=(1, 2, 0)),
        np.transpose(np.reshape(observation['graph_image'],
                                [3, FLAGS.graph_height, FLAGS.graph_width]),
                     axes=(1, 2, 0)),
        observation['latlng'],
        observation['target_latlng'],
        observation['yaw'],
        observation['yaw_label'],
        observation['latlng_label'],
        observation['target_latlng_label'],
    ]

  @staticmethod
  def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
    """Returns a nest of `TensorSpec` with the method's output specification."""
    observation_spec = [
        contrib_framework.TensorSpec([FLAGS.height, FLAGS.width, 3], tf.uint8),
        contrib_framework.TensorSpec([FLAGS.graph_height, FLAGS.graph_width, 3],
                                     tf.uint8),
        contrib_framework.TensorSpec([
            2,
        ], tf.float64),
        contrib_framework.TensorSpec([
            2,
        ], tf.float64),
        contrib_framework.TensorSpec([], tf.float64),
        contrib_framework.TensorSpec([], tf.uint8),
        contrib_framework.TensorSpec([], tf.int32),
        contrib_framework.TensorSpec([], tf.int32),
    ]

    if method_name == 'initial':
      return observation_spec
    elif method_name == 'step':
      return (
          contrib_framework.TensorSpec([], tf.float32),
          contrib_framework.TensorSpec([], tf.bool),
          observation_spec,
      )


def build_actor(agent, env, level_name, action_set):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  initial_agent_state = agent.initial_state(1)
  initial_action = tf.zeros([1], dtype=tf.int32)
  dummy_agent_output, _ = agent(
      (initial_action,
       nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
      initial_agent_state)
  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

  # All state that needs to persist across training iterations. This includes
  # the last environment output, agent state and last agent output. These
  # variables should never go on the parameter servers.
  def create_state(t):
    # Creates a unique variable scope to ensure the variable name is unique.
    with tf.variable_scope(None, default_name='state'):
      return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  persistent_state = nest.map_structure(
      create_state, (initial_env_state, initial_env_output, initial_agent_state,
                     initial_agent_output))

  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_state, agent_output = input_

    # Run agent.
    action = agent_output[0]
    batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            env_output)
    agent_output, agent_state = agent((action, batched_env_output), agent_state)

    # Convert action index to the native action.
    action = agent_output[0][0]
    raw_action = tf.gather(action_set, action)

    env_output, env_state = env.step(raw_action, env_state)

    return env_state, env_output, agent_state, agent_output

  # Run the unroll. `read_value()` is needed to make sure later usage will
  # return the first values and not a new snapshot of the variables.
  first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
  _, first_env_output, first_agent_state, first_agent_output = first_values

  # Use scan to apply `step` multiple times, therefore unrolling the agent
  # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
  # the output of each call of `step` as input of the subsequent call of `step`.
  # The unroll sequence is initialized with the agent and environment states
  # and outputs as stored at the end of the previous unroll.
  # `output` stores lists of all states and outputs stacked along the entire
  # unroll. Note that the initial states and outputs (fed through `initializer`)
  # are not in `output` and will need to be added manually later.
  output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
  _, env_outputs, _, agent_outputs = output

  # Update persistent state with the last output from the loop.
  assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                  persistent_state, output)

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
    first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
    agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output), (agent_outputs, env_outputs))

    output = ActorOutput(
        level_name=level_name, agent_state=first_agent_state,
        env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)

    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, output)


def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_classification_loss(logits, labels):
  classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
  return tf.reduce_sum(classification_loss)


def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)


def plot_logits_2d(logits, num_x, num_y):
  """Plot logits as 2D images."""
  logits_2d = tf.reshape(logits, shape=[-1, num_y, num_x])
  logits_2d = tf.expand_dims(tf.expand_dims(logits_2d[:, ::-1, :], 1), -1)
  return logits_2d


def build_learner(agent, agent_state, env_outputs, agent_outputs):
  """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

  Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
  """
  learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs,
                                    agent_state)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, the environment outputs at time step `t` are the inputs that
  # lead to the learner_outputs at time step `t`. After the following shifting,
  # the actions in agent_outputs and learner_outputs at time step `t` is what
  # leads to the environment outputs at time step `t`.
  agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
  rewards, infos, done, observations = nest.map_structure(
      lambda t: t[1:], env_outputs)
  learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)
  observation_names = FLAGS.observations.split(';')

  if FLAGS.reward_clipping == 'abs_one':
    clipped_rewards = tf.clip_by_value(rewards, -1, 1)
  elif FLAGS.reward_clipping == 'soft_asymmetric':
    squeezed = tf.tanh(rewards / 5.0)
    # Negative rewards are given less weight than positive rewards.
    clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

  discounts = tf.to_float(~done) * FLAGS.discounting

  # Compute V-trace returns and weights.
  # Note, this is put on the CPU because it's faster than on GPU. It can be
  # improved further with XLA-compilation or with a custom TensorFlow operation.
  with tf.device('/cpu'):
    vtrace_returns = vtrace.from_logits(
        behaviour_policy_logits=agent_outputs.policy_logits,
        target_policy_logits=learner_outputs.policy_logits,
        actions=agent_outputs.action,
        discounts=discounts,
        rewards=clipped_rewards,
        values=learner_outputs.baseline,
        bootstrap_value=bootstrap_value)

  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  rl_loss_policy_gradient = compute_policy_gradient_loss(
      learner_outputs.policy_logits, agent_outputs.action,
      vtrace_returns.pg_advantages)
  rl_loss_baseline = FLAGS.baseline_cost * compute_baseline_loss(
      vtrace_returns.vs - learner_outputs.baseline)
  rl_loss_entropy = FLAGS.entropy_cost * compute_entropy_loss(
      learner_outputs.policy_logits)
  total_loss = rl_loss_policy_gradient + rl_loss_baseline + rl_loss_entropy

  # Add auxiliary loss for heading prediction.
  if 'yaw_label' in observation_names:
    idx_yaw_label = observation_names.index('yaw_label')
    yaw_logits = learner_outputs.heading
    yaw_labels = tf.cast(observations[idx_yaw_label], dtype=tf.int32)
    heading_loss = FLAGS.heading_prediction_cost * compute_classification_loss(
        yaw_logits, yaw_labels)
    total_loss += heading_loss

  # Add auxiliary loss for XY position and XY target position prediction.
  if 'latlng_label' in observation_names:
    idx_latlng_label = observation_names.index('latlng_label')
    xy_logits = learner_outputs.xy
    xy_labels = tf.cast(observations[idx_latlng_label], dtype=tf.int32)
    xy_loss = FLAGS.xy_prediction_cost * compute_classification_loss(
        xy_logits, xy_labels)
    total_loss += xy_loss
  if 'target_latlng_label' in observation_names:
    idx_target_latlng_label = observation_names.index('target_latlng_label')
    target_xy_logits = learner_outputs.target_xy
    target_xy_labels = tf.cast(observations[idx_target_latlng_label],
                               dtype=tf.int32)
    target_xy_loss = (
        FLAGS.target_xy_prediction_cost * compute_classification_loss(
            target_xy_logits, target_xy_labels))
    total_loss += target_xy_loss

  # Optimization
  num_env_frames = tf.train.get_global_step()
  learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                            FLAGS.total_environment_frames, 0)
  optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                        FLAGS.momentum, FLAGS.epsilon)
  train_op = optimizer.minimize(total_loss)

  # Merge updating the network and environment frames into a single tensor.
  with tf.control_dependencies([train_op]):
    num_env_frames_and_train = num_env_frames.assign_add(
        FLAGS.batch_size * FLAGS.unroll_length)

  # Adding a few summaries: RL losses and actions.
  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('rl_loss_policy_gradient',
                    rl_loss_policy_gradient)
  tf.summary.scalar('rl_loss_baseline', rl_loss_baseline)
  tf.summary.scalar('rl_loss_entropy', rl_loss_entropy)
  if 'yaw_label' in observation_names:
    tf.summary.scalar('heading_loss', heading_loss)
  if 'latlng_label' in observation_names:
    tf.summary.scalar('xy_loss', xy_loss)
  if 'target_latlng_label' in observation_names:
    tf.summary.scalar('target_xy_loss', target_xy_loss)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.histogram('action', agent_outputs.action)

  # Adding a few summaries: agent's view and graph.
  idx_frame = observation_names.index('view_image')
  frame = observations[idx_frame]
  tf.summary.image('frame', frame[:3, 0, :, :, :])
  idx_graph = observation_names.index('graph_image')
  street_graph = observations[idx_graph]
  tf.summary.image('street_graph', street_graph[:3, 0, :, :, :])

  # Adding a few summaries: current and target lat/lng.
  idx_latlng = observation_names.index('latlng')
  latlng = observations[idx_latlng]
  tf.summary.histogram('current_lat', latlng[:, 0, 0])
  tf.summary.histogram('current_lng', latlng[:, 0, 1])
  idx_target_latlng = observation_names.index('target_latlng')
  target_latlng = observations[idx_target_latlng]
  target_latlng = tf.Print(target_latlng, [target_latlng])
  tf.summary.histogram('target_lat', target_latlng[:, 0, 0])
  tf.summary.histogram('target_lng', target_latlng[:, 0, 1])

  # Adding a few summaries: yaw.
  if 'yaw' in observation_names:
    idx_yaw = observation_names.index('yaw')
    yaw = observations[idx_yaw]
    tf.summary.histogram('yaw', yaw[:, 0])

  # Adding a few summaries: heading prediction.
  if 'yaw_label' in observation_names:
    img_yaw_labels = tf.expand_dims(
        tf.expand_dims(tf.one_hot(tf.cast(yaw_labels, tf.int32), 16), 1), -1)
    img_yaw_logits = tf.expand_dims(
        tf.expand_dims(tf.nn.softmax(tf.cast(yaw_logits, tf.float32)), 1), -1)
    tf.summary.image("yaw_labels", img_yaw_labels[:, :, 0, :, :])
    tf.summary.image("yaw_logits", img_yaw_logits[:, :, 0, :, :])

  # Adding a few summaries: XY position prediction.
  if 'latlng_label' in observation_names:
    img_xy_labels = plot_logits_2d(
        tf.one_hot(tf.cast(xy_labels[:, 0], tf.int32), 32*32), 32, 32)
    img_xy_logits = plot_logits_2d(
        tf.nn.softmax(tf.cast(xy_logits[:, 0, :], tf.float32)), 32, 32)
    tf.summary.image("xy_labels", img_xy_labels[:, 0, :, :, :])
    tf.summary.image("xy_logits", img_xy_logits[:, 0, :, :, :])

  # Adding a few summaries: XY position prediction.
  if 'target_latlng_label' in observation_names:
    img_target_xy_labels = plot_logits_2d(
        tf.one_hot(tf.cast(target_xy_labels[:, 0], tf.int32), 32*32), 32, 32)
    img_target_xy_logits = plot_logits_2d(
        tf.nn.softmax(tf.cast(target_xy_logits, tf.float32)), 32, 32)
    tf.summary.image("target_xy_labels", img_target_xy_labels[:, 0, :, :, :])
    tf.summary.image("target_xy_logits", img_target_xy_logits[:, 0, :, :, :])

  return done, infos, num_env_frames_and_train


def create_environment(level_name, seed, is_test=False):
  """Creates an environment wrapped in a `FlowEnvironment`."""
  observations = FLAGS.observations.split(';')
  tf.logging.info('Observations requested:')
  tf.logging.info(observations)
  config = {
      'status_height': 0,
      'width': FLAGS.width,
      'height': FLAGS.height,
      'graph_width': FLAGS.graph_width,
      'graph_height': FLAGS.graph_height,
      'graph_zoom': FLAGS.graph_zoom,
      'game_name': FLAGS.game_name,
      'goal_timeout': FLAGS.frame_cap,
      'frame_cap': FLAGS.frame_cap,
      'full_graph': (FLAGS.start_pano == ''),
      'start_pano': FLAGS.start_pano,
      'min_graph_depth': FLAGS.graph_depth,
      'max_graph_depth': FLAGS.graph_depth,
      'proportion_of_panos_with_coins':
          FLAGS.proportion_of_panos_with_coins,
      'timestamp_start_curriculum': FLAGS.timestamp_start_curriculum,
      'hours_curriculum_part_1': FLAGS.hours_curriculum_part_1,
      'hours_curriculum_part_2': FLAGS.hours_curriculum_part_2,
      'min_goal_distance_curriculum': FLAGS.min_goal_distance_curriculum,
      'max_goal_distance_curriculum': FLAGS.max_goal_distance_curriculum,
      'observations': observations,
      'bbox_lat_min': FLAGS.bbox_lat_min,
      'bbox_lat_max': FLAGS.bbox_lat_max,
      'bbox_lng_min': FLAGS.bbox_lng_min,
      'bbox_lng_max': FLAGS.bbox_lng_max,
      'min_radius_meters': FLAGS.min_radius_meters,
      'max_radius_meters': FLAGS.max_radius_meters,
  }

  config = default_config.ApplyDefaults(config)
  tf.logging.info(config)
  game = default_config.CreateGame(config['game_name'], config)
  dataset_path = FLAGS.dataset_paths + '/' + level_name
  tf.logging.info(dataset_path)
  p = py_process.PyProcess(
      StreetLearnImpalaAdapter, dataset_path, config, game)
  return FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
  """Pins global variables to the specified device."""

  def getter(getter, *args, **kwargs):
    var_collections = kwargs.get('collections', None)
    if var_collections is None:
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
      with tf.device(device):
        return getter(*args, **kwargs)
    else:
      return getter(*args, **kwargs)

  with tf.variable_scope('', custom_getter=getter) as vs:
    yield vs


def create_agent(num_actions):
  """Create the agent."""
  assert FLAGS.agent in ['goal_nav_agent', 'city_nav_agent']
  if FLAGS.agent == 'city_nav_agent':
    agent = city_nav_agent.CityNavAgent(
        num_actions, observation_names=FLAGS.observations)
  else:
    agent = goal_nav_agent.GoalNavAgent(
        num_actions, observation_names=FLAGS.observations)
  return agent


def train(action_set, level_names):
  """Train."""

  if is_single_machine():
    local_job_device = ''
    shared_job_device = ''
    is_actor_fn = lambda i: True
    is_learner = True
    global_variable_device = '/gpu'
    server = tf.train.Server.create_local_server()
    server_target = FLAGS.master
    filters = []
  else:
    local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    # Placing the variable on CPU, makes it cheaper to send it to all the
    # actors. Continual copying the variables from the GPU is slow.
    global_variable_device = shared_job_device + '/cpu'
    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
        'learner': ['localhost:8000']
    })
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
    server_target = server.target
    filters = [shared_job_device, local_job_device]

  # Only used to find the actor output structure.
  with tf.Graph().as_default():
    agent = create_agent(len(action_set))
    env = create_environment(level_names[0], seed=1)
    structure = build_actor(agent, env, level_names[0], action_set)
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]
    shapes = [t.shape.as_list() for t in flattened_structure]

  with tf.Graph().as_default(), \
       tf.device(local_job_device + '/cpu'), \
       pin_global_variables(global_variable_device):
    tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

    # Create Queue and Agent on the learner.
    with tf.device(shared_job_device):
      queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
      agent = create_agent(len(action_set))

    # Build actors and ops to enqueue their output.
    enqueue_ops = []
    for i in range(FLAGS.num_actors):
      if is_actor_fn(i):
        level_name = level_names[i % len(level_names)]
        tf.logging.info('Creating actor %d with level %s', i, level_name)
        env = create_environment(level_name, seed=i + 1)
        actor_output = build_actor(agent, env, level_name, action_set)
        with tf.device(shared_job_device):
          enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

    # If running in a single machine setup, run actors with QueueRunners
    # (separate threads).
    if is_learner and enqueue_ops:
      tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    # Build learner.
    if is_learner:
      # Create global step, which is the number of environment frames processed.
      tf.get_variable(
          'num_environment_frames',
          initializer=tf.zeros_initializer(),
          shape=[],
          dtype=tf.int64,
          trainable=False,
          collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

      # Create batch (time major) and recreate structure.
      dequeued = queue.dequeue_many(FLAGS.batch_size)
      dequeued = nest.pack_sequence_as(structure, dequeued)

      def make_time_major(s):
        return nest.map_structure(
            lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]),
            s)

      dequeued = dequeued._replace(
          env_outputs=make_time_major(dequeued.env_outputs),
          agent_outputs=make_time_major(dequeued.agent_outputs))

      with tf.device('/gpu'):
        # Using StagingArea allows us to prepare the next batch and send it to
        # the GPU while we're performing a training step. This adds up to 1 step
        # policy lag.
        flattened_output = nest.flatten(dequeued)
        area = contrib_staging.StagingArea([t.dtype for t in flattened_output],
                                           [t.shape for t in flattened_output])
        stage_op = area.put(flattened_output)

        data_from_actors = nest.pack_sequence_as(structure, area.get())

        # Unroll agent on sequence, create losses and update ops.
        output = build_learner(agent, data_from_actors.agent_state,
                               data_from_actors.env_outputs,
                               data_from_actors.agent_outputs)

    # Create MonitoredSession (to run the graph, checkpoint and log).
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
    # config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
    with tf.train.MonitoredTrainingSession(
        server_target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=config,
        hooks=[py_process.PyProcessHook()]) as session:

      if is_learner:
        tf.logging.info('is_learner')
        # Logging.
        level_returns = {level_name: [] for level_name in level_names}
        summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

        # Prepare data for first run.
        session.run_step_fn(
            lambda step_context: step_context.session.run(stage_op))

        # Execute learning and track performance.
        num_env_frames_v = 0
        while num_env_frames_v < FLAGS.total_environment_frames:
          tf.logging.info(num_env_frames_v)
          level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
              (data_from_actors.level_name,) + output + (stage_op,))
          level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)

          for level_name, episode_return, episode_step in zip(
              level_names_v[done_v],
              infos_v.episode_return[done_v],
              infos_v.episode_step[done_v]):
            episode_frames = episode_step

            tf.logging.info('Level: %s Episode return: %f',
                            level_name, episode_return)

            summary = tf.summary.Summary()
            summary.value.add(tag=level_name + '/episode_return',
                              simple_value=episode_return)
            summary.value.add(tag=level_name + '/episode_frames',
                              simple_value=episode_frames)
            summary_writer.add_summary(summary, num_env_frames_v)

      else:
        tf.logging.info('actor')
        # Execute actors (they just need to enqueue their output).
        while True:
          session.run(enqueue_ops)


def test(action_set, level_names):
  """Test."""

  level_returns = {level_name: [] for level_name in level_names}
  with tf.Graph().as_default():
    agent = create_agent(len(action_set))
    outputs = {}
    for level_name in level_names:
      env = create_environment(level_name, seed=1, is_test=True)
      outputs[level_name] = build_actor(agent, env, level_name, action_set)

    with tf.train.SingularMonitoredSession(
        checkpoint_dir=FLAGS.logdir,
        hooks=[py_process.PyProcessHook()]) as session:
      for level_name in level_names:
        tf.logging.info('Testing level: %s', level_name)
        while True:
          done_v, infos_v = session.run((
              outputs[level_name].env_outputs.done,
              outputs[level_name].env_outputs.info
          ))
          returns = level_returns[level_name]
          returns.extend(infos_v.episode_return[1:][done_v[1:]])

          if len(returns) >= FLAGS.test_num_episodes:
            tf.logging.info('Mean episode return: %f', np.mean(returns))
            break


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  action_set = streetlearn.get_action_set(FLAGS.action_set,
                                          FLAGS.rotation_speed)
  tf.logging.info(action_set)
  level_names = FLAGS.level_names.split(',')
  tf.logging.info(level_names)

  if FLAGS.mode == 'train':
    train(action_set, level_names)
  else:
    test(action_set, level_names)


if __name__ == '__main__':
  tf.app.run()
