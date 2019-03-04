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

"""StreetLearn level for the instruction-following game with a curriculum.

This environment implements the instruction-following game and selects levels
given a particular curriculum strategy, either by slowly increasing the number
of instructions per episode, the maximum distance of routes, or both.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from absl import logging
import numpy as np
import six
from six.moves import range

from streetlearn.python.environment import instructions_base

_SECONDS_IN_HOUR = 3600


# Curriculum constants.
CURRICULUM_NONE = 0
CURRICULUM_LENGTH_BASED = 1
CURRICULUM_INSTR_BASED = 2
CURRICULUM_LENGTH_INSTR_BASED = 3

# Verbosity constants
NUM_TRAJECTORIES_VERBOSE = 10000


class InstructionsCurriculum(instructions_base.InstructionsBase):
  """Instruction following game with curriculum on distance or #instructions."""

  def __init__(self, config):
    """Creates an instance of the StreetLearn level.

    Args:
      config: config dict of various settings.
    """
    super(InstructionsCurriculum, self).__init__(config)

    # Curriculum types: 0 = none, 1 = dist. to goal, 2 = instructions
    self._curriculum_type = config['instruction_curriculum_type']
    self._timestamp_start = config['timestamp_start_curriculum']
    self._hours_curriculum_part_1 = config['hours_curriculum_part_1']
    self._hours_curriculum_part_2 = config['hours_curriculum_part_2']
    self._steps_plateau = config.get('curriculum_steps_plateau', 0)
    self._steps_ramp = config.get('curriculum_steps_ramp', 0)
    self._curriculum_num_instructions_part_1 = config[
        'curriculum_num_instructions_part_1']
    self._min_goal_distance = config['min_goal_distance_curriculum']
    self._max_goal_distance = config['max_goal_distance_curriculum']
    self._curriculum_bin_distance = config['curriculum_bin_distance']
    if self._curriculum_type != CURRICULUM_NONE:
      logging.info('Curriculum starting at time %f', self._timestamp_start)
    if (self._curriculum_type == CURRICULUM_LENGTH_BASED) or (
        self._curriculum_type == CURRICULUM_LENGTH_INSTR_BASED):
      logging.info('Initial plateau: trajectories of distance at most %f',
                   self._min_goal_distance)
      logging.info('Training ramps up to trajectories of distance at most %f',
                   self._max_goal_distance)
      logging.info('Trajectories sorted in bins of distance length %f',
                   self._curriculum_bin_distance)
    if (self._curriculum_type == CURRICULUM_INSTR_BASED) or (
        self._curriculum_type == CURRICULUM_LENGTH_INSTR_BASED):
      logging.info('Initial plateau: trajectories with %d instructions',
                   self._curriculum_num_instructions_part_1)
      logging.info('Training ramps up to trajectories with %d instructions',
                   self._num_instructions)
    logging.info('Initial training plateau lasts for %f hours',
                 self._hours_curriculum_part_1)
    logging.info('Training ramps up to longer traj. for %f hours',
                 self._hours_curriculum_part_2)

    # Frame cap curriculum
    self._curriculum_frame_cap = config['curriculum_frame_cap']
    self._curriculum_frame_cap_part_1 = config['curriculum_frame_cap_part_1']
    self._curriculum_frame_cap_extra_steps = max(
        0, config['frame_cap'] - self._curriculum_frame_cap_part_1)
    if self._curriculum_frame_cap:
      logging.info('Initial plateau: trajectories with %d frames',
                   self._curriculum_frame_cap_part_1)
      logging.info('Training ramps up to trajectories with %d extra frames',
                   self._curriculum_frame_cap_extra_steps)

    self._init_complete = self.initialize_curricula(True)

  def initialize_curricula(self, first_init=False):
    """Initializes the curriculum code.

    Args:
      first_init: If true, container variables are created. Should
        be false for all subsequent calls.

    Returns:
      True if curriculum has been fully established, False otherwise.
    """

    num_bins_distance = int(math.ceil(
        self._max_goal_distance / self._curriculum_bin_distance))
    if first_init:
      self._curriculum_count = 0
      self._trajectory_data_map_per_distance = {
          k: [] for k in range(num_bins_distance + 1)
      }
      self._trajectory_data_map_per_waypoints = {
          k: [] for k in range(self._num_instructions + 1)
      }

    if self._curriculum_type == CURRICULUM_LENGTH_BASED:
      # Bin the trajectories by length
      for index in range(self._num_trajectories):
        v = self._trajectory_data[index]
        self._curriculum_count += 1
        # Note: goal.length stores the overall length of a route.
        bin_distance = int(
            math.ceil(v.goal.length / self._curriculum_bin_distance))
        for k in range(bin_distance, num_bins_distance + 1):
          self._trajectory_data_map_per_distance[k].append(index)
        if (self._curriculum_count % NUM_TRAJECTORIES_VERBOSE) == 0:
          logging.info('Processed %d trajectories', self._curriculum_count)
          return False
      for k in range(num_bins_distance + 1):
        logging.info('%d trajectories with distance at most %f',
                     len(self._trajectory_data_map_per_distance[k]),
                     k * self._curriculum_bin_distance)

    if self._curriculum_type == CURRICULUM_INSTR_BASED:
      # Bin the trajectories by number of instructions (waypoints)
      for index in range(self._num_trajectories):
        v = self._trajectory_data[index]
        self._curriculum_count += 1
        num_waypoints = len(v.steps)
        for k in range(num_waypoints, self._num_instructions + 1):
          self._trajectory_data_map_per_waypoints[k].append(index)
        if (self._curriculum_count % NUM_TRAJECTORIES_VERBOSE) == 0:
          logging.info('Processed %d trajectories', self._curriculum_count)
          return False
      for k in range(self._num_instructions + 1):
        logging.info('%d trajectories with %d instructions',
                     len(self._trajectory_data_map_per_waypoints[k]), k)

    if self._curriculum_type == CURRICULUM_LENGTH_INSTR_BASED:
      # Bin the trajectories by length and instructions
      for index in range(self._num_trajectories):
        v = self._trajectory_data[index]
        self._curriculum_count += 1
        bin_distance = int(
            math.ceil(v.goal.length / self._curriculum_bin_distance))
        for k in range(bin_distance, num_bins_distance + 1):
          self._trajectory_data_map_per_distance[k].append(index)
        num_waypoints = len(v.steps)
        for k in range(num_waypoints, self._num_instructions + 1):
          self._trajectory_data_map_per_waypoints[k].append(index)
        if (self._curriculum_count % NUM_TRAJECTORIES_VERBOSE) == 0:
          logging.info('Processed %d trajectories', self._curriculum_count)
          return False
      for k in range(num_bins_distance + 1):
        logging.info('%d trajectories with distance at most %f',
                     len(self._trajectory_data_map_per_distance[k]),
                     k * self._curriculum_bin_distance)
      for k in range(self._num_instructions + 1):
        logging.info('%d trajectories with %d instructions',
                     len(self._trajectory_data_map_per_waypoints[k]), k)

    return True

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    # Continue initialization of the curricula.
    if not self._init_complete:
      self._init_complete = self.initialize_curricula()
    return super(InstructionsCurriculum, self).on_reset(streetlearn)

  def _ratio_training(self):
    """Updates the fraction of training curriculum based on elapsed time."""
    hours_train = (time.time() - self._timestamp_start) / _SECONDS_IN_HOUR
    if hours_train > self._hours_curriculum_part_1:
      ratio_training = hours_train - self._hours_curriculum_part_1
      ratio_training /= self._hours_curriculum_part_2
      ratio_training = max(min(ratio_training, 1.0), 0.0)
    else:
      ratio_training = 0
    logging.info('Hours elapsed: %f, ratio: %f', hours_train, ratio_training)
    return ratio_training

  def _sample_trajectory(self, streetlearn):
    """Sample a trajectory.

    Args:
      streetlearn: Streetlearn instance.
    Returns:
      trajectory object.
    """
    if self._curriculum_type != CURRICULUM_NONE or self._curriculum_frame_cap:
      ratio_training = self._ratio_training()

    if self._curriculum_frame_cap:
      # Is there a curriculum on the cap on the number of frames?
      prev_frame_cap = streetlearn.frame_cap
      frame_cap = int(
          math.ceil(self._curriculum_frame_cap_part_1 + ratio_training *
                    self._curriculum_frame_cap_extra_steps))
      streetlearn.frame_cap = frame_cap
      if prev_frame_cap != frame_cap:
        logging.info('Changing frame cap from %d to %d', prev_frame_cap,
                     frame_cap)

    if self._curriculum_type == CURRICULUM_NONE:
      # Skip the curriculum sampling
      return super(InstructionsCurriculum, self)._sample_trajectory(streetlearn)

    if self._curriculum_type == CURRICULUM_LENGTH_BASED:
      # Curriculum based on the length/distance (in m) from start to goal.
      max_distance = self._min_goal_distance
      extra_distance = max(0, self._max_goal_distance - self._min_goal_distance)
      max_distance += math.ceil(extra_distance * ratio_training)
      logging.info('Max distance: %f', max_distance)
      bin_distance = int(math.ceil(
          max_distance / self._curriculum_bin_distance))
      map_trajectories = self._trajectory_data_map_per_distance[bin_distance]
    if self._curriculum_type == CURRICULUM_INSTR_BASED:
      # Curriculum based on the number of instructions/waypoints.
      max_num_instructions = self._curriculum_num_instructions_part_1
      num_extra_instructions = max(
          0, self._num_instructions - self._curriculum_num_instructions_part_1)
      max_num_instructions += math.ceil(num_extra_instructions * ratio_training)
      logging.info('Max #instructions: %d', max_num_instructions)
      map_trajectories = self._trajectory_data_map_per_waypoints[
          max_num_instructions]
    if self._curriculum_type == CURRICULUM_LENGTH_INSTR_BASED:
      # Curriculum based both on the number of instructions and on length;
      # at the beginning, only short trajectories with few waypoints are sampled
      # and at the end, long trajectories with may waypoints are sampled too.
      # The final set of trajectories from which one can sample is the
      # intersection of the set of length-based curriculum trajectories and of
      # the set of instruction-based curriculum trajectories.
      max_distance = self._min_goal_distance
      extra_distance = max(0, self._max_goal_distance - self._min_goal_distance)
      max_distance += math.ceil(extra_distance * ratio_training)
      logging.info('Max distance: %f', max_distance)
      bin_distance = int(math.ceil(
          max_distance / self._curriculum_bin_distance))
      map_trajectories_1 = self._trajectory_data_map_per_distance[bin_distance]
      max_num_instructions = self._curriculum_num_instructions_part_1
      num_extra_instructions = max(
          0, self._num_instructions - self._curriculum_num_instructions_part_1)
      max_num_instructions += math.ceil(num_extra_instructions * ratio_training)
      logging.info('Max #instructions: %d', max_num_instructions)
      map_trajectories_2 = self._trajectory_data_map_per_waypoints[
          max_num_instructions]
      map_trajectories = list(set(map_trajectories_1) & set(map_trajectories_2))
      logging.info('Intersection of two sets: %d & %d -> %d',
                   len(map_trajectories_1), len(map_trajectories_2),
                   len(map_trajectories))

    if map_trajectories:
      i = np.random.choice(map_trajectories)
      self._trajectory = self._trajectory_data[i]
      return self._trajectory

    logging.info('Could not find trajectories for ratio training time/steps %f',
                 ratio_training)
    logging.info('Sampling trajectory without curriculum')
    self._trajectory = super(StreetLangTimedCurriculum,
                             self)._sample_trajectory(streetlearn)
    return self._trajectory
