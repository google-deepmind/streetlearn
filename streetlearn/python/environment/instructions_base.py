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

"""StreetLearn basic level for the instruction-following task.

In this environment, the agent receives a reward for every waypoint it hits
as well as a larger reward for reaching the final goal.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging
import numpy as np

from streetlearn.python.environment import coin_game
from streetlearn.python.environment import thumbnail_helper

TrajectoryStep = collections.namedtuple(
    'TrajectoryStep',
    'waypoint_index pano lat lng heading_deg length instruction')
Trajectory = collections.namedtuple('Trajectory', 'steps goal')


class InstructionsBase(coin_game.CoinGame):
  """Instruction following game."""

  def __init__(self, config):
    """Creates an instance of the StreetLearn level.

    Args:
      config: config dict of various settings.
    """
    super(InstructionsBase, self).__init__(config)

    self._colors.update({
        'goal': config['color_for_goal'],
        'waypoint': config['color_for_waypoint'],
        'shortest_path': config['color_for_shortest_path'],
    })
    self._reward_at_waypoint = config['reward_at_waypoint']
    self._reward_at_goal = config['reward_at_goal']
    self._instruction_file = config['instruction_file']
    self._num_instructions = config['num_instructions']
    self._max_instructions = config['max_instructions']
    self._thumbnail_helper = thumbnail_helper.ThumbnailHelper()
    self._thumbnails = np.zeros(
        [self._max_instructions + 1, 3, config['width'], config['height']],
        dtype=np.uint8)
    logging.info('Using %d instructions', self._num_instructions)
    logging.info('Padding to %d instructions', self._max_instructions)
    self._instructions = []
    self._step_counter = 1
    self._reward_pano_id_list = {}
    self._reward_pano_id_to_family = {}
    self._reward_family = {}
    self._pano_id_to_color = {}
    self._goal_pano_id = None
    self._trajectory = None
    self._show_shortest_path = config['show_shortest_path']
    self._calculate_ground_truth = config['calculate_ground_truth']

    # Ground truth direction (for imitation learning agents).
    self._gt_direction = 0

    # Trajectories
    self._num_trajectories = 0
    self._trajectory_data = []
    self._loaded_trajectories = False

  def _load_trajectories(self):
    """Load the trajectories into memory."""
    logging.info('Loading trajectories from %s', self._instruction_file)
    steps = []
    current_trajectory_index = 0
    with open(self._instruction_file, 'r') as f:
      for line in f:
        tokens = line.strip().split('\t')
        trajectory_index = int(tokens[0])
        waypoint_index = int(tokens[1])
        lat = float(tokens[2])
        lng = float(tokens[3])
        heading_deg = float(tokens[4])
        length = float(tokens[5])
        pano_id = tokens[6]
        instruction = tokens[7]
        step = TrajectoryStep(
            waypoint_index=waypoint_index,
            pano=pano_id,
            lat=lat,
            lng=lng,
            heading_deg=heading_deg,
            length=length,
            instruction=instruction)
        if trajectory_index != current_trajectory_index:
          self._add_trajectory(steps)
          steps = []
          current_trajectory_index = trajectory_index
        steps.append(step)
      self._add_trajectory(steps)

    logging.info('Loaded %d trajectories', self._num_trajectories)
    self._loaded_trajectories = True

  def _add_trajectory(self, steps):
    """Store a trajectory."""
    num_steps = len(steps)
    if num_steps > 0:

      # Separate goal from waypoints.
      goal = steps[num_steps-1]
      steps = steps[:(num_steps-1)]

      # Store the trajectory in a hashtable.
      trajectory = Trajectory(steps=steps, goal=goal)
      self._trajectory_data.append(trajectory)
      self._num_trajectories += 1
      if self._num_trajectories % 1000 == 0:
        logging.info('Stored %d trajectories', self._num_trajectories)

  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Selects a random trajectory, extracts the instructions and panos at goal and
    waypoints, computes the shortest paths between each start, each waypoint and
    the goal.

    Args:
      streetlearn: a streetlearn instance.
    Returns:
      A newly populated pano_id_to_color dictionary.
    """
    # Initialise graph of rewards and colors with coins
    super(InstructionsBase, self).on_reset(streetlearn)

    self._current_step = 0
    self._step_counter = 1
    self._step_by_pano = {}
    self._pano_by_step = {}

    self._reward_pano_id_list = {}
    self._reward_pano_id_to_family = {}
    self._reward_family = {}
    self._pano_id_to_color = {}
    self._num_steps_this_goal = 0

    # Randomly sample a trajectory.
    if self._loaded_trajectories == False:
      self._load_trajectories()
    trajectory = self._sample_trajectory(streetlearn)

    start = max(len(trajectory.steps) - self._num_instructions, 0)
    logging.info('Trajectory of length %d (max %d), starting at %d',
                 len(trajectory.steps), self._num_instructions, start)
    num_steps = 0
    start_pano_id = None
    self._instructions = []
    self._thumbnails[:] = 0
    pano_list = []

    for step in trajectory.steps[start:]:
      pano_id = step.pano
      pano_list.append(pano_id)

      # Even if we do not take rewards for waypoints, we store them to keep
      # track of the agent's position along the trajectory.
      if num_steps == 0:
        start_pano_id = pano_id
      if num_steps > 0:
        self._add_reward_to_pano(pano_id, self._reward_at_waypoint,
                                 self._colors['waypoint'], streetlearn)
      self._instructions.append(step.instruction)

      # Fetch the thumbnail for the current step of the trajectory.
      step_thumbnail = self._thumbnail_helper.get_thumbnail(
          streetlearn, pano_id, step.heading_deg)
      if step_thumbnail is not None:
        self._thumbnails[num_steps] = step_thumbnail

      if self._reward_at_waypoint:
        logging.info('Waypoint %d at pano %s, yields reward of %f',
                     num_steps, pano_id, self._reward_at_waypoint)
      else:
        logging.info('Waypoint %d at pano %s', num_steps, pano_id)
      num_steps += 1

    # Set the goal.
    self._goal_pano_id = trajectory.goal.pano
    self._add_reward_to_pano(self._goal_pano_id, self._reward_at_goal,
                             self._colors['goal'], streetlearn)
    pano_list.append(self._goal_pano_id)

    # Store the previously defined coin rewards and colours
    for pano_id in self._coin_pano_id_set:
      self._add_coin_reward_to_pano(pano_id)

    # Add goal pano thumbnail at the end.
    goal_thumbnail = self._thumbnail_helper.get_thumbnail(
        streetlearn, self._goal_pano_id, trajectory.goal.heading_deg)
    if goal_thumbnail is not None:
      self._thumbnails[num_steps] = goal_thumbnail

    # Move and rotate player into start position.
    streetlearn.engine.SetPosition(start_pano_id)
    streetlearn.currentpano_id = start_pano_id
    streetlearn.engine.RotateObserver(trajectory.steps[start].heading_deg, 0)

    logging.info('From: %s (%f, %f), To: %s', start_pano_id,
                 trajectory.steps[start].lat,
                 trajectory.steps[start].lng, self._goal_pano_id)
    logging.info('Trajectory with %d waypoints (goal included)', num_steps)

    if self._calculate_ground_truth or self._show_shortest_path:
      # Update the shortest path to the goal or first waypoint.
      self._update_shortest_path(streetlearn, start_pano_id)
    if self._show_shortest_path:
      # Use the computed shortest path to color the panos.
      self._color_shortest_path(streetlearn)
    # By default, direction is forward.
    self._gt_direction = 0

    return self._pano_id_to_color

  def _update_shortest_path(self, streetlearn, start_pano_id):
    """Update the target of the shortest paths and color panos along that path.

    Args:
      streetlearn: the streetlearn environment.
      start_pano_id: a string for the current pano ID, for computing the optimal
          path.
    """
    step = self._current_step + 1
    logging.info(self._pano_by_step)
    logging.info('Reached step %d', step)
    if step in self._pano_by_step:
      target_pano_id = self._pano_by_step[step]
      self._shortest_path, num_panos = self._shortest_paths(
          streetlearn, target_pano_id, start_pano_id)
      logging.info('Shortest path from %s to waypoint/goal %s covers %d panos',
                   start_pano_id, target_pano_id, num_panos)

  def _color_shortest_path(self, streetlearn):
    """Color panos along the current shortest path to the current target.

    Args:
      streetlearn: the streetlearn environment.
    """
    for pano_id in self._shortest_path:
      self._pano_id_to_color.setdefault(pano_id, self._colors['shortest_path'])

  @property
  def trajectory(self):
    return self._trajectory

  def _sample_trajectory(self, streetlearn):
    """Sample a trajectory.

    Args:
      streetlearn: Streetlearn instance.
    Returns:
      trajectory object.
    """
    trajectory_index = np.random.randint(len(self._trajectory_data))
    self._trajectory = self._trajectory_data[trajectory_index]
    return self.trajectory

  def _add_reward_to_pano(self, pano_id, reward, color, streetlearn):
    """Add reward to a pano and all its neighbours.

    Args:
      pano_id: centroid pano id.
      reward: Amount of reward to attach to this and neighbouring panos.
      color: Color for the goal in the minimap.
      streetlearn: Streetlearn instance
    """
    # If this already has a reward indirectly through a neighbour, undo that.
    if pano_id in self._reward_pano_id_list:
      if self._reward_pano_id_to_family[pano_id] == pano_id:
        # This was already made a reward field; update reward only.
        for neighbor in self._reward_family[pano_id]:
          # Replace reward and colour.
          self._reward_pano_id_list[neighbor] = reward
          self._pano_id_to_color[neighbor] = color
        return
      else:
        # This was previously an indirect reward field.
        # Remove from other family,: continue with default operation.
        self._reward_family[self._reward_pano_id_to_family[pano_id]].remove(
            pano_id)
        self._reward_pano_id_to_family[pano_id] = None

    # Define family around this id.
    self._add_family(pano_id, streetlearn)

    # Add reward and colour to family and links into family.
    for neighbor in self._reward_family[pano_id]:
      self._reward_pano_id_list[neighbor] = reward
      self._reward_pano_id_to_family[neighbor] = pano_id
      self._pano_id_to_color[neighbor] = color

  def _add_coin_reward_to_pano(self, pano_id):
    """Add coin reward to a pano, but only if that pano has no reward yet.

    Args:
      pano_id: centroid pano id.
    """
    if pano_id not in self._reward_pano_id_list:
      self._reward_pano_id_list[pano_id] = self._reward_per_coin
      self._reward_pano_id_to_family[pano_id] = pano_id
      self._reward_family[pano_id] = {pano_id}
      self._pano_id_to_color[pano_id] = self._colors['coin']

  def _add_family(self, pano_id, streetlearn):
    """Add all neighbours of a pano to a list (family) of pano IDs.

    Args:
      pano_id: centroid pano id.
      streetlearn: streetlearn graph for establishing neighbours.
    """
    # If the pano is already part of a reward, do not mess with it.
    if pano_id in self._reward_family:
      return

    # Assign each waypoint with a pano group counter. Used when adding waypoints
    # one by one, in the order of the trajectory.
    if pano_id not in self._step_by_pano:
      logging.info('Added waypoint %d at pano %s', self._step_counter, pano_id)
      self._step_by_pano[pano_id] = self._step_counter
      self._pano_by_step[self._step_counter] = pano_id
      self._step_counter += 1

    # Add the same logic to the immediate neighbours of the pano.
    self._reward_family[pano_id] = set({pano_id})
    pano_metadata = streetlearn.engine.GetMetadata(pano_id)
    for neighbor in pano_metadata.neighbors:
      if neighbor.id not in self._reward_pano_id_to_family:
        self._reward_pano_id_to_family[neighbor.id] = pano_id
        self._reward_family[pano_id].add(neighbor.id)

  def _check_reward(self, pano_id, streetlearn):
    """Check what reward the current pano yields, based on instructions.

    Args:
      pano_id: centroid pano id.
      streetlearn: streetlearn graph for establishing neighbours.
    Returns:
      The reward for the current step.
    """
    reward = 0
    self._reached_goal = False

    # Check if pano ID is in the list of pano IDs that yield rewards.
    if pano_id in self._reward_pano_id_list:
      reward = self._reward_pano_id_list[pano_id]
      family_id = self._reward_pano_id_to_family[pano_id]

      # If the family_id matches the goal, we have finished the trajectory.
      previous_step = self._current_step
      self._current_step = self._step_by_pano[family_id]
      if family_id == self._goal_pano_id:
        self._reached_goal = True
        logging.info('%d: Completed level', streetlearn.frame_count)
        # It appears the level does not end immediately, so we need to reset the
        # step counter manually at this stage to prevent overflow.
        self._current_step = 0
      else:
        logging.info('%d: Moved from %d to %d', streetlearn.frame_count,
                     previous_step, self._current_step)
        if self._calculate_ground_truth or self._show_shortest_path:
          # Update the shortest path to the goal or next waypoint.
          self._update_shortest_path(streetlearn, pano_id)
        if self._show_shortest_path:
          # Use the computed shortest path to color the panos.
          self._color_shortest_path(streetlearn)

      for i in self._reward_family[family_id]:
        del self._reward_pano_id_list[i]
        del self._reward_pano_id_to_family[i]
        del self._pano_id_to_color[i]
      del self._reward_family[family_id]

      # The value of the reward determines if the goal was reached and the
      # episode can now end.
      logging.info('%d: Picked up reward of %f at pano %s.',
                   streetlearn.frame_count, reward, pano_id)

    # Add optional coin rewards.
    if pano_id in self._coin_pano_id_set:
      reward += self._reward_per_coin
      self._coin_pano_id_set.remove(pano_id)

    return reward

  def get_reward(self, streetlearn):
    """Looks at current_pano_id and collects any reward found there.

    Args:
      streetlearn: the streetlearn environment.
    Returns:
      reward: the reward from the last step.
    """
    # Calculate coin, waypoint and goal rewards, determine if end of episode.
    current_pano_id = streetlearn.current_pano_id
    reward = self._check_reward(current_pano_id, streetlearn)
    self._num_steps_this_goal += 1
    return reward

  def get_info(self, streetlearn):
    """"Returns current information about the state of the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """
    info = super(InstructionsBase, self).get_info(streetlearn)
    info['num_steps_this_goal'] = self._num_steps_this_goal
    info['current_step'] = self._current_step
    info['current_goal_id'] = self._goal_pano_id
    info['distance_to_goal'] = streetlearn.engine.GetPanoDistance(
        streetlearn.current_pano_id, self._goal_pano_id)
    info['reward_current_goal'] = self._reward_at_goal
    if self._calculate_ground_truth:
      current_pano_id = streetlearn.current_pano_id
      next_pano_id = self._panos_to_goal[current_pano_id]
      info['next_pano_id'] = next_pano_id
      if next_pano_id:
        bearing_to_next_pano = streetlearn.engine.GetPanoBearing(
            current_pano_id, next_pano_id) - streetlearn.engine.GetYaw()
      else:
        bearing_to_next_pano = 0
      info['bearing_to_next_pano'] = (bearing_to_next_pano + 180) % 360 - 180
    return info

  def done(self):
    """Returns a flag indicating the end of the current episode.

    This game ends only at the end of the episode or if the goal is reached.
    """
    if self._reached_goal:
      self._reached_goal = False
      return True
    else:
      return False

  def thumbnails(self):
    """Returns extra observation thumbnails.

    Args:
      include_goal_thumb: Bool (default: False) of whether we add the goal.
    Returns:
      thumbnails: Thumbnails array of shape (batch_size, 3, h, w)
    """
    return self._thumbnails

  def instructions(self):
    """Returns instructions.

    Args:
      None
    Returns:
      instructions: string containing game specific instructions.
    """
    return self._instructions

  @property
  def goal_id(self):
    """Returns the id of the goal Pano."""
    return self._goal_pano_id

  def on_step(self, streetlearn):
    """Update the ground truth direction to take and the set of touched panos.

    Args:
      streetlearn: the streetlearn environment.
    """
    super(InstructionsBase, self).on_step(streetlearn)

    if self._calculate_ground_truth:
      # streetlearn.current_pano_id is not always updated.
      current_pano_id = streetlearn.engine.GetPano().id
      # What is the next pano and what is the direction to the pano?
      next_pano_id = self._panos_to_goal[current_pano_id]
      if next_pano_id:
        yaw = streetlearn.engine.GetYaw()
        bearing = streetlearn.engine.GetPanoBearing(
            current_pano_id, next_pano_id) - yaw
        self._gt_direction = (bearing + 180) % 360 - 180
      else:
        self._gt_direction = 0

  def ground_truth_direction(self):
    """Returns the ground truth direction to take.

    Returns:
      ground_truth_direction: Float angle with the ground truth direction
          to be taken for the agent to go towards the goal.
    """
    return self._gt_direction
