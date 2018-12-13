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

"""Base class for all StreetLearn levels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
from absl import logging
import six

from streetlearn.engine.python import color

TOL_ALT = 2.0
TOL_DEPTH = 3
TOL_BEARING = 30


@six.add_metaclass(abc.ABCMeta)
class Game(object):
  """Base class for streetlearn levels."""

  @abc.abstractmethod
  def on_reset(self, streetlearn):
    """Gets called after StreetLearn:reset().

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      Dictionary that maps certain pano IDs to colors.
    """
    return {}

  @abc.abstractmethod
  def on_step(self, streetlearn):
    """"Gets called after StreetLearn:step().

    Args:
      streetlearn: a StreetLearn instance.
    """

  @abc.abstractmethod
  def get_reward(self, streetlearn):
    """Returns the reward from the last step.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      reward: the reward from the last step.
    """

  @property
  def goal_id(self):
    """Returns the id of the goal pano, if there is one."""
    return None

  @property
  def done(self):
    """Returns a flag indicating the end of the current episode."""
    return True

  # TODO(b/120770007) Move this code out to a different function.
  def _compute_extended_graphs(self, streetlearn):
    """Compute an extended directed graph accessible to the StreeLearn agent.

    Args:
      streetlearn: the streetlearn environment.
    """
    logging.info('Computing the extended directed graph.')
    self._extended_graph_from = {}
    self._extended_graph_to = {}
    num_panos = 0
    num_panos_total = len(streetlearn.graph)
    num_edges_extended = 0
    for current_id in streetlearn.graph.iterkeys():

      # Find the neighbors up to depth 3 of the current pano, not separated
      # by a drop of 2m in altitude.
      current_metadata = streetlearn.engine.GetMetadata(current_id).pano
      visited = {}
      queue_panos = [(current_id, 0)]
      while queue_panos:
        elem = queue_panos.pop(0)
        pano_id = elem[0]
        depth = elem[1]
        visited[pano_id] = depth
        if depth > 0:
          # Store the distance and bearing to each neighbor.
          dist = streetlearn.engine.GetPanoDistance(current_id, pano_id)
          bearing = streetlearn.engine.GetPanoBearing(current_id, pano_id)
          visited[pano_id] = (dist, bearing)
        # Look for new neighbors recursively.
        if depth < TOL_DEPTH:
          neighbors = streetlearn.graph[pano_id]
          for neighbor_id in neighbors:
            if neighbor_id not in visited:
              neighbor_metadata = streetlearn.engine.GetMetadata(
                  neighbor_id).pano
              if (depth == 0 or
                  abs(neighbor_metadata.alt - current_metadata.alt) < TOL_ALT):
                queue_panos.append((neighbor_id, depth+1))
      visited.pop(current_id)

      # Select only neighbors that are the closest within a tolerance cone,
      # and create extended graphs.
      self._extended_graph_from[current_id] = []
      for pano_id, (dist, bearing) in visited.iteritems():
        retain_pano_id = True
        for other_id, (other_dist, other_bearing) in visited.iteritems():
          if ((pano_id != other_id) and
              (180 - abs(abs(bearing - other_bearing) - 180) < TOL_BEARING) and
              (other_dist < dist)):
            retain_pano_id = False
        if retain_pano_id:
          self._extended_graph_from[current_id].append((pano_id, dist, bearing))
          num_edges_extended += 1
          if pano_id in self._extended_graph_to:
            self._extended_graph_to[pano_id].append((current_id, dist, bearing))
          else:
            self._extended_graph_to[pano_id] = [(current_id, dist, bearing)]

      num_panos += 1
      if num_panos % 1000 == 0:
        logging.info('Processed %d/%d panos, %d extended directed edges',
                     num_panos, num_panos_total, num_edges_extended)

  def _shortest_paths(self, streetlearn, target_pano_id, start_pano_id):
    """Compute the shortest paths from all the panos to a given start pano.

    Args:
      streetlearn: the streetlearn environment.
      target_pano_id: a string for the target pano ID.
      start_pano_id: a string for the current pano ID, for computing the optimal
          path.
    Returns:
      shortest_path: dictionary containing (current_pano_id, next_pano_id)
          as (key, value) pairs.
      num_panos: integer number of panos in the shortest path.
    """
    # The shortest path relies on the extended directed graph.
    if not hasattr(self, '_extended_graph_from'):
      self._compute_extended_graphs(streetlearn)

    # Compute the shortest paths from all the panos to the target pano.
    logging.info('Computing shortest path to %s using BFS on the graph',
                 target_pano_id)
    visited = {}
    flagged = {}
    queue_panos = [(target_pano_id, None, 0)]
    flagged[target_pano_id] = True
    graph = self._extended_graph_to
    while queue_panos:
      # Mark the pano at the top of the queue as visited.
      elem = queue_panos.pop(0)
      current_pano_id = elem[0]
      parent_pano_id = elem[1]
      depth = elem[2]
      visited[current_pano_id] = parent_pano_id
      # Add the neighbors of the pano.
      if current_pano_id in graph:
        neighbors = graph[current_pano_id]
        for neighbor in neighbors:
          neighbor_id = neighbor[0]
          if neighbor_id not in flagged:
            flagged[neighbor_id] = True
            queue_panos.append((neighbor_id, current_pano_id, depth+1))
    logging.info('Computed the shortest paths to the goal using BFS')
    self._panos_to_goal = visited

    # Compute the shortest path from the current starting position.
    shortest_path = {}
    num_panos = 0
    current_pano_id = start_pano_id
    next_pano_id = self._panos_to_goal[current_pano_id]
    while next_pano_id:
      shortest_path[current_pano_id] = next_pano_id
      current_pano_id = next_pano_id
      next_pano_id = self._panos_to_goal[current_pano_id]
      num_panos += 1
    return shortest_path, num_panos

