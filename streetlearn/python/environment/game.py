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
import collections
import math
import numpy as np
from absl import logging
import six

from streetlearn.engine.python import color

EMPTY_THUMBMNAILS = np.empty((0, 3, 0, 0))
EMPTY_GT_DIRECTION = np.empty((0,))
# When computing which panos B_i are immediately reachable from a given pano A,
# we look at all panos B_i up to depth TOL_DEPTH in a graph whose root is A,
# with a difference in altitude less than TOL_ALT meters, and within a cone
# of TOL_BEARING degrees.
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

  @abc.abstractmethod
  def get_info(self, streetlearn):
    """"Returns current information about the environment.

    Args:
      streetlearn: a StreetLearn instance.
    Returns:
      info: information from the environment at the last step.
    """

  @property
  def goal_id(self):
    """Returns the id of the goal pano, if there is one."""
    return None

  def ground_truth_direction(self):
    """Returns the float angle with the ground truth direction for the agent."""
    return EMPTY_GT_DIRECTION

  def thumbnails(self):
    """Returns observation thumbnails array of shape (batch_size, 3, h, w)."""
    return EMPTY_THUMBMNAILS

  def instructions(self):
    """Returns a string containing game specific instructions."""
    return str()

  def highlighted_panos(self):
    """Returns the list of highlighted panos and their colors."""
    return {}

  @property
  def done(self):
    """Returns a flag indicating the end of the current episode."""
    return True

  def _compute_extended_graphs(self, streetlearn):
    """Compute an extended directed graph accessible to the StreeLearn agent.

    Args:
      streetlearn: the streetlearn environment.
    """
    logging.info('Storing the altitudes of each pano for faster retrieval.')
    altitudes = {}
    for pano_id in six.iterkeys(streetlearn.graph):
      altitudes[pano_id] = streetlearn.engine.GetMetadata(pano_id).pano.alt
    logging.info('Computing the extended directed graph.')
    self._extended_graph_from = {}
    self._extended_graph_to = collections.defaultdict(list)
    num_panos = 0
    num_panos_total = len(streetlearn.graph)
    num_edges_extended = 0
    for current_id in six.iterkeys(streetlearn.graph):

      # Find the neighbors up to depth 3 of the current pano, not separated
      # by a drop of 2m in altitude.
      visited = {}
      queue_panos = [(current_id, 0)]
      while queue_panos:
        elem = queue_panos.pop(0)
        pano_id = elem[0]
        depth = elem[1]
        current_alt = altitudes[current_id]
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
              neighbor_alt = altitudes[neighbor_id]
              if depth == 0 or abs(neighbor_alt - current_alt) < TOL_ALT:
                queue_panos.append((neighbor_id, depth+1))
      visited.pop(current_id)

      # Select only neighbors that are the closest within a tolerance cone,
      # and create extended graphs.
      self._extended_graph_from[current_id] = {}
      for pano_id, (dist, bearing) in six.iteritems(visited):
        retain_pano_id = True
        for other_id, (other_dist, other_bearing) in six.iteritems(visited):
          if ((pano_id != other_id) and
              (180 - abs(abs(bearing - other_bearing) - 180) < TOL_BEARING) and
              (other_dist < dist)):
            retain_pano_id = False
        if retain_pano_id:
          self._extended_graph_from[current_id][pano_id] = (dist, bearing)
          num_edges_extended += 1
          self._extended_graph_to[pano_id].append((current_id, dist, bearing))

      num_panos += 1
      if num_panos % 1000 == 0:
        logging.info('Processed %d/%d panos, %d extended directed edges',
                     num_panos, num_panos_total, num_edges_extended)

  def _bfs(self, queue_panos, graph, flagged, visited):
    """Compute the shortest paths using BFS given a queue and pano graph.

    Args:
      queue_panos: list of tuples (parent_pano_id, child_pano_id).
      graph: dictionary with pano_id keys and lists of pano_id values.
      flagged: dictionary with pano_id keys and boolean values.
      visited: dictionary with child pano_id keys and parent pano id values.
    Returns:
      flagged: dictionary with pano_id keys and boolean values.
      visited: dictionary with child pano_id keys and parent pano id values.
    """
    while queue_panos:
      # Mark the pano at the top of the queue as visited.
      elem = queue_panos.pop(0)
      current_pano_id = elem[0]
      parent_pano_id = elem[1]
      depth = elem[2]
      visited[current_pano_id] = (parent_pano_id, depth)
      # Add the neighbors of the pano.
      if current_pano_id in graph:
        neighbors = graph[current_pano_id]
        for neighbor_id in neighbors:
          if isinstance(neighbor_id, tuple):
            neighbor_id = neighbor_id[0]
          if neighbor_id not in flagged:
            flagged.add(neighbor_id)
            queue_panos.append((neighbor_id, current_pano_id, depth+1))
    return flagged, visited

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

    # Compute the shortest paths from all the panos to the target pano
    # using the direct connection graph.
    logging.info('Computing shortest paths to %s using BFS on direct graph',
                 target_pano_id)
    flagged_direct = set([target_pano_id])
    (_, visited_direct) = self._bfs(
        [(target_pano_id, None, 0)], streetlearn.graph, flagged_direct, {})

    # Compute the shortest paths from all the panos to the target pano
    # using the extended (reachable) graph, with shortcuts.
    logging.info('Computing shortest paths to %s using BFS on extended graph',
                 target_pano_id)
    flagged_extended = set([target_pano_id])
    (_, visited_extended) = self._bfs(
        [(target_pano_id, None, 0)], self._extended_graph_to, flagged_extended,
        {})

    # Some panos may have been missed during the shortest path computation
    # on the extended graph because of the preferential choice of one pano
    # over the other. In order to make sure that there is a path from every
    # pano of the graph to the goal pano, we backfill visited_extended
    # with visited_direct, which is computed on the direct connection graph.
    self._panos_to_goal = {}
    for child, (parent, _) in six.iteritems(visited_direct):
      if child in visited_extended:
        (parent, _) = visited_extended[child]
      self._panos_to_goal[child] = parent

    # Extract the shortest path, from the current starting position, by
    # following the panos to goal as computed by the BFS search that started
    # from the goal.
    current_pano_id = start_pano_id
    list_panos = [current_pano_id]
    next_pano_id = self._panos_to_goal[current_pano_id]
    while next_pano_id:
      list_panos.append(next_pano_id)
      current_pano_id = next_pano_id
      next_pano_id = self._panos_to_goal[current_pano_id]

    # Because of the Street View direct graph connectivity and because of how
    # the StreetLearn extended graph adds edges when two panos that are distant
    # by up to 2 links can still be directly reached, we need to "iron out"
    # the path at street intersections. This code transforms a -> b -> c -> d
    # into a -> b' -> c' -> d if the latter is shorter (in metric distance)
    # and a -> b -> c into a -> b' -> c' -> c if the latter is shorter.
    shortest_path = {}
    num_panos = 0
    while num_panos < len(list_panos)-3:
      a = list_panos[num_panos]
      b = list_panos[num_panos+1]
      c = list_panos[num_panos+2]
      d = list_panos[num_panos+3]
      skipped = 1
      shortest_path[a] = b
      if (a in self._extended_graph_from and
          b in self._extended_graph_from and
          c in self._extended_graph_from and
          b in self._extended_graph_from[a] and
          c in self._extended_graph_from[b] and
          d in self._extended_graph_from[c]):
        (dist_ab, _) = self._extended_graph_from[a][b]
        (dist_bc, _) = self._extended_graph_from[b][c]
        (dist_cd, _) = self._extended_graph_from[c][d]
        dist_abc = dist_ab + dist_bc
        dist_abcd = dist_abc + dist_cd
        for b2 in six.iterkeys(self._extended_graph_from[a]):
          if b2 != b:
            for c2 in six.iterkeys(self._extended_graph_from[b2]):
              for d2 in six.iterkeys(self._extended_graph_from[c2]):
                if d2 == c or d2 == d:
                  (dist_ab2, _) = self._extended_graph_from[a][b2]
                  (dist_bc2, _) = self._extended_graph_from[b2][c2]
                  (dist_cd2, _) = self._extended_graph_from[c2][d2]
                  dist_abcd2 = dist_ab2 + dist_bc2 + dist_cd2
                  if d2 == c and dist_abcd2 < dist_abc:
                    self._panos_to_goal[a] = b2
                    self._panos_to_goal[b2] = c2
                    self._panos_to_goal[c2] = c
                    shortest_path[a] = b2
                    shortest_path[b2] = c2
                    shortest_path[c2] = c
                    logging.info('Replaced %s, %s, %s by %s, %s, %s, %s',
                                 a, b, c, a, b2, c2, d2)
                    skipped = 2
                  if d2 == d and dist_abcd2 < dist_abcd:
                    self._panos_to_goal[a] = b2
                    self._panos_to_goal[b2] = c2
                    self._panos_to_goal[c2] = d
                    shortest_path[a] = b2
                    shortest_path[b2] = c2
                    shortest_path[c2] = d
                    logging.info('Replaced %s, %s, %s, %s by %s, %s, %s, %s',
                                 a, b, c, d, a, b2, c2, d2)
                    skipped = 3
      num_panos += skipped
    while num_panos < len(list_panos)-1:
      shortest_path[list_panos[num_panos]] = list_panos[num_panos+1]
      num_panos +=1

    return shortest_path, num_panos

