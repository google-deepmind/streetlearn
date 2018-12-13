# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for streetlearn_engine clif bindings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from absl.testing import absltest
from streetlearn.engine.python import streetlearn_engine
from streetlearn.engine.python.test_dataset import TestDataset

_PANO_OFFSET = 32


class StreetlearnEngineTest(absltest.TestCase):

  def setUp(self):
    TestDataset.Generate()
    self.dataset_path = TestDataset.GetPath()

  def test_build_graph(self):
    engine = streetlearn_engine.StreetLearnEngine.Create(
        self.dataset_path, TestDataset.kImageWidth, TestDataset.kImageHeight)
    engine.InitEpisode(0, 0)

    root = engine.BuildGraphWithRoot('1')
    self.assertEqual(root, '1')

    # Check that the right sized graph is returned.
    engine.BuildRandomGraph()
    graph = engine.GetGraph()
    self.assertEqual(len(graph), TestDataset.kPanoCount)

  def test_set_position(self):
    engine = streetlearn_engine.StreetLearnEngine.Create(
        self.dataset_path, TestDataset.kImageWidth, TestDataset.kImageHeight)
    engine.InitEpisode(0, 0)
    engine.BuildGraphWithRoot('1')

     # Set position a couple of times and check the result.
    self.assertEqual(engine.SetPosition('1'), '1')
    self.assertEqual(engine.GetPano().id, '1')
    self.assertEqual(engine.SetPosition('2'), '2')
    self.assertEqual(engine.GetPano().id, '2')

    # Currently facing north so cannot move to the next pano.
    self.assertEqual(engine.MoveToNextPano(), '2')

    # Rotate to face the next pano and move should succeed.
    engine.RotateObserver(_PANO_OFFSET, 0.0)
    self.assertEqual(engine.MoveToNextPano(), '3')
    self.assertEqual(engine.GetPano().id, '3')

  def test_pano_calculations(self):
    engine = streetlearn_engine.StreetLearnEngine.Create(
        self.dataset_path, TestDataset.kImageWidth, TestDataset.kImageHeight)
    engine.InitEpisode(0, 0)
    engine.BuildGraphWithRoot('1')

    self.assertEqual(engine.GetPitch(), 0)
    self.assertEqual(engine.GetYaw(), 0)
    self.assertAlmostEqual(engine.GetPanoDistance('1', '2'), 130.902, 3)

  def test_observation(self):
    engine = streetlearn_engine.StreetLearnEngine.Create(
        self.dataset_path, TestDataset.kImageWidth, TestDataset.kImageHeight)
    engine.InitEpisode(0, 0)
    engine.BuildGraphWithRoot('1')

    # Check that obervations have the right values.
    buffer_size = 3 * TestDataset.kImageWidth * TestDataset.kImageHeight
    obs = np.zeros(buffer_size, dtype=np.ubyte)
    engine.RenderObservation(obs)
    for i in range(0, TestDataset.kImageHeight):
      for j in range(0, TestDataset.kImageWidth):
        index = i * TestDataset.kImageWidth + j
        self.assertIn(obs[index], range(0, 232))

  def test_neighbors(self):
    engine = streetlearn_engine.StreetLearnEngine.Create(
        self.dataset_path, TestDataset.kImageWidth, TestDataset.kImageHeight)
    engine.InitEpisode(0, 0)
    engine.BuildGraphWithRoot('1')
    engine.SetPosition('2')

    # Should have two neighbors.
    occupancy = engine.GetNeighborOccupancy(4)
    self.assertEqual(len(occupancy), 4)
    self.assertEqual(occupancy[0], 1)
    self.assertEqual(occupancy[1], 0)
    self.assertEqual(occupancy[2], 1)
    self.assertEqual(occupancy[3], 0)

  def test_metadata(self):
    engine = streetlearn_engine.StreetLearnEngine.Create(
        self.dataset_path, TestDataset.kImageWidth, TestDataset.kImageHeight)
    engine.InitEpisode(0, 0)
    engine.BuildGraphWithRoot('1')

    # Check that the right metadata is returned.
    metadata = engine.GetMetadata('1')
    self.assertEqual(metadata.pano.id, '1')
    self.assertEqual(len(metadata.neighbors), 1)
    self.assertEqual(metadata.neighbors[0].id, '2')
    self.assertEqual(metadata.graph_depth, 10)

if __name__ == '__main__':
  absltest.main()
