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

"""Classes to handle the various StreetLearn observation types."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np

_METADATA_COUNT = 14
_NUM_HEADING_BINS = 16
_NUM_LAT_BINS = 32
_NUM_LNG_BINS = 32


class Observation(object):
  """Base class for all observations."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, streetlearn):
    self._streetlearn = streetlearn

  @abc.abstractproperty
  def observation_spec(self):
    """The observation_spec for this observation."""
    pass

  @abc.abstractproperty
  def observation(self):
    """The observation data."""
    pass

  @classmethod
  def create(cls, name, streetlearn):
    """Dispatches an Observation based on `name`."""
    observations = [ViewImage, GraphImage, Yaw, Pitch, Metadata, TargetMetadata,
                    LatLng, TargetLatLng, YawLabel, Neighbors, LatLngLabel,
                    TargetLatLngLabel, Thumbnails, Instructions,
                    GroundTruthDirection]
    dispatch = {o.name: o for o in observations}
    try:
      return dispatch[name](streetlearn)
    except KeyError:
      raise ValueError('No Observation named %s found' % name)


class ViewImage(Observation):
  """RGB pixel data of the view."""
  name = 'view_image'
  observation_spec_dtypes = np.uint8

  def __init__(self, streetlearn):
    super(ViewImage, self).__init__(streetlearn)
    self._width = streetlearn.config["width"]
    self._height = streetlearn.config["height"]
    self._depth = 3
    self._buffer = np.empty(self._depth * self._width * self._height,
                            dtype=np.uint8)

  @property
  def observation_spec(self):
    return [self._depth, self._height, self._width]

  @property
  def observation(self):
    self._streetlearn.engine.RenderObservation(self._buffer)
    return self._buffer


class GraphImage(Observation):
  """RGB pixel data of the graph."""
  name = 'graph_image'
  observation_spec_dtypes = np.uint8

  def __init__(self, streetlearn):
    super(GraphImage, self).__init__(streetlearn)
    self._width = streetlearn.config["graph_width"]
    self._height = streetlearn.config["graph_height"]
    self._depth = 3
    self._buffer = np.empty(self._depth * self._width * self._height,
                            dtype=np.uint8)

  @property
  def observation_spec(self):
    return [self._depth, self._height, self._width]

  @property
  def observation(self):
    highlighted_panos = self._streetlearn.game.highlighted_panos()
    self._streetlearn.engine.DrawGraph(highlighted_panos, self._buffer)
    return self._buffer


class Yaw(Observation):
  """The agent's current yaw (different from the current pano heading)."""
  name = 'yaw'
  observation_spec_dtypes = np.float64

  @property
  def observation_spec(self):
    return [0]

  @property
  def observation(self):
    return np.array(self._streetlearn.engine.GetYaw(), dtype=np.float64)


class Pitch(Observation):
  """The agent's current pitch."""
  name = 'pitch'
  observation_spec_dtypes = np.float64

  @property
  def observation_spec(self):
    return [0]

  @property
  def observation(self):
    return np.array(self._streetlearn.engine.GetPitch(), dtype=np.float64)


class YawLabel(Observation):
  """The agent's current yaw (different from the current pano heading)."""
  name = 'yaw_label'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return [0]

  @property
  def observation(self):
    yaw = self._streetlearn.engine.GetYaw() % 360.0
    return np.array(yaw * _NUM_HEADING_BINS / 360.0, dtype=np.uint8)


class Metadata(Observation):
  """Metadata about the current pano."""
  name = 'metadata'
  observation_spec_dtypes = bytearray

  @property
  def observation_spec(self):
    return [_METADATA_COUNT]

  @property
  def observation(self):
    pano_id = self._streetlearn.current_pano_id
    return bytearray(self._streetlearn.engine.GetMetadata(
        pano_id).pano.SerializeToString())


class TargetMetadata(Observation):
  """Metadata about the target pano."""
  name = 'target_metadata'
  observation_spec_dtypes = bytearray

  @property
  def observation_spec(self):
    return [_METADATA_COUNT]

  @property
  def observation(self):
    goal_id = self._streetlearn.game.goal_id
    if goal_id:
      return bytearray(self._streetlearn.engine.GetMetadata(
          goal_id).pano.SerializeToString())
    return bytearray()


class LatLng(Observation):
  """The agent's current lat/lng coordinates, scaled according to config params
  using bbox_lat_min and bbox_lat_max, as well as bbox_lng_min and bbox_lng_max,
  so that scaled lat/lng take values between 0 and 1 within the bounding box.
  """
  name = 'latlng'
  observation_spec_dtypes = np.float64

  def _scale_lat(self, lat):
    den = self._streetlearn.bbox_lat_max - self._streetlearn.bbox_lat_min
    return ((lat - self._streetlearn.bbox_lat_min) / den if (den > 0) else 0)

  def _scale_lng(self, lng):
    den = self._streetlearn.bbox_lng_max - self._streetlearn.bbox_lng_min
    return ((lng - self._streetlearn.bbox_lng_min) / den if (den > 0) else 0)

  @property
  def observation_spec(self):
    return [2]

  @property
  def observation(self):
    pano_id = self._streetlearn.current_pano_id
    pano_data = self._streetlearn.engine.GetMetadata(pano_id).pano
    lat_scaled = self._scale_lat(pano_data.coords.lat)
    lng_scaled = self._scale_lng(pano_data.coords.lng)
    return np.array([lat_scaled, lng_scaled], dtype=np.float64)


class LatLngLabel(LatLng):
  """The agent's current yaw (different from the current pano heading)."""
  name = 'latlng_label'
  observation_spec_dtypes = np.int32

  def _latlng_bin(self, pano_id):
    pano_data = self._streetlearn.engine.GetMetadata(pano_id).pano
    lat_bin = np.floor(self._scale_lat(pano_data.coords.lat) * _NUM_LAT_BINS)
    lat_bin = np.max([np.min([lat_bin, _NUM_LAT_BINS-1]), 0])
    lng_bin = np.floor(self._scale_lng(pano_data.coords.lng) * _NUM_LNG_BINS)
    lng_bin = np.max([np.min([lng_bin, _NUM_LNG_BINS-1]), 0])
    latlng_bin = lat_bin * _NUM_LNG_BINS + lng_bin
    return latlng_bin

  @property
  def observation_spec(self):
    return [0]

  @property
  def observation(self):
    pano_id = self._streetlearn.current_pano_id
    return np.array(self._latlng_bin(pano_id), dtype=np.int32)


class TargetLatLng(LatLng):
  """The agent's target lat/lng coordinates."""
  name = 'target_latlng'

  @property
  def observation(self):
    goal_id = self._streetlearn.game.goal_id
    if goal_id:
      pano_data = self._streetlearn.engine.GetMetadata(goal_id).pano
      lat_scaled = self._scale_lat(pano_data.coords.lat)
      lng_scaled = self._scale_lng(pano_data.coords.lng)
      return np.array([lat_scaled, lng_scaled], dtype=np.float64)
    return np.array([0, 0], dtype=np.float64)


class TargetLatLngLabel(LatLngLabel):
  """The agent's current yaw (different from the current pano heading)."""
  name = 'target_latlng_label'

  @property
  def observation(self):
    goal_id = self._streetlearn.game.goal_id
    if goal_id:
      return np.array(self._latlng_bin(goal_id), dtype=np.int32)
    return np.array(0, dtype=np.int32)


class Thumbnails(Observation):
  """Thumbnails' pixel data."""
  name = 'thumbnails'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return self._streetlearn.game.thumbnails().shape

  @property
  def observation(self):
    return self._streetlearn.game.thumbnails()


class Instructions(Observation):
  """StreetLang instructions."""
  name = 'instructions'
  observation_spec_dtypes = str

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    return ('|'.join(self._streetlearn.game.instructions())).encode('utf-8')


class GroundTruthDirection(Observation):
  """Direction in degrees that the agent needs to take now."""
  name = 'ground_truth_direction'
  observation_spec_dtypes = np.float32

  @property
  def observation_spec(self):
    return [0]

  @property
  def observation(self):
    return np.array(self._streetlearn.game.ground_truth_direction(),
                    dtype=np.float32)


class Neighbors(Observation):
  """IDs of neighboring panos."""
  name = 'neighbors'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return [self._streetlearn.neighbor_resolution]

  @property
  def observation(self):
    return np.asarray(
        self._streetlearn.engine.GetNeighborOccupancy(
            self._streetlearn.neighbor_resolution),
        dtype=np.uint8)

