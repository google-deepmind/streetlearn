"""Thumbnail helper class used in Taxi and Streetlang levels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ThumbnailHelper(object):
  """Thumbnail helper class."""

  def __init__(self):
    self._width = None
    self._height = None

  def get_thumbnail(self, streetlearn, pano_id, heading):
    """Fetch the thumbnail from the environment.

    Args:
      streetlearn: a streetlearn instance.
      pano_id: Pano id of the thumbnail.
      heading: Heading in degrees for the thumbnail.

    Returns:
      Thumbnail ndarray.
    """
    observation = streetlearn.goto(pano_id, heading)
    thumbnail = observation['view_image_hwc']
    if not self._width:
      self._width = streetlearn.config['width']
      self._height = streetlearn.config['height']
    thumbnail = thumbnail.reshape([self._height, self._width, 3])
    return thumbnail
