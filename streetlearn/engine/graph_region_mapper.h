// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_REGION_MAPPER_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_REGION_MAPPER_H_

#include "streetlearn/engine/vector.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"

namespace streetlearn {

// Helper class for calculating mappings from graph to screen regions and
// between lat/lng and screen co-ordinates.
class GraphRegionMapper {
 public:
  explicit GraphRegionMapper(const Vector2_i& screen_size)
      : screen_size_(screen_size) {}

  // GraphRegionManager is neither copyable nor movable.
  GraphRegionMapper(const GraphRegionMapper&) = delete;
  GraphRegionMapper& operator=(const GraphRegionMapper&) = delete;

  // Sets graph region and the display margin required.
  void SetGraphBounds(const S2LatLngRect& graph_bounds);

  // Sets the bounds of the current view around the centre point.
  void SetCurrentBounds(double zoom, const S2LatLng& image_centre);

  // Re-centers the current view.
  void ResetCurrentBounds(double zoom);

  // Maps the lat/lng provided to screen coordinates.
  Vector2_d MapToScreen(double lat, double lng) const;

  // Maps the lat/lng to coordinates in the buffer of the given dimensions.
  Vector2_d MapToBuffer(double lat, double lng, int image_width,
                        int image_height) const;

 private:
  // Calculates the margin around the graph region to fit the screen shape.
  void CalculateSceneMargin();

  // The size of the screen into which to render.
  Vector2_i screen_size_;

  // The bounding lat/lng of the graph.
  S2LatLngRect graph_bounds_;

  // The bounds of the current display.
  S2LatLngRect current_bounds_;

  // The margin to leave around the map to fit the aspect ratio of the screen.
  S2LatLng margin_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_REGION_MAPPER_H_
