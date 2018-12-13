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

#include "streetlearn/engine/graph_region_mapper.h"

#include "streetlearn/engine/logging.h"
#include "streetlearn/engine/vector.h"
#include "s2/s1angle.h"

namespace streetlearn {

constexpr double kMargin = 0.05;

void GraphRegionMapper::SetGraphBounds(const S2LatLngRect& graph_bounds) {
  graph_bounds_ = graph_bounds;
  CalculateSceneMargin();
}

void GraphRegionMapper::ResetCurrentBounds(double zoom) {
  S2LatLng image_centre((graph_bounds_.lat_lo() + graph_bounds_.lat_hi()) / 2,
                        (graph_bounds_.lng_lo() + graph_bounds_.lng_hi()) / 2);
  SetCurrentBounds(zoom, image_centre);
}

void GraphRegionMapper::SetCurrentBounds(double zoom,
                                         const S2LatLng& image_centre) {
  S1Angle lat_min = graph_bounds_.lat_lo() - margin_.lat();
  S1Angle lat_max = graph_bounds_.lat_hi() + margin_.lat();
  S1Angle lng_min = graph_bounds_.lng_lo() - margin_.lng();
  S1Angle lng_max = graph_bounds_.lng_hi() + margin_.lng();

  CHECK_GT(zoom, 0);

  S1Angle lat_range = (lat_max - lat_min) / zoom;
  S1Angle lng_range = (lng_max - lng_min) / zoom;

  // Shift the bounds if any lie off-screen.
  S1Angle minLat = image_centre.lat() - lat_range / 2;
  S1Angle lat_diff;
  if (minLat < lat_min) {
    lat_diff = lat_min - minLat;
    minLat = lat_min;
  }
  S1Angle maxLat = image_centre.lat() + lat_range / 2 + lat_diff;
  if (maxLat > lat_max) {
    minLat -= maxLat - lat_max;
    maxLat = lat_max;
  }

  S1Angle minLng = image_centre.lng() - lng_range / 2;
  S1Angle lng_diff;
  if (minLng < lng_min) {
    lng_diff = lng_min - minLng;
    minLng = lng_min;
  }
  S1Angle maxLng = image_centre.lng() + lng_range / 2 + lng_diff;
  if (maxLng > lng_max) {
    minLng -= maxLng - lng_max;
    maxLng = lng_max;
  }

  current_bounds_ =
      S2LatLngRect(S2LatLng(minLat, minLng), S2LatLng(maxLat, maxLng));
}

Vector2_d GraphRegionMapper::MapToScreen(double lat, double lng) const {
  double lat_range =
      (current_bounds_.lat_hi() - current_bounds_.lat_lo()).degrees();
  double lng_range =
      (current_bounds_.lng_hi() - current_bounds_.lng_lo()).degrees();

  double x_coord =
      (lng - current_bounds_.lng_lo().degrees()) * screen_size_.x() / lng_range;
  double y_coord =
      (lat - current_bounds_.lat_lo().degrees()) * screen_size_.y() / lat_range;

  return {x_coord, screen_size_.y() - y_coord};
}

Vector2_d GraphRegionMapper::MapToBuffer(double lat, double lng,
                                         int image_width,
                                         int image_height) const {
  S1Angle lat_range =
      (graph_bounds_.lat_hi() - graph_bounds_.lat_lo() + 2.0 * margin_.lat());
  S1Angle lng_range =
      (graph_bounds_.lng_hi() - graph_bounds_.lng_lo() + 2.0 * margin_.lng());

  double x_coord = (lng - (graph_bounds_.lng_lo() - margin_.lng()).degrees()) *
                   image_width / lng_range.degrees();
  double y_coord = (lat - (graph_bounds_.lat_lo() - margin_.lat()).degrees()) *
                   image_height / lat_range.degrees();

  return {x_coord, image_height - y_coord};
}

void GraphRegionMapper::CalculateSceneMargin() {
  double lat_range =
      (graph_bounds_.lat_hi() - graph_bounds_.lat_lo()).degrees();
  double lng_range =
      (graph_bounds_.lng_hi() - graph_bounds_.lng_lo()).degrees();
  double aspect_ratio =
      static_cast<double>(screen_size_.x()) / screen_size_.y();

  if (lat_range > lng_range) {
    lat_range *= aspect_ratio;
    double lat_margin = lat_range * kMargin;
    double deg_ppx = (lat_range + 2.0 * lat_margin) / screen_size_.y();
    double total_lng = screen_size_.x() * deg_ppx;
    double lng_margin = (total_lng - lng_range) / 2;
    margin_ = S2LatLng::FromDegrees(lat_margin, lng_margin);
  } else {
    lng_range *= aspect_ratio;
    double lng_margin = lng_range * kMargin;
    double deg_ppx = (lng_range + 2.0 * lng_margin) / screen_size_.x();
    double total_lat = screen_size_.y() * deg_ppx;
    double lat_margin = (total_lat - lat_range) / 2;
    margin_ = S2LatLng::FromDegrees(lat_margin, lng_margin);
  }
}

}  // namespace streetlearn
