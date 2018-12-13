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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_PANO_CALCULATIONS_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_PANO_CALCULATIONS_H_

#include <cmath>
#include <random>

#include "streetlearn/engine/pano_graph_node.h"

namespace streetlearn {

constexpr double kEarthRadiusInMetres = 6371000.0;

// Converts the given angle from degrees to radians.
inline double DegreesToRadians(double angle_in_degrees) {
  return angle_in_degrees * M_PI / 180;
}

// Returns the bearing between the panos in degrees.
inline double BearingBetweenPanos(const streetlearn::Pano& pano1,
                                  const streetlearn::Pano& pano2) {
  double lat1 = DegreesToRadians(pano1.coords().lat());
  double lon1 = DegreesToRadians(pano1.coords().lng());
  double lat2 = DegreesToRadians(pano2.coords().lat());
  double lon2 = DegreesToRadians(pano2.coords().lng());
  double x = std::cos(lat1) * std::sin(lat2) -
             std::sin(lat1) * std::cos(lat2) * std::cos(lon2 - lon1);
  double y = std::sin(lon2 - lon1) * std::cos(lat2);
  return 180.0 * std::atan2(y, x) / M_PI;
}

// Uses the Haversine formula to compute the distance between the panos in
// meters.
inline double DistanceBetweenPanos(const streetlearn::Pano& pano1,
                                   const streetlearn::Pano& pano2) {
  double lat1 = DegreesToRadians(pano1.coords().lat());
  double lon1 = DegreesToRadians(pano1.coords().lng());
  double lat2 = DegreesToRadians(pano2.coords().lat());
  double lon2 = DegreesToRadians(pano2.coords().lng());
  double u = std::sin((lat2 - lat1) / 2);
  double v = std::sin((lon2 - lon1) / 2);
  return 2.0 * kEarthRadiusInMetres *
         std::asin(std::sqrt(u * u + std::cos(lat1) * std::cos(lat2) * v * v));
}

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_PANO_CALCULATIONS_H_
