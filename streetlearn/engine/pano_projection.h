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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_PANO_PROJECTION_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_PANO_PROJECTION_H_

#include <memory>

#include "streetlearn/engine/image.h"

namespace streetlearn {

// Each Street View panorama is an image that provides a full 360-degree view
// from a single location. Images conform to the equirectangular projection,
// which contains 360 degrees of horizontal view (a full wrap-around) and
// 180 degrees of vertical view (from straight up to straight down). The
// resulting 360-degree panorama defines a projection on a sphere with the image
// wrapped to the two-dimensional surface of that sphere.
// Panorama images are projected onto a narrow field of view image at a given
// pitch and yaw w.r.t. the panorama heading.
class PanoProjection {
 public:
  PanoProjection(double fov_deg, int proj_width, int proj_height);
  ~PanoProjection();

  // Projects a panorama image input into an output image at specified yaw and
  // pitch angles.
  void Project(const Image3_b& input, double yaw_deg, double pitch_deg,
               Image3_b* output);

  // Changes the field of view.
  void ChangeFOV(double fov_deg);

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_PANO_PROJECTION_H_
