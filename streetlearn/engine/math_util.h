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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_MATHUTIL_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_MATHUTIL_H_

#include <cmath>
#include <random>

#include "streetlearn/engine/logging.h"
#include "absl/base/attributes.h"

namespace streetlearn {
namespace math {

// Returns a random integer in the range [0, max].
inline int UniformRandomInt(std::mt19937* random, int max) {
  std::uniform_int_distribution<int> uniform_dist(0, max);
  return uniform_dist(*random);
}

// Clamps value to the range [low, high].  Requires low <= high.
template <typename T>
ABSL_MUST_USE_RESULT inline const T Clamp(const T& low, const T& high,
                                          const T& value) {
  // Detects errors in ordering the arguments.
  CHECK(!(high < low));
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

// Converts the given angle from degrees to radians.
inline double DegreesToRadians(double degrees) {
  return degrees * M_PI / 180.0;
}

// Converts the given angle from radians to degrees.
inline double RadiansToDegrees(double radians) {
  return radians * 180.0 / M_PI;
}

}  // namespace math
}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_MATHUTIL_H_
