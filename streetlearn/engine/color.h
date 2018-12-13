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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_COLOR_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_COLOR_H_

namespace streetlearn {

// Struct to allow colors to be exposed to Python via CLIF in a natural way.
struct Color {
  float red;
  float green;
  float blue;

  constexpr Color(float r, float g, float b) : red(r), green(g), blue(b) {}
  Color() : Color(0, 0, 0) {}
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_COLOR_H_
