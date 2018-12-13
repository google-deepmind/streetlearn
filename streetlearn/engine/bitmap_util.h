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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_IMAGE_UTIL_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_IMAGE_UTIL_H_

#include <cstdint>

namespace streetlearn {

// Converts the input packed RGB pixels to planar. The buffers must both be
// at least 3 * width * height in size.
void ConvertRGBPackedToPlanar(const uint8_t* rgb_buffer, int width, int height,
                              uint8_t* out_buffer);

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_IMAGE_UTIL_H_
