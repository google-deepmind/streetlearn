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

#include "streetlearn/engine/bitmap_util.h"

namespace streetlearn {

void ConvertRGBPackedToPlanar(const uint8_t* rgb_buffer, int width, int height,
                              uint8_t* out_buffer) {
  int g_offset = width * height;
  int b_offset = 2 * g_offset;
  for (int y = 0; y < height; ++y) {
    int row_offset = y * width;
    for (int x = 0; x < width; ++x) {
      int index = row_offset + x;
      out_buffer[index] = rgb_buffer[3 * index];
      out_buffer[index + g_offset] = rgb_buffer[3 * index + 1];
      out_buffer[index + b_offset] = rgb_buffer[3 * index + 2];
    }
  }
}

}  // namespace streetlearn
