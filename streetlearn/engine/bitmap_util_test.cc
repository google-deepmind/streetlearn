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

#include <array>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace streetlearn {
namespace {

using ::testing::ElementsAreArray;

constexpr int kWidth = 2;
constexpr int kHeight = 2;

const std::array<uint8_t, kWidth * kHeight * 3> kBitmapRGB{{
    11, 12, 13,  // (0,0)
    21, 22, 23,  // (1,0)
    31, 32, 33,  // (0,1)
    41, 42, 255  // (1,1)
}};

const std::array<uint8_t, kWidth * kHeight * 3> kBitmapPlanarRGB{{
    11, 21, 31, 41,   // R
    12, 22, 32, 42,   // G
    13, 23, 33, 255,  // B
}};

TEST(BitmapUtilTest, TestRGBToPlanar) {
  std::vector<uint8_t> out_buffer(kBitmapPlanarRGB.size());
  ConvertRGBPackedToPlanar(kBitmapRGB.data(), kWidth, kHeight,
                           out_buffer.data());
  EXPECT_THAT(out_buffer, ElementsAreArray(kBitmapPlanarRGB));
}

}  // namespace
}  // namespace streetlearn
