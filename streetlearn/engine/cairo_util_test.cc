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

#include "streetlearn/engine/cairo_util.h"

#include <vector>

#include "gtest/gtest.h"
#include <cairo/cairo.h>

namespace streetlearn {
namespace {

TEST(CairoUtilTest, TestEmpty) {
  std::vector<unsigned char> buffer;
  CairoRenderHelper helper(buffer.data(), 0, 0);

  EXPECT_EQ(helper.width(), 0);
  EXPECT_EQ(helper.height(), 0);

  EXPECT_EQ(cairo_status(helper.context()), CAIRO_STATUS_SUCCESS);
  EXPECT_EQ(cairo_image_surface_get_data(helper.surface()), nullptr);
  EXPECT_EQ(cairo_image_surface_get_width(helper.surface()), 0);
  EXPECT_EQ(cairo_image_surface_get_height(helper.surface()), 0);
  EXPECT_EQ(cairo_image_surface_get_stride(helper.surface()), 0);
}

TEST(CairoUtilTest, TestNonEmpty) {
  constexpr int kWidth = 32;
  constexpr int kHeight = 64;
  constexpr int kChannels = 1;

  std::vector<unsigned char> buffer(kWidth * kHeight * kChannels);
  CairoRenderHelper helper(buffer.data(), kWidth, kHeight, CAIRO_FORMAT_A8);

  EXPECT_EQ(helper.width(), kWidth);
  EXPECT_EQ(helper.height(), kHeight);

  EXPECT_EQ(cairo_status(helper.context()), CAIRO_STATUS_SUCCESS);
  EXPECT_EQ(cairo_image_surface_get_data(helper.surface()), buffer.data());
  EXPECT_EQ(cairo_image_surface_get_width(helper.surface()), kWidth);
  EXPECT_EQ(cairo_image_surface_get_height(helper.surface()), kHeight);
  EXPECT_EQ(cairo_image_surface_get_stride(helper.surface()),
            kWidth * kChannels);
}

}  // namespace
}  // namespace streetlearn
