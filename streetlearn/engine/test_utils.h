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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_TESTING_TEST_UTILS_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_TESTING_TEST_UTILS_H_

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "streetlearn/engine/image.h"

namespace streetlearn {

namespace test_utils {

// Returns the name of the directory in which the test binary is located.
std::string TestSrcDir();

// Custom assertion to compare to compare two images pixel by pixel. The
// expected image is loaded from `expected_image_path` which is a sub path
// inside TEST_SRCDIR.
::testing::AssertionResult CompareImages(const ImageView4_b& image,
                                         absl::string_view expected_image_path);

// Supported pixel formats for comparison.
enum PixelFormat { kPackedRGB, kPlanarRGB };

// Custom assertion to compare to compare two images pixel by pixel. The source
// image is provided as a continous buffer where each pixel is represented
// as as 24 bit R, G, B value. PixelFormat determines whether channels are
// are stored in packed or planar format. The expected image is loaded from
// `expected_image_path` which is a sub path inside TEST_SRCDIR.
::testing::AssertionResult CompareRGBBufferWithImage(
    absl::Span<const uint8_t> buffer, int width, int height, PixelFormat format,
    absl::string_view expected_image_path);

}  // namespace test_utils
}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_TESTING_TEST_UTILS_H_
