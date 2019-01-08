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

#include "streetlearn/engine/test_utils.h"

#include <cstdlib>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "streetlearn/engine/pano_graph_node.h"

#ifndef STREETLEARN_SUPPRESS_COMMANDLINE_FLAGS

#include "absl/flags/flag.h"
DECLARE_string(test_srcdir);
#endif

namespace streetlearn {
namespace test_utils {

std::string TestSrcDir() {
#ifndef STREETLEARN_SUPPRESS_COMMANDLINE_FLAGS
  return absl::GetFlag(FLAGS_test_srcdir) + "/streetlearn";
#else
  const char* const test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir) {
    std::string path = absl::StrCat(test_srcdir, "/");
    const char* const test_workspace = std::getenv("TEST_WORKSPACE");

    if (test_workspace) {
      return absl::StrCat(path, test_workspace, "/streetlearn/");
    }

    return path;
  }
  return "[undefined TEST_SRCDIR environment variable]";
#endif
}

::testing::AssertionResult CompareImages(
    const ImageView4_b& image, absl::string_view expected_image_path) {
  std::vector<uint8_t> buf(image.width() * image.height() * 3);
  for (int y = 0; y < image.height(); ++y) {
    int row_offset = y * image.width();
    for (int x = 0; x < image.width(); ++x) {
      int out_offset = 3 * (row_offset + x);
      const uint8_t* pixel = image.pixel(x, y);
      buf[out_offset] = pixel[2];
      buf[out_offset + 1] = pixel[1];
      buf[out_offset + 2] = pixel[0];
    }
  }

  return CompareRGBBufferWithImage(absl::MakeSpan(buf), image.width(),
                                   image.height(), kPackedRGB,
                                   expected_image_path);
}

::testing::AssertionResult CompareRGBBufferWithImage(
    absl::Span<const uint8_t> buffer, int width, int height, PixelFormat format,
    absl::string_view expected_image_path) {
  cv::Mat original_image =
      cv::imread(test_utils::TestSrcDir() + std::string(expected_image_path));

  // Treat empty images as identical.
  if (original_image.empty() && width == 0 && height == 0) {
    return ::testing::AssertionSuccess();
  }

  if (width != original_image.cols || height != original_image.rows) {
    return ::testing::AssertionFailure()
           << "Image size does not match: (" << width << "," << height << ","
           << ") != (" << original_image.cols << "," << original_image.rows
           << ")";
  }

  // Construct a Mat from the buffer.
  cv::Mat image;
  uint8_t* data = const_cast<uint8_t*>(buffer.data());

  if (format == kPackedRGB) {
    image = cv::Mat(height, width, CV_8UC3, data);
  } else {
    cv::Mat channel_r(height, width, CV_8UC1, data);
    cv::Mat channel_g(height, width, CV_8UC1, data + width * height);
    cv::Mat channel_b(height, width, CV_8UC1, data + 2 * width * height);
    cv::merge(std::vector<cv::Mat>{channel_r, channel_g, channel_b}, image);
  }

  cv::Mat image_gray;
  cvtColor(image, image_gray, CV_RGB2GRAY);
  cv::Mat original_image_gray;
  cvtColor(original_image, original_image_gray, CV_BGR2GRAY);

  cv::Mat diff;
  cv::compare(image_gray, original_image_gray, diff, cv::CMP_NE);

  int diff_count = cv::countNonZero(diff);
  if (diff_count) {
    std::string diff_image_path = absl::StrCat(
        ::testing::TempDir(), "diff_", absl::ToUnixNanos(absl::Now()), ".png");
    cv::imwrite(diff_image_path, diff);

    return ::testing::AssertionFailure()
           << "Pixels do not match. Number of differences: " << diff_count
           << ". Binary diff saved to: " << diff_image_path;
  }

  return ::testing::AssertionSuccess();
}

}  // namespace test_utils
}  // namespace streetlearn
