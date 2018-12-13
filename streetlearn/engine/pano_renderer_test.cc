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

#include "streetlearn/engine/pano_renderer.h"

#include <string>

#include "gtest/gtest.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/pano_renderer.h"
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

constexpr int kTolerance = 2;
constexpr int kFovDegrees = 30;
constexpr int kStatusHeight = 10;
// Color components for bearings.
constexpr int kCenterBearing = 128;
constexpr int kSideBearing = 230;

TEST(StreetLearn, PanoRendererTest) {
  PanoRenderer pano_renderer(TestDataset::kImageWidth,
                             TestDataset::kImageHeight, kStatusHeight,
                             kFovDegrees);

  const Image3_b test_image = TestDataset::GenerateTestImage();
  pano_renderer.RenderScene(test_image, 0, 0, 0, 0, {}, {});
  auto pixels = pano_renderer.Pixels();

  for (int y = 0; y < TestDataset::kImageHeight / 2; ++y) {
    for (int x = 0; x < TestDataset::kImageWidth; ++x) {
      int index = 3 * (y * TestDataset::kImageWidth + x);
      const auto* expected_pixel = test_image.pixel(x, y);
      EXPECT_LE(std::abs(pixels[index] - expected_pixel[0]), kTolerance);
      EXPECT_LE(std::abs(pixels[index + 1] - expected_pixel[1]), kTolerance);
      EXPECT_LE(std::abs(pixels[index + 2] - expected_pixel[2]), kTolerance);
    }
  }
}

TEST(StreetLearn, PanoRendererBearingsTest) {
  PanoRenderer pano_renderer(TestDataset::kImageWidth,
                             TestDataset::kImageHeight, kStatusHeight,
                             kFovDegrees);

  Image3_b pano_image(TestDataset::kImageWidth, TestDataset::kImageHeight);

  std::vector<PanoNeighborBearing> bearings = {
      {"0", -90, 0.0}, {"1", 0, 1.0}, {"2", 90, 2.0}};

  // Look at the center of the view.
  pano_renderer.RenderScene(pano_image, 0, 0, 0, 0, bearings, {});

  auto pixels = pano_renderer.Pixels();
  for (int j = 0; j < TestDataset::kImageWidth; ++j) {
    int index = (TestDataset::kImageHeight - 5) * TestDataset::kImageWidth + j;
    if (TestDataset::kImageWidth / 2 == j) {
      EXPECT_LE(abs(pixels[3 * index] - kCenterBearing), kTolerance);
    } else if (TestDataset::kImageWidth / 4 == j ||
               3 * TestDataset::kImageWidth / 4 == j) {
      EXPECT_LE(abs(pixels[3 * index] - kSideBearing), kTolerance);
    }
  }

  // Rotate by 90 degrees.
  pano_renderer.RenderScene(pano_image, 0, 90, 0, 0, bearings, {});
  pixels = pano_renderer.Pixels();
  for (int j = 0; j < TestDataset::kImageWidth; ++j) {
    int index = (TestDataset::kImageHeight - 5) * TestDataset::kImageWidth + j;
    if (TestDataset::kImageWidth / 2 == j) {
      EXPECT_LE(abs(pixels[3 * index] - kCenterBearing), kTolerance);
    } else if (0 == j || TestDataset::kImageWidth / 4 == j) {
      EXPECT_LE(abs(pixels[3 * index] - kSideBearing), kTolerance);
    }
  }
}

TEST(StreetLearn, PanoRendererConstrainAngleTest) {
  double angle = PanoRenderer::ConstrainAngle(180, 400);
  EXPECT_EQ(angle, 40);

  angle = PanoRenderer::ConstrainAngle(360, 400);
  EXPECT_EQ(angle, -320);

  angle = PanoRenderer::ConstrainAngle(180, -400);
  EXPECT_EQ(angle, -40);

  angle = PanoRenderer::ConstrainAngle(360, -420);
  EXPECT_EQ(angle, 300);

  angle = PanoRenderer::ConstrainAngle(180, -180);
  EXPECT_EQ(angle, -180);

  angle = PanoRenderer::ConstrainAngle(180, 180);
  EXPECT_EQ(angle, -180);
}

}  // namespace
}  // namespace streetlearn
