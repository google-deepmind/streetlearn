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

#include "streetlearn/engine/pano_projection.h"

#include <cmath>

#include "gtest/gtest.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

// Panorama and projection image specs.
constexpr int kPanoramaLargeWidth = 6656;
constexpr int kPanoramaLargeHeight = 3328;
constexpr int kPanoramaSmallWidth = 416;
constexpr int kPanoramaSmallHeight = 208;
constexpr int kProjectionLargeWidth = 640;
constexpr int kProjectionLargeHeight = 480;
constexpr int kProjectionSmallWidth = 84;
constexpr int kProjectionSmallHeight = 84;
constexpr int kXCenterLarge = kProjectionLargeWidth / 2;
constexpr int kXCenterSmall = kProjectionSmallWidth / 2;
constexpr int kXRightLarge = kProjectionLargeWidth - 1;
constexpr int kXRightSmall = kProjectionSmallWidth - 1;
constexpr int kYCenterLarge = kProjectionLargeHeight / 2;
constexpr int kYCenterSmall = kProjectionSmallHeight / 2;
constexpr int kYBottomLarge = kProjectionLargeHeight - 1;
constexpr int kYBottomSmall = kProjectionSmallHeight - 1;
constexpr double kHorizontalFieldOfView = 60.0;
constexpr double kLongitudeStep = 30.0;
constexpr double kLongitudeTolerance = 5.0;
constexpr double kLatitudeStep = 30.0;
constexpr double kLatitudeTolerance = 5.0;
constexpr int kImageWidth = 64;
constexpr int kImageHeight = 48;
constexpr int kImageDepth = 3;

// Generate a panorama image.
Image3_b GeneratePanorama(int width, int height) {
  // Create grayscale image of the same size as the panorama.
  Image3_b panorama(width, height);
  // Draw white rectangles on the black panoramic image at regular intervals.
  for (int y = 0; y < height; y++) {
    double latitude = y * 180.0 / height;
    latitude = std::fmod(latitude, kLatitudeStep);
    if ((latitude < kLatitudeTolerance) ||
        ((kLatitudeStep - latitude) < kLatitudeTolerance)) {
      for (int x = 0; x < width; x++) {
        double longitude = x * 360.0 / width;
        longitude = std::fmod(longitude, kLongitudeStep);
        if ((longitude < kLongitudeTolerance) ||
            ((kLongitudeStep - longitude) < kLongitudeTolerance)) {
          auto pixel = panorama.pixel(x, y);
          pixel[0] = 255;
          pixel[1] = 255;
          pixel[2] = 255;
        }
      }
    }
  }

  return panorama;
}

// Return grayscale color of pixel.
double GrayScaleColorAt(const Image3_b& image, int x, int y) {
  EXPECT_GE(x, 0);
  EXPECT_GE(y, 0);
  EXPECT_LT(x, image.width());
  EXPECT_LT(y, image.height());
  auto pixel = image.pixel(x, y);
  return (pixel[0] + pixel[1] + pixel[2]) / 3;
}

// Helper function for testing values at 9 points on the projected image.
void TestProjection(const Image3_b& image, int topLeft, int topMiddle,
                    int topRight, int centerLeft, int center, int centerRight,
                    int bottomLeft, int bottom, int bottomRight) {
  int width = image.width();
  int height = image.height();
  int halfWidth = width / 2;
  int halfHeight = height / 2;
  EXPECT_EQ(GrayScaleColorAt(image, 0, 0), topLeft);
  EXPECT_EQ(GrayScaleColorAt(image, 0, halfHeight), topMiddle);
  EXPECT_EQ(GrayScaleColorAt(image, 0, height - 1), topRight);
  EXPECT_EQ(GrayScaleColorAt(image, halfWidth, 0), centerLeft);
  EXPECT_EQ(GrayScaleColorAt(image, halfWidth, halfHeight), center);
  EXPECT_EQ(GrayScaleColorAt(image, halfWidth, height - 1), centerRight);
  EXPECT_EQ(GrayScaleColorAt(image, width - 1, 0), bottomLeft);
  EXPECT_EQ(GrayScaleColorAt(image, width - 1, halfHeight), bottom);
  EXPECT_EQ(GrayScaleColorAt(image, width - 1, height - 1), bottomRight);
}

TEST(StreetLearn, PanoProjectionOpenCVLargeTest) {
  Image3_b panorama =
      GeneratePanorama(kPanoramaLargeWidth, kPanoramaLargeHeight);
  Image3_b image(kProjectionLargeWidth, kProjectionLargeHeight);

  // Creates the pano projection object.
  PanoProjection projector(kHorizontalFieldOfView, kProjectionLargeWidth,
                           kProjectionLargeHeight);
  // Projects the panorama at zero pitch and yaw.
  projector.Project(panorama, 0.0, 0.0, &image);
  TestProjection(image, 0, 255, 0, 0, 255, 0, 0, 255, 0);
  // Projects the panorama at zero pitch and 30 yaw.
  projector.Project(panorama, kLongitudeStep, 0.0, &image);
  TestProjection(image, 0, 255, 0, 0, 255, 0, 0, 255, 0);
  // Projects the panorama at zero pitch and -30 yaw.
  projector.Project(panorama, kLongitudeStep, 0.0, &image);
  TestProjection(image, 0, 255, 0, 0, 255, 0, 0, 255, 0);
  // Projects the panorama at zero pitch and 15 yaw.
  projector.Project(panorama, kLongitudeStep / 2, 0.0, &image);
  TestProjection(image, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // Projects the panorama at 15 pitch and zero yaw.
  projector.Project(panorama, 0.0, kLatitudeStep / 2, &image);
  TestProjection(image, 255, 0, 0, 0, 0, 0, 255, 0, 0);
  // Projects the panorama at 30 pitch and zero yaw.
  projector.Project(panorama, 0.0, kLatitudeStep, &image);
  TestProjection(image, 0, 255, 0, 0, 255, 0, 0, 255, 0);
  // Projects the panorama at 60 pitch and zero yaw.
  projector.Project(panorama, 0.0, kLatitudeStep * 2, &image);
  TestProjection(image, 0, 0, 255, 0, 255, 0, 0, 0, 255);
}

TEST(StreetLearn, PanoProjectionOpenCVSmallTest) {
  Image3_b panorama =
      GeneratePanorama(kPanoramaSmallWidth, kPanoramaSmallHeight);
  Image3_b image(kProjectionSmallWidth, kProjectionSmallHeight);

  // Creates the pano projection object.
  PanoProjection projector(kHorizontalFieldOfView, kProjectionSmallWidth,
                           kProjectionSmallHeight);
  // Projects the panorama at zero pitch and yaw.
  projector.Project(panorama, 0.0, 0.0, &image);
  TestProjection(image, 255, 255, 255, 255, 255, 255, 255, 255, 255);
  // Projects the panorama at zero pitch and 15 yaw.
  projector.Project(panorama, kLongitudeStep / 2, 0.0, &image);
  TestProjection(image, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // Projects the panorama at 15 pitch and zero yaw.
  projector.Project(panorama, 0.0, kLatitudeStep / 2, &image);
  TestProjection(image, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // Projects the panorama at 30 pitch and zero yaw.
  projector.Project(panorama, 0.0, kLatitudeStep, &image);
  TestProjection(image, 0, 255, 255, 255, 255, 255, 0, 255, 255);
  // Projects the panorama at 60 pitch and zero yaw.
  projector.Project(panorama, 0.0, kLatitudeStep * 2, &image);
  TestProjection(image, 255, 0, 255, 255, 255, 255, 255, 0, 255);
}

TEST(StreetLearn, PanoProjectionPanoImageTest) {
  // Creates the panorama
  const auto test_image = TestDataset::GenerateTestImage();
  // Creates the pano projection object.
  PanoProjection projector(kHorizontalFieldOfView, kImageWidth, kImageHeight);
  // Projects the panorama at zero pitch and yaw.
  Image3_b proj_image(kImageWidth, kImageHeight);
  projector.Project(test_image, 0.0, 0.0, &proj_image);
  TestProjection(proj_image, 37, 37, 37, 37, 37, 37, 37, 37, 37);
}

}  // namespace
}  // namespace streetlearn
