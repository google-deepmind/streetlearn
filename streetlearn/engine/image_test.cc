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

#include "streetlearn/engine/image.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace streetlearn {
namespace {

using ::testing::Eq;

template <typename T>
class ImageTest : public ::testing::Test {};
TYPED_TEST_CASE_P(ImageTest);

TYPED_TEST_P(ImageTest, EmptyImageTest) {
  constexpr int kChannels = 3;
  Image<TypeParam, kChannels> image;

  EXPECT_EQ(image.width(), 0);
  EXPECT_EQ(image.height(), 0);
  EXPECT_EQ(image.channels(), kChannels);
  EXPECT_TRUE(image.data().empty());
  EXPECT_EQ(image.pixel(0, 0), nullptr);
}

TYPED_TEST_P(ImageTest, EmptyImageViewTest) {
  constexpr int kChannels = 3;
  Image<TypeParam, kChannels> image;
  ImageView<TypeParam, kChannels> image_view(&image, 0, 0, image.width(),
                                             image.height());

  EXPECT_EQ(image_view.width(), 0);
  EXPECT_EQ(image_view.height(), 0);
  EXPECT_EQ(image_view.channels(), kChannels);
  EXPECT_EQ(image_view.pixel(0, 0), nullptr);
}

TYPED_TEST_P(ImageTest, CreationTest) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 7;
  constexpr int kChannels = 4;
  Image<TypeParam, kChannels> image(kWidth, kHeight);
  EXPECT_EQ(image.width(), kWidth);
  EXPECT_EQ(image.height(), kHeight);
  EXPECT_EQ(image.channels(), kChannels);
  EXPECT_EQ(image.data().size(), kWidth * kHeight * kChannels);

  ImageView<TypeParam, kChannels> image_view(&image, 1, 1, kWidth - 1,
                                             kHeight - 1);
  EXPECT_EQ(image_view.width(), kWidth - 1);
  EXPECT_EQ(image_view.height(), kHeight - 1);
  EXPECT_EQ(image_view.channels(), kChannels);
}

TYPED_TEST_P(ImageTest, ImageAccessTest) {
  constexpr int kSize = 5;
  Image<TypeParam, 1> image(kSize, kSize);

  for (int i = 0; i < kSize; ++i) {
    EXPECT_THAT(*image.pixel(i, i), Eq(0));
    const TypeParam pixel = i;
    *image.pixel(i, i) = pixel;
    EXPECT_THAT(*image.pixel(i, i), Eq(pixel));

    for (int j = 0; j < kSize; ++j) {
      if (j <= i) {
        EXPECT_THAT(*image.pixel(j, j), Eq(static_cast<TypeParam>(j)));
      } else {
        EXPECT_THAT(*image.pixel(j, j), Eq(static_cast<TypeParam>(0)));
      }
    }
  }
}

TYPED_TEST_P(ImageTest, ImageViewAccessTest) {
  constexpr int kSize = 5;
  Image<TypeParam, 1> image(kSize, kSize);
  ImageView<TypeParam, 1> image_view(&image, 1, 1, kSize - 2, kSize - 2);

  for (int i = 0; i < kSize - 2; ++i) {
    *image_view.pixel(i, i) = static_cast<TypeParam>(i);
  }

  EXPECT_THAT(*image.pixel(0, 0), Eq(0));
  EXPECT_THAT(*image.pixel(kSize - 1, kSize - 1), Eq(0));

  for (int j = 1; j < kSize - 2; ++j) {
    EXPECT_THAT(*image.pixel(j, j), Eq(static_cast<TypeParam>(j - 1)));
  }
}

REGISTER_TYPED_TEST_CASE_P(ImageTest, EmptyImageTest, EmptyImageViewTest,
                           CreationTest, ImageAccessTest, ImageViewAccessTest);

using Types = ::testing::Types<unsigned char, float, double>;
INSTANTIATE_TYPED_TEST_CASE_P(TypedImageTests, ImageTest, Types);

}  // namespace
}  // namespace streetlearn
