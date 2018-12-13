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

#include "streetlearn/engine/pano_calculations.h"

#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace streetlearn {
namespace {

constexpr double kLatitude1 = 51.5335010;
constexpr double kLatitude2 = 51.5380236;
constexpr double kLongitude1 = -0.1256744;
constexpr double kLongitude2 = -0.1268474;
constexpr double kTolerance = 0.001;

class PanoCalculationsTest : public ::testing::Test {
 public:
  PanoCalculationsTest() {
    LatLng* coords1 = pano1_.mutable_coords();
    coords1->set_lat(kLatitude1);
    coords1->set_lng(kLongitude1);

    LatLng* coords2 = pano2_.mutable_coords();
    coords2->set_lat(kLatitude2);
    coords2->set_lng(kLongitude2);
  }

  Pano pano1_;
  Pano pano2_;
};

TEST(StreetLearn, AngleConversionsTest) {
  EXPECT_EQ(DegreesToRadians(0), 0);
  EXPECT_EQ(DegreesToRadians(90), M_PI / 2);
  EXPECT_EQ(DegreesToRadians(-90), -M_PI / 2);
  EXPECT_EQ(DegreesToRadians(180), M_PI);
}

TEST_F(PanoCalculationsTest, BearingBetweenPanosTest) {
  EXPECT_THAT(BearingBetweenPanos(pano1_, pano2_),
              testing::DoubleNear(-9.16417, kTolerance));
}

TEST_F(PanoCalculationsTest, DistanceBetweenPanos) {
  EXPECT_THAT(DistanceBetweenPanos(pano1_, pano2_),
              testing::DoubleNear(509.393, kTolerance));
}

}  // namespace
}  // namespace streetlearn
