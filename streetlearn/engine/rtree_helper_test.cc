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

#include "streetlearn/engine/rtree_helper.h"

#include "s2/s1angle.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace streetlearn {
namespace geometry {
namespace {

using ::testing::ElementsAre;

constexpr int kNumBounds = 1000;

// Create d by d S2LatLngRect centered at (c, c).
S2LatLngRect GetRect(double c, double d) {
  auto r = d / 2.0;
  auto lo = S2LatLng::FromDegrees(c - r, c - r);
  auto hi = S2LatLng::FromDegrees(c + r, c + r);
  return S2LatLngRect(lo, hi);
}

TEST(RTreeHelperTest, EmptyTree) {
  S2LatLngRect rect;
  rect.mutable_lng()->set_hi(M_PI);
  rect.mutable_lng()->set_lo(0);

  RTree rtree;
  EXPECT_TRUE(rtree.empty());

  std::vector<int> out;
  EXPECT_EQ(rtree.FindIntersecting(S2LatLngRect::Full(), &out), 0);
  EXPECT_TRUE(out.empty());
}

TEST(RTreeHelperTest, Insert_Basic) {
  RTree rtree;
  S2LatLngRect bound = GetRect(10, 5);
  rtree.Insert(bound, 1);

  std::vector<int> out;
  EXPECT_EQ(rtree.FindIntersecting(bound, &out), 1);
  EXPECT_THAT(out, ElementsAre(1));
}

TEST(RTreeHelperTest, Insert_NonOverlapping) {
  RTree rtree;
  std::vector<S2LatLngRect> bounds;

  constexpr double start = 0.0;
  constexpr double step = 0.01;

  for (int i = 0; i < kNumBounds; ++i) {
    const double lat = start + step * i;
    bounds.push_back(S2LatLngRect(S2LatLng::FromDegrees(lat, 0),
                                  S2LatLng::FromDegrees(lat + (step / 2), 10)));
    std::cout << lat << ";" << 0 << " - " << lat + step / 2 << ";" << 10
              << std::endl;
    rtree.Insert(bounds[i], i);
  }

  for (int i = 0; i < bounds.size(); ++i) {
    std::vector<int> out;
    EXPECT_EQ(rtree.FindIntersecting(bounds[i], &out), 1);
    EXPECT_THAT(out, ElementsAre(i));
  }
}

TEST(RTreeHelperTest, Insert_Overlapping) {
  RTree rtree;
  constexpr double kCenter = 50.0;
  constexpr double kMin = 10.0;
  constexpr double kMax = 20.0;

  for (int i = 0; i < kNumBounds; ++i) {
    S2LatLngRect rect =
        GetRect(kCenter, kMax - (i * (kMax - kMin) / (kNumBounds - 1)));
    rtree.Insert(rect, i);
  }

  std::vector<int> out;
  EXPECT_EQ(rtree.FindIntersecting(GetRect(kCenter, kMin), &out), kNumBounds);
  EXPECT_EQ(kNumBounds, out.size());
}

}  // namespace
}  // namespace geometry
}  // namespace streetlearn
