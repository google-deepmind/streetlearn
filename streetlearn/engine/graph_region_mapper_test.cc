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

#include "streetlearn/engine/graph_region_mapper.h"

#include <cmath>

#include "gtest/gtest.h"
#include "streetlearn/engine/test_dataset.h"
#include "streetlearn/engine/vector.h"
#include "s2/s1angle.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"

namespace streetlearn {
namespace {

constexpr int kScreenWidth = 640;
constexpr int kScreenHeight = 480;
constexpr int kMarginX = 38;
constexpr int kMarginY = 22;
constexpr double kMinLatitude = TestDataset::kMinLatitude;
constexpr double kMinLongitude = TestDataset::kMinLongitude;
constexpr double kMaxLatitude = TestDataset::kMaxLatitude;
constexpr double kMaxLongitude = TestDataset::kMaxLongitude;
constexpr double kLatCentre = (kMaxLatitude + kMinLatitude) / 2;
constexpr double kLngCentre = (kMaxLongitude + kMinLongitude) / 2;
constexpr double kLatRange = kMaxLatitude - kMinLatitude;
constexpr double kLngRange = kMaxLongitude - kMinLongitude;

S2LatLng image_centre() {
  return S2LatLng::FromDegrees(kLatCentre, kLngCentre);
}

class RegionMapperTest : public testing::Test {
 public:
  RegionMapperTest() : region_mapper_({kScreenWidth, kScreenHeight}) {
    S2LatLng bounds_min(S1Angle::Degrees(kMinLatitude),
                        S1Angle::Degrees(kMinLongitude));
    S2LatLng bounds_max(S1Angle::Degrees(kMaxLatitude),
                        S1Angle::Degrees(kMaxLongitude));
    S2LatLngRect graph_bounds(bounds_min, bounds_max);
    region_mapper_.SetGraphBounds(graph_bounds);
  }

  void SetCurrentBounds(double zoom, const S2LatLng& image_centre) {
    region_mapper_.SetCurrentBounds(zoom, image_centre);
  }

  Vector2_d MapToScreen(double lat, double lng) const {
    return region_mapper_.MapToScreen(lat, lng);
  }

  Vector2_d MapToBuffer(double lat, double lng, int width, int height) const {
    return region_mapper_.MapToBuffer(lat, lng, width, height);
  }

  void ResetCurrentBounds(int zoom) { region_mapper_.ResetCurrentBounds(zoom); }

 private:
  streetlearn::GraphRegionMapper region_mapper_;
};

TEST_F(RegionMapperTest, MapToScreenTest) {
  // Test at full zoom.
  S2LatLng centre = image_centre();
  SetCurrentBounds(1.0 /* zoom_factor */, centre);

  Vector2_d mid = MapToScreen(centre.lat().degrees(), centre.lng().degrees());
  EXPECT_EQ(kScreenWidth / 2, std::lround(mid.x()));
  EXPECT_EQ(kScreenHeight / 2, std::lround(mid.y()));

  Vector2_d min = MapToScreen(kMinLatitude, kMinLongitude);
  EXPECT_EQ(kMarginX, std::lround(min.x()));
  EXPECT_EQ(kScreenHeight - kMarginY, std::lround(min.y()));

  Vector2_d max = MapToScreen(kMaxLatitude, kMaxLongitude);
  EXPECT_EQ(kScreenWidth - kMarginX, std::lround(max.x()));
  EXPECT_EQ(kMarginY, std::lround(max.y()));

  // Test at zoom factor 2.
  SetCurrentBounds(2.0 /* zoom_factor */, centre);

  Vector2_d new_centre =
      MapToScreen(centre.lat().degrees(), centre.lng().degrees());
  EXPECT_EQ(kScreenWidth / 2, std::lround(new_centre.x()));
  EXPECT_EQ(kScreenHeight / 2, std::lround(new_centre.y()));

  Vector2_d new_min =
      MapToScreen(kMinLatitude + kLatRange / 4, kMinLongitude + kLngRange / 4);
  EXPECT_EQ(kMarginX, std::lround(new_min.x()));
  EXPECT_EQ(kScreenHeight - kMarginY, std::lround(new_min.y()));

  Vector2_d new_max =
      MapToScreen(kMaxLatitude - kLatRange / 4, kMaxLongitude - kLngRange / 4);
  EXPECT_EQ(kScreenWidth - kMarginX, std::lround(new_max.x()));
  EXPECT_EQ(kMarginY, std::lround(new_max.y()));
}

TEST_F(RegionMapperTest, OffsetCentreTest) {
  // Test with centre offset.
  S2LatLng offset_centre(S1Angle::Degrees(kLatCentre - kLatRange / 2),
                         S1Angle::Degrees(kLngCentre - kLngRange / 2));
  SetCurrentBounds(2.0 /* zoom_factor */, offset_centre);

  S2LatLng centre = image_centre();
  Vector2_d offset_test =
      MapToScreen(centre.lat().degrees(), centre.lng().degrees());
  EXPECT_EQ(kScreenWidth, std::lround(offset_test.x()));
  EXPECT_EQ(0, std::lround(offset_test.y()));
}

TEST_F(RegionMapperTest, ResetCurrentBoundsTest) {
  // Offset the centre firstly.
  S2LatLng offset_centre(S1Angle::Degrees(kLatCentre - kLatRange / 2),
                         S1Angle::Degrees(kLngCentre - kLngRange / 2));
  SetCurrentBounds(2.0 /* zoom_factor */, offset_centre);

  // Then test ResetCurrentBounds.
  ResetCurrentBounds(2.0 /* zoom_factor */);
  S2LatLng centre = image_centre();
  Vector2_d reset_test =
      MapToScreen(centre.lat().degrees(), centre.lng().degrees());
  EXPECT_EQ(kScreenWidth / 2, std::lround(reset_test.x()));
  EXPECT_EQ(kScreenHeight / 2, std::lround(reset_test.y()));
}

TEST_F(RegionMapperTest, MapToBufferTest) {
  S2LatLng centre = image_centre();
  SetCurrentBounds(1.0 /* zoom_factor */, centre);

  Vector2_d mid =
      MapToBuffer(kLatCentre, kLngCentre, kScreenWidth, kScreenHeight);
  EXPECT_EQ(kScreenWidth / 2, std::lround(mid.x()));
  EXPECT_EQ(kScreenHeight / 2, std::lround(mid.y()));

  Vector2_d min =
      MapToBuffer(kMinLatitude, kMinLongitude, kScreenWidth, kScreenHeight);
  EXPECT_EQ(kMarginX, std::lround(min.x()));
  EXPECT_EQ(kScreenHeight - kMarginY, std::lround(min.y()));

  Vector2_d max =
      MapToBuffer(kMaxLatitude, kMaxLongitude, kScreenWidth, kScreenHeight);
  EXPECT_EQ(kScreenWidth - kMarginX, std::lround(max.x()));
  EXPECT_EQ(kMarginY, std::lround(max.y()));
}

}  // namespace
}  // namespace streetlearn
