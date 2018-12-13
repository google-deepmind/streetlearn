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

#include "streetlearn/engine/metadata_cache.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

TEST(StreetLearn, MetadataCacheTest) {
  ASSERT_TRUE(TestDataset::Generate());
  std::unique_ptr<const Dataset> dataset =
      Dataset::Create(TestDataset::GetPath());
  ASSERT_TRUE(dataset != nullptr);

  auto metadata_cache =
      MetadataCache::Create(dataset.get(), TestDataset::kMinGraphDepth);
  ASSERT_TRUE(metadata_cache);

  EXPECT_EQ(metadata_cache->min_lat(), TestDataset::kMinLatitude);
  EXPECT_EQ(metadata_cache->max_lat(), TestDataset::kMaxLatitude);
  EXPECT_EQ(metadata_cache->min_lng(), TestDataset::kMinLongitude);
  EXPECT_EQ(metadata_cache->max_lng(), TestDataset::kMaxLongitude);
  EXPECT_EQ(metadata_cache->size(), TestDataset::kPanoCount);

  // Invalis Pano ID
  auto* invalid_pano = metadata_cache->GetPanoMetadata(
      absl::StrCat(2 * TestDataset::kPanoCount));
  EXPECT_EQ(invalid_pano, nullptr);

  // Valid indices.
  for (int pano_id = 1; pano_id <= TestDataset::kPanoCount; ++pano_id) {
    auto metadata = metadata_cache->GetPanoMetadata(absl::StrCat(pano_id));
    ASSERT_NE(metadata, nullptr);

    // Pano Coords
    LatLng lat_lng = metadata->pano.coords();
    EXPECT_EQ(lat_lng.lat(),
              TestDataset::kLatitude + TestDataset::kIncrement * pano_id);
    EXPECT_EQ(lat_lng.lng(),
              TestDataset::kLongitude + TestDataset::kIncrement * pano_id);

    // Graph Depth - zero for the isolated temporal neighbor.
    if (pano_id < TestDataset::kPanoCount + 1) {
      EXPECT_EQ(metadata->graph_depth, TestDataset::kPanoCount);
    } else {
      EXPECT_EQ(metadata->graph_depth, 0);
    }

    // Neighbors
    auto neighbors = metadata->neighbors;
    if (pano_id == 1 || pano_id == TestDataset::kPanoCount) {
      EXPECT_EQ(neighbors.size(), 1);
    } else if (pano_id < TestDataset::kPanoCount + 1) {
      EXPECT_EQ(neighbors.size(), TestDataset::kNeighborCount);
    } else {
      EXPECT_EQ(neighbors.size(), 0);
    }
    for (const auto& neighbor : neighbors) {
      int id;
      EXPECT_TRUE(absl::SimpleAtoi(neighbor.id(), &id));
      const auto& coords = neighbor.coords();
      EXPECT_EQ(coords.lat(),
                TestDataset::kLatitude + TestDataset::kIncrement * id);
      EXPECT_EQ(coords.lng(),
                TestDataset::kLongitude + TestDataset::kIncrement * id);
      int neighbor_id;
      ASSERT_TRUE(absl::SimpleAtoi(neighbor.id(), &neighbor_id));
      EXPECT_GE(neighbor_id, 1);
      EXPECT_LE(neighbor_id, TestDataset::kPanoCount);
    }
  }
}

TEST(StreetLearn, InvalidDataTest) {
  ASSERT_TRUE(TestDataset::GenerateInvalid());
  std::unique_ptr<const Dataset> invalid_dataset =
      Dataset::Create(TestDataset::GetInvalidDatasetPath());
  ASSERT_TRUE(invalid_dataset != nullptr);

  auto metadata_cache =
      MetadataCache::Create(invalid_dataset.get(), TestDataset::kMinGraphDepth);
  EXPECT_FALSE(metadata_cache);
}

}  // namespace
}  // namespace streetlearn
