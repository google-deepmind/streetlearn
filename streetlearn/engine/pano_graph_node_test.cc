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

#include "streetlearn/engine/pano_graph_node.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

TEST(StreetLearn, PanoGraphNodeTest) {
  constexpr int kTestPanoId = 1;
  Pano pano = TestDataset::GeneratePano(kTestPanoId);
  const auto compressed_pano_image = TestDataset::GenerateCompressedTestImage();
  pano.mutable_compressed_image()->assign(compressed_pano_image.begin(),
                                          compressed_pano_image.end());
  const PanoGraphNode graph_node(pano);

  EXPECT_EQ(graph_node.id(), absl::StrCat(kTestPanoId));
  EXPECT_EQ(graph_node.date(), TestDataset::kPanoDate);
  EXPECT_EQ(graph_node.latitude(),
            TestDataset::kLatitude + TestDataset::kIncrement);
  EXPECT_EQ(graph_node.longitude(),
            TestDataset::kLongitude + TestDataset::kIncrement);
  EXPECT_EQ(graph_node.bearing(), TestDataset::kHeadingDegrees);

  auto image = graph_node.image();
  EXPECT_EQ(image->width(), TestDataset::kImageWidth);
  EXPECT_EQ(image->height(), TestDataset::kImageHeight);
  EXPECT_EQ(image->data(), TestDataset::GenerateTestImage().data());

  auto metadata = graph_node.GetPano();
  EXPECT_EQ(metadata->street_name(), TestDataset::kStreetName);
  EXPECT_EQ(metadata->full_address(), TestDataset::kFullAddress);
  EXPECT_EQ(metadata->region(), TestDataset::kRegion);
  EXPECT_EQ(metadata->country_code(), TestDataset::kCountryCode);
}

TEST(StreetLearn, InvalidPanoTest) {
  Pano pano;
  PanoGraphNode graph_node(pano);
  EXPECT_EQ(graph_node.image().get(), nullptr);
}

}  // namespace
}  // namespace streetlearn
