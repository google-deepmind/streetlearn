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

#include "streetlearn/engine/graph_image_cache.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "streetlearn/engine/cairo_util.h"
#include "streetlearn/engine/dataset_factory.h"
#include "streetlearn/engine/node_cache.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/test_dataset.h"
#include "streetlearn/engine/test_utils.h"
#include "streetlearn/engine/vector.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"

namespace streetlearn {
namespace {

constexpr int kScreenWidth = 640;
constexpr int kScreenHeight = 480;
constexpr int kGraphDepth = 10;
constexpr char kTestFilePath[] = "engine/test_data/";
constexpr bool kBlackOnWhiteFalse = false;
constexpr bool kBlackOnWhiteTrue = true;

class GraphImageCacheTest : public testing::Test {
 public:
  static void SetUpTestSuite() { ASSERT_TRUE(TestDataset::Generate()); }

  void SetUp() {
    dataset_ = CreateDataset(TestDataset::GetPath());
    ASSERT_TRUE(dataset_ != nullptr);
    node_cache_ = CreateNodeCache(dataset_.get(), TestDataset::kThreadCount,
                                  TestDataset::kMaxCacheSize);
    ASSERT_TRUE(node_cache_ != nullptr);

    pano_graph_ = absl::make_unique<PanoGraph>(
        TestDataset::kMaxGraphDepth, TestDataset::kMinGraphDepth,
        TestDataset::kMaxGraphDepth, std::move(dataset_),
        std::move(node_cache_));

    // Build the pano graph.
    pano_graph_->SetRandomSeed(1);
    ASSERT_TRUE(pano_graph_->Init());
    ASSERT_TRUE(pano_graph_->BuildGraphWithRoot("1"));

    // Initialise the graph renderer.
    auto bounds_min = S2LatLng::FromDegrees(TestDataset::kMinLatitude,
                                            TestDataset::kMinLongitude);
    auto bounds_max = S2LatLng::FromDegrees(TestDataset::kMaxLatitude,
                                            TestDataset::kMaxLongitude);
    S2LatLngRect graph_bounds(bounds_min, bounds_max);
    graph_image_cache_ = absl::make_unique<GraphImageCache>(
        Vector2_i{kScreenWidth, kScreenHeight}, std::map<std::string, Color>{},
        kBlackOnWhiteFalse);

    ASSERT_TRUE(
        graph_image_cache_->InitCache(*pano_graph_.get(), graph_bounds));

    graph_image_cache_black_on_white_ = absl::make_unique<GraphImageCache>(
        Vector2_i{kScreenWidth, kScreenHeight}, std::map<std::string, Color>{},
        kBlackOnWhiteTrue);

    ASSERT_TRUE(
        graph_image_cache_black_on_white_->InitCache(*pano_graph_.get(),
                                                     graph_bounds));
  }

  std::unique_ptr<GraphImageCache> graph_image_cache_;
  std::unique_ptr<GraphImageCache> graph_image_cache_black_on_white_;

 private:
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<NodeCache> node_cache_;
  std::unique_ptr<PanoGraph> pano_graph_;
};

TEST_F(GraphImageCacheTest, GraphImageCacheTest) {
  const S2LatLng image_centre = S2LatLng::FromDegrees(
      (TestDataset::kMaxLatitude + TestDataset::kMinLatitude) / 2,
      (TestDataset::kMaxLongitude + TestDataset::kMinLongitude) / 2);

  // Test rendering.
  EXPECT_TRUE(test_utils::CompareImages(
      graph_image_cache_->Pixels(image_centre),
      absl::StrCat(kTestFilePath, "image_cache_test.png")));

  // Test zoom.
  graph_image_cache_->SetZoom(2.0);
  EXPECT_DOUBLE_EQ(graph_image_cache_->current_zoom(), 2.0);
  EXPECT_TRUE(test_utils::CompareImages(
      graph_image_cache_->Pixels(image_centre),
      absl::StrCat(kTestFilePath, "image_cache_test_zoomed.png")));

  graph_image_cache_->SetZoom(1.0);
  EXPECT_DOUBLE_EQ(graph_image_cache_->current_zoom(), 1.0);
  EXPECT_TRUE(test_utils::CompareImages(
      graph_image_cache_->Pixels(image_centre),
      absl::StrCat(kTestFilePath, "image_cache_test.png")));

  // Test black on white graphs.
  EXPECT_FALSE(test_utils::CompareImages(
      graph_image_cache_black_on_white_->Pixels(image_centre),
      absl::StrCat(kTestFilePath, "image_cache_test.png")));
}

}  // namespace
}  // namespace streetlearn
