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

#include "streetlearn/engine/graph_renderer.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "streetlearn/engine/color.h"
#include "streetlearn/engine/graph_image_cache.h"
#include "streetlearn/engine/math_util.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/test_dataset.h"
#include "streetlearn/engine/test_utils.h"
#include "streetlearn/engine/vector.h"

namespace streetlearn {
namespace {

constexpr int kScreenWidth = 640;
constexpr int kScreenHeight = 480;
constexpr int kGraphDepth = 10;
constexpr double kObserverYawDegrees = 45;
constexpr double kObserverFovDegrees = 30;
constexpr char kTestPanoID[] = "5";
constexpr char kTestFilePath[] = "engine/test_data/";
constexpr Color kConeColor = {0.4, 0.6, 0.9};
constexpr Color kNodeColor = {0.1, 0.9, 0.1};
constexpr Color kHighlightColor = {0.9, 0.1, 0.1};

static const auto* const kColoredPanos = new std::map<std::string, Color>{
    {"2", kNodeColor}, {"4", kNodeColor}, {"6", kNodeColor}};

static const auto* const kHighlightedPanos = new std::map<std::string, Color>{
    {"1", kHighlightColor}, {"3", kHighlightColor}, {"5", kHighlightColor}};

class GraphRendererTest : public testing::Test {
 public:
  static void SetUpTestCase() { ASSERT_TRUE(TestDataset::Generate()); }

  void SetUp() override {
    dataset_ = Dataset::Create(TestDataset::GetPath());
    ASSERT_TRUE(dataset_ != nullptr);

    pano_graph_ = absl::make_unique<PanoGraph>(
        TestDataset::kMaxGraphDepth, TestDataset::kMaxCacheSize,
        TestDataset::kMinGraphDepth, TestDataset::kMaxGraphDepth,
        dataset_.get());
    graph_image_cache_ = absl::make_unique<GraphImageCache>(
        Vector2_i{kScreenWidth, kScreenHeight}, std::map<std::string, Color>{});

    // Build the pano graph.
    pano_graph_->SetRandomSeed(0);
    ASSERT_TRUE(pano_graph_->Init());
    ASSERT_TRUE(pano_graph_->BuildGraphWithRoot("1"));

    // Create the graph renderer.
    graph_renderer_ = GraphRenderer::Create(
        *pano_graph_, Vector2_i(kScreenWidth, kScreenHeight), *kColoredPanos);
  }

  std::unique_ptr<GraphRenderer> graph_renderer_;

 private:
  std::unique_ptr<const Dataset> dataset_;
  std::unique_ptr<PanoGraph> pano_graph_;
  std::unique_ptr<GraphImageCache> graph_image_cache_;
};

TEST_F(GraphRendererTest, Test) {
  ASSERT_TRUE(graph_renderer_->RenderScene(*kHighlightedPanos, absl::nullopt));
  std::vector<uint8_t> pixel_buffer(3 * kScreenWidth * kScreenHeight);
  graph_renderer_->GetPixels(absl::MakeSpan(pixel_buffer));
  EXPECT_TRUE(test_utils::CompareRGBBufferWithImage(
      pixel_buffer, kScreenWidth, kScreenHeight,
      test_utils::PixelFormat::kPackedRGB,
      absl::StrCat(kTestFilePath, "graph_test.png")));
}

TEST_F(GraphRendererTest, ZoomTest) {
  const std::map<double, std::string> kZoomLevels = {
      {2.0, "graph_test_zoom2.png"},
      {4.0, "graph_test_zoom4.png"},
      {8.0, "graph_test_zoom8.png"}};

  std::vector<uint8_t> pixel_buffer(3 * kScreenWidth * kScreenHeight);
  // Zoom in all the way.
  for (const auto& zoom_image_pair : kZoomLevels) {
    ASSERT_TRUE(graph_renderer_->SetZoom(zoom_image_pair.first));
    ASSERT_TRUE(
        graph_renderer_->RenderScene(*kHighlightedPanos, absl::nullopt));
    graph_renderer_->GetPixels(absl::MakeSpan(pixel_buffer));
    EXPECT_TRUE(test_utils::CompareRGBBufferWithImage(
        pixel_buffer, kScreenWidth, kScreenHeight,
        test_utils::PixelFormat::kPackedRGB,
        absl::StrCat(kTestFilePath, zoom_image_pair.second)));
  }

  // Zoom out.
  for (auto it = kZoomLevels.rbegin(); it != kZoomLevels.rend(); ++it) {
    ASSERT_TRUE(graph_renderer_->SetZoom(it->first));
    ASSERT_TRUE(
        graph_renderer_->RenderScene(*kHighlightedPanos, absl::nullopt));

    graph_renderer_->GetPixels(absl::MakeSpan(pixel_buffer));
    EXPECT_TRUE(test_utils::CompareRGBBufferWithImage(
        pixel_buffer, kScreenWidth, kScreenHeight,
        test_utils::PixelFormat::kPackedRGB,
        absl::StrCat(kTestFilePath, it->second)));
  }
}

TEST_F(GraphRendererTest, ObserverTest) {
  // Place an observer.
  Observer observer;
  observer.pano_id = kTestPanoID;
  observer.yaw_radians = math::DegreesToRadians(kObserverYawDegrees);
  observer.fov_yaw_radians = math::DegreesToRadians(kObserverFovDegrees);
  observer.color = kConeColor;

  ASSERT_TRUE(graph_renderer_->RenderScene(*kHighlightedPanos, observer));

  std::vector<uint8_t> pixel_buffer(3 * kScreenWidth * kScreenHeight);
  graph_renderer_->GetPixels(absl::MakeSpan(pixel_buffer));
  EXPECT_TRUE(test_utils::CompareRGBBufferWithImage(
      pixel_buffer, kScreenWidth, kScreenHeight,
      test_utils::PixelFormat::kPackedRGB,
      absl::StrCat(kTestFilePath, "graph_test_observer.png")));
}

}  // namespace
}  // namespace streetlearn
