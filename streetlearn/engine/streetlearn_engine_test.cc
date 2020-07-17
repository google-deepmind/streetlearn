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

#include "streetlearn/engine/streetlearn_engine.h"

#include <array>
#include <map>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "streetlearn/engine/dataset_factory.h"
#include "streetlearn/engine/math_util.h"
#include "streetlearn/engine/node_cache.h"
#include "streetlearn/engine/pano_calculations.h"
#include "streetlearn/engine/test_dataset.h"
#include "streetlearn/engine/test_utils.h"

namespace streetlearn {
namespace {

constexpr int kScreenWidth = 640;
constexpr int kScreenHeight = 480;
constexpr int kStatusHeight = 10;
constexpr int kFieldOfView = 30;
constexpr double kPanoOffset = 32.0;
constexpr int kMinGraphDepth = 10;
constexpr int kMaxGraphDepth = 50;
constexpr int kImageDepth = 3;
constexpr int kBufferSize =
    kImageDepth * TestDataset::kImageWidth * TestDataset::kImageHeight;

constexpr Color kObserverColor = {0.4, 0.6, 0.9};
constexpr Color kNodeColor = {0.1, 0.9, 0.1};
constexpr bool kBlackOnWhite = false;

using ::testing::DoubleEq;
using ::testing::Eq;
using ::testing::Optional;

class StreetLearnEngineTest : public testing::Test {
 public:
  static void SetUpTestSuite() { ASSERT_TRUE(TestDataset::Generate()); }

  void SetUp() override {
    std::shared_ptr<Dataset> dataset = CreateDataset(TestDataset::GetPath());
    ASSERT_TRUE(dataset != nullptr);
    std::shared_ptr<NodeCache> node_cache = CreateNodeCache(
        dataset.get(), TestDataset::kThreadCount, TestDataset::kMaxCacheSize);
    ASSERT_TRUE(node_cache != nullptr);

    engine_ = absl::make_unique<StreetLearnEngine>(
        std::move(dataset), std::move(node_cache),
        Vector2_i(TestDataset::kImageWidth, TestDataset::kImageHeight),
        Vector2_i(kScreenWidth, kScreenHeight), kStatusHeight, kFieldOfView,
        kMinGraphDepth, kMaxGraphDepth);

    engine_->InitEpisode(0 /*episode_index*/, 0 /*random_seed*/);
    ASSERT_THAT(engine_->BuildGraphWithRoot("1"), Optional(Eq("1")));

    ASSERT_TRUE(engine_->InitGraphRenderer(
        kObserverColor,
        {{"2", kNodeColor}, {"4", kNodeColor}, {"6", kNodeColor}},
        kBlackOnWhite));
  }

  std::unique_ptr<StreetLearnEngine> engine_;
};

TEST_F(StreetLearnEngineTest, TestEngine) {
  std::array<int, 3> dims = {
      {kImageDepth, TestDataset::kImageHeight, TestDataset::kImageWidth}};
  EXPECT_EQ(engine_->ObservationDims(), dims);

  // Set position a couple of times and check the result.
  EXPECT_THAT(engine_->SetPosition("1"), Optional(Eq("1")));
  EXPECT_THAT(engine_->GetPano()->id(), Eq("1"));
  EXPECT_THAT(engine_->SetPosition("2"), Optional(Eq("2")));
  EXPECT_THAT(engine_->GetPano()->id(), Eq("2"));

  // Currently facing north so cannot move to the next pano.
  EXPECT_THAT(engine_->MoveToNextPano(), Optional(Eq("2")));
  EXPECT_EQ(engine_->GetPitch(), 0);
  EXPECT_DOUBLE_EQ(engine_->GetYaw(), 0);
  const auto pano2 = engine_->GetPano();

  // Rotate to face the next pano and move should succeed.
  engine_->RotateObserver(kPanoOffset, 0.0);
  EXPECT_THAT(engine_->MoveToNextPano(), Optional(Eq("3")));
  EXPECT_EQ(engine_->GetPano()->id(), "3");
  const auto pano3 = engine_->GetPano();

  EXPECT_THAT(engine_->GetPanoDistance("2", "3"),
              Optional(DoubleEq(DistanceBetweenPanos(*pano2, *pano3))));

  // Check that obervations are the right size.
  auto obs = engine_->RenderObservation();
  EXPECT_EQ(obs.size(), kBufferSize);

  // Should have two neighbors.
  auto occupancy = engine_->GetNeighborOccupancy(4);
  EXPECT_THAT(occupancy, ::testing::ElementsAre(1, 0, 1, 0));

  // Check that the right metadata is returned.
  auto metadata = engine_->GetMetadata("1");
  EXPECT_EQ(metadata->pano.id(), "1");
}

TEST_F(StreetLearnEngineTest, TestGraphRendering) {
  constexpr char kTestPanoID[] = "5";
  constexpr double kYawDegrees = 45.0;

  engine_->SetPosition(kTestPanoID);
  engine_->RotateObserver(kYawDegrees, 0.0);
  constexpr Color kHighlightColor = {0.9, 0.1, 0.1};
  const std::map<std::string, Color> kHighlightedPanos = {
      {"1", kHighlightColor}, {"3", kHighlightColor}, {"5", kHighlightColor}};

  std::vector<uint8_t> image_buffer(kImageDepth * kScreenWidth * kScreenHeight);
  ASSERT_TRUE(
      engine_->DrawGraph(kHighlightedPanos, absl::MakeSpan(image_buffer)));
  EXPECT_TRUE(test_utils::CompareRGBBufferWithImage(
      absl::MakeSpan(image_buffer), kScreenWidth, kScreenHeight,
      test_utils::PixelFormat::kPlanarRGB,
      "engine/test_data/graph_test_observer.png"));
}

TEST(StreetLearn, StreetLearnEngineCreateTest) {
  TestDataset::Generate();

  auto engine = StreetLearnEngine::Create(
      TestDataset::GetPath(), TestDataset::kImageWidth,
      TestDataset::kImageHeight, TestDataset::kImageWidth,
      TestDataset::kImageHeight, kStatusHeight, kFieldOfView);
  engine->InitEpisode(0 /*episode_index*/, 0 /*random_seed*/);

  // Do some basic checks on the graph.
  auto opt_id = engine->BuildRandomGraph();
  ASSERT_TRUE(opt_id);
  EXPECT_EQ(engine->SetPosition(opt_id.value()), opt_id.value());
  EXPECT_EQ(engine->GetPano()->id(), opt_id.value());

  auto obs = engine->RenderObservation();
  EXPECT_EQ(obs.size(), kBufferSize);

  auto graph = engine->GetGraph();
  EXPECT_EQ(graph.size(), TestDataset::kPanoCount);
}


TEST(StreetLearn, StreetLearnEngineCloneTest) {
  TestDataset::Generate();

  auto engine1 = StreetLearnEngine::Create(
      TestDataset::GetPath(), TestDataset::kImageWidth,
      TestDataset::kImageHeight, TestDataset::kImageWidth,
      TestDataset::kImageHeight, kStatusHeight, kFieldOfView);
  engine1->InitEpisode(0 /*episode_index*/, 0 /*random_seed*/);

  auto engine2 = engine1->Clone(
      TestDataset::kImageWidth, TestDataset::kImageHeight,
      TestDataset::kImageWidth, TestDataset::kImageHeight,
      kStatusHeight, kFieldOfView);
  engine2->InitEpisode(0 /*episode_index*/, 1 /*random_seed*/);

  auto engine3 = engine2->Clone(
      TestDataset::kImageWidth, TestDataset::kImageHeight,
      TestDataset::kImageWidth, TestDataset::kImageHeight,
      kStatusHeight, kFieldOfView);
  engine3->InitEpisode(0 /*episode_index*/, 2 /*random_seed*/);

  // Do some basic checks on the graph.
  auto opt_id = engine3->BuildRandomGraph();
  ASSERT_TRUE(opt_id);
  EXPECT_EQ(engine3->SetPosition(opt_id.value()), opt_id.value());
  EXPECT_EQ(engine3->GetPano()->id(), opt_id.value());

  auto obs = engine3->RenderObservation();
  EXPECT_EQ(obs.size(), kBufferSize);

  auto graph = engine3->GetGraph();
  EXPECT_EQ(graph.size(), TestDataset::kPanoCount);
}

}  // namespace
}  // namespace streetlearn
