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

#include "streetlearn/engine/pano_graph.h"

#include <cmath>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

using ::testing::DoubleNear;
using ::testing::Optional;

class PanoGraphTest : public ::testing::Test {
 public:
  static void SetUpTestCase() { ASSERT_TRUE(TestDataset::Generate()); }

  void SetUp() {
    dataset_ = Dataset::Create(TestDataset::GetPath());
    ASSERT_TRUE(dataset_ != nullptr);
  }

  std::unique_ptr<const Dataset> dataset_;
};

void TestGraph(PanoGraph* pano_graph, const std::string& root) {
  auto root_node = pano_graph->Root();
  if (!root.empty()) {
    EXPECT_EQ(root_node.id(), root);
  }
  auto root_image = pano_graph->RootImage();
  int graph_size = pano_graph->GetGraph().size();
  EXPECT_NE(root_image.get(), nullptr);

  PanoMetadata metadata;
  EXPECT_FALSE(pano_graph->Metadata("bogus", &metadata));
  ASSERT_TRUE(pano_graph->Metadata(root_node.id(), &metadata));
  EXPECT_LE(metadata.neighbors.size(), TestDataset::kNeighborCount);

  for (const auto& neighbor : metadata.neighbors) {
    EXPECT_TRUE(pano_graph->BuildGraphWithRoot(neighbor.id()));
    auto root_node = pano_graph->Root();
    EXPECT_EQ(root_node.id(), neighbor.id());
    EXPECT_EQ(pano_graph->GetGraph().size(), graph_size);
    auto neighbor_bearings = pano_graph->GetNeighborBearings(1);
    EXPECT_GE(neighbor_bearings.size(), 1);
  }

  auto old_graph = pano_graph->GetGraph();
  EXPECT_TRUE(pano_graph->MoveToNeighbor(30 /* bearing */, 5 /* tol */));
  EXPECT_TRUE(pano_graph->MoveToNeighbor(-150 /* bearing */, 5 /* tol */));
  auto new_graph = pano_graph->GetGraph();
  EXPECT_EQ(old_graph.size(), new_graph.size());

  // The panos are in a straight line at 32 degrees to each other.
  auto bearings =
      pano_graph->GetNeighborBearings(TestDataset::kMaxNeighborDepth);
  for (const auto& neighbor : bearings) {
    EXPECT_TRUE(round(fabs(neighbor.bearing)) == 32 ||
                round(fabs(neighbor.bearing)) == 148);
  }
}

TEST(StreetLearn, PanoGraphInitFailureTest) {
  ASSERT_TRUE(TestDataset::GenerateInvalid());
  std::unique_ptr<Dataset> invalid_dataset =
      Dataset::Create(TestDataset::GetInvalidDatasetPath());
  ASSERT_TRUE(invalid_dataset != nullptr);

  PanoGraph pano_graph(TestDataset::kMaxGraphDepth, TestDataset::kMaxCacheSize,
                       TestDataset::kMinGraphDepth, TestDataset::kMaxGraphDepth,
                       invalid_dataset.get());
  // Invalid dataset.
  EXPECT_FALSE(pano_graph.Init());
}

TEST_F(PanoGraphTest, TestPanoGraph) {
  PanoGraph pano_graph(TestDataset::kMaxGraphDepth, TestDataset::kMaxCacheSize,
                       TestDataset::kMinGraphDepth, TestDataset::kMaxGraphDepth,
                       dataset_.get());
  pano_graph.SetRandomSeed(0);
  // Valid metadata filename.
  ASSERT_TRUE(pano_graph.Init());
  EXPECT_FALSE(pano_graph.MoveToNeighbor(0, 0));

  // BuildRandomGraph test. The root is chosen randomly so we don't
  // test what it's set to.
  EXPECT_TRUE(pano_graph.BuildRandomGraph());
  TestGraph(&pano_graph, "" /* Don't check root */);
  EXPECT_FALSE(pano_graph.SetPosition("-1"));
  EXPECT_FALSE(pano_graph.MoveToNeighbor(270, 0));
  EXPECT_TRUE(pano_graph.SetPosition("2"));

  // BuildGraphWithRoot tests.
  EXPECT_TRUE(pano_graph.BuildGraphWithRoot("1"));
  TestGraph(&pano_graph, "1");

  for (int root_index = 1; root_index <= TestDataset::kPanoCount;
       ++root_index) {
    const std::string root_id = absl::StrCat(root_index);

    EXPECT_TRUE(pano_graph.BuildGraphWithRoot(root_id));
    TestGraph(&pano_graph, root_id);
  }

  // BuildEntireGraph test.
  EXPECT_TRUE(pano_graph.BuildEntireGraph());
  const auto graph = pano_graph.GetGraph();
  EXPECT_EQ(graph.size(), TestDataset::GetPanoIDs().size());
  TestGraph(&pano_graph, "" /* Don't check root */);
}

TEST_F(PanoGraphTest, InvalidGraphSizeTest) {
  PanoGraph pano_graph(TestDataset::kMaxGraphDepth, TestDataset::kMaxCacheSize,
                       2 * TestDataset::kMaxGraphDepth,
                       2 * TestDataset::kMaxGraphDepth, dataset_.get());
  ASSERT_TRUE(pano_graph.Init());
  EXPECT_FALSE(pano_graph.BuildRandomGraph());
}

TEST_F(PanoGraphTest, InvalidPanosTest) {
  PanoGraph pano_graph(TestDataset::kMaxGraphDepth, TestDataset::kMaxCacheSize,
                       TestDataset::kMinGraphDepth, TestDataset::kMaxGraphDepth,
                       dataset_.get());
  ASSERT_TRUE(pano_graph.Init());

  EXPECT_FALSE(pano_graph.BuildGraphWithRoot(""));

  std::string bogus_id = "-1";
  EXPECT_FALSE(pano_graph.BuildGraphWithRoot(bogus_id));
}

TEST_F(PanoGraphTest, TerminalBearingTest) {
  PanoGraph pano_graph(TestDataset::kMaxGraphDepth, TestDataset::kMaxCacheSize,
                       TestDataset::kMinGraphDepth / 2,
                       TestDataset::kMaxGraphDepth / 2, dataset_.get());
  ASSERT_TRUE(pano_graph.Init());
  EXPECT_TRUE(pano_graph.BuildGraphWithRoot("5"));

  const auto& terminalBearings = pano_graph.TerminalBearings("9");
  ASSERT_EQ(terminalBearings.size(), 1);
  const auto& bearings = *terminalBearings.begin();
  EXPECT_EQ(bearings.first, 1);
  ASSERT_EQ(bearings.second.size(), 1);
  const auto& terminal = *bearings.second.begin();
  EXPECT_EQ(std::round(terminal.distance), 131);
  EXPECT_EQ(std::round(terminal.bearing), 32);
}

TEST_F(PanoGraphTest, PanoCalculationTest) {
  PanoGraph pano_graph(TestDataset::kMaxGraphDepth, 42 /* large cache */,
                       TestDataset::kMinGraphDepth / 2,
                       TestDataset::kMaxGraphDepth / 2, dataset_.get());
  ASSERT_TRUE(pano_graph.Init());
  EXPECT_TRUE(pano_graph.BuildGraphWithRoot("1"));

  auto pano_ids = TestDataset::GetPanoIDs();
  PanoMetadata metadata1;
  ASSERT_TRUE(pano_graph.Metadata(pano_ids[0], &metadata1));
  const auto& pano1 = metadata1.pano;
  PanoMetadata metadata2;
  ASSERT_TRUE(pano_graph.Metadata(pano_ids[1], &metadata2));
  const auto& pano2 = metadata2.pano;

  // TODO(b/117477463): elminitate all magic numbers
  EXPECT_THAT(pano_graph.GetPanoDistance(pano1.id(), pano2.id()),
              Optional(DoubleNear(131, 0.1)));
  EXPECT_THAT(pano_graph.GetPanoBearing(pano1.id(), pano2.id()),
              Optional(DoubleNear(31.9, 0.1)));

  std::string bogus_id = "-1";
  EXPECT_FALSE(pano_graph.GetPanoDistance(pano1.id(), bogus_id).has_value());
  EXPECT_FALSE(pano_graph.GetPanoDistance(bogus_id, pano1.id()).has_value());
  EXPECT_FALSE(pano_graph.GetPanoBearing(pano1.id(), bogus_id).has_value());
  EXPECT_FALSE(pano_graph.GetPanoBearing(bogus_id, pano1.id()).has_value());
}

}  // namespace
}  // namespace streetlearn
