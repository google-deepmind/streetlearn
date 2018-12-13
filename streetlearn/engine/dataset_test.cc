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

#include "streetlearn/engine/dataset.h"

#include <cstdint>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "leveldb/db.h"
#include "leveldb/iterator.h"
#include "leveldb/options.h"
#include "leveldb/slice.h"
#include "leveldb/status.h"
#include "leveldb/write_batch.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

class MockLevelDB : public ::leveldb::DB {
 public:
  MockLevelDB() = default;
  ~MockLevelDB() override = default;

  MOCK_METHOD3(Put, ::leveldb::Status(const ::leveldb::WriteOptions& options,
                                      const ::leveldb::Slice& key,
                                      const ::leveldb::Slice& value));

  MOCK_METHOD2(Delete, ::leveldb::Status(const ::leveldb::WriteOptions& options,
                                         const ::leveldb::Slice& key));

  MOCK_METHOD2(Write, ::leveldb::Status(const ::leveldb::WriteOptions& options,
                                        ::leveldb::WriteBatch* updates));

  MOCK_METHOD3(Get, ::leveldb::Status(const ::leveldb::ReadOptions& options,
                                      const ::leveldb::Slice& key,
                                      std::string* value));
  MOCK_METHOD1(NewIterator,
               ::leveldb::Iterator*(const ::leveldb::ReadOptions& options));

  MOCK_METHOD0(GetSnapshot, const ::leveldb::Snapshot*());

  MOCK_METHOD1(ReleaseSnapshot, void(const ::leveldb::Snapshot*));

  MOCK_METHOD2(GetProperty,
               bool(const ::leveldb::Slice& property, std::string* value));

  MOCK_METHOD3(GetApproximateSizes,
               void(const ::leveldb::Range* range, int n, uint64_t* sizes));

  MOCK_METHOD2(CompactRange, void(const ::leveldb::Slice* begin,
                                  const ::leveldb::Slice* end));
};

TEST(DatasetTest, TestAccessWithNonExistingKeys) {
  constexpr char kTestPanoId[] = "non_existing_pano_id";
  auto mock_leveldb = absl::make_unique<::testing::StrictMock<MockLevelDB>>();

  EXPECT_CALL(*mock_leveldb, Get(_, ::leveldb::Slice(kTestPanoId), _))
      .WillOnce(Return(leveldb::Status::NotFound("not found")));
  EXPECT_CALL(*mock_leveldb, Get(_, ::leveldb::Slice(Dataset::kGraphKey), _))
      .WillOnce(Return(leveldb::Status::NotFound("not found")));

  Dataset dataset(std::move(mock_leveldb));

  StreetLearnGraph graph;
  EXPECT_FALSE(dataset.GetGraph(&graph));

  Pano pano;
  EXPECT_FALSE(dataset.GetPano(kTestPanoId, &pano));
}

TEST(DatasetTest, TestAccessWithExistingKeys) {
  constexpr char kTestPanoId[] = "test_pano_id";
  auto mock_leveldb = absl::make_unique<::testing::StrictMock<MockLevelDB>>();

  Pano test_pano;
  test_pano.set_id(kTestPanoId);
  const std::string test_pano_string = test_pano.SerializeAsString();

  StreetLearnGraph test_graph;
  test_graph.mutable_min_coords()->set_lat(42);
  const std::string test_graph_string = test_graph.SerializeAsString();

  EXPECT_CALL(*mock_leveldb, Get(_, ::leveldb::Slice(kTestPanoId), _))
      .WillOnce(DoAll(SetArgPointee<2>(test_pano_string),
                      Return(leveldb::Status::OK())));
  EXPECT_CALL(*mock_leveldb, Get(_, ::leveldb::Slice(Dataset::kGraphKey), _))
      .WillOnce(DoAll(SetArgPointee<2>(test_graph_string),
                      Return(leveldb::Status::OK())));

  Dataset dataset(std::move(mock_leveldb));

  StreetLearnGraph graph;
  EXPECT_TRUE(dataset.GetGraph(&graph));
  EXPECT_EQ(graph.SerializeAsString(), test_graph_string);

  Pano pano;
  EXPECT_TRUE(dataset.GetPano(kTestPanoId, &pano));
  EXPECT_EQ(pano.SerializeAsString(), test_pano_string);
}

}  // namespace
}  // namespace streetlearn
