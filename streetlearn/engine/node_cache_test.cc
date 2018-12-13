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

#include "streetlearn/engine/node_cache.h"

#include <map>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

constexpr int kThreadCount = 8;

class CacheTest : public ::testing::Test {
 public:
  static void SetUpTestCase() { ASSERT_TRUE(TestDataset::Generate()); }

  CacheTest()
      : dataset_(Dataset::Create(TestDataset::GetPath())),
        node_cache_(dataset_.get(), kThreadCount, TestDataset::kMaxCacheSize) {}

  void LoadPano(const std::string& pano_id) {
    notification_ = absl::make_unique<absl::Notification>();
    node_cache_.Lookup(pano_id,
                       [this](const PanoGraphNode* node) { PanoLoaded(node); });
    notification_->WaitForNotification();
  }

  void LoadPanoAsync(const std::string& pano_id) {
    node_cache_.Lookup(pano_id,
                       [this](const PanoGraphNode* node) { PanoLoaded(node); });
  }

  const PanoGraphNode* GetNode(const std::string& pano_id)
      LOCKS_EXCLUDED(mutex_) {
    absl::ReaderMutexLock lock(&mutex_);
    return loaded_panos_.find(pano_id) != loaded_panos_.end()
               ? loaded_panos_[pano_id]
               : nullptr;
  }

  bool CacheContains(const std::string& pano_id) {
    return node_cache_.Lookup(pano_id, [](const PanoGraphNode* node) {});
  }

  void InitBlockingCounter(int size) {
    blockingCounter_ = absl::make_unique<absl::BlockingCounter>(size);
  }

  void WaitForAll() { blockingCounter_->Wait(); }

 private:
  void PanoLoaded(const PanoGraphNode* node) LOCKS_EXCLUDED(mutex_) {
    if (node != nullptr) {
      absl::WriterMutexLock lock(&mutex_);
      loaded_panos_[node->id()] = node;
    }
    if (notification_) {
      notification_->Notify();
    }
    if (blockingCounter_) {
      blockingCounter_->DecrementCount();
    }
  }

  std::unique_ptr<const Dataset> dataset_;
  absl::Mutex mutex_;
  NodeCache node_cache_;
  absl::flat_hash_map<std::string, const PanoGraphNode*> loaded_panos_
      GUARDED_BY(mutex_);
  std::unique_ptr<absl::Notification> notification_;
  std::unique_ptr<absl::BlockingCounter> blockingCounter_;
};

TEST_F(CacheTest, NodeCacheTest) {
  // Fill the cache.
  for (int i = 1; i <= TestDataset::kMaxCacheSize; ++i) {
    auto node_id = absl::StrCat(i);
    LoadPano(node_id);
    const PanoGraphNode* node = GetNode(node_id);
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->id(), node_id);
  }

  // Test an eviction. The first element should now have been removed.
  auto node_id = absl::StrCat(TestDataset::kMaxCacheSize + 1);
  LoadPano(node_id);
  const PanoGraphNode* node = GetNode(node_id);
  EXPECT_NE(node, nullptr);
  EXPECT_EQ(node->id(), node_id);
  EXPECT_FALSE(CacheContains(absl::StrCat(1)));
}

TEST_F(CacheTest, NodeCacheTestAsync) {
  // Load the panos asynchronously.
  InitBlockingCounter(TestDataset::kMaxCacheSize);
  for (int i = 1; i <= TestDataset::kMaxCacheSize; ++i) {
    LoadPanoAsync(absl::StrCat(i));
  }

  WaitForAll();

  for (int i = 1; i <= TestDataset::kMaxCacheSize; ++i) {
    auto node_id = absl::StrCat(i);
    const PanoGraphNode* node = GetNode(node_id);
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->id(), node_id);
  }
}

}  // namespace
}  // namespace streetlearn
