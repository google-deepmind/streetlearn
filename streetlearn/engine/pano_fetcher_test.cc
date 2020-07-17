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

#include "streetlearn/engine/pano_fetcher.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "streetlearn/engine/dataset_factory.h"
#include "streetlearn/engine/pano_graph_node.h"
#include "streetlearn/engine/test_dataset.h"

namespace streetlearn {
namespace {

class PanoFetcherTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() { ASSERT_TRUE(TestDataset::Generate()); }

  void SetUp() {
    dataset_ = CreateDataset(TestDataset::GetPath());
    ASSERT_TRUE(dataset_ != nullptr);
  }

  std::unique_ptr<const Dataset> dataset_;
};

// Class to fetch a single batch of panos of the required size. Provides a wait
// method for asynchronous loads.
class FetchTest {
 public:
  explicit FetchTest(int panos_required) : counter_(panos_required) {}

  // Callback for the PanoFetcher when a new pano arrives. Optionally block on
  // the last fetch to test cancellation.
  void PanoLoaded(absl::string_view pano_file,
                  std::shared_ptr<const PanoGraphNode> pano)
      STREETLEARN_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock scope_lock(&mutex_);
    if (pano != nullptr) {
      loaded_.push_back(std::move(pano));
    }
    counter_.DecrementCount();
  }

  int LoadedCount() STREETLEARN_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock scope_lock(&mutex_);
    return loaded_.size();
  }

  bool PanoWithID(const std::string& pano_id, const PanoGraphNode* pano_node)
      STREETLEARN_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock scope_lock(&mutex_);
    auto it = std::find_if(
        loaded_.begin(), loaded_.end(),
        [&pano_id](const std::shared_ptr<const PanoGraphNode>& arg) {
          return arg->id() == pano_id;
        });
    if (it == loaded_.end()) {
      return false;
    }
    pano_node = it->get();
    return true;
  }

  // Wait for all remaining panos to be downloaded.
  void WaitForAsyncLoad(int cancelled) {
    for (int i = 0; i < cancelled; ++i) {
      counter_.DecrementCount();
    }
    counter_.Wait();
  }

  PanoFetcher::FetchCallback MakeCallback() {
    return [this](absl::string_view pano_file,
                  std::shared_ptr<const PanoGraphNode> node) {
      this->PanoLoaded(pano_file, std::move(node));
    };
  }

 private:
  std::vector<std::shared_ptr<const PanoGraphNode>> loaded_
      STREETLEARN_GUARDED_BY(mutex_);
  absl::BlockingCounter counter_;
  absl::Mutex mutex_;
};

TEST_F(PanoFetcherTest, TestPanoFetcher) {
  FetchTest fetch_test(TestDataset::kPanoCount);
  PanoFetcher pano_fetcher(dataset_.get(), TestDataset::kThreadCount,
                           fetch_test.MakeCallback());

  for (int i = 1; i <= TestDataset::kPanoCount; ++i) {
    pano_fetcher.FetchAsync(absl::StrCat(i));
  }

  fetch_test.WaitForAsyncLoad(0);

  EXPECT_EQ(fetch_test.LoadedCount(), TestDataset::kPanoCount);

  for (int i = 1; i <= TestDataset::kPanoCount; ++i) {
    PanoGraphNode* pano = nullptr;
    EXPECT_TRUE(fetch_test.PanoWithID(absl::StrCat(i), pano));
  }
}

TEST_F(PanoFetcherTest, CancelFetchTest) {
  FetchTest fetch_test(TestDataset::kPanoCount);
  PanoFetcher pano_fetcher(dataset_.get(), TestDataset::kThreadCount,
                           fetch_test.MakeCallback());

  std::vector<std::string> pano_ids(TestDataset::kPanoCount);
  for (int i = 1; i <= TestDataset::kPanoCount; ++i) {
    auto pano_id = absl::StrCat(i);
    pano_ids.emplace_back(pano_id);
    pano_fetcher.FetchAsync(pano_id);
  }

  // Since the fetching is asynchronous there is no way of knowing which panos
  // have been cancelled - can only check the right number have taken place.
  std::vector<std::string> cancelled = pano_fetcher.CancelPendingFetches();
  fetch_test.WaitForAsyncLoad(cancelled.size());
  EXPECT_EQ(fetch_test.LoadedCount(),
            TestDataset::kPanoCount - cancelled.size());
}

TEST_F(PanoFetcherTest, InvalidPanoTests) {
  FetchTest fetch_test(TestDataset::kPanoCount);
  PanoFetcher pano_fetcher(dataset_.get(), TestDataset::kThreadCount,
                           fetch_test.MakeCallback());

  auto pano_node = pano_fetcher.Fetch("Pano1");
  EXPECT_EQ(pano_node, nullptr);
}

}  // namespace
}  // namespace streetlearn
