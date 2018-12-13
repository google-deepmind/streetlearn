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

#include <fstream>
#include <memory>
#include <streambuf>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "streetlearn/engine/pano_graph_node.h"

namespace streetlearn {

PanoFetcher::PanoFetcher(const Dataset* dataset, int thread_count,
                         FetchCallback callback)
    : dataset_(dataset),
      threads_(thread_count),
      callback_(std::move(callback)),
      shutdown_(false) {
  for (int i = 0; i < thread_count; ++i) {
    threads_[i] = std::thread(&PanoFetcher::ThreadExecute, this);
  }
}

PanoFetcher::~PanoFetcher() {
  {
    absl::MutexLock lock(&queue_mutex_);
    shutdown_ = true;
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}

std::shared_ptr<const PanoGraphNode> PanoFetcher::Fetch(
    absl::string_view pano_id) {
  Pano pano;
  if (dataset_->GetPano(pano_id, &pano)) {
    return std::make_shared<PanoGraphNode>(std::move(pano));
  }

  return nullptr;
}

void PanoFetcher::FetchAsync(absl::string_view pano_id) {
  absl::MutexLock lock(&queue_mutex_);
  fetch_queue_.emplace_front(pano_id);
}

bool PanoFetcher::MonitorRequests(std::string* pano_id) {
  auto lock_condition = [](PanoFetcher* fetcher) {
    fetcher->queue_mutex_.AssertHeld();
    return !fetcher->fetch_queue_.empty() || fetcher->shutdown_;
  };

  queue_mutex_.LockWhen(absl::Condition(+lock_condition, this));
  if (!shutdown_) {
    *pano_id = fetch_queue_.back();
    fetch_queue_.pop_back();
    queue_mutex_.Unlock();
    return true;
  } else {
    queue_mutex_.Unlock();
    return false;
  }
}

void PanoFetcher::ThreadExecute() {
  std::string pano_id;
  while (MonitorRequests(&pano_id)) {
    callback_(pano_id, Fetch(pano_id));
  }
}

std::vector<std::string> PanoFetcher::CancelPendingFetches() {
  absl::MutexLock lock(&queue_mutex_);
  std::vector<std::string> cancelled(
      {fetch_queue_.begin(), fetch_queue_.end()});
  fetch_queue_.clear();
  return cancelled;
}

}  // namespace streetlearn
