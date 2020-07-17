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

namespace streetlearn {

std::unique_ptr<NodeCache> CreateNodeCache(const Dataset* dataset,
                                           int thread_count,
                                           std::size_t max_size) {
  if (!dataset) {
    LOG(ERROR) << "No dataset supplied. Cannot create node cache.";
    return nullptr;
  }

  return absl::make_unique<NodeCache>(dataset, thread_count, max_size);
}

NodeCache::NodeCache(const Dataset* dataset, int thread_count,
                     std::size_t max_size)
    : max_size_(max_size),
      cache_size_(0),
      pano_fetcher_(dataset, thread_count,
                    [this](absl::string_view pano_id,
                           std::shared_ptr<const PanoGraphNode> node) {
                      this->Insert(std::string(pano_id), std::move(node));
                    }) {}

bool NodeCache::Lookup(const std::string& pano_id, LoadCallback callback) {
  absl::MutexLock lock(&mutex_);
  auto it = cache_lookup_.find(pano_id);
  if (it == cache_lookup_.end()) {
    cache_size_ = cache_list_.size();
    pano_fetcher_.FetchAsync(pano_id);
    callbacks_[pano_id].push_front(std::move(callback));
    return false;
  }
  cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
  callback(it->second->pano_node.get());
  return true;
}

void NodeCache::Insert(const std::string& pano_id,
                       std::shared_ptr<const PanoGraphNode> node) {
  std::list<LoadCallback> callbacks = InsertImpl(pano_id, node);
  RunCallbacks(callbacks, node);
}

void NodeCache::CancelPendingFetches() {
  std::vector<std::string> cancelled = pano_fetcher_.CancelPendingFetches();
  for (const auto& pano_id : cancelled) {
    std::list<LoadCallback> callbacks;
    {
      absl::MutexLock lock(&mutex_);
      callbacks = GetAndClearCallbacks(pano_id);
    }
    RunCallbacks(callbacks, nullptr);
  }
}

std::list<NodeCache::LoadCallback> NodeCache::InsertImpl(
    const std::string& pano_id, std::shared_ptr<const PanoGraphNode> node) {
  absl::MutexLock lock(&mutex_);
  // Insert into the cache.
  auto it = cache_lookup_.find(pano_id);
  if (it != cache_lookup_.end()) {
    cache_list_.erase(it->second);
    cache_lookup_.erase(it);
  }
  cache_list_.push_front(CacheEntry(pano_id, std::move(node)));
  cache_lookup_[pano_id] = cache_list_.begin();

  // Evict an item if necessary.
  if (cache_list_.size() > max_size_) {
    cache_lookup_.erase(cache_list_.back().pano_id);
    cache_list_.pop_back();
  }

  return GetAndClearCallbacks(pano_id);
}

std::list<NodeCache::LoadCallback> NodeCache::GetAndClearCallbacks(
    const std::string& pano_id) {
  std::list<LoadCallback> callbacks;
  auto it = callbacks_.find(pano_id);
  if (it != callbacks_.end()) {
    callbacks = std::move(it->second);
    callbacks_.erase(it);
  }
  return callbacks;
}

void NodeCache::RunCallbacks(const std::list<LoadCallback>& callbacks,
                             const std::shared_ptr<const PanoGraphNode>& node) {
  const PanoGraphNode* pano_node = node ? node.get() : nullptr;
  for (const auto& callback : callbacks) {
    callback(pano_node);
  }
}

}  // namespace streetlearn
