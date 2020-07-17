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

#ifndef STREETLEARN_NODE_CACHE_H_
#define STREETLEARN_NODE_CACHE_H_

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "streetlearn/engine/pano_fetcher.h"
#include "streetlearn/engine/pano_graph_node.h"

namespace streetlearn {

// A Least Recently Used (LRU) cache for PanoGraphNodes. It uses a PanoFetcher
// to fetch nodes from disk and uses a callback mechanism to inform the callee
// when nodes are available. It works like this:
//
//            +-------+            +---------+
//   Lookup   |       |   Fetch    |         |
// ---------> |  LRU  | ---------> | Fetch   |
// <--------- | Cache | <--------- | Routine |
//  Callback  |       |  Callback  |         |
//            +-------+            +---------+
//
// The cache reads panos from a single directory passed to the constructor. All
// lookups and insertions are thread safe, but the copies of the callbacks
// passed to Lookup are called concurrently, the most recently added first.
class NodeCache {
 public:
  // Type of the client callback.
  using LoadCallback = std::function<void(const PanoGraphNode*)>;

  // Create a Cache that will use thread_count threads for fetching panos from
  // the dataset and store up to max_size pano nodes.
  static std::unique_ptr<NodeCache> CreateNodeCache(
      const Dataset* dataset, int thread_count, std::size_t max_size);

  // The Cache will use thread_count threads for fetching panos from the dataset
  // and store up to max_size pano nodes.
  NodeCache(const Dataset* dataset, int thread_count, std::size_t max_size);

  ~NodeCache() { CancelPendingFetches(); }

  // Looks up a pano node and runs the callback when it is available. Returns
  // true if the pano is already in the cache and available immediately and
  // false if the fetcher has been used.
  bool Lookup(const std::string& pano_id, LoadCallback callback)
      STREETLEARN_LOCKS_EXCLUDED(mutex_);

  // Called by the fetcher when panos are loaded.
  void Insert(const std::string& pano_id,
              std::shared_ptr<const PanoGraphNode> node)
      STREETLEARN_LOCKS_EXCLUDED(mutex_);

  // Cancel any outstanding fetches.
  void CancelPendingFetches() STREETLEARN_LOCKS_EXCLUDED(mutex_);

  // Return the cache size.
  int GetSize() { return cache_size_; }

 private:
  // Updates the cache with the give node and returns any callbacks required.
  ABSL_MUST_USE_RESULT
  std::list<LoadCallback> InsertImpl(const std::string& pano_id,
                                     std::shared_ptr<const PanoGraphNode> node)
      STREETLEARN_LOCKS_EXCLUDED(mutex_);

  // Remove and return any callbacks that are required for the given pano.
  ABSL_MUST_USE_RESULT
  std::list<LoadCallback> GetAndClearCallbacks(const std::string& pano_id);

  // Run the callbacks provided.
  void RunCallbacks(const std::list<LoadCallback>& callbacks,
                    const std::shared_ptr<const PanoGraphNode>& node)
      STREETLEARN_LOCKS_EXCLUDED(mutex_);

  // An entry in the LRU list the cache maintains.
  struct CacheEntry {
    std::string pano_id;
    std::shared_ptr<const PanoGraphNode> pano_node;

    CacheEntry(const std::string& id, std::shared_ptr<const PanoGraphNode> node)
        : pano_id(id), pano_node(std::move(node)) {}
  };

  // Insertions/lookups will happen from multiple fetch threads, so require
  // this mutex.
  absl::Mutex mutex_;

  // Store cache entries. The list is sorted by recency of entry lookups, with
  // the most recent entry at the front.
  std::list<CacheEntry> cache_list_ STREETLEARN_GUARDED_BY(mutex_);

  // Store lookup from key to entry in the cache list.
  absl::flat_hash_map<std::string, std::list<CacheEntry>::iterator>
      cache_lookup_ STREETLEARN_GUARDED_BY(mutex_);

  // Client callbacks when nodes become available.
  absl::flat_hash_map<std::string, std::list<LoadCallback>> callbacks_;

  // Maximum size of the cache.
  const std::size_t max_size_;

  // Current size of the cache.
  int cache_size_;

  // The pano fetcher
  PanoFetcher pano_fetcher_;
};

// Instantiates a NodeCache object.
std::unique_ptr<NodeCache> CreateNodeCache(const Dataset* dataset,
                                           int thread_count,
                                           std::size_t max_size);

}  //  namespace streetlearn

#endif  // STREETLEARN_NODE_CACHE_H_
