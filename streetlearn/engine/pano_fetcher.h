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

#ifndef STREETLEARN_PANO_FETCHER_H_
#define STREETLEARN_PANO_FETCHER_H_

#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "streetlearn/engine/dataset.h"
#include "streetlearn/engine/pano_graph_node.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// Class to fetch panos from disk for the LRU Cache maintained by the
// PanoGraph. Conceptually, it consists of a synchronized queue of items to
// be fetched and a pool of worker threads that pop items from the queue and
// process them. Non-binding attempts to remove items from the queue are
// supported. All queue operations are serialized.
//
// The fetcher owns a callback to handle notifications when asynchronous fetches
// have completed. This callback is called concurrently by the worker threads
// and must therefore be thread-safe. Any state referenced by the callback must
// outlive the fetcher.
class PanoFetcher {
 public:
  using FetchCallback = std::function<void(
      absl::string_view pano_file, std::shared_ptr<const PanoGraphNode>)>;

  // Constructs a PanoFetcher using data from the dataset provided using the
  // given callback. The fetcher uses thread_count threads for fetching.
  PanoFetcher(const Dataset* dataset, int thread_count, FetchCallback callback);

  // Cancels and joins all of the fetching threads.
  ~PanoFetcher() LOCKS_EXCLUDED(queue_mutex_);

  // Fetch a pano synchronously by ID. Returns null if the current graph does
  // not contain a pano with the given ID.
  std::shared_ptr<const PanoGraphNode> Fetch(absl::string_view pano_id);

  // Fetch a pano asynchronously by ID and calls the callback when complete.
  void FetchAsync(absl::string_view pano_id) LOCKS_EXCLUDED(queue_mutex_);

  // Clears the fetch queue and returns the IDs of panos cancelled.
  std::vector<std::string> CancelPendingFetches() LOCKS_EXCLUDED(queue_mutex_);

 private:
  // Thread routine for loading panos.
  void ThreadExecute();

  // Waits until a pano request is pending on the queue or shutdown has been
  // requested. Returns true when a request is pending and false on shutdown.
  bool MonitorRequests(std::string* filename) LOCKS_EXCLUDED(queue_mutex_);

  // Dataset to read the panos.
  const Dataset* dataset_;

  // The threads used for fetching.
  std::vector<std::thread> threads_;

  // Queue for fetch requests.
  std::deque<std::string> fetch_queue_ GUARDED_BY(queue_mutex_);

  // Fetch queue mutex.
  absl::Mutex queue_mutex_;

  // Callback for asynchronous requests.
  FetchCallback callback_;

  // Shutdown flag to terminate fetch threads.
  bool shutdown_ GUARDED_BY(queue_mutex_);
};

}  // namespace streetlearn

#endif  // STREETLEARN_PANO_FETCHER_H_
