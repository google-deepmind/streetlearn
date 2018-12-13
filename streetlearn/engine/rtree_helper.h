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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_RTREE_HELPER_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_RTREE_HELPER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "s2/s2latlng_rect.h"

namespace streetlearn {

// Helper class that encapsulates a generic RTree implementation for the purpose
// of executing intersection queries against S2LatLngRects. This RTree is only
// capable of storing int values and only supports insertion and intersection
// query.
class RTree {
 public:
  RTree();
  ~RTree();

  RTree(const RTree&) = delete;
  RTree& operator=(const RTree&) = delete;

  // Inserts S2LatLngRect, int pairs.
  void Insert(const S2LatLngRect& rect, int value);

  // Query boxes that intersect `rect` and insert values to the result vector.
  // Returns the number of intersecting boxes.
  std::size_t FindIntersecting(const S2LatLngRect& rect,
                               std::vector<int>* out) const;

  // Returns whether the tree is empty.
  bool empty() const;

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_RTREE_HELPER_H_
