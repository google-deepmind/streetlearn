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

#include "streetlearn/engine/rtree_helper.h"

#include <utility>

#include <boost/function_output_iterator.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/core/tags.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/parameters.hpp>
#include <boost/geometry/index/predicates.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "absl/memory/memory.h"
#include "s2/s1angle.h"
#include "s2/s2latlng_rect.h"

namespace streetlearn {
namespace {

namespace bg = boost::geometry;

using point_t = bg::model::point<double, 2, bg::cs::geographic<bg::radian>>;
using box_t = bg::model::box<point_t>;
using rtree_t =
    boost::geometry::index::rtree<std::pair<box_t, int>,
                                  boost::geometry::index::quadratic<16>>;

box_t CreateBox(const S2LatLngRect& rect) {
  return box_t{{rect.lng_lo().radians(), rect.lat_lo().radians()},
               {rect.lng_hi().radians(), rect.lat_hi().radians()}};
}

}  // namespace

struct RTree::Impl {
  // Helper function to insert S2LatLngRect, insert pairs to the rtree;
  void Insert(const S2LatLngRect& rect, int value) {
    tree_.insert({CreateBox(rect), value});
  }

  // Helper function to query intersecting boxes in the RTree and insert values
  // to the result vector.
  std::size_t FindIntersecting(const S2LatLngRect& rect,
                               std::vector<int>* out) {
    return tree_.query(boost::geometry::index::intersects(CreateBox(rect)),
                       boost::make_function_output_iterator(
                           [&](std::pair<box_t, int> const& value) {
                             out->push_back(value.second);
                           }));
  }

  bool empty() const { return tree_.empty(); }

  rtree_t tree_;
};

RTree::RTree() : impl_(absl::make_unique<RTree::Impl>()) {}

RTree::~RTree() {}

void RTree::Insert(const S2LatLngRect& rect, int value) {
  impl_->Insert(rect, value);
}

std::size_t RTree::FindIntersecting(const S2LatLngRect& rect,
                                    std::vector<int>* out) const {
  return impl_->FindIntersecting(rect, out);
}

bool RTree::empty() const { return impl_->empty(); }

}  // namespace streetlearn
