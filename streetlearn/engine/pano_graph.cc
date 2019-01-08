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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <utility>

#include "streetlearn/engine/logging.h"
#include "absl/synchronization/notification.h"
#include "streetlearn/engine/math_util.h"
#include "streetlearn/engine/metadata_cache.h"
#include "streetlearn/engine/pano_calculations.h"

namespace streetlearn {
namespace {

constexpr int kThreadCount = 16;
constexpr int kMinGraphDepth = 10;
constexpr double kNeighborAltituteThresholdMeters = 2.0;

}  // namespace

PanoGraph::PanoGraph(int prefetch_depth, int max_cache_size,
                     int min_graph_depth, int max_graph_depth,
                     const Dataset* dataset)
    : dataset_(dataset),
      min_graph_depth_(min_graph_depth),
      max_graph_depth_(max_graph_depth),
      prefetch_depth_(prefetch_depth),
      node_cache_(dataset, kThreadCount, max_cache_size) {}

PanoGraph::~PanoGraph() { CancelPendingFetches(); }

bool PanoGraph::Init() {
  metadata_cache_ = MetadataCache::Create(dataset_, kMinGraphDepth);
  if (!metadata_cache_) {
    LOG(ERROR) << "Unable to initialize Pano graph";
    return false;
  }
  return true;
}

void PanoGraph::SetRandomSeed(int random_seed) { random_.seed(random_seed); }

bool PanoGraph::BuildGraphWithRoot(const std::string& pano_id) {
  if (pano_id.empty()) {
    LOG(ERROR) << "No pano ID has been specified" << std::endl;
    return false;
  }

  if (!BuildGraph(pano_id, GraphRegion::kDepthBounded)) {
    return false;
  }
  PrefetchGraph(pano_id);
  return true;
}

bool PanoGraph::BuildRandomGraph() {
  std::vector<std::string> panos =
      metadata_cache_->PanosInGraphsOfSize(min_graph_depth_);
  if (panos.empty()) {
    LOG(ERROR) << "No graphs available of the required depth between "
               << min_graph_depth_ << " and " << max_graph_depth_;
    return false;
  }

  int pano_index = math::UniformRandomInt(&random_, panos.size() - 1);
  const auto& root_node = panos[pano_index];
  if (!BuildGraph(root_node, GraphRegion::kDepthBounded)) {
    return false;
  }
  PrefetchGraph(root_node);
  return true;
}

bool PanoGraph::BuildEntireGraph() {
  std::string pano = metadata_cache_->PanoInLargestGraph();
  if (!BuildGraph(pano, GraphRegion::kEntire)) {
    return false;
  }

  auto iter = node_tree_.begin();
  std::advance(iter, math::UniformRandomInt(&random_, node_tree_.size() - 1));
  const auto& root_node = iter->first;
  PrefetchGraph(root_node);
  return true;
}

void PanoGraph::SetGraphDepth(int min_depth, int max_depth) {
  min_graph_depth_ = min_depth;
  max_graph_depth_ = max_depth;
}

bool PanoGraph::SetPosition(const std::string& pano_id) {
  if (node_tree_.find(pano_id) == node_tree_.end()) {
    LOG(ERROR) << "Pano " << pano_id << " is not in the current graph";
    return false;
  }

  PrefetchGraph(pano_id);
  return true;
}

void PanoGraph::PrefetchGraph(const std::string& new_root) {
  std::vector<std::string> new_tree;
  std::vector<std::string> current_level = {new_root};
  std::set<std::string> already_in_tree = {new_root};

  // Iterate to the required depth fetching asynchronously and adding metadata
  // for children to the node tree.
  std::vector<std::string> temp_level;
  for (int i = 0; i < prefetch_depth_; ++i) {
    temp_level.clear();

    for (const auto& node_id : current_level) {
      const auto& metadata = node_tree_[node_id];
      new_tree.push_back(node_id);

      for (const auto& neighbour : metadata.neighbors) {
        if (node_tree_.find(neighbour.id()) == node_tree_.end()) {
          continue;
        }

        auto insert_result = already_in_tree.insert(neighbour.id());
        if (insert_result.second) {
          temp_level.push_back(neighbour.id());
        }
      }
    }

    current_level.swap(temp_level);
  }

  // Cancel any outstanding fetches.
  CancelPendingFetches();

  // Fetch the root of the new graph synchronously.
  root_node_ = FetchNode(new_root);

  // Fetch any new nodes required.
  current_subtree_ = std::move(new_tree);
  blockingCounter_ =
      absl::make_unique<absl::BlockingCounter>(current_subtree_.size());
  for (const auto& node_id : current_subtree_) {
    PrefetchNode(node_id);
  }
}

std::shared_ptr<Image3_b> PanoGraph::RootImage() const {
  return root_node_.image();
}

bool PanoGraph::Metadata(const std::string& pano_id,
                         PanoMetadata* metadata) const {
  const PanoMetadata* node_metadata = GetNodeMetadata(pano_id);
  if (!node_metadata) {
    return false;
  }

  *metadata = *node_metadata;
  return true;
}

bool PanoGraph::MoveToNeighbor(double current_bearing, double tolerance) {
  if (node_tree_.empty()) {
    LOG(ERROR) << "Cannot move to a neighbor until the graph has been built!";
    return false;
  }

  // Our current bearing.
  double current_bearing_rads = math::DegreesToRadians(current_bearing);
  if (-M_PI > current_bearing_rads || current_bearing_rads > M_PI) {
    LOG(ERROR) << "current_bearing must be in range [-PI, PI]!";
    return false;
  }
  double tolerance_rads = math::DegreesToRadians(tolerance);

  // Work out which neighbor is closest to the current bearing.
  // TODO: reintroduce max_neighbor_depth.
  auto neighbour_bearings = GetNeighborBearings(3 /* max_neighbor_depth */);

  std::string node_id;
  double min_distance = std::numeric_limits<double>::max();
  for (const auto& neighbour_bearing : neighbour_bearings) {
    double bearing = math::DegreesToRadians(neighbour_bearing.bearing);
    double offset = fabs(current_bearing_rads - bearing);
    // When either side of PI use the smaller angle between them.
    if (offset > M_PI) {
      offset = 2.0 * M_PI - offset;
    }

    if (offset < tolerance_rads && neighbour_bearing.distance < min_distance) {
      min_distance = neighbour_bearing.distance;
      node_id = neighbour_bearing.pano_id;
    }
  }

  if (!node_id.empty()) {
    PrefetchGraph(node_id);
  }

  return true;
}

std::vector<PanoNeighborBearing> PanoGraph::GetNeighborBearings(
    int max_neighbor_depth) const {
  std::vector<PanoNeighborBearing> retval;
  auto root_iter = node_tree_.find(root_node_.id());
  if (root_iter != node_tree_.end()) {
    std::set<std::string> visited;
    visited.insert(root_node_.id());

    const auto& root_metadata = root_iter->second;

    std::vector<std::string> nodes;
    nodes.reserve(root_metadata.neighbors.size());

    auto get_pano_id = [](const PanoIdAndGpsCoords& id_and_coords) {
      return id_and_coords.id();
    };
    std::transform(root_metadata.neighbors.begin(),
                   root_metadata.neighbors.end(), std::back_inserter(nodes),
                   get_pano_id);
    std::vector<std::string> next_nodes;
    for (int depth = 0; depth < max_neighbor_depth; ++depth) {
      for (const auto& node_id : nodes) {
        if (visited.insert(node_id).second) {
          auto iter = node_tree_.find(node_id);
          if (iter != node_tree_.end()) {
            const auto& node = iter->second;
            if (depth == 0 ||
                std::abs(root_metadata.pano.alt() - node.pano.alt()) <
                    kNeighborAltituteThresholdMeters) {
              auto bearing = BearingBetweenPanos(root_metadata.pano, node.pano);
              retval.emplace_back(PanoNeighborBearing{
                  node_id, bearing,
                  DistanceBetweenPanos(root_metadata.pano, node.pano)});
              next_nodes.reserve(next_nodes.size() + node.neighbors.size());
              std::transform(node.neighbors.begin(), node.neighbors.end(),
                             std::back_inserter(next_nodes), get_pano_id);
            }
          }
        }
      }
      nodes.clear();
      nodes.swap(next_nodes);
    }
  }
  return retval;
}

std::map<int, std::vector<TerminalGraphNode>>
PanoGraph::InteriorTerminalBearings(const PanoMetadata& node) const {
  std::map<int, std::vector<TerminalGraphNode>> retval;
  if (node.neighbors.size() == 1) {
    auto neighbor_iter = node_tree_.find(node.neighbors[0].id());
    if (neighbor_iter != node_tree_.end()) {
      // Use the bearing from the neighbor to the current position.
      auto bearing = BearingBetweenPanos(neighbor_iter->second.pano, node.pano);
      auto distance =
          DistanceBetweenPanos(node.pano, neighbor_iter->second.pano) / 4;
      retval[1].emplace_back(distance, bearing);
    }
  }
  return retval;
}

std::map<int, std::vector<TerminalGraphNode>> PanoGraph::TerminalBearings(
    const std::string& node_id) const {
  auto iter = node_tree_.find(node_id);
  if (iter == node_tree_.end()) {
    return {};
  }
  const auto& node = iter->second;

  // 1. Deal with terminal nodes inside the graph
  if (terminal_nodes_.find(node_id) != terminal_nodes_.end()) {
    return InteriorTerminalBearings(node);
  }

  // 2. Deal with terminal nodes outside, cutoff by breadth-first traversal.
  std::map<int, std::vector<TerminalGraphNode>> retval;
  for (const auto& neighbor : node.neighbors) {
    auto terminal_iter = terminal_nodes_.find(neighbor.id());
    if (terminal_iter != terminal_nodes_.end()) {  // Immediate neighbors.
      auto bearing = BearingBetweenPanos(node.pano, terminal_iter->second.pano);
      auto distance =
          DistanceBetweenPanos(node.pano, terminal_iter->second.pano);
      retval[1].emplace_back(distance, bearing);
    } else {
      auto neighbor_iter = node_tree_.find(neighbor.id());
      if (neighbor_iter == node_tree_.end()) {  // Neighbors two panos away.
        continue;
      }
      for (const auto& second_neighbor : neighbor_iter->second.neighbors) {
        if (node_tree_.find(second_neighbor.id()) != node_tree_.end()) {
          continue;  // Terminal node inside the graph.
        }
        auto terminal_iter = terminal_nodes_.find(second_neighbor.id());
        if (terminal_iter != terminal_nodes_.end()) {
          auto bearing =
              BearingBetweenPanos(node.pano, terminal_iter->second.pano);
          auto distance =
              DistanceBetweenPanos(node.pano, terminal_iter->second.pano);
          retval[2].emplace_back(distance, bearing);
        }
      }
    }
  }
  return retval;
}

absl::optional<double> PanoGraph::GetPanoDistance(const std::string& pano_id1,
                                                  const std::string& pano_id2) {
  const PanoMetadata* metadata1 = GetNodeMetadata(pano_id1);
  if (metadata1) {
    const PanoMetadata* metadata2 = GetNodeMetadata(pano_id2);
    if (metadata2) {
      return DistanceBetweenPanos(metadata1->pano, metadata2->pano);
    }
  }

  return absl::nullopt;
}

absl::optional<double> PanoGraph::GetPanoBearing(const std::string& pano_id1,
                                                 const std::string& pano_id2) {
  const PanoMetadata* metadata1 = GetNodeMetadata(pano_id1);
  if (metadata1) {
    const PanoMetadata* metadata2 = GetNodeMetadata(pano_id2);
    if (metadata2) {
      return BearingBetweenPanos(metadata1->pano, metadata2->pano);
    }
  }

  return absl::nullopt;
}

const PanoMetadata* PanoGraph::GetNodeMetadata(
    const std::string& pano_id) const {
  const auto it = node_tree_.find(pano_id);
  if (it == node_tree_.end()) {
    LOG(ERROR) << "Invalid pano id: " << pano_id;
    return nullptr;
  }
  return &it->second;
}

std::map<std::string, std::vector<std::string>> PanoGraph::GetGraph() const {
  std::map<std::string, std::vector<std::string>> retval;
  for (const auto& node : node_tree_) {
    for (const auto& neighbor : node.second.neighbors) {
      // Only include neighbors that are in the graph.
      if (node_tree_.find(neighbor.id()) != node_tree_.end()) {
        retval[node.first].push_back(neighbor.id());
      }
    }
  }
  return retval;
}

bool PanoGraph::BuildGraph(const std::string& root_node_id,
                           GraphRegion graph_region) {
  node_tree_.clear();
  terminal_nodes_.clear();

  // Start at start_node and iterate to max_depth.
  std::vector<std::string> current_level = {root_node_id};
  std::set<std::string> already_in_tree = {root_node_id};

  int max_depth = graph_region == GraphRegion::kEntire
                      ? std::numeric_limits<int>::max()
                      : max_graph_depth_;

  std::vector<std::string> temp_level;
  for (int i = 0; i < max_depth && !current_level.empty(); ++i) {
    temp_level.clear();

    for (const auto& node_id : current_level) {
      auto* metadata = metadata_cache_->GetPanoMetadata(node_id);
      if (metadata == nullptr) {
        LOG(ERROR) << "Unknown pano: " << node_id;
        return false;
      }
      node_tree_.emplace(node_id, *metadata);

      // Nodes with a single neighbour must be terminal.
      if (metadata->neighbors.size() == 1) {
        terminal_nodes_.emplace(node_id, *metadata);
      }

      for (const auto& neighbour : metadata->neighbors) {
        auto insert_result = already_in_tree.insert(neighbour.id());
        if (insert_result.second) {
          temp_level.push_back(neighbour.id());
        }
      }
    }

    current_level.swap(temp_level);
  }

  // Add terminal nodes outside the graph.
  for (const auto& node_id : current_level) {
    auto* metadata = metadata_cache_->GetPanoMetadata(node_id);
    terminal_nodes_.emplace(node_id, *metadata);
  }

  return true;
}

PanoGraphNode PanoGraph::FetchNode(const std::string& node_id) {
  PanoGraphNode retval;
  absl::Notification notification;
  node_cache_.Lookup(node_id,
                     [&retval, &notification](const PanoGraphNode* node) {
                       retval = *node;
                       notification.Notify();
                     });
  notification.WaitForNotification();
  return retval;
}

void PanoGraph::PrefetchNode(const std::string& node_id) {
  node_cache_.Lookup(node_id, [this](const PanoGraphNode* node) {
    if (blockingCounter_) {
      blockingCounter_->DecrementCount();
    }
  });
}

void PanoGraph::CancelPendingFetches() {
  node_cache_.CancelPendingFetches();
  if (blockingCounter_) {
    blockingCounter_->Wait();
  }
  blockingCounter_ = nullptr;
}

}  // namespace streetlearn
