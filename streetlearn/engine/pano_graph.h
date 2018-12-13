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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_PANO_GRAPH_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_PANO_GRAPH_H_

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/synchronization/blocking_counter.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "streetlearn/engine/dataset.h"
#include "streetlearn/engine/metadata_cache.h"
#include "streetlearn/engine/node_cache.h"
#include "streetlearn/engine/pano_graph_node.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// Class that represents a graph of PanoGraphNodes. It provides methods for
// creating random graphs and those containing given panos, as well as
// methods for navigating and querying the graph.
// The graph is stored as a lookup from pano ID to metadata which includes
// neighbor information. The PanoGraphNodes are loaded asyncronously on demand,
// as loading and decompressing the imagery is a costly operation.
class PanoGraph {
 public:
  // prefetch_depth: the depth of the node prefetch neighborhood.
  // max_cache_size: the maximum size of the node cache.
  // min_graph_depth: the minimum depth of graphs required.
  // max_graph_depth: the maximum depth of graphs required.
  // dataset: StreetLearn dataset wrapper.
  PanoGraph(int prefetch_depth, int max_cache_size, int min_graph_depth,
            int max_graph_depth, const Dataset* dataset);
  ~PanoGraph();

  // Loads metadata for the region. Returns false if metadata is not available.
  bool Init();

  // Initialises the random number generator.
  void SetRandomSeed(int random_seed);

  // Builds a graph with the given pano as root. Returns false if the pano
  // supplied is not in the graph.
  bool BuildGraphWithRoot(const std::string& pano_id);

  // Builds a graph between min_graph_depth_ and max_graph_depth_, chosen
  // randomly from those available. Returns false if none are available.
  bool BuildRandomGraph();

  // Builds the largest connected graph in the region. Returns false if no
  // graphs are available.
  bool BuildEntireGraph();

  // Sets the min and max graph depth
  void SetGraphDepth(int min_depth, int max_depth);

  // Teleports to the given pano, if it's in the current graph. Returns false
  // if the teleport is not possible.
  bool SetPosition(const std::string& pano_id);

  // Gets the image from the root node.
  std::shared_ptr<Image3_b> RootImage() const;

  // Moves to neighbor if there is one within tolerance of our bearing. Returns
  // false if the move is not possible.
  bool MoveToNeighbor(double current_bearing, double tolerance);

  // Returns the bearings of all neighbors of the root node in the graph.
  std::vector<PanoNeighborBearing> GetNeighborBearings(
      int max_neighbor_depth) const;

  // Returns all terminal neighbors of the node with their distance in numbers
  // of nodes, up to two hops from the current node.
  std::map<int, std::vector<TerminalGraphNode>> TerminalBearings(
      const std::string& node_id) const;

  // Returns the distance in meters between two panos.
  absl::optional<double> GetPanoDistance(const std::string& pano_id1,
                                         const std::string& pano_id2);

  // Returns the bearing in degrees between two panos.
  absl::optional<double> GetPanoBearing(const std::string& pano_id1,
                                        const std::string& pano_id2);

  // Returns the current root node.
  const PanoGraphNode Root() const { return root_node_; }

  // Returns metadata for the node identified by pano_id. Returns false if the
  // node is not in the graph.
  bool Metadata(const std::string& pano_id, PanoMetadata* metadata) const;

  // Returns a map of pano_ids to their neighbors in the current graph.
  std::map<std::string, std::vector<std::string>> GetGraph() const;

 private:
  // Returns metadata for a pano in the node tree. Returns nullptr if not found.
  const PanoMetadata* GetNodeMetadata(const std::string& pano_id) const;

  // Returns terminal bearings for nodes inside the graph.
  std::map<int, std::vector<TerminalGraphNode>> InteriorTerminalBearings(
      const PanoMetadata& node) const;

  // Prefetches a section of graph up to prefetch_depth_ from the new root.
  void PrefetchGraph(const std::string& new_root);

  // Builds the node_tree_ of the entire graph or to the required depth.
  enum class GraphRegion { kDepthBounded, kEntire };
  bool BuildGraph(const std::string& root_node_id, GraphRegion graph_region);

  // Gets the node with the given id synchronously.
  PanoGraphNode FetchNode(const std::string& node_id);

  // Gets the node with the given id asynchronously.
  void PrefetchNode(const std::string& node_id);

  // Cancels any ongoing fetch requests.
  void CancelPendingFetches();

  const Dataset* dataset_;

  // Limits to the depth of graphs built.
  int min_graph_depth_;
  int max_graph_depth_;

  // The depth of asynchronous prefetch.
  const int prefetch_depth_;

  // The ID of the root node.
  std::string root_id_;

  // The root node of the graph.
  PanoGraphNode root_node_;

  // Metadata reader and cache.
  std::unique_ptr<MetadataCache> metadata_cache_;

  // Counts unfinished prefetch requests.
  std::unique_ptr<absl::BlockingCounter> blockingCounter_;

  // Node cache - owns the nodes.
  NodeCache node_cache_;

  // Node tree - gets built once when a graph is constructed.
  std::map<std::string, PanoMetadata> node_tree_;

  // Neighbors of lead nodes that lie outside the graph.
  std::map<std::string, PanoMetadata> terminal_nodes_;

  // Nodes currently being requested - changes every time the root node changes.
  std::vector<std::string> current_subtree_;

  // Used for choosing random panos when no root ID is supplied.
  std::mt19937 random_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_PANO_GRAPH_H_
