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

#ifndef STREETLEARN_STREETLEARN_ENGINE_H_
#define STREETLEARN_STREETLEARN_ENGINE_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "streetlearn/engine/dataset.h"
#include "streetlearn/engine/graph_renderer.h"
#include "streetlearn/engine/math_util.h"
#include "streetlearn/engine/metadata_cache.h"
#include "streetlearn/engine/node_cache.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/pano_graph_node.h"
#include "streetlearn/engine/pano_renderer.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// The controller for building and navigating around StreetLearn graphs.
// Streetlearn pano and metadata should be in the same directory. Each pano
// file needs to use the ID of the pano it contains as its name.
class StreetLearnEngine {
 public:
  // Creates an instance of the StreetLearnEngine.
  // Arguments:
  //   data_path: full path to the directory containing the dataset.
  //   width: width of the street image.
  //   height: height of the street image.
  //   graph_width: width of the map graph image.
  //   graph_height: height of the map graph image.
  //   status_height: height of the status bar at the bottom of the screen.
  //   field_of_view: field of view covered by the screen.
  //   min_graph_depth: minimum depth of graphs created.
  //   max_graph_depth: maximum depth of graphs created.
  //   max_cache_size: maximum capacity of the node cache.
  static std::unique_ptr<StreetLearnEngine> Create(
      const std::string& data_path, int width = 320, int height = 240,
      int graph_width = 320, int graph_height = 240, int status_height = 10,
      int field_of_view = 60, int min_graph_depth = 10,
      int max_graph_depth = 15, int max_cache_size = 1000);

  // Clones an instance of the StreetLearnEngine, that shares the same dataset
  // and node_cache as the original.
  // Arguments:
  //   width: width of the street image.
  //   height: height of the street image.
  //   graph_width: width of the map graph image.
  //   graph_height: height of the map graph image.
  //   status_height: height of the status bar at the bottom of the screen.
  //   field_of_view: field of view covered by the screen.
  //   min_graph_depth: minimum depth of graphs created.
  //   max_graph_depth: maximum depth of graphs created.
  std::unique_ptr<StreetLearnEngine> Clone(
      int width = 320, int height = 240, int graph_width = 320,
      int graph_height = 240, int status_height = 10, int field_of_view = 60,
      int min_graph_depth = 10, int max_graph_depth = 15);

  StreetLearnEngine(std::shared_ptr<Dataset> dataset,
                    std::shared_ptr<NodeCache> node_cache,
                    const Vector2_i& pano_size, const Vector2_i& graph_size,
                    int status_height, int field_of_view, int min_graph_depth,
                    int max_graph_depth);

  // Initialises the random number generator.
  void InitEpisode(int episode_index, int random_seed);

  // Builds a random graph of the required depth bounds, if possible. Returns
  // the root node of the graph if successful.
  absl::optional<std::string> BuildRandomGraph();

  // Builds a graph around the given root. Returns the root node of the graph
  // if successful.
  absl::optional<std::string> BuildGraphWithRoot(const std::string& pano_id);

  // Builds an entire graph for a region.
  absl::optional<std::string> BuildEntireGraph();

  // Sets the current Pano or chooses a random one if no ID is specified.
  absl::optional<std::string> SetPosition(const std::string& pano_id);

  // Moves to the neighbor ahead and if possible returns the new panoID.
  absl::optional<std::string> MoveToNextPano();

  // Sets the min and max graph depth.
  void SetGraphDepth(int min_depth, int max_depth);

  // Rotates the observer by the given amounts.
  void RotateObserver(double yaw_deg, double pitch_deg);

  // Renders and returns the observation of the game at the current timestep.
  absl::Span<const uint8_t> RenderObservation();

  // Renders and returns the observation of the game at the current timestep.
  // The user is responsible for providing a buffer of the correct size.
  void RenderObservation(absl::Span<uint8_t> buffer);

  // Calculates a binary occupancy vector of neighbors to the current pano.
  std::vector<uint8_t> GetNeighborOccupancy(int resolution);

  // Returns the pano for the root of the current graph.
  std::shared_ptr<const Pano> GetPano() const;

  // Returns metadata for the given pano if it's in the current graph.
  absl::optional<PanoMetadata> GetMetadata(const std::string& pano_id) const;

  // Returns a list of panoIDs and their neighbors in the current graph.
  std::map<std::string, std::vector<std::string>> GetGraph() const;

  // Returns the agent's current yaw rotation.
  double GetYaw() const { return rotation_yaw_; }

  // Returns the agent's current pitch rotation.
  double GetPitch() const { return -rotation_pitch_; }

  // Calculates the distance in meters between the two panos.
  absl::optional<double> GetPanoDistance(const std::string& pano_id1,
                                         const std::string& pano_id2);

  // Returns the bearing in degrees between two panos.
  absl::optional<double> GetPanoBearing(const std::string& pano_id1,
                                        const std::string& pano_id2);

  // The shape of the observation tensor.
  std::array<int, 3> ObservationDims() const {
    return {{3, pano_size_.y(), pano_size_.x()}};
  }

  // Creates a new graph renderer with the given panos highlighted. Returns true
  // if graph renderer is initialised correctly.
  bool InitGraphRenderer(
      const Color& observer_color,
      const std::map<std::string, Color>& panos_to_highlight,
      const bool black_on_white);

  // Renders the current graph into the buffer provided. The user is
  // responsible for providing a buffer of the correct size. Returns true is
  // rendering is successful.
  bool DrawGraph(const std::map<std::string, Color>& pano_id_to_color,
                 absl::Span<uint8_t> buffer);

  // Sets the current zoom factor. Returns true if zoom is successfully set.
  bool SetZoom(double zoom);

  // Return the current size of the node cache.
  int GetNodeCacheSize() { return pano_graph_.GetNodeCacheSize(); };

 private:
  // Sets internal state once a graph is loaded.
  std::string SetupCurrentGraph();

  // Renders the current pano view.
  void RenderScene();

  // StreetLearn dataset.
  std::shared_ptr<Dataset> dataset_;

  // Cache of StreetLearn panoramas.
  std::shared_ptr<NodeCache> node_cache_;

  // The width/height of the output pano images.
  Vector2_i pano_size_;

  // The width/height of the output graph images.
  Vector2_i graph_size_;

  // Indicate where the graph has been cutoff.
  bool show_stop_signs_;

  // The agent's current orientation.
  double rotation_yaw_;
  double rotation_pitch_;

  // The agent's field of view.
  double field_of_view_;

  // Buffer in which to do pixel transforms when drawing panos.
  std::vector<uint8_t> pano_buffer_;

  // Buffer in which to do pixel transforms when drawing the graph.
  std::vector<uint8_t> graph_buffer_;

  // The observer used for the view cone when rendering the graph.
  Observer observer_;

  // The graph of panos.
  PanoGraph pano_graph_;

  // Rotates, projects and draws subsections of panos.
  PanoRenderer pano_renderer_;

  // Draws a pictorial representation of graphs.
  std::unique_ptr<GraphRenderer> graph_renderer_;
};

}  // namespace streetlearn

#endif  // STREETLEARN_STREETLEARN_ENGINE_H_
