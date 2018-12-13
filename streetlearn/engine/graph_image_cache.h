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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_IMAGE_CACHE_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_IMAGE_CACHE_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include <cairo/cairo.h>
#include "streetlearn/engine/color.h"
#include "streetlearn/engine/graph_region_mapper.h"
#include "streetlearn/engine/image.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/vector.h"
#include "s2/s1angle.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"

namespace streetlearn {

// Cache of graph images at different levels of zoom showing the graph edges.
// Used by the GraphRenderer which also draws nodes when zoomed in sufficiently.
class GraphImageCache {
 public:
  explicit GraphImageCache(const Vector2_i& screen_size,
                           const std::map<std::string, Color>& highlighted)
      : screen_size_(screen_size),
        region_manager_(screen_size),
        highlighted_nodes_(highlighted) {}

  GraphImageCache(const GraphRegionMapper&) = delete;
  GraphImageCache& operator=(const GraphRegionMapper&) = delete;

  // Populates the edge cache and draws the top level image.
  bool InitCache(const PanoGraph& pano_graph, const S2LatLngRect& graph_bounds);

  // Updates the map region being rendered for the current level at the given
  // centre point. Returns a view into the image buffer at this region.
  ImageView4_b Pixels(const S2LatLng& image_centre);

  // Sets the current zoom and creates the level image if it's not in the cache.
  void SetZoom(double zoom);

  // Returns the current zoom.
  double current_zoom() const { return current_zoom_; }

  // Maps the lat/lng provided to screen coordinates.
  Vector2_d MapToScreen(const S2LatLng& lat_lng) const {
    return region_manager_.MapToScreen(lat_lng.lat().degrees(),
                                       lat_lng.lng().degrees());
  }

 private:
  // Collect all the node and edges in the graph.
  bool SetGraph(const PanoGraph& pano_graph);

  // Creates and caches a graph image for the current level.
  void RenderLevel();

  // Draws an edge between each pair of nodes in the graph.
  void DrawEdges(int image_width, int image_height, cairo_t* context);

  // Draws a filled circle marker for any nodes that need highlighted.
  void DrawNodes(int image_width, int image_height, cairo_t* context);

  // The size of the screen into which to render.
  Vector2_i screen_size_;

  // The current zoom level.
  double current_zoom_ = 1.0;

  // Helper to map lat/lng regions to screen coordinates.
  GraphRegionMapper region_manager_;

  // Representation of a graph edge.
  struct GraphEdge {
    GraphEdge() = default;
    GraphEdge(S2LatLng start, S2LatLng end)
        : start(std::move(start)), end(std::move(end)) {}
    S2LatLng start;
    S2LatLng end;
  };

  // Collection of nodes that need rendered.
  std::map<std::string, S2LatLng> nodes_;

  // Collection of graph edges that need rendered.
  std::vector<GraphEdge> graph_edges_;

  // Collection of nodes that need colored.
  std::map<std::string, Color> highlighted_nodes_;

  // Cache of images at different zoom levels.
  absl::node_hash_map<double, Image4_b> image_cache_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_IMAGE_CACHE_H_
