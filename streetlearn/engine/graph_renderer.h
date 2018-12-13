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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_RENDERER_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_RENDERER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include <cairo/cairo.h>
#include "streetlearn/engine/cairo_util.h"
#include "streetlearn/engine/color.h"
#include "streetlearn/engine/graph_image_cache.h"
#include "streetlearn/engine/image.h"
#include "streetlearn/engine/metadata_cache.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/rtree_helper.h"
#include "streetlearn/engine/vector.h"
#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"

namespace streetlearn {

// Information for drawing an observer view cone. The cone is drawn at a bearing
// of yaw_radians for an angle fov_yaw_radians in the color indicated.
struct Observer {
  std::string pano_id;
  Color color;
  double yaw_radians = 0.0;
  double fov_yaw_radians = 0.0;
};

// Draws a graphical representation of a pano graph. Nodes are marked as filled
// circles and connected with edges drawn in grey. The Observer allows the
// current position and field of view to be shown.
class GraphRenderer {
 public:
  static std::unique_ptr<GraphRenderer> Create(
      const PanoGraph& graph, const Vector2_i& screen_size,
      const std::map<std::string, Color>& highlight);

  GraphRenderer(const GraphRenderer&) = delete;
  GraphRenderer& operator=(const GraphRenderer&) = delete;

  // Sets up the graph bounds and observer cone size.
  bool InitRenderer();

  // Renders the scene into an internal screen buffer. Nodes are draw using the
  // default color unless in pano_id_to_color in which case the associated color
  // is used. The graph is drawn to the depth indicated.
  bool RenderScene(const std::map<std::string, Color>& pano_id_to_color,
                   const absl::optional<Observer>& observer);

  // Sets the current zoom.
  bool SetZoom(double zoom);

  // Draws the pixel buffer int `rgb_buffer`. The color order is RGB. Caller of
  // this function has to make sure that the size of `rgb_buffer` exactly
  // matches the size of the image.
  void GetPixels(absl::Span<uint8_t> rgb_buffer) const;

 private:
  GraphRenderer(const PanoGraph& graph, const Vector2_i& screen_size,
                const std::map<std::string, Color>& highlight);

  // Builds an RTree of graph nodes.
  bool BuildRTree();

  // Calculates lat/lon bounds for the scene and the mapping factors to screen
  // coordinates centered on the pano provided.
  bool SetSceneBounds(absl::string_view pano_id);

  // Marks the current observer position and orientation using a cone that
  // extends to three neighboring panos.
  void DrawObserver(const PanoMetadata& metadata, const Observer& observer);

  // Draws the background of graph edges.
  void DrawGraphEdges();

  // Draw an edge between two points. Used for the observer cone.
  void DrawEdge(const Vector2_d& start, const Vector2_d& end,
                const Color& color);

  // Draws a filled circle of the given radius in the color provided.
  void DrawPosition(const Vector2_d& coords, const Color& color);

  // Transform the node coordinates from lat/lon to screen.
  Vector2_d MapCoordinates(const PanoMetadata& node);

  // The output screen size.
  const Vector2_i screen_size_;

  // The graph bounds in degrees.
  S2LatLngRect graph_bounds_;

  // The user's current location in the graph.
  S2LatLng graph_centre_;

  // The pano currently being displayed.
  std::string current_pano_;

  // The pixel buffer is used to hold the current render.
  Image4_b pixel_buffer_;

  // Wrapper around Cairo surface creation.
  CairoRenderHelper cairo_;

  // The graph this object renders.
  const PanoGraph& pano_graph_;

  // RTree of point rects for all nodes in the graph, associated to indices in
  // the nodes_ collection below. Keys in the RTree need to be POD types.
  RTree rtree_;

  // Pano IDs of the nodes in the RTree.
  std::vector<std::string> nodes_;

  // Nodes in the current view.
  std::vector<std::string> current_nodes_;

  // Cache of images of the graph edges.
  GraphImageCache image_cache_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_GRAPH_RENDERER_H_
