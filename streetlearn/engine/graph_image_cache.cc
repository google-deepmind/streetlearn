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

#include "streetlearn/engine/graph_image_cache.h"

#include <cmath>
#include <string>

#include "streetlearn/engine/logging.h"
#include "absl/container/node_hash_set.h"
#include <cairo/cairo.h>
#include "streetlearn/engine/cairo_util.h"
#include "streetlearn/engine/color.h"
#include "streetlearn/engine/math_util.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/engine/vector.h"

namespace streetlearn {
namespace {

constexpr int kMinZoom = 1;
constexpr int kMaxZoom = 32;
constexpr int kNodeZoomCutoff = 8;
constexpr int kNodeRadius = 2;
constexpr double kLineWidth = 0.5;

// All color vectors interpreted as {r, g, b}
Color BackColor() { return {0, 0, 0}; }
Color NodeColor() { return {1, 1, 1}; }
Color EdgeColor() { return {0.4, 0.4, 0.4}; }

}  // namespace

bool GraphImageCache::InitCache(const PanoGraph& pano_graph,
                                const S2LatLngRect& graph_bounds) {
  image_cache_.clear();
  region_manager_.SetGraphBounds(graph_bounds);
  if (!SetGraph(pano_graph)) {
    return false;
  }

  // Create and cache the top-level image.
  region_manager_.ResetCurrentBounds(1.0 /* zoom factor */);
  RenderLevel();
  return true;
}

void GraphImageCache::SetZoom(double zoom) {
  CHECK_GE(zoom, kMinZoom);
  CHECK_LE(zoom, kMaxZoom);
  current_zoom_ = zoom;

  // Create the image for the level if it's not in the cache.
  if (image_cache_.find(current_zoom_) == image_cache_.end()) {
    region_manager_.ResetCurrentBounds(current_zoom_);
    RenderLevel();
  }
}

// ImageView4_b GraphImageCache::Pixels(const S2LatLng& image_centre) {
ImageView4_b GraphImageCache::Pixels(const S2LatLng& image_centre) {
  auto it = image_cache_.find(current_zoom_);
  CHECK(it != image_cache_.end());
  auto& current_image = it->second;

  int image_width = screen_size_.x() * current_zoom_;
  int image_height = screen_size_.y() * current_zoom_;
  region_manager_.SetCurrentBounds(current_zoom_, image_centre);

  Vector2_d centre = region_manager_.MapToBuffer(image_centre.lat().degrees(),
                                                 image_centre.lng().degrees(),
                                                 image_width, image_height);

  // Constrain the limits to be within the image.
  int left = static_cast<int>(centre.x() - screen_size_.x() / 2);
  left = math::Clamp(0, image_width - screen_size_.x(), left);
  int top = static_cast<int>(centre.y() - screen_size_.y() / 2);
  top = math::Clamp(0, image_height - screen_size_.y(), top);

  return ImageView4_b(&current_image, left, top, screen_size_.x(),
                      screen_size_.y());
}

void GraphImageCache::RenderLevel() {
  int image_width = screen_size_.x() * current_zoom_;
  int image_height = screen_size_.y() * current_zoom_;
  auto it =
      image_cache_.emplace(current_zoom_, Image4_b(image_width, image_height))
          .first;
  CairoRenderHelper cairo(it->second.pixel(0, 0), it->second.width(),
                          it->second.height());

  // Draw the background.
  auto back_color = BackColor();
  cairo_set_source_rgb(cairo.context(), back_color.red, back_color.green,
                       back_color.blue);
  cairo_paint(cairo.context());

  auto edge_color = EdgeColor();
  cairo_set_source_rgb(cairo.context(), edge_color.red, edge_color.green,
                       edge_color.blue);
  cairo_set_line_width(cairo.context(), kLineWidth);

  // Draw all the edges.
  DrawEdges(image_width, image_height, cairo.context());

  // Draw highlighted nodes if sufficiently zoomed in.
  if (current_zoom_ >= kNodeZoomCutoff) {
    DrawNodes(image_width, image_height, cairo.context());
  }
}

void GraphImageCache::DrawEdges(int image_width, int image_height,
                                cairo_t* context) {
  for (const auto& edge : graph_edges_) {
    Vector2_d start = region_manager_.MapToBuffer(edge.start.lat().degrees(),
                                                  edge.start.lng().degrees(),
                                                  image_width, image_height);
    Vector2_d end = region_manager_.MapToBuffer(edge.end.lat().degrees(),
                                                edge.end.lng().degrees(),
                                                image_width, image_height);

    cairo_move_to(context, start.x(), start.y());
    cairo_line_to(context, end.x(), end.y());
    cairo_stroke(context);
  }
}

void GraphImageCache::DrawNodes(int image_width, int image_height,
                                cairo_t* context) {
  // Draw the nodes
  for (const auto& node : nodes_) {
    Vector2_d coords = region_manager_.MapToBuffer(node.second.lat().degrees(),
                                                   node.second.lng().degrees(),
                                                   image_width, image_height);
    auto iter = highlighted_nodes_.find(node.first);
    const Color& color =
        iter != highlighted_nodes_.end() ? iter->second : NodeColor();
    cairo_set_source_rgb(context, color.red, color.green, color.blue);
    cairo_arc(context, coords.x(), coords.y(), kNodeRadius, 0, 2 * M_PI);
    cairo_fill(context);
  }
}

bool GraphImageCache::SetGraph(const PanoGraph& pano_graph) {
  std::map<std::string, std::vector<std::string>> graph_nodes =
      pano_graph.GetGraph();

  // Populate the edge cache.
  absl::node_hash_set<std::string> already_done;
  for (const auto& node : graph_nodes) {
    PanoMetadata node_data;
    if (!pano_graph.Metadata(node.first, &node_data)) {
      return false;
    }
    double start_lat = node_data.pano.coords().lat();
    double start_lng = node_data.pano.coords().lng();
    already_done.insert(node.first);

    auto start = S2LatLng::FromDegrees(start_lat, start_lng);
    nodes_[node.first] = start;

    for (const auto& neighbor_id : node.second) {
      if (already_done.find(neighbor_id) != already_done.end()) {
        continue;
      }
      PanoMetadata neighbor_data;
      if (!pano_graph.Metadata(neighbor_id, &neighbor_data)) {
        return false;
      }

      double end_lat = neighbor_data.pano.coords().lat();
      double end_lng = neighbor_data.pano.coords().lng();
      graph_edges_.emplace_back(start, S2LatLng::FromDegrees(end_lat, end_lng));
    }
  }

  return true;
}

}  // namespace streetlearn
