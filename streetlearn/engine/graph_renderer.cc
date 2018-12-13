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

#include "streetlearn/engine/graph_renderer.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

#include "streetlearn/engine/logging.h"
#include "absl/container/node_hash_set.h"
#include "absl/memory/memory.h"
#include "streetlearn/engine/metadata_cache.h"
#include "streetlearn/engine/pano_graph.h"
#include "streetlearn/proto/streetlearn.pb.h"
#include "s2/s1angle.h"

namespace streetlearn {
namespace {

constexpr double kMaxLat = 90.0;
constexpr double kMaxLon = 180.0;
constexpr double kShowNodesLimit = 0.004;
constexpr double kViewConeLength = 0.05;
constexpr int kNodeRadius = 2;
constexpr int kMaxNodes = 5000;

// Extends the bounds to encompass the node.
void AddNodeToBounds(const PanoMetadata& node, double* min_lat, double* max_lat,
                     double* min_lng, double* max_lng) {
  double latitude = node.pano.coords().lat();
  *min_lat = std::min(latitude, *min_lat);
  *max_lat = std::max(latitude, *max_lat);

  double longitude = node.pano.coords().lng();
  *min_lng = std::min(longitude, *min_lng);
  *max_lng = std::max(longitude, *max_lng);
}

// Calculates the bounding box around the panos in the graph.
bool CalculateBounds(const PanoGraph& graph, double* min_lat, double* max_lat,
                     double* min_lng, double* max_lng) {
  std::map<std::string, std::vector<std::string>> graph_nodes =
      graph.GetGraph();
  absl::node_hash_set<std::string> already_done;
  for (const auto& node : graph_nodes) {
    PanoMetadata node_data;
    if (!graph.Metadata(node.first, &node_data)) {
      return false;
    }
    AddNodeToBounds(node_data, min_lat, max_lat, min_lng, max_lng);
    already_done.insert(node.first);

    for (const auto& neighbor_id : node.second) {
      if (already_done.find(neighbor_id) != already_done.end()) {
        continue;
      }
      PanoMetadata neighbor_data;
      if (!graph.Metadata(neighbor_id, &neighbor_data)) {
        return false;
      }
      AddNodeToBounds(neighbor_data, min_lat, max_lat, min_lng, max_lng);
      already_done.insert(node.first);
    }
  }
  return true;
}

}  // namespace

std::unique_ptr<GraphRenderer> GraphRenderer::Create(
    const PanoGraph& graph, const Vector2_i& screen_size,
    const std::map<std::string, Color>& highlight) {
  auto graph_renderer =
      absl::WrapUnique(new GraphRenderer(graph, screen_size, highlight));
  if (!graph_renderer->InitRenderer()) {
    LOG(ERROR) << "Failed to initialize GraphRenderer";
    return nullptr;
  }
  return graph_renderer;
}

GraphRenderer::GraphRenderer(const PanoGraph& graph,
                             const Vector2_i& screen_size,
                             const std::map<std::string, Color>& highlight)
    : screen_size_(screen_size),
      pixel_buffer_(screen_size.x(), screen_size.y()),
      cairo_(pixel_buffer_.pixel(0, 0), screen_size.x(), screen_size.y()),
      pano_graph_(graph),
      image_cache_(screen_size, highlight) {}

bool GraphRenderer::InitRenderer() {
  // Initialise to the extents and calculate bounds.
  double min_lat = kMaxLat;
  double max_lat = -kMaxLat;
  double min_lng = kMaxLon;
  double max_lng = -kMaxLon;
  if (!CalculateBounds(pano_graph_, &min_lat, &max_lat, &min_lng, &max_lng)) {
    return false;
  }
  S2LatLng graph_min = S2LatLng::FromDegrees(min_lat, min_lng);
  S2LatLng graph_max = S2LatLng::FromDegrees(max_lat, max_lng);
  graph_bounds_ = S2LatLngRect(graph_min, graph_max);

  if (!image_cache_.InitCache(pano_graph_, graph_bounds_)) {
    return false;
  }

  if (!BuildRTree()) {
    return false;
  }

  return SetSceneBounds("");
}

bool GraphRenderer::BuildRTree() {
  std::map<std::string, std::vector<std::string>> graph_nodes =
      pano_graph_.GetGraph();
  if (graph_nodes.empty()) {
    return false;
  }
  nodes_.clear();
  nodes_.reserve(graph_nodes.size());

  for (const auto& node : graph_nodes) {
    PanoMetadata node_data;
    if (!pano_graph_.Metadata(node.first, &node_data)) {
      return false;
    }

    nodes_.emplace_back(node.first);
    auto node_coords = S2LatLng::FromDegrees(node_data.pano.coords().lat(),
                                             node_data.pano.coords().lng());
    rtree_.Insert(S2LatLngRect(node_coords, node_coords), nodes_.size() - 1);
  }

  return true;
}

bool GraphRenderer::SetSceneBounds(absl::string_view pano_id) {
  // Center the graph and choose a range in degrees to encompass both bounds.
  double current_zoom = image_cache_.current_zoom();
  double lat_range =
      (graph_bounds_.lat_hi() - graph_bounds_.lat_lo()).degrees() /
      current_zoom;
  double lng_range =
      (graph_bounds_.lng_hi() - graph_bounds_.lng_lo()).degrees() /
      current_zoom;
  double max_range = std::numeric_limits<double>::max();
  if (current_zoom > 1) {
    // If we're zoomed in use a square region on the max dimension.
    max_range = std::max(lat_range, lng_range);
    lat_range = lng_range = max_range * 2;
  }

  double lat_centre, lng_centre;
  if (current_zoom > 1 && !pano_id.empty()) {
    PanoMetadata node_data;
    if (!pano_graph_.Metadata(string(pano_id), &node_data)) {
      return false;
    }
    lat_centre = node_data.pano.coords().lat();
    lng_centre = node_data.pano.coords().lng();
  } else {
    lat_centre =
        (graph_bounds_.lat_hi() + graph_bounds_.lat_lo()).degrees() / 2;
    lng_centre =
        (graph_bounds_.lng_hi() + graph_bounds_.lng_lo()).degrees() / 2;
  }
  graph_centre_ = S2LatLng::FromDegrees(lat_centre, lng_centre);

  S2LatLng new_min = S2LatLng::FromDegrees(lat_centre - lat_range / 2,
                                           lng_centre - lng_range / 2);
  S2LatLng new_max = S2LatLng::FromDegrees(lat_centre + lat_range / 2,
                                           lng_centre + lng_range / 2);

  std::vector<int> nodes;
  if (!rtree_.FindIntersecting(S2LatLngRect(new_min, new_max), &nodes)) {
    return false;
  }

  current_nodes_.clear();
  if (nodes.size() < kMaxNodes || max_range < kShowNodesLimit) {
    std::transform(nodes.begin(), nodes.end(),
                   std::back_inserter(current_nodes_),
                   [this](int pos) { return this->nodes_[pos]; });
  }

  return true;
}

bool GraphRenderer::SetZoom(double zoom) {
  image_cache_.SetZoom(zoom);
  return SetSceneBounds(current_pano_);
}

void GraphRenderer::GetPixels(absl::Span<uint8_t> rgb_buffer) const {
  constexpr int kRGBPixelSize = 3;
  constexpr int kBGRAPixelSize = 4;
  CHECK_EQ(rgb_buffer.size(),
           pixel_buffer_.width() * pixel_buffer_.height() * kRGBPixelSize);

  absl::Span<const uint8_t> pixel_data = pixel_buffer_.data();

  int total_size =
      pixel_buffer_.width() * pixel_buffer_.height() * kRGBPixelSize;
  int pixel_offset = 0;
  int out_offset = 0;
  while (out_offset < total_size) {
    rgb_buffer[out_offset++] = pixel_data[pixel_offset + 2];
    rgb_buffer[out_offset++] = pixel_data[pixel_offset + 1];
    rgb_buffer[out_offset++] = pixel_data[pixel_offset];
    pixel_offset += kBGRAPixelSize;
  }
}

bool GraphRenderer::RenderScene(
    const std::map<std::string, Color>& pano_id_to_color,
    const absl::optional<Observer>& observer) {
  if (observer && observer->pano_id != current_pano_) {
    current_pano_ = observer->pano_id;
    if (!SetSceneBounds(observer->pano_id)) {
      return false;
    }
  }

  // Draw the background of graph edges.
  DrawGraphEdges();

  // Draw highlighted nodes.
  for (const auto& node : pano_id_to_color) {
    PanoMetadata metadata;
    if (!pano_graph_.Metadata(node.first, &metadata)) {
      return false;
    }
    Vector2_d coords = MapCoordinates(metadata);
    DrawPosition(coords, node.second);
  }

  // Draw the observer cone.
  if (observer) {
    PanoMetadata metadata;
    if (!pano_graph_.Metadata(observer->pano_id, &metadata)) {
      return false;
    }
    DrawObserver(metadata, *observer);
  }

  return true;
}

void GraphRenderer::DrawObserver(const PanoMetadata& metadata,
                                 const Observer& observer) {
  Vector2_d coords = MapCoordinates(metadata);
  double min_bearing = observer.yaw_radians - observer.fov_yaw_radians / 2;
  double max_bearing = observer.yaw_radians + observer.fov_yaw_radians / 2;
  int cone_size = kViewConeLength * screen_size_.x();
  Vector2_d coord1 = {coords.x() + cone_size * sin(min_bearing),
                      coords.y() - cone_size * cos(min_bearing)};
  Vector2_d coord2 = {coords.x() + cone_size * sin(max_bearing),
                      coords.y() - cone_size * cos(max_bearing)};
  cairo_move_to(cairo_.context(), coord1.x(), coord1.y());
  cairo_line_to(cairo_.context(), coords.x(), coords.y());
  cairo_line_to(cairo_.context(), coord2.x(), coord2.y());
  cairo_close_path(cairo_.context());
  cairo_set_source_rgba(cairo_.context(), observer.color.red,
                        observer.color.green, observer.color.blue, 0.5);
  cairo_fill(cairo_.context());
  DrawEdge(coords, coord1, observer.color);
  DrawEdge(coords, coord2, observer.color);
  DrawPosition(coords, observer.color);
}

void GraphRenderer::DrawPosition(const Vector2_d& coords, const Color& color) {
  cairo_set_source_rgb(cairo_.context(), color.red, color.green, color.blue);
  cairo_arc(cairo_.context(), coords.x(), coords.y(), kNodeRadius, 0, 2 * M_PI);
  cairo_fill(cairo_.context());
}

void GraphRenderer::DrawGraphEdges() {
  double current_zoom = image_cache_.current_zoom();
  auto pixels = image_cache_.Pixels(graph_centre_);
  int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32,
                                             screen_size_.x() * current_zoom);
  cairo_surface_t* surf = cairo_image_surface_create_for_data(
      pixels.pixel(0, 0), CAIRO_FORMAT_ARGB32, screen_size_.x(),
      screen_size_.y(), stride);

  cairo_set_source_surface(cairo_.context(), surf, 0, 0);
  cairo_surface_flush(surf);
  cairo_paint(cairo_.context());
  cairo_surface_destroy(surf);
}

void GraphRenderer::DrawEdge(const Vector2_d& start, const Vector2_d& end,
                             const Color& color) {
  cairo_set_source_rgb(cairo_.context(), color.red, color.green, color.blue);
  cairo_set_line_width(cairo_.context(), 0.5);
  cairo_move_to(cairo_.context(), start.x(), start.y());
  cairo_line_to(cairo_.context(), end.x(), end.y());
  cairo_stroke(cairo_.context());
}

Vector2_d GraphRenderer::MapCoordinates(const PanoMetadata& node) {
  return image_cache_.MapToScreen(S2LatLng::FromDegrees(
      node.pano.coords().lat(), node.pano.coords().lng()));
}

}  // namespace streetlearn
