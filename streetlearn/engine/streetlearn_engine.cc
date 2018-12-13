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

#include "streetlearn/engine/streetlearn_engine.h"

#include <functional>

#include "streetlearn/engine/logging.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "streetlearn/engine/bitmap_util.h"
#include "streetlearn/engine/math_util.h"

namespace streetlearn {
namespace {

constexpr int kDefaultPrefetchGraphDepth = 20;
constexpr int kDefaultMaxNeighborDepth = 3;
constexpr bool kShowStopSigns = true;
constexpr double kTolerance = 30;
constexpr double kMaxYaw = 180;
constexpr double kMaxPitch = 90;

// Constrain angle to be in [-constraint, constraint].
void ConstrainAngle(double constraint, double* angle) {
  while (*angle < -constraint) {
    *angle += 2.0 * constraint;
  }
  while (*angle > constraint) {
    *angle -= 2.0 * constraint;
  }
}

}  // namespace

std::unique_ptr<StreetLearnEngine> StreetLearnEngine::Create(
    const std::string& data_path, int width, int height, int graph_width,
    int graph_height, int status_height, int field_of_view, int min_graph_depth,
    int max_graph_depth, int max_cache_size) {
  auto dataset = Dataset::Create(data_path);
  if (!dataset) {
    return nullptr;
  }

  return absl::make_unique<StreetLearnEngine>(
      std::move(dataset), Vector2_i(width, height),
      Vector2_i(graph_width, graph_height), status_height, field_of_view,
      min_graph_depth, max_graph_depth, max_cache_size);
}

StreetLearnEngine::StreetLearnEngine(std::unique_ptr<Dataset> dataset,
                                     const Vector2_i& pano_size,
                                     const Vector2_i& graph_size,
                                     int status_height, int field_of_view,
                                     int min_graph_depth, int max_graph_depth,
                                     int max_cache_size)
    : dataset_(std::move(dataset)),
      pano_size_(pano_size),
      graph_size_(graph_size),
      show_stop_signs_(kShowStopSigns),
      rotation_yaw_(0),
      rotation_pitch_(0),
      field_of_view_(field_of_view),
      pano_buffer_(3 * pano_size.x() * pano_size.y()),
      graph_buffer_(3 * graph_size.x() * graph_size.y()),
      pano_graph_(kDefaultPrefetchGraphDepth, max_cache_size, min_graph_depth,
                  max_graph_depth, dataset_.get()),
      pano_renderer_(pano_size.x(), pano_size.y(), status_height,
                     field_of_view) {
  // TODO(b/117756079): Init return value unused.
  pano_graph_.Init();
}

void StreetLearnEngine::InitEpisode(int episode_index, int random_seed) {
  absl::Hash<std::tuple<int, int>> hasher;
  int seed = hasher(std::make_tuple(episode_index, random_seed));
  pano_graph_.SetRandomSeed(seed);
}

std::string StreetLearnEngine::SetupCurrentGraph() {
  rotation_yaw_ = 0;
  rotation_pitch_ = 0;
  return pano_graph_.Root().id();
}

absl::optional<std::string> StreetLearnEngine::BuildRandomGraph() {
  if (!pano_graph_.BuildRandomGraph()) {
    return absl::nullopt;
  }
  return SetupCurrentGraph();
}

absl::optional<std::string> StreetLearnEngine::BuildGraphWithRoot(
    const std::string& pano_id) {
  if (!pano_graph_.BuildGraphWithRoot(pano_id)) {
    return absl::nullopt;
  }
  return SetupCurrentGraph();
}

void StreetLearnEngine::SetGraphDepth(const int min_depth,
                                      const int max_depth) {
  pano_graph_.SetGraphDepth(min_depth, max_depth);
}

absl::optional<std::string> StreetLearnEngine::BuildEntireGraph() {
  if (!pano_graph_.BuildEntireGraph()) {
    return absl::nullopt;
  }
  return SetupCurrentGraph();
}

absl::optional<std::string> StreetLearnEngine::SetPosition(
    const std::string& pano_id) {
  if (!pano_graph_.SetPosition(pano_id)) {
    return absl::nullopt;
  }
  return SetupCurrentGraph();
}

absl::optional<std::string> StreetLearnEngine::MoveToNextPano() {
  if (!pano_graph_.MoveToNeighbor(rotation_yaw_, kTolerance)) {
    return absl::nullopt;
  }
  return pano_graph_.Root().id();
}

void StreetLearnEngine::RotateObserver(double yaw_deg, double pitch_deg) {
  rotation_yaw_ += yaw_deg;
  ConstrainAngle(kMaxYaw, &rotation_yaw_);
  rotation_pitch_ -= pitch_deg;
  ConstrainAngle(kMaxPitch, &rotation_pitch_);
}

void StreetLearnEngine::RenderScene() {
  const auto& neighbor_bearings =
      pano_graph_.GetNeighborBearings(kDefaultMaxNeighborDepth);
  std::map<int, std::vector<TerminalGraphNode>> terminal_bearings;
  if (show_stop_signs_) {
    terminal_bearings = pano_graph_.TerminalBearings(pano_graph_.Root().id());
  }
  pano_renderer_.RenderScene(
      *pano_graph_.RootImage(), pano_graph_.Root().bearing(), rotation_yaw_,
      rotation_pitch_, kTolerance, neighbor_bearings, terminal_bearings);
}

absl::Span<const uint8_t> StreetLearnEngine::RenderObservation() {
  RenderScene();
  const auto& pixels = pano_renderer_.Pixels();

  // Convert from packed to planar.
  ConvertRGBPackedToPlanar(pixels.data(), pano_size_.x(), pano_size_.y(),
                           pano_buffer_.data());

  return pano_buffer_;
}

void StreetLearnEngine::RenderObservation(absl::Span<uint8_t> buffer) {
  RenderScene();

  const auto& pixels = pano_renderer_.Pixels();
  if (buffer.size() != pixels.size()) {
    LOG(ERROR)
        << "Input buffer is not the right size for rendering. Buffer size: "
        << buffer.size() << " Required size: " << pixels.size();
    return;
  }

  ConvertRGBPackedToPlanar(pixels.data(), pano_size_.x(), pano_size_.y(),
                           pano_buffer_.data());
  std::copy(pano_buffer_.begin(), pano_buffer_.end(), buffer.begin());
}

std::vector<uint8_t> StreetLearnEngine::GetNeighborOccupancy(
    const int resolution) {
  const auto& neighbor_bearings =
      pano_graph_.GetNeighborBearings(kDefaultMaxNeighborDepth);
  std::vector<uint8_t> neighborOccupancy(resolution, 0);

  // For each neighbor in bearings, decide which bin to place it in.
  // Normalize the bearings by subtracting the agent's orientation,
  // then add a fixed offset that is half the size of one bin to make the
  // agent's current orientation be centered in a bin rather than in between.
  double double_offset = 360.0 / static_cast<double>(resolution) / 2.0;
  for (auto bearing : neighbor_bearings) {
    double recentered = bearing.bearing - rotation_yaw_;
    double offset = recentered + double_offset;
    offset -= 360.0 * floor(offset / 360.0);

    int bin = static_cast<int>(offset / 360.0 * resolution);
    neighborOccupancy[bin] = 1;
  }
  return neighborOccupancy;
}

bool StreetLearnEngine::InitGraphRenderer(
    const Color& observer_color,
    const std::map<std::string, streetlearn::Color>& panos_to_highlight) {
  observer_.color = observer_color;
  graph_renderer_ =
      GraphRenderer::Create(pano_graph_, graph_size_, panos_to_highlight);
  return graph_renderer_ != nullptr;
}

bool StreetLearnEngine::DrawGraph(
    const std::map<std::string, streetlearn::Color>& pano_id_to_color,
    absl::Span<uint8> buffer) {
  if (!graph_renderer_) return false;

  observer_.pano_id = pano_graph_.Root().id();
  observer_.yaw_radians = math::DegreesToRadians(rotation_yaw_);
  observer_.fov_yaw_radians = math::DegreesToRadians(field_of_view_);

  if (!graph_renderer_->RenderScene(pano_id_to_color, observer_)) {
    return false;
  }

  graph_renderer_->GetPixels(absl::MakeSpan(graph_buffer_));
  ConvertRGBPackedToPlanar(graph_buffer_.data(), graph_size_.x(),
                           graph_size_.y(), buffer.data());
  return true;
}

bool StreetLearnEngine::SetZoom(double zoom) {
  if (!graph_renderer_) return false;
  return graph_renderer_->SetZoom(zoom);
}

std::shared_ptr<const Pano> StreetLearnEngine::GetPano() const {
  return pano_graph_.Root().GetPano();
}

absl::optional<double> StreetLearnEngine::GetPanoDistance(
    const std::string& pano_id1, const std::string& pano_id2) {
  return pano_graph_.GetPanoDistance(pano_id1, pano_id2);
}

absl::optional<double> StreetLearnEngine::GetPanoBearing(
    const std::string& pano_id1, const std::string& pano_id2) {
  return pano_graph_.GetPanoBearing(pano_id1, pano_id2);
}

absl::optional<PanoMetadata> StreetLearnEngine::GetMetadata(
    const std::string& pano_id) const {
  PanoMetadata metadata;
  if (!pano_graph_.Metadata(pano_id, &metadata)) {
    return absl::nullopt;
  }
  return metadata;
}

std::map<std::string, std::vector<std::string>> StreetLearnEngine::GetGraph()
    const {
  return pano_graph_.GetGraph();
}

}  // namespace streetlearn
