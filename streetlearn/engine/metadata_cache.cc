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

#include "streetlearn/engine/metadata_cache.h"

#include <fstream>
#include <iostream>

#include "streetlearn/engine/logging.h"
#include "absl/memory/memory.h"

namespace streetlearn {

namespace {

void PreprocessPano(Pano* pano) {
  for (auto& neighbor : *pano->mutable_neighbor()) {
    if (neighbor.has_snapped_coords()) {
      *neighbor.mutable_coords() = neighbor.snapped_coords();
    }
  }
}

void PreprocessGraph(StreetLearnGraph* graph) {
  for (auto& pano : *graph->mutable_pano()) {
    PreprocessPano(&pano);
  }
}

}  // namespace

std::unique_ptr<MetadataCache> MetadataCache::Create(const Dataset* dataset,
                                                     int min_graph_depth) {
  auto metadataCache =
      absl::WrapUnique<MetadataCache>(new MetadataCache(min_graph_depth));
  if (!metadataCache->ReadData(dataset)) {
    return nullptr;
  }
  return metadataCache;
}

MetadataCache::MetadataCache(int min_graph_depth)
    : min_lat_(-1),
      min_lng_(-1),
      max_lat_(-1),
      max_lng_(-1),
      min_graph_depth_(min_graph_depth) {}

bool MetadataCache::ReadData(const Dataset* dataset) {
  StreetLearnGraph streetlearn_graph;
  if (!dataset->GetGraph(&streetlearn_graph)) {
    LOG(ERROR) << "Cannot read StreetLearn graph!";
    return false;
  }

  PreprocessGraph(&streetlearn_graph);

  // Min and max coords.
  const auto& min_coords = streetlearn_graph.min_coords();
  min_lat_ = min_coords.lat();
  min_lng_ = min_coords.lng();
  const auto& max_coords = streetlearn_graph.max_coords();
  max_lat_ = max_coords.lat();
  max_lng_ = max_coords.lng();

  // Panos - for full metadata.
  std::map<std::string, Pano> panos;
  for (const auto& pano : streetlearn_graph.pano()) {
    auto it_inserted = panos.insert({pano.id(), pano});
    if (!it_inserted.second) {
      LOG(ERROR) << "Pano " << pano.id() << " has two metadata entries!";
    }
  }

  ProcessNeighbors(streetlearn_graph, panos);

  return true;
}

void MetadataCache::ProcessNeighbors(const StreetLearnGraph& streetlearn_graph,
                                     const std::map<std::string, Pano>& panos) {
  for (const auto& neighbor : streetlearn_graph.connection()) {
    if (neighbor.subgraph_size() < min_graph_depth_) {
      continue;
    }
    const auto& pano_id = neighbor.id();

    const auto& pano_iter = panos.find(pano_id);
    if (pano_iter == panos.end()) {
      return;
    }
    const auto& pano = pano_iter->second;

    PanoMetadata metadata;
    metadata.pano = pano;
    metadata.graph_depth = neighbor.subgraph_size();

    for (const auto& neighbor_id : neighbor.neighbor()) {
      const auto& neighbor_iter = panos.find(neighbor_id);
      CHECK(neighbor_iter != panos.end());
      const auto& neighbor = neighbor_iter->second;

      PanoIdAndGpsCoords panoIdAndCoords;
      panoIdAndCoords.set_id(neighbor_id);
      LatLng* coords = panoIdAndCoords.mutable_coords();
      coords->set_lat(neighbor.coords().lat());
      coords->set_lng(neighbor.coords().lng());
      metadata.neighbors.push_back(panoIdAndCoords);
    }
    pano_metadata_.emplace(pano_id, metadata);
  }
}

const PanoMetadata* MetadataCache::GetPanoMetadata(
    const std::string& pano_id) const {
  auto cache_iter = pano_metadata_.find(pano_id);
  if (cache_iter != pano_metadata_.end()) {
    return &cache_iter->second;
  }

  return nullptr;
}

std::vector<std::string> MetadataCache::PanosInGraphsOfSize(
    int min_size) const {
  std::vector<std::string> pano_ids;
  for (const auto& entry : pano_metadata_) {
    if (entry.second.graph_depth >= min_size) {
      pano_ids.push_back(entry.first);
    }
  }
  return pano_ids;
}

std::string MetadataCache::PanoInLargestGraph() const {
  return std::max_element(pano_metadata_.begin(), pano_metadata_.end(),
                          [](const std::pair<std::string, PanoMetadata>& p1,
                             const std::pair<std::string, PanoMetadata>& p2) {
                            return p1.second.graph_depth <
                                   p2.second.graph_depth;
                          })
      ->second.pano.id();
}

}  // namespace streetlearn
