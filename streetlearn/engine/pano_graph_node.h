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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_PANO_GRAPH_NODE_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_PANO_GRAPH_NODE_H_

#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "streetlearn/engine/image.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// Holds metadata and de-compressed image for a pano.
class PanoGraphNode {
 public:
  PanoGraphNode() = default;
  explicit PanoGraphNode(const Pano& pano);

  // The ID of the pano.
  std::string id() const { return pano_->id(); }

  // The date on which the pano was recorded.
  std::string date() const { return pano_->pano_date(); }

  // The latitude at which the pano was recorded.
  double latitude() const { return pano_->coords().lat(); }

  // The longitude at which the pano was recorded.
  double longitude() const { return pano_->coords().lng(); }

  // The camera bearing at which the pano was recorded.
  double bearing() const { return pano_->heading_deg(); }

  // Returns the node's image.
  const std::shared_ptr<Image3_b> image() const { return image_; }

  // Returns the metadata.
  std::shared_ptr<Pano> GetPano() const { return pano_; }

 private:
  // The protocol buffer containing the pano's data.
  std::shared_ptr<Pano> pano_;

  // The de-compressed image for the pano.
  std::shared_ptr<Image3_b> image_;
};

// A struct representing a neighbor bearing.
struct PanoNeighborBearing {
  std::string pano_id;
  double bearing;
  double distance;
};

// Struct that holds the distance in meters and the bearing in degrees from the
// current position to a terminal pano.
struct TerminalGraphNode {
  TerminalGraphNode(double distance, double bearing)
      : distance(distance), bearing(bearing) {}

  double distance;
  double bearing;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_PANO_GRAPH_NODE_H_
