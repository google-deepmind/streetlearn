// Copyright 2019 Google LLC
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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_DATASET_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_DATASET_H_

#include "absl/strings/string_view.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// Interface class for reading datasets that may be stored in different formats
// or on different mediums. Implmenentations of the Dataset interface must
// support instantiation using the DatasetFactory.
class Dataset {
 public:
  virtual ~Dataset() = default;

  // Gets the connectivity graph from the dataset. Returns false if
  // unsuccessful.
  virtual bool GetGraph(StreetLearnGraph* graph) const = 0;

  // Gets a Pano associated with `pano_id` from the dataset. Returns false if
  // the `pano_id` is unknown.
  virtual bool GetPano(absl::string_view pano_id, Pano* pano) const = 0;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_DATASET_H_
