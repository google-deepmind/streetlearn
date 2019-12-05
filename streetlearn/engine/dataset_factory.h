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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_DATASET_FACTORY_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_DATASET_FACTORY_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "streetlearn/engine/dataset.h"

namespace streetlearn {

// Creates a Dataset object by parsing the passed url like path and
// instantiating a Dataset child class capable of handling the dataset at that
// path.
std::unique_ptr<Dataset> CreateDataset(absl::string_view dataset_url);

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_DATASET_FACTORY_H_
