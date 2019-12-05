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

#include <memory>

#include "absl/strings/string_view.h"
#include "streetlearn/engine/dataset.h"
#include "streetlearn/engine/leveldb_dataset.h"

namespace streetlearn {

std::unique_ptr<Dataset> CreateDataset(absl::string_view dataset_url) {
  // Add support for other dataset formats by eg. parsing `dataset_url`.
  return LevelDBDataset::Create(std::string(dataset_url));
}

}  // namespace streetlearn
