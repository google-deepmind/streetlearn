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

#include "streetlearn/engine/dataset.h"

#include <string>
#include <utility>

#include "streetlearn/engine/logging.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "leveldb/options.h"
#include "leveldb/status.h"

namespace streetlearn {

constexpr char Dataset::kGraphKey[];

std::unique_ptr<Dataset> Dataset::Create(const std::string& dataset_path) {
  leveldb::DB* db;
  leveldb::Status status =
      leveldb::DB::Open(leveldb::Options(), dataset_path, &db);
  if (!status.ok()) {
    LOG(ERROR) << "Unable initialize dataset: " << status.ToString();
    return nullptr;
  }

  return absl::make_unique<Dataset>(absl::WrapUnique(db));
}

Dataset::Dataset(std::unique_ptr<leveldb::DB> db) { db_ = std::move(db); }

bool Dataset::GetGraph(StreetLearnGraph* graph) const {
  std::string graph_proto_string;
  if (GetValue(kGraphKey, &graph_proto_string)) {
    return graph->ParseFromString(graph_proto_string);
  }
  return false;
}

bool Dataset::GetPano(absl::string_view pano_id, Pano* pano) const {
  std::string pano_proto_string;
  if (GetValue(pano_id, &pano_proto_string)) {
    return pano->ParseFromString(pano_proto_string);
  }

  return false;
}

bool Dataset::GetValue(absl::string_view key, std::string* value) const {
  leveldb::Status status =
      db_->Get(leveldb::ReadOptions(), std::string(key), value);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to get key/value: " << status.ToString();
    return false;
  }

  return true;
}

}  // namespace streetlearn
