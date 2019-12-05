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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_LEVELDB_DATASET_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_LEVELDB_DATASET_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "leveldb/db.h"
#include "streetlearn/engine/dataset.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// Dataset is a wrapper around a StreetLearn dataset, that currently resides in
// LevelDB files.
class LevelDBDataset : public Dataset {
 public:
  // The key used to access the connectivity graph in the underlying database.
  static constexpr char kGraphKey[] = "panos_connectivity";

  // Create a Dataset instance that is initialised to use the levelDB database
  // at `dataset_path` on the filesystem.
  static std::unique_ptr<LevelDBDataset> Create(
      const std::string& dataset_path);

  // Construct Dataset using an already open levelDB instance. Only for testing
  // purposes. Regular users should use the Create factory method.
  LevelDBDataset(std::unique_ptr<leveldb::DB> db);

  LevelDBDataset(const LevelDBDataset&) = delete;
  LevelDBDataset& operator=(const LevelDBDataset&) = delete;

  // Get the connectivity graph from the dataset.
  bool GetGraph(StreetLearnGraph* graph) const override;

  // Get a Pano associated with `pano_id` from the dataset. If the `pano_id` is
  // unknown, the function returns false.
  bool GetPano(absl::string_view pano_id, Pano* pano) const override;

 private:
  // Try to get a string value stored in the dataset under `key`.
  bool GetValue(absl::string_view key, std::string* value) const;

  std::unique_ptr<leveldb::DB> db_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_LEVELDB_DATASET_H_
