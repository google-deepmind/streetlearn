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

#include "streetlearn/engine/test_dataset.h"
#include "gtest/gtest.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "leveldb/db.h"
#include "leveldb/options.h"
#include "streetlearn/engine/dataset.h"

namespace streetlearn {

constexpr char TestDataset::kStreetName[];
constexpr char TestDataset::kRegion[];
constexpr char TestDataset::kFullAddress[];
constexpr char TestDataset::kDescription[];
constexpr char TestDataset::kCountryCode[];
constexpr char TestDataset::kPanoDate[];
constexpr float TestDataset::kAltitude;
constexpr float TestDataset::kLatitude;
constexpr float TestDataset::kLongitude;
constexpr float TestDataset::kIncrement;
constexpr float TestDataset::kRollDegrees;
constexpr float TestDataset::kPitchDegrees;
constexpr float TestDataset::kHeadingDegrees;
constexpr float TestDataset::kMinLatitude;
constexpr float TestDataset::kMinLongitude;
constexpr float TestDataset::kMaxLatitude;
constexpr float TestDataset::kMaxLongitude;
constexpr int TestDataset::kImageWidth;
constexpr int TestDataset::kImageHeight;
constexpr int TestDataset::kImageDepth;
constexpr int TestDataset::kPanoCount;
constexpr int TestDataset::kNeighborCount;
constexpr int TestDataset::kMinGraphDepth;
constexpr int TestDataset::kMaxGraphDepth;
constexpr int TestDataset::kMaxNeighborDepth;
constexpr int TestDataset::kMaxCacheSize;

Pano TestDataset::GeneratePano(int pano_id) {
  Pano pano;
  pano.set_id(absl::StrCat(pano_id));
  pano.set_street_name(kStreetName);
  pano.set_full_address(kFullAddress);
  pano.set_region(kRegion);
  pano.set_country_code(kCountryCode);
  pano.set_pano_date(kPanoDate);
  pano.set_alt(kAltitude);
  pano.set_roll_deg(kRollDegrees);
  pano.set_pitch_deg(kPitchDegrees);
  pano.set_heading_deg(kHeadingDegrees);
  pano.set_pano_date(kPanoDate);
  LatLng* coords = pano.mutable_coords();
  // Place the panos in a line.
  coords->set_lat(kLatitude + kIncrement * pano_id);
  coords->set_lng(kLongitude + kIncrement * pano_id);
  return pano;
}

std::vector<uint8_t> TestDataset::GenerateCompressedTestImage() {
  auto image = GenerateTestImage();
  cv::Mat mat(image.height(), image.width(), CV_8UC3,
              const_cast<uint8_t*>(image.pixel(0, 0)));
  // Convert to BGR as the defaul OpenCV format is BGR.
  cv::Mat bgr_mat;
  cvtColor(mat, bgr_mat, CV_RGB2BGR);

  std::vector<uint8_t> compressed_image;
  cv::imencode(".png", bgr_mat, compressed_image);
  return compressed_image;
}

// The panos are numbered 1-10 and arranged in a line, each connecting to both
// its immediate neighbors, except the ends.
void TestDataset::AddNeighbors(int pano_id, Pano* pano) {
  if (pano_id < kPanoCount) {
    int next = pano_id + 1;
    PanoNeighbor* neighbor = pano->add_neighbor();
    neighbor->set_id(absl::StrCat(next));
    neighbor->set_pano_date(kPanoDate);
    LatLng* coords = neighbor->mutable_coords();
    coords->set_lat(kLatitude + kIncrement * next);
    coords->set_lng(kLongitude + kIncrement * next);
  }

  if (pano_id > 1) {
    int previous = pano_id - 1;
    PanoNeighbor* neighbor = pano->add_neighbor();
    neighbor->set_id(absl::StrCat(previous));
    neighbor->set_pano_date(kPanoDate);
    LatLng* coords = neighbor->mutable_coords();
    coords->set_lat(kLatitude + kIncrement * previous);
    coords->set_lng(kLongitude + kIncrement * previous);
  }
}

// The panos are numbered 1-10 and arranged in a line, each connecting to both
// its immediate neighbors except the ends.
void TestDataset::AddNeighborMetadata(int pano_id, PanoConnection* connection) {
  connection->set_subgraph_size(kPanoCount);
  connection->set_id(absl::StrCat(pano_id));

  if (pano_id < kPanoCount) {
    auto* neighbor = connection->add_neighbor();
    *neighbor = absl::StrCat(pano_id + 1);
  }

  if (pano_id > 1) {
    auto* neighbor = connection->add_neighbor();
    *neighbor = absl::StrCat(pano_id - 1);
  }
}

bool TestDataset::Generate() {
  leveldb::DB* db_ptr;
  leveldb::Options options;
  options.create_if_missing = true;

  leveldb::Status leveldb_status =
      leveldb::DB::Open(options, GetPath(), &db_ptr);
  if (!leveldb_status.ok()) {
    LOG(ERROR) << "Cannot initialize dataset: " << leveldb_status.ToString();
    return false;
  }

  auto db = absl::WrapUnique<leveldb::DB>(db_ptr);
  const auto compressed_pano_image = GenerateCompressedTestImage();

  // Create panos.
  for (int i = 1; i <= kPanoCount; ++i) {
    Pano pano = GeneratePano(i);
    AddNeighbors(i, &pano);
    pano.mutable_compressed_image()->assign(compressed_pano_image.begin(),
                                            compressed_pano_image.end());

    leveldb_status =
        db->Put(leveldb::WriteOptions(), pano.id(), pano.SerializeAsString());
    if (!leveldb_status.ok()) {
      return false;
    }
  }

  // Create connectivity graph.
  StreetLearnGraph metadata;
  metadata.mutable_min_coords()->set_lat(kMinLatitude);
  metadata.mutable_min_coords()->set_lng(kMinLongitude);
  metadata.mutable_max_coords()->set_lat(kMaxLatitude);
  metadata.mutable_max_coords()->set_lng(kMaxLongitude);

  for (int i = 1; i <= kPanoCount; ++i) {
    PanoConnection* connection = metadata.add_connection();
    *metadata.add_pano() = GeneratePano(i);
    AddNeighborMetadata(i, connection);
  }

  leveldb_status = db->Put(leveldb::WriteOptions(), Dataset::kGraphKey,
                           metadata.SerializeAsString());
  return leveldb_status.ok();
}

std::string TestDataset::GetPath() {
  return ::testing::TempDir() + "/streetlearn_test_dataset";
}

bool TestDataset::GenerateInvalid() {
  leveldb::DB* db_ptr;
  leveldb::Options options;
  options.create_if_missing = true;

  auto status = leveldb::DB::Open(options, GetInvalidDatasetPath(), &db_ptr);
  if (!status.ok()) {
    LOG(ERROR) << "Cannot initialize dataset: " << status.ToString();
    return false;
  }

  auto db = absl::WrapUnique<leveldb::DB>(db_ptr);
  return true;
}

std::string TestDataset::GetInvalidDatasetPath() {
  return ::testing::TempDir() + "/streetlearn_invalid_dataset";
}

Image3_b TestDataset::GenerateTestImage() {
  Image3_b test_image(kImageWidth, kImageHeight);
  for (int x = 0; x < kImageWidth; ++x) {
    for (int y = 0; y < kImageHeight; ++y) {
      auto pixel = test_image.pixel(x, y);
      pixel[0] = 16;
      pixel[1] = 32;
      pixel[2] = 64;
    }
  }

  return test_image;
}

std::vector<std::string> TestDataset::GetPanoIDs() {
  std::vector<std::string> pano_ids(kPanoCount);
  for (int i = 0; i < kPanoCount; ++i) {
    pano_ids[i] = absl::StrCat(i + 1);
  }
  return pano_ids;
}

}  // namespace streetlearn
