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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_TEST_DATASET_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_TEST_DATASET_H_

#include <cstdint>
#include <string>
#include <vector>

#include "streetlearn/engine/image.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// This class is used in tests to generate a test dataset existing of kPanoCount
// panos that are connected and are on a line.
class TestDataset {
 public:
  static constexpr char kStreetName[] = "Pancras Square";
  static constexpr char kRegion[] = "London, England";
  static constexpr char kFullAddress[] = "Pancras Square, London, England";
  static constexpr char kDescription[] = "Sample Pano";
  static constexpr char kCountryCode[] = "GB";
  static constexpr char kPanoDate[] = "12/12/2012";
  static constexpr float kAltitude = 70.10226140993;
  static constexpr float kLatitude = 51.530589522519;
  static constexpr float kLongitude = -0.12275015773299;

  static constexpr float kIncrement = 0.001;
  static constexpr float kRollDegrees = 3.335879;
  static constexpr float kPitchDegrees = 12.23486;
  static constexpr float kHeadingDegrees = -9.98770;
  static constexpr float kMinLatitude = 51.5315895;
  static constexpr float kMinLongitude = -0.1217502;
  static constexpr float kMaxLatitude = 51.5405884;
  static constexpr float kMaxLongitude = -0.1127502;

  static constexpr int kImageWidth = 64;
  static constexpr int kImageHeight = 48;
  static constexpr int kImageDepth = 3;

  static constexpr int kPanoCount = 10;

  static constexpr int kNeighborCount = 2;
  static constexpr int kMinGraphDepth = 5;
  static constexpr int kMaxGraphDepth = 10;
  static constexpr int kMaxNeighborDepth = 1;

  static constexpr int kThreadCount = 8;
  static constexpr int kMaxCacheSize = 8;

  // Create a test dataset of kPanoCount panos on a line with IDs 1..kPanoCount.
  static bool Generate();

  // Return the path where the generated test dataset is.
  static std::string GetPath();

  // Create an invalid dataset.
  static bool GenerateInvalid();

  // Return the path where the invalid dataset is.
  static std::string GetInvalidDatasetPath();

  // Generate a test pano filled with the data defined in the constants.
  static Pano GeneratePano(int pano_id);

  // Generate a test pano image of size (kImageWidth, kImageHeight). The pixels
  // of the test image are set to RGB values of 16, 32, 64 respectively.
  static Image3_b GenerateTestImage();

  // Generate a test pano image but return a compressed (PNG) buffer.
  static std::vector<uint8_t> GenerateCompressedTestImage();

  // Returns the pano IDs in the test data.
  static std::vector<std::string> GetPanoIDs();

 private:
  // Add neighbors to the pano. Neighbors are the previous and next panos on the
  // line if they exist.
  static void AddNeighbors(int pano_id, Pano* pano);

  // Add neigbor metadata to the connection.
  static void AddNeighborMetadata(int pano_id, PanoConnection* connection);
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_TEST_DATASET_H_
