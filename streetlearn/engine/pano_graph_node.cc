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

#include "streetlearn/engine/pano_graph_node.h"

#include <memory>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace streetlearn {

PanoGraphNode::PanoGraphNode(const Pano& pano) {
  pano_ = std::make_shared<Pano>(pano);

  const std::string& compressed_image = pano.compressed_image();
  if (!compressed_image.empty()) {
    cv::Mat mat =
        cv::imdecode(cv::Mat(1, compressed_image.size(), CV_8UC1,
                             const_cast<char*>(compressed_image.data())),
                     CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    CHECK_EQ(mat.channels(), 3);
    cv::Mat rgb_mat;
    cv::cvtColor(mat, rgb_mat, CV_BGR2RGB);

    image_ = std::make_shared<Image3_b>(mat.cols, mat.rows);
    std::copy(rgb_mat.datastart, rgb_mat.dataend, image_->pixel(0, 0));
  }
}

}  // namespace streetlearn
