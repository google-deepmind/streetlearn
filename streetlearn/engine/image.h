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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_IMAGE_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_IMAGE_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "streetlearn/engine/logging.h"
#include "absl/types/span.h"

namespace streetlearn {

// An ImageFrame is a common superclass for Image and ImageView storing
// dimensions and implements pixel offset calculation.
template <int kChannels>
class ImageFrame {
  static_assert(kChannels > 0, "Channels must be greater than zero");

 public:
  // Construct an ImageFrame for an image of size width x height. The width_step
  // defines the step in pixels for to the next row which is useful if
  // ImageFrame refers to a sub image.
  ImageFrame(int width, int height, int width_step)
      : width_(width), height_(height), width_step_(width_step) {
    CHECK_GE(width, 0);
    CHECK_GE(height, 0);
  }

  virtual ~ImageFrame() = default;

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return kChannels; }

  int offset(int col, int row) const {
    if (0 <= col && col < width_ && 0 <= row && row < height_) {
      return (width_step_ * row + col) * kChannels;
    }

    return 0;
  }

 private:
  const int width_;
  const int height_;
  const int width_step_;
};

// The Image class represents a continous memory buffer of `PixelType` to help
// storing and accessing images easier.
template <typename PixelType, int kChannels>
class Image : public ImageFrame<kChannels> {
 public:
  Image() : ImageFrame<kChannels>(0, 0, 0) {}

  Image(int width, int height)
      : ImageFrame<kChannels>(width, height, width),
        pixels_(width * height * kChannels) {}

  Image(const Image& other)
      : ImageFrame<kChannels>(other.width(), other.height(), other.width()),
        pixels_(other.pixels_) {}

  ~Image() override = default;

  PixelType* pixel(int col, int row) {
    if (pixels_.empty()) return nullptr;
    return &pixels_[ImageFrame<kChannels>::offset(col, row)];
  }

  const PixelType* pixel(int col, int row) const {
    if (pixels_.empty()) return nullptr;
    return &pixels_[ImageFrame<kChannels>::offset(col, row)];
  }

  absl::Span<const PixelType> data() const { return pixels_; }

 private:
  std::vector<PixelType> pixels_;
};

// ImageView defines a readable and writable view of an Image. As a convenience,
// ImageView makes it possible to decorate any buffer of type PixelType and use
// it as if it were an image.
template <typename PixelType, int kChannels>
class ImageView : public ImageFrame<kChannels> {
 public:
  ImageView(Image<PixelType, kChannels>* image, int col, int row, int width,
            int height)
      : ImageFrame<kChannels>(width, height, image->width()),
        pixels_(image->pixel(col, row)) {
    CHECK_LE(0, col);
    CHECK_LE(0, row);
    CHECK_LE(col + width, image->width());
    CHECK_LE(row + height, image->height());
  }

  ImageView(PixelType* data, int width, int height)
      : ImageFrame<kChannels>(width, height, width), pixels_(data) {}

  ~ImageView() override = default;

  PixelType* pixel(int col, int row) {
    return &pixels_[ImageFrame<kChannels>::offset(col, row)];
  }

  const PixelType* pixel(int col, int row) const {
    return &pixels_[ImageFrame<kChannels>::offset(col, row)];
  }

 private:
  PixelType* const pixels_;
};

using Image3_b = Image<uint8_t, 3>;
using Image4_b = Image<uint8_t, 4>;
using ImageView3_b = ImageView<uint8_t, 3>;
using ImageView4_b = ImageView<uint8_t, 4>;

};  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_IMAGE_H_
