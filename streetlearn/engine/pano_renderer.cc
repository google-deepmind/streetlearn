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

#include "streetlearn/engine/pano_renderer.h"

#include <cmath>
#include <cstdint>
#include <iostream>

#include "streetlearn/engine/logging.h"
#include "streetlearn/engine/math_util.h"
#include "streetlearn/engine/pano_graph_node.h"

namespace streetlearn {
namespace {

// Cairo uses [0, 1] range for colors.
constexpr Color kBackColor = {0.0, 0.0, 0.0};
constexpr Color kBarColor = {0.5, 0.5, 0.5};
constexpr Color kRedColor = {0.9, 0.0, 0.0};
constexpr Color kAmberColor = {1.0, 0.49, 0.0};
constexpr Color kGreenColor = {0.0, 0.9, 0.0};

constexpr Color kNoEntryColor = {230, 0, 0};
constexpr double kNoEntryProportion = 0.1;  // proportion of screen
constexpr double kNoEntrySignWidthMeters = 3;
constexpr int kConstraintMultiplier = 2;

// Scale the rgb vector from [0, 1] to [0, 255] and set at the point provided.
void PutPixel(int screen_width, int screen_height, int x, int y,
              const Color& color, uint8_t* pixels) {
  if (0 <= x && x < screen_width && 0 <= y && y < screen_height) {
    int index = 3 * (y * screen_width + x);
    pixels[index] = color.red;
    pixels[index + 1] = color.green;
    pixels[index + 2] = color.blue;
  }
}

// Draw a filled circle at the centre point provided in the given color.
void DrawCircle(int screen_width, int screen_height, int center_x, int center_y,
                int radius, const Color& color, uint8_t* pixels) {
  const float sin_45_degrees = std::sin(math::DegreesToRadians(45.0));
  // This is the distance on the axis from sin(90) to sin(45).
  int range = radius / (2 * sin_45_degrees);
  for (int i = radius; i >= range; --i) {
    int j = sqrt(radius * radius - i * i);
    for (int k = -j; k <= j; k++) {
      // We draw all the 4 sides at the same time.
      PutPixel(screen_width, screen_height, center_x - k, center_y + i, color,
               pixels);
      PutPixel(screen_width, screen_height, center_x - k, center_y - i, color,
               pixels);
      PutPixel(screen_width, screen_height, center_x + i, center_y + k, color,
               pixels);
      PutPixel(screen_width, screen_height, center_x - i, center_y - k, color,
               pixels);
    }
  }
  // To fill the circle we draw the circumscribed square.
  range = radius * sin_45_degrees;
  for (int i = center_x - range + 1; i < center_x + range; i++) {
    for (int j = center_y - range + 1; j < center_y + range; j++) {
      PutPixel(screen_width, screen_height, i, j, color, pixels);
    }
  }
}
}  // namespace

PanoRenderer::PanoRenderer(int screen_width, int screen_height,
                           int status_height, double fov_deg)
    : width_(screen_width),
      height_(screen_height),
      fov_deg_(fov_deg),
      status_height_(status_height),
      no_entry_width_(width_ * kNoEntryProportion),
      degrees_pp_(360.0 / width_),
      context_(nullptr),
      pano_buffer_(screen_width, screen_height - status_height),
      pano_projection_(fov_deg, screen_width, screen_height - status_height) {
  pixels_.resize(width_ * height_ * 3);
  if (status_height_ > 0 && status_height_ < height_) {
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, width_);
    status_pixels_.resize(stride * status_height_);
    surface_ = cairo_image_surface_create_for_data(status_pixels_.data(),
                                                   CAIRO_FORMAT_RGB24, width_,
                                                   status_height_, stride);
    context_ = cairo_create(surface_);

  } else if (status_height_ != 0) {
    LOG(ERROR) << "Invalid status bar height: " << status_height_;
    status_height_ = 0;
  }
}

PanoRenderer::~PanoRenderer() {
  if (context_ != nullptr) {
    cairo_destroy(context_);
    cairo_surface_destroy(surface_);
  }
}

void PanoRenderer::RenderScene(
    const Image3_b& input, double global_yaw, double yaw, double pitch,
    double tolerance, const std::vector<PanoNeighborBearing>& bearings,
    const std::map<int, std::vector<TerminalGraphNode>>& inaccessible) {
  if (status_height_ > 0) {
    DrawStatusBar(tolerance, yaw, bearings);
  }

  ProjectPano(input, yaw - global_yaw, pitch);
  DrawNoEntrySigns(global_yaw, yaw, pitch, inaccessible);
}

void PanoRenderer::ProjectPano(const Image3_b& input, double yaw,
                               double pitch) {
  yaw = ConstrainAngle(360, yaw);
  pitch = ConstrainAngle(90, pitch);
  pano_projection_.Project(input, yaw, pitch, &pano_buffer_);
  std::copy(pano_buffer_.data().begin(), pano_buffer_.data().end(),
            pixels_.begin());
}

void PanoRenderer::DrawBearing(double current_bearing, double bearing,
                               const Color& color) {
  double pos = ConstrainAngle(180, bearing - current_bearing);
  double x_pos = width_ / 2.0 + pos / degrees_pp_;
  cairo_arc(context_, x_pos, status_height_ / 2, status_height_ / 2 - 1, 0,
            2 * M_PI);
  cairo_set_source_rgb(context_, color.red, color.green, color.blue);
  cairo_fill(context_);
}

void PanoRenderer::DrawStatusBar(
    double tolerance, double current_bearing,
    const std::vector<PanoNeighborBearing>& bearings) {
  // Background
  cairo_set_source_rgb(context_, kBackColor.red, kBackColor.green,
                       kBackColor.blue);
  cairo_paint(context_);

  // Bearings - need to center everything around the current bearing. When a
  // point is in both quadrants 3 and 4 need to use the smaller angle between
  // them.

  // Work out the bearing closest to our current bearing down which travel is
  // possible.
  double chosen_dir(std::numeric_limits<double>::max());
  double min_distance(std::numeric_limits<double>::max());
  for (auto neighbor : bearings) {
    double diff = fabs(neighbor.bearing - current_bearing);
    if (diff > 180.0) {
      diff = 360.0 - diff;
    }
    if (diff < tolerance && neighbor.distance < min_distance) {
      chosen_dir = neighbor.bearing;
      min_distance = neighbor.distance;
    }
  }

  // Draw all the bearings except for the chosen direction, which will be drawn
  // last so that no other bearing is superimposed.
  for (auto neighbor : bearings) {
    if (neighbor.bearing == chosen_dir) {
      continue;
    }

    double pos = fabs(neighbor.bearing - current_bearing);
    if (pos > 180.0) {
      pos = 360.0 - pos;
    }
    const auto& color = fabs(pos) < tolerance ? kAmberColor : kRedColor;
    DrawBearing(current_bearing, neighbor.bearing, color);
  }

  if (chosen_dir < std::numeric_limits<double>::max()) {
    DrawBearing(current_bearing, chosen_dir, kGreenColor);
  }

  // Tolerance bars for forward movement.
  cairo_set_source_rgb(context_, kBarColor.red, kBarColor.green,
                       kBarColor.blue);
  double lower_tol = (180.0 - tolerance) / degrees_pp_;
  cairo_move_to(context_, lower_tol, 0.0);
  cairo_line_to(context_, lower_tol, status_height_);
  double upper_tol = (180.0 + tolerance) / degrees_pp_;
  cairo_move_to(context_, upper_tol, 0.0);
  cairo_line_to(context_, upper_tol, status_height_);
  cairo_stroke(context_);

  // Cairo swaps g and b and only supports 4 bytes per pixel - need to convert.
  int image_height = height_ - status_height_;
  for (int i = 0; i < status_height_; i++) {
    const int image_row_offset = (i + image_height) * width_;
    const int status_row_offset = i * width_;
    for (int j = 0; j < width_; j++) {
      int index = 3 * (image_row_offset + j);
      int status_index = 4 * (status_row_offset + j);
      pixels_[index] = status_pixels_[status_index + 2];
      pixels_[index + 1] = status_pixels_[status_index + 1];
      pixels_[index + 2] = status_pixels_[status_index];
    }
  }
}

// The co-ordinates are calculated by uniformly scaling along the axes, which
// may cause the signs to drift slightly towards the edges of the screen.
void PanoRenderer::DrawNoEntrySigns(
    double global_yaw, double current_bearing, double current_pitch,
    const std::map<int, std::vector<TerminalGraphNode>>& inaccessible) {
  // Draw the farthest away first.

  for (auto iter = inaccessible.rbegin(); iter != inaccessible.rend(); ++iter) {
    for (const auto& terminal : iter->second) {
      int sign_width =
          no_entry_width_ *
          (2.0 * atan(kNoEntrySignWidthMeters / terminal.distance)) /
          math::DegreesToRadians(fov_deg_);
      double offset_x = ConstrainAngle(180, terminal.bearing - current_bearing);
      double offset_y = current_pitch;
      int x_pos = width_ * (0.5 + offset_x / fov_deg_);
      int y_pos = height_ * (0.5 - (offset_y * width_) / (fov_deg_ * height_));

      DrawCircle(width_, height_ - status_height_, x_pos, y_pos, sign_width,
                 kNoEntryColor, pixels_.data());
    }
  }
}

double PanoRenderer::ConstrainAngle(double constraint, double input_angle) {
  // The idea here is to shift the problem so that we're remapping the angle to
  // [0.. 2*constraint), which makes it solvable in one line. Having clamped to
  // that range, we then unshift back to [-constraint, constraint).
  double value_range = kConstraintMultiplier * constraint;
  double value = input_angle + constraint;
  value -= value_range * std::floor(value / value_range);
  value -= constraint;
  return value;
}

}  // namespace streetlearn
