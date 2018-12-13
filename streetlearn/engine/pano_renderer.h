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

#ifndef STREETLEARN_PANO_RENDERER_H_
#define STREETLEARN_PANO_RENDERER_H_

#include <cstdint>
#include <map>
#include <vector>

#include <cairo/cairo.h>
#include "streetlearn/engine/color.h"
#include "streetlearn/engine/pano_graph_node.h"
#include "streetlearn/engine/pano_projection.h"
#include "streetlearn/proto/streetlearn.pb.h"

namespace streetlearn {

// Renders a StreetLearn pano image along with an (optional) status bar along
// the bottom indicating which directions of travel are available. All angles
// are specified in degrees.
class PanoRenderer {
 public:
  // Height is the total screen height, including status_height.
  PanoRenderer(int screen_width, int screen_height, int status_height,
               double fov_deg);
  ~PanoRenderer();

  // Renders the input image at the given orientation. The global_yaw is the
  // start heading for the episode and yaw is any additional offset from it.
  // The tolerance determines which bearings are regarded as being in range and
  // can be moved to - these are rendered a different color from the rest.
  // Inaccessible nodes are marked with a no-entry sign.
  void RenderScene(
      const Image3_b& input, double global_yaw, double yaw, double pitch,
      double tolerance, const std::vector<PanoNeighborBearing>& bearings,
      const std::map<int, std::vector<TerminalGraphNode>>& inaccessible);

  // Returns the contents of the pixel buffer which has the format 1 byte per
  // pixel RGB.
  const std::vector<uint8_t>& Pixels() const { return pixels_; }

  // Returns the unique value in the range [-constraint, +constraint) that is
  // equivalent to input_angle. This considers two angles a, b to be equivalent
  // whenever a - b is an integral multiple of 2 * constraint.
  // For example:
  //    ConstrainAngle(187, 180) --> 7
  //    ConstrainAngle(180, 180) --> -180
  static double ConstrainAngle(double constraint, double input_angle);

 private:
  // Projects the portion of the pano at the given yaw and pitch.
  void ProjectPano(const Image3_b& input, double yaw, double pitch);

  // Draws the status bar into the image.
  void DrawStatusBar(double tolerance, double current_bearing,
                     const std::vector<PanoNeighborBearing>& bearings);

  // Draws a bearing marker on the status bar.
  void DrawBearing(double current_bearing, double bearing, const Color& color);

  // Draw no entry signs where the graph has been cut at terminal nodes.
  void DrawNoEntrySigns(
      double global_yaw, double current_bearing, double current_pitch,
      const std::map<int, std::vector<TerminalGraphNode>>& inaccessible);

  // The dimensions of the output image.
  int width_;
  int height_;

  // The field of view of the output image in degrees.
  double fov_deg_;

  // The height of the status bar in pixels.
  int status_height_;

  // The width of a no entry sign in pixels.
  int no_entry_width_;

  // Degrees per-pixel in the output.
  double degrees_pp_;

  // Cairo objects used for rendering the status bar. Owned by PanoRenderer.
  cairo_surface_t* surface_;
  cairo_t* context_;

  // Buffer into which to re-project the StreetLearn image.
  Image3_b pano_buffer_;

  // Buffer for the status bar.
  std::vector<uint8_t> status_pixels_;

  // Buffer for the whole screen.
  std::vector<uint8_t> pixels_;

  // Handles the projection from equiractangular co-ordinates.
  PanoProjection pano_projection_;
};

}  // namespace streetlearn
#endif  // STREETLEARN_PANO_RENDERER_H_
