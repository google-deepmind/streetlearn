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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_CAIRO_UTIL_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_CAIRO_UTIL_H_

#include <cairo/cairo.h>

namespace streetlearn {

// A helper class to aid rendering with Cairo.
class CairoRenderHelper {
 public:
  // Create a CairoRenderHelper for the image pointed to by `pixels`. The data
  // pointed to by `pixels` must have a longer life time than the created
  // CairoRenderHelper object.
  CairoRenderHelper(unsigned char* pixels, int width, int height,
                    cairo_format_t format = CAIRO_FORMAT_ARGB32) {
    int stride = cairo_format_stride_for_width(format, width);
    surface_ = cairo_image_surface_create_for_data(pixels, format, width,
                                                   height, stride);
    context_ = cairo_create(surface_);
  }

  ~CairoRenderHelper() {
    cairo_surface_flush(surface_);
    cairo_destroy(context_);
    cairo_surface_destroy(surface_);
  }

  CairoRenderHelper(const CairoRenderHelper&) = delete;
  CairoRenderHelper& operator=(const CairoRenderHelper&) = delete;

  int width() const { return cairo_image_surface_get_width(surface_); }
  int height() const { return cairo_image_surface_get_height(surface_); }

  // Returns the current cairo context.
  cairo_t* context() { return context_; }

  // Returns the surface of the current cairo context.
  cairo_surface_t* surface() { return surface_; }

 private:
  // Offscreen buffer used to render game state.
  cairo_surface_t* surface_;
  // Current rendering context.
  cairo_t* context_;
};

}  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_CAIRO_UTIL_H_
