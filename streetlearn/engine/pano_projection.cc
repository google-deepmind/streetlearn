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

#include "streetlearn/engine/pano_projection.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "absl/memory/memory.h"
#include "streetlearn/engine/math_util.h"

namespace {

// Radius used in calculations of the X, Y, Z point cloud.
static constexpr double PANO_RADIUS = 120.0;

}  // namespace

namespace streetlearn {

namespace {

// Conversion from Image to OpenCV Mat.
void ImageToOpenCV(const Image3_b& input, cv::Mat* output) {
  const int width = input.width();
  const int height = input.height();
  output->create(height, width, CV_8UC3);
  const auto image_data = input.data();
  std::copy(image_data.begin(), image_data.end(), output->data);
}

}  // namespace

struct PanoProjection::Impl {
  Impl(double fov_deg, int proj_width, int proj_height)
      : fov_deg_(0), proj_width_(0), proj_height_(0) {
    // Update the projection size and pre-compute the centered XYZ point cloud.
    UpdateProjectionSize(fov_deg, proj_width, proj_height);
  }

  void Project(const cv::Mat& input, double yaw_deg, double pitch_deg,
               cv::Mat* output);

  // Prepares the centered XYZ point cloud for a specified field of view and
  // projection image width and height.
  cv::Mat PrepareXYZPointCloud();
  // Rotates a centered XYZ point cloud to given yaw and pitch angles.
  cv::Mat RotateXYZPointCloud(double yaw_deg, double pitch_deg);

  // Projects an XYZ point cloud to latitude and longitude maps, given panorama
  // image and projected image widths and heights.
  void ProjectXYZPointCloud(const cv::Mat& xyz, int pano_width, int pano_height,
                            cv::Mat* map_lon, cv::Mat* map_lat);

  // Decodes the image.
  void Decode(int image_width, int image_height, int image_depth,
              int compressed_size, const char* compressed_image,
              std::vector<uint8_t>* output);

  // Updates the projected image size and field of view, and if there are
  // changes then recomputes the centered XYZ point cloud.
  void UpdateProjectionSize(double fov_deg, int proj_width, int proj_height);

  // Degrees of horizontal field of view.
  double fov_deg_;

  // Projected image size.
  int proj_width_;
  int proj_height_;

  // Point cloud of projected image coordinates, with zero yaw and pitch.
  cv::Mat xyz_centered_;
};

cv::Mat PanoProjection::Impl::PrepareXYZPointCloud() {
  // Vertical and horizontal fields of view.
  double fov_width_deg = fov_deg_;
  double fov_height_deg = fov_deg_ * proj_height_ / proj_width_;
  // Scale of the Y and Z grid.
  double half_fov_width_rad = math::DegreesToRadians(fov_width_deg / 2.0);
  double half_afov_width_rad =
      math::DegreesToRadians((180.0 - fov_width_deg) / 2.0);
  double half_fov_height_rad = math::DegreesToRadians(fov_height_deg / 2.0);
  double half_afov_height_rad =
      math::DegreesToRadians((180.0 - fov_height_deg) / 2.0);
  double ratio_width =
      std::sin(half_fov_width_rad) / std::sin(half_afov_width_rad);
  double scale_width = 2.0 * PANO_RADIUS * ratio_width / (proj_width_ - 1);
  double ratio_height =
      std::sin(half_fov_height_rad) / std::sin(half_afov_height_rad);
  double scale_height = 2.0 * PANO_RADIUS * ratio_height / (proj_height_ - 1);
  // Image centres for the projection.
  double c_x = (proj_width_ - 1.0) / 2.0;
  double c_y = (proj_height_ - 1.0) / 2.0;

  // Create the 3D point cloud for X, Y, Z coordinates of the image projected
  // on a sphere.
  cv::Mat xyz_centered;
  xyz_centered.create(3, proj_height_ * proj_width_, CV_32FC1);
  for (int i = 0, k = 0; i < proj_height_; i++) {
    for (int j = 0; j < proj_width_; j++, k++) {
      double x_ij = PANO_RADIUS;
      double y_ij = (j - c_x) * scale_width;
      double z_ij = -(i - c_y) * scale_height;
      double d_ij = std::sqrt(x_ij * x_ij + y_ij * y_ij + z_ij * z_ij);
      double coeff = PANO_RADIUS / d_ij;
      xyz_centered.at<float>(0, k) = coeff * x_ij;
      xyz_centered.at<float>(1, k) = coeff * y_ij;
      xyz_centered.at<float>(2, k) = coeff * z_ij;
    }
  }
  return xyz_centered;
}

cv::Mat PanoProjection::Impl::RotateXYZPointCloud(double yaw_deg,
                                                  double pitch_deg) {
  // Rotation matrix for yaw.
  float z_axis_data[3] = {0.0, 0.0, 1.0};
  cv::Mat z_axis(3, 1, CV_32FC1, z_axis_data);
  cv::Mat z_rotation_vec = z_axis * math::DegreesToRadians(yaw_deg);
  cv::Mat z_rotation_mat(3, 3, CV_32FC1);
  cv::Rodrigues(z_rotation_vec, z_rotation_mat);

  // Rotation matrix for pitch.
  float y_axis_data[3] = {0.0, 1.0, 0.0};
  cv::Mat y_axis(3, 1, CV_32FC1, y_axis_data);
  cv::Mat y_rotation_vec =
      z_rotation_mat * y_axis * math::DegreesToRadians(-pitch_deg);
  cv::Mat y_rotation_mat(3, 3, CV_32FC1);
  cv::Rodrigues(y_rotation_vec, y_rotation_mat);

  // Apply yaw and pitch rotation to the point cloud.
  cv::Mat xyz_rotated = y_rotation_mat * (z_rotation_mat * xyz_centered_);

  return xyz_rotated;
}

void PanoProjection::Impl::ProjectXYZPointCloud(const cv::Mat& xyz,
                                                int pano_width, int pano_height,
                                                cv::Mat* map_lon,
                                                cv::Mat* map_lat) {
  // Image centres for the panorama.
  double pano_c_x = (pano_width - 1.0) / 2.0;
  double pano_c_y = (pano_height - 1.0) / 2.0;

  map_lat->create(proj_height_, proj_width_, CV_32FC1);
  map_lon->create(proj_height_, proj_width_, CV_32FC1);

  for (int i = 0, k = 0; i < proj_height_; i++) {
    for (int j = 0; j < proj_width_; j++, k++) {
      // Project the Z coordinate into latitude.
      double z_ij = xyz.at<float>(2, k);
      double sin_lat_ij = z_ij / PANO_RADIUS;
      double lat_ij = std::asin(sin_lat_ij);
      map_lat->at<float>(i, j) = -lat_ij / CV_PI * 2.0 * pano_c_y + pano_c_y;
      // Project the X and Y coordinates into longitude.
      double x_ij = xyz.at<float>(0, k);
      double y_ij = xyz.at<float>(1, k);
      double tan_theta_ij = y_ij / x_ij;
      double theta_ij = std::atan(tan_theta_ij);
      double lon_ij;
      if (x_ij > 0) {
        lon_ij = theta_ij;
      } else {
        if (y_ij > 0) {
          lon_ij = theta_ij + CV_PI;
        } else {
          lon_ij = theta_ij - CV_PI;
        }
      }
      map_lon->at<float>(i, j) = lon_ij / CV_PI * pano_c_x + pano_c_x;
    }
  }
}

void PanoProjection::Impl::UpdateProjectionSize(double fov_deg, int proj_width,
                                                int proj_height) {
  // Update only if there is a change.
  if ((proj_width_ != proj_width) || (proj_height_ != proj_height) ||
      (fov_deg != fov_deg_)) {
    fov_deg_ = fov_deg;
    proj_width_ = proj_width;
    proj_height_ = proj_height;

    // If updated the projected image or field of view parameters,
    // recompute the centered XYZ point cloud.
    xyz_centered_ = PrepareXYZPointCloud();
  }
}

void PanoProjection::Impl::Decode(int image_width, int image_height,
                                  int image_depth, int compressed_size,
                                  const char* compressed_image,
                                  std::vector<uint8_t>* output) {
  std::vector<uint8_t> input(compressed_image,
                             compressed_image + compressed_size);
  cv::Mat mat = cv::imdecode(cv::Mat(input), CV_LOAD_IMAGE_ANYDEPTH);
  output->resize(image_width * image_height * image_depth);
  output->assign(mat.datastart, mat.dataend);
}

void PanoProjection::Impl::Project(const cv::Mat& input, double yaw_deg,
                                   double pitch_deg, cv::Mat* output) {
  // Assuming that the field of view has not changed, check for updates of
  // the projected image size, and optionally recompute the centered XYZ
  // point cloud.
  int proj_width = output->cols;
  int proj_height = output->rows;
  UpdateProjectionSize(fov_deg_, proj_width, proj_height);

  // Rotate the XYZ point cloud by given yaw and pitch.
  cv::Mat xyz = RotateXYZPointCloud(yaw_deg, pitch_deg);

  // Project the rotated XYZ point cloud to latitude and longitude maps.
  int pano_width = input.cols;
  int pano_height = input.rows;
  cv::Mat map_lon;
  cv::Mat map_lat;
  ProjectXYZPointCloud(xyz, pano_width, pano_height, &map_lon, &map_lat);

  // Remap from input to output given the latitude and longitude maps.
  cv::remap(input, *output, map_lon, map_lat, CV_INTER_CUBIC);
}

PanoProjection::PanoProjection(double fov_deg, int proj_width, int proj_height)
    : impl_(absl::make_unique<PanoProjection::Impl>(fov_deg, proj_width,
                                                    proj_height)) {}

PanoProjection::~PanoProjection() {}

void PanoProjection::Project(const Image3_b& input, double yaw_deg,
                             double pitch_deg, Image3_b* output) {
  cv::Mat input_cv;
  ImageToOpenCV(input, &input_cv);
  cv::Mat output_cv;
  output_cv.create(impl_->proj_height_, impl_->proj_width_, CV_8UC3);
  impl_->Project(input_cv, yaw_deg, pitch_deg, &output_cv);
  std::memcpy(output->pixel(0, 0), output_cv.data,
              impl_->proj_width_ * impl_->proj_height_ * 3);
}

void PanoProjection::ChangeFOV(double fov_deg) {
  impl_->UpdateProjectionSize(fov_deg, impl_->proj_width_, impl_->proj_height_);
}

}  // namespace streetlearn
