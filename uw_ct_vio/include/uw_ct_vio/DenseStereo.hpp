// Dense stereo (rectification + SGBM) turned into an organised point cloud with normals.
#pragma once

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

#include "uw_ct_vio/KltStereoTracker.hpp"

namespace uw_ct_vio {

struct DenseStereoConfig {
  // SGBM on rectified images downscaled by 2 (parameters of the working scripts/sgbm_probe.py).
  int num_disparities = 64;          ///< [px at half resolution], multiple of 16
  int block_size = 5;
  int disp12_max_diff = 1;           ///< left-right consistency [px]
  int uniqueness_ratio = 10;
  int speckle_window = 100, speckle_range = 2;
  double min_disparity_px = 0.5;     ///< [px at half resolution]; below this the disparity is invalid
  // Cloud
  int grid_stride = 4;               ///< cloud grid spacing [px at half resolution]
  double min_depth_m = 0.3, max_depth_m = 8.0;  ///< along the rectified optical axis
  double disparity_sigma_px = 0.5;   ///< 1-sigma disparity error [px at half resolution]
  /// Neighbouring grid points whose range differs by more than this fraction of the range, or whose
  /// second difference does, belong to different surfaces: no normal there.
  double max_neighbour_range_frac = 0.1;
};

/// Organised (grid) point cloud of one stereo frame. Points and normals are in the ORIGINAL left camera
/// frame; the grid and (fx, fy, cx, cy) belong to the rectified left image at half resolution, so a
/// point p_cam is looked up with R_rect_cam * p_cam projected through the pinhole.
struct OrganisedCloud {
  static constexpr uint8_t kPoint = 1, kNormal = 2;
  int rows = 0, cols = 0, stride = 1;  ///< grid size; cell (r, c) sits at pixel (c * stride, r * stride)
  double fx = 0, fy = 0, cx = 0, cy = 0;      ///< rectified half-resolution pinhole
  Eigen::Matrix3d R_rect_cam = Eigen::Matrix3d::Identity();  ///< x_rect = R_rect_cam * x_cam
  std::vector<Eigen::Vector3d> p, n;  ///< [m]; n unit, facing the camera, valid where kNormal is set
  std::vector<float> sigma_z;         ///< 1-sigma depth error along the optical axis [m]
  std::vector<uint8_t> flags;
  size_t index(int r, int c) const { return static_cast<size_t>(r) * cols + c; }
  int validPoints() const;
  int validNormals() const;
  void resize(int rows_, int cols_);
};

/// Fills n and the kNormal flags from the grid neighbours (left/right/up/down) of every point.
void estimateNormals(OrganisedCloud* cloud, double max_neighbour_range_frac);

class DenseStereo {
 public:
  /// T_C0C1: right camera pose in the left camera frame. Throws std::invalid_argument on bad geometry.
  DenseStereo(const DenseStereoConfig& cfg, const CameraModel& left, const CameraModel& right, const gtsam::Pose3& T_C0C1);

  /// Raw grey images of one stereo frame (the camera models' resolution).
  OrganisedCloud compute(const cv::Mat& left, const cv::Mat& right) const;
  /// Cloud from an SGBM disparity map (CV_16S, 1/16 px, half resolution).
  OrganisedCloud cloudFromDisparity(const cv::Mat& disp16) const;

  /// Distorted pixels of the left / right camera -> full-resolution rectified pixels.
  std::vector<cv::Point2f> rectifyPixels(const std::vector<cv::Point2f>& px, bool right) const;
  /// Point in the ORIGINAL left camera frame from a half-resolution rectified left pixel and its disparity.
  Eigen::Vector3d pointFromDisparity(double u_half, double v_half, double disparity_half_px) const;

  double baselineM() const { return baseline_m_; }
  double focalHalfPx() const { return fx_half_; }

 private:
  DenseStereoConfig cfg_;
  CameraModel left_, right_;
  cv::Mat Kl_, Kr_, Dl_, Dr_, R1_, R2_, P1_, P2_;
  cv::Mat mapL1_, mapL2_, mapR1_, mapR2_;
  Eigen::Matrix3d R1e_ = Eigen::Matrix3d::Identity();
  double baseline_m_ = 0, fx_half_ = 0, fy_half_ = 0, cx_half_ = 0, cy_half_ = 0;
  cv::Size size_;
  mutable cv::Ptr<cv::StereoSGBM> sgbm_;  // compute() is logically const; SGBM keeps scratch buffers
};

}  // namespace uw_ct_vio
