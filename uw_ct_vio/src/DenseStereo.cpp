#include "uw_ct_vio/DenseStereo.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uw_ct_vio {

int OrganisedCloud::validPoints() const {
  int n_ = 0;
  for (uint8_t f : flags) n_ += (f & kPoint) ? 1 : 0;
  return n_;
}

int OrganisedCloud::validNormals() const {
  int n_ = 0;
  for (uint8_t f : flags) n_ += (f & kNormal) ? 1 : 0;
  return n_;
}

void OrganisedCloud::resize(int rows_, int cols_) {
  rows = rows_, cols = cols_;
  const size_t cells = static_cast<size_t>(rows) * cols;
  p.assign(cells, Eigen::Vector3d::Zero());
  n.assign(cells, Eigen::Vector3d::Zero());
  sigma_z.assign(cells, 0.0f);
  flags.assign(cells, 0);
}

void estimateNormals(OrganisedCloud* cloud, double max_neighbour_range_frac) {
  OrganisedCloud& c = *cloud;
  for (auto& f : c.flags) f &= static_cast<uint8_t>(~OrganisedCloud::kNormal);
  for (int r = 1; r + 1 < c.rows; ++r) {
    for (int col = 1; col + 1 < c.cols; ++col) {
      const size_t i = c.index(r, col), l = c.index(r, col - 1), rt = c.index(r, col + 1), u = c.index(r - 1, col),
                   d = c.index(r + 1, col);
      constexpr uint8_t kP = OrganisedCloud::kPoint;
      if (!((c.flags[i] & kP) && (c.flags[l] & kP) && (c.flags[rt] & kP) && (c.flags[u] & kP) && (c.flags[d] & kP))) continue;
      const double range = c.p[i].norm();
      const double tol = max_neighbour_range_frac * range;
      // Same surface: neighbours at a similar range, and a small second difference along both grid axes.
      if (std::abs(c.p[l].norm() - range) > tol || std::abs(c.p[rt].norm() - range) > tol ||
          std::abs(c.p[u].norm() - range) > tol || std::abs(c.p[d].norm() - range) > tol)
        continue;
      if ((c.p[l] + c.p[rt] - 2.0 * c.p[i]).norm() > tol || (c.p[u] + c.p[d] - 2.0 * c.p[i]).norm() > tol) continue;
      Eigen::Vector3d nrm = (c.p[rt] - c.p[l]).cross(c.p[d] - c.p[u]);
      const double len = nrm.norm();
      if (!(len > 1e-12) || !std::isfinite(len)) continue;
      nrm /= len;
      if (nrm.dot(c.p[i]) > 0.0) nrm = -nrm;  // face the camera
      c.n[i] = nrm;
      c.flags[i] |= OrganisedCloud::kNormal;
    }
  }
}

namespace {

cv::Mat cameraMatrix(const CameraModel& c) { return (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1); }

cv::Mat distCoeffs(const CameraModel& c) { return (cv::Mat_<double>(4, 1) << c.coeffs[0], c.coeffs[1], c.coeffs[2], c.coeffs[3]); }

Eigen::Matrix3d toEigen(const cv::Mat& m) {
  Eigen::Matrix3d e;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) e(r, c) = m.at<double>(r, c);
  return e;
}

}  // namespace

DenseStereo::DenseStereo(const DenseStereoConfig& cfg, const CameraModel& left, const CameraModel& right,
                         const gtsam::Pose3& T_C0C1)
    : cfg_(cfg), left_(left), right_(right), size_(left.width, left.height) {
  left_.validate("dense left");
  right_.validate("dense right");
  if (left.width != right.width || left.height != right.height)
    throw std::invalid_argument("DenseStereo: left and right image sizes differ");
  if (left_.distortion != right_.distortion) throw std::invalid_argument("DenseStereo: mixed distortion models");
  if (cfg_.num_disparities <= 0 || cfg_.num_disparities % 16 != 0 || cfg_.block_size < 3 || cfg_.block_size % 2 == 0 ||
      cfg_.grid_stride < 1 || !(cfg_.min_depth_m > 0.0 && cfg_.max_depth_m > cfg_.min_depth_m) ||
      !(cfg_.disparity_sigma_px > 0.0) || !(cfg_.min_disparity_px >= 0.0))
    throw std::invalid_argument("DenseStereoConfig: invalid SGBM or cloud parameters");
  Kl_ = cameraMatrix(left_), Kr_ = cameraMatrix(right_), Dl_ = distCoeffs(left_), Dr_ = distCoeffs(right_);
  // OpenCV maps left-camera points into the right camera: x_right = R x_left + T.
  const gtsam::Matrix3 R_C0C1 = T_C0C1.rotation().matrix();
  const gtsam::Vector3 t_C0C1 = T_C0C1.translation();
  const gtsam::Matrix3 Rm = R_C0C1.transpose();
  const gtsam::Vector3 Tm = -Rm * t_C0C1;
  cv::Mat R = (cv::Mat_<double>(3, 3) << Rm(0, 0), Rm(0, 1), Rm(0, 2), Rm(1, 0), Rm(1, 1), Rm(1, 2), Rm(2, 0), Rm(2, 1), Rm(2, 2));
  cv::Mat T = (cv::Mat_<double>(3, 1) << Tm(0), Tm(1), Tm(2));
  cv::Mat Q;
  if (left_.distortion == CameraModel::Distortion::Equidistant) {
    cv::fisheye::stereoRectify(Kl_, Dl_, Kr_, Dr_, size_, R, T, R1_, R2_, P1_, P2_, Q, cv::CALIB_ZERO_DISPARITY, size_, 0.0, 1.0);
    cv::fisheye::initUndistortRectifyMap(Kl_, Dl_, R1_, P1_, size_, CV_32FC1, mapL1_, mapL2_);
    cv::fisheye::initUndistortRectifyMap(Kr_, Dr_, R2_, P2_, size_, CV_32FC1, mapR1_, mapR2_);
  } else {
    cv::stereoRectify(Kl_, Dl_, Kr_, Dr_, size_, R, T, R1_, R2_, P1_, P2_, Q, cv::CALIB_ZERO_DISPARITY, 0.0, size_);
    cv::initUndistortRectifyMap(Kl_, Dl_, R1_, P1_, size_, CV_32FC1, mapL1_, mapL2_);
    cv::initUndistortRectifyMap(Kr_, Dr_, R2_, P2_, size_, CV_32FC1, mapR1_, mapR2_);
  }
  const double f = P1_.at<double>(0, 0);
  if (!(f > 0.0) || !(P1_.at<double>(1, 1) > 0.0) || std::abs(P2_.at<double>(0, 0) - f) > 1e-6 * f)
    throw std::invalid_argument("DenseStereo: rectification produced inconsistent projections");
  baseline_m_ = -P2_.at<double>(0, 3) / P2_.at<double>(0, 0);
  if (!(baseline_m_ > 0.0) || !std::isfinite(baseline_m_))
    throw std::invalid_argument("DenseStereo: non-positive baseline after rectification (right camera must be right of left)");
  R1e_ = toEigen(R1_);
  // pyrDown keeps every second pixel, so half-resolution pixel u corresponds to full-resolution 2u.
  fx_half_ = 0.5 * f, fy_half_ = 0.5 * P1_.at<double>(1, 1);
  cx_half_ = 0.5 * P1_.at<double>(0, 2), cy_half_ = 0.5 * P1_.at<double>(1, 2);
  sgbm_ = cv::StereoSGBM::create(0, cfg_.num_disparities, cfg_.block_size, 8 * cfg_.block_size * cfg_.block_size,
                                 32 * cfg_.block_size * cfg_.block_size, cfg_.disp12_max_diff, 0, cfg_.uniqueness_ratio,
                                 cfg_.speckle_window, cfg_.speckle_range, cv::StereoSGBM::MODE_SGBM_3WAY);
}

std::vector<cv::Point2f> DenseStereo::rectifyPixels(const std::vector<cv::Point2f>& px, bool right) const {
  std::vector<cv::Point2f> out;
  if (px.empty()) return out;
  const cv::Mat& K = right ? Kr_ : Kl_;
  const cv::Mat& D = right ? Dr_ : Dl_;
  const cv::Mat& R = right ? R2_ : R1_;
  const cv::Mat& P = right ? P2_ : P1_;
  if (left_.distortion == CameraModel::Distortion::Equidistant) cv::fisheye::undistortPoints(px, out, K, D, R, P);
  else cv::undistortPoints(px, out, K, D, R, P);
  return out;
}

Eigen::Vector3d DenseStereo::pointFromDisparity(double u_half, double v_half, double disparity_half_px) const {
  const double z = fx_half_ * baseline_m_ / disparity_half_px;
  const Eigen::Vector3d p_rect((u_half - cx_half_) * z / fx_half_, (v_half - cy_half_) * z / fy_half_, z);
  return R1e_.transpose() * p_rect;
}

OrganisedCloud DenseStereo::compute(const cv::Mat& left, const cv::Mat& right) const {
  if (left.size() != size_ || right.size() != size_ || left.type() != CV_8UC1 || right.type() != CV_8UC1)
    throw std::invalid_argument("DenseStereo::compute: images must be 8-bit grey at the camera resolution");
  cv::Mat l, r, l2, r2, disp;
  cv::remap(left, l, mapL1_, mapL2_, cv::INTER_LINEAR);
  cv::remap(right, r, mapR1_, mapR2_, cv::INTER_LINEAR);
  cv::pyrDown(l, l2);
  cv::pyrDown(r, r2);
  sgbm_->compute(l2, r2, disp);
  return cloudFromDisparity(disp);
}

OrganisedCloud DenseStereo::cloudFromDisparity(const cv::Mat& disp16) const {
  if (disp16.type() != CV_16S) throw std::invalid_argument("DenseStereo::cloudFromDisparity: need a CV_16S disparity map");
  OrganisedCloud c;
  const int s = cfg_.grid_stride;
  c.stride = s;
  c.fx = fx_half_, c.fy = fy_half_, c.cx = cx_half_, c.cy = cy_half_;
  c.R_rect_cam = R1e_;
  c.resize((disp16.rows - 1) / s + 1, (disp16.cols - 1) / s + 1);
  for (int r = 0; r < c.rows; ++r) {
    const int16_t* row = disp16.ptr<int16_t>(r * s);
    for (int col = 0; col < c.cols; ++col) {
      const double d = row[col * s] / 16.0;
      if (!(d >= cfg_.min_disparity_px)) continue;
      const Eigen::Vector3d p = pointFromDisparity(col * s, r * s, d);
      const double z = fx_half_ * baseline_m_ / d;  // depth along the rectified optical axis
      if (!p.allFinite() || z < cfg_.min_depth_m || z > cfg_.max_depth_m) continue;
      const size_t i = c.index(r, col);
      c.p[i] = p;
      c.sigma_z[i] = static_cast<float>(z * z * cfg_.disparity_sigma_px / (fx_half_ * baseline_m_));
      c.flags[i] = OrganisedCloud::kPoint;
    }
  }
  estimateNormals(&c, cfg_.max_neighbour_range_frac);
  return c;
}

}  // namespace uw_ct_vio
