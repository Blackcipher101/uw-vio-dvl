// End-to-end estimator test on a simulated stereo + IMU + depth sequence with known truth.
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "uw_ct_vio/Estimator.hpp"

using uw_ct_vio::Estimator;
using uw_ct_vio::EstimatorConfig;
using uw_ct_vio::FeatureObs;

namespace {

constexpr double kG = 9.81, kImuHz = 200.0, kCamHz = 20.0, kDepthHz = 10.0, kDuration = 40.0;
constexpr double kFocal = 460.0, kBaseline = 0.11;

gtsam::Point3 position(double t) {
  return {2.0 * std::sin(0.2 * t), 1.5 * std::sin(0.15 * t + 0.5) - 1.5 * std::sin(0.5), 0.3 * std::sin(0.3 * t)};
}
gtsam::Rot3 rotation(double t) {
  return gtsam::Rot3::Ypr(0.6 * std::sin(0.1 * t), 0.04 * std::sin(0.4 * t), 0.05 * std::sin(0.5 * t));
}

struct Truth {
  std::vector<double> t;
  std::vector<gtsam::Pose3> T_WB;
};

struct Result {
  double rmse_m = 1e9;
  int frames = 0, frames_with_vision = 0;
};

Result run(double outlier_fraction, bool use_depth, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  EstimatorConfig cfg;
  // Forward-looking stereo: optical z = body x, optical x = -body y, optical y = -body z.
  const gtsam::Rot3 R_BC(gtsam::Matrix3((gtsam::Matrix3() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished()));
  cfg.T_BC0 = gtsam::Pose3(R_BC, gtsam::Point3(0.1, 0.0, 0.0));
  cfg.T_BC1 = cfg.T_BC0 * gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(kBaseline, 0, 0));
  cfg.focal_px = kFocal;
  cfg.acc_noise_density = 0.01;
  cfg.gyr_noise_density = 0.001;
  cfg.acc_random_walk = 0.001;
  cfg.gyr_random_walk = 0.0001;
  Estimator est(cfg);

  // Landmarks on a cylinder (radius 6 m) around the trajectory, like tank walls.
  std::vector<gtsam::Point3> lms;
  for (int i = 0; i < 600; ++i) {
    const double a = 2 * M_PI * u01(rng);
    lms.emplace_back(6.0 * std::cos(a), 6.0 * std::sin(a), -3.0 + 6.0 * u01(rng));
  }
  const gtsam::Vector3 ba(0.05, -0.03, 0.02), bg(0.002, -0.001, 0.0015);  // true constant biases
  const double sa = cfg.acc_noise_density * std::sqrt(kImuHz), sg = cfg.gyr_noise_density * std::sqrt(kImuHz);
  const double h = 1e-4;

  Truth truth;
  std::vector<double> estT;
  std::vector<gtsam::Point3> estP;
  Result res;
  double nextCam = 0.5, nextDepth = 0.5;  // first 0.5 s: IMU only (initialisation window)
  for (int i = 0; i <= static_cast<int>(kDuration * kImuHz); ++i) {
    const double t = i / kImuHz;
    // IMU: specific force and body rate from the analytic trajectory (finite differences).
    const gtsam::Vector3 acc_w = (position(t + h) - 2 * position(t) + position(t - h)) / (h * h);
    const gtsam::Rot3 R = rotation(t);
    const gtsam::Vector3 f_b = R.unrotate(acc_w + gtsam::Vector3(0, 0, kG));
    const gtsam::Vector3 w_b = gtsam::Rot3::Logmap(rotation(t - h).between(rotation(t + h))) / (2 * h);
    est.addImu({t, f_b + ba + sa * gtsam::Vector3(n01(rng), n01(rng), n01(rng)),
                w_b + bg + sg * gtsam::Vector3(n01(rng), n01(rng), n01(rng))});
    if (use_depth && t >= nextDepth - 1e-9) {
      est.addDepth(t, -position(t).z() + 0.01 * n01(rng));  // surface at world z = 0 of the truth
      nextDepth += 1.0 / kDepthHz;
    }
    if (t < nextCam - 1e-9) continue;
    nextCam += 1.0 / kCamHz;
    const gtsam::Pose3 T_WB(R, position(t));
    std::vector<FeatureObs> obs;
    for (size_t j = 0; j < lms.size(); ++j) {
      const gtsam::Point3 p0 = (T_WB * cfg.T_BC0).transformTo(lms[j]);
      const gtsam::Point3 p1 = (T_WB * cfg.T_BC1).transformTo(lms[j]);
      if (p0.z() < 0.3 || std::abs(p0.x() / p0.z()) > 0.7 || std::abs(p0.y() / p0.z()) > 0.55) continue;
      FeatureObs f;
      f.track_id = j;
      const double s = 1.0 / kFocal;
      f.left = gtsam::Point2(p0.x() / p0.z() + s * n01(rng), p0.y() / p0.z() + s * n01(rng));
      if (p1.z() > 0.3) {
        f.has_right = true;
        f.right = gtsam::Point2(p1.x() / p1.z() + s * n01(rng), p1.y() / p1.z() + s * n01(rng));
      }
      if (u01(rng) < outlier_fraction) f.left += gtsam::Point2(0.2 * (u01(rng) - 0.5), 0.2 * (u01(rng) - 0.5));
      obs.push_back(f);
    }
    uw_ct_vio::FrameStats st;
    if (!est.addFrame(t, obs, &st)) continue;
    ++res.frames;
    res.frames_with_vision += st.vision_used;
    truth.t.push_back(t);
    truth.T_WB.push_back(T_WB);
    estT.push_back(t);
    estP.push_back(est.latestState().pose().translation());
  }
  if (truth.t.empty()) return res;
  // Compare displacements from the first state; the estimator starts at the origin with yaw 0,
  // and the truth's yaw at t = 0.5 s is 0.03 rad, so rotate the truth into that frame first.
  const gtsam::Rot3 yaw0 = gtsam::Rot3::Yaw(rotation(truth.t.front()).yaw());
  double se = 0;
  for (size_t i = 0; i < truth.t.size(); ++i) {
    const gtsam::Vector3 d_true = yaw0.unrotate(truth.T_WB[i].translation() - truth.T_WB.front().translation());
    const gtsam::Vector3 d_est = estP[i] - estP.front();
    se += (d_true - d_est).squaredNorm();
  }
  res.rmse_m = std::sqrt(se / truth.t.size());
  return res;
}

}  // namespace

TEST(EstimatorSynthetic, TracksCleanStereoImu) {
  const Result r = run(0.0, false, 1);
  std::cout << "[ result ] TracksCleanStereoImu: rmse " << r.rmse_m << " m, frames " << r.frames << ", with vision " << r.frames_with_vision << "\n";
  EXPECT_GT(r.frames, 700);
  EXPECT_GT(r.frames_with_vision, r.frames - 5);
  EXPECT_LT(r.rmse_m, 0.10) << "frames " << r.frames;
}

TEST(EstimatorSynthetic, ToleratesTenPercentOutliers) {
  const Result r = run(0.10, false, 2);
  std::cout << "[ result ] ToleratesTenPercentOutliers: rmse " << r.rmse_m << " m, frames " << r.frames << ", with vision " << r.frames_with_vision << "\n";
  EXPECT_GT(r.frames, 700);
  EXPECT_LT(r.rmse_m, 0.15);
}

TEST(EstimatorSynthetic, DepthFactorDoesNotHurt) {
  const Result r = run(0.0, true, 3);
  std::cout << "[ result ] DepthFactorDoesNotHurt: rmse " << r.rmse_m << " m, frames " << r.frames << ", with vision " << r.frames_with_vision << "\n";
  EXPECT_GT(r.frames, 700);
  EXPECT_LT(r.rmse_m, 0.10);
}

