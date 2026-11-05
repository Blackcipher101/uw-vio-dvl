// Estimator + DVL on a simulated stereo + IMU + DVL sequence with known truth: dead reckoning through
// camera loss, outlier rejection, DVL dropout, and "DVL off changes nothing".
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "uw_ct_vio/Estimator.hpp"

using uw_ct_vio::Estimator;
using uw_ct_vio::EstimatorConfig;
using uw_ct_vio::FeatureObs;

namespace {

constexpr double kG = 9.81, kImuHz = 200.0, kCamHz = 20.0, kDvlHz = 8.0, kDuration = 40.0;
constexpr double kFocal = 460.0, kBaseline = 0.11;

gtsam::Point3 position(double t) {
  return {2.0 * std::sin(0.2 * t), 1.5 * std::sin(0.15 * t + 0.5) - 1.5 * std::sin(0.5), 0.3 * std::sin(0.3 * t)};
}
gtsam::Rot3 rotation(double t) {
  return gtsam::Rot3::Ypr(0.6 * std::sin(0.1 * t), 0.04 * std::sin(0.4 * t), 0.05 * std::sin(0.5 * t));
}

/// DVL mount in the body frame: yawed 45 deg, 0.15 m below and 0.1 m ahead of the IMU.
gtsam::Pose3 dvlMount() { return gtsam::Pose3(gtsam::Rot3::Yaw(M_PI / 4), gtsam::Point3(0.1, 0.05, -0.15)); }

struct Window {
  double t0 = -1.0, dur = 0.0;
  bool contains(double t) const { return dur > 0.0 && t >= t0 && t < t0 + dur; }
};

struct Scenario {
  bool dvl_enabled = false;     ///< estimator config
  bool feed_dvl = false;        ///< samples are generated and passed to addDvl
  Window vision_blackout;       ///< empty feature lists
  Window dvl_gap;               ///< no DVL samples
  Window dvl_outliers;          ///< constant wrong velocity, valid = 1
  Window imu_kick;              ///< unmodelled 0.15 m/s^2 along body x added to the accelerometer
  double relock_after_s = 2.0;
  unsigned seed = 1;
};

struct Result {
  double final_drift_m = 1e9, max_step_err_m = 1e9, max_step_t = 0.0, err_before_m = 0.0, err_after_m = 0.0;
  int frames = 0;
  Estimator::Counters counters;
  std::vector<gtsam::Point3> est;
};

Result run(const Scenario& sc) {
  std::mt19937 rng(sc.seed), dvlRng(sc.seed + 1000);  // separate stream: DVL noise never perturbs the rest
  std::normal_distribution<double> n01(0.0, 1.0), dvlN01(0.0, 1.0);  // own distribution: it caches a draw
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  EstimatorConfig cfg;
  const gtsam::Rot3 R_BC(gtsam::Matrix3((gtsam::Matrix3() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished()));
  cfg.T_BC0 = gtsam::Pose3(R_BC, gtsam::Point3(0.1, 0.0, 0.0));
  cfg.T_BC1 = cfg.T_BC0 * gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(kBaseline, 0, 0));
  cfg.focal_px = kFocal;
  cfg.acc_noise_density = 0.01;
  cfg.gyr_noise_density = 0.001;
  cfg.acc_random_walk = 0.001;
  cfg.gyr_random_walk = 0.0001;
  cfg.dvl_enabled = sc.dvl_enabled;
  cfg.dvl_T_BD = dvlMount();
  cfg.dvl_relock_after_s = sc.relock_after_s;
  Estimator est(cfg);

  std::vector<gtsam::Point3> lms;
  for (int i = 0; i < 300; ++i) {
    const double a = 2 * M_PI * u01(rng);
    lms.emplace_back(6.0 * std::cos(a), 6.0 * std::sin(a), -3.0 + 6.0 * u01(rng));
  }
  const gtsam::Vector3 ba(0.05, -0.03, 0.02), bg(0.002, -0.001, 0.0015);
  const double sa = cfg.acc_noise_density * std::sqrt(kImuHz), sg = cfg.gyr_noise_density * std::sqrt(kImuHz);
  const double h = 1e-4;
  const gtsam::Pose3 T_BD = dvlMount();

  std::vector<double> truthT;
  std::vector<gtsam::Pose3> truth;
  Result res;
  double nextCam = 0.5, nextDvl = 0.5;
  for (int i = 0; i <= static_cast<int>(kDuration * kImuHz); ++i) {
    const double t = i / kImuHz;
    const gtsam::Vector3 acc_w = (position(t + h) - 2 * position(t) + position(t - h)) / (h * h);
    const gtsam::Rot3 R = rotation(t);
    const gtsam::Vector3 f_b = R.unrotate(acc_w + gtsam::Vector3(0, 0, kG));
    const gtsam::Vector3 w_b = gtsam::Rot3::Logmap(rotation(t - h).between(rotation(t + h))) / (2 * h);
    const gtsam::Vector3 kick = sc.imu_kick.contains(t) ? gtsam::Vector3(0.15, 0, 0) : gtsam::Vector3::Zero();
    est.addImu({t, f_b + ba + kick + sa * gtsam::Vector3(n01(rng), n01(rng), n01(rng)),
                w_b + bg + sg * gtsam::Vector3(n01(rng), n01(rng), n01(rng))});
    if (sc.feed_dvl && t >= nextDvl - 1e-9) {
      nextDvl += 1.0 / kDvlHz;
      if (!sc.dvl_gap.contains(t)) {
        const gtsam::Vector3 vW = (position(t + h) - position(t - h)) / (2 * h);
        gtsam::Vector3 vD = T_BD.rotation().unrotate(R.unrotate(vW) + w_b.cross(T_BD.translation()));
        if (sc.dvl_outliers.contains(t)) vD = gtsam::Vector3(1.2, -0.8, 0.3);
        else vD += 0.01 * gtsam::Vector3(dvlN01(dvlRng), dvlN01(dvlRng), dvlN01(dvlRng));
        est.addDvl(t, vD, true);
      }
    }
    if (t < nextCam - 1e-9) continue;
    nextCam += 1.0 / kCamHz;
    const gtsam::Pose3 T_WB(R, position(t));
    std::vector<FeatureObs> obs;
    for (size_t j = 0; j < lms.size() && !sc.vision_blackout.contains(t); ++j) {
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
      obs.push_back(f);
    }
    if (!est.addFrame(t, obs)) continue;
    ++res.frames;
    truthT.push_back(t);
    truth.push_back(T_WB);
    res.est.push_back(est.latestState().pose().translation());
  }
  res.counters = est.counters();
  if (truth.empty()) return res;
  // The estimator starts at the origin with yaw 0: express the truth displacement in that frame.
  const gtsam::Rot3 yaw0 = gtsam::Rot3::Yaw(rotation(truthT.front()).yaw());
  gtsam::Vector3 prevErr = gtsam::Vector3::Zero();
  res.max_step_err_m = 0.0;
  for (size_t i = 0; i < truth.size(); ++i) {
    const gtsam::Vector3 err =
        res.est[i] - res.est.front() - yaw0.unrotate(truth[i].translation() - truth.front().translation());
    if (i > 0 && (err - prevErr).norm() > res.max_step_err_m) {
      res.max_step_err_m = (err - prevErr).norm();
      res.max_step_t = truthT[i];
      res.err_before_m = prevErr.norm();
      res.err_after_m = err.norm();
    }
    prevErr = err;
  }
  res.final_drift_m = prevErr.norm();
  return res;
}

void print(const char* name, const Result& r) {
  std::cout << "[ result ] " << name << ": drift " << r.final_drift_m << " m, max step error " << r.max_step_err_m << " m at t=" << r.max_step_t << " (error " << r.err_before_m << " -> " << r.err_after_m << " m)"
            << ", frames " << r.frames << ", dvl factors " << r.counters.dvl_factors << ", gated "
            << r.counters.dvl_gated << ", relocks " << r.counters.dvl_relocks << ", gaps " << r.counters.dvl_gaps
            << ", solver failures " << r.counters.solver_failures << "\n";
}

}  // namespace

TEST(EstimatorDvl, DeadReckonsThroughVisionBlackout) {
  Scenario sc;
  sc.vision_blackout = {15.0, 10.0};
  const Result off = run(sc);
  sc.dvl_enabled = sc.feed_dvl = true;
  const Result on = run(sc);
  print("blackout, no DVL", off);
  print("blackout, DVL   ", on);
  EXPECT_GT(on.counters.dvl_factors, 250u);
  EXPECT_EQ(on.counters.solver_failures, 0u);
  EXPECT_LT(on.final_drift_m, 0.5 * off.final_drift_m);
  EXPECT_LT(on.final_drift_m, 0.15);
}

TEST(EstimatorDvl, RejectsOutlierBurstWithoutJump) {
  Scenario sc;
  sc.dvl_enabled = sc.feed_dvl = true;
  sc.dvl_outliers = {20.0, 3.0};
  const Result r = run(sc);
  print("outlier burst", r);
  EXPECT_GT(r.counters.dvl_gated, 15u);
  EXPECT_EQ(r.counters.dvl_relocks, 0u) << "vision is healthy: the DVL must stay rejected";
  EXPECT_LT(r.max_step_err_m, 0.05);
  EXPECT_LT(r.final_drift_m, 0.15);
}

TEST(EstimatorDvl, DvlGapThenReturnDoesNotJump) {
  Scenario sc;
  sc.dvl_enabled = sc.feed_dvl = true;
  sc.dvl_gap = {20.0, 5.0};
  const Result r = run(sc);
  print("DVL gap, vision on", r);
  EXPECT_EQ(r.counters.dvl_gaps, 1u);
  EXPECT_GT(r.counters.dvl_longest_gap_s, 4.9);
  EXPECT_LT(r.max_step_err_m, 0.05);

  // Worse: the camera is out as well, so only the IMU carries the state across the gap.
  sc.vision_blackout = {17.0, 10.0};
  const Result b = run(sc);
  print("DVL gap inside blackout", b);
  // The returning DVL reveals the velocity drift built up over the gap; the smoother corrects the window
  // retroactively, so the newest pose steps, and the step must move it substantially toward the truth.
  EXPECT_LT(b.err_after_m, 0.5 * b.err_before_m);
  EXPECT_LT(b.final_drift_m, 0.15);
  EXPECT_EQ(b.counters.solver_failures, 0u);
}

TEST(EstimatorDvl, RelocksWhenImuPredictionIsTheCulprit) {
  // The camera is out for 20 s and the DVL is silent for the first 10 of them; an unmodelled
  // accelerometer kick meanwhile makes the IMU prediction 0.3 m/s wrong. The returning DVL is
  // rejected by the gate, but vision is unhealthy, so after dvl_relock_after_s it is trusted.
  Scenario sc;
  sc.dvl_enabled = sc.feed_dvl = true;
  sc.vision_blackout = {15.0, 20.0};
  sc.dvl_gap = {15.0, 10.0};
  sc.imu_kick = {16.0, 2.0};
  const Result relock = run(sc);
  sc.relock_after_s = 1e9;
  const Result never = run(sc);
  print("relock enabled", relock);
  print("relock disabled", never);
  EXPECT_GE(relock.counters.dvl_relocks, 1u);
  EXPECT_EQ(never.counters.dvl_relocks, 0u);
  EXPECT_GT(never.counters.dvl_gated, relock.counters.dvl_gated);
  EXPECT_LT(relock.final_drift_m, 0.5 * never.final_drift_m);
  EXPECT_EQ(relock.counters.solver_failures, 0u);
}

TEST(EstimatorDvl, DvlOffChangesNothing) {
  Scenario plain;
  Scenario fed = plain;
  fed.feed_dvl = true;  // samples reach addDvl, which must ignore them while dvl_enabled is false
  const Result a = run(plain), b = run(fed);
  ASSERT_EQ(a.est.size(), b.est.size());
  for (size_t i = 0; i < a.est.size(); ++i) ASSERT_EQ(a.est[i], b.est[i]) << "frame " << i;
  EXPECT_EQ(b.counters.dvl_received, 0u);
  EXPECT_EQ(b.counters.dvl_factors, 0u);
}
