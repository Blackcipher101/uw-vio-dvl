#include "uw_ct_vio/Estimator.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <boost/math/distributions/chi_squared.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "uw_ct_vio/DenseBetweenFactor.hpp"
#include "uw_ct_vio/DepthGpFactor.hpp"
#include "uw_ct_vio/DvlFactor.hpp"
#include "uw_ct_vio/PartialPosePrior.hpp"

#include <gtsam/navigation/AttitudeFactor.h>

namespace uw_ct_vio {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using ProjFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;

namespace {

constexpr double kInitWindowS = 0.5;       // accelerometer averaging window for roll/pitch
constexpr size_t kInitMinSamples = 10;
constexpr double kInitAccWarnMps2 = 0.3;   // |mean acc| - g beyond this: vehicle was accelerating
constexpr double kMinParallaxRad = 1e-3;   // stereo rays closer than this are not triangulated
constexpr int kMinCandidatesForGateDecision = 10;
// Weak prior on each new landmark at its triangulated position. Projection factors zero their
// Jacobian when a point falls behind a camera after relinearisation; without this prior such a
// landmark is unconstrained and iSAM2 throws IndeterminantLinearSystemException. 5 m is ~1e8
// times weaker than one pixel measurement, so it does not bias the estimate.
constexpr double kLandmarkPriorSigmaM = 5.0;

// Throttled stderr warning: first few, then every 100th.
void warn(uint64_t count, const std::string& msg) {
  if (count <= 5 || count % 100 == 0) std::cerr << "[uw_ct_vio] WARN (" << count << "x): " << msg << "\n";
}

}  // namespace

Estimator::Estimator(const EstimatorConfig& config) : cfg_(config) {
  if (!(cfg_.lag_s > 0.0) || !(cfg_.pixel_sigma > 0.0) || !(cfg_.focal_px > 0.0) ||
      !(cfg_.min_depth_m > 0.0 && cfg_.max_depth_m > cfg_.min_depth_m)) {
    throw std::invalid_argument("EstimatorConfig: need lag_s, pixel_sigma, focal_px > 0 and 0 < min_depth < max_depth");
  }
  if (!(cfg_.dense_cov_scale > 0.0) || !(cfg_.dense_min_eig > 0.0) ||
      !(cfg_.dense_gate_quantile > 0.0 && cfg_.dense_gate_quantile < 1.0)) {
    throw std::invalid_argument("EstimatorConfig: need dense_cov_scale, dense_min_eig > 0 and 0 < dense_gate_quantile < 1");
  }
  if (cfg_.dvl_enabled) {
    if (!cfg_.dvl_T_BD.matrix().allFinite() || !(cfg_.dvl_sigma_abs_mps > 0.0) || !(cfg_.dvl_sigma_rel >= 0.0) ||
        !(cfg_.dvl_max_speed_mps > 0.0) || !(cfg_.dvl_gate_pred_sigma_mps >= 0.0) || !(cfg_.dvl_relock_after_s >= 0.0) ||
        !(cfg_.dvl_recent_s >= 0.0) || !(cfg_.dvl_recovery_vel_sigma_mps > 0.0) || !(cfg_.dvl_gap_s > 0.0) ||
        !(cfg_.dvl_gate_quantile > 0.0 && cfg_.dvl_gate_quantile < 1.0) || !std::isfinite(cfg_.dvl_time_offset_s)) {
      throw std::invalid_argument(
          "EstimatorConfig: invalid DVL settings (need finite extrinsic and time offset, sigma_abs, max_speed, gap_s, "
          "recovery sigma > 0, rel / pred sigma / relock / recent >= 0, 0 < gate_quantile < 1)");
    }
    dvlGateLimit_ = boost::math::quantile(boost::math::chi_squared(3), cfg_.dvl_gate_quantile);
  }
  imuParams_ = gtsam::PreintegrationCombinedParams::MakeSharedU(cfg_.gravity_mps2);
  imuParams_->accelerometerCovariance = gtsam::I_3x3 * std::pow(cfg_.acc_noise_density, 2);
  imuParams_->gyroscopeCovariance = gtsam::I_3x3 * std::pow(cfg_.gyr_noise_density, 2);
  imuParams_->integrationCovariance = gtsam::I_3x3 * std::pow(cfg_.integration_sigma, 2);
  imuParams_->biasAccCovariance = gtsam::I_3x3 * std::pow(cfg_.acc_random_walk, 2);
  imuParams_->biasOmegaCovariance = gtsam::I_3x3 * std::pow(cfg_.gyr_random_walk, 2);
  imuParams_->biasAccOmegaInt = gtsam::I_6x6 * 1e-5;

  K_ = boost::make_shared<gtsam::Cal3_S2>(1.0, 1.0, 0.0, 0.0, 0.0);
  pixelNoise_ = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(cfg_.huber_k),
                                                  gtsam::noiseModel::Isotropic::Sigma(2, cfg_.pixel_sigma / cfg_.focal_px));
  depthNoise_ = gtsam::noiseModel::Isotropic::Sigma(1, cfg_.depth_sigma_m);

  gtsam::ISAM2Params isam;
  isam.relinearizeThreshold = 0.1;
  isam.relinearizeSkip = 1;
  smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(cfg_.lag_s, isam);
}

void Estimator::addImu(const ImuSample& s) {
  if (!std::isfinite(s.t) || !s.acc.allFinite() || !s.gyr.allFinite() || (!imu_.empty() && s.t <= imu_.back().t)) {
    warn(++counters_.imu_dropped, "dropped non-finite or out-of-order IMU sample");
    return;
  }
  imu_.push_back(s);
}

void Estimator::addDepth(double t, double depth_m) {
  if (!std::isfinite(t) || !std::isfinite(depth_m) || (!depth_.empty() && t <= depth_.back().first)) {
    warn(++counters_.depth_dropped, "dropped non-finite or out-of-order depth sample");
    return;
  }
  depth_.emplace_back(t, depth_m);
}

void Estimator::addDvl(double t, const gtsam::Vector3& v_D, bool valid, const std::optional<gtsam::Matrix3>& cov) {
  if (!cfg_.dvl_enabled) return;
  ++counters_.dvl_received;
  const double te = t + cfg_.dvl_time_offset_s;
  if (!std::isfinite(te) || !v_D.allFinite()) {
    warn(++counters_.dvl_invalid, "dropped non-finite DVL sample");
    return;
  }
  if (!dvl_.empty() && te <= dvl_.back().t) {
    warn(++counters_.dvl_invalid, "dropped out-of-order DVL sample");
    return;
  }
  if (!valid) {
    warn(++counters_.dvl_invalid, "dropped DVL sample without bottom lock (valid=0)");
    return;
  }
  if (v_D.norm() > cfg_.dvl_max_speed_mps) {
    warn(++counters_.dvl_invalid, "dropped DVL sample faster than dvl_max_speed_mps: " + std::to_string(v_D.norm()) + " m/s");
    return;
  }
  gtsam::Vector3 sigma;
  for (int i = 0; i < 3; ++i)
    sigma(i) = std::sqrt(std::pow(cfg_.dvl_sigma_abs_mps, 2) + std::pow(cfg_.dvl_sigma_rel * std::abs(v_D(i)), 2));
  if (cov && cov->allFinite() && cov->diagonal().minCoeff() > 0.0) sigma = cov->diagonal().cwiseSqrt();
  dvl_.push_back({te, v_D, sigma});
}

gtsam::Vector3 Estimator::gyroAt(double t) const {
  size_t i = 1;
  while (i + 1 < imu_.size() && imu_[i].t < t) ++i;
  const ImuSample& a = imu_[std::min(i - 1, imu_.size() - 1)];
  const ImuSample& b = imu_[std::min(i, imu_.size() - 1)];
  const double w = b.t > a.t ? std::clamp((t - a.t) / (b.t - a.t), 0.0, 1.0) : 0.0;
  return (1 - w) * a.gyr + w * b.gyr - latestBias_.gyroscope();
}

bool Estimator::initialize(double t) {
  gtsam::Vector3 sum = gtsam::Vector3::Zero();
  size_t n = 0;
  for (const auto& s : imu_) {
    if (s.t > t - kInitWindowS && s.t <= t) {
      sum += s.acc;
      ++n;
    }
  }
  if (n < kInitMinSamples) return false;
  const gtsam::Vector3 f = sum / static_cast<double>(n);  // specific force, ~ R^T * (0, 0, g) when still
  if (std::abs(f.norm() - cfg_.gravity_mps2) > kInitAccWarnMps2) {
    std::cerr << "[uw_ct_vio] WARN: initial accelerometer norm " << f.norm() << " vs g " << cfg_.gravity_mps2
              << "; roll/pitch initialisation will be biased by vehicle acceleration\n";
  }
  // Rotation taking the measured up-direction (body) onto world +z (Rodrigues); yaw set to 0.
  const gtsam::Vector3 a = f.normalized(), z = gtsam::Vector3::UnitZ(), v = a.cross(z);
  const double c = a.dot(z), s2 = v.squaredNorm();
  gtsam::Matrix3 Ralign = gtsam::I_3x3;
  if (s2 > 1e-12) {
    const gtsam::Matrix3 vx = gtsam::skewSymmetric(v);
    Ralign += vx + vx * vx * ((1.0 - c) / s2);
  } else if (c < 0) {
    Ralign = gtsam::Rot3::Rx(M_PI).matrix();  // upside down
  }
  const gtsam::Rot3 R_align(Ralign);
  const gtsam::Vector3 rpy = R_align.rpy();
  const gtsam::Rot3 R0 = gtsam::Rot3::RzRyRx(rpy.x(), rpy.y(), 0.0);

  latest_ = gtsam::NavState(gtsam::Pose3(R0, gtsam::Point3::Zero()), gtsam::Vector3::Zero());
  latestBias_ = gtsam::imuBias::ConstantBias();
  k_ = 0;
  tFrame_ = t;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::FixedLagSmoother::KeyTimestampMap stamps;
  gtsam::Vector6 poseSigmas;
  poseSigmas << cfg_.prior_rot_sigma_rad, cfg_.prior_rot_sigma_rad, cfg_.prior_yaw_sigma_rad,
      gtsam::Vector3::Constant(cfg_.prior_pos_sigma_m);
  gtsam::Vector6 biasSigmas;
  biasSigmas << gtsam::Vector3::Constant(cfg_.prior_acc_bias_sigma), gtsam::Vector3::Constant(cfg_.prior_gyr_bias_sigma);
  graph.addPrior(X(0), latest_.pose(), gtsam::noiseModel::Diagonal::Sigmas(poseSigmas));
  graph.addPrior(V(0), latest_.v(), gtsam::noiseModel::Isotropic::Sigma(3, cfg_.prior_vel_sigma_mps));
  graph.addPrior(B(0), latestBias_, gtsam::noiseModel::Diagonal::Sigmas(biasSigmas));
  values.insert(X(0), latest_.pose());
  values.insert(V(0), latest_.v());
  values.insert(B(0), latestBias_);
  stamps[X(0)] = stamps[V(0)] = stamps[B(0)] = t;
  smoother_->update(graph, values, stamps);
  stateIndex_.clear();
  stateIndex_[t] = 0;
  initialized_ = true;
  return true;
}

void Estimator::reset(double t_last, double vel_sigma_mps, bool zero_velocity) {
  if (zero_velocity) latest_ = gtsam::NavState(latest_.pose(), gtsam::Vector3::Zero());
  gtsam::ISAM2Params isam;
  isam.relinearizeThreshold = 0.1;
  isam.relinearizeSkip = 1;
  smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(cfg_.lag_s, isam);
  landmarks_.clear();
  ++k_;  // fresh keys: the old smoother's variables are gone
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::FixedLagSmoother::KeyTimestampMap stamps;
  gtsam::Vector6 poseSigmas;
  poseSigmas << gtsam::Vector3::Constant(cfg_.prior_rot_sigma_rad), gtsam::Vector3::Constant(0.05);
  gtsam::Vector6 biasSigmas;
  biasSigmas << gtsam::Vector3::Constant(cfg_.prior_acc_bias_sigma), gtsam::Vector3::Constant(cfg_.prior_gyr_bias_sigma);
  dvlSkipGate_ = true;
  dvlRejectSince_.reset();
  graph.addPrior(X(k_), latest_.pose(), gtsam::noiseModel::Diagonal::Sigmas(poseSigmas));
  graph.addPrior(V(k_), latest_.v(), gtsam::noiseModel::Isotropic::Sigma(3, vel_sigma_mps));
  graph.addPrior(B(k_), latestBias_, gtsam::noiseModel::Diagonal::Sigmas(biasSigmas));
  values.insert(X(k_), latest_.pose());
  values.insert(V(k_), latest_.v());
  values.insert(B(k_), latestBias_);
  stamps[X(k_)] = stamps[V(k_)] = stamps[B(k_)] = t_last;
  smoother_->update(graph, values, stamps);
  stateIndex_.clear();
  stateIndex_[t_last] = k_;
}

bool Estimator::preintegrate(double t0, double t1, gtsam::PreintegratedCombinedMeasurements* pim) {
  double maxGap = 0.0;
  if (!integrateImu(t0, t1, pim, &maxGap)) return false;
  if (maxGap > cfg_.max_imu_gap_s) warn(++counters_.imu_gaps, "IMU gap of " + std::to_string(maxGap) + " s bridged");
  // Keep one sample at or before t1 for the next interval.
  while (imu_.size() > 2 && imu_[1].t <= t1) imu_.pop_front();
  return true;
}

bool Estimator::integrateImu(double t0, double t1, gtsam::PreintegratedCombinedMeasurements* pim,
                             double* max_gap_s) const {
  *max_gap_s = 0.0;
  if (imu_.empty() || imu_.back().t < t1 || imu_.front().t > t0) return false;
  // Break points: t0, every sample time inside (t0, t1), t1. Each segment uses the IMU value
  // linearly interpolated at its midpoint.
  auto at = [this](double t) {
    size_t i = 1;
    while (i + 1 < imu_.size() && imu_[i].t < t) ++i;
    const ImuSample& a = imu_[i - 1];
    const ImuSample& b = imu_[i];
    const double w = std::clamp((t - a.t) / (b.t - a.t), 0.0, 1.0);
    return std::make_pair(gtsam::Vector3((1 - w) * a.acc + w * b.acc), gtsam::Vector3((1 - w) * a.gyr + w * b.gyr));
  };
  std::vector<double> bp{t0};
  for (const auto& s : imu_)
    if (s.t > t0 && s.t < t1) bp.push_back(s.t);
  bp.push_back(t1);
  for (size_t i = 0; i + 1 < bp.size(); ++i) {
    const double dt = bp[i + 1] - bp[i];
    if (dt <= 0) continue;
    *max_gap_s = std::max(*max_gap_s, dt);
    const auto m = at(0.5 * (bp[i] + bp[i + 1]));
    pim->integrateMeasurement(m.first, m.second, dt);
  }
  return true;
}

std::optional<gtsam::Pose3> Estimator::predictLeftCamera(double t) const {
  if (!initialized_ || !(t > tFrame_)) return std::nullopt;
  gtsam::PreintegratedCombinedMeasurements pim(imuParams_, latestBias_);
  double maxGap = 0.0;
  if (!integrateImu(tFrame_, t, &pim, &maxGap)) return std::nullopt;
  return pim.predict(latest_, latestBias_).pose() * cfg_.T_BC0;
}

std::optional<gtsam::Pose3> Estimator::leftCameraAt(double t) const {
  const auto it = stateIndex_.find(t);
  if (it == stateIndex_.end()) return std::nullopt;
  try {
    return smoother_->calculateEstimate<gtsam::Pose3>(X(it->second)) * cfg_.T_BC0;
  } catch (const std::exception&) {  // marginalised meanwhile
    return std::nullopt;
  }
}

std::vector<std::pair<uint64_t, gtsam::Point3>> Estimator::liveLandmarks() const {
  std::vector<std::pair<uint64_t, gtsam::Point3>> out;
  out.reserve(landmarks_.size());
  for (const auto& [id, lm] : landmarks_) {
    try {
      out.emplace_back(id, smoother_->calculateEstimate<gtsam::Point3>(lm.key));
    } catch (const std::exception&) {  // marginalised: addFrame forgets it on its next sighting
    }
  }
  return out;
}

gtsam::Point3 Estimator::triangulate(const FeatureObs& f, const gtsam::Pose3& T_WB, bool* ok) const {
  *ok = false;
  const gtsam::Pose3 T_C0C1 = cfg_.T_BC0.between(cfg_.T_BC1);
  const gtsam::Vector3 d0(f.left.x(), f.left.y(), 1.0);
  const gtsam::Vector3 d1 = T_C0C1.rotation().matrix() * gtsam::Vector3(f.right.x(), f.right.y(), 1.0);
  const gtsam::Vector3 c1 = T_C0C1.translation();
  // Closest points of the two rays: minimise |l0 d0 - (c1 + l1 d1)|.
  gtsam::Matrix32 A;
  A << d0, -d1;
  const gtsam::Vector2 l = A.colPivHouseholderQr().solve(c1);
  const double parallax = std::acos(std::clamp(d0.normalized().dot(d1.normalized()), -1.0, 1.0));
  if (!(l.allFinite()) || l(0) <= 0 || l(1) <= 0 || parallax < kMinParallaxRad) return gtsam::Point3();
  const gtsam::Point3 p_C0 = 0.5 * (l(0) * d0 + c1 + l(1) * d1);
  if (p_C0.z() < cfg_.min_depth_m || p_C0.z() > cfg_.max_depth_m) return gtsam::Point3();
  *ok = true;
  return (T_WB * cfg_.T_BC0).transformFrom(p_C0);
}

bool Estimator::addFrame(double t, const std::vector<FeatureObs>& features, FrameStats* stats,
                         const DenseConstraint* dense) {
  FrameStats local;
  FrameStats& st = stats ? *stats : local;
  st = FrameStats();
  st.observations = static_cast<int>(features.size());

  gtsam::NonlinearFactorGraph imuGraph, visionGraph;
  gtsam::Values values;
  gtsam::FixedLagSmoother::KeyTimestampMap stamps;
  gtsam::Pose3 T_WB_pred;
  gtsam::NavState navPred;
  std::vector<std::pair<DvlSample, gtsam::Vector3>> dvlBatch;  // samples of this interval with their body rates

  if (!initialized_) {
    if (!initialize(t)) return false;
    T_WB_pred = latest_.pose();
  } else {
    if (!(t > tFrame_)) return false;
    // Mean specific force and rotation rate over the interval, for the gravity prior.
    gtsam::Vector3 accSum = gtsam::Vector3::Zero(), gyrSum = gtsam::Vector3::Zero();
    int nImu = 0;
    for (const auto& s : imu_)
      if (s.t > tFrame_ && s.t <= t) accSum += s.acc, gyrSum += s.gyr, ++nImu;
    // Body rate at each DVL stamp, read before preintegrate() prunes imu_.
    if (cfg_.dvl_enabled && !imu_.empty()) {
      for (const DvlSample& d : dvl_)
        if (d.t > tFrame_ && d.t <= t) dvlBatch.emplace_back(d, gyroAt(d.t));
    }
    gtsam::PreintegratedCombinedMeasurements pim(imuParams_, latestBias_);
    if (!preintegrate(tFrame_, t, &pim)) return false;
    while (!dvl_.empty() && dvl_.front().t <= t) {
      if (dvl_.front().t <= tFrame_) warn(++counters_.dvl_invalid, "dropped stale DVL sample (older than the previous frame)");
      dvl_.pop_front();
    }
    const gtsam::NavState pred = pim.predict(latest_, latestBias_);
    navPred = pred;
    const uint64_t k = k_ + 1;
    imuGraph.emplace_shared<gtsam::CombinedImuFactor>(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim);
    values.insert(X(k), pred.pose());
    values.insert(V(k), pred.v());
    values.insert(B(k), latestBias_);
    stamps[X(k)] = stamps[V(k)] = stamps[B(k)] = t;
    // Depth readings since the last frame: continuous-time factors between the two states.
    if (!depthSurfaceZ_ && !depth_.empty()) {
      // Surface height from the first reading after initialisation (body z ~ known there).
      const auto& [td, d] = depth_.front();
      depthSurfaceZ_ = latest_.pose().z() + d;
      (void)td;
    }
    while (!depth_.empty() && depth_.front().first <= t) {
      const auto [td, d] = depth_.front();
      depth_.pop_front();
      if (td <= tFrame_ || !depthSurfaceZ_) continue;  // before this interval: too late to use
      imuGraph.emplace_shared<DepthGpFactor>(X(k - 1), V(k - 1), X(k), V(k), (td - tFrame_) / (t - tFrame_),
                                             t - tFrame_, d, *depthSurfaceZ_, cfg_.depth_lever_body, depthNoise_);
    }
    if (cfg_.gravity_prior && nImu > 0) {
      const gtsam::Vector3 f = accSum / nImu - latestBias_.accelerometer();
      const double w = (gyrSum / nImu - latestBias_.gyroscope()).norm();
      const double sigma = cfg_.gravity_sigma_base_rad + cfg_.gravity_sigma_per_mps2 * std::abs(f.norm() - cfg_.gravity_mps2) +
                           cfg_.gravity_sigma_per_radps * w;
      if (sigma <= cfg_.gravity_sigma_max_rad && f.norm() > 1e-3) {
        // At rest the specific force points up in the world: R * f_body ~ (0, 0, 1).
        imuGraph.emplace_shared<gtsam::Pose3AttitudeFactor>(X(k), gtsam::Unit3(0, 0, 1),
                                                            gtsam::noiseModel::Isotropic::Sigma(2, sigma), gtsam::Unit3(f));
        ++counters_.gravity_factors;
      }
    }
    k_ = k;
    T_WB_pred = pred.pose();
  }

  // Vision: existing landmarks (gated against the IMU prediction), then new stereo landmarks.
  const gtsam::Pose3 T_WC0 = T_WB_pred * cfg_.T_BC0;
  // Per-observation noise: the tracker may report a larger sigma for low-contrast patches.
  auto noiseFor = [this](const FeatureObs& f) -> gtsam::SharedNoiseModel {
    if (!(f.sigma_px > 0.0) || f.sigma_px == cfg_.pixel_sigma) return pixelNoise_;
    return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(cfg_.huber_k),
                                             gtsam::noiseModel::Isotropic::Sigma(2, f.sigma_px / cfg_.focal_px));
  };
  int candidates = 0, stereo = 0;
  gtsam::Values newLandmarks;
  std::vector<std::pair<uint64_t, Landmark>> created;
  std::vector<uint64_t> refreshed;
  gtsam::Matrix66 visInfo = gtsam::Matrix66::Zero();  // information vision gives the new pose
  for (const FeatureObs& f : features) {
    if (f.has_right) ++stereo;
    auto it = landmarks_.find(f.track_id);
    if (it != landmarks_.end()) {
      ++candidates;
      gtsam::Point3 P;
      try {
        P = smoother_->calculateEstimate<gtsam::Point3>(it->second.key);
      } catch (const std::exception&) {  // marginalised meanwhile
        landmarks_.erase(it);
        continue;
      }
      const gtsam::Point3 p = T_WC0.transformTo(P);
      const double sigmaPx = f.sigma_px > 0.0 ? f.sigma_px : cfg_.pixel_sigma;
      const double gateNorm = cfg_.gate_sigma * sigmaPx / cfg_.focal_px;
      if (p.z() < cfg_.min_depth_m || (gtsam::Point2(p.x() / p.z(), p.y() / p.z()) - f.left).norm() > gateNorm) {
        ++st.gated;
        continue;
      }
      const auto noise = noiseFor(f);
      visionGraph.emplace_shared<ProjFactor>(f.left, noise, X(k_), it->second.key, K_, cfg_.T_BC0);
      if (f.has_right)
        visionGraph.emplace_shared<ProjFactor>(f.right, noise, X(k_), it->second.key, K_, cfg_.T_BC1);
      if (cfg_.degeneracy_aware) {
        // Landmark held at its estimate: the pose Jacobian of each projection, whitened.
        const double s = (f.sigma_px > 0.0 ? f.sigma_px : cfg_.pixel_sigma) / cfg_.focal_px;
        for (int cam = 0; cam < (f.has_right ? 2 : 1); ++cam) {
          ProjFactor pf(cam == 0 ? f.left : f.right, gtsam::noiseModel::Isotropic::Sigma(2, s), X(k_), it->second.key, K_,
                        cam == 0 ? cfg_.T_BC0 : cfg_.T_BC1);
          gtsam::Matrix Hx, Hl;
          pf.evaluateError(T_WB_pred, P, Hx, Hl);
          visInfo += Hx.transpose() * Hx / (s * s);
        }
      }
      refreshed.push_back(f.track_id);
      ++st.landmark_factors;
    } else if (f.has_right) {
      bool ok = false;
      const gtsam::Point3 Pw = triangulate(f, T_WB_pred, &ok);
      if (!ok) continue;
      const gtsam::Key key = L(nextLandmark_++);
      newLandmarks.insert(key, Pw);
      visionGraph.addPrior(key, Pw, gtsam::noiseModel::Isotropic::Sigma(3, kLandmarkPriorSigmaM));
      const auto noise = noiseFor(f);
      visionGraph.emplace_shared<ProjFactor>(f.left, noise, X(k_), key, K_, cfg_.T_BC0);
      visionGraph.emplace_shared<ProjFactor>(f.right, noise, X(k_), key, K_, cfg_.T_BC1);
      created.emplace_back(f.track_id, Landmark{key, t});
      ++st.new_landmarks;
    }
  }
  st.candidates = candidates;
  st.stereo = stereo;
  if (!visionStarted_ && stereo >= cfg_.min_stereo_to_start) visionStarted_ = true;
  st.vision_used = visionStarted_ && !(candidates >= kMinCandidatesForGateDecision &&
                                       st.gated > cfg_.max_gated_fraction * candidates);
  if (!st.vision_used && visionStarted_) warn(++counters_.frames_no_vision, "frame vision factors dropped by the gate");
  consecutiveGated_ = (!st.vision_used && visionStarted_) ? consecutiveGated_ + 1 : 0;
  st.vision_healthy = st.vision_used && st.landmark_factors >= cfg_.vision_healthy_min_factors;

  // Dense registration: its own acceptance (Mahalanobis against the prior estimates in the kept
  // subspace), independent of the point-vision gate, since low-texture frames are what it is for.
  boost::shared_ptr<DenseBetweenFactor> denseFactor;
  DenseSubspace denseSub;
  if (dense) {
    const auto ref = stateIndex_.find(dense->t_ref);
    std::optional<gtsam::Pose3> X_ref;
    if (ref != stateIndex_.end() && ref->second != k_ && dense->t_ref >= t - cfg_.lag_s + 1e-6) {
      try {
        X_ref = smoother_->calculateEstimate<gtsam::Pose3>(X(ref->second));
      } catch (const std::exception&) {  // marginalised meanwhile
      }
    }
    if (!X_ref) {
      warn(++counters_.dense_stale, "dense constraint dropped: reference frame t=" + std::to_string(dense->t_ref) + " left the window");
    } else if (!dense->T_Cref_Ccur.matrix().allFinite() || !dense->info_cam.allFinite()) {
      warn(++counters_.dense_invalid, "dense constraint dropped: non-finite registration");
    } else {
      denseSub = denseSubspace(dense->info_cam, cfg_.dense_min_eig);
      if (denseSub.rank() > 0) {
        auto f = boost::make_shared<DenseBetweenFactor>(X(ref->second), X(k_), dense->T_Cref_Ccur, cfg_.T_BC0, denseSub,
                                                        cfg_.dense_cov_scale, cfg_.huber_k);
        const double limit = boost::math::quantile(boost::math::chi_squared(denseSub.rank()), cfg_.dense_gate_quantile);
        if (f->mahalanobisSq(*X_ref, T_WB_pred) <= limit) {
          denseFactor = f;
        } else {
          warn(++counters_.dense_gated, "dense constraint rejected: disagrees with the estimate beyond the chi-square gate");
        }
      }
    }
  }

  if (denseFactor && st.vision_used && cfg_.degeneracy_aware) visInfo += denseInfoInBody(denseSub, cfg_.dense_cov_scale, cfg_.T_BC0);

  if (st.vision_used && cfg_.degeneracy_aware && k_ > 0) {
    // Directions vision pins down worse than degeneracy_sigma_max stay with the IMU prediction.
    Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> es(visInfo);
    const double lamMin = 1.0 / (cfg_.degeneracy_sigma_max * cfg_.degeneracy_sigma_max);
    std::vector<int> weak;
    for (int i = 0; i < 6; ++i)
      if (es.eigenvalues()(i) < lamMin) weak.push_back(i);
    if (!weak.empty()) {
      gtsam::Matrix U(6, weak.size());
      for (size_t j = 0; j < weak.size(); ++j) U.col(j) = es.eigenvectors().col(weak[j]);
      visionGraph.emplace_shared<PartialPosePrior>(X(k_), T_WB_pred, U, cfg_.degeneracy_prior_sigma);
      ++counters_.degenerate_frames;
      counters_.degenerate_dirs += weak.size();
    }
  }

  // DVL: continuous-time velocity factors between the two states. Gated against the IMU prediction;
  // placed in imuGraph so they carry the state through frames whose vision was gated out or empty.
  if (!dvlBatch.empty()) {
    const uint64_t k = k_;
    const double dt = t - tFrame_;
    const double chi2Floor = std::pow(cfg_.dvl_gate_pred_sigma_mps, 2);
    for (const auto& [d, omega] : dvlBatch) {
      const auto noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(cfg_.huber_k),
                                                           gtsam::noiseModel::Diagonal::Sigmas(d.sigma));
      const auto f = boost::make_shared<DvlFactor>(X(k - 1), V(k - 1), X(k), V(k), (d.t - tFrame_) / dt, dt, d.v, omega,
                                                   cfg_.dvl_T_BD, noise);
      bool accept = true, relock = false;
      if (!dvlSkipGate_) {
        const gtsam::Vector r = f->evaluateError(latest_.pose(), latest_.v(), navPred.pose(), navPred.v());
        double chi2 = 0.0;
        for (int i = 0; i < 3; ++i) chi2 += r(i) * r(i) / (d.sigma(i) * d.sigma(i) + chi2Floor);
        if (chi2 > dvlGateLimit_) {
          if (!dvlRejectSince_) dvlRejectSince_ = d.t;
          relock = !st.vision_healthy && d.t - *dvlRejectSince_ >= cfg_.dvl_relock_after_s;
          accept = relock;
          if (!accept) {
            ++st.dvl_gated;
            warn(++counters_.dvl_gated, "DVL sample at t=" + std::to_string(d.t) + " rejected by the innovation gate (chi2 " +
                                            std::to_string(chi2) + " > " + std::to_string(dvlGateLimit_) + ")");
          }
        }
      }
      if (!accept) continue;
      if (relock) {
        warn(++counters_.dvl_relocks, "DVL re-lock: " + std::to_string(d.t - *dvlRejectSince_) +
                                          " s of rejections with unhealthy vision; trusting the DVL over the IMU prediction");
      }
      dvlRejectSince_.reset();
      dvlSkipGate_ = false;
      if (dvlLastAccepted_ && d.t - *dvlLastAccepted_ > cfg_.dvl_gap_s) {
        const double gap = d.t - *dvlLastAccepted_;
        warn(++counters_.dvl_gaps, "DVL gap of " + std::to_string(gap) + " s");
        counters_.dvl_longest_gap_s = std::max(counters_.dvl_longest_gap_s, gap);
      }
      dvlLastAccepted_ = d.t;
      imuGraph.push_back(f);
      ++st.dvl_factors;
      ++counters_.dvl_factors;
    }
  }

  gtsam::NonlinearFactorGraph graph = imuGraph;
  gtsam::Values all = values;
  if (denseFactor) graph.push_back(denseFactor);
  if (st.vision_used) {
    graph.add(visionGraph);
    all.insert(newLandmarks);
    for (const auto& [id, lm] : created) stamps[lm.key] = t;
    for (uint64_t id : refreshed) stamps[landmarks_.at(id).key] = t;
  }
  try {
    smoother_->update(graph, all, stamps);
  } catch (const std::exception& e) {
    // iSAM2 has already absorbed the new variables, so the update cannot be retried. Restart the
    // smoother from the last good state and drop this frame.
    warn(++counters_.solver_failures, std::string("smoother update failed; resetting from the last state: ") + e.what());
    reset(tFrame_);
    return false;
  }
  if (denseFactor) {
    ++counters_.dense_factors;
    counters_.dense_rank_sum += denseSub.rank();
  }
  stateIndex_[t] = k_;
  stateIndex_.erase(stateIndex_.begin(), stateIndex_.lower_bound(t - cfg_.lag_s + 1e-6));
  if (st.vision_used) {
    for (const auto& [id, lm] : created) landmarks_[id] = lm;
    for (uint64_t id : refreshed) landmarks_.at(id).last_seen = t;
  }
  // Landmarks about to leave the window are forgotten; a re-observed track starts a new landmark.
  for (auto it = landmarks_.begin(); it != landmarks_.end();) {
    it = (it->second.last_seen < t - cfg_.lag_s + 1e-6) ? landmarks_.erase(it) : std::next(it);
  }
  latest_ = gtsam::NavState(smoother_->calculateEstimate<gtsam::Pose3>(X(k_)),
                            smoother_->calculateEstimate<gtsam::Vector3>(V(k_)));
  latestBias_ = smoother_->calculateEstimate<gtsam::imuBias::ConstantBias>(B(k_));
  tFrame_ = t;
  if (st.vision_used) lastGoodVelocity_ = latest_.v();
  if (consecutiveGated_ >= cfg_.max_consecutive_gated) {
    // Vision has disagreed with the IMU prediction for too long: the prediction (velocity) is the
    // likelier culprit. Keep the pose, loosen velocity, drop landmarks so vision re-anchors.
    warn(++counters_.gate_recoveries, "vision gated for " + std::to_string(consecutiveGated_) +
                                          " frames; restarting with a loose velocity prior");
    if (cfg_.dvl_enabled && dvlLastAccepted_ && t - *dvlLastAccepted_ <= cfg_.dvl_recent_s) {
      // The DVL has just constrained the velocity: keep it instead of zeroing, with a tight prior.
      warn(++counters_.dvl_recoveries_kept, "restart keeps the DVL-constrained velocity");
      reset(t, cfg_.dvl_recovery_vel_sigma_mps, false);
      dvlSkipGate_ = false;
    } else if (cfg_.recovery_keep_velocity) {
      latest_ = gtsam::NavState(latest_.pose(), lastGoodVelocity_);
      reset(t, cfg_.recovery_vel_sigma_mps, false);
    } else {
      reset(t, cfg_.recovery_vel_sigma_mps, true);
    }
    consecutiveGated_ = 0;
  }
  return true;
}

}  // namespace uw_ct_vio
