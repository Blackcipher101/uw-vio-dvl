// Fixed-lag stereo-inertial(-depth) estimator on GTSAM.
#pragma once

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace uw_ct_vio {

/// Configuration. Frames: world z-up (gravity -z), body = IMU, cameras are OpenCV optical frames.
struct EstimatorConfig {
  gtsam::Pose3 T_BC0, T_BC1;  ///< left / right camera in the body (IMU) frame
  double gravity_mps2 = 9.81;
  // IMU noise (Kalibr continuous-time densities)
  double acc_noise_density = 0.02, gyr_noise_density = 0.002;     // [m/s^2/sqrt(Hz)], [rad/s/sqrt(Hz)]
  double acc_random_walk = 0.004, gyr_random_walk = 0.0004;       // [m/s^3/sqrt(Hz)], [rad/s^2/sqrt(Hz)]
  double integration_sigma = 1e-4;
  // Priors at start
  double prior_rot_sigma_rad = 0.05, prior_yaw_sigma_rad = 1e-3, prior_pos_sigma_m = 1e-3;
  double prior_vel_sigma_mps = 0.5, prior_acc_bias_sigma = 0.1, prior_gyr_bias_sigma = 0.01;
  // Vision
  double pixel_sigma = 1.0;          ///< reprojection noise [px], converted with focal_px below
  double focal_px = 460.0;           ///< typical focal length, to turn pixels into normalised units
  double huber_k = 1.345;
  double min_depth_m = 0.2, max_depth_m = 15.0;
  double gate_sigma = 5.0;           ///< drop observations predicted worse than this many sigma
  double max_gated_fraction = 0.7;   ///< more dropped than this: frame adds no vision factors
  int min_stereo_to_start = 15;      ///< stereo landmarks needed before vision factors start
  /// After this many consecutive frames whose vision was gated out, the IMU prediction is assumed to
  /// have drifted: restart from the last pose with a loose velocity prior and fresh landmarks.
  int max_consecutive_gated = 10;
  double recovery_vel_sigma_mps = 0.5;
  // Step 3 switches.
  /// Roll/pitch from the mean specific force over each frame interval (GeVI-SLAM style); sigma grows
  /// with |a| - g and with the rotation rate, and the factor is skipped above gravity_sigma_max_rad.
  bool gravity_prior = false;
  double gravity_sigma_base_rad = 0.02, gravity_sigma_per_mps2 = 0.2, gravity_sigma_per_radps = 0.5,
         gravity_sigma_max_rad = 0.2;
  /// Directions of the new pose that this frame's vision constrains worse than degeneracy_sigma_max
  /// are held to the IMU prediction by a partial prior (Hinduja et al. style).
  bool degeneracy_aware = false;
  double degeneracy_sigma_max = 0.05;       ///< [rad or m] along an eigen-direction of the visual information
  double degeneracy_prior_sigma = 0.02;
  /// On a gate recovery keep the last velocity estimated while vision was in use, instead of zero.
  bool recovery_keep_velocity = false;
  // Step 6: dense stereo registration factors (the registration itself lives in DenseRegistration).
  /// Directions of the registration Hessian (left camera frame, [rad; m] units) below this eigenvalue are not
  /// constrained by the factor; sigma_i = dense_cov_scale / sqrt(lambda_i) for the others.
  double dense_min_eig = 1e3;
  double dense_cov_scale = 3.0;        ///< inflates the (over-confident) ICP covariance, sigma multiplier
  double dense_gate_quantile = 0.999;  ///< registrations beyond this chi-square quantile of Mahalanobis distance are rejected
  // Depth sensor
  double depth_sigma_m = 0.02;
  gtsam::Vector3 depth_lever_body = gtsam::Vector3::Zero();
  // DVL (tightly coupled velocity factors, DvlFactor). Off unless dvl_enabled.
  bool dvl_enabled = false;
  gtsam::Pose3 dvl_T_BD;                    ///< DVL frame in the body (IMU) frame
  double dvl_sigma_abs_mps = 0.01;          ///< per-axis noise sigma = sqrt(abs^2 + (rel * |v|)^2)
  double dvl_sigma_rel = 0.01;
  double dvl_time_offset_s = 0.0;           ///< added to DVL stamps to bring them into IMU time
  double dvl_max_speed_mps = 5.0;           ///< faster samples are dropped as invalid
  /// Innovation gate: the residual at the IMU prediction, whitened with sigma_meas^2 +
  /// dvl_gate_pred_sigma_mps^2, must stay below this chi-square(3) quantile.
  double dvl_gate_quantile = 0.999;
  double dvl_gate_pred_sigma_mps = 0.05;    ///< velocity uncertainty of the prediction over one frame interval
  /// Gate rejections lasting this long while vision is not healthy: the prediction is the likelier culprit.
  double dvl_relock_after_s = 2.0;
  double dvl_recent_s = 1.0;                ///< an accepted sample this young lets a vision restart keep its velocity
  double dvl_recovery_vel_sigma_mps = 0.05;
  double dvl_gap_s = 1.0;                   ///< spacing of accepted samples beyond this counts as a gap
  /// Vision counts as healthy (FrameStats::vision_healthy) when vision_used and at least this many
  /// existing landmarks passed the gate on the frame. Used only by the DVL re-lock rule and logs.
  int vision_healthy_min_factors = 10;
  // Smoother
  double lag_s = 3.0;
  double max_imu_gap_s = 0.05;       ///< larger gaps are bridged with a warning
};

/// IMU sample, body frame, measurement time in seconds.
struct ImuSample {
  double t;
  gtsam::Vector3 acc, gyr;
};

/// One tracked feature in a stereo frame, in undistorted normalised image coordinates.
struct FeatureObs {
  uint64_t track_id;
  gtsam::Point2 left;
  bool has_right = false;
  gtsam::Point2 right = gtsam::Point2::Zero();
  double sigma_px = 0.0;  ///< per-observation reprojection noise [px]; 0 = EstimatorConfig::pixel_sigma
};

struct FrameStats {
  int observations = 0, new_landmarks = 0, landmark_factors = 0, gated = 0;
  bool vision_used = false;
  int candidates = 0, stereo = 0;  ///< features matching a live landmark; features with a right observation
  /// vision_used and >= vision_healthy_min_factors landmark factors. Unlike vision_used it is false on
  /// frames with (almost) no features, where the gate has too few candidates to decide.
  bool vision_healthy = false;
  int dvl_factors = 0, dvl_gated = 0;  ///< DVL samples of this frame interval that became factors / were gated out
};

/// Relative pose between two earlier-seen frames from dense stereo registration, in left camera coordinates.
struct DenseConstraint {
  double t_ref;                  ///< time of the reference frame, exactly as passed to addFrame earlier
  gtsam::Pose3 T_Cref_Ccur;      ///< current left camera in the reference left camera frame
  gtsam::Matrix66 info_cam;      ///< registration information, tangent [omega; v] in the current left camera frame
};

class Estimator {
 public:
  explicit Estimator(const EstimatorConfig& config);

  /// IMU samples must arrive in increasing time; out-of-order or non-finite samples are dropped.
  void addImu(const ImuSample& s);
  /// Depth below the surface [m] at time t; used once the frame after t arrives.
  void addDepth(double t, double depth_m);
  /// DVL velocity of the DVL frame, expressed in it [m/s], stamped t [s] (shifted by dvl_time_offset_s).
  /// valid = bottom-lock flag. cov: 3x3 velocity covariance from the driver, used if its diagonal is
  /// positive. Invalid, non-finite, too fast or out-of-order samples are dropped and counted.
  /// No-op unless EstimatorConfig::dvl_enabled. Used once the frame after t arrives.
  void addDvl(double t, const gtsam::Vector3& v_D, bool valid, const std::optional<gtsam::Matrix3>& cov = std::nullopt);
  /// Adds a frame at time t. Returns false (and adds nothing) if it cannot be processed yet,
  /// e.g. no IMU coverage up to t.
  /// dense (optional): registration of this frame against the frame at dense->t_ref, accepted by its own
  /// Mahalanobis gate independently of the point-vision gate; dropped (counted) if that frame has left the window.
  bool addFrame(double t, const std::vector<FeatureObs>& features, FrameStats* stats = nullptr,
                const DenseConstraint* dense = nullptr);

  bool initialized() const { return initialized_; }
  gtsam::NavState latestState() const { return latest_; }
  gtsam::imuBias::ConstantBias latestBias() const { return latestBias_; }
  gtsam::Pose3 latestLeftCamera() const { return latest_.pose() * cfg_.T_BC0; }
  double latestTime() const { return tFrame_; }
  /// Left camera pose at time t (> latestTime()) propagated with the IMU from the latest state,
  /// without changing the estimator; nullopt before initialisation or without IMU coverage up to t.
  std::optional<gtsam::Pose3> predictLeftCamera(double t) const;
  /// Smoother estimate of the left camera at a frame time passed earlier to addFrame (exact match);
  /// nullopt if that frame is no longer in the window or was lost by a restart.
  std::optional<gtsam::Pose3> leftCameraAt(double t) const;
  /// World positions of the landmarks still in the smoother window, keyed by track id.
  std::vector<std::pair<uint64_t, gtsam::Point3>> liveLandmarks() const;

  struct Counters {
    uint64_t imu_dropped = 0, imu_gaps = 0, depth_dropped = 0, frames_no_vision = 0, solver_failures = 0,
             gate_recoveries = 0, gravity_factors = 0, degenerate_frames = 0, degenerate_dirs = 0,
             dense_factors = 0, dense_gated = 0, dense_stale = 0, dense_invalid = 0, dense_rank_sum = 0,
             dvl_received = 0, dvl_invalid = 0, dvl_gated = 0, dvl_factors = 0, dvl_relocks = 0, dvl_gaps = 0,
             dvl_recoveries_kept = 0;
    double dvl_longest_gap_s = 0.0;
  };
  const Counters& counters() const { return counters_; }
  int consecutiveGated() const { return consecutiveGated_; }

 private:
  bool initialize(double t);
  /// Restart the smoother from the last good state (latest_, latestBias_) at time t_last.
  void reset(double t_last, double vel_sigma_mps = 0.1, bool zero_velocity = false);
  bool preintegrate(double t0, double t1, gtsam::PreintegratedCombinedMeasurements* pim);
  /// preintegrate() without pruning imu_ or counting gaps; the largest bridged gap goes to max_gap_s.
  bool integrateImu(double t0, double t1, gtsam::PreintegratedCombinedMeasurements* pim, double* max_gap_s) const;
  struct DvlSample {
    double t;
    gtsam::Vector3 v;
    gtsam::Vector3 sigma;  ///< per-axis sigma [m/s]
  };
  /// Gyro (bias-corrected) interpolated from imu_ at t; must be called before preintegrate prunes imu_.
  gtsam::Vector3 gyroAt(double t) const;
  gtsam::Point3 triangulate(const FeatureObs& f, const gtsam::Pose3& T_WB, bool* ok) const;

  EstimatorConfig cfg_;
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> imuParams_;
  boost::shared_ptr<gtsam::Cal3_S2> K_;  // identity calibration: measurements are normalised
  gtsam::SharedNoiseModel pixelNoise_, depthNoise_;
  std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;

  std::deque<ImuSample> imu_;
  std::deque<std::pair<double, double>> depth_;  // (t, depth)
  bool initialized_ = false;
  uint64_t k_ = 0;  // index of the latest state
  double tFrame_ = 0.0;
  gtsam::NavState latest_;
  gtsam::imuBias::ConstantBias latestBias_;

  struct Landmark {
    gtsam::Key key;
    double last_seen;
  };
  std::unordered_map<uint64_t, Landmark> landmarks_;  // track id -> live landmark
  uint64_t nextLandmark_ = 0;
  bool visionStarted_ = false;
  int consecutiveGated_ = 0;
  gtsam::Vector3 lastGoodVelocity_ = gtsam::Vector3::Zero();
  std::map<double, uint64_t> stateIndex_;  ///< frame time -> state index k, pruned to the window, cleared on restarts
  std::deque<DvlSample> dvl_;
  std::optional<double> dvlLastAccepted_;   ///< stamp of the newest sample that became a factor (IMU time)
  std::optional<double> dvlRejectSince_;    ///< stamp of the first of the current run of gate rejections
  double dvlGateLimit_ = 0.0;               ///< chi-square(3) quantile of dvl_gate_quantile
  bool dvlSkipGate_ = true;                 ///< velocity unknown (start or restart): accept the next sample
  std::optional<double> depthSurfaceZ_;  ///< world z of the water surface, from the first depth reading
  Counters counters_;
};

}  // namespace uw_ct_vio
