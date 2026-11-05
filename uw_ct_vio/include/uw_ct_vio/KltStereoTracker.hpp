// Patch (KLT) stereo feature tracker: survives low-texture scenes where corner matching fails.
#pragma once

#include <gtsam/geometry/Pose3.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "uw_ct_vio/Estimator.hpp"

namespace uw_ct_vio {

/// Pinhole camera with radial-tangential (k1 k2 p1 p2) or equidistant (k1..k4) distortion.
struct CameraModel {
  enum class Distortion { RadTan, Equidistant };
  Distortion distortion = Distortion::RadTan;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::vector<double> coeffs;  ///< 4 values
  int width = 0, height = 0;

  /// Pixels (distorted) -> undistorted normalised coordinates (x/z, y/z).
  std::vector<cv::Point2f> undistort(const std::vector<cv::Point2f>& px) const;
  /// Points in the camera frame (z > 0) -> distorted pixels.
  std::vector<cv::Point2f> project(const std::vector<cv::Point3f>& p) const;
  void validate(const std::string& name) const;
};

struct KltConfig {
  int max_features = 150;
  int grid_cols = 8, grid_rows = 6;      ///< new features only in cells below their share
  int min_distance_px = 15;
  double quality_level = 0.001;          ///< low: dark, low-contrast underwater images
  int lk_window_px = 21, lk_levels = 3;
  int stereo_lk_levels = 4;              ///< left->right disparities are larger than frame-to-frame motion
  double fb_max_px = 1.0;                ///< forward-backward consistency
  double ransac_px = 1.0;                ///< temporal essential-matrix RANSAC threshold
  double epipolar_px = 2.0;              ///< stereo epipolar distance threshold
  double clahe_clip = 1.0;
  int clahe_tiles = 2;
  // Step 2 switches (raw images in; these change how patches are compared and weighted only).
  /// Track on locally normalised intensities (z = (I - mean) / sqrt(var + (3 sigma_n)^2)) instead of
  /// CLAHE: invariant to slowly varying gain and offset (lamp falloff, exposure), with a noise floor
  /// so flat dark regions are not amplified.
  bool lcn = false;
  /// Decide LCN from the scene instead: on when the median dynamic range (p99 - p1 grey levels) of
  /// the first lcn_auto_frames frames is below lcn_auto_max_range. LCN helps in low-contrast scenes
  /// (NTNU pool: range 67-86) and hurts bright hazy ones (AFRL lake: 138-206), where it turns glare
  /// ramps and particles into texture. A single first frame is unreliable (exposure settling).
  bool lcn_auto = false;
  /// Set between the two datasets seen so far (NTNU pool 79-117, AFRL lake 194 over the first 30
  /// frames); new sites need checking.
  double lcn_auto_max_range = 150.0;
  int lcn_auto_frames = 30;
  int lcn_window_px = 15;
  /// Lower bound of the LCN noise floor [grey levels]. JPEG-compressed images hide sensor noise
  /// (estimated sigma ~0.4), so 3 sigma alone lets normalisation amplify blocking and glare ramps.
  double lcn_min_floor = 0.0;
  /// Use LCN for left-right matching too. Off: stereo uses CLAHE images, because LCN removes the
  /// low-frequency structure coarse pyramid levels need to find large disparities (wide baselines).
  bool lcn_stereo = true;
  /// Zero-mean NCC between the raw patch before and after tracking; tracks below are dropped (0 = off).
  double ncc_min = 0.0;
  int ncc_patch_px = 11;
  /// Scale the NCC threshold by the NCC two noisy copies of the same patch would reach,
  /// var_s / (var_s + sigma_n^2): dark, low-contrast patches cannot score high even when tracked well.
  bool ncc_noise_aware = false;
  /// Per-observation sigma from the patch contrast-to-noise ratio: sigma = base * clamp(cnr_ref / CNR, 1, 5).
  bool snr_sigma = false;
  double snr_cnr_ref = 8.0;
  double snr_max_scale = 5.0;
  double base_sigma_px = 1.0;
  // Step 5 switches (frame-to-map, need a MapPrediction from the estimator).
  /// Start temporal LK at the predicted position: the landmark's projection if the track has one,
  /// else the previous ray rotated by the IMU-predicted camera rotation. A track the prediction
  /// loses is retried from its old position, so a poor landmark estimate costs time, not the track.
  bool map_predict = false;
  /// Re-find recently lost tracks whose landmark is still in the estimator window: LK from the
  /// stored image where the track was last seen, started at the landmark's projection. Accepted
  /// only within reacq_gate_px of the projection, with raw-patch NCC >= reacq_ncc_min and a stereo
  /// match, because the estimator's gate uses the same prediction and cannot confirm identity.
  bool reacquire = false;
  int reacq_history = 30;        ///< frames kept for re-acquisition (~ the estimator lag)
  double reacq_gate_px = 8.0;
  double reacq_ncc_min = 0.7;
};

/// The estimator's expectation for the frame about to be tracked (all in the current left camera).
struct MapPrediction {
  /// Current left camera pose in the left camera frame of the previous track() call; empty if the
  /// estimator has no state for that frame.
  std::optional<gtsam::Pose3> T_prev_cur;
  /// Live landmarks by track id, in the IMU-predicted current left camera frame.
  std::vector<std::pair<uint64_t, gtsam::Point3>> landmarks_cur;
};

class KltStereoTracker {
 public:
  /// T_C0C1: right camera pose in the left camera frame.
  KltStereoTracker(const KltConfig& cfg, const CameraModel& left, const CameraModel& right, const gtsam::Pose3& T_C0C1);

  /// Grey 8-bit images of the same stereo frame. Returns features in normalised coordinates.
  /// pred (optional) is used only by the step-5 switches.
  std::vector<FeatureObs> track(const cv::Mat& left, const cv::Mat& right, const MapPrediction* pred = nullptr);

  struct Stats {
    int tracked = 0, fb_rejected = 0, ransac_rejected = 0, detected = 0, stereo = 0, stereo_rejected = 0,
        ncc_rejected = 0, predicted = 0, predict_retried = 0, reacq_tried = 0, reacquired = 0, reacq_no_stereo = 0;
    double noise_sigma = 0.0, mean_sigma_px = 0.0;
    bool lcn_active = false;
    double auto_range = -1.0;
  };
  const Stats& lastStats() const { return stats_; }

 private:
  void detect(const cv::Mat& img);
  cv::Mat trackingImage(const cv::Mat& raw, double sigmaNoise) const;
  /// Predicted pixel of each current track (pts_ where there is no prediction).
  std::vector<cv::Point2f> predictTracks(const MapPrediction& pred,
                                         const std::unordered_map<uint64_t, gtsam::Point3>& lms) const;
  void reacquire(const cv::Mat& left, const cv::Mat& leftRaw, const std::unordered_map<uint64_t, gtsam::Point3>& lms,
                 std::vector<bool>* reacquired);

  KltConfig cfg_;
  CameraModel left_, right_;
  cv::Matx33d E_;  ///< essential matrix of the stereo pair: x1^T E x0 = 0 (normalised coordinates)
  cv::Ptr<cv::CLAHE> clahe_;
  cv::Mat prev_, prevRaw_;
  bool lcnActive_ = false, lcnDecided_ = false;
  std::vector<double> autoRanges_;
  double autoRange_ = -1.0;
  std::vector<cv::Point2f> pts_;
  std::vector<uint64_t> ids_;
  uint64_t nextId_ = 1;
  Stats stats_;
  // Re-acquisition state: recent tracking/raw images by frame index, and where lost tracks were last seen.
  struct HistFrame {
    int64_t idx;
    cv::Mat img, raw;
  };
  struct Lost {
    int64_t idx;
    cv::Point2f px;
  };
  int64_t frameIdx_ = 0;
  std::deque<HistFrame> hist_;
  std::unordered_map<uint64_t, Lost> lost_;
};

}  // namespace uw_ct_vio
