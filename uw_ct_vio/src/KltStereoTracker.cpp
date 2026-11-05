#include "uw_ct_vio/KltStereoTracker.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace uw_ct_vio {

namespace {

constexpr double kEssentialConfidence = 0.999;
constexpr int kMinPointsForRansac = 8;
constexpr double kLcnNoiseFloor = 3.0;  // in units of the image noise sigma
constexpr double kLcnScale = 40.0;      // z-score -> 8-bit, centred on 128
constexpr double kMinPredictDepthM = 0.2;  // landmarks closer than this are not projected

/// Immerkaer (1996) noise estimate over the flatter half of the image [grey levels].
double noiseSigma(const cv::Mat& g8) {
  cv::Mat g, resp, gx, gy, mag;
  g8.convertTo(g, CV_32F);
  const cv::Mat k = (cv::Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
  cv::filter2D(g, resp, CV_32F, k);
  cv::Sobel(g, gx, CV_32F, 1, 0);
  cv::Sobel(g, gy, CV_32F, 0, 1);
  cv::magnitude(gx, gy, mag);
  cv::Mat m = mag.reshape(1, 1).clone();
  std::nth_element(m.begin<float>(), m.begin<float>() + m.cols / 2, m.end<float>());
  const float med = m.at<float>(m.cols / 2);
  double sum = 0;
  size_t n = 0;
  for (int y = 1; y < g.rows - 1; ++y)
    for (int x = 1; x < g.cols - 1; ++x)
      if (mag.at<float>(y, x) <= med) {
        sum += std::abs(resp.at<float>(y, x));
        ++n;
      }
  return n ? std::sqrt(M_PI / 2.0) / 6.0 * sum / static_cast<double>(n) : 0.0;
}

/// Zero-mean NCC of two equal-size float patches; -1 if either is flat.
double zncc(const cv::Mat& a, const cv::Mat& b) {
  cv::Scalar ma, sa, mb, sb;
  cv::meanStdDev(a, ma, sa);
  cv::meanStdDev(b, mb, sb);
  if (sa[0] < 1e-6 || sb[0] < 1e-6) return -1.0;
  return cv::mean((a - ma[0]).mul(b - mb[0]))[0] / (sa[0] * sb[0]);
}

}  // namespace

void CameraModel::validate(const std::string& name) const {
  if (!(fx > 0 && fy > 0 && width > 0 && height > 0) || coeffs.size() != 4)
    throw std::invalid_argument("camera '" + name + "': need fx, fy, width, height > 0 and 4 distortion coefficients");
}

std::vector<cv::Point2f> CameraModel::undistort(const std::vector<cv::Point2f>& px) const {
  std::vector<cv::Point2f> out;
  if (px.empty()) return out;
  const cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
  const cv::Vec4d D(coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
  if (distortion == Distortion::Equidistant) {
    cv::fisheye::undistortPoints(px, out, K, D);
  } else {
    cv::undistortPoints(px, out, K, D);
  }
  return out;
}

std::vector<cv::Point2f> CameraModel::project(const std::vector<cv::Point3f>& p) const {
  std::vector<cv::Point2f> out;
  if (p.empty()) return out;
  const cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
  const cv::Vec4d D(coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
  const cv::Vec3d zero(0, 0, 0);
  if (distortion == Distortion::Equidistant) {
    cv::fisheye::projectPoints(p, out, zero, zero, K, D);
  } else {
    cv::projectPoints(p, zero, zero, K, D, out);
  }
  return out;
}

KltStereoTracker::KltStereoTracker(const KltConfig& cfg, const CameraModel& left, const CameraModel& right,
                                   const gtsam::Pose3& T_C0C1)
    : cfg_(cfg), left_(left), right_(right) {
  left_.validate("left");
  right_.validate("right");
  // Points map cam0 -> cam1 with R = R_C0C1^T, t = -R_C0C1^T t_C0C1; E = [t]x R.
  const gtsam::Pose3 T_C1C0 = T_C0C1.inverse();
  const gtsam::Matrix3 E = gtsam::skewSymmetric(T_C1C0.translation()) * T_C1C0.rotation().matrix();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) E_(r, c) = E(r, c);
  clahe_ = cv::createCLAHE(cfg_.clahe_clip, cv::Size(cfg_.clahe_tiles, cfg_.clahe_tiles));
}

void KltStereoTracker::detect(const cv::Mat& img) {
  const int cells = cfg_.grid_cols * cfg_.grid_rows;
  const int perCell = std::max(1, cfg_.max_features / cells);
  const double cw = static_cast<double>(img.cols) / cfg_.grid_cols, ch = static_cast<double>(img.rows) / cfg_.grid_rows;
  std::vector<int> count(cells, 0);
  cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
  for (const auto& p : pts_) {
    const int cx = std::clamp(static_cast<int>(p.x / cw), 0, cfg_.grid_cols - 1);
    const int cy = std::clamp(static_cast<int>(p.y / ch), 0, cfg_.grid_rows - 1);
    ++count[cy * cfg_.grid_cols + cx];
    cv::circle(mask, p, cfg_.min_distance_px, cv::Scalar(0), -1);
  }
  for (int gy = 0; gy < cfg_.grid_rows; ++gy) {
    for (int gx = 0; gx < cfg_.grid_cols; ++gx) {
      const int need = perCell - count[gy * cfg_.grid_cols + gx];
      if (need <= 0) continue;
      const cv::Rect cell(static_cast<int>(gx * cw), static_cast<int>(gy * ch), static_cast<int>(cw), static_cast<int>(ch));
      std::vector<cv::Point2f> found;
      cv::goodFeaturesToTrack(img(cell), found, need, cfg_.quality_level, cfg_.min_distance_px, mask(cell));
      for (auto& f : found) {
        f += cv::Point2f(static_cast<float>(cell.x), static_cast<float>(cell.y));
        pts_.push_back(f);
        ids_.push_back(nextId_++);
        ++stats_.detected;
      }
    }
  }
}

cv::Mat KltStereoTracker::trackingImage(const cv::Mat& raw, double sigmaNoise) const {
  if (!lcnActive_) {
    cv::Mat out;
    clahe_->apply(raw, out);
    return out;
  }
  cv::Mat f, mu, mu2, var, z, out;
  raw.convertTo(f, CV_32F);
  const cv::Size k(cfg_.lcn_window_px, cfg_.lcn_window_px);
  cv::boxFilter(f, mu, CV_32F, k);
  cv::boxFilter(f.mul(f), mu2, CV_32F, k);
  const double floor2 = std::pow(std::max(kLcnNoiseFloor * std::max(sigmaNoise, 0.5), cfg_.lcn_min_floor), 2);
  var = mu2 - mu.mul(mu) + floor2;
  cv::sqrt(cv::max(var, floor2), var);
  cv::divide(f - mu, var, z);
  z.convertTo(out, CV_8U, kLcnScale, 128.0);  // saturates beyond about +-3 sigma
  return out;
}

std::vector<cv::Point2f> KltStereoTracker::predictTracks(const MapPrediction& pred,
                                                         const std::unordered_map<uint64_t, gtsam::Point3>& lms) const {
  std::vector<cv::Point2f> guess = pts_;
  std::vector<cv::Point3f> rays;
  std::vector<size_t> idx;
  std::vector<cv::Point2f> prevNorm;
  gtsam::Matrix3 R_cur_prev = gtsam::Matrix3::Identity();
  if (pred.T_prev_cur) {
    prevNorm = left_.undistort(pts_);
    R_cur_prev = pred.T_prev_cur->rotation().matrix().transpose();
  }
  for (size_t i = 0; i < pts_.size(); ++i) {
    const auto it = lms.find(ids_[i]);
    if (it != lms.end() && it->second.z() > kMinPredictDepthM) {
      rays.emplace_back(it->second.x(), it->second.y(), it->second.z());
      idx.push_back(i);
    } else if (pred.T_prev_cur) {
      // Unknown depth: rotate the ray only (exact for distant points, the common case underwater).
      const gtsam::Vector3 v = R_cur_prev * gtsam::Vector3(prevNorm[i].x, prevNorm[i].y, 1.0);
      if (v.z() > 1e-3) {
        rays.emplace_back(v.x(), v.y(), v.z());
        idx.push_back(i);
      }
    }
  }
  const auto px = left_.project(rays);
  for (size_t j = 0; j < idx.size(); ++j)
    if (px[j].x >= 0 && px[j].y >= 0 && px[j].x < left_.width && px[j].y < left_.height) guess[idx[j]] = px[j];
  return guess;
}

void KltStereoTracker::reacquire(const cv::Mat& left, const cv::Mat& leftRaw,
                                 const std::unordered_map<uint64_t, gtsam::Point3>& lms, std::vector<bool>* reacquired) {
  const int64_t cur = frameIdx_;
  for (auto it = lost_.begin(); it != lost_.end();)
    it = (hist_.empty() || it->second.idx < hist_.front().idx || cur - it->second.idx > cfg_.reacq_history)
             ? lost_.erase(it)
             : std::next(it);
  // Candidates: lost tracks with a live landmark in front of the camera, grouped by source frame.
  std::vector<uint64_t> cid;
  std::vector<cv::Point3f> rays;
  for (const auto& [id, l] : lost_) {
    const auto lm = lms.find(id);
    if (lm == lms.end() || lm->second.z() <= kMinPredictDepthM) continue;
    cid.push_back(id);
    rays.emplace_back(lm->second.x(), lm->second.y(), lm->second.z());
  }
  const auto proj = left_.project(rays);
  const float margin = 0.5f * static_cast<float>(cfg_.lk_window_px);
  const double minD2 = static_cast<double>(cfg_.min_distance_px) * cfg_.min_distance_px;
  std::map<int64_t, std::vector<size_t>> bySource;
  for (size_t j = 0; j < cid.size(); ++j) {
    const cv::Point2f& p = proj[j];
    if (!(p.x >= margin && p.y >= margin && p.x < left.cols - margin && p.y < left.rows - margin)) continue;
    bool crowded = false;
    for (const auto& q : pts_) {
      const cv::Point2f d = q - p;
      if (d.dot(d) < minD2) {
        crowded = true;
        break;
      }
    }
    if (!crowded) bySource[lost_.at(cid[j]).idx].push_back(j);
  }
  const cv::Size win(cfg_.lk_window_px, cfg_.lk_window_px);
  const cv::Size ps(cfg_.ncc_patch_px, cfg_.ncc_patch_px);
  const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  for (const auto& [srcIdx, js] : bySource) {
    const auto h = std::find_if(hist_.begin(), hist_.end(), [&](const HistFrame& f) { return f.idx == srcIdx; });
    if (h == hist_.end()) continue;
    std::vector<cv::Point2f> src, fwd, back;
    for (size_t j : js) {
      src.push_back(lost_.at(cid[j]).px);
      fwd.push_back(proj[j]);
    }
    std::vector<uchar> s1, s2;
    std::vector<float> err;
    stats_.reacq_tried += static_cast<int>(js.size());
    cv::calcOpticalFlowPyrLK(h->img, left, src, fwd, s1, err, win, cfg_.lk_levels, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    back = src;
    cv::calcOpticalFlowPyrLK(left, h->img, fwd, back, s2, err, win, cfg_.lk_levels, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    for (size_t k = 0; k < js.size(); ++k) {
      const size_t j = js[k];
      const bool inside = fwd[k].x >= 0 && fwd[k].y >= 0 && fwd[k].x < left.cols && fwd[k].y < left.rows;
      if (!s1[k] || !s2[k] || !inside || cv::norm(back[k] - src[k]) > cfg_.fb_max_px ||
          cv::norm(fwd[k] - proj[j]) > cfg_.reacq_gate_px)
        continue;
      cv::Mat a, b;
      cv::getRectSubPix(h->raw, ps, src[k], a, CV_32F);
      cv::getRectSubPix(leftRaw, ps, fwd[k], b, CV_32F);
      if (zncc(a, b) < cfg_.reacq_ncc_min) continue;
      pts_.push_back(fwd[k]);
      ids_.push_back(cid[j]);
      reacquired->push_back(true);
    }
  }
}

std::vector<FeatureObs> KltStereoTracker::track(const cv::Mat& leftIn, const cv::Mat& rightIn, const MapPrediction* pred) {
  if (leftIn.type() != CV_8UC1 || rightIn.type() != CV_8UC1 || leftIn.size() != rightIn.size())
    throw std::invalid_argument("KltStereoTracker: expects two 8-bit grey images of equal size");
  stats_ = Stats();
  if (!lcnDecided_) {
    if (!cfg_.lcn_auto) {
      lcnActive_ = cfg_.lcn;
      lcnDecided_ = true;
    } else {
      // Dynamic range of this frame from its grey-level histogram.
      int hist[256] = {0};
      for (int y = 0; y < leftIn.rows; ++y)
        for (int x = 0; x < leftIn.cols; ++x) ++hist[leftIn.at<uchar>(y, x)];
      const double total = static_cast<double>(leftIn.total());
      double acc = 0;
      int p1 = 0, p99 = 255;
      for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc < 0.01 * total) p1 = v + 1;
        if (acc >= 0.99 * total) { p99 = v; break; }
      }
      autoRanges_.push_back(p99 - p1);
      if (static_cast<int>(autoRanges_.size()) >= cfg_.lcn_auto_frames) {
        std::vector<double> r = autoRanges_;
        std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
        autoRange_ = r[r.size() / 2];
        lcnActive_ = autoRange_ < cfg_.lcn_auto_max_range;
        lcnDecided_ = true;
        // Re-express the previous frame in the new representation so tracks continue cleanly.
        if (lcnActive_ && !prevRaw_.empty()) prev_ = trackingImage(prevRaw_, noiseSigma(prevRaw_));
        if (lcnActive_) {  // stored images are in the old representation
          hist_.clear();
          lost_.clear();
        }
      }
    }
  }
  stats_.lcn_active = lcnActive_;
  stats_.auto_range = autoRange_;
  const double sigmaN = (lcnActive_ || cfg_.snr_sigma || cfg_.ncc_noise_aware) ? noiseSigma(leftIn) : 0.0;
  stats_.noise_sigma = sigmaN;
  const cv::Mat left = trackingImage(leftIn, sigmaN), right = trackingImage(rightIn, sigmaN);
  const cv::Size win(cfg_.lk_window_px, cfg_.lk_window_px);
  const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  std::unordered_map<uint64_t, gtsam::Point3> lms;
  if (pred && (cfg_.map_predict || cfg_.reacquire))
    for (const auto& [id, P] : pred->landmarks_cur) lms.emplace(id, P);
  const bool reacq = cfg_.reacquire && pred;
  auto markLost = [&](uint64_t id, const cv::Point2f& px) {
    if (cfg_.reacquire) lost_[id] = Lost{frameIdx_ - 1, px};
  };

  // Temporal tracking with forward-backward check and essential-matrix RANSAC.
  if (!prev_.empty() && !pts_.empty()) {
    std::vector<cv::Point2f> fwd, back;
    std::vector<uchar> s1, s2;
    std::vector<float> err;
    if (pred && cfg_.map_predict) {
      const std::vector<cv::Point2f> guess = predictTracks(*pred, lms);
      fwd = guess;
      // Backward pass starts from the old position shifted by the same prediction error, so the
      // forward-backward check does not start at its own answer.
      cv::calcOpticalFlowPyrLK(prev_, left, pts_, fwd, s1, err, win, cfg_.lk_levels, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
      back.resize(pts_.size());
      for (size_t i = 0; i < pts_.size(); ++i) back[i] = pts_[i] + (fwd[i] - guess[i]);
      cv::calcOpticalFlowPyrLK(left, prev_, fwd, back, s2, err, win, cfg_.lk_levels, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
      // A prediction that loses the track (e.g. a poor landmark estimate) is retried from the old position.
      std::vector<size_t> retry;
      for (size_t i = 0; i < pts_.size(); ++i) {
        const bool moved = guess[i] != pts_[i];
        stats_.predicted += moved;
        const bool inside = fwd[i].x >= 0 && fwd[i].y >= 0 && fwd[i].x < left.cols && fwd[i].y < left.rows;
        if (moved && (!s1[i] || !s2[i] || !inside || cv::norm(back[i] - pts_[i]) > cfg_.fb_max_px)) retry.push_back(i);
      }
      if (!retry.empty()) {
        std::vector<cv::Point2f> rp, rf, rb;
        std::vector<uchar> r1, r2;
        for (size_t i : retry) rp.push_back(pts_[i]);
        cv::calcOpticalFlowPyrLK(prev_, left, rp, rf, r1, err, win, cfg_.lk_levels, crit);
        cv::calcOpticalFlowPyrLK(left, prev_, rf, rb, r2, err, win, cfg_.lk_levels, crit);
        for (size_t k = 0; k < retry.size(); ++k) {
          fwd[retry[k]] = rf[k];
          back[retry[k]] = rb[k];
          s1[retry[k]] = r1[k];
          s2[retry[k]] = r2[k];
        }
        stats_.predict_retried = static_cast<int>(retry.size());
      }
    } else {
      cv::calcOpticalFlowPyrLK(prev_, left, pts_, fwd, s1, err, win, cfg_.lk_levels, crit);
      cv::calcOpticalFlowPyrLK(left, prev_, fwd, back, s2, err, win, cfg_.lk_levels, crit);
    }
    std::vector<cv::Point2f> keptPrev, keptCur;
    std::vector<uint64_t> keptIds;
    for (size_t i = 0; i < pts_.size(); ++i) {
      const bool inside = fwd[i].x >= 0 && fwd[i].y >= 0 && fwd[i].x < left.cols && fwd[i].y < left.rows;
      if (!s1[i] || !s2[i] || !inside || cv::norm(back[i] - pts_[i]) > cfg_.fb_max_px) {
        ++stats_.fb_rejected;
        markLost(ids_[i], pts_[i]);
        continue;
      }
      if (cfg_.ncc_min > 0.0) {
        // cuVSLAM-style check on the raw images: the tracked patch must still look like the old one.
        const cv::Size ps(cfg_.ncc_patch_px, cfg_.ncc_patch_px);
        cv::Mat a, b;
        cv::getRectSubPix(prevRaw_, ps, pts_[i], a, CV_32F);
        cv::getRectSubPix(leftIn, ps, fwd[i], b, CV_32F);
        double thresh = cfg_.ncc_min;
        if (cfg_.ncc_noise_aware) {
          cv::Scalar m, sd;
          cv::meanStdDev(a, m, sd);
          const double varS = std::max(sd[0] * sd[0] - sigmaN * sigmaN, 1e-6), varN = sigmaN * sigmaN;
          thresh *= varS / (varS + varN);
        }
        if (zncc(a, b) < thresh) {
          ++stats_.ncc_rejected;
          markLost(ids_[i], pts_[i]);
          continue;
        }
      }
      keptPrev.push_back(pts_[i]);
      keptCur.push_back(fwd[i]);
      keptIds.push_back(ids_[i]);
    }
    pts_.clear();
    ids_.clear();
    if (static_cast<int>(keptCur.size()) >= kMinPointsForRansac) {
      const auto a = left_.undistort(keptPrev), b = left_.undistort(keptCur);
      cv::Mat inl;
      cv::findEssentialMat(a, b, cv::Mat::eye(3, 3, CV_64F), cv::RANSAC, kEssentialConfidence,
                           cfg_.ransac_px / left_.fx, inl);
      for (size_t i = 0; i < keptCur.size(); ++i) {
        if (!inl.empty() && !inl.at<uchar>(static_cast<int>(i))) {
          ++stats_.ransac_rejected;
          markLost(keptIds[i], keptPrev[i]);
          continue;
        }
        pts_.push_back(keptCur[i]);
        ids_.push_back(keptIds[i]);
      }
    } else {
      pts_ = keptCur;
      ids_ = keptIds;
    }
    stats_.tracked = static_cast<int>(pts_.size());
  }
  std::vector<bool> reacquired(pts_.size(), false);
  if (reacq) reacquire(left, leftIn, lms, &reacquired);
  if (static_cast<int>(pts_.size()) < cfg_.max_features) detect(left);
  reacquired.resize(pts_.size(), false);
  prev_ = left;
  prevRaw_ = leftIn.clone();
  if (cfg_.reacquire) {
    hist_.push_back({frameIdx_, left, prevRaw_});
    while (static_cast<int>(hist_.size()) > cfg_.reacq_history) hist_.pop_front();
  }
  ++frameIdx_;

  // Stereo: LK left -> right, forward-backward and epipolar checks.
  std::vector<FeatureObs> out(pts_.size());
  const auto nl = left_.undistort(pts_);
  double sigmaSum = 0;
  for (size_t i = 0; i < pts_.size(); ++i) {
    out[i].track_id = ids_[i];
    out[i].left = gtsam::Point2(nl[i].x, nl[i].y);
    if (cfg_.snr_sigma) {
      // Contrast-to-noise of the raw patch around the feature: weak patches count less.
      cv::Mat patch;
      cv::getRectSubPix(leftIn, cv::Size(cfg_.lk_window_px, cfg_.lk_window_px), pts_[i], patch, CV_32F);
      cv::Scalar m, sd;
      cv::meanStdDev(patch, m, sd);
      const double cnr = sd[0] / std::max(sigmaN, 0.5);
      const double scale = std::clamp(cfg_.snr_cnr_ref / std::max(cnr, 1e-3), 1.0, cfg_.snr_max_scale);
      out[i].sigma_px = cfg_.base_sigma_px * scale;
      sigmaSum += out[i].sigma_px;
    }
  }
  if (cfg_.snr_sigma && !pts_.empty()) stats_.mean_sigma_px = sigmaSum / static_cast<double>(pts_.size());
  if (!pts_.empty()) {
    std::vector<cv::Point2f> r, back;
    std::vector<uchar> s1, s2;
    std::vector<float> err;
    // Fresh buffers: cv::Mat assignment shares pixels, and prev_ aliases `left`.
    cv::Mat sl, sr;
    if (lcnActive_ && !cfg_.lcn_stereo) {
      clahe_->apply(leftIn, sl);
      clahe_->apply(rightIn, sr);
    } else {
      sl = left;
      sr = right;
    }
    cv::calcOpticalFlowPyrLK(sl, sr, pts_, r, s1, err, win, cfg_.stereo_lk_levels, crit);
    cv::calcOpticalFlowPyrLK(sr, sl, r, back, s2, err, win, cfg_.stereo_lk_levels, crit);
    const auto nr = right_.undistort(r);
    const double epiMax = cfg_.epipolar_px / right_.fx;
    for (size_t i = 0; i < pts_.size(); ++i) {
      if (!s1[i] || !s2[i] || cv::norm(back[i] - pts_[i]) > cfg_.fb_max_px) continue;
      const cv::Vec3d x0(nl[i].x, nl[i].y, 1.0), x1(nr[i].x, nr[i].y, 1.0);
      const cv::Vec3d line = E_ * x0;  // epipolar line in the right image (normalised)
      const double d = std::abs(x1.dot(line)) / std::hypot(line[0], line[1]);
      if (!(d < epiMax)) {
        ++stats_.stereo_rejected;
        continue;
      }
      out[i].has_right = true;
      out[i].right = gtsam::Point2(nr[i].x, nr[i].y);
      ++stats_.stereo;
    }
  }
  if (reacq) {
    // A re-acquired track must also match in the right image; otherwise it stays lost (keeping its
    // original reference) and the point is dropped.
    size_t w = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      if (reacquired[i]) {
        if (!out[i].has_right) {
          ++stats_.reacq_no_stereo;
          continue;
        }
        lost_.erase(ids_[i]);
        ++stats_.reacquired;
      }
      out[w] = out[i];
      pts_[w] = pts_[i];
      ids_[w] = ids_[i];
      ++w;
    }
    out.resize(w);
    pts_.resize(w);
    ids_.resize(w);
  }
  return out;
}

}  // namespace uw_ct_vio
