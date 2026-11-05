#include "uw_ct_vio/DenseRegistration.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace uw_ct_vio {

namespace {

constexpr double kCorrShrink = 0.7;     // per-iteration factor of the correspondence gate
constexpr double kCauchyC = 2.3849;     // 95 % efficiency of the Cauchy kernel, on the normalised residual
constexpr double kMinDepthM = 1e-3;     // guards the depth division in the noise model

struct Linearisation {
  gtsam::Matrix66 H = gtsam::Matrix66::Zero();
  gtsam::Vector6 g = gtsam::Vector6::Zero();
  int inliers = 0;
  double sse = 0.0;
};

/// Standard deviation of a point's error along its normal: the depth error moves the point along its
/// viewing ray, whose component along n is |n . p| / z (z in the rectified frame).
double sigmaAlongNormal(const OrganisedCloud& c, size_t i) {
  const double z = std::max((c.R_rect_cam * c.p[i]).z(), kMinDepthM);
  return c.sigma_z[i] * std::abs(c.n[i].dot(c.p[i])) / z;
}

Linearisation linearise(const OrganisedCloud& ref, const OrganisedCloud& cur, const gtsam::Pose3& T, double max_corr_m,
                        const IcpConfig& cfg) {
  Linearisation L;
  const double cosMin = std::cos(cfg.max_normal_angle_rad);
  const Eigen::Matrix3d R = T.rotation().matrix();
  const Eigen::Vector3d t = T.translation();
  for (int r = 0; r < cur.rows; ++r) {
    for (int c = 0; c < cur.cols; ++c) {
      const size_t i = cur.index(r, c);
      if (!(cur.flags[i] & OrganisedCloud::kNormal)) continue;
      const Eigen::Vector3d p_ref = R * cur.p[i] + t;
      const Eigen::Vector3d q_rect = ref.R_rect_cam * p_ref;
      if (!(q_rect.z() > kMinDepthM)) continue;
      const double u = ref.fx * q_rect.x() / q_rect.z() + ref.cx, v = ref.fy * q_rect.y() / q_rect.z() + ref.cy;
      const int tc = static_cast<int>(std::lround(u / ref.stride)), tr = static_cast<int>(std::lround(v / ref.stride));
      if (tr < 0 || tr >= ref.rows || tc < 0 || tc >= ref.cols) continue;
      const size_t j = ref.index(tr, tc);
      if (!(ref.flags[j] & OrganisedCloud::kNormal)) continue;
      const Eigen::Vector3d m = R.transpose() * ref.n[j];  // target normal in the current frame
      if (m.dot(cur.n[i]) < cosMin) continue;
      const double res = ref.n[j].dot(p_ref - ref.p[j]);
      const double sigma2 = std::pow(sigmaAlongNormal(cur, i), 2) + std::pow(sigmaAlongNormal(ref, j), 2) +
                            cfg.min_sigma_m * cfg.min_sigma_m;
      // Far points are noisier than the metric gate: let them in up to max_corr_sigmas of their own noise.
      if (std::abs(res) > std::max(max_corr_m, cfg.max_corr_sigmas * std::sqrt(sigma2))) continue;
      const double rho = res / (kCauchyC * std::sqrt(sigma2));
      const double w = 1.0 / (sigma2 * (1.0 + rho * rho));
      // r(T Exp(d)) = r + J d with d = [omega; v]: dr = n^T R (omega x p + v).
      gtsam::Vector6 J;
      J << cur.p[i].cross(m), m;
      L.H += w * J * J.transpose();
      L.g += w * res * J;
      L.sse += res * res;
      ++L.inliers;
    }
  }
  return L;
}

}  // namespace

const char* toString(IcpFailure f) {
  switch (f) {
    case IcpFailure::None: return "none";
    case IcpFailure::NonFinite: return "non-finite";
    case IcpFailure::TooFewInliers: return "too-few-inliers";
    case IcpFailure::LowInlierFraction: return "low-inlier-fraction";
    case IcpFailure::Diverged: return "diverged";
  }
  return "unknown";
}

IcpResult registerClouds(const OrganisedCloud& ref, const OrganisedCloud& cur, const gtsam::Pose3& T_ref_cur_init,
                         const IcpConfig& cfg) {
  IcpResult res;
  res.T_ref_cur = T_ref_cur_init;
  if (!T_ref_cur_init.matrix().allFinite()) {
    res.reason = IcpFailure::NonFinite;
    return res;
  }
  const int sources = cur.validNormals();
  gtsam::Pose3 T = T_ref_cur_init;
  double corr = cfg.max_corr_start_m;
  for (int it = 0; it < cfg.max_iterations; ++it) {
    const Linearisation L = linearise(ref, cur, T, corr, cfg);
    res.iterations = it + 1;
    if (L.inliers < 6) break;  // nothing to step on; the acceptance checks below report it
    // Pseudo-inverse over the well-observed eigen-directions: a degenerate scene (one plane) must
    // leave the unobserved directions at the initial guess instead of sliding along them.
    Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> es(L.H);
    const double lamMax = es.eigenvalues()(5), lamCut = std::min(cfg.solve_rel_eig * lamMax, cfg.solve_abs_eig);
    gtsam::Vector6 step = gtsam::Vector6::Zero();
    for (int k = 0; k < 6; ++k) {
      if (es.eigenvalues()(k) > lamCut && es.eigenvalues()(k) > 0.0) {
        const gtsam::Vector6 e = es.eigenvectors().col(k);
        step -= e * (e.dot(L.g) / es.eigenvalues()(k));
      }
    }
    if (!step.allFinite()) {
      res.reason = IcpFailure::NonFinite;
      return res;
    }
    T = T.compose(gtsam::Pose3::Expmap(step));
    corr = std::max(cfg.max_corr_end_m, corr * kCorrShrink);
    if (step.norm() < cfg.stop_step && corr <= cfg.max_corr_end_m) {
      res.converged = true;
      break;
    }
  }
  // Statistics and information at the final pose, with the final gate.
  const Linearisation F = linearise(ref, cur, T, cfg.max_corr_end_m, cfg);
  res.T_ref_cur = T;
  res.H = F.H;
  res.inliers = F.inliers;
  res.inlier_fraction = sources > 0 ? static_cast<double>(F.inliers) / sources : 0.0;
  res.rmse_m = F.inliers > 0 ? std::sqrt(F.sse / F.inliers) : 0.0;
  if (!T.matrix().allFinite() || !F.H.allFinite() || !std::isfinite(res.rmse_m)) {
    res.reason = IcpFailure::NonFinite;
    return res;
  }
  if (F.inliers < cfg.min_inliers) {
    res.reason = IcpFailure::TooFewInliers;
    return res;
  }
  if (res.inlier_fraction < cfg.min_inlier_fraction) {
    res.reason = IcpFailure::LowInlierFraction;
    return res;
  }
  const gtsam::Vector6 dev = gtsam::Pose3::Logmap(T_ref_cur_init.between(T));
  if (dev.head<3>().norm() > cfg.max_rot_dev_rad || dev.tail<3>().norm() > cfg.max_trans_dev_m) {
    res.reason = IcpFailure::Diverged;
    return res;
  }
  res.ok = true;
  res.reason = IcpFailure::None;
  return res;
}

}  // namespace uw_ct_vio
