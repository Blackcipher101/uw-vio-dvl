#include "uw_ct_vio/DepthGpFactor.hpp"

#include <gtsam/base/Matrix.h>

#include <cmath>
#include <stdexcept>

namespace uw_ct_vio {

DepthGpFactor::DepthGpFactor(gtsam::Key xi, gtsam::Key vi, gtsam::Key xj, gtsam::Key vj, double tau, double dt_s,
                             double depth_m, double z_surface_m, const gtsam::Vector3& l_body,
                             const gtsam::SharedNoiseModel& noise)
    : Base(noise, xi, vi, xj, vj), tau_(tau), dt_(dt_s), depth_(depth_m), zSurface_(z_surface_m), lBody_(l_body) {
  if (!(tau >= 0.0 && tau <= 1.0)) throw std::invalid_argument("DepthGpFactor: tau must be in [0, 1]");
  if (!(dt_s > 0.0)) throw std::invalid_argument("DepthGpFactor: dt_s must be > 0");
  if (!std::isfinite(depth_m) || !std::isfinite(z_surface_m) || !l_body.allFinite())
    throw std::invalid_argument("DepthGpFactor: non-finite depth, surface or lever arm");
  if (!noise || noise->dim() != 1) throw std::invalid_argument("DepthGpFactor: noise model must be 1-D");
}

gtsam::Vector4 DepthGpFactor::hermite(double tau) {
  const double t2 = tau * tau, t3 = t2 * tau;
  return gtsam::Vector4(2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + tau, -2 * t3 + 3 * t2, t3 - t2);
}

gtsam::Vector DepthGpFactor::evaluateError(const gtsam::Pose3& Xi, const gtsam::Vector3& Vi, const gtsam::Pose3& Xj,
                                           const gtsam::Vector3& Vj, boost::optional<gtsam::Matrix&> H1,
                                           boost::optional<gtsam::Matrix&> H2, boost::optional<gtsam::Matrix&> H3,
                                           boost::optional<gtsam::Matrix&> H4) const {
  const gtsam::Vector4 h = hermite(tau_);
  const bool nearI = tau_ <= 0.5;
  const gtsam::Rot3& Rn = nearI ? Xi.rotation() : Xj.rotation();
  const gtsam::Vector3 p = h(0) * Xi.translation() + h(1) * dt_ * Vi + h(2) * Xj.translation() + h(3) * dt_ * Vj +
                           Rn.matrix() * lBody_;
  const gtsam::Vector3 e3(0, 0, 1);

  // Pose3 tangent is [omega; v] with right perturbation: d(t)/d(v) = R, d(R l)/d(omega) = -R [l]x.
  const gtsam::Matrix13 dLever = -e3.transpose() * Rn.matrix() * gtsam::skewSymmetric(lBody_);
  if (H1) {
    *H1 = gtsam::Matrix::Zero(1, 6);
    if (nearI) H1->block<1, 3>(0, 0) = dLever;
    H1->block<1, 3>(0, 3) = h(0) * e3.transpose() * Xi.rotation().matrix();
  }
  if (H2) *H2 = h(1) * dt_ * e3.transpose();
  if (H3) {
    *H3 = gtsam::Matrix::Zero(1, 6);
    if (!nearI) H3->block<1, 3>(0, 0) = dLever;
    H3->block<1, 3>(0, 3) = h(2) * e3.transpose() * Xj.rotation().matrix();
  }
  if (H4) *H4 = h(3) * dt_ * e3.transpose();
  return gtsam::Vector1(p.z() - (zSurface_ - depth_));
}

}  // namespace uw_ct_vio
