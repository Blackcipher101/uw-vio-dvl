#include "uw_ct_vio/DvlFactor.hpp"

#include <gtsam/base/Matrix.h>

#include <cmath>
#include <stdexcept>

namespace uw_ct_vio {

DvlFactor::DvlFactor(gtsam::Key xi, gtsam::Key vi, gtsam::Key xj, gtsam::Key vj, double tau, double dt_s,
                     const gtsam::Vector3& v_meas_D, const gtsam::Vector3& omega_body, const gtsam::Pose3& T_BD,
                     const gtsam::SharedNoiseModel& noise)
    : Base(noise, xi, vi, xj, vj), tau_(tau), dt_(dt_s), vMeas_(v_meas_D), omega_(omega_body), T_BD_(T_BD) {
  if (!(tau >= 0.0 && tau <= 1.0)) throw std::invalid_argument("DvlFactor: tau must be in [0, 1]");
  if (!(dt_s > 0.0)) throw std::invalid_argument("DvlFactor: dt_s must be > 0");
  if (!v_meas_D.allFinite() || !omega_body.allFinite() || !T_BD.matrix().allFinite())
    throw std::invalid_argument("DvlFactor: non-finite velocity, angular rate or extrinsic");
  if (!noise || noise->dim() != 3) throw std::invalid_argument("DvlFactor: noise model must be 3-D");
}

gtsam::Vector4 DvlFactor::hermiteDerivative(double tau) {
  const double t2 = tau * tau;
  return gtsam::Vector4(6 * t2 - 6 * tau, 3 * t2 - 4 * tau + 1, -6 * t2 + 6 * tau, 3 * t2 - 2 * tau);
}

gtsam::Vector DvlFactor::evaluateError(const gtsam::Pose3& Xi, const gtsam::Vector3& Vi, const gtsam::Pose3& Xj,
                                       const gtsam::Vector3& Vj, boost::optional<gtsam::Matrix&> H1,
                                       boost::optional<gtsam::Matrix&> H2, boost::optional<gtsam::Matrix&> H3,
                                       boost::optional<gtsam::Matrix&> H4) const {
  const gtsam::Vector4 d = hermiteDerivative(tau_);
  const bool nearI = tau_ <= 0.5;
  const gtsam::Matrix3 Rn = (nearI ? Xi.rotation() : Xj.rotation()).matrix();
  const gtsam::Matrix3 A = T_BD_.rotation().matrix().transpose() * Rn.transpose();  // world -> D
  // dp/dt of the cubic Hermite curve; the dt factors of the velocity terms cancel.
  const gtsam::Vector3 vW =
      (d(0) * Xi.translation() + d(2) * Xj.translation()) / dt_ + d(1) * Vi + d(3) * Vj;
  const gtsam::Vector3 vBody = Rn.transpose() * vW;  // body-frame velocity of the body origin
  const gtsam::Vector3 pred =
      T_BD_.rotation().matrix().transpose() * (vBody + omega_.cross(T_BD_.translation()));

  // Pose3 tangent is [omega; v] with right perturbation: d(t)/d(v) = R, d(R^T w)/d(omega) = [R^T w]x.
  const gtsam::Matrix3 dRot = T_BD_.rotation().matrix().transpose() * gtsam::skewSymmetric(vBody);
  if (H1) {
    *H1 = gtsam::Matrix::Zero(3, 6);
    if (nearI) H1->block<3, 3>(0, 0) = dRot;
    H1->block<3, 3>(0, 3) = A * (d(0) / dt_) * Xi.rotation().matrix();
  }
  if (H2) *H2 = A * d(1);
  if (H3) {
    *H3 = gtsam::Matrix::Zero(3, 6);
    if (!nearI) H3->block<3, 3>(0, 0) = dRot;
    H3->block<3, 3>(0, 3) = A * (d(2) / dt_) * Xj.rotation().matrix();
  }
  if (H4) *H4 = A * d(3);
  return pred - vMeas_;
}

}  // namespace uw_ct_vio
