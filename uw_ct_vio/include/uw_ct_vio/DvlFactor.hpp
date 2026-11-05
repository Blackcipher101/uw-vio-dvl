// Continuous-time DVL (Doppler velocity log) velocity factor between two consecutive states.
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace uw_ct_vio {

/**
 * DVL velocity measured at time t, with t_i <= t <= t_j, between states (X_i, V_i) and (X_j, V_j).
 *
 * The DVL frame D is rigidly mounted on the body (= IMU) frame B at T_BD = (R_BD, l_B). It reports
 * the velocity of its own origin expressed in D:
 *
 *   v_D = R_BD^T ( R_WB(t)^T v_W(t) + omega_B x l_B )
 *
 * v_W(t) is the time derivative of the cubic Hermite position curve of (p_i, v_i, p_j, v_j), i.e. the
 * same curve DepthGpFactor evaluates, so depth and DVL constrain one trajectory. R_WB(t) is taken
 * from the nearer state (error ~ |v| * |dR| / 2, < 0.5 deg between states 50 ms apart at 10 deg/s);
 * omega_B is the bias-corrected gyro rate at t, a constant of the factor (not a variable).
 *
 * World frame is z-up; the state velocity V is the world-frame velocity of the body origin.
 * Residual = predicted v_D - measured v_D [m/s], 3-D.
 */
class DvlFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3>;

  /// tau = (t - t_i) / dt in [0, 1]; dt = t_j - t_i > 0 [s]; v_meas_D: measured velocity in D [m/s];
  /// omega_body: bias-corrected body angular rate at t [rad/s]; T_BD: DVL frame in the body frame;
  /// noise: 3-D (sigmas in m/s, DVL axes).
  DvlFactor(gtsam::Key xi, gtsam::Key vi, gtsam::Key xj, gtsam::Key vj, double tau, double dt_s,
            const gtsam::Vector3& v_meas_D, const gtsam::Vector3& omega_body, const gtsam::Pose3& T_BD,
            const gtsam::SharedNoiseModel& noise);

  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Vector3& Vi, const gtsam::Pose3& Xj,
                              const gtsam::Vector3& Vj, boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none,
                              boost::optional<gtsam::Matrix&> H4 = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new DvlFactor(*this)));
  }

  /// Derivatives of the Hermite basis (h00', h10', h01', h11') w.r.t. tau.
  static gtsam::Vector4 hermiteDerivative(double tau);

 private:
  double tau_, dt_;
  gtsam::Vector3 vMeas_, omega_;
  gtsam::Pose3 T_BD_;
};

}  // namespace uw_ct_vio
