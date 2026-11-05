// Continuous-time depth (pressure) factor between two consecutive states.
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <string>

namespace uw_ct_vio {

/**
 * Depth measured at time t, with t_i <= t <= t_j, between states (X_i, V_i) and (X_j, V_j).
 *
 * The body position at t is the cubic Hermite interpolant of (p_i, v_i, p_j, v_j): the mean of a
 * white-noise-on-acceleration Gaussian process in translation. The sensor sits at lever arm
 * `l_body` in the body frame; its rotation is taken from the nearer state (error below
 * |l| * |dR| / 2, < 1 mm for |l| = 0.3 m and 0.4 deg between states at 20 Hz).
 *
 * World frame is z-up (gtsam PreintegrationParams::MakeSharedU), depth is positive downwards, so
 * the residual is z_sensor(t) - (z_surface - depth).
 */
class DepthGpFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3>;

  /// tau = (t - t_i) / dt in [0, 1]; dt = t_j - t_i > 0 [s]; depth [m]; z_surface: world z of the
  /// water surface [m]; noise: 1-D (sigma of the depth reading).
  DepthGpFactor(gtsam::Key xi, gtsam::Key vi, gtsam::Key xj, gtsam::Key vj, double tau, double dt_s, double depth_m,
                double z_surface_m, const gtsam::Vector3& l_body, const gtsam::SharedNoiseModel& noise);

  gtsam::Vector evaluateError(const gtsam::Pose3& Xi, const gtsam::Vector3& Vi, const gtsam::Pose3& Xj,
                              const gtsam::Vector3& Vj, boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none,
                              boost::optional<gtsam::Matrix&> H4 = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new DepthGpFactor(*this)));
  }

  /// Hermite basis (h00, h10, h01, h11) at tau.
  static gtsam::Vector4 hermite(double tau);

 private:
  double tau_, dt_, depth_, zSurface_;
  gtsam::Vector3 lBody_;
};

}  // namespace uw_ct_vio
