// Prior on a pose restricted to a subspace of its tangent space (degeneracy-aware fusion).
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace uw_ct_vio {

/**
 * Residual U^T Log(T_ref^-1 T), with U (6 x k, orthonormal columns) spanning the directions of
 * the pose tangent space [omega; v] that the frame's vision constrains poorly. Holds those
 * directions to T_ref (the IMU prediction) and leaves the well-constrained ones to vision, in the
 * spirit of partially constrained factors (Hinduja, Ho, Kaess, IROS 2019).
 */
class PartialPosePrior : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3>;

  PartialPosePrior(gtsam::Key key, const gtsam::Pose3& T_ref, const gtsam::Matrix& U, double sigma);

  gtsam::Vector evaluateError(const gtsam::Pose3& T, boost::optional<gtsam::Matrix&> H = boost::none) const override;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new PartialPosePrior(*this)));
  }

 private:
  gtsam::Pose3 ref_;
  gtsam::Matrix Ut_;  // k x 6
};

}  // namespace uw_ct_vio
