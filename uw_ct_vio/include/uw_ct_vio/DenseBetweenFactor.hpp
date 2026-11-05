// Relative-pose factor from dense stereo registration, restricted to the directions the registration constrains.
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace uw_ct_vio {

/// Eigen-directions of an ICP information matrix above a floor: U is 6 x k (orthonormal columns,
/// tangent order [omega; v]), lambda their eigenvalues.
struct DenseSubspace {
  gtsam::Matrix U;
  gtsam::Vector lambda;
  int rank() const { return static_cast<int>(lambda.size()); }
};
DenseSubspace denseSubspace(const gtsam::Matrix66& H_cam, double min_eig);

/**
 * Registration measurement meas = T_Cref_Ccur (left camera of X_b in the left camera of X_a) between two
 * body poses, in CAMERA coordinates so the ICP Hessian applies unchanged:
 *   error = U^T Log( meas^-1 * (X_a T_BC0)^-1 * (X_b T_BC0) ),
 * whitened with sigma_i = cov_scale / sqrt(lambda_i) (ICP Hessians are over-confident) and Huber-robustified.
 * Directions the registration does not constrain (e.g. in-plane motion against one wall) are not touched.
 */
class DenseBetweenFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3>;

  /// Throws std::invalid_argument if the subspace is empty or cov_scale / huber_k are not positive.
  DenseBetweenFactor(gtsam::Key key_a, gtsam::Key key_b, const gtsam::Pose3& meas_Cref_Ccur, const gtsam::Pose3& T_BC0,
                     const DenseSubspace& subspace, double cov_scale, double huber_k);

  gtsam::Vector evaluateError(const gtsam::Pose3& Xa, const gtsam::Pose3& Xb, boost::optional<gtsam::Matrix&> Ha = boost::none,
                              boost::optional<gtsam::Matrix&> Hb = boost::none) const override;

  /// Squared Mahalanobis norm of the error at the given body poses (unrobustified).
  double mahalanobisSq(const gtsam::Pose3& Xa, const gtsam::Pose3& Xb) const;
  int rank() const { return static_cast<int>(Ut_.rows()); }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new DenseBetweenFactor(*this)));
  }

 private:
  gtsam::Pose3 meas_, T_BC0_;
  gtsam::Matrix Ut_;       // k x 6
  gtsam::Vector sigmas_;   // k
};

/// Information the factor gives the pose X_b (body tangent [omega; v], X_a held fixed) at zero error:
/// delta_cam = Ad(T_BC0^-1) delta_body, so H_body = A^T U diag(lambda / cov_scale^2) U^T A.
gtsam::Matrix66 denseInfoInBody(const DenseSubspace& subspace, double cov_scale, const gtsam::Pose3& T_BC0);

}  // namespace uw_ct_vio
