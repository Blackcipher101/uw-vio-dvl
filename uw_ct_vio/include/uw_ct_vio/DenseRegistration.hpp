// Point-to-plane ICP between two organised stereo clouds with projective data association.
#pragma once

#include <gtsam/geometry/Pose3.h>

#include "uw_ct_vio/DenseStereo.hpp"

namespace uw_ct_vio {

struct IcpConfig {
  int max_iterations = 15;
  double stop_step = 1e-5;             ///< stop when the Gauss-Newton step norm [rad, m] is below this
  double max_corr_start_m = 0.3;       ///< point-to-plane gate at the first iteration [m]
  double max_corr_end_m = 0.05;        ///< ... shrunk geometrically to this
  double max_corr_sigmas = 3.0;        ///< a far point's gate widens to this many of its own sigmas along the normal
  double max_normal_angle_rad = 0.5236;  ///< 30 deg: correspondences with more differing normals are rejected
  double min_sigma_m = 0.01;           ///< model-error floor added to the stereo noise along the normal [m]
  double solve_rel_eig = 1e-4;         ///< eigenvalues below this fraction of the largest are not stepped along ...
  double solve_abs_eig = 1e3;          ///< ... unless above this; set to the factor's min_eig so every direction
                                       ///< the factor keeps was actually solved for, not left at the initial guess
  int min_inliers = 300;
  double min_inlier_fraction = 0.3;    ///< of the source points that have a normal
  double max_trans_dev_m = 0.5;        ///< registration this far from the initial guess is rejected [m]
  double max_rot_dev_rad = 0.1745;     ///< ... or this far in rotation (10 deg)
};

enum class IcpFailure { None, NonFinite, TooFewInliers, LowInlierFraction, Diverged };
const char* toString(IcpFailure f);

struct IcpResult {
  bool ok = false;
  IcpFailure reason = IcpFailure::TooFewInliers;
  gtsam::Pose3 T_ref_cur;     ///< pose of the current left camera frame in the reference left camera frame
  gtsam::Matrix66 H = gtsam::Matrix66::Zero();  ///< information of T_ref_cur in the tangent [omega; v] (right perturbation) [1/m^2-ish]
  int inliers = 0, iterations = 0;
  double inlier_fraction = 0.0, rmse_m = 0.0;   ///< rmse of the point-to-plane residuals of the inliers [m]
  bool converged = false;
};

/// Both clouds are in their own original left camera frame. T_ref_cur_init is the initial guess
/// (e.g. from the IMU prediction); the result is rejected if it ends up far from it.
IcpResult registerClouds(const OrganisedCloud& ref, const OrganisedCloud& cur, const gtsam::Pose3& T_ref_cur_init,
                         const IcpConfig& cfg = IcpConfig());

}  // namespace uw_ct_vio
