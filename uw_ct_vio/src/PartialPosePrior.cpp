#include "uw_ct_vio/PartialPosePrior.hpp"

#include <gtsam/linear/NoiseModel.h>

#include <stdexcept>

namespace uw_ct_vio {

PartialPosePrior::PartialPosePrior(gtsam::Key key, const gtsam::Pose3& T_ref, const gtsam::Matrix& U, double sigma)
    : Base(gtsam::noiseModel::Isotropic::Sigma(static_cast<size_t>(U.cols()), sigma), key), ref_(T_ref), Ut_(U.transpose()) {
  if (U.rows() != 6 || U.cols() < 1 || U.cols() > 6) throw std::invalid_argument("PartialPosePrior: U must be 6 x k, 1 <= k <= 6");
  if (!(sigma > 0.0)) throw std::invalid_argument("PartialPosePrior: sigma must be > 0");
}

gtsam::Vector PartialPosePrior::evaluateError(const gtsam::Pose3& T, boost::optional<gtsam::Matrix&> H) const {
  if (!H) return Ut_ * gtsam::Pose3::Logmap(ref_.between(T));
  gtsam::Matrix66 Hb, Hl;  // d(between)/dT, d(Logmap)/d(between)
  const gtsam::Pose3 d = ref_.between(T, boost::none, Hb);
  const gtsam::Vector6 xi = gtsam::Pose3::Logmap(d, Hl);
  *H = Ut_ * Hl * Hb;
  return Ut_ * xi;
}

}  // namespace uw_ct_vio
