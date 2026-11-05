#include "uw_ct_vio/DenseBetweenFactor.hpp"

#include <Eigen/Eigenvalues>
#include <gtsam/linear/NoiseModel.h>

#include <cmath>
#include <stdexcept>

namespace uw_ct_vio {

DenseSubspace denseSubspace(const gtsam::Matrix66& H_cam, double min_eig) {
  DenseSubspace s;
  s.U = gtsam::Matrix::Zero(6, 0);
  s.lambda = gtsam::Vector::Zero(0);
  if (!H_cam.allFinite()) return s;
  Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> es(0.5 * (H_cam + H_cam.transpose()));
  int k = 0;
  for (int i = 0; i < 6; ++i) k += es.eigenvalues()(i) >= min_eig && es.eigenvalues()(i) > 0.0;
  s.U.resize(6, k);
  s.lambda.resize(k);
  int j = 0;
  for (int i = 5; i >= 0 && j < k; --i) {  // strongest direction first
    if (es.eigenvalues()(i) >= min_eig && es.eigenvalues()(i) > 0.0) {
      s.U.col(j) = es.eigenvectors().col(i);
      s.lambda(j++) = es.eigenvalues()(i);
    }
  }
  return s;
}

namespace {

gtsam::SharedNoiseModel makeNoise(const DenseSubspace& s, double cov_scale, double huber_k) {
  if (s.rank() < 1 || s.rank() > 6 || s.U.rows() != 6 || s.U.cols() != s.rank())
    throw std::invalid_argument("DenseBetweenFactor: subspace must be 6 x k with 1 <= k <= 6");
  if (!(cov_scale > 0.0) || !(huber_k > 0.0)) throw std::invalid_argument("DenseBetweenFactor: cov_scale and huber_k must be > 0");
  if (!(s.lambda.minCoeff() > 0.0)) throw std::invalid_argument("DenseBetweenFactor: eigenvalues must be > 0");
  return gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(huber_k),
                                           gtsam::noiseModel::Diagonal::Sigmas(cov_scale * s.lambda.array().rsqrt().matrix()));
}

}  // namespace

DenseBetweenFactor::DenseBetweenFactor(gtsam::Key key_a, gtsam::Key key_b, const gtsam::Pose3& meas_Cref_Ccur,
                                       const gtsam::Pose3& T_BC0, const DenseSubspace& subspace, double cov_scale,
                                       double huber_k)
    : Base(makeNoise(subspace, cov_scale, huber_k), key_a, key_b),
      meas_(meas_Cref_Ccur),
      T_BC0_(T_BC0),
      Ut_(subspace.U.transpose()),
      sigmas_(cov_scale * subspace.lambda.array().rsqrt().matrix()) {}

gtsam::Vector DenseBetweenFactor::evaluateError(const gtsam::Pose3& Xa, const gtsam::Pose3& Xb,
                                                boost::optional<gtsam::Matrix&> Ha, boost::optional<gtsam::Matrix&> Hb) const {
  if (!Ha && !Hb) {
    const gtsam::Pose3 d = (Xa * T_BC0_).between(Xb * T_BC0_);
    return Ut_ * gtsam::Pose3::Logmap(meas_.between(d));
  }
  gtsam::Matrix66 Hca, Hcb, Hd1, Hd2, He, Hl;
  const gtsam::Pose3 Ca = Xa.compose(T_BC0_, Hca);
  const gtsam::Pose3 Cb = Xb.compose(T_BC0_, Hcb);
  const gtsam::Pose3 d = Ca.between(Cb, Hd1, Hd2);
  const gtsam::Pose3 e = meas_.between(d, boost::none, He);
  const gtsam::Vector6 xi = gtsam::Pose3::Logmap(e, Hl);
  if (Ha) *Ha = Ut_ * Hl * He * Hd1 * Hca;
  if (Hb) *Hb = Ut_ * Hl * He * Hd2 * Hcb;
  return Ut_ * xi;
}

double DenseBetweenFactor::mahalanobisSq(const gtsam::Pose3& Xa, const gtsam::Pose3& Xb) const {
  return evaluateError(Xa, Xb).cwiseQuotient(sigmas_).squaredNorm();
}

gtsam::Matrix66 denseInfoInBody(const DenseSubspace& subspace, double cov_scale, const gtsam::Pose3& T_BC0) {
  const gtsam::Matrix66 A = T_BC0.inverse().AdjointMap();
  const gtsam::Matrix66 Hc = subspace.U * (subspace.lambda / (cov_scale * cov_scale)).asDiagonal() * subspace.U.transpose();
  return A.transpose() * Hc * A;
}

}  // namespace uw_ct_vio
