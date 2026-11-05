#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <cmath>

#include "uw_ct_vio/DvlFactor.hpp"

using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using uw_ct_vio::DvlFactor;

namespace {

const auto kNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.02);

gtsam::Values states(const gtsam::Pose3& xi, const gtsam::Vector3& vi, const gtsam::Pose3& xj, const gtsam::Vector3& vj) {
  gtsam::Values v;
  v.insert(X(0), xi);
  v.insert(V(0), vi);
  v.insert(X(1), xj);
  v.insert(V(1), vj);
  return v;
}

}  // namespace

TEST(DvlFactor, JacobiansMatchNumericalOnBothSidesOfMidpoint) {
  const gtsam::Pose3 xi(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.7), gtsam::Point3(1.0, -2.0, -3.0));
  const gtsam::Pose3 xj(gtsam::Rot3::RzRyRx(0.15, -0.1, 0.9), gtsam::Point3(1.03, -1.98, -3.05));
  const gtsam::Vector3 vi(0.4, 0.1, -0.3), vj(0.35, 0.2, -0.25);
  const gtsam::Pose3 T_BD(gtsam::Rot3::RzRyRx(0.02, -0.03, M_PI / 4), gtsam::Point3(0.12, -0.05, -0.15));
  for (double tau : {0.0, 0.3, 0.5, 0.7, 1.0}) {
    DvlFactor f(X(0), V(0), X(1), V(1), tau, 0.05, gtsam::Vector3(0.3, -0.1, 0.05), gtsam::Vector3(0.1, -0.2, 0.3),
                T_BD, kNoise);
    EXPECT_TRUE(gtsam::internal::testFactorJacobians("DvlFactor", f, states(xi, vi, xj, vj), 1e-6, 1e-6))
        << "tau=" << tau;
  }
}

TEST(DvlFactor, ExactForCubicMotionWithoutLeverArm) {
  // The derivative of the Hermite interpolant is exact for any cubic position curve.
  const gtsam::Vector3 a(0.2, -1, -2), b(0.5, 0.1, -0.4), c(-0.3, 0.2, 0.6), d(0.8, -0.1, -1.2);
  auto pos = [&](double t) { return gtsam::Vector3(a + b * t + c * t * t + d * t * t * t); };
  auto vel = [&](double t) { return gtsam::Vector3(b + 2 * c * t + 3 * d * t * t); };
  const double ti = 0.4, dt = 0.3;
  const gtsam::Rot3 R(gtsam::Rot3::Yaw(0.3));
  const gtsam::Pose3 xi(R, pos(ti)), xj(R, pos(ti + dt));
  const gtsam::Rot3 R_BD = gtsam::Rot3::RzRyRx(0.1, 0.2, -0.4);
  const gtsam::Pose3 T_BD(R_BD, gtsam::Point3::Zero());
  for (double tau : {0.1, 0.45, 0.9}) {
    // Same rotation at both states, so the nearer-state rule is exact here.
    const gtsam::Vector3 vD = R_BD.unrotate(R.unrotate(vel(ti + tau * dt)));
    DvlFactor f(X(0), V(0), X(1), V(1), tau, dt, vD, gtsam::Vector3::Zero(), T_BD, kNoise);
    EXPECT_NEAR(f.evaluateError(xi, vel(ti), xj, vel(ti + dt)).norm(), 0.0, 1e-12) << "tau=" << tau;
  }
}

TEST(DvlFactor, LeverArmAddsRotationRateCrossLever) {
  // Stationary body spinning about z: the DVL 0.3 m along body x sees omega x l = (0, 0.3 w, 0).
  const gtsam::Pose3 x(gtsam::Rot3(), gtsam::Point3(0, 0, 0));
  const gtsam::Vector3 zero = gtsam::Vector3::Zero();
  const gtsam::Pose3 T_BD(gtsam::Rot3(), gtsam::Point3(0.3, 0, 0));
  DvlFactor f(X(0), V(0), X(1), V(1), 0.4, 0.05, gtsam::Vector3(0, 0.15, 0), gtsam::Vector3(0, 0, 0.5), T_BD, kNoise);
  EXPECT_NEAR(f.evaluateError(x, zero, x, zero).norm(), 0.0, 1e-12);
}

TEST(DvlFactor, RejectsInvalidArguments) {
  const gtsam::Vector3 v = gtsam::Vector3::Zero();
  const gtsam::Pose3 I;
  const auto n1 = gtsam::noiseModel::Isotropic::Sigma(1, 0.02);
  const gtsam::Vector3 nan(NAN, 0, 0);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), -0.1, 0.05, v, v, I, kNoise), std::invalid_argument);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), 1.1, 0.05, v, v, I, kNoise), std::invalid_argument);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), 0.5, 0.0, v, v, I, kNoise), std::invalid_argument);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), 0.5, 0.05, nan, v, I, kNoise), std::invalid_argument);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), 0.5, 0.05, v, nan, I, kNoise), std::invalid_argument);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), 0.5, 0.05, v, v, I, n1), std::invalid_argument);
  EXPECT_THROW(DvlFactor(X(0), V(0), X(1), V(1), 0.5, 0.05, v, v, I, nullptr), std::invalid_argument);
}
