#include <gtest/gtest.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

#include "uw_ct_vio/DepthGpFactor.hpp"

using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using uw_ct_vio::DepthGpFactor;

namespace {

const auto kNoise = gtsam::noiseModel::Isotropic::Sigma(1, 0.02);

gtsam::Values states(const gtsam::Pose3& xi, const gtsam::Vector3& vi, const gtsam::Pose3& xj, const gtsam::Vector3& vj) {
  gtsam::Values v;
  v.insert(X(0), xi);
  v.insert(V(0), vi);
  v.insert(X(1), xj);
  v.insert(V(1), vj);
  return v;
}

}  // namespace

TEST(DepthGpFactor, JacobiansMatchNumericalOnBothSidesOfMidpoint) {
  const gtsam::Pose3 xi(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.7), gtsam::Point3(1.0, -2.0, -3.0));
  const gtsam::Pose3 xj(gtsam::Rot3::RzRyRx(0.15, -0.1, 0.9), gtsam::Point3(1.03, -1.98, -3.05));
  const gtsam::Vector3 vi(0.4, 0.1, -0.3), vj(0.35, 0.2, -0.25);
  for (double tau : {0.0, 0.3, 0.5, 0.7, 1.0}) {
    DepthGpFactor f(X(0), V(0), X(1), V(1), tau, 0.05, 3.1, 0.2, gtsam::Vector3(0.12, -0.05, 0.3), kNoise);
    EXPECT_TRUE(gtsam::internal::testFactorJacobians("DepthGpFactor", f, states(xi, vi, xj, vj), 1e-6, 1e-6))
        << "tau=" << tau;
  }
}

TEST(DepthGpFactor, ExactForCubicMotionWithoutLeverArm) {
  // Hermite interpolation reproduces any cubic position curve from its end points and velocities.
  const gtsam::Vector3 a(0.2, -1, -2), b(0.5, 0.1, -0.4), c(-0.3, 0.2, 0.6), d(0.8, -0.1, -1.2);
  auto pos = [&](double t) { return gtsam::Vector3(a + b * t + c * t * t + d * t * t * t); };
  auto vel = [&](double t) { return gtsam::Vector3(b + 2 * c * t + 3 * d * t * t); };
  const double ti = 0.4, dt = 0.3, zSurface = 0.5;
  const gtsam::Pose3 xi(gtsam::Rot3::Yaw(0.3), pos(ti)), xj(gtsam::Rot3::Yaw(0.5), pos(ti + dt));
  for (double tau : {0.1, 0.45, 0.9}) {
    const double depth = zSurface - pos(ti + tau * dt).z();
    DepthGpFactor f(X(0), V(0), X(1), V(1), tau, dt, depth, zSurface, gtsam::Vector3::Zero(), kNoise);
    EXPECT_NEAR(f.evaluateError(xi, vel(ti), xj, vel(ti + dt))(0), 0.0, 1e-12) << "tau=" << tau;
  }
}

TEST(DepthGpFactor, LeverArmUsesNearerStateRotation) {
  const gtsam::Pose3 xi(gtsam::Rot3::Roll(M_PI / 2), gtsam::Point3(0, 0, -1));  // lever y-axis -> world z
  const gtsam::Pose3 xj(gtsam::Rot3(), gtsam::Point3(0, 0, -1));
  const gtsam::Vector3 zero = gtsam::Vector3::Zero(), lever(0, 0.3, 0);
  DepthGpFactor near_i(X(0), V(0), X(1), V(1), 0.2, 0.05, 1.0, 0.0, lever, kNoise);
  DepthGpFactor near_j(X(0), V(0), X(1), V(1), 0.8, 0.05, 1.0, 0.0, lever, kNoise);
  EXPECT_NEAR(near_i.evaluateError(xi, zero, xj, zero)(0), 0.3, 1e-12);  // Roll(90deg) maps +y to +z
  EXPECT_NEAR(near_j.evaluateError(xi, zero, xj, zero)(0), 0.0, 1e-12);
}

TEST(DepthGpFactor, RejectsInvalidArguments) {
  const gtsam::Vector3 l = gtsam::Vector3::Zero();
  EXPECT_THROW(DepthGpFactor(X(0), V(0), X(1), V(1), 1.2, 0.05, 1, 0, l, kNoise), std::invalid_argument);
  EXPECT_THROW(DepthGpFactor(X(0), V(0), X(1), V(1), 0.5, 0.0, 1, 0, l, kNoise), std::invalid_argument);
  EXPECT_THROW(DepthGpFactor(X(0), V(0), X(1), V(1), 0.5, 0.05, NAN, 0, l, kNoise), std::invalid_argument);
  EXPECT_THROW(DepthGpFactor(X(0), V(0), X(1), V(1), 0.5, 0.05, 1, 0, l, gtsam::noiseModel::Isotropic::Sigma(2, 0.1)),
               std::invalid_argument);
}
