#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>

#include "uw_ct_vio/PartialPosePrior.hpp"

using gtsam::symbol_shorthand::X;
using uw_ct_vio::PartialPosePrior;

TEST(PartialPosePrior, JacobiansMatchNumerical) {
  const gtsam::Pose3 ref(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3), gtsam::Point3(1, 2, 3));
  const gtsam::Pose3 T = ref * gtsam::Pose3(gtsam::Rot3::RzRyRx(0.05, 0.02, -0.04), gtsam::Point3(0.1, -0.05, 0.2));
  gtsam::Matrix U = gtsam::Matrix::Zero(6, 2);
  U(5, 0) = 1.0;                                        // translation along body z
  U(0, 1) = U(1, 1) = std::sqrt(0.5);                   // a mixed rotation direction
  PartialPosePrior f(X(0), ref, U, 0.02);
  gtsam::Values v;
  v.insert(X(0), T);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("PartialPosePrior", f, v, 1e-6, 1e-6));
}

TEST(PartialPosePrior, IgnoresUnconstrainedDirections) {
  const gtsam::Pose3 ref;
  gtsam::Matrix U = gtsam::Matrix::Zero(6, 1);
  U(3, 0) = 1.0;  // only translation x
  PartialPosePrior f(X(0), ref, U, 0.02);
  // A pure y translation and a pure yaw are invisible to this prior. (Combined, they are not:
  // Pose3::Logmap couples rotation and translation.)
  EXPECT_NEAR(f.evaluateError(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 1.5, 0)))(0), 0.0, 1e-12);
  EXPECT_NEAR(f.evaluateError(gtsam::Pose3(gtsam::Rot3::Yaw(0.3), gtsam::Point3()))(0), 0.0, 1e-12);
  EXPECT_NEAR(f.evaluateError(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.4, 0, 0)))(0), 0.4, 1e-12);
}

TEST(PartialPosePrior, RejectsBadShapes) {
  EXPECT_THROW(PartialPosePrior(X(0), gtsam::Pose3(), gtsam::Matrix::Zero(5, 1), 0.1), std::invalid_argument);
  EXPECT_THROW(PartialPosePrior(X(0), gtsam::Pose3(), gtsam::Matrix::Zero(6, 1), 0.0), std::invalid_argument);
}
