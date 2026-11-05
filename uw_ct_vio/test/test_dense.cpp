// Step 6 building blocks: stereo rectification, ICP on organised clouds, the dense between factor, and the
// estimator's handling of dense constraints.
#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <Eigen/QR>

#include <cmath>
#include <vector>

#include "uw_ct_vio/DenseBetweenFactor.hpp"
#include "uw_ct_vio/DenseRegistration.hpp"
#include "uw_ct_vio/DenseStereo.hpp"
#include "uw_ct_vio/Estimator.hpp"

using gtsam::symbol_shorthand::X;
using namespace uw_ct_vio;

namespace {

struct Plane {
  Eigen::Vector3d n;  // unit normal pointing into the free space
  double d;           // n . x = d
};

constexpr double kFx = 175.0, kBaselineM = 0.12;

/// Organised cloud of a plane scene seen from T_W_C (identity rectification), noise free.
OrganisedCloud renderCloud(const std::vector<Plane>& planes, const gtsam::Pose3& T_W_C) {
  OrganisedCloud c;
  c.stride = 4;
  c.fx = c.fy = kFx, c.cx = 180.0, c.cy = 135.0;
  c.resize(68, 90);
  const Eigen::Matrix3d R = T_W_C.rotation().matrix();
  const Eigen::Vector3d o = T_W_C.translation();
  for (int r = 0; r < c.rows; ++r) {
    for (int col = 0; col < c.cols; ++col) {
      const Eigen::Vector3d dir_cam((col * c.stride - c.cx) / c.fx, (r * c.stride - c.cy) / c.fy, 1.0);
      const Eigen::Vector3d dir_w = R * dir_cam;
      double best = 1e9;
      for (const Plane& pl : planes) {
        const double den = pl.n.dot(dir_w);
        if (den >= -1e-9) continue;
        const double t = (pl.d - pl.n.dot(o)) / den;
        if (t > 0.3 && t < best) best = t;
      }
      if (best > 8.0) continue;
      const size_t i = c.index(r, col);
      c.p[i] = dir_cam * best;  // z = best: depth along the optical axis
      c.sigma_z[i] = static_cast<float>(best * best * 0.5 / (kFx * kBaselineM));
      c.flags[i] = OrganisedCloud::kPoint;
    }
  }
  estimateNormals(&c, 0.1);
  return c;
}

const std::vector<Plane> kBoxCorner = {{{0, 0, -1}, -3.0}, {{-1, 0, 0}, -1.0}, {{0, -1, 0}, -1.0}};  // wall, wall, floor

gtsam::Matrix66 randomRotation6() {
  Eigen::Matrix<double, 6, 6> A;
  A << 1, 2, 0, 1, 0, 3, 0, 1, 4, 0, 2, 1, 2, 0, 1, 5, 1, 0, 1, 1, 0, 2, 6, 1, 3, 0, 1, 0, 1, 7, 0, 4, 1, 1, 0, 2;
  return Eigen::HouseholderQR<Eigen::Matrix<double, 6, 6>>(A).householderQ();
}

CameraModel fisheye(double fx, double fy) {
  CameraModel c;
  c.distortion = CameraModel::Distortion::Equidistant;
  c.fx = fx, c.fy = fy, c.cx = 360.0, c.cy = 270.0, c.width = 720, c.height = 540;
  c.coeffs = {-0.02, 0.005, -0.003, 0.0005};
  return c;
}

}  // namespace

TEST(DenseRegistration, BoxCornerRecoversRelativePose) {
  const gtsam::Pose3 T_W_ref(gtsam::Rot3::RzRyRx(0.02, -0.03, 0.01), gtsam::Point3(0.05, 0.0, 0.0));
  const gtsam::Pose3 T_ref_cur(gtsam::Rot3::RzRyRx(0.015, -0.02, 0.03), gtsam::Point3(0.10, -0.05, 0.08));
  const OrganisedCloud ref = renderCloud(kBoxCorner, T_W_ref), cur = renderCloud(kBoxCorner, T_W_ref * T_ref_cur);
  ASSERT_GT(ref.validNormals(), 1000);
  const gtsam::Pose3 init = T_ref_cur * gtsam::Pose3(gtsam::Rot3::RzRyRx(0.01, 0.0, 0.015), gtsam::Point3(0.03, -0.02, 0.02));
  const IcpResult r = registerClouds(ref, cur, init);
  ASSERT_TRUE(r.ok) << toString(r.reason);
  const gtsam::Vector6 err = gtsam::Pose3::Logmap(T_ref_cur.between(r.T_ref_cur));
  EXPECT_LT(err.tail<3>().norm(), 1e-3);
  EXPECT_LT(err.head<3>().norm(), 0.05 * M_PI / 180.0);
  EXPECT_EQ(denseSubspace(r.H, 1e3).rank(), 6) << "three orthogonal planes constrain all six directions";
}

TEST(DenseRegistration, SinglePlaneKeepsRankThreeAndDoesNotSlide) {
  const gtsam::Pose3 T_W_ref;
  const gtsam::Pose3 T_ref_cur(gtsam::Rot3(), gtsam::Point3(0.10, 0.05, 0.08));
  const std::vector<Plane> wall = {kBoxCorner[0]};
  const OrganisedCloud ref = renderCloud(wall, T_W_ref), cur = renderCloud(wall, T_ref_cur);
  const gtsam::Pose3 init(gtsam::Rot3(), gtsam::Point3(0.13, 0.03, 0.10));
  const IcpResult r = registerClouds(ref, cur, init);
  ASSERT_TRUE(r.ok) << toString(r.reason);
  EXPECT_LT(r.rmse_m, 1e-4);
  EXPECT_NEAR(r.T_ref_cur.x(), init.x(), 1e-9) << "translation along the wall is unobservable and must stay at the initial guess";
  const DenseSubspace s = denseSubspace(r.H, 1e3);
  EXPECT_EQ(s.rank(), 3);
  DenseBetweenFactor f(X(0), X(1), r.T_ref_cur, gtsam::Pose3(), s, 3.0, 1.345);
  EXPECT_EQ(f.rank(), 3);
}

TEST(DenseRegistration, RejectsInitialGuessFarFromTheScene) {
  const OrganisedCloud ref = renderCloud(kBoxCorner, gtsam::Pose3()), cur = renderCloud(kBoxCorner, gtsam::Pose3());
  const IcpResult r = registerClouds(ref, cur, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0, 5.0)));
  EXPECT_FALSE(r.ok);
  const IcpResult nan = registerClouds(ref, cur, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(NAN, 0, 0)));
  EXPECT_FALSE(nan.ok);
  EXPECT_EQ(nan.reason, IcpFailure::NonFinite);
}

namespace {

struct FactorFixture {
  gtsam::Pose3 T_BC0{gtsam::Rot3::RzRyRx(0.3, -0.2, 0.5), gtsam::Point3(0.1, 0.05, -0.2)};
  gtsam::Pose3 Xa{gtsam::Rot3::RzRyRx(0.1, 0.2, -0.3), gtsam::Point3(1, 2, 3)};
  gtsam::Pose3 Xb = Xa * gtsam::Pose3(gtsam::Rot3::RzRyRx(0.05, -0.02, 0.04), gtsam::Point3(0.2, -0.1, 0.05));
  gtsam::Pose3 meas = (Xa * T_BC0).between(Xb * T_BC0);
  DenseSubspace sub;
  FactorFixture() {
    const gtsam::Matrix66 Q = randomRotation6();
    gtsam::Vector6 lam;
    lam << 1e5, 5e4, 2e4, 1e4, 10, 1;
    sub = denseSubspace(Q * lam.asDiagonal() * Q.transpose(), 1e3);
  }
};

}  // namespace

TEST(DenseBetweenFactor, ZeroErrorAtTruthAndJacobiansMatchNumerical) {
  FactorFixture fx;
  ASSERT_EQ(fx.sub.rank(), 4);
  DenseBetweenFactor f(X(0), X(1), fx.meas, fx.T_BC0, fx.sub, 3.0, 1.345);
  EXPECT_LT(f.evaluateError(fx.Xa, fx.Xb).norm(), 1e-9);
  // Small offset: inside the Huber quadratic zone, where linearize() applies no robust down-weighting.
  const gtsam::Pose3 Xb_off = fx.Xb * gtsam::Pose3(gtsam::Rot3::RzRyRx(0.001, 0.002, -0.001), gtsam::Point3(0.003, -0.002, 0.001));
  EXPECT_GT(f.evaluateError(fx.Xa, Xb_off).norm(), 1e-4);
  gtsam::Values v;
  v.insert(X(0), fx.Xa);
  v.insert(X(1), Xb_off);
  EXPECT_TRUE(gtsam::internal::testFactorJacobians("DenseBetweenFactor", f, v, 1e-6, 1e-6));
}

TEST(DenseBetweenFactor, BodyInformationMatchesFactorJacobian) {
  FactorFixture fx;
  DenseBetweenFactor f(X(0), X(1), fx.meas, fx.T_BC0, fx.sub, 3.0, 1.345);
  gtsam::Matrix Ha, Hb;
  f.evaluateError(fx.Xa, fx.Xb, Ha, Hb);
  gtsam::Matrix W = Hb;
  for (int i = 0; i < W.rows(); ++i) W.row(i) /= 3.0 / std::sqrt(fx.sub.lambda(i));
  const gtsam::Matrix66 expected = W.transpose() * W;
  const gtsam::Matrix66 got = denseInfoInBody(fx.sub, 3.0, fx.T_BC0);
  EXPECT_LT((got - expected).norm(), 1e-6 * expected.norm());
}

TEST(DenseBetweenFactor, RejectsEmptySubspaceAndBadScale) {
  FactorFixture fx;
  EXPECT_THROW(DenseBetweenFactor(X(0), X(1), fx.meas, fx.T_BC0, denseSubspace(gtsam::Matrix66::Zero(), 1e3), 3.0, 1.345),
               std::invalid_argument);
  EXPECT_THROW(DenseBetweenFactor(X(0), X(1), fx.meas, fx.T_BC0, fx.sub, 0.0, 1.345), std::invalid_argument);
  EXPECT_EQ(denseSubspace(gtsam::Matrix66::Constant(NAN), 1e3).rank(), 0);
}

TEST(DenseStereo, RectificationDisparityRoundTripEquidistant) {
  const CameraModel left = fisheye(350.0, 351.0), right = fisheye(348.0, 349.5);
  const gtsam::Pose3 T_C0C1(gtsam::Rot3::RzRyRx(0.004, -0.01, 0.006), gtsam::Point3(kBaselineM, 0.002, 0.001));
  DenseStereo ds(DenseStereoConfig(), left, right, T_C0C1);
  EXPECT_NEAR(ds.baselineM(), kBaselineM, 0.01);
  for (const gtsam::Point3& P : {gtsam::Point3(0.3, -0.2, 2.5), gtsam::Point3(-0.8, 0.4, 1.5), gtsam::Point3(0.0, 0.0, 4.0)}) {
    const gtsam::Point3 P1 = T_C0C1.transformTo(P);
    const auto pl = left.project({cv::Point3f(P.x(), P.y(), P.z())});
    const auto pr = right.project({cv::Point3f(P1.x(), P1.y(), P1.z())});
    const auto rl = ds.rectifyPixels(pl, false), rr = ds.rectifyPixels(pr, true);
    ASSERT_EQ(rl.size(), 1u);
    ASSERT_EQ(rr.size(), 1u);
    EXPECT_NEAR(rl[0].y, rr[0].y, 0.05) << "rectified rows must coincide";
    const double disp = rl[0].x - rr[0].x;  // full resolution
    ASSERT_GT(disp, 0.0);
    const Eigen::Vector3d p = ds.pointFromDisparity(0.5 * rl[0].x, 0.5 * rl[0].y, 0.5 * disp);
    EXPECT_LT((p - Eigen::Vector3d(P.x(), P.y(), P.z())).norm(), 0.01 * P.norm()) << "point " << P.transpose();
  }
}

TEST(Estimator, DenseConstraintOutsideWindowIsDroppedNotThrown) {
  EstimatorConfig cfg;
  Estimator est(cfg);
  for (int i = 0; i <= 400; ++i) est.addImu({i / 200.0, gtsam::Vector3(0, 0, cfg.gravity_mps2), gtsam::Vector3::Zero()});
  ASSERT_TRUE(est.addFrame(0.5, {}));
  const gtsam::Matrix66 info = gtsam::Matrix66::Identity() * 1e6;
  const DenseConstraint stale{0.1, gtsam::Pose3(), info};
  ASSERT_TRUE(est.addFrame(0.55, {}, nullptr, &stale));
  EXPECT_EQ(est.counters().dense_stale, 1u);
  EXPECT_EQ(est.counters().dense_factors, 0u);
  EXPECT_FALSE(est.leftCameraAt(0.1).has_value());
  ASSERT_TRUE(est.leftCameraAt(0.5).has_value());

  const DenseConstraint good{0.5, gtsam::Pose3(), info};
  ASSERT_TRUE(est.addFrame(0.6, {}, nullptr, &good));
  EXPECT_EQ(est.counters().dense_factors, 1u);
  EXPECT_EQ(est.counters().dense_rank_sum, 6u);

  const DenseConstraint wrong{0.5, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0, 0)), info};
  ASSERT_TRUE(est.addFrame(0.65, {}, nullptr, &wrong));
  EXPECT_EQ(est.counters().dense_gated, 1u) << "a registration 1 m off the IMU-consistent estimate is rejected";

  gtsam::Matrix66 bad = info;
  bad(0, 0) = NAN;
  const DenseConstraint nan{0.5, gtsam::Pose3(), bad};
  ASSERT_TRUE(est.addFrame(0.7, {}, nullptr, &nan));
  EXPECT_EQ(est.counters().dense_invalid, 1u);
}
