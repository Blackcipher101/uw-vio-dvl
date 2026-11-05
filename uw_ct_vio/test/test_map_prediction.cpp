// Step 5 building blocks: camera projection and the estimator's side-effect-free IMU prediction.
#include <gtest/gtest.h>

#include <vector>

#include "uw_ct_vio/Estimator.hpp"
#include "uw_ct_vio/KltStereoTracker.hpp"

using uw_ct_vio::CameraModel;
using uw_ct_vio::Estimator;
using uw_ct_vio::EstimatorConfig;

namespace {

CameraModel makeCamera(CameraModel::Distortion d) {
  CameraModel c;
  c.distortion = d;
  c.fx = 350.0, c.fy = 351.0, c.cx = 360.0, c.cy = 270.0, c.width = 720, c.height = 540;
  c.coeffs = d == CameraModel::Distortion::Equidistant ? std::vector<double>{-0.02, 0.005, -0.003, 0.0005}
                                                       : std::vector<double>{-0.25, 0.07, 0.0005, -0.0003};
  return c;
}

void expectRoundTrip(const CameraModel& c) {
  std::vector<cv::Point2f> px;
  for (int y = 40; y < c.height; y += 100)
    for (int x = 40; x < c.width; x += 100) px.emplace_back(static_cast<float>(x), static_cast<float>(y));
  const auto n = c.undistort(px);
  std::vector<cv::Point3f> rays;
  for (const auto& p : n) rays.emplace_back(2.0f * p.x, 2.0f * p.y, 2.0f);  // any depth > 0
  const auto back = c.project(rays);
  ASSERT_EQ(back.size(), px.size());
  // cv::undistortPoints (radtan) iterates only 5 times, leaving a few hundredths of a pixel near the
  // corners; far below what an LK initial guess needs.
  for (size_t i = 0; i < px.size(); ++i) EXPECT_LT(cv::norm(back[i] - px[i]), 0.1) << "pixel " << px[i];
}

}  // namespace

TEST(CameraModel, ProjectInvertsUndistortFisheye) { expectRoundTrip(makeCamera(CameraModel::Distortion::Equidistant)); }

TEST(CameraModel, ProjectInvertsUndistortRadTan) { expectRoundTrip(makeCamera(CameraModel::Distortion::RadTan)); }

// predictLeftCamera() must not consume IMU samples: the frame it predicted must still be addable,
// and the IMU-only estimate of that frame must match the prediction.
TEST(Estimator, PredictionHasNoSideEffects) {
  EstimatorConfig cfg;
  Estimator est(cfg);
  const double yawRate = 0.5;  // [rad/s], so the prediction is not trivially the identity
  for (int i = 0; i <= 200; ++i)
    est.addImu({i / 200.0, gtsam::Vector3(0, 0, cfg.gravity_mps2), gtsam::Vector3(0, 0, yawRate)});
  EXPECT_FALSE(est.predictLeftCamera(0.55).has_value()) << "no prediction before initialisation";
  ASSERT_TRUE(est.addFrame(0.5, {}));
  const auto p1 = est.predictLeftCamera(0.55);
  const auto p2 = est.predictLeftCamera(0.55);
  ASSERT_TRUE(p1 && p2);
  EXPECT_TRUE(p1->equals(*p2, 1e-12));
  EXPECT_FALSE(est.predictLeftCamera(2.0).has_value()) << "no IMU coverage up to t";
  ASSERT_TRUE(est.addFrame(0.55, {})) << "prediction consumed the IMU samples addFrame needs";
  const gtsam::Pose3 T = est.latestLeftCamera();
  EXPECT_LT(gtsam::Rot3::Logmap(T.rotation().between(p1->rotation())).norm(), 1e-3);
  EXPECT_LT((T.translation() - p1->translation()).norm(), 1e-3);
  EXPECT_NEAR(gtsam::Rot3::Logmap(est.latestLeftCamera().rotation()).z(), 0.05 * yawRate, 2e-3);
}
