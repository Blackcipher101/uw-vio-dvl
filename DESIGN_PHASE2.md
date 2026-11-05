# Phase 2 design: `uw_ct_vio`, a continuous-time GTSAM stereo-inertial odometry

Status: draft for review, 2026-10-29. Nothing implemented yet.

## Why, and what it must beat

Phase 0 showed: on bus_outside (COLMAP GT ~20 % small; Sim3 ATE / RPE) SVIn BRISK ~0.29–0.33 m /
4.4 %; on NTNU traj_06 (mocap GT, hand-eye evaluation) ROVIO 0.65 m / 11.7 % and SVIn BRISK 2.44–2.86 m
/ 27–33 %. SVIn fails in the dark, low-texture pool (FAST: 5–49 corners/frame; strong CLAHE gives
thousands of noise corners and makes SVIn diverge). Targets: NTNU < 0.65 m, bus_outside ≤ 0.29 m Sim3.

## Scope of v1

- Stereo + IMU odometry, optional depth (pressure) as a continuous-time factor. No loop closure (Phase 4).
- Offline runner reading the rosbag2 directly (`rosbag2_cpp::Reader`), deterministic, no ROS graph:
  no orphan nodes, no replay-rate effects. A ROS 2 node wrapper comes after the offline numbers.
- Same TUM output and evaluator (`scripts/evaluate.py`, `--hand-eye` for NTNU).

## Package layout

`~/uw-vslam-bench/uw_ct_vio/` (ament_cmake, C++17), built with colcon against
`~/uw-vslam-bench/deps/gtsam-4.2.0` (official borglab 4.2.0 + `GTSAM_BUILD_UNSTABLE=ON`, system Eigen).
- `core/` (no ROS): `StereoCamera` (radtan or equidistant via OpenCV `cv::undistortPoints` /
  `cv::fisheye::undistortPoints`), `KltStereoTracker`, `Estimator`, custom factors, config.
- `app/run_bag.cpp`: offline runner (rosbag2_cpp, sensor_msgs Image/CompressedImage/Imu/FluidPressure).
- `test/`: gtest; factor Jacobians via `EXPECT_TRUE(gtsam::internal::testFactorJacobians(name, factor,
  values, step, tol))` (gtsam/nonlinear/factorTesting.h; the `EXPECT_CORRECT_FACTOR_JACOBIANS` macro is
  CppUnitLite-only); synthetic stereo+IMU world with known truth.

## Front end: patch tracking, not corner matching

ROVIO (patch-based) survives the pool where BRISK does not. `KltStereoTracker`:
- CLAHE (same clip/tiles as SVIn's config), then `cv::goodFeaturesToTrack` on a grid with a low
  quality level to keep ~150 tracks spread over the image (new features only in empty cells).
- Temporal: `cv::calcOpticalFlowPyrLK` left(t-1)->left(t), forward-backward check (< 1 px), then
  RANSAC on undistorted bearings (essential matrix) to drop outliers.
- Stereo: LK left->right seeded at the IMU/depth-predicted disparity, epipolar check on undistorted
  normalised coordinates (|dy| < 2 px-equivalent), forward-backward check.
- Output per frame: track id, undistorted normalised (x, y) in left and, when matched, right camera.
- Alternative input: the existing track file (`TrackFileReader` format) so SP+LG / cuVSLAM tracks can be
  compared through the same back end.

## Back end

- **Variables** per camera frame k: `X(k)` Pose3 (T_world_body, body = IMU), `V(k)` Vector3 world
  velocity, `B(k)` imuBias::ConstantBias. Landmarks `L(j)` Point3 world.
- **IMU:** `PreintegratedCombinedMeasurements` with `PreintegrationCombinedParams::MakeSharedU(g)`, noise
  from the Kalibr IMU yaml; one `CombinedImuFactor(X(k-1),V(k-1),X(k),V(k),B(k-1),B(k))` per frame.
- **Vision:** for each observation, `GenericProjectionFactor<Pose3, Point3, Cal3_S2>` with
  `Cal3_S2(1,1,0,0,0)` on the undistorted normalised point (works for fisheye too, since undistortion is
  done in the front end), `body_P_sensor = T_BC` for left and right camera, isotropic sigma
  `pixel_sigma / f`, Huber loss (k = 1.345). Landmarks are created by stereo triangulation in the left
  camera when first seen in both views, with depth limits.
- **Continuous time:** depth readings arrive between camera frames. `DepthGpFactor(X(i),V(i),X(j),V(j))`
  evaluates the world z of the body at time t ∈ (t_i, t_j) by cubic Hermite interpolation of position
  from the two poses and velocities (the mean of a white-noise-on-acceleration GP in translation) plus
  the body-to-pressure-sensor lever arm, residual = z(t) − (−depth). Implemented with
  `NoiseModelFactorN<Pose3, Vector3, Pose3, Vector3>`; Jacobians analytic, checked numerically.
  Pressure to depth: (p − p_surface)/(ρ g), ρ configurable (fresh vs sea water); p_surface from the first
  second of data, flagged in the log.
- **Solver:** `IncrementalFixedLagSmoother` (gtsam_unstable) with lag 3 s; every variable gets a
  timestamp in the `KeyTimestampMap`; a landmark's timestamp is updated to its latest observation so it
  stays while observed. Output = the latest state after each update (causal), written as TUM of the left
  camera (`T_world_body * T_BC`).
- **Gating (GenZ-style deviation gate):** before adding a frame's vision factors, project each landmark
  with the IMU-predicted pose; observations with reprojection error > 5 sigma are dropped. If more than
  70 % are dropped, the frame adds no vision factors (logged), so a burst of bad tracks cannot pull the
  state away from the IMU.
- **Initialisation:** roll/pitch from the mean accelerometer over the first 0.5 s (vehicle need not be
  static: warn if accel-norm deviates > 0.3 m/s² from g), yaw 0, position 0, velocity prior 0 ± 0.5 m/s,
  bias priors from config. Vision factors start once a frame has ≥ 15 stereo landmarks.

## Failure handling

- IMU gap > 50 ms between frames: propagate with the last sample, inflate covariance, log WARN; > 1 s:
  reset the estimator (new keys, priors from last good state) and log ERROR.
- Non-finite / out-of-order stamps: drop the message, count, log throttled.
- Smoother `IndeterminantLinearSystemException`: log, drop that frame's vision factors, retry once with
  IMU only; on second failure reset.

## Tests and gates

1. Unit: `DepthGpFactor` Jacobians (`testFactorJacobians` under gtest), Hermite interpolation exact for
   constant-acceleration motion, stereo triangulation, undistortion round trips (radtan, equidistant).
2. Synthetic: simulated stereo + IMU on a 60 s trajectory with known landmarks and noise; ATE < 5 cm,
   no divergence with 20 % outlier tracks.
3. Bags: NTNU traj_06 and bus_outside through `evaluate.py`; must beat the targets above outside the
   run-to-run spread (3 runs; deterministic offline runner should make spread ~0).

## Review log

- 2026-10-29 Codex (medium, pack = gtsam 4.2.0 source): flagged scripts/evaluate.py and TrackFileReader
  as missing — rejected, they live in uw-vslam-bench/scripts and svin_ws okvis_frontend (outside the
  pack). Confirmed: EXPECT_CORRECT_FACTOR_JACOBIANS is CppUnitLite-only (factorTesting.h:115) -> fixed.

## Open questions for review

1. States at every frame (20 Hz, 60 states in a 3 s lag) vs keyframes: is iSAM2 fast enough with
   ~150 landmarks/frame, or should the window hold keyframes only?
2. Projection factors per observation vs `SmartStereoProjectionPoseFactor` (gtsam_unstable): smart
   factors avoid landmark variables but interact awkwardly with fixed-lag marginalisation.
3. Is cubic Hermite position interpolation a sound stand-in for the WNOA GP mean, given IMU factors
   already constrain the states?
4. Underwater refraction: Kalibr intrinsics calibrated in water absorb most of it; is a pinhole/fisheye
   model on in-water intrinsics adequate at this scale (2–15 m), or is an explicit refractive model needed?
