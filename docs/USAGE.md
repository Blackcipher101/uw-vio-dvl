# Build, run and validation status (details)

[Back to README](../README.md)

## Build

GTSAM 4.2.0 with `GTSAM_BUILD_UNSTABLE=ON` installed to a prefix, ROS 2 Jazzy (rosbag2, geometry_msgs,
sensor_msgs), OpenCV 4, Eigen 3. Source ROS from **bash** (zsh leaves PYTHONPATH unset).

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths uw_ct_vio --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_DIR=$GTSAM_PREFIX/lib/cmake/GTSAM -DGTSAM_UNSTABLE_DIR=$GTSAM_PREFIX/lib/cmake/GTSAM_UNSTABLE -DBUILD_TESTING=ON
export LD_LIBRARY_PATH=$GTSAM_PREFIX/lib:$LD_LIBRARY_PATH
colcon test --packages-select uw_ct_vio && colcon test-result --verbose   # test_estimator_dvl takes ~4 min
python3 -m pytest -q scripts/test_sim_dvl.py
```

## Run (NTNU example)

```bash
C=<ntnu>/calibration
run_bag --bag <ntnu>/traj_01/traj_01_ros2 \
  --intrinsics $C/stereo_intrinsics_under_water/camchain-ariel_stereo_underwater.yaml --stereo-from-intrinsics 1 \
  --imucam $C/stereo_extrinsics_above_water/camchain-imucam-ariel_stereo_extrinsics_above_water.yaml \
  --imu-yaml <imu noise yaml> --left cam1 --right cam0 \
  --left-topic /alphasense_driver_ros/cam1 --right-topic /alphasense_driver_ros/cam0 \
  --imu-topic /alphasense_driver_ros/imu --pressure-topic /mavros/imu/static_pressure \
  --lcn 2 --lcn-stereo 0 --huber-k 0.5 \
  [--dvl-file dvl.csv --dvl-extrinsic dvl.yaml | --dvl-topic /dvl/twist] [--vision-blackout 60:20] \
  [--frame-log frames.csv] --out traj_01.tum
python3 scripts/sim_dvl.py <mocap.tum> --hand-eye-est <reference est.tum> --imucam <imucam.yaml> --cam cam1 \
  --scenario none|periodic|long|bursts|outliers --out dvl.csv
python3 scripts/evaluate.py <mocap.tum> traj_01.tum --hand-eye --hand-eye-first-s 60
```
`scripts/dvl_bench_ws1.sh` is the 6-trajectory benchmark runner used for the table (paths for our machine;
set `UWVIO`, `DATA`, `RUNNER`).

## Not validated

Real DVL data; the `--dvl-topic` reader; DVL extrinsic / scale / time-offset estimation; vehicle runs.
An independent code review of the DVL change was started and not completed.

Licence: none yet (choose before publishing).
