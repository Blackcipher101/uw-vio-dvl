# uw-vio-dvl

Underwater stereo-inertial-pressure VIO (GTSAM fixed-lag) with a tightly coupled DVL velocity factor whose gating
and re-lock logic keep DVL dropouts from corrupting the estimate. Research code; never run on a vehicle.

| 20 s camera blackout in Stonefish: vision-only vs with DVL | Top-down trajectory, NTNU traj_03 |
|---|---|
| [![Stonefish camera blackout](docs/media/blackout_clear.gif)](sim/README.md) | [![traj_03 trajectory](docs/figures/hero_trajectory.png)](docs/figures/hero_trajectory.png) |
| physically simulated DVL, single run, aligned on the pre-blackout segment: vision-only error grows to 8.8 m, with the DVL it stays under 0.5 m ([sim details](sim/README.md)) | mocap ground truth vs 20 s blackout without DVL and with simulated DVL |

## Results

**NTNU underwater 2023**, 6 trajectories with mocap (mean ATE SE3 with hand-eye, RPE over 5 m). **The DVL is
simulated** from mocap (noise, scale bias, ~2.5 deg mount error, dropouts, outliers):

| configuration | mean ATE (m) | mean RPE (%) | worst traj (m) |
|---|---|---|---|
| VIO + pressure, cameras normal (base) | 0.401 | 6.75 | 0.642 |
| + DVL (sim), periodic / long / burst dropouts, outliers | 0.393-0.400 | 6.71 | 0.658 |
| 20 s camera blackout, no DVL | 3.876 | 33.81 | 10.711 |
| 20 s camera blackout, **DVL (sim)** | **0.663** | 7.31 | 0.936 |
| 20 s camera blackout, DVL also lost for 7 s inside it | 0.888 | 9.39 | 1.527 |

Baselines on the same bags: Basalt 1.10 m (in-air stereo extrinsic), SVIn 1.80 m, ORB-SLAM3 initialises on 3/6,
OpenVINS 3/6.

**Stonefish seabed** (physically simulated DVL with natural bottom-lock loss, exact ground truth, single runs):

| clear water | ATE (m) |
|---|---|
| no blackout | 0.30 |
| 20 s camera blackout, no DVL | 3.69 |
| 20 s camera blackout, with DVL | **0.32** |

In turbid water vision-only diverges and the DVL keeps it at 4.23 m.

## Quick start

```bash
source /opt/ros/jazzy/setup.bash    # from bash, not zsh
colcon build --base-paths uw_ct_vio --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_DIR=$GTSAM_PREFIX/lib/cmake/GTSAM -DGTSAM_UNSTABLE_DIR=$GTSAM_PREFIX/lib/cmake/GTSAM_UNSTABLE -DBUILD_TESTING=ON
colcon test --packages-select uw_ct_vio && colcon test-result --verbose
run_bag --bag <ntnu>/traj_01/traj_01_ros2 ... --out traj_01.tum   # full flag list in docs/USAGE.md
```

## More

- [docs/RESULTS.md](docs/RESULTS.md): estimator overview, the full results table, all figures (error vs time,
  per-trajectory ATE, dropout robustness, simulated DVL example) and the Stonefish simulation.
- [docs/USAGE.md](docs/USAGE.md): build dependencies, the NTNU run example, the DVL simulator, evaluation, the
  benchmark runner and what is not validated.
- [sim/README.md](sim/README.md): Stonefish benchmark details and caveats.
- [DESIGN_DVL.md](DESIGN_DVL.md) (DVL design, as-built notes), [DESIGN_V2.md](DESIGN_V2.md) and
  [DESIGN_PHASE2.md](DESIGN_PHASE2.md) (estimator history and ablations).

Not validated: real DVL data, the `--dvl-topic` reader, DVL extrinsic / scale / time-offset estimation, vehicle runs.
Licence: none yet (choose before publishing).
