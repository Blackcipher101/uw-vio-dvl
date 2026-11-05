# uw_ct_vio: tightly coupled DVL that survives DVL dropouts (plan)

Status: implemented 2026-11-04 (see "As built" at the end; deviations from the plan are listed there). Base = uw_ct_vio as of 2026-10-30 (pressure depth, LCN, in-water stereo,
gw 2.866e-5, huber 0.5 best on dev). Data reality: none of our bags has a DVL. Real DVL data =
Heriot-Watt Tank dataset (form-gated, CC BY-NC-SA) / SOLAQUA (unverified). Until then a DVL is
SIMULATED from NTNU mocap; every number from it is labelled "sim-DVL" and proves plumbing and
dropout behaviour, not real-sensor accuracy.

## Measurement model

A DVL reports the velocity of its own frame D, expressed in D, at time t_d (mid-ping), 4-15 Hz,
with a validity flag (bottom lock) and sometimes a figure of merit.

    v_D = R_BD^T ( R_WB(t)^T v_W(t) + omega_B(t) x l_B )

- R_BD, l_B: DVL extrinsic in the body (= IMU) frame, from config (`--dvl-extrinsic` yaml, 4x4
  T_B_D). Not estimated (yet).
- v_W(t): derivative of the same cubic Hermite position curve the depth factor uses between states
  (X_i, V_i, X_j, V_j), so depth and DVL see one consistent trajectory.
- R_WB(t): rotation of the nearer state (as DepthGpFactor); states are ~50 ms apart and the ROV
  turns slowly, so the error is < 0.5 deg at 10 deg/s. Upgrade to geodesic interpolation if the
  Jacobian test shows it matters.
- omega_B(t): bias-corrected gyro, linearly interpolated from imu_ at factor creation, held
  constant in the factor (not a variable). Only matters with a lever arm.
- t_d is shifted by `dvl_time_offset_s` (config, default 0).

Noise: per-axis sigma = sqrt(sigma_abs^2 + (rel * |v_D|)^2), defaults 0.01 m/s and 1 %
(Waterlinked A50-class); Huber(huber_k) robust kernel.

## Dropout and outlier handling (the point of this work)

1. Validity: samples with valid=0, non-finite values, or |v| > dvl_max_speed_mps are dropped and
   counted (bottom-lock loss, out of range).
2. Innovation gate: at creation the factor's residual is evaluated at the IMU prediction (X_j, V_j
   predicted, X_i, V_i = latest). Whitened with sigma_meas^2 + dvl_gate_pred_sigma_mps^2; rejected
   above the chi2(3) quantile dvl_gate_quantile (0.999). Catches water-column lock, fish, spikes.
3. Re-lock rule: if the gate rejects for > dvl_relock_after_s AND vision is not healthy (vision gated
   on the current frame), the IMU prediction is the likelier culprit -> accept (counted as relock).
   If vision is healthy and the DVL keeps disagreeing, the DVL stays rejected (consistently wrong
   lock). This mirrors the existing vision-gate reasoning.
4. Through a DVL gap nothing special happens: IMU + vision + depth keep going; when the DVL returns
   it goes through the same gate (no reinitialisation, no jump).
5. Vision recovery with DVL: when the vision gate triggers a restart and an accepted DVL sample is
   younger than dvl_recent_s (1 s), the restart keeps the current (DVL-constrained) velocity with
   sigma dvl_recovery_vel_sigma_mps (0.05) instead of zeroing it.
6. Counters (printed in "done:"): dvl_received, dvl_invalid, dvl_gated, dvl_factors, dvl_relocks,
   dvl_gaps (> dvl_gap_s), dvl_longest_gap_s. Throttled warnings on gate rejects and gaps.

## Inputs

- `--dvl-file <csv>`: `t,vx,vy,vz,valid` (DVL frame, m/s, same clock as the bag header stamps).
  Sidecar file so simulated DVL needs no bag rewrite (laptop disk is full).
- `--dvl-topic <topic>`: geometry_msgs/TwistWithCovarianceStamped or TwistStamped (real bags);
  covariance used if non-zero, else the model above.
- `--dvl-extrinsic <yaml>`, `--dvl-sigma-abs`, `--dvl-sigma-rel`, `--dvl-time-offset`.
- Test-only: `--vision-blackout <t0>:<dur>[,...]` (seconds from bag start) feeds empty feature lists
  in those windows, to measure dead reckoning through camera loss.

## Simulated DVL (scripts/sim_dvl.py)

From NTNU mocap TUM (mocap body M) + hand-eye X = T_M_C (from evaluate.py on a reference run) +
Kalibr T_C_I -> T_M_I. DVL placed at T_I_D (default: 0.15 m below the IMU, yawed 45 deg, A50 style).
Positions smoothed (Savitzky-Golay), differentiated for v_W and omega; v_D at the DVL point.
Noise: white sigma_abs + rel, slow scale bias (0.3 %), optional mis-mount (0.5 deg).
Rate 8 Hz with jitter. Dropout scenarios: none, periodic (5 s off / 30 s), long (20 s altitude
loss), random bursts, outliers (2 % spikes + 3 s water-column lock at ~0 m/s with valid=1).

## Acceptance tests

Unit (gtest test_dvl_factor): Jacobians vs numerical on both sides of the midpoint, with lever arm
and extrinsic rotation; exact for a cubic trajectory without lever; rejects invalid arguments.
Synthetic estimator: (a) vision blackout 10 s, drift with DVL << without; (b) outlier burst
rejected by the gate (counter > 0, no drift jump); (c) 5 s DVL gap then return: no pose jump
beyond noise.
NTNU 6 traj (sim-DVL, ws1): no-DVL base vs DVL full vs each dropout scenario vs 20 s camera
blackout with/without DVL. Pass: DVL off is byte-identical to the base; DVL on never diverges;
dropout scenarios stay between DVL-full and no-DVL; blackout drift with DVL well below without.
Real-data validation (Tank dataset) is pending the user's download request.

## As built (2026-11-04)

Files: `include/uw_ct_vio/DvlFactor.hpp`, `src/DvlFactor.cpp`, `Estimator.{hpp,cpp}`, `app/run_bag.cpp`,
`test/test_dvl_factor.cpp`, `test/test_estimator_dvl.cpp`, `scripts/sim_dvl.py`, `scripts/test_sim_dvl.py`,
`scripts/dvl_bench_ws1.sh`. New dependency: `geometry_msgs` (run_bag only).

### Estimator behaviour that differs from or refines the plan

- **Time**: `addDvl(t, v_D, valid[, cov])` adds `dvl_time_offset_s` to `t` (so t is in the same clock as
  the IMU: the bag header stamps). Samples are consumed in `addFrame` for `(tFrame_, t]`, after the vision
  section, so the vision-health of the *current* frame is known to the re-lock rule. Samples older than the
  previous frame (e.g. everything before initialisation) are dropped and counted in `dvl_invalid`.
- **Gyro for the lever term** is read from `imu_` before `preintegrate()` prunes it (bias-corrected with
  `latestBias_`, linearly interpolated at the DVL stamp).
- **Noise**: per-axis `sqrt(abs^2 + (rel |v_i|)^2)`; the diagonal of the message covariance replaces it when
  all three entries are > 0 (off-diagonals ignored). Huber(`huber_k`) wraps the diagonal model.
- **Gate**: residual of the new factor evaluated at the IMU prediction (state i = latest estimate, state j =
  `pim.predict`), `chi2 = sum r_i^2 / (sigma_i^2 + dvl_gate_pred_sigma_mps^2)` vs the chi2(3) quantile
  `dvl_gate_quantile`. Default `dvl_gate_pred_sigma_mps` = 0.05 m/s (my choice, tune on ws1).
  The gate is skipped for the first sample after initialisation and after every smoother restart
  (`reset()`), because the velocity is then unknown (prior 0.5 m/s); with a DVL-kept velocity (below) it is on.
- **Vision health (new, separate signal)**: `FrameStats::vision_healthy = vision_used && landmark_factors >=
  vision_healthy_min_factors` (10). The existing gate (`vision_used`, `consecutiveGated_`) is unchanged, so
  empty feature lists still count as "used" there; only the DVL re-lock rule and the frame log read the new
  signal. New `FrameStats` fields: `candidates`, `stereo`, `vision_healthy`, `dvl_factors`, `dvl_gated`.
- **Re-lock**: a run of gate rejections starts at the first rejected stamp; once it is >= `dvl_relock_after_s`
  (2 s) old and the current frame is not `vision_healthy`, the next sample is accepted (`dvl_relocks`). Any
  accepted sample clears the run. With healthy vision the DVL stays rejected.
- **Recovery**: at a vision-gate restart, if the newest accepted DVL sample is <= `dvl_recent_s` old, the
  restart keeps the current velocity with `dvl_recovery_vel_sigma_mps` (counter `dvl_recoveries_kept`);
  otherwise the old behaviour (zero / `recovery_keep_velocity`).
- **Counters**: `dvl_received` (every call while enabled), `dvl_invalid` (valid=0, non-finite, out of order,
  > `dvl_max_speed_mps`, stale), `dvl_gated`, `dvl_factors`, `dvl_relocks`, `dvl_gaps` /
  `dvl_longest_gap_s` (spacing between consecutive *accepted* samples above `dvl_gap_s` = 1 s; so gated
  runs count as gaps, a trailing gap at the end of a run is not counted). Warnings use the existing throttle.
- DVL off (`dvl_enabled=false`, the default): `addDvl` returns immediately; no factor, queue or output
  changes. Verified byte-identical, see below.

### run_bag

New flags: `--dvl-file`, `--dvl-topic`, `--dvl-extrinsic` (yaml `T_B_D: 4x4`), `--dvl-sigma-abs`,
`--dvl-sigma-rel`, `--dvl-time-offset`, `--dvl-gate-quantile`, plus `--dvl-gate-pred-sigma`,
`--dvl-relock-after`, `--dvl-max-speed`, `--dvl-recovery-vel-sigma`; test flags `--vision-blackout
t0:dur[,t0:dur]` (seconds after the first image; the tracker still runs, the estimator gets an empty
feature list) and `--frame-log <csv>`. The `done:` line gets `; dvl_received ... dvl_recoveries_kept N`
inserted before ` -> out` only when a DVL is configured (and `; blackout_frames N` with a blackout), so the
line is unchanged otherwise and the `uwct_ablate2.sh` regex still matches. `--dvl-topic` reads the type
from the bag metadata (TwistStamped or TwistWithCovarianceStamped, twist.linear taken as the DVL frame).
CSV samples are fed when `t + offset <= current IMU stamp`.

Frame log columns (fixed order, header row, one row per frame that `addFrame` accepted):
`t, tracked, stereo, new_landmarks, landmark_factors, gated, candidates, vision_used, vision_healthy,
consecutive_gated, recoveries, dvl_factors, dvl_gated, img_mean, img_std, n_features`
(`tracked`/`stereo` = tracker stats of the frame; `candidates` = features matching a live landmark;
`consecutive_gated`/`recoveries` after the frame; `dvl_*` = samples of this frame interval; `img_*` = mean and
standard deviation of the left image the tracker saw (resized grey, before LCN); `n_features` = features
handed to the estimator, 0 inside a blackout.)

### sim_dvl.py

`scripts/sim_dvl.py <mocap.tum> (--T-M-C file | --hand-eye-est est.tum) --imucam <yaml> --cam cam1
--scenario none|periodic|long|bursts|outliers --out <csv> [--seed N]`; writes `<csv>`, `<stem>.json` and
`<stem>.yaml` (T_B_D). Notes:
- `evaluate.py --json` stores only the hand-eye translation and rotation angle, so X cannot be rebuilt from
  it: pass the reference run with `--hand-eye-est` (it calls `evaluate.hand_eye`, first 60 s by default) or
  a 4x4 file with `--T-M-C`.
- Default mount: 0.15 m along the IMU axis that points down, yaw 45 deg about it. Down is derived from the
  mocap start pose assuming a z-up mocap world and snapped to an IMU axis (NTNU traj_06: +z of the IMU);
  `--down-axis` overrides. Confirm against the still-bag accelerometer mean (specific force points up).
- Mocap is split into segments at stamp gaps > 0.2 s; per segment Savitzky-Golay (0.25 s window, order 3)
  gives the IMU-point velocity, body rate from consecutive slerped rotations; no samples within 0.3 s of a
  segment edge or inside a mocap gap (traj_06 has 27 segments, ~1600 samples at 8 Hz).
- Scenario dropouts are `valid=0` rows with zero velocity; the water-column lock is ~0 m/s with valid=1.
- Scale bias is constant per run (0.3 %, sign from the seed), not slowly varying.

### Verification done (laptop)

- `test_dvl_factor` (4 tests): Jacobians on both sides of the midpoint with lever arm + extrinsic rotation,
  exact for cubic motion, omega x l term, invalid arguments.
- `test_estimator_dvl` (5 tests; ~5 min, 300 landmarks, 40 s sim): (a) 10 s blackout drift 0.264 m without vs
  0.031 m with DVL; (b) 3 s outlier burst: 24 samples gated, 0 re-locks, drift 0.014 m, no step above the noise
  floor; (c) 5 s DVL gap with vision: no step above noise; inside a blackout the returning DVL corrects the
  drift (error 0.21 -> 0.05 m in one frame: a smoother correction, not a spurious jump); (d) re-lock when the
  IMU prediction is the culprit (1 re-lock, drift 1.3 m vs 6.8 m with the rule disabled); (e) DVL off with
  samples being fed is bit-identical to no samples.
- Smoke run, NTNU traj_06, sim-DVL (scenario none) + `--vision-blackout 100:20`, hand-eye ATE SE3 [m]: base
  0.589; blackout without DVL 1.409; blackout with DVL 0.685 (`dvl_received 1613, gated 96, factors 1517,
  gaps 24, longest 16.7 s, solver failures 0`). 96 gated samples (6 %) on clean sim data means the gate is
  tight for this base (its scale is off by ~9 %); consider `--dvl-gate-pred-sigma 0.1` on ws1. One run only,
  not a benchmark.
- DVL off vs the previous binary on NTNU traj_06: TUM byte-identical (`cmp`), identical `done:` counters.

### Not validated / open

- Sim-DVL only; no real DVL data. The simulator shares the estimator's kinematic model (same Hermite /
  extrinsic conventions apart from independent smoothing), so lever-arm or frame mistakes common to both
  would not show. Real validation needs the Tank dataset.
- `dvl_gate_pred_sigma_mps`, `dvl_relock_after_s`, `vision_healthy_min_factors` are untuned.
- Extrinsic (R_BD, l_B), scale and time offset are not estimated.

### NTNU 6-trajectory benchmark on ws1

Paths on ws1: bags `~/uw_datasets/ntnu/subset-3-mocap-gt/traj_0X/traj_0X_ros2`, mocap `.../traj_0X_mocap.tum`,
calibration `~/uwvio/calibration/...`, runner `~/uwvio/ntnu_all.sh <tag> [flags]` ({T} -> trajectory number),
reference run for the hand-eye `~/uwvio/results/ntnu_all/swgw/traj_0X.tum`.

1. Edit `BASE_FLAGS` (and check `IMUCAM`) at the top of `scripts/dvl_bench_ws1.sh`; the committed value
   (`--lcn 2 --huber-k 0.5`) is a placeholder for the exact swgw flag string.
2. Build: `colcon build --packages-select uw_ct_vio --parallel-workers 1` (needs geometry_msgs).
3. Run everything: `scripts/dvl_bench_ws1.sh` (simulates 5 scenarios x 6 trajectories into
   `~/uwvio/results/dvl_sim/traj_0X/<scenario>.{csv,json,yaml}`, then runs the tags in order):

   | tag | flags added to BASE_FLAGS |
   |---|---|
   | dvl_none_off | none (baseline) |
   | dvl_full | `--dvl-file .../none.csv --dvl-extrinsic .../none.yaml` |
   | dvl_periodic / dvl_long / dvl_bursts / dvl_outliers | same with the scenario CSV |
   | blackout20_nodvl | `--vision-blackout 60:20` |
   | blackout20_dvl | `--vision-blackout 60:20` + the dvl_full DVL flags |

   One tag: `scripts/dvl_bench_ws1.sh dvl_full`; reuse CSVs: `SKIP_SIM=1 ...`; other blackout window:
   `BLACKOUT=90:20 ...`. The blackout must fall inside the trajectory and ideally inside a stretch with mocap.
   Add `--frame-log ~/uwvio/results/dvl_sim/frames_{T}_<tag>.csv` to a tag's flags to collect the failure-model
   training data (every row carries vision_healthy, DVL counts and image statistics).
4. Pass criteria (plan): dvl_none_off identical to the existing swgw+huber0.5 result; DVL on never diverges
   (check `solver failures`, `gate recoveries` in the log); dropout scenarios between dvl_full and no-DVL;
   blackout20_dvl drift well below blackout20_nodvl.

## Results: sim-DVL benchmark, NTNU 2023 traj_01..06 (ws1, 2026-11-04)

All DVL numbers use a SIMULATED DVL (scripts/sim_dvl.py from mocap). They show plumbing and dropout
behaviour, not real-sensor accuracy. Base = tag h05 (`--lcn 2 --lcn-stereo 0 --stereo-from-intrinsics 1
--imu-yaml configs_dev/imu_gw1e-2.yaml --huber-k 0.5`), reproduced exactly before the change (0.401 m / 6.75 %).
Mean over 6 trajectories, ATE SE3 with hand-eye (first 60 s), RPE % over 5 m. Camera blackout = 20 s of empty
feature lists starting 60 s after the first image (`--vision-blackout 60:20`).

| Tag | DVL | Camera | Mean ATE (m) | Mean RPE (%) | Worst traj ATE (m) |
|---|---|---|---|---|---|
| dvl_none_off | off | normal | 0.401 | 6.75 | 0.642 |
| dvl_full | sim, no dropouts | normal | 0.393 | 6.71 | 0.646 |
| dvl_periodic | 5 s off every 30 s | normal | 0.398 | 6.71 | 0.658 |
| dvl_long | one 20 s gap | normal | 0.393 | 6.71 | 0.646 |
| dvl_bursts | random bursts | normal | 0.396 | 6.71 | 0.650 |
| dvl_outliers | 2 % spikes + 3 s water-column lock (valid=1) | normal | 0.400 | 6.71 | 0.650 |
| blackout20_nodvl | off | 20 s blackout | **3.876** | 33.81 | 10.711 |
| blackout20_dvl | sim, no dropouts | 20 s blackout | **0.663** | 7.31 | 0.936 |
| blackout20_dvlgap7 | sim, 7 s DVL loss inside the blackout (65-72 s) | 20 s blackout | **0.888** | 9.39 | 1.527 |

Reading:
- DVL off is byte-identical to the base (DVL-off output = h05, also checked with cmp on the laptop).
- With working cameras the DVL adds little (0.401 -> 0.393 m): stereo VIO already constrains velocity, and
  the sim DVL carries a realistic ~2.4-2.5 deg mount error (the hand-eye rotation residual used to place it).
- Through a 20 s camera blackout the DVL cuts the mean ATE 5.8x (3.88 -> 0.66 m) and the worst case from
  10.7 m to 0.94 m; the two vision-gate restarts without DVL (traj_02, 03) do not happen with it.
- DVL lost for 7 s in the middle of the camera blackout (both sensors down at once): 0.888 m, between full DVL
  (0.663) and no DVL (3.876). Samples actually dropped per trajectory: 2/41/29/0/6/0 (mocap gaps already
  remove DVL there on traj_04/06), so the effect is concentrated on traj_02/03/05.
- No dropout or outlier scenario does worse than DVL off. Outlier scenario: 89-185 samples gated per
  trajectory, 0 re-locks (vision healthy); in the blackout run 1 re-lock (traj_03).
- Weakness of the scenario design: the sim emits no samples inside mocap gaps, so the scheduled `long`
  20 s gap mostly fell into an existing mocap gap (only 19 samples dropped on traj_06); real coverage of the
  "long DVL loss" case comes from the mocap gaps themselves (longest DVL gap per run 16-42 s).

Not validated: real DVL hardware/data (Tank dataset pending the user's request), `--dvl-topic` path,
extrinsic/scale/time-offset estimation, keep-velocity-on-recovery path in a real run (0 triggers).
