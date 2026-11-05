# uw_ct_vio v2: robust front end + degeneracy-aware GTSAM back end (plan)

Status: plan, 2026-10-29. Builds on uw_ct_vio v4 (KLT + GTSAM fixed-lag, NTNU traj_06 raw 1.07 m /
9.4 %, bus_outside 0.54–1.07 m Sim3). Baselines in comparisons: SVIn, ORB-SLAM3 (stereo-inertial),
OpenVINS, Basalt — all on the same raw bags. ROVIO is out of scope.

## Principle

Raw images in. No enhancement, no denoised or re-lit images. Robustness to darkness, lamp falloff,
turbidity and featureless scenes lives in the tracker (how it compares patches) and in the estimator
(how much it trusts each measurement, and in which directions). What the lamp-correction trial taught
us is used as a *noise model*, not as an image edit.

## Where each idea comes from

| Idea | Source | Becomes |
|---|---|---|
| Grid Shi-Tomasi + pyramidal LK + NCC check, keyframe when tracked fraction drops | cuVSLAM (arXiv 2506.04359) | F1, F4 |
| Per-patch affine brightness (gain + bias) in alignment | DSO, ROVIO, UniVIO | F1 |
| Lamp falloff / vignetting field as per-pixel noise | our lightcone trial | F1 (covariance, not image) |
| SuperPoint/ALIKED + LightGlue for association in dim light | SuperVINS, AirVO/AirSLAM, DIVO | F2 |
| Point-line features for low texture | AirVO/AirSLAM, PL-VINS | F3 |
| Direct frame-to-map registration, incremental local map | FAST-LIO2 | F4 |
| Adaptive budget by scene scale (PD controller) | GenZ-LIO | F5 |
| Gravity prior decoupling roll/pitch, adaptively weighted | GeVI-SLAM (arXiv 2510.24533) | B2 |
| Update only well-constrained directions | Hinduja, Ho, Kaess IROS 2019; X-ICP; GenZ | B3 |
| Hybrid point-to-plane / point-to-point with per-point covariance | GenZ-LIO | B5 |
| Dense registration factors, fixed-lag + submap global graph | GLIM / gtsam_points | B5, B6 |
| Continuous-time async sensors (depth, DVL) | DIVO | B1 |

## Front end

**F1. Photometrically robust KLT (raw images).** cuVSLAM-style grid Shi-Tomasi + pyramidal LK, with:
- affine brightness per patch (gain a, bias b) estimated inside the LK iteration, so lamp falloff,
  auto-exposure and turbidity changes do not break tracks;
- zero-mean NCC acceptance test after convergence (reject < 0.8) instead of raw forward-backward only;
- a per-pixel noise model sigma(x) = sigma_sensor / g(x), with g(x) the lamp/vignetting field estimated
  online from running per-pixel percentiles (the lightcone fit, done on statistics, never applied to
  pixels). Each observation's reprojection sigma scales with the patch's local SNR, so dark-corner
  tracks count less instead of being amplified.

**F2. Learned association on keyframes.** ALIKED (commercially usable; SuperPoint weights are not) +
LightGlue on keyframes only (~5 Hz, GPU/TensorRT later):
- re-detect and re-associate after KLT loss (the bus_outside hard stretches);
- merge tracks that re-observe an existing landmark (longer tracks, less drift);
- descriptors for loop-closure candidates (B6).

**F3. Line features (later).** LSD/ELSED lines anchored to nearby points (AirVO); pool walls, cables and
hull plates are line-rich where points are scarce. GTSAM line factors (Plücker or endpoints).

**F4. Frame-to-map tracking.** Project live landmarks with the IMU-predicted pose and refine with the
F1 patch alignment against each landmark's reference patch (FAST-LIO2's "register to the map, not
the last frame"). Keyframe when the tracked fraction falls below a threshold (cuVSLAM).

**F5. Adaptive budget.** GenZ-style PD controller on grid density / features per cell, driven by the
median stereo depth and the fraction of cells with usable texture.

## Back end (GTSAM)

**B1. Keep:** IMU preintegration, fixed-lag iSAM2 (3 s), stereo projection factors, weak landmark prior,
reset on solver failure. **Add data:** continuous-time depth factor fed from pressure (NTNU
FluidPressure; bus_outside /bar30/depth custom message decoded by hand), DVL factor when available.

**B2. Gravity prior.** Roll/pitch factor from the accelerometer, weight adapted to |a| - g and gyro
rate (GeVI-SLAM): with slow ROV motion roll/pitch stay observable even when vision is weak.

**B3. Degeneracy-aware vision.** Each frame, form the visual information on the new pose (6x6 block
after marginalising its landmarks). Eigen-decompose it; directions with eigenvalue below a dynamic
threshold are degenerate. Apply vision only in the well-constrained subspace (projected / partial
constraint factor, Hinduja et al.); IMU, gravity and depth carry the rest. Replaces today's
all-or-nothing gate and reset. Logged per frame (which directions, eigenvalues).

**B4. Recovery without jumps.** When a reset is unavoidable, carry velocity from the last good state
and the vision-only relative motion instead of zeroing it.

**B5. Dense stereo factor.** SGBM depth (already in dehaze.py) -> per-keyframe cloud with
sigma_z = z^2 sigma_d / (f B) -> voxel-plane local map -> GenZ hybrid residual
(point-to-plane where planar, point-to-point otherwise), correspondences re-evaluated at each
relinearisation (GLIM matching-cost style). CPU first, gtsam_points GPU later.

**B6. Global (Phase 4).** GLIM-style submaps every few metres, F2-descriptor loop candidates, dense
submap alignment factors, global iSAM2 / GTSAM CUDA LM.

## Order, cost, and the test each step must pass

Each step is one switch in the config, measured on all 6 NTNU subset-3 trajectories (mocap, hand-eye)
plus bus_outside (Sim3), with the deterministic runner. A step stays only if the mean ATE and RPE over
the 7 sequences improve and no sequence diverges.

| # | Step | Effort | Expected effect |
|---|---|---|---|
| 0 | All 6 NTNU trajectories + SVIn, ORB-SLAM3, OpenVINS, Basalt on raw bags (build on workstation1) | 1–2 days, mostly unattended | the honest baseline table |
| 1 | B1 depth data (pressure) | hours | vertical drift down |
| 2 | F1 affine-brightness KLT + NCC + SNR-weighted sigma | 1 day | the lightcone gain without touching images |
| 3 | B2 gravity prior + B3 degeneracy-aware + B4 | 2 days | bus_outside hard stretches, no resets |
| 4 | F2 ALIKED + LightGlue keyframe association | 2 days | re-association after loss, longer tracks |
| 5 | F4 frame-to-map + F5 adaptive budget | 2 days | drift, keyframe count |
| 6 | B5 dense stereo factor | ~1 week | featureless-but-structured scenes |
| 7 | F3 lines, B6 global graph | later | low-texture structure, loop closure |

## Licences

GTSAM BSD; LightGlue Apache-2.0; ALIKED BSD-3; SuperPoint weights non-commercial (avoid); GenZ-LIO
GPL v2 and AirSLAM GPL v3 (ideas only, no code); cuVSLAM NVIDIA licence (ideas only).

## Results so far (2026-10-29)

Gate: mean ATE (SE3, hand-eye) over NTNU subset-3 traj_01..06, plus bus_outside (Sim3) as a check;
no divergence. Raw images, deterministic offline runner.

| Step | Variant | NTNU mean ATE / RPE | bus Sim3 / RPE | Verdict |
|---|---|---|---|---|
| — | no depth | 1.226 m / 12.56 % | — | reference |
| 1 | pressure depth (B1) | 1.154 / 11.90 | 0.581 / 4.38 | **kept** |
| 2 | LCN on tracking, temporal only, auto (`--lcn 2 --lcn-stereo 0`) | **0.690 / 8.70** | 0.581 / 4.38 (auto off) | **kept: current base** |
| 2 | NCC check, noise-aware NCC, SNR sigma | — | — | rejected (diverge on traj_06 or bus) |
| 3 | gravity prior / degeneracy-aware / keep-velocity | 1.139 / 1.162 / 1.160 (on step-1 base) | 0.583 / 2.25 / 0.581 | rejected (gravity marginal) |
| 4 | ALIKED+LightGlue tracks, keyframe re-association, augment / replace | 1.092 / 1.352 | 0.351 / 0.340 (step-1 base) | rejected: traj_06 breaks (wrong re-associations) |
| 4 | same, re-association off, augment / replace | 0.694 / 0.850 | 0.535 / 0.472, RPE worse | rejected: no NTNU gain, 3-8x estimator time |
| 5 | IMU/map-predicted KLT (`--map-predict`) | 0.668 / 8.70 | 0.818 / 9.40, 2 resets | rejected |
| 5 | re-acquisition of lost landmarks (`--reacquire`) | 0.693 / 8.83 | — | rejected: ~1 % accepted |
| 6 | dense SGBM keyframe ICP factor (`--dense`), cov scale 3 / 1 / 0.3 | 0.690 / 0.692 / 0.691 | — | rejected: ICP less accurate than the prediction (below) |

Step 6 diagnosis (`--dense-log` + `scripts/dense_vs_gt.py`, keyframes 0.5 s apart, ~20 cm motion): median
relative-pose error vs mocap, IMU/VIO prediction 3.2 cm vs ICP 5.1 cm (traj_01), 3.3 vs 8.4 cm (traj_05); ICP
better on 10-22 % of keyframes. At weight scale 1 or 0.3 the chi2 gate rejects most registrations (337/534 and
534/536 on traj_04), consistent with ICP being worse than its Hessian claims. Half-res SGBM through water on
pool surfaces at 1.5-3.5 m is not accurate enough to add information here. Code stays, default off (off =
byte-identical output, checked on all 6).

Baselines on the same 6 NTNU bags: SVIn 1.798 m (1.08 / 0.77 / 0.52 / 2.05 / 3.81 / 2.55, full coverage),
Basalt 1.104 m (full coverage; the evaluator coverage of ~0.80 is the ceiling for 20 Hz output vs 50 Hz GT, ours is 0.80 too), ORB-SLAM3 initialises on only 3 of 6 (coverage 0.24-0.64 of the 0.80 ceiling). OpenVINS (euroc_mav defaults): never initialises on 0/6 (static init needs a stop + jerk); with init_dyn_use=true 3/6 track (01-03 mean 1.21 m), 3/6 diverge.

## Held-out check: NTNU 2025 (2026-10-30)

7 sequences never used for tuning (subset-mclab 1-2 pool, subset-fjord 1-5 sea; fjord_6 kept blind, not yet run).
Reference = the dataset's ReAqROVIO trajectories, not mocap, so ATE is against another estimator; RPE (% per 5 m)
is the steadier number. Same frozen config as the 2023 base (`--lcn 2 --lcn-stereo 0`), calibration files
byte-identical to 2023.

Data-ingestion bug found on the way: the 2025 `/mavros/imu/static_pressure` is not in Pa but ~241 units per metre
(dataset card: `depth = -(p - 2660) / 241`, fresh water; matches the reference z with corr -0.99 at ~250/m on all
bags). The first run read depth changes 39x too small, which pinned the height (RPE 7-114 %, up to 13 restarts).
Fixed by `--pressure-scale 40.705` (= 1000 * 9.81 / 241, from the card, not fitted to the reference).

ATE m / RPE %:

| Seq | ours (pressure fixed) | ours (no pressure) | Basalt | SVIn | ORB-SLAM3 |
|---|---|---|---|---|---|
| mclab_1 | 0.916 / 5.16 | 0.759 / 4.73 | 0.668 / 3.09 | 0.727 / 3.02 | 0.495 / 4.32 |
| mclab_2 | 0.591 / 4.37 | 0.547 / 4.26 | 0.505 / 3.86 | 0.493 / 3.78 | 0.508 / 2.18 |
| fjord_1 | 0.867 / 4.99 | 0.960 / 5.59 | 0.708 / 4.06 | 3.885 / 34.5 | crash |
| fjord_2 | 1.271 / 5.24 | 1.477 / 4.95 | 1.032 / 5.04 | 1.307 / 4.58 | crash |
| fjord_3 | 0.792 / 5.42 | 0.728 / 5.68 | 0.616 / 4.80 | 0.429 / 3.99 | crash |
| fjord_4 | **0.578** / 3.58 | 0.906 / 3.33 | 0.841 / 3.23 | 1.308 / 7.52 | crash |
| fjord_5 | 4.218 / 12.7 | 4.309 / 13.4 | 1.176 / 2.98 | 2.865 / 6.75 | diverged (4988 m) |
| **mean** | **1.319 / 5.93** | 1.384 / 5.99 | **0.792 / 3.87** | 1.573 / 9.17 | 2/7 |

Verdict: on unseen data Basalt is better than us (ATE and RPE, 6 of 7 sequences); we beat SVIn and ORB-SLAM3 on
the mean, no divergence, at most 2 restarts per sequence. The 2023 lead (0.69 vs Basalt 1.10) came from surviving the
hard 2023 runs, not from accuracy. fjord_5 (4.2 m) is our clear failure. Tried and rejected on held-out: a widened
start-up gate (no gain; code removed) and `--lag 6` (worse). mclab_1/2 and fjord_1-5 were inspected while debugging,
so fjord_6 is the only blind sequence left.

### Calibration fix: in-water stereo pose (2026-10-30)

The left-right stereo pose came from the in-air cam-IMU chain; the dataset also ships an in-water stereo camchain.
They differ by 0.28 deg, almost all about the camera y axis: a 2.2 px disparity bias, i.e. 7-15 % depth error at
1.5-3 m (pool) and 28-54 % at 5-8 m (sea). Wrong depths turn part of each rotation into translation; fjord_5 lost
60 deg of yaw over +1017 deg of circling. `--stereo-from-intrinsics 1` takes T_cn_cnm1 from the `--intrinsics`
camchain (left cam-IMU pose kept). Ruled out on the way: gyro scale (raw gyro matches the reference turn, slope
1.000), fisheye undistortion (exact round trip), time offset (<= 15 ms, same as Basalt), camera-fixed features (none).

Config `swgw` = in-water stereo + documented pressure scale + gyro_random_walk 2.866e-5 (100x below the dataset file).
Basalt rerun with the same in-water stereo pose (and the same gyro value) for fairness:

| | 2023 (6, mocap) | dev 2025 (7) ATE / RPE |
|---|---|---|
| ours base (in-air stereo, pressure fixed) | 0.690 | 1.319 / 5.93 |
| ours swgw | **0.403** | 0.632 / 3.29 |
| Basalt in-air stereo | 1.104 | 0.792 / 3.87 |
| Basalt in-water stereo + gyro 2.866e-5 | not run | **0.583 / 2.24** |

Per dev sequence (ATE) ours / Basalt-water-gw: mclab_1 0.648 / 0.880, mclab_2 0.536 / 0.335, fjord_1 0.244 / 0.291,
fjord_2 0.778 / 0.923, fjord_3 0.252 / 0.286, fjord_4 0.528 / 0.786, fjord_5 1.437 / 0.579. Ours better on 5 of 7 by
ATE, Basalt better on the mean and on RPE (6 of 7): our local accuracy is the gap. fjord_6 still blind.

Open idea from step 4: re-association helped bus_outside a lot but merged wrong landmarks on traj_06;
verifying each re-association in 3D against the stereo landmark (not just epipolar RANSAC) may keep
the gain without the failure.
