# Stonefish underwater benchmark

A physics-simulated seabed survey for `uw_ct_vio`. It has:

- a DVL simulated in Stonefish: ray-cast bottom lock, which is naturally lost for about 18 s over a 27 m deep pit;
- 720x540 stereo cameras at 10 Hz;
- a pressure sensor;
- a synthetic 200 Hz IMU;
- exact ground truth.

The scene is a 330 s survey: a 40 s hold, two legs over a shelf at about 10 m depth (one across the pit), and an orbit
around a container-and-pipe structure. There are two water variants, clear (Jerlov 0.10) and turbid (Jerlov 0.55).
Frames and constants are in `rig.py`; the scene and trajectory are in `trajectory.py`.

## Results (ws1, 2026-11-06)

| Variant | Config | ATE SE3 [m] | RPE 5 m [%] | Notes |
|---|---|---|---|---|
| clear | vision (stereo + IMU + pressure) | **0.30** | 3.08 | |
| clear | + DVL | **0.26** | 2.81 | 1539 DVL factors; bottom lock lost over the pit, 0 relocks needed |
| clear | 20 s camera blackout | 3.69 | 17.4 | |
| clear | 20 s camera blackout + DVL | **0.32** | 2.90 | the DVL holds the blackout at the no-blackout level |
| turbid | vision | 157 (diverged) | 107 | vision fails in turbid water |
| turbid | + DVL | **4.23** | 18.0 | the DVL keeps the estimate bounded |

The trajectory is about 100 m long. ATE uses an SE(3) alignment of the estimate to the ground-truth left camera,
and the Sim(3) scale is 0.995–0.998 for every run that did not diverge. Each config ran once, so these are single runs,
not means. The 20 s blackout starts 195 s after the first image, in the middle of leg 2.

## Run

```
sim/run_sim_ws1.sh both                      # record + convert (CAM_RATE=10 recommended; KEEP_RAW=1 keeps the raw bag)
RUNS=<out dir> sim/bench_sim.sh              # run_bag on all configs + evaluate.py -> results/results.tsv
python3 sim/diag.py --bench <run>/bench      # independent IMU/camera consistency checks against the ground truth
python3 -m pytest sim/tests                  # laptop unit tests (the ROS bag test needs stonefish_ros2)
```

`convert.py` prints a sanity report and flags a run if any of these happen:

- the vehicle is not still when the first stereo pair arrives;
- there are stereo gaps over 1 s;
- frames fail to pair;
- the pressure slope is off;
- the DVL frame is wrong;
- bottom lock is never lost.

## Bugs found on the way (each now has a check or a test)

- **Stereo pairing tolerance.** Stonefish stamps every image with `now()` when it publishes it, and the right image trails the left by about 24 ms (the left publish callback's cost). A fixed 25 ms pairing tolerance silently dropped about 30 % of the frames in long runs. The dropped frames left a stereo gap of up to 63 s, and the first pair arrived late, after the vehicle had started moving. The estimator initialises from rest, so it blew up at start-up (vision-only ATE 21–49 m). The tolerance is now 0.45 of the camera period.
- **Start in motion.** The 4 s hold was shorter than the first-pair delay, so the hold is now 40 s and the converter flags any run where it is not long enough.

## Limits

- The visuals are rasterised by Stonefish: no refraction model, no lens distortion, no rolling shutter.
- The Stonefish DVL velocity is derived from the vehicle's motion, not from a Doppler model.
- The IMU is synthesised from the analytic trajectory (a Kalibr noise model), not from Stonefish's own IMU.
- Not yet investigated: why vision-only diverges in turbid water.
