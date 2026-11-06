# Results, figures and simulation (details)

[Back to README](../README.md)

`uw_ct_vio` is a GTSAM fixed-lag (iSAM2) estimator for underwater vehicles: IMU preintegration, stereo
projection factors from a photometrically robust KLT front end (auto local-contrast normalisation for dark /
turbid images), a continuous-time pressure-depth factor, and a **DVL body-velocity factor** with validity,
innovation-gate, re-lock and recovery logic so DVL dropouts (bottom-lock loss, water-column lock, spikes) do
not corrupt the estimate. Offline runner `run_bag` replays ROS 2 bags. Research code; never run on a vehicle.

![Top-down trajectory on traj_03: mocap ground truth, 20 s blackout without DVL and with simulated DVL](figures/hero_trajectory.png)

*traj_03 (NTNU underwater 2023, subset-3), x-y after hand-eye + SE3 alignment as in `scripts/evaluate.py`; thick
segments mark the 20 s camera blackout. The DVL is simulated from mocap, so this shows the factor and gating, not real-sensor accuracy.*

## Results (NTNU underwater dataset 2023, 6 trajectories with mocap; mean ATE SE3 with hand-eye, RPE over 5 m)

| Configuration | Mean ATE (m) | Mean RPE (%) | Worst traj (m) |
|---|---|---|---|
| VIO + pressure, cameras normal (base) | 0.401 | 6.75 | 0.642 |
| + DVL (sim), no dropouts | 0.393 | 6.71 | 0.646 |
| + DVL (sim), periodic / long / burst dropouts, outliers | 0.393-0.400 | 6.71 | 0.658 |
| 20 s camera blackout, **no DVL** | 3.876 | 33.81 | 10.711 |
| 20 s camera blackout, **DVL (sim)** | **0.663** | 7.31 | 0.936 |
| 20 s camera blackout, DVL also lost for 7 s inside it | 0.888 | 9.39 | 1.527 |

**The DVL is simulated** (`scripts/sim_dvl.py`, from mocap, with noise, scale bias, a ~2.5 deg mount error from
the hand-eye calibration, dropouts and outliers). These numbers show the factor, gate and dropout handling
work; real-sensor accuracy needs a real DVL dataset (e.g. the Heriot-Watt Tank dataset). Baselines on the same
bags: Basalt 1.10 m (in-air stereo extrinsic), SVIn 1.80 m, ORB-SLAM3 initialises on 3/6, OpenVINS 3/6.
Details, design, as-built notes and the full table: `DESIGN_DVL.md`. Estimator history and ablations:
`DESIGN_V2.md`, `DESIGN_PHASE2.md`.

### Figures

![Position error vs time on traj_03](figures/error_vs_time.png)

*Position error vs time on traj_03 (log scale). Errors are after the same global SE3 alignment that gives the reported ATE, so a
late blackout drift also shows up as offset before it; the DVL-lost-for-7 s run is the purple curve. Simulated DVL.*

![ATE per trajectory](figures/ate_blackout_bars.png)

*ATE per trajectory (log axis): baseline, 20 s blackout without DVL, and with simulated DVL. Values from the table above (`ntnu_all.tsv` of the benchmark runs). On traj_01 and traj_04 the DVL gives little or no gain.*

![DVL dropout robustness](figures/dvl_dropout_robustness.png)

*Mean ATE per simulated DVL scenario with cameras normal; dots are the 6 trajectories. The means are not above DVL-off, but
single trajectories can be slightly worse (worst 0.658 m vs 0.642 m).*

![Simulated DVL example](figures/simulated_dvl_example.png)

*Example simulated DVL velocity (body frame, traj_02), periodic dropouts and outliers; samples exist only where mocap does. Outlier spikes are clipped to +-1.2 m/s.*

Regenerate all figures (needs matplotlib, numpy and the run outputs; paths are for our machine):
```bash
python3 scripts/make_figures.py --data <uw_figdata> --mocap-root <results>/ws1_sync/mocap
```

## Simulation

`sim/` holds a Stonefish seabed benchmark with a physically simulated DVL (natural bottom-lock loss over a pit) and exact
ground truth. On the clear-water run (single runs), a 20 s camera blackout raises ATE from 0.30 m to 3.69 m without the DVL; with the DVL it is **0.32 m**.
In turbid water vision-only diverges and the DVL keeps it at 4.23 m. Details and caveats are in [`sim/README.md`](../sim/README.md).

![Stonefish camera blackout (animated)](media/blackout_clear.gif)

*Video ([mp4](media/blackout_clear.mp4)), with the estimates aligned on the pre-blackout segment only: during the 20 s blackout the vision-only error grows to 8.8 m, while with the DVL it stays under 0.5 m. Rendered by `sim/make_video.py`.*
