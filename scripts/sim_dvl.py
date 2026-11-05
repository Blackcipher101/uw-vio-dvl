#!/usr/bin/env python3
"""Simulate a DVL from motion-capture ground truth (sim-DVL: proves plumbing and dropout behaviour,
not real-sensor accuracy).

Inputs
  mocap TUM          t x y z qx qy qz qw of the mocap body M (T_world_M), world z-up, bag clock.
  T_M_C              left camera in M, i.e. the hand-eye X = T_body_cam of evaluate.py. Either
                     --T-M-C <file> (4x4 as JSON list / {"T_M_C" | "X": 4x4} / whitespace text) or
                     --hand-eye-est <est.tum> (a reference run; evaluate.hand_eye is run on it).
                     evaluate.py --json only keeps the hand-eye translation and rotation *angle*, not
                     R, so it cannot be used directly.
  Kalibr imucam yaml T_cam_imu of the left camera (NTNU: the left camera is cam1, pass --cam cam1).
  Mount              T_I_D: DVL --mount-height-m (0.15) below the IMU, yawed --mount-yaw-deg (45) about
                     the "down" axis. Down is an IMU axis chosen from the mocap start pose: -z_world
                     expressed in I, snapped to the nearest +-x/+-y/+-z (--down-axis overrides).
                     Check it against the mean accelerometer vector of a still stretch of the bag
                     (specific force points UP, so down = -f/|f|).

Model   v_D = R_ID^T (R_WI^T v_W + w_I x l_I) at the DVL origin, then
        v_meas = (1 + scale_bias) * R_mis * v_D + N(0, sigma_abs^2 + (rel |v_D|)^2) per axis.

Mocap drops out for seconds (see evaluate.py): the stream is split into contiguous segments at gaps
> --gap-s, v_W and w are smoothed (Savitzky-Golay) and differentiated per segment, and no sample is
emitted in gaps or within --edge-margin-s of a segment edge (truth unknown / filter edge effects).
Scenario dropouts emit valid=0 rows with zero velocity (bottom lock lost).

Outputs (for run_bag): <out> CSV `t,vx,vy,vz,valid`; <out stem>.json and <out stem>.yaml with T_B_D
(body = IMU) for --dvl-extrinsic; the JSON also records the generator settings.

Scenarios: none | periodic (5 s off per 30 s) | long (one 20 s gap) | bursts (random 1-4 s losses,
~1 per 40 s) | outliers (2 % spikes of 1-3 m/s plus one 3 s water-column lock at ~0 m/s, valid=1).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

SCENARIOS = ("none", "periodic", "long", "bursts", "outliers")
AXES = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}


@dataclass(frozen=True)
class SimConfig:
    rate_hz: float = 8.0
    jitter_frac: float = 0.05          # timing jitter sigma as a fraction of the period
    sigma_abs_mps: float = 0.01
    sigma_rel: float = 0.01
    scale_bias: float = 0.003          # constant scale error amplitude, sign drawn from the seed
    mis_mount_deg: float = 0.0
    gap_s: float = 0.2                 # mocap stamp gap that starts a new segment
    edge_margin_s: float = 0.3         # no samples this close to a segment edge
    smooth_window_s: float = 0.25
    scenario: str = "none"
    seed: int = 0


def load_tum(path: Path) -> np.ndarray:
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.ndim != 2 or a.shape[1] != 8 or len(a) < 2:
        raise ValueError(f"{path}: expected >= 2 rows of 8 columns (t x y z qx qy qz qw)")
    a = a[np.all(np.isfinite(a), axis=1)]
    return a[np.argsort(a[:, 0])]


def read_matrix(path: Path) -> np.ndarray:
    """4x4 matrix from a JSON list, a JSON dict with T_M_C / X, or whitespace-separated text."""
    text = path.read_text()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("T_M_C", obj.get("X"))
            if obj is None:
                raise ValueError(f"{path}: JSON has no T_M_C or X key (evaluate.py --json has no rotation)")
        m = np.asarray(obj, dtype=float)
    except json.JSONDecodeError:
        m = np.loadtxt(path)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"{path}: expected a finite 4x4 matrix, got shape {m.shape}")
    return m


def load_T_cam_imu(yaml_path: Path, cam: str) -> np.ndarray:
    import yaml  # local import: only this loader needs it

    node = yaml.safe_load(yaml_path.read_text())
    if cam not in node:
        raise ValueError(f"{yaml_path}: no camera '{cam}' (found {sorted(node)})")
    m = np.asarray(node[cam]["T_cam_imu"], dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"{yaml_path}: {cam}.T_cam_imu is not 4x4")
    return m


def inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def snap_axis(v: np.ndarray) -> np.ndarray:
    i = int(np.argmax(np.abs(v)))
    out = np.zeros(3)
    out[i] = np.sign(v[i])
    return out


def mount_from_down(down_I: np.ndarray, height_m: float, yaw_deg: float) -> np.ndarray:
    """T_I_D: origin height_m along down_I, rotated yaw_deg about down_I."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(down_I, float) * np.radians(yaw_deg)).as_matrix()
    T[:3, 3] = np.asarray(down_I, float) * height_m
    return T


def split_segments(t: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Index ranges [i0, i1) of contiguous mocap data (consecutive stamps closer than gap_s)."""
    cuts = np.nonzero(np.diff(t) > gap_s)[0] + 1
    edges = np.concatenate([[0], cuts, [len(t)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]


def smooth_kinematics(t: np.ndarray, p: np.ndarray, R: Rotation, window_s: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World velocity of p and body-frame angular rate of R on one contiguous segment, on its own time
    grid: returns (grid_t, v_W, w_B). The stream is resampled to a uniform grid first."""
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    grid = grid[grid <= t[-1]]
    pg = np.stack([np.interp(grid, t, p[:, k]) for k in range(3)], axis=1)
    Rg = Slerp(t, R)(grid)
    win = max(5, int(round(window_s / dt)) | 1)  # odd, >= 5
    if win > len(grid):
        win = (len(grid) - 1) | 1
    if win < 5:
        raise ValueError("segment too short to smooth")
    order = 3
    v = savgol_filter(pg, win, order, deriv=1, delta=dt, axis=0)
    rel = Rg[:-1].inv() * Rg[1:]
    w = rel.as_rotvec() / dt                      # body-frame rate between consecutive samples
    w = np.vstack([w, w[-1:]])
    w = savgol_filter(w, win, 2, axis=0)
    return grid, v, w


def dropout_masks(times: np.ndarray, cfg: SimConfig, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lost, spike, lock) boolean masks over the sample times for the scenario."""
    n = len(times)
    lost, spike, lock = (np.zeros(n, bool) for _ in range(3))
    t0, span = times[0], times[-1] - times[0]
    if cfg.scenario == "periodic":
        lost = ((times - t0) % 30.0) >= 25.0
    elif cfg.scenario == "long":
        start = t0 + 0.4 * span
        lost = (times >= start) & (times < start + 20.0)
    elif cfg.scenario == "bursts":
        t = t0 + rng.exponential(40.0)
        while t < times[-1]:
            dur = rng.uniform(1.0, 4.0)
            lost |= (times >= t) & (times < t + dur)
            t += dur + rng.exponential(40.0)
    elif cfg.scenario == "outliers":
        spike = rng.random(n) < 0.02
        start = t0 + 0.5 * span
        lock = (times >= start) & (times < start + 3.0)
    return lost, spike, lock


def simulate(mocap: np.ndarray, T_M_I: np.ndarray, T_I_D: np.ndarray, cfg: SimConfig
             ) -> tuple[np.ndarray, dict]:
    """Rows (t, vx, vy, vz, valid) in the DVL frame and a stats dict."""
    if cfg.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {cfg.scenario}; choose from {SCENARIOS}")
    if not (cfg.rate_hz > 0 and cfg.gap_s > 0 and cfg.edge_margin_s >= 0 and cfg.smooth_window_s > 0):
        raise ValueError("rate_hz, gap_s, smooth_window_s must be > 0 and edge_margin_s >= 0")
    rng = np.random.default_rng(cfg.seed)
    R_MI, t_MI = T_M_I[:3, :3], T_M_I[:3, 3]
    R_ID, l_I = T_I_D[:3, :3], T_I_D[:3, 3]
    scale = 1.0 + cfg.scale_bias * rng.choice([-1.0, 1.0])
    R_mis = np.eye(3)
    if cfg.mis_mount_deg > 0:
        axis = rng.normal(size=3)
        R_mis = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(cfg.mis_mount_deg)).as_matrix()

    t_all = mocap[:, 0]
    period = 1.0 / cfg.rate_hz
    times, v_true = [], []
    n_segments = 0
    for i0, i1 in split_segments(t_all, cfg.gap_s):
        seg = mocap[i0:i1]
        R_WM = Rotation.from_quat(seg[:, 4:8])
        p_I = seg[:, 1:4] + R_WM.apply(t_MI)
        R_WI = R_WM * Rotation.from_matrix(R_MI)
        try:
            grid, v_W, w_I = smooth_kinematics(seg[:, 0], p_I, R_WI, cfg.smooth_window_s)
        except ValueError:
            continue
        lo, hi = grid[0] + cfg.edge_margin_s, grid[-1] - cfg.edge_margin_s
        if hi <= lo:
            continue
        n_segments += 1
        ts = lo + np.arange(0.0, hi - lo, period)
        ts = ts + rng.normal(0.0, cfg.jitter_frac * period, len(ts))
        ts = np.sort(ts[(ts >= lo) & (ts <= hi)])
        if len(ts) == 0:  # segment shorter than one DVL period after the margins
            continue
        Rg =Slerp(seg[:, 0], R_WI)(np.clip(ts, seg[0, 0], seg[-1, 0]))
        vW = np.stack([np.interp(ts, grid, v_W[:, k]) for k in range(3)], axis=1)
        wI = np.stack([np.interp(ts, grid, w_I[:, k]) for k in range(3)], axis=1)
        v_I = Rg.inv().apply(vW) + np.cross(wI, l_I)
        times.append(ts)
        v_true.append(v_I @ R_ID)                      # R_ID^T v_I, row-vector form
    if not times:
        raise ValueError("no contiguous mocap segment is long enough for the smoothing window plus margins")
    t = np.concatenate(times)
    v_D = np.concatenate(v_true)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, v_D = t[keep], v_D[keep]

    sigma = np.sqrt(cfg.sigma_abs_mps ** 2 + (cfg.sigma_rel * np.abs(v_D)) ** 2)
    v = scale * (v_D @ R_mis.T) + rng.normal(size=v_D.shape) * sigma
    valid = np.ones(len(t))
    lost, spike, lock = dropout_masks(t, cfg, rng)
    for k in np.nonzero(spike)[0]:
        d = rng.normal(size=3)
        v[k] = d / np.linalg.norm(d) * rng.uniform(1.0, 3.0)
    v[lock] = rng.normal(size=(int(lock.sum()), 3)) * 0.005
    v[lost] = 0.0
    valid[lost] = 0.0
    rows = np.column_stack([t, v, valid])
    stats = {"samples": int(len(t)), "invalid": int(lost.sum()), "spikes": int(spike.sum()),
             "lock_samples": int(lock.sum()), "segments": n_segments, "scale": float(scale)}
    return rows, stats


def write_outputs(out: Path, rows: np.ndarray, T_B_D: np.ndarray, meta: dict) -> tuple[Path, Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("t,vx,vy,vz,valid\n")
        for r in rows:
            f.write(f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{int(r[4])}\n")
    js, ym = out.with_suffix(".json"), out.with_suffix(".yaml")
    js.write_text(json.dumps({"T_B_D": T_B_D.tolist(), **meta}, indent=1))
    ym.write_text("T_B_D:\n" + "".join("  - [" + ", ".join(f"{x:.9g}" for x in row) + "]\n" for row in T_B_D))
    return js, ym


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mocap", type=Path)
    ap.add_argument("--T-M-C", type=Path, dest="T_M_C")
    ap.add_argument("--hand-eye-est", type=Path, help="reference estimate TUM; hand-eye X computed with evaluate.py")
    ap.add_argument("--hand-eye-first-s", type=float, default=60.0)
    ap.add_argument("--imucam", type=Path, required=True, help="Kalibr camchain-imucam yaml")
    ap.add_argument("--cam", default="cam1", help="left camera key in --imucam (NTNU: cam1)")
    ap.add_argument("--scenario", choices=SCENARIOS, default="none")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-hz", type=float, default=SimConfig.rate_hz)
    ap.add_argument("--sigma-abs", type=float, default=SimConfig.sigma_abs_mps)
    ap.add_argument("--sigma-rel", type=float, default=SimConfig.sigma_rel)
    ap.add_argument("--scale-bias", type=float, default=SimConfig.scale_bias)
    ap.add_argument("--mis-mount-deg", type=float, default=SimConfig.mis_mount_deg)
    ap.add_argument("--gap-s", type=float, default=SimConfig.gap_s)
    ap.add_argument("--edge-margin-s", type=float, default=SimConfig.edge_margin_s)
    ap.add_argument("--down-axis", choices=["auto", *AXES], default="auto")
    ap.add_argument("--mount-height-m", type=float, default=0.15)
    ap.add_argument("--mount-yaw-deg", type=float, default=45.0)
    a = ap.parse_args()

    mocap = load_tum(a.mocap)
    if (a.T_M_C is None) == (a.hand_eye_est is None):
        ap.error("give exactly one of --T-M-C and --hand-eye-est")
    if a.T_M_C is not None:
        T_M_C = read_matrix(a.T_M_C)
    else:
        from evaluate import hand_eye
        T_M_C, info = hand_eye(mocap, load_tum(a.hand_eye_est), a.hand_eye_first_s)
        print("hand-eye", info)
    T_M_I = T_M_C @ load_T_cam_imu(a.imucam, a.cam)   # T_C_I = Kalibr T_cam_imu
    if a.down_axis == "auto":
        R_WM0 = Rotation.from_quat(mocap[0, 4:8]).as_matrix()
        down_I = snap_axis(T_M_I[:3, :3].T @ R_WM0.T @ np.array([0.0, 0.0, -1.0]))
    else:
        down_I = np.array(AXES[a.down_axis], float)
    T_I_D = mount_from_down(down_I, a.mount_height_m, a.mount_yaw_deg)
    cfg = SimConfig(rate_hz=a.rate_hz, sigma_abs_mps=a.sigma_abs, sigma_rel=a.sigma_rel, scale_bias=a.scale_bias,
                    mis_mount_deg=a.mis_mount_deg, gap_s=a.gap_s, edge_margin_s=a.edge_margin_s,
                    scenario=a.scenario, seed=a.seed)
    rows, stats = simulate(mocap, T_M_I, T_I_D, cfg)
    js, ym = write_outputs(a.out, rows, T_I_D, {"scenario": a.scenario, "seed": a.seed, "down_axis_I": down_I.tolist(),
                                                "T_M_I": T_M_I.tolist(), "config": cfg.__dict__, "stats": stats})
    print(f"{a.out}: {stats}  down_axis_I={down_I.tolist()}  sidecars {js.name} {ym.name}")


if __name__ == "__main__":
    main()
