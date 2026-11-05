"""pytest for sim_dvl.py on a synthetic circle with a mocap dropout."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import sim_dvl
from sim_dvl import SimConfig, simulate

RADIUS, OMEGA, HZ = 1.5, 0.4, 100.0
GAP = (10.0, 13.0)  # mocap dropout [s]


def circle_mocap() -> np.ndarray:
    t = np.arange(0.0, 30.0, 1.0 / HZ)
    t = t[(t < GAP[0]) | (t >= GAP[1])]
    p = np.stack([RADIUS * np.cos(OMEGA * t), RADIUS * np.sin(OMEGA * t), -0.5 + 0.1 * np.sin(0.3 * t)], axis=1)
    q = Rotation.from_euler("z", OMEGA * t + np.pi / 2).as_quat()  # heading along the tangent
    return np.column_stack([t, p, q])


def analytic_v_D(t: np.ndarray, T_M_I: np.ndarray, T_I_D: np.ndarray) -> np.ndarray:
    """DVL-frame velocity of the circle, computed independently of sim_dvl's smoothing."""
    v_W = np.stack([-RADIUS * OMEGA * np.sin(OMEGA * t), RADIUS * OMEGA * np.cos(OMEGA * t), 0.03 * np.cos(0.3 * t)], axis=1)
    R_WM = Rotation.from_euler("z", OMEGA * t + np.pi / 2)
    R_WI = R_WM * Rotation.from_matrix(T_M_I[:3, :3])
    # the I origin sits at p_M + R_WM t_MI: its velocity adds w x (R_WM t_MI), w = OMEGA about world z
    v_I_W = v_W + np.cross([0, 0, OMEGA], R_WM.apply(T_M_I[:3, 3]))
    w_I = R_WI.inv().apply(np.tile([0, 0, OMEGA], (len(t), 1)))
    v_I = R_WI.inv().apply(v_I_W) + np.cross(w_I, T_I_D[:3, 3])
    return v_I @ T_I_D[:3, :3]


T_M_I = np.eye(4)
T_M_I[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.05, 0.3]).as_matrix()
T_M_I[:3, 3] = [0.1, -0.05, 0.02]
T_I_D = sim_dvl.mount_from_down(np.array([0.0, 0.0, -1.0]), 0.15, 45.0)
T_I_D[:3, 3] += [0.3, 0.1, 0.0]  # sideways lever so the w x l term is exercised (0.4 rad/s * 0.3 m)


def test_recovered_velocity_matches_analytic_within_noise():
    cfg = SimConfig(scale_bias=0.0, seed=3)
    rows, stats = simulate(circle_mocap(), T_M_I, T_I_D, cfg)
    t, v, valid = rows[:, 0], rows[:, 1:4], rows[:, 4]
    assert stats["samples"] > 150 and np.all(valid == 1)
    err = v - analytic_v_D(t, T_M_I, T_I_D)
    sigma = np.sqrt(cfg.sigma_abs_mps ** 2 + (cfg.sigma_rel * 0.6) ** 2)
    assert abs(err.mean()) < 0.5 * sigma                 # smoothing adds no bias
    no_lever = T_I_D.copy()
    no_lever[:3, 3] = 0.0
    assert np.sqrt(((v - analytic_v_D(t, T_M_I, no_lever)) ** 2).mean()) > 3 * sigma  # the test sees the lever
    assert np.sqrt((err ** 2).mean()) < 1.3 * sigma      # residual is the injected noise


def test_gap_and_edges_produce_no_samples():
    cfg = SimConfig(seed=1)
    rows, stats = simulate(circle_mocap(), T_M_I, T_I_D, cfg)
    t = rows[:, 0]
    assert stats["segments"] == 2
    lo, hi = GAP[0] - cfg.edge_margin_s, GAP[1] + cfg.edge_margin_s
    assert not np.any((t > lo) & (t < hi))
    assert t.min() >= cfg.edge_margin_s - 1e-6 and np.all(np.diff(t) > 0)


def test_scenarios_dropouts_and_outliers():
    mocap = circle_mocap()
    base, _ = simulate(mocap, T_M_I, T_I_D, SimConfig(scenario="none", seed=5))
    assert np.all(base[:, 4] == 1)
    per, st = simulate(mocap, T_M_I, T_I_D, SimConfig(scenario="periodic", seed=5))
    assert st["invalid"] > 0 and np.all(per[per[:, 4] == 0][:, 1:4] == 0)
    out, st = simulate(mocap, T_M_I, T_I_D, SimConfig(scenario="outliers", seed=5))
    assert np.all(out[:, 4] == 1) and (st["spikes"] > 0 or st["lock_samples"] > 0)
    lock = np.linalg.norm(out[:, 1:4], axis=1)
    assert np.sum(lock < 0.05) == st["lock_samples"] > 0   # water-column lock: ~0 m/s with valid=1
    a, _ = simulate(mocap, T_M_I, T_I_D, SimConfig(scenario="bursts", seed=7))
    b, _ = simulate(mocap, T_M_I, T_I_D, SimConfig(scenario="bursts", seed=7))
    assert np.array_equal(a, b)                             # seeded


def test_extrinsic_sidecars(tmp_path):
    rows, _ = simulate(circle_mocap(), T_M_I, T_I_D, SimConfig(seed=2))
    js, ym = sim_dvl.write_outputs(tmp_path / "d.csv", rows, T_I_D, {})
    import json
    import yaml
    assert np.allclose(json.loads(js.read_text())["T_B_D"], T_I_D)
    assert np.allclose(yaml.safe_load(ym.read_text())["T_B_D"], T_I_D)
    assert (tmp_path / "d.csv").read_text().startswith("t,vx,vy,vz,valid\n")
