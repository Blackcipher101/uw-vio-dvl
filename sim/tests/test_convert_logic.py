"""convert.py core logic on synthetic data (no ROS needed)."""
import math

import numpy as np

import convert
import rig
import trajectory as tj


def _synthetic_traj_state(traj, t0, seconds, rtf=0.98, jitter_s=2e-4, drop=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / rig.SIM_RATE_HZ
    n = int(seconds / dt)
    sim = t0 + np.arange(n) * dt
    keep = rng.random(n) >= drop
    p, R, _, _ = traj.state_at(sim)
    q = np.stack([rig.quat_xyzw_from_rot(r) for r in R])
    wall = 7000.0 + (sim - t0) / rtf + jitter_s * rng.standard_normal(n)
    return sim[keep], wall[keep], p[keep], q[keep]


def test_time_map_recovers_sim_time(traj):
    sim, wall, p, q = _synthetic_traj_state(traj, t0=0.005, seconds=120.0)
    tm = convert.fit_time_map(wall, p, q, traj, rig.SIM_RATE_HZ)
    assert tm.info["mode"] == "regular"
    assert np.abs(tm.sim - sim).max() < 1.5e-3
    assert math.isclose(tm.info["realtime_factor"], 0.98, rel_tol=0.01)


def test_time_map_survives_dropped_messages(traj):
    sim, wall, p, q = _synthetic_traj_state(traj, t0=0.005, seconds=120.0, drop=0.05)
    tm = convert.fit_time_map(wall, p, q, traj, rig.SIM_RATE_HZ)
    assert tm.info["mode"] == "interpolated"
    assert np.abs(tm.sim - sim).max() < 6e-3


def test_scalar_and_image_time_mapping(traj):
    sim, wall, p, q = _synthetic_traj_state(traj, 0.005, 60.0)
    tm = convert.fit_time_map(wall, p, q, traj, rig.SIM_RATE_HZ)
    w = np.array([wall[1000] + 1e-4, wall[2000] - 5e-4])  # same step, published slightly before/after the pose message
    s, lag = convert.scalar_sim_time(tm, w)
    np.testing.assert_allclose(s, sim[[1000, 2000]], atol=1.5e-3)
    img = convert.image_sim_time(tm, np.array([wall[3000] + 0.02]), latency_s=0.02)
    assert abs(img[0] - sim[3000]) < 1.5e-3


def test_pair_images_tolerates_drops_and_skew():
    l = np.arange(0, 10, 0.05)
    r = l + 0.008
    r = np.delete(r, [10, 11, 50])
    l = np.delete(l, [20])
    pairs = convert.pair_images(l, r, convert.PAIR_TOL_PERIOD_FRACTION * 0.05)
    assert len(pairs) == len(l) - 3  # left frames 10, 11 and 50 lost their partner
    assert all(abs(l[i] - r[j]) < 0.025 for i, j in pairs)


def test_pair_images_right_published_after_left():
    # Regression: Stonefish stamps the right image ~22-27 ms after the left (sequential publish callbacks). At 10 Hz a
    # fixed 25 ms tolerance lost every frame whose lag exceeded 25 ms; the period-relative tolerance pairs all of them.
    rng = np.random.default_rng(3)
    l = np.arange(0, 60, 0.1)
    r = l + rng.uniform(0.022, 0.0275, len(l))
    assert len(convert.pair_images(l, r, 0.025)) < 0.8 * len(l)  # the old behaviour
    pairs = convert.pair_images(l, r, convert.PAIR_TOL_PERIOD_FRACTION * 0.1)
    assert pairs == [(i, i) for i in range(len(l))]
    # a missing right frame must not be paired with the next one (0.1 s later)
    r2 = np.delete(r, 100)
    p2 = convert.pair_images(l, r2, convert.PAIR_TOL_PERIOD_FRACTION * 0.1)
    assert len(p2) == len(l) - 1 and all(abs(l[i] - r2[j]) < 0.03 for i, j in p2)


def test_camera_time_snapping():
    rng = np.random.default_rng(1)
    t = 3.0 + 0.05 * np.arange(400) + 0.001 * rng.standard_normal(400)
    snapped, info = convert.assign_camera_times(t, "auto")
    assert info["mode"] == "snap" and info["rms_s"] < 0.002
    assert np.allclose(np.diff(snapped), 0.05, atol=1e-4)
    noisy = t + 0.02 * rng.standard_normal(400)
    _, info2 = convert.assign_camera_times(noisy, "auto")
    assert info2["mode"] == "mapped"
    t_gap = np.delete(t, [100, 101])
    s2, info3 = convert.assign_camera_times(t_gap, "snap")
    assert info3["dropped_frames"] == 2 and np.allclose(np.diff(s2)[98:101], [0.05, 0.15, 0.05], atol=1e-3)


def test_dvl_validity_follows_altitude_flag():
    alt = np.array([3.0, -1.0, 2.0, np.nan])
    vel = np.zeros((4, 3))
    vel[3, 0] = np.nan
    np.testing.assert_array_equal(convert.dvl_validity(alt, vel), [True, False, True, False])


def test_dvl_expected_velocity_matches_finite_difference(traj):
    """v_D = R_BD^T (R_WB^T v_W + w x l) is the velocity of the DVL point seen from the DVL frame."""
    r = rig.build_rig()
    t = np.array([60.0, 130.0, 200.0])
    exp = convert.dvl_expected_velocity(traj, t, r)
    h = 1e-3
    pa, Ra, _, _ = traj.state_at(t - h)
    pb, Rb, _, _ = traj.state_at(t + h)
    p, R, _, _ = traj.state_at(t)
    pd_a = pa + Ra @ r.T_B_D[:3, 3]
    pd_b = pb + Rb @ r.T_B_D[:3, 3]
    v_world = (pd_b - pd_a) / (2 * h)
    v_body = np.einsum("nji,nj->ni", R, v_world)
    np.testing.assert_allclose(exp["body_frame"], v_body, atol=2e-4)
    np.testing.assert_allclose(exp["dvl_frame"], v_body @ r.T_B_D[:3, :3], atol=2e-4)
    np.testing.assert_allclose(exp["world_ned"], v_world, atol=2e-4)


def test_frame_hypothesis_is_identifiable(traj):
    t = np.arange(20.0, 200.0, 0.2)
    exp = convert.dvl_expected_velocity(traj, t)
    meas = exp["dvl_frame"]
    res = {k: np.sqrt(np.mean((meas - v) ** 2)) for k, v in exp.items()}
    assert min(res, key=res.get) == "dvl_frame" and res["dvl_frame"] < 1e-9
    assert res["body_frame"] > 0.1 and res["world_ned"] > 0.1


def test_circular_phase():
    t = 0.013 + 0.05 * np.arange(200)
    ph, conc = convert.circular_phase(t, 0.05)
    assert abs(ph - 0.013) < 1e-6 and conc > 0.999
