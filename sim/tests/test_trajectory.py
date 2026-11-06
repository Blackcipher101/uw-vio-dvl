"""The scripted trajectory meets the benchmark spec, and the synthetic IMU is consistent with it."""
import json
import math

import numpy as np

import convert
import rig
import trajectory as tj

from conftest import SIM


def test_spec(traj):
    assert 270.0 <= traj.duration_s <= 330.0
    move = (traj.t > tj.HOLD_START_S + 2) & (traj.t < traj.duration_s - tj.HOLD_END_S - 1)
    speed = np.linalg.norm(traj.v_W, axis=1)[move]
    assert 0.2 <= speed.min() and speed.max() <= 0.5
    assert np.linalg.norm(traj.a_W, axis=1).max() < 0.3
    assert np.linalg.norm(traj.w_B, axis=1).max() < 0.5
    assert traj.p[:, 2].min() > 2.0  # well below the surface
    yaw = traj.euler[:, 2]
    assert yaw.max() - yaw.min() > math.radians(300)  # yaw changes: sway, half turn, orbit


def test_altitude_and_single_lock_loss(traj):
    ts = np.arange(0.0, traj.duration_s, 1.0 / rig.DVL_RATE_HZ)
    valid, _, vertical = tj.dvl_expected(traj, ts)
    intervals = tj.invalid_intervals(ts, valid)
    assert len(intervals) == 1
    assert 15.0 <= intervals[0][1] - intervals[0][0] <= 22.0
    t_move = ts - tj.HOLD_START_S  # time since the vehicle started moving
    outside = (t_move < 26.0) | (t_move > 86.0)  # not over the pit (leg 1 is above it ~29-83 s after the start)
    assert 1.8 <= vertical[outside].min() and vertical[outside].max() <= 4.5
    # the other documented-beam-pattern hypothesis must also lose lock for 15+ s (possibly split in two)
    v2, _, _ = tj.dvl_expected(traj, ts, azimuths_deg=(0.0, 90.0, 180.0, 270.0))
    assert (~v2).sum() / rig.DVL_RATE_HZ >= 15.0


def test_keypoints_reproduce_analytic_motion(traj):
    t, p, e = traj.keypoints()
    mid = np.arange(0.01, traj.duration_s - 0.05, 0.37)
    i = np.minimum((mid / 0.02).astype(int), len(t) - 2)
    f = ((mid - t[i]) / 0.02)[:, None]
    p_pwl = p[i] * (1 - f) + p[i + 1] * f
    p_true, R_true, _, _ = traj.state_at(mid)
    assert np.abs(p_pwl - p_true).max() < 1e-3
    # unwrapped yaw: raw (sign-unfixed) quaternions of consecutive keypoints never flip sign
    raw = np.stack([_raw_quat(*row) for row in e[::10]])
    assert np.all(np.einsum("ij,ij->i", raw[1:], raw[:-1]) > 0)
    for row in e[::500]:  # the hand-written half-angle quaternion is the rotation matrix we expect
        np.testing.assert_allclose(rig.rot_from_quat_xyzw(_raw_quat(*row)), rig.rot_from_rpy(*row), atol=1e-12)


def _raw_quat(roll, pitch, yaw):
    """x, y, z, w of Rz(yaw) Ry(pitch) Rx(roll) built from half angles (what a parser does), no sign fixing."""
    cr, sr, cp, sp, cy, sy = (fn(a / 2) for a, fn in ((roll, math.cos), (roll, math.sin), (pitch, math.cos),
                                                      (pitch, math.sin), (yaw, math.cos), (yaw, math.sin)))
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


def test_meta_matches_generator(traj):
    meta = json.loads((SIM / "scenarios" / "trajectory_meta.json").read_text())
    assert math.isclose(meta["duration_s"], traj.duration_s, abs_tol=1e-3)
    import hashlib

    t, p, e = traj.keypoints()
    assert hashlib.sha256(np.round(np.column_stack([t, p, e]), 6).tobytes()).hexdigest() == meta["keypoint_sha256"]


def test_imu_noise_free_integration_reproduces_trajectory(traj):
    """Integrating the noise-free synthetic IMU (specific force + rate) recovers the analytic pose: checks the sign
    conventions of gravity (NED, +z down), body-frame rate and specific force."""
    imu = convert.synth_imu(traj, noise=False)
    dt = 1.0 / rig.IMU_RATE_HZ
    n = int(40.0 * rig.IMU_RATE_HZ)
    p0, R0, v0, _ = traj.state_at(np.array([0.0]))
    R, v, p = R0[0].copy(), v0[0].copy(), p0[0].copy()
    g = np.array([0.0, 0.0, rig.GRAVITY_MPS2])
    for k in range(n):
        w = 0.5 * (imu["gyro"][k] + imu["gyro"][k + 1]) * dt
        th = np.linalg.norm(w)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        dR = np.eye(3) + K + K @ K * 0.5 if th < 1e-9 else (
            np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * K @ K)
        a = R @ (0.5 * (imu["acc"][k] + imu["acc"][k + 1])) + g
        p += v * dt + 0.5 * a * dt * dt
        v += a * dt
        R = R @ dR
    p_ref, R_ref, v_ref, _ = traj.state_at(np.array([n * dt]))
    assert np.linalg.norm(p - p_ref[0]) < 0.02
    assert np.linalg.norm(v - v_ref[0]) < 0.01
    assert math.acos(np.clip((np.trace(R.T @ R_ref[0]) - 1) / 2, -1, 1)) < 0.005


def test_imu_static_specific_force_points_up(traj):
    imu = convert.synth_imu(traj, noise=False)
    first = imu["acc"][: int(2 * rig.IMU_RATE_HZ)].mean(axis=0)
    np.testing.assert_allclose(first, [0.0, 0.0, -rig.GRAVITY_MPS2], atol=0.01)  # FRD level: f = -g (z down => -9.81)


def test_imu_noise_matches_kalibr_parameters(traj):
    imu = convert.synth_imu(traj, seed=3)
    clean = convert.synth_imu(traj, noise=False)
    dt = 1.0 / rig.IMU_RATE_HZ
    for key, nd in (("gyro", rig.GYR_NOISE_DENSITY), ("acc", rig.ACC_NOISE_DENSITY)):
        d = np.diff(imu[key] - clean[key], axis=0)  # differencing removes the (slow) bias, leaves white noise * sqrt(2)
        std = d.std(axis=0) / math.sqrt(2.0)
        np.testing.assert_allclose(std, nd / math.sqrt(dt), rtol=0.05)
