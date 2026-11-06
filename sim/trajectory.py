"""Seabed inspection scene and the scripted vehicle trajectory (analytic, smooth, repeatable).

The same functions drive the scenario/mesh generators, the synthetic IMU, the exact ground truth and the expected
DVL model used by convert.py as a cross-check, so there is exactly one definition of the world.

World W = Stonefish NED (x north, y east, z down, sea surface at z = 0). See rig.py for all frames.

Layout (metres): flat-ish shelf at ~10 m depth with relief; a rectangular pit (floor 27 m) at PIT_C that the
vehicle crosses on leg 1 (rectangular, so every beam is >6 m from a wall in its middle), so the seabed is further than the DVL's altitude_max from the DVL for ~15-20 s
(natural bottom-lock loss); a container + pipe "structure" at STRUCT_C that the vehicle orbits at the end.
Path: hold 40 s -> leg 1 (+x, y = -10, across the pit) -> half turn -> leg 2 (-x, y = -1) -> 0.75 orbit around the
structure facing it -> hold 3 s. Speed 0.35-0.45 m/s, yaw sway on the legs, small roll/pitch oscillation, altitude
2.2-3.8 m above the shelf (the vehicle holds depth over the pit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import rig

GRID_DT_S = 0.001  # resolution of the stored analytic trajectory (1 kHz)

SHELF_Z_M = 10.0
PIT_C = np.array([14.0, -14.75])
PIT_HX_M, PIT_HY_M = 9.0, 12.25  # half extents of the rectangular pit (walls at x 5..23, y -27..-2.5)
PIT_WALL_M = 1.4
PIT_FLOOR_Z_M = 27.0
STRUCT_C = np.array([-9.0, 6.0])
STRUCT_FLAT_R_M = 7.0
ORBIT_R_M = 7.0
ORBIT_FRACTION = 0.75
# Stonefish publishes the first camera frames only ~25 s of sim time after start (renderer warm-up); the estimator
# initialises from rest, so the vehicle must still be holding when the first stereo pair arrives (convert.py flags it).
HOLD_START_S = 40.0
HOLD_END_S = 3.0
LEG1_X0, LEG1_X1, LEG1_Y = -8.0, 26.0, -10.0
TURN_R_M = 4.5
LEG2_X1 = STRUCT_C[0]  # leg 2 is tangent to the orbit circle at (STRUCT_C x, STRUCT_C y - ORBIT_R)
SPEED_LEG_MPS, SPEED_SLOW_MPS, SPEED_ARC_MPS = 0.45, 0.35, 0.40
ALT_MEAN_M, ALT_AMP_M, ALT_PERIOD_S = 3.0, 0.8, 45.0
SWAY_AMP_RAD, SWAY_PERIOD_S = math.radians(20.0), 24.0
PITCH_AMP_RAD, PITCH_PERIOD_S = math.radians(3.0), 9.0
ROLL_AMP_RAD, ROLL_PERIOD_S = math.radians(4.0), 13.0
TERRAIN_X = (-34.0, 46.0)  # mesh extent (m)
TERRAIN_Y = (-36.0, 26.0)
TERRAIN_CELL_M = 0.5


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _relief_components():
    rng = np.random.default_rng(7)
    comps = []
    for lam_lo, lam_hi, amp_sum, n in ((6.0, 30.0, 0.35, 10), (1.5, 4.0, 0.08, 10)):
        for _ in range(n):
            lam = rng.uniform(lam_lo, lam_hi)
            th = rng.uniform(0, 2 * math.pi)
            comps.append((2 * math.pi / lam * math.cos(th), 2 * math.pi / lam * math.sin(th),
                          rng.uniform(0, 2 * math.pi), amp_sum / math.sqrt(n) * rng.uniform(0.5, 1.5)))
    return comps


_RELIEF = _relief_components()


def _pit_sdf(x, y):
    """Signed distance-like value to the pit rectangle (negative inside)."""
    return np.maximum(np.abs(x - PIT_C[0]) - PIT_HX_M, np.abs(y - PIT_C[1]) - PIT_HY_M)


def terrain_z(x, y):
    """Seabed depth z [m] (down positive) at world (x, y); arrays broadcast."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    relief = np.zeros(np.broadcast(x, y).shape)
    for kx, ky, ph, a in _RELIEF:
        relief += a * np.sin(kx * x + ky * y + ph)
    d_struct = np.hypot(x - STRUCT_C[0], y - STRUCT_C[1])
    relief *= 0.15 + 0.85 * _smoothstep((d_struct - STRUCT_FLAT_R_M) / 4.0)  # flat pad under the structure
    base = SHELF_Z_M + relief
    w = _smoothstep(-_pit_sdf(x, y) / PIT_WALL_M + 0.5)  # 1 inside the pit, 0 outside, smooth wall
    return base + (PIT_FLOOR_Z_M - base) * w


@dataclass(frozen=True)
class SceneObjects:
    container_center: np.ndarray  # world xyz of the box centre
    container_size: np.ndarray  # x, y, z extents
    pipe_p0: np.ndarray
    pipe_p1: np.ndarray
    pipe_radius_m: float
    rocks: tuple  # (x, y, radius, seed)


def scene_objects() -> SceneObjects:
    sx, sy = STRUCT_C
    size = np.array([6.0, 2.4, 1.6])
    ground = float(terrain_z(sx, sy)) + 0.15
    cc = np.array([sx, sy, ground - size[2] / 2.0])
    px = sx + 4.5
    ys = np.linspace(sy - 6.0, sy + 6.0, 25)
    pz = float(np.max(terrain_z(np.full_like(ys, px), ys))) - 0.30 + 0.12
    rng = np.random.default_rng(11)
    rocks = []
    while len(rocks) < 16:
        x = rng.uniform(-22.0, 34.0)
        y = rng.uniform(-14.0, 16.0)
        rad = rng.uniform(0.4, 1.2)
        if _pit_sdf(x, y) < rad + 1.0:
            continue
        if abs(x - sx) < size[0] / 2 + rad + 0.5 and abs(y - sy) < size[1] / 2 + rad + 0.5:
            continue
        if abs(x - px) < rad + 0.6 and abs(y - sy) < 6.5:
            continue
        rocks.append((float(x), float(y), float(rad), int(rng.integers(1 << 30))))
    return SceneObjects(cc, size, np.array([px, sy - 6.0, pz]), np.array([px, sy + 6.0, pz]), 0.30, tuple(rocks))


def _path(ds: float = 0.005):
    """Geometric path sampled every ds metres. Returns xy (N,2), tangent heading (N,), segment id (N,),
    orbit angle phi (N,) (0 outside the orbit), and the arc length at which each segment starts."""
    pts, seg, phi = [], [], []

    def add(xy, sid, ph=None):
        pts.append(xy)
        seg.append(np.full(len(xy), sid))
        phi.append(np.zeros(len(xy)) if ph is None else ph)

    n1 = int((LEG1_X1 - LEG1_X0) / ds)
    add(np.stack([np.linspace(LEG1_X0, LEG1_X1, n1, endpoint=False), np.full(n1, LEG1_Y)], 1), 1)
    n2 = int(math.pi * TURN_R_M / ds)
    ph = np.linspace(0, math.pi, n2, endpoint=False)
    cy = LEG1_Y + TURN_R_M
    add(np.stack([LEG1_X1 + TURN_R_M * np.sin(ph), cy - TURN_R_M * np.cos(ph)], 1), 2)
    y2 = cy + TURN_R_M
    n3 = int((LEG1_X1 - LEG2_X1) / ds)
    add(np.stack([np.linspace(LEG1_X1, LEG2_X1, n3, endpoint=False), np.full(n3, y2)], 1), 3)
    assert abs(y2 - (STRUCT_C[1] - ORBIT_R_M)) < 1e-9, "leg 2 must be tangent to the orbit circle"
    n4 = int(2 * math.pi * ORBIT_R_M * ORBIT_FRACTION / ds)
    ph = np.linspace(0, 2 * math.pi * ORBIT_FRACTION, n4)
    add(np.stack([STRUCT_C[0] - ORBIT_R_M * np.sin(ph), STRUCT_C[1] - ORBIT_R_M * np.cos(ph)], 1), 4, ph)
    xy = np.concatenate(pts)
    sid = np.concatenate(seg)
    phi = np.concatenate(phi)
    d = np.gradient(xy, ds, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return xy, heading, sid, phi


@dataclass
class Trajectory:
    t: np.ndarray  # (N,) s from sim start, uniform GRID_DT_S
    p: np.ndarray  # (N,3) body origin in W
    euler: np.ndarray  # (N,3) roll, pitch, yaw (yaw unwrapped), R_WB = Rz Ry Rx
    R: np.ndarray  # (N,3,3) R_WB
    v_W: np.ndarray  # (N,3) m/s
    a_W: np.ndarray  # (N,3) m/s^2 (kinematic, gravity not included)
    w_B: np.ndarray  # (N,3) rad/s body angular velocity
    seg: np.ndarray  # (N,) path segment id (0 = hold)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    def _interp(self, times, arr):
        times = np.clip(np.asarray(times, dtype=float), 0.0, self.t[-1])
        i = np.minimum((times / GRID_DT_S).astype(int), len(self.t) - 2)
        f = (times / GRID_DT_S - i)[..., None]
        return arr[i] * (1 - f) + arr[i + 1] * f

    def state_at(self, times):
        """p (M,3), R (M,3,3) (nlerp of the 1 kHz rotations), v_W (M,3), w_B (M,3) at arbitrary sim times."""
        times = np.asarray(times, dtype=float)
        p = self._interp(times, self.p)
        v = self._interp(times, self.v_W)
        w = self._interp(times, self.w_B)
        i = np.minimum((np.clip(times, 0, self.t[-1]) / GRID_DT_S).astype(int), len(self.t) - 2)
        f = (np.clip(times, 0, self.t[-1]) / GRID_DT_S - i)[:, None, None]
        r = self.R[i] * (1 - f) + self.R[i + 1] * f
        u, _, vt = np.linalg.svd(r)  # project the blend back onto SO(3) (angle between samples is ~1e-3 rad)
        return p, u @ vt, v, w

    def keypoints(self, rate_hz: float = rig.KEYPOINT_RATE_HZ):
        """t (K,), p (K,3), roll/pitch/yaw (K,3) for the Stonefish pwl trajectory (unwrapped yaw: no quaternion
        sign flips between keypoints)."""
        step = int(round(1.0 / rate_hz / GRID_DT_S))
        idx = np.arange(0, len(self.t), step)
        return self.t[idx], self.p[idx], self.euler[idx]


def build_trajectory() -> Trajectory:
    xy, heading_path, sid, phi_path = _path()
    ds = 0.005
    s = np.arange(len(xy)) * ds
    # speed profile along arc length: slow in the pit zone, arcs slightly slower, smoothed over 1.5 m
    v_s = np.where(sid == 1, SPEED_LEG_MPS, np.where(sid == 3, SPEED_LEG_MPS, SPEED_ARC_MPS))
    pit_zone = (sid == 1) & (xy[:, 0] > PIT_C[0] - PIT_HX_M - 4.0) & (xy[:, 0] < PIT_C[0] + PIT_HX_M + 4.0)
    v_s = np.where(pit_zone, SPEED_SLOW_MPS, v_s)
    v_s = gaussian_filter1d(v_s, 1.5 / ds, mode="nearest")
    t_of_s = np.concatenate([[0.0], np.cumsum(ds / v_s[:-1])]) + HOLD_START_S
    t_end = t_of_s[-1] + HOLD_END_S
    t = np.arange(0.0, t_end + 0.5 * GRID_DT_S, GRID_DT_S)
    s_t = np.interp(t, np.concatenate([[0.0], t_of_s]), np.concatenate([[0.0], s]))
    idx = s_t / ds
    i0 = np.minimum(idx.astype(int), len(xy) - 2)
    fr = (idx - i0)[:, None]
    pos_xy = xy[i0] * (1 - fr) + xy[i0 + 1] * fr
    seg_t = sid[i0]
    seg_t = np.where(t < HOLD_START_S, 0, seg_t)
    sig = lambda sec: sec / GRID_DT_S  # noqa: E731
    pos_xy = gaussian_filter1d(pos_xy, sig(1.0), axis=0, mode="nearest")

    # windows: 0 during the holds, 1 on the move
    win = _smoothstep((t - (HOLD_START_S - 1.0)) / 4.0) * _smoothstep((t_end - HOLD_END_S + 1.0 - t) / 4.0 + 1.0)
    # yaw: path heading + sway on the legs, facing the structure during the orbit
    head = np.interp(s_t, s, heading_path)
    phi = np.interp(s_t, s, phi_path)
    s_orbit0 = float(s[np.argmax(sid == 4)])
    w_orb = _smoothstep((s_t - (s_orbit0 - 3.0)) / 6.0)
    yaw_leg = head + SWAY_AMP_RAD * np.sin(2 * math.pi * t / SWAY_PERIOD_S) * win * (1 - w_orb)
    yaw_orb = math.pi / 2 - phi
    yaw = gaussian_filter1d((1 - w_orb) * yaw_leg + w_orb * yaw_orb, sig(0.8), mode="nearest")
    pitch = PITCH_AMP_RAD * np.sin(2 * math.pi * t / PITCH_PERIOD_S + 0.6) * win
    roll = ROLL_AMP_RAD * np.sin(2 * math.pi * t / ROLL_PERIOD_S) * win

    # depth: hold the altitude above the (pit-capped) shelf, smoothed so the vehicle does not dive into the pit
    z_floor = np.minimum(terrain_z(pos_xy[:, 0], pos_xy[:, 1]), SHELF_Z_M + 0.9)
    z_floor = gaussian_filter1d(z_floor, sig(2.5), mode="nearest")
    alt = ALT_MEAN_M + ALT_AMP_M * np.sin(2 * math.pi * t / ALT_PERIOD_S + 1.0) * win
    z = gaussian_filter1d(np.maximum(z_floor - alt, 2.0), sig(0.5), mode="nearest")

    p = np.column_stack([pos_xy, z])
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    R = np.empty((len(t), 3, 3))
    R[:, 0] = np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], 1)
    R[:, 1] = np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], 1)
    R[:, 2] = np.stack([-sp, cp * sr, cp * cr], 1)
    v = np.gradient(p, GRID_DT_S, axis=0)
    a = np.gradient(v, GRID_DT_S, axis=0)
    Rd = np.gradient(R, GRID_DT_S, axis=0)
    wh = np.einsum("nji,njk->nik", R, Rd)  # R^T Rdot = skew(w_B)
    w_B = np.stack([wh[:, 2, 1], wh[:, 0, 2], wh[:, 1, 0]], 1)
    return Trajectory(t, p, np.column_stack([roll, pitch, yaw]), R, v, a, w_B, seg_t)


def dvl_expected(traj: Trajectory, times, r: rig.Rig | None = None, azimuths_deg=rig.DVL_BEAM_AZIMUTHS_DEG,
                 alt_max_m: float = rig.DVL_ALT_MAX_M, alt_min_m: float = rig.DVL_ALT_MIN_M):
    """Expected DVL altitude model against the analytic terrain (no structures): returns
    (valid (M,), slant_min (M,), vertical (M,)); valid = nearest beam hit within [alt_min, alt_max].
    The beam azimuth pattern inside Stonefish is not documented, hence the azimuths argument."""
    r = r or rig.build_rig()
    times = np.asarray(times, dtype=float)
    p, R, _, _ = traj.state_at(times)
    p_d = p + R @ r.T_B_D[:3, 3]
    R_wd = R @ r.T_B_D[:3, :3]
    th = math.radians(rig.DVL_BEAM_ANGLE_DEG)
    step = 0.05
    ranges = np.arange(step, alt_max_m / 1.0 + 5.0, step)
    slant = np.full((len(times), len(azimuths_deg)), np.inf)
    for j, az in enumerate(azimuths_deg):
        d_d = np.array([math.sin(th) * math.cos(math.radians(az)), math.sin(th) * math.sin(math.radians(az)),
                        -math.cos(th)])  # beams along the sensor -z axis (Stonefish default, see rig.py)
        d_w = R_wd @ d_d
        pts = p_d[:, None, :] + ranges[None, :, None] * d_w[:, None, :]
        below = pts[:, :, 2] >= terrain_z(pts[:, :, 0], pts[:, :, 1])
        hit = below.any(axis=1)
        first = np.argmax(below, axis=1)
        slant[hit, j] = ranges[first[hit]]
    smin = slant.min(axis=1)
    vertical = terrain_z(p_d[:, 0], p_d[:, 1]) - p_d[:, 2]
    return (smin >= alt_min_m) & (smin <= alt_max_m), smin, vertical


def invalid_intervals(t: np.ndarray, valid: np.ndarray):
    """[(t_start, t_end)] of consecutive invalid samples."""
    out, start = [], None
    for ti, ok in zip(t, valid):
        if not ok and start is None:
            start = ti
        if ok and start is not None:
            out.append((float(start), float(ti)))
            start = None
    if start is not None:
        out.append((float(start), float(t[-1])))
    return out
