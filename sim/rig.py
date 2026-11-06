"""Single source of truth for the simulated vehicle: frames, sensor geometry, noise, rates.

Everything that appears in the Stonefish scenario (gen_scenario.py), the Kalibr yamls (gen_calib.py) and the
conversion (convert.py) is derived from the constants here, so the three cannot drift apart.

Frames (all right-handed)
  W  Stonefish world = NED: x north, y east, z down. The sea surface is the plane z = 0, depth = z.
  B  body = animated frame = IMU frame, FRD: x forward, y right, z down. T_W_B is the animated trajectory.
  C0, C1  left / right camera, OpenCV optical convention: x right, y down, z forward (optical axis).
     Stonefish's camera axes are inferred from stonefish/Tests/CameraTest/CameraTestManager.cpp (a camera attached with
     sf::Quaternion(-pi/2, 0, pi/2) to a frame that moves through the scene at heading yaw looks along the frame's
     -x axis, which only works if the optical axis is the sensor +z, x right, y down); see README "Assumptions".
  D  DVL sensor frame. The Stonefish default has the four beams along the sensor -z axis (the Girona500 scenario,
     stonefish/Tests/Data/girona500auv_full.scn, mounts its downward DVL with roll = pi), so D is B rotated by
     roll = pi (z up) and a 45 deg mount yaw about it.

Stonefish orientation convention for <origin rpy="roll pitch yaw">: R = Rz(yaw) Ry(pitch) Rx(roll)
(stonefish/docs/animated.rst: keypoint rpy; btQuaternion(yaw, pitch, roll)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GRAVITY_MPS2 = 9.81  # run_bag kGravity; Stonefish pressure sensor assumed to use the same value (checked by convert.py)
WATER_DENSITY_KGPM3 = 1000.0  # <water density=...> in the scenario and --water-density of run_bag

# --- timing ------------------------------------------------------------------------------------------------------
SIM_RATE_HZ = 200.0  # Stonefish physics step rate (stonefish_simulator <rate> argument)
CAM_RATE_HZ = 20.0
IMU_RATE_HZ = 200.0
PRESSURE_RATE_HZ = 10.0
DVL_RATE_HZ = 5.0
GT_ODOM_RATE_HZ = 50.0
KEYPOINT_RATE_HZ = 50.0  # animated-trajectory keypoint spacing (pwl interpolation)
BAG_T0_S = 1000.0  # sim time 0 is written as this stamp in the converted bag / tum / csv

# --- camera ----------------------------------------------------------------------------------------------------
IMG_W, IMG_H = 720, 540
HFOV_DEG = 80.0
BASELINE_M = 0.12
CAM_TILT_DOWN_DEG = 25.0  # optical axis pitched down from body +x
T_B_CAMS_ORIGIN_M = np.array([0.15, 0.0, 0.03])  # midpoint of the two cameras in B


def intrinsics() -> tuple[float, float, float, float]:
    """Pinhole fx, fy, cx, cy [px] for the OpenCV convention (pixel centres at integer coordinates).

    fx = (W/2) / tan(hfov/2); Stonefish's GenerateCameraMsgPrototypes derives fy = (H/2) / tan(vfov/2) with
    tan(vfov/2) = H/W tan(hfov/2), hence fy = fx (square pixels). The principal point of a symmetric GL frustum is
    at (W-1)/2, (H-1)/2 in pixel-centre coordinates (Stonefish's CameraInfo says W/2, H/2: a 0.5 px convention
    difference that is written down in the README)."""
    fx = (IMG_W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    return fx, fx, (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0


# --- IMU (Kalibr continuous-time noise parameters, truth for the synthetic IMU and for imu.yaml) --------------
ACC_NOISE_DENSITY = 2.0e-3  # m/s^2/sqrt(Hz)
ACC_RANDOM_WALK = 3.0e-4  # m/s^3/sqrt(Hz)
GYR_NOISE_DENSITY = 1.7e-4  # rad/s/sqrt(Hz)
GYR_RANDOM_WALK = 2.0e-5  # rad/s^2/sqrt(Hz)
ACC_INIT_BIAS_SIGMA = 0.02  # m/s^2
GYR_INIT_BIAS_SIGMA = 1.0e-3  # rad/s

# --- pressure --------------------------------------------------------------------------------------------------
PRESSURE_NOISE_PA = 20.0  # ~2 mm of water, a Bar30-class sensor
PRESSURE_RANGE_PA = 500000.0

# --- DVL -------------------------------------------------------------------------------------------------------
DVL_BEAM_ANGLE_DEG = 30.0  # from the sensor axis
DVL_ALT_MIN_M = 0.3
DVL_ALT_MAX_M = 12.0  # bottom lock is lost beyond this (stonefish <range altitude_max>)
DVL_VEL_MAX_MPS = 3.0
DVL_NOISE_VEL_PERCENT = 0.5  # <noise velocity_percent>; Stonefish docs: "percentage of the measured velocity"
DVL_NOISE_VEL_MPS = 0.003  # <noise velocity>, constant additive std
DVL_NOISE_ALT_M = 0.03
DVL_POS_B_M = np.array([-0.10, 0.0, 0.12])  # DVL origin in B (under the hull, aft of the IMU)
DVL_MOUNT_YAW_DEG = -45.0  # about the D z axis (Stonefish rpy yaw), applied after the roll = pi flip
DVL_BEAM_AZIMUTHS_DEG = (45.0, 135.0, 225.0, 315.0)  # beam azimuth in D; ONLY used by the expected-lock model, the
# real pattern inside Stonefish is not documented (the model is also evaluated with 0/90/180/270)

# --- pose helpers ----------------------------------------------------------------------------------------------


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Stonefish <origin rpy>: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rz(yaw) @ ry(pitch) @ rx(roll)


def rpy_from_rot(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rot_from_rpy for |pitch| < pi/2."""
    pitch = -math.asin(max(-1.0, min(1.0, r[2, 0])))
    return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])


def make_t(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def inv_t(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


# R_B_C for a forward-looking camera: C x = B y (right), C y = B z (down), C z = B x (forward).
R_B_CFWD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Rig:
    T_B_C0: np.ndarray  # left camera pose in B
    T_B_C1: np.ndarray  # right camera pose in B
    T_B_D: np.ndarray  # DVL pose in B

    @property
    def T_C1_C0(self) -> np.ndarray:
        """Kalibr T_cn_cnm1 of cam1: maps cam0 points into cam1 (cam0 pose in cam1)."""
        return inv_t(self.T_B_C1) @ self.T_B_C0


def build_rig() -> Rig:
    r_b_c = ry(-math.radians(CAM_TILT_DOWN_DEG)) @ R_B_CFWD  # tilt the optical axis (B +x) towards +z (down)
    # Baseline along the camera x axis (= B y, unaffected by the tilt about B y).
    off = np.array([0.0, BASELINE_M / 2.0, 0.0])
    t0 = T_B_CAMS_ORIGIN_M - off
    t1 = T_B_CAMS_ORIGIN_M + off
    r_b_d = rz(math.radians(DVL_MOUNT_YAW_DEG)) @ rx(math.pi)
    return Rig(make_t(r_b_c, t0), make_t(r_b_c, t1), make_t(r_b_d, DVL_POS_B_M))


def quat_xyzw_from_rot(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [x, y, z, w] with w >= 0."""
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def rot_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
