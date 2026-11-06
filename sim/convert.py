#!/usr/bin/env python3
"""Raw Stonefish rosbag2 -> benchmark inputs for uw_ct_vio run_bag, plus a sanity report.

  python3 convert.py --bag <raw_bag> --out <dir> [--cam-latency-s 0.0] [--cam-time auto|mapped|snap]
                     [--imu synth|stonefish] [--img-quality 95] [--dump-frames 3] [--seed 1]

Outputs in <dir>: bench_bag/ (rosbag2, sqlite3), gt_leftcam.tum, dvl.csv, sanity.json, convert_info.json, frames/*.png.

WHY RE-STAMP. stonefish_ros2 stamps every message with the ROS clock at publication (ROS2ScenarioParser/ROS2Interface:
header.stamp = nh_->get_clock()->now()), not with simulation time, and images are stamped when the GPU read-back
finishes. The animated trajectory is published every physics step (topic /sim/traj_state) with a pose that is an exact
function of simulation time, so the pose gives the simulation time of each step; every other message is assigned the
simulation time of the traj_state message that is nearest in wall stamp (same step: sensors are published inside the
same SimulationStepCompleted call). Camera frames get the mapped time of the LEFT frame's delivery stamp minus --cam-latency-s,
and left/right frames of a pair share that stamp. All output stamps are BAG_T0_S + simulation seconds.

SYNTHETIC IMU. The benchmark IMU is generated from the analytic trajectory (specific force f = R_WB^T (a_W - g_W),
g_W = (0, 0, +9.81) NED, angular rate in B) with Kalibr-style white noise and bias random walk (rig.py), on an exact
200 Hz grid. Reason: the animated body has piece-wise-linear velocity, so a simulator-side accelerometer would see
acceleration impulses at every keypoint, and whether Stonefish adds gravity to its accelerometer is undocumented.
The simulator IMU is still recorded (/sim/imu_sf) and checked in the report; --imu stonefish uses it instead.

DVL validity. stonefish_ros2 sets DVL.altitude = -1 unless the sensor status is 0 or 2 (ROS2Interface::PublishDVL) and
never sets velocity_covariance to -1, so a row is valid iff altitude >= 0 (and finite and velocity finite). Invalid rows
are written with zero velocity and valid = 0.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import rig
import trajectory as tj

HERE = Path(__file__).resolve().parent
IMG_TOPICS = ("/sim/cam_left/image_color", "/sim/cam_right/image_color")
TRAJ_TOPIC, DVL_TOPIC, PRESS_TOPIC, IMU_SF_TOPIC, ODOM_TOPIC = (
    "/sim/traj_state", "/sim/dvl", "/sim/pressure", "/sim/imu_sf", "/sim/gt_odom")
# Left/right frames whose wall stamps differ by less than this fraction of the camera period belong together.
# Stonefish renders both cameras in the same step but stamps each image with now() when its callback publishes it,
# so the right image trails the left by the left callback's cost (~22-27 ms for 720x540 rgb8 on ws1); a fixed 25 ms
# tolerance silently dropped ~30 % of the frames in long runs. Half a period keeps the pairing unambiguous.
PAIR_TOL_PERIOD_FRACTION = 0.45
SNAP_RMS_MAX_S = 0.004  # snap camera stamps to a regular grid only if they already sit within this of it


# --------------------------------------------------------------------------------------------------------------
# Pure logic (numpy only; unit-tested without ROS)
# --------------------------------------------------------------------------------------------------------------
@dataclass
class TimeMap:
    wall: np.ndarray  # (N,) wall stamps of the traj_state messages [s]
    sim: np.ndarray  # (N,) simulation time of each of them [s]
    info: dict = field(default_factory=dict)


def quat_to_rot_batch(q: np.ndarray) -> np.ndarray:
    return np.stack([rig.rot_from_quat_xyzw(x) for x in q])


def invert_trajectory(traj: tj.Trajectory, pos: np.ndarray, quat: np.ndarray, coarse_stride: int = 5) -> np.ndarray:
    """Simulation time [s] at which the analytic trajectory has the given poses (min of |dp| + 0.5 angle).
    Ambiguous while the vehicle is stationary, so callers should only pass moving poses."""
    Rq = quat_to_rot_batch(quat)
    gp, gR = traj.p[::coarse_stride], traj.R[::coarse_stride]
    out = np.empty(len(pos))
    for s in range(0, len(pos), 64):
        sl = slice(s, s + 64)
        dp = np.linalg.norm(pos[sl, None, :] - gp[None, :, :], axis=2)
        tr = np.einsum("qij,gij->qg", Rq[sl], gR)
        cost = dp + 0.5 * np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
        out[sl] = np.argmin(cost, axis=1) * coarse_stride
    for k in range(len(pos)):  # refine on the 1 kHz grid
        lo, hi = max(0, int(out[k]) - coarse_stride), min(len(traj.t), int(out[k]) + coarse_stride + 1)
        dp = np.linalg.norm(pos[k] - traj.p[lo:hi], axis=1)
        tr = np.einsum("ij,gij->g", Rq[k], traj.R[lo:hi])
        out[k] = lo + int(np.argmin(dp + 0.5 * np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))
    return out * tj.GRID_DT_S


def fit_time_map(wall: np.ndarray, pos: np.ndarray, quat: np.ndarray, traj: tj.Trajectory, sim_rate_hz: float,
                 n_queries: int = 600, regular_tol_s: float = 0.002) -> TimeMap:
    """Assign a simulation time to every traj_state message (see module docstring)."""
    n = len(wall)
    if n < 100:
        raise ValueError(f"only {n} {TRAJ_TOPIC} messages in the bag")
    idx = np.unique(np.linspace(5, n - 6, min(n_queries, n - 10)).astype(int))
    step = np.linalg.norm(pos[idx + 5] - pos[idx - 5], axis=1)  # displacement over 10 messages
    moving = step > 0.02
    idx = idx[moving]
    if len(idx) < 20:
        raise ValueError("too few moving traj_state poses to recover the simulation time")
    t_inv = invert_trajectory(traj, pos[idx], quat[idx])
    keep = t_inv < traj.duration_s - 0.05  # past the end the pose is constant
    idx, t_inv = idx[keep], t_inv[keep]
    dt = 1.0 / sim_rate_hz
    a = float(np.median(t_inv - idx * dt))
    resid = t_inv - (a + idx * dt)
    p95 = float(np.percentile(np.abs(resid), 95))
    if p95 < regular_tol_s:
        sim = a + np.arange(n) * dt
        mode = "regular"
    else:  # steps were dropped or time stretched: follow the inverted poses as a function of wall time
        w_q = wall[idx]
        rate = (t_inv[-1] - t_inv[0]) / (w_q[-1] - w_q[0])  # sim seconds per wall second, to extrapolate the holds
        sim = np.interp(wall, w_q, t_inv)
        sim = np.where(wall < w_q[0], t_inv[0] + (wall - w_q[0]) * rate, sim)
        sim = np.where(wall > w_q[-1], t_inv[-1] + (wall - w_q[-1]) * rate, sim)
        sim = np.maximum.accumulate(sim)
        mode = "interpolated"
    slope = float(np.polyfit(np.arange(n), wall, 1)[0])
    gaps = int(np.sum(np.diff(wall) > 3.0 * dt))
    info = {"mode": mode, "n_messages": n, "intercept_s": a, "resid_p95_s": p95, "resid_max_s": float(np.abs(resid).max()),
            "wall_per_step_s": slope, "realtime_factor": dt / slope if slope > 0 else float("nan"),
            "wall_gaps_gt_3_steps": gaps, "sim_first_s": float(sim[0]), "sim_last_s": float(sim[-1])}
    return TimeMap(np.asarray(wall, float), sim, info)


def scalar_sim_time(tm: TimeMap, wall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulation time of scalar-sensor messages: the nearest traj_state message in wall stamp. Returns (sim, |wall lag|)."""
    j = np.clip(np.searchsorted(tm.wall, wall), 1, len(tm.wall) - 1)
    j = np.where(np.abs(tm.wall[j - 1] - wall) <= np.abs(tm.wall[j] - wall), j - 1, j)
    return tm.sim[j], np.abs(tm.wall[j] - wall)


def image_sim_time(tm: TimeMap, wall: np.ndarray, latency_s: float) -> np.ndarray:
    return np.interp(np.asarray(wall) - latency_s, tm.wall, tm.sim)


def pair_images(wall_l: np.ndarray, wall_r: np.ndarray, tol_s: float) -> list[tuple[int, int]]:
    """Greedy in-order pairing of left/right frames by wall stamp."""
    pairs, j = [], 0
    for i, w in enumerate(wall_l):
        while j < len(wall_r) and wall_r[j] < w - tol_s:
            j += 1
        if j < len(wall_r) and abs(wall_r[j] - w) <= tol_s:
            pairs.append((i, j))
            j += 1
    return pairs


def assign_camera_times(t: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    """mapped: keep; snap: regular grid t0 + k T (T = median spacing); auto: snap if the residual RMS is small."""
    if len(t) < 3:
        return t.copy(), {"mode": "mapped", "period_s": float("nan"), "rms_s": float("nan")}
    period = float(np.median(np.diff(t)))
    k = np.round((t - t[0]) / period)
    A = np.stack([np.ones_like(k), k], 1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    grid = A @ coef
    rms = float(np.sqrt(np.mean((t - grid) ** 2)))
    info = {"period_s": float(coef[1]), "rms_s": rms, "dropped_frames": int(k[-1] - (len(t) - 1))}
    use = mode == "snap" or (mode == "auto" and rms < SNAP_RMS_MAX_S)
    info["mode"] = "snap" if use else "mapped"
    return (grid if use else t.copy()), info


def synth_imu(traj: tj.Trajectory, rate_hz: float = rig.IMU_RATE_HZ, seed: int = 1, noise: bool = True) -> dict:
    """Synthetic 200 Hz IMU from the analytic trajectory. Returns t (sim s), gyro (B, rad/s), acc (B, m/s^2 specific force)."""
    dt = 1.0 / rate_hz
    t = np.arange(0.0, traj.duration_s, dt)
    _, R, _, _ = traj.state_at(t)
    ia = np.round(t / tj.GRID_DT_S).astype(int)
    a_W, w_B = traj.a_W[ia], traj.w_B[ia]
    g_W = np.array([0.0, 0.0, rig.GRAVITY_MPS2])
    f_B = np.einsum("nji,nj->ni", R, a_W - g_W)
    gyro, acc = w_B.copy(), f_B.copy()
    if noise:
        rng = np.random.default_rng(seed)
        for arr, nd, rw, b0 in ((gyro, rig.GYR_NOISE_DENSITY, rig.GYR_RANDOM_WALK, rig.GYR_INIT_BIAS_SIGMA),
                                (acc, rig.ACC_NOISE_DENSITY, rig.ACC_RANDOM_WALK, rig.ACC_INIT_BIAS_SIGMA)):
            bias = b0 * rng.standard_normal(3) + np.cumsum(rw * math.sqrt(dt) * rng.standard_normal((len(t), 3)), axis=0)
            arr += bias + nd / math.sqrt(dt) * rng.standard_normal(arr.shape)
    return {"t": t, "gyro": gyro, "acc": acc}


def dvl_validity(altitude: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return (altitude >= 0.0) & np.isfinite(altitude) & np.all(np.isfinite(vel), axis=1)


def dvl_expected_velocity(traj: tj.Trajectory, t: np.ndarray, r: rig.Rig | None = None) -> dict:
    """Noise-free DVL velocity at the DVL origin under three hypotheses about the frame it is expressed in."""
    r = r or rig.build_rig()
    _, R, v_W, w_B = traj.state_at(t)
    l = r.T_B_D[:3, 3]
    v_B = np.einsum("nji,nj->ni", R, v_W) + np.cross(w_B, l)
    return {"dvl_frame": v_B @ r.T_B_D[:3, :3], "body_frame": v_B, "world_ned": np.einsum("nij,nj->ni", R, v_B)}


def gt_leftcam(traj: tj.Trajectory, t: np.ndarray, r: rig.Rig | None = None) -> np.ndarray:
    """TUM rows (t_bag, x, y, z, qx, qy, qz, qw) of the LEFT CAMERA in world NED: T_W_C0 = T_W_B T_B_C0."""
    r = r or rig.build_rig()
    p, R, _, _ = traj.state_at(t)
    rows = np.empty((len(t), 8))
    for i in range(len(t)):
        T = rig.make_t(R[i], p[i]) @ r.T_B_C0
        rows[i] = [rig.BAG_T0_S + t[i], *T[:3, 3], *rig.quat_xyzw_from_rot(T[:3, :3])]
    return rows


def invalid_stats(t: np.ndarray, valid: np.ndarray) -> list[list[float]]:
    return [[round(a, 2), round(b, 2)] for a, b in tj.invalid_intervals(t, valid)]


def stream_rate(t: np.ndarray) -> dict:
    if len(t) < 2:
        return {"n": int(len(t))}
    d = np.diff(t)
    return {"n": int(len(t)), "median_dt_s": float(np.median(d)), "rate_hz": float(1.0 / np.median(d)),
            "max_dt_s": float(d.max()), "span_s": float(t[-1] - t[0])}


def circular_phase(t: np.ndarray, period: float) -> tuple[float, float]:
    """(mean phase [s] in [0, period), concentration R in [0, 1]) of t modulo period."""
    z = np.exp(2j * np.pi * (t % period) / period)
    m = z.mean()
    return float((np.angle(m) % (2 * np.pi)) / (2 * np.pi) * period), float(np.abs(m))


def build_report(traj, tm, streams: dict, cam: dict, dvl: dict, pressure: dict, imu_sf: dict | None) -> dict:
    rep = {"time_map": tm.info, "camera": cam, "dvl": dvl, "pressure": pressure, "imu_stonefish": imu_sf}
    rep["trajectory"] = {"duration_s": traj.duration_s, "path_length_m": float(np.sum(np.linalg.norm(np.diff(traj.p[::10], axis=0), axis=1))),
                         "speed_mean_mps": float(np.mean(np.linalg.norm(traj.v_W, axis=1)[int(5 / tj.GRID_DT_S):-int(4 / tj.GRID_DT_S)]))}
    rep["streams"] = streams
    return rep


# --------------------------------------------------------------------------------------------------------------
# Bag IO (rosbag2_py; needs a sourced ROS 2 environment with stonefish_ros2 for the DVL message type)
# --------------------------------------------------------------------------------------------------------------
def _ros():
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message, serialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:  # pragma: no cover
        sys.exit(f"convert.py needs a sourced ROS 2 environment (rosbag2_py, rclpy): {e}")
    return rosbag2_py, deserialize_message, serialize_message, get_message


def topic_metadata(idx: int, name: str, typ: str):
    """rosbag2_py.TopicMetadata across distros (Jazzy takes an id first, Humble does not)."""
    rosbag2_py, _, _, _ = _ros()
    try:
        return rosbag2_py.TopicMetadata(id=idx, name=name, type=typ, serialization_format="cdr")
    except TypeError:
        return rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")


def _storage_id(path: Path) -> str:
    meta = path / "metadata.yaml"
    if meta.exists():
        for line in meta.read_text().splitlines():
            if line.strip().startswith("storage_identifier:"):
                return line.split(":", 1)[1].strip()
    return "sqlite3"


def _open_reader(path: Path, required, optional=()):
    rosbag2_py, _, _, _ = _ros()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=_storage_id(path)), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in required if t not in types]
    if missing:
        sys.exit(f"bag {path} lacks topics {missing}; has {sorted(types)}")
    wanted = [*required, *[t for t in optional if t in types]]
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    return reader, {t: types[t] for t in wanted}


def _header_stamp_fast(raw: bytes) -> float | None:
    """Header stamp of a CDR little-endian message whose first field is std_msgs/Header (offset 4)."""
    if len(raw) > 12 and raw[0] == 0 and raw[1] == 1:
        sec, nsec = struct.unpack_from("<iI", raw, 4)
        return sec + 1e-9 * nsec
    return None


def _stamp(msg) -> float:
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def read_light_streams(path: Path) -> dict:
    """Everything except image pixels: header stamps of the images, full content of the rest."""
    _, deserialize, _, get_message = _ros()
    reader, types = _open_reader(path, [TRAJ_TOPIC, DVL_TOPIC, PRESS_TOPIC, *IMG_TOPICS], [IMU_SF_TOPIC])
    classes = {n: get_message(t) for n, t in types.items()}
    out = {k: [] for k in ("traj", "dvl", "press", "imu", "img_l", "img_r")}
    while reader.has_next():
        name, raw, _ = reader.read_next()
        if name in IMG_TOPICS:
            w = _header_stamp_fast(raw)
            if w is None:
                w = _stamp(deserialize(raw, classes[name]))
            out["img_l" if name == IMG_TOPICS[0] else "img_r"].append(w)
            continue
        m = deserialize(raw, classes[name])
        w = _stamp(m)
        if name == TRAJ_TOPIC:
            p, q = m.pose.pose.position, m.pose.pose.orientation
            out["traj"].append((w, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        elif name == DVL_TOPIC:
            out["dvl"].append((w, m.velocity.x, m.velocity.y, m.velocity.z, m.altitude))
        elif name == PRESS_TOPIC:
            out["press"].append((w, m.fluid_pressure, m.variance))
        elif name == IMU_SF_TOPIC:
            out["imu"].append((w, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                               m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
    arrays = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    for k in ("img_l", "img_r"):
        arrays[k] = arrays[k].reshape(-1)
    for k, cols in (("traj", 8), ("dvl", 5), ("press", 3), ("imu", 7)):
        arrays[k] = arrays[k].reshape(len(out[k]), cols)
    return arrays


def _encoder():
    """(name, fn(gray uint8 HxW, quality) -> bytes) for the first available JPEG encoder, or None."""
    try:
        import cv2

        return "cv2", lambda g, q: cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1].tobytes()
    except ImportError:
        pass
    try:
        import io

        from PIL import Image

        def enc(g, q):
            b = io.BytesIO()
            Image.fromarray(g).save(b, format="JPEG", quality=int(q))
            return b.getvalue()

        return "PIL", enc
    except ImportError:
        return None


def rgb8_to_gray(msg) -> np.ndarray:
    if msg.encoding not in ("rgb8", "bgr8", "mono8"):
        raise ValueError(f"unexpected image encoding {msg.encoding}")
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    if msg.encoding == "mono8":
        return a[:, :msg.width].copy()
    a = a[:, :msg.width * 3].reshape(msg.height, msg.width, 3).astype(np.float32)
    w = (0.299, 0.587, 0.114) if msg.encoding == "rgb8" else (0.114, 0.587, 0.299)
    return np.clip(np.rint(a @ np.array(w, np.float32)), 0, 255).astype(np.uint8)


def write_bench_bag(out_dir: Path, path_raw: Path, pairs, t_pairs, wall_l, wall_r, small_msgs, quality: int,
                    dump_frames: int) -> dict:
    """Stream the images once, writing them (mono8, JPEG when an encoder exists) merged with the small messages in time order."""
    rosbag2_py, deserialize, serialize, get_message = _ros()
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import CompressedImage, Image

    enc = _encoder()
    suffix = "compressed" if enc else "image_raw"
    topics = (f"/sim/cam0/{suffix}", f"/sim/cam1/{suffix}")
    writer = rosbag2_py.SequentialWriter()
    bag = out_dir / "bench_bag"
    if bag.exists():
        sys.exit(f"{bag} already exists; remove it or pick another --out")
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    type_of = {topics[0]: "sensor_msgs/msg/CompressedImage" if enc else "sensor_msgs/msg/Image",
               topics[1]: "sensor_msgs/msg/CompressedImage" if enc else "sensor_msgs/msg/Image"}
    for idx, (name, typ) in enumerate({**type_of, **{n: ty for n, (ty, _) in small_msgs.items()}}.items()):
        writer.create_topic(topic_metadata(idx, name, typ))
    merged = sorted((tt, n, buf) for n, (_, items) in small_msgs.items() for tt, buf in items)
    mi = 0

    def stamp_of(t_sim: float):
        tb = rig.BAG_T0_S + t_sim
        return Time(sec=int(tb), nanosec=int(round((tb - int(tb)) * 1e9)))

    reader, types = _open_reader(path_raw, IMG_TOPICS)
    cls = get_message(types[IMG_TOPICS[0]])
    pair_of = {0: {i: k for k, (i, _) in enumerate(pairs)}, 1: {j: k for k, (_, j) in enumerate(pairs)}}
    counters = [0, 0]
    n_written = 0
    dump_dir = out_dir / "frames"
    from gen_assets import write_png

    while reader.has_next():
        name, raw, _ = reader.read_next()
        side = 0 if name == IMG_TOPICS[0] else 1
        i = counters[side]
        counters[side] += 1
        k = pair_of[side].get(i)
        if k is None:
            continue
        t_sim = float(t_pairs[k])
        gray = rgb8_to_gray(deserialize(raw, cls))
        if dump_frames and k in (0, len(pairs) // 2, len(pairs) - 1):
            dump_dir.mkdir(exist_ok=True)
            write_png(dump_dir / f"cam{side}_frame{k:05d}.png", gray)
        while mi < len(merged) and merged[mi][0] <= t_sim:
            writer.write(merged[mi][1], merged[mi][2], int((rig.BAG_T0_S + merged[mi][0]) * 1e9))
            mi += 1
        if enc:
            msg = CompressedImage()
            msg.format = "jpeg"
            msg.data = np.frombuffer(enc[1](gray, quality), dtype=np.uint8)
        else:
            msg = Image()
            msg.height, msg.width, msg.encoding, msg.step = gray.shape[0], gray.shape[1], "mono8", gray.shape[1]
            msg.data = np.ascontiguousarray(gray).reshape(-1)
        msg.header.stamp = stamp_of(t_sim)
        msg.header.frame_id = f"cam{side}"
        writer.write(topics[side], serialize(msg), int((rig.BAG_T0_S + t_sim) * 1e9))
        n_written += 1
    for tt, n, buf in merged[mi:]:
        writer.write(n, buf, int((rig.BAG_T0_S + tt) * 1e9))
    del writer
    return {"left_topic": topics[0], "right_topic": topics[1], "image_encoder": enc[0] if enc else "none (raw mono8)",
            "images_written": n_written}


# --------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cam-latency-s", type=float, default=0.0,
                    help="seconds between rendering and delivery (GPU read-back) of the LEFT frame, subtracted from the image stamps")
    ap.add_argument("--cam-time", choices=("auto", "mapped", "snap"), default="auto")
    ap.add_argument("--imu", choices=("synth", "stonefish"), default="synth")
    ap.add_argument("--img-quality", type=int, default=95)
    ap.add_argument("--dump-frames", type=int, default=1, help="1: write first/middle/last frame pair as PNG, 0: none")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, _, serialize, _ = _ros()
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from sensor_msgs.msg import FluidPressure, Imu

    meta = json.loads((HERE / "scenarios" / "trajectory_meta.json").read_text())
    traj = tj.build_trajectory()
    t_kp, p_kp, e_kp = traj.keypoints()
    import hashlib

    if hashlib.sha256(np.round(np.column_stack([t_kp, p_kp, e_kp]), 6).tobytes()).hexdigest() != meta["keypoint_sha256"]:
        sys.exit("trajectory.py no longer reproduces scenarios/trajectory_meta.json: regenerate the scenarios (gen_scenario.py) "
                 "and re-run the simulation, or the ground truth will not match the recorded bag")
    print("reading bag (headers + small topics) ...", flush=True)
    s = read_light_streams(a.bag)
    tr = s["traj"]
    tm = fit_time_map(tr[:, 0], tr[:, 1:4], tr[:, 4:8], traj, meta["sim_rate_hz"])
    print(f"time map: {tm.info}", flush=True)

    # ---- scalar sensors -------------------------------------------------------------------------------------
    t_dvl, lag_dvl = scalar_sim_time(tm, s["dvl"][:, 0])
    ok_t = (t_dvl >= 0) & (t_dvl <= traj.duration_s)
    t_dvl, dv = t_dvl[ok_t], s["dvl"][ok_t]
    keep = np.concatenate([[True], np.diff(t_dvl) > 0])
    t_dvl, dv = t_dvl[keep], dv[keep]
    valid = dvl_validity(dv[:, 4], dv[:, 1:4])
    vel = np.where(valid[:, None], dv[:, 1:4], 0.0)
    with open(a.out / "dvl.csv", "w") as fh:
        fh.write("t,vx,vy,vz,valid\n")
        for tt, v, ok in zip(t_dvl, vel, valid):
            fh.write(f"{rig.BAG_T0_S + tt:.6f},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f},{int(ok)}\n")
    twist_items = []
    for tt, v, c in zip(t_dvl[valid], vel[valid], dv[valid]):
        m = TwistWithCovarianceStamped()
        sec = rig.BAG_T0_S + tt
        m.header.stamp.sec, m.header.stamp.nanosec = int(sec), int(round((sec - int(sec)) * 1e9))
        m.header.frame_id = "dvl"
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = map(float, v)
        sig2 = (rig.DVL_NOISE_VEL_MPS ** 2)
        for d in (0, 7, 14):
            m.twist.covariance[d] = sig2
        twist_items.append((float(tt), serialize(m)))
    small = {"/sim/dvl_twist": ("geometry_msgs/msg/TwistWithCovarianceStamped", twist_items)}

    t_p, lag_p = scalar_sim_time(tm, s["press"][:, 0])
    okp = (t_p >= 0) & (t_p <= traj.duration_s)
    t_p, pr = t_p[okp], s["press"][okp]
    items = []
    for tt, row in zip(t_p, pr):
        m = FluidPressure()
        sec = rig.BAG_T0_S + tt
        m.header.stamp.sec, m.header.stamp.nanosec = int(sec), int(round((sec - int(sec)) * 1e9))
        m.header.frame_id = "pressure"
        m.fluid_pressure, m.variance = float(row[1]), float(row[2])
        items.append((float(tt), serialize(m)))
    small["/sim/pressure"] = ("sensor_msgs/msg/FluidPressure", items)

    # ---- IMU --------------------------------------------------------------------------------------------------
    imu_sf_report = None
    if len(s["imu"]):
        t_i, _ = scalar_sim_time(tm, s["imu"][:, 0])
        hold = t_i < tj.HOLD_START_S - 0.5
        if hold.sum() > 10:
            imu_sf_report = {"static_gyro_mean_radps": s["imu"][hold, 1:4].mean(0).tolist(),
                             "static_acc_mean_mps2": s["imu"][hold, 4:7].mean(0).tolist(),
                             "static_acc_norm_mps2": float(np.linalg.norm(s["imu"][hold, 4:7].mean(0))),
                             "expected_if_gravity_included": [0.0, 0.0, -rig.GRAVITY_MPS2],
                             "rate": stream_rate(t_i)}
    if a.imu == "synth":
        imu = synth_imu(traj, seed=a.seed)
        t_imu, gyro, acc = imu["t"], imu["gyro"], imu["acc"]
    else:
        if not len(s["imu"]):
            sys.exit("--imu stonefish but the bag has no /sim/imu_sf")
        t_imu, _ = scalar_sim_time(tm, s["imu"][:, 0])
        m_ok = (t_imu >= 0) & (t_imu <= traj.duration_s)
        t_imu, gyro, acc = t_imu[m_ok], s["imu"][m_ok, 1:4], s["imu"][m_ok, 4:7]
    items = []
    for tt, g, f_ in zip(t_imu, gyro, acc):
        m = Imu()
        sec = rig.BAG_T0_S + tt
        m.header.stamp.sec, m.header.stamp.nanosec = int(sec), int(round((sec - int(sec)) * 1e9))
        m.header.frame_id = "imu"
        m.orientation_covariance[0] = -1.0  # no orientation estimate
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, g)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, f_)
        items.append((float(tt), serialize(m)))
    small["/sim/imu"] = ("sensor_msgs/msg/Imu", items)

    # ---- cameras -----------------------------------------------------------------------------------------------
    cam_rate_hz = float(meta.get("cam_rate_hz", rig.CAM_RATE_HZ))
    pairs = pair_images(s["img_l"], s["img_r"], PAIR_TOL_PERIOD_FRACTION / cam_rate_hz)
    tl = image_sim_time(tm, s["img_l"][[i for i, _ in pairs]], a.cam_latency_s)
    t_pair0 = tl  # both cameras are rendered at the same simulation step; the left frame's delivery defines the pair stamp
    lr_dt = np.abs(s["img_l"][[i for i, _ in pairs]] - s["img_r"][[j for _, j in pairs]])
    t_cam, snap_info = assign_camera_times(t_pair0, a.cam_time)
    in_range = (t_cam >= 0.0) & (t_cam <= traj.duration_s)
    pairs = [p for p, ok in zip(pairs, in_range) if ok]
    t_cam = t_cam[in_range]
    phase, conc = circular_phase(t_pair0[in_range], 1.0 / cam_rate_hz)
    print(f"{len(pairs)} stereo pairs of {len(s['img_l'])} left / {len(s['img_r'])} right frames", flush=True)

    # ---- ground truth ----------------------------------------------------------------------------------------
    gt = gt_leftcam(traj, t_cam)
    np.savetxt(a.out / "gt_leftcam.tum", gt, fmt="%.9f", header="t x y z qx qy qz qw (left camera in world NED)")

    info = write_bench_bag(a.out, a.bag, pairs, t_cam, s["img_l"], s["img_r"], small, a.img_quality, a.dump_frames)

    # ---- sanity report ---------------------------------------------------------------------------------------
    streams = {"traj_state": stream_rate(tm.wall), "dvl": stream_rate(t_dvl), "pressure": stream_rate(t_p),
               "image_pairs": stream_rate(t_cam), "image_left_wall": stream_rate(s["img_l"]),
               "image_right_wall": stream_rate(s["img_r"])}
    exp_v = dvl_expected_velocity(traj, t_dvl)
    res = {}
    for name, ev in exp_v.items():
        d = (vel - ev)[valid]
        res[name] = {"rms_mps": float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"),
                     "bias_mps": d.mean(0).tolist() if len(d) else None}
    best = min(res, key=lambda k: res[k]["rms_mps"]) if valid.any() else None
    alt_valid = dv[valid, 4]
    exp_ok = {n: tj.dvl_expected(traj, t_dvl, azimuths_deg=az)[0] for n, az in
              (("beams_45_135_225_315", rig.DVL_BEAM_AZIMUTHS_DEG), ("beams_0_90_180_270", (0.0, 90.0, 180.0, 270.0)))}
    _, _, vert = tj.dvl_expected(traj, t_dvl)
    dvl_rep = {"n": int(len(t_dvl)), "n_invalid": int((~valid).sum()), "invalid_intervals_s": invalid_stats(t_dvl, valid),
               "expected_invalid_intervals_s": {n: invalid_stats(t_dvl, v) for n, v in exp_ok.items()},
               "agreement_with_expected_valid": {n: float(np.mean(v == valid)) for n, v in exp_ok.items()},
               "altitude_reported_m": ({"min": float(alt_valid.min()), "median": float(np.median(alt_valid)),
                                        "max": float(alt_valid.max())} if valid.any() else None),
               "altitude_over_vertical_expected": float(np.median(alt_valid / vert[valid])) if valid.any() else None,
               "velocity_residual_by_frame_hypothesis": res, "best_frame_hypothesis": best,
               "expected_noise_std_mps_at_0.4mps": rig.DVL_NOISE_VEL_MPS + 0.004 * rig.DVL_NOISE_VEL_PERCENT,
               "wall_lag_to_traj_msg_s_p95": float(np.percentile(lag_dvl, 95))}
    p_true, _, _, _ = traj.state_at(t_p)
    A = np.stack([p_true[:, 2], np.ones(len(t_p))], 1)
    coef, *_ = np.linalg.lstsq(A, pr[:, 1], rcond=None)
    press_rep = {"n": int(len(t_p)), "pa_per_m_depth": float(coef[0]), "expected_pa_per_m": rig.WATER_DENSITY_KGPM3 * rig.GRAVITY_MPS2,
                 "offset_pa": float(coef[1]), "resid_std_pa": float(np.std(pr[:, 1] - A @ coef)), "configured_noise_pa": rig.PRESSURE_NOISE_PA}
    gaps = np.diff(t_cam) if len(t_cam) > 1 else np.zeros(1)
    k_gap = int(np.argmax(gaps))
    cam_rep = {"pairs": len(pairs), "left_frames": int(len(s["img_l"])), "right_frames": int(len(s["img_r"])),
               "first_pair_sim_s": float(t_cam[0]) if len(t_cam) else None,
               "static_s_after_first_pair": float(tj.HOLD_START_S - t_cam[0]) if len(t_cam) else None,
               "max_pair_gap_s": float(gaps[k_gap]), "max_pair_gap_at_sim_s": float(t_cam[k_gap]) if len(t_cam) else None,
               "pair_gaps_gt_1s": int(np.sum(gaps > 1.0)),
               "lr_wall_diff_s_p95": float(np.percentile(lr_dt, 95)) if len(lr_dt) else None, "stamping": snap_info,
               "delivery_phase_mod_50ms_s": phase, "delivery_phase_concentration": conc, "cam_latency_applied_s": a.cam_latency_s,
               **info}
    rep = build_report(traj, tm, streams, cam_rep, dvl_rep, press_rep, imu_sf_report)
    rep["bench"] = {"left_topic": info["left_topic"], "right_topic": info["right_topic"], "imu_topic": "/sim/imu",
                    "pressure_topic": "/sim/pressure", "dvl_topic": "/sim/dvl_twist", "bag": str(a.out / "bench_bag"),
                    "bag_t0_s": rig.BAG_T0_S, "suggested_blackout": meta["suggested_blackout"], "imu_source": a.imu}
    (a.out / "sanity.json").write_text(json.dumps(rep, indent=1, default=float))
    (a.out / "convert_info.json").write_text(json.dumps(rep["bench"], indent=1))
    print_report(rep)


def print_report(rep: dict) -> None:
    print("\n==== sanity report ====")
    print(json.dumps(rep, indent=1, default=float))
    tm, d, c, p = rep["time_map"], rep["dvl"], rep["camera"], rep["pressure"]
    flags = []
    if tm["mode"] != "regular":
        flags.append(f"time map is '{tm['mode']}' (traj_state steps dropped or sim not stepping uniformly)")
    if tm["realtime_factor"] < 0.9:
        flags.append(f"realtime factor {tm['realtime_factor']:.2f} < 0.9")
    if c["pairs"] < 0.95 * min(c["left_frames"], c["right_frames"]):
        flags.append("more than 5% of the camera frames could not be paired")
    if c["static_s_after_first_pair"] is None or c["static_s_after_first_pair"] < 5.0:
        flags.append(f"vehicle holds still for only {c['static_s_after_first_pair']} s after the first stereo pair "
                     "(estimator initialises from rest; raise trajectory.HOLD_START_S)")
    if c["max_pair_gap_s"] > 1.0:
        flags.append(f"{c['pair_gaps_gt_1s']} stereo gaps > 1 s, longest {c['max_pair_gap_s']:.1f} s at sim "
                     f"{c['max_pair_gap_at_sim_s']:.1f} s (frames lost before the recorder)")
    if abs(p["pa_per_m_depth"] / p["expected_pa_per_m"] - 1.0) > 0.02:
        flags.append("pressure slope differs from rho*g by more than 2%")
    if d["best_frame_hypothesis"] != "dvl_frame":
        flags.append(f"DVL velocity matches the '{d['best_frame_hypothesis']}' hypothesis best, not the DVL frame")
    if d["n_invalid"] == 0:
        flags.append("no invalid DVL sample: bottom lock was never lost (the lock-loss test is void)")
    print("\nFLAGS:" if flags else "\nFLAGS: none")
    for f_ in flags:
        print("  -", f_)


if __name__ == "__main__":
    main()
