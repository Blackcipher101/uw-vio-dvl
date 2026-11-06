"""End-to-end convert.py on a synthetic raw bag (needs a sourced ROS 2 environment with stonefish_ros2 msgs).

The bag imitates stonefish_ros2: wall-clock header stamps that run at 0.97x sim time with jitter, camera frames delivered
12 ms after rendering with left/right skew, DVL altitude = -1 while bottom lock is lost, pose-exact traj_state at 200 Hz."""
import json
import subprocess
import sys

import numpy as np
import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("stonefish_ros2.msg")

from builtin_interfaces.msg import Time  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402
from sensor_msgs.msg import CompressedImage, FluidPressure, Image  # noqa: E402
from stonefish_ros2.msg import DVL  # noqa: E402

import convert  # noqa: E402
import rig  # noqa: E402

from conftest import SIM  # noqa: E402

T0, T1 = 10.0, 25.0
RTF, LATENCY = 0.97, 0.012


def _stamp(w):
    return Time(sec=int(w), nanosec=int(round((w - int(w)) * 1e9)))


def _write_raw_bag(path, traj):
    rng = np.random.default_rng(0)
    w_of = lambda t: 5000.0 + (t - T0) / RTF  # noqa: E731
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    for idx, (name, typ) in enumerate((("/sim/traj_state", "nav_msgs/msg/Odometry"), ("/sim/dvl", "stonefish_ros2/msg/DVL"),
                                       ("/sim/pressure", "sensor_msgs/msg/FluidPressure"),
                                       ("/sim/cam_left/image_color", "sensor_msgs/msg/Image"),
                                       ("/sim/cam_right/image_color", "sensor_msgs/msg/Image"))):
        writer.create_topic(convert.topic_metadata(idx, name, typ))
    events = []
    for t in np.arange(T0, T1, 0.005):
        p, R, _, _ = traj.state_at(np.array([t]))
        m = Odometry()
        w = w_of(t) + 2e-4 * rng.standard_normal()
        m.header.stamp = _stamp(w)
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, p[0])
        q = rig.quat_xyzw_from_rot(R[0])
        m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w = map(float, q)
        events.append((w, "/sim/traj_state", m))
    for t in T0 + 0.2 * np.arange(int((T1 - T0) / 0.2)):
        exp = convert.dvl_expected_velocity(traj, np.array([t]))["dvl_frame"][0]
        m = DVL()
        w = w_of(t) - 1e-4
        m.header.stamp = _stamp(w)
        m.velocity.x, m.velocity.y, m.velocity.z = map(float, exp)
        m.altitude = -1.0 if 18.0 <= round(float(t), 3) < 20.0 else 3.0
        events.append((w, "/sim/dvl", m))
    for t in np.arange(T0, T1, 0.1):
        _, _, _, _ = traj.state_at(np.array([t]))
        m = FluidPressure()
        w = w_of(t) - 1e-4
        m.header.stamp = _stamp(w)
        m.fluid_pressure = float(rig.WATER_DENSITY_KGPM3 * rig.GRAVITY_MPS2 * traj.state_at(np.array([t]))[0][0, 2])
        events.append((w, "/sim/pressure", m))
    for t in np.arange(T0 + 0.05, T1, 0.05):
        for side, name in enumerate(("/sim/cam_left/image_color", "/sim/cam_right/image_color")):
            m = Image()
            m.height, m.width, m.encoding, m.step = 24, 32, "rgb8", 96
            m.data = rng.integers(0, 255, 24 * 96, dtype=np.uint8)
            w = w_of(t) + LATENCY + 0.006 * side
            m.header.stamp = _stamp(w)
            events.append((w, name, m))
    for w, name, m in sorted(events, key=lambda e: e[0]):
        writer.write(name, serialize_message(m), int(w * 1e9))
    del writer


def test_convert_end_to_end(tmp_path, traj):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_raw_bag(raw, traj)
    r = subprocess.run([sys.executable, str(SIM / "convert.py"), "--bag", str(raw), "--out", str(out),
                        "--cam-latency-s", str(LATENCY), "--img-quality", "90"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads((out / "sanity.json").read_text())
    assert rep["time_map"]["mode"] == "regular"
    assert abs(rep["time_map"]["realtime_factor"] - RTF) < 0.01
    assert rep["dvl"]["best_frame_hypothesis"] == "dvl_frame"
    assert rep["dvl"]["velocity_residual_by_frame_hypothesis"]["dvl_frame"]["rms_mps"] < 1e-6
    assert rep["dvl"]["n_invalid"] == 10  # 2 s at 5 Hz
    assert abs(rep["pressure"]["pa_per_m_depth"] - rig.WATER_DENSITY_KGPM3 * rig.GRAVITY_MPS2) < 1.0

    rows = [ln.split(",") for ln in (out / "dvl.csv").read_text().splitlines()[1:]]
    t = np.array([float(x[0]) for x in rows]) - rig.BAG_T0_S
    assert np.all(np.diff(t) > 0) and abs(t[0] - T0) < 0.003
    valid = np.array([int(x[4]) for x in rows])
    assert set(t[valid == 0].round(1)) == {18.0, 18.2, 18.4, 18.6, 18.8, 19.0, 19.2, 19.4, 19.6, 19.8}

    gt = np.loadtxt(out / "gt_leftcam.tum")
    assert gt.shape[1] == 8 and len(gt) >= 290
    t_gt = gt[:, 0] - rig.BAG_T0_S
    np.testing.assert_allclose(t_gt, T0 + 0.05 + 0.05 * np.round((t_gt - T0 - 0.05) / 0.05), atol=0.0015)  # on the render grid

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(out / "bench_bag"), storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    info = json.loads((out / "convert_info.json").read_text())
    assert info["left_topic"] in topics and info["right_topic"] in topics and "/sim/imu" in topics
    last, counts, imu_t, img_t = -1, {}, [], {}
    while reader.has_next():
        name, raw_msg, ts = reader.read_next()
        assert ts >= last  # written in time order
        last = ts
        counts[name] = counts.get(name, 0) + 1
        if name == "/sim/imu":
            imu_t.append(ts)
        if name in (info["left_topic"], info["right_topic"]):
            img_t.setdefault(name, []).append(ts)
    assert counts[info["left_topic"]] == counts[info["right_topic"]] == len(gt)
    assert np.allclose(np.diff(imu_t), 5e6, atol=2)  # exact 200 Hz grid [ns]
    np.testing.assert_array_equal(img_t[info["left_topic"]], img_t[info["right_topic"]])  # pairs share stamps
