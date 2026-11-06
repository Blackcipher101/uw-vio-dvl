"""Frames: the yamls, the scenario XML and the ground-truth export describe the same geometry."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import convert
import gen_calib
import rig

from conftest import SIM


def _scn_origin(sensor_name: str) -> np.ndarray:
    root = ET.parse(SIM / "scenarios" / "seabed_clear.scn").getroot()
    for s in root.iter("sensor"):
        if s.get("name") == sensor_name:
            o = s.find("origin")
            xyz = [float(x) for x in o.get("xyz").split()]
            r, p, y = [float(x) for x in o.get("rpy").split()]
            return rig.make_t(rig.rot_from_rpy(r, p, y), np.array(xyz))
    raise KeyError(sensor_name)


def _yaml_mat(rows) -> np.ndarray:
    return np.array(rows, dtype=float)


def test_rpy_roundtrip():
    rg = rig.build_rig()
    for T in (rg.T_B_C0, rg.T_B_C1, rg.T_B_D):
        np.testing.assert_allclose(rig.rot_from_rpy(*rig.rpy_from_rot(T[:3, :3])), T[:3, :3], atol=1e-12)


def test_scenario_origins_match_yaml():
    cc = yaml.safe_load(gen_calib.camchain_yaml())
    ic = yaml.safe_load(gen_calib.imucam_yaml())
    dv = yaml.safe_load(gen_calib.dvl_yaml())
    T_B_C0, T_B_C1, T_B_D = _scn_origin("CamLeft"), _scn_origin("CamRight"), _scn_origin("DVL")
    np.testing.assert_allclose(rig.inv_t(T_B_C0), _yaml_mat(ic["cam0"]["T_cam_imu"]), atol=1e-9)
    np.testing.assert_allclose(rig.inv_t(T_B_C1), _yaml_mat(ic["cam1"]["T_cam_imu"]), atol=1e-9)
    np.testing.assert_allclose(T_B_D, _yaml_mat(dv["T_B_D"]), atol=1e-9)
    # Kalibr: cam1.T_cn_cnm1 maps cam0 points into cam1
    np.testing.assert_allclose(rig.inv_t(T_B_C1) @ T_B_C0, _yaml_mat(cc["cam1"]["T_cn_cnm1"]), atol=1e-9)
    # stereo geometry as run_bag requires it: right camera to the right of the left one, baseline 0.12 m, parallel axes
    rel = rig.inv_t(T_B_C0) @ T_B_C1
    np.testing.assert_allclose(rel[:3, 3], [rig.BASELINE_M, 0, 0], atol=1e-9)
    np.testing.assert_allclose(rel[:3, :3], np.eye(3), atol=1e-9)


def test_camera_axes_and_tilt():
    T = rig.build_rig().T_B_C0
    R = T[:3, :3]
    np.testing.assert_allclose(R[:, 0], [0, 1, 0], atol=1e-12)  # image x = body right
    tilt = math.radians(rig.CAM_TILT_DOWN_DEG)
    np.testing.assert_allclose(R[:, 2], [math.cos(tilt), 0, math.sin(tilt)], atol=1e-12)  # optical axis forward, down-tilted
    assert np.linalg.det(R) > 0.999999


def test_intrinsics_match_scenario_fov():
    root = ET.parse(SIM / "scenarios" / "seabed_clear.scn").getroot()
    cam = next(s for s in root.iter("sensor") if s.get("name") == "CamLeft").find("specs")
    w, h, fov = int(cam.get("resolution_x")), int(cam.get("resolution_y")), float(cam.get("horizontal_fov"))
    cc = yaml.safe_load(gen_calib.camchain_yaml())
    fx, fy, cx, cy = cc["cam0"]["intrinsics"]
    assert cc["cam0"]["resolution"] == [w, h]
    assert math.isclose(fx, (w / 2) / math.tan(math.radians(fov) / 2), rel_tol=1e-8)
    assert math.isclose(fx, fy) and (cx, cy) == ((w - 1) / 2, (h - 1) / 2)
    # a point on the optical axis projects to the principal point; a point to the right lands at larger u
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    for X, du in (([0, 0, 5.0], 0.0), ([1.0, 0, 5.0], fx / 5.0)):
        u = K @ np.array(X)
        assert math.isclose(u[0] / u[2] - cx, du, abs_tol=1e-9)


def test_gt_leftcam_matches_body_pose_and_yaml(traj):
    ic = yaml.safe_load(gen_calib.imucam_yaml())
    T_C0_B = _yaml_mat(ic["cam0"]["T_cam_imu"])
    t = np.array([5.0, 33.3, 120.07, 250.0])
    rows = convert.gt_leftcam(traj, t)
    p, R, _, _ = traj.state_at(t)
    for k in range(len(t)):
        T_W_C0 = rig.make_t(rig.rot_from_quat_xyzw(rows[k, 4:8]), rows[k, 1:4])
        T_W_B = T_W_C0 @ T_C0_B  # body (= IMU) pose recovered through the yaml extrinsic
        np.testing.assert_allclose(T_W_B[:3, :3], R[k], atol=1e-9)
        np.testing.assert_allclose(T_W_B[:3, 3], p[k], atol=1e-9)
        assert math.isclose(rows[k, 0], rig.BAG_T0_S + t[k])


def test_dvl_beams_point_down_in_world():
    T = rig.build_rig().T_B_D
    beam_axis_B = T[:3, :3] @ np.array([0, 0, -1.0])  # Stonefish default: beams along the sensor -z
    np.testing.assert_allclose(beam_axis_B, [0, 0, 1.0], atol=1e-12)  # body +z = down for a level FRD body
