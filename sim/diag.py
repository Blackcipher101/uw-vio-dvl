#!/usr/bin/env python3
"""Independent checks of the converted bench data against the exact ground truth (run on ws1).

  python3 diag.py --bench ~/uwvio/sim_runs/clear/bench [--pairs 300] [--max-gap-s 0.2]

Reads bench/{bench_bag, gt_leftcam.tum, convert_info.json} and ../../sim/calib. Prints:
  IMU  1 s windows: gyro-integrated rotation, and gravity+specific-force double integration, vs GT body poses
       (GT body = GT left camera * T_cam_imu from imucam.yaml); sweep of an IMU time offset.
  CAM  stereo: depth from left-right disparity (fx, baseline from camchain.yaml) of KLT-tracked corners, then consecutive
       left frames: relative rotation from the essential matrix vs GT rotation and vs integrated gyro, reprojection error of
       the triangulated points moved with the GT relative pose, and sweeps of a camera time offset.
Rotation errors are angles of R_est^T R_ref in degrees. Needs numpy, scipy, cv2 and a sourced ROS 2 (rosbag2_py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

import rig

HERE = Path(__file__).resolve().parent
OFFSETS_S = np.arange(-0.10, 0.1001, 0.01)


def load_tum(path):
    a = np.loadtxt(path, comments="#")
    return a[:, 0], a[:, 1:4], Rotation.from_quat(a[:, 4:8])


class GtBody:
    """Body (IMU) pose at arbitrary times from GT left-camera poses and the yaml T_cam_imu."""

    def __init__(self, t, p_c, rot_c, T_cam_imu):
        T_imu_cam = rig.inv_t(T_cam_imu)
        R_cb = Rotation.from_matrix(T_cam_imu[:3, :3])  # body -> camera
        self.t = t
        self.rot = rot_c * R_cb  # R_WB = R_WC R_CB
        self.p = p_c + rot_c.apply(T_cam_imu[:3, 3])  # IMU origin in the camera frame
        self.slerp = Slerp(t, self.rot)

    def at(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])
        return np.stack([np.interp(tq, self.t, self.p[:, i]) for i in range(3)], -1), self.slerp(tq)

    def vel(self, tq, h=0.05):
        return (self.at(tq + h)[0] - self.at(tq - h)[0]) / (2 * h)


def _ang(R):
    return np.degrees(np.linalg.norm(R.as_rotvec(), axis=-1))


def integrate_imu(t, gyro, acc, t0, t1, R0, v0, p0, g=np.array([0.0, 0.0, rig.GRAVITY_MPS2])):
    """Euler/exp-map integration of the raw IMU between t0 and t1 starting from the given state."""
    m = (t >= t0) & (t <= t1)
    tt, w, f = t[m], gyro[m], acc[m]
    R, v, p = R0, v0.copy(), p0.copy()
    R_hist = R0
    for k in range(len(tt) - 1):
        dt = tt[k + 1] - tt[k]
        a = R.apply(0.5 * (f[k] + f[k + 1])) + g
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        R = R * Rotation.from_rotvec(0.5 * (w[k] + w[k + 1]) * dt)
    return R, v, p


def imu_window_residuals(gt: GtBody, t, gyro, acc, offsets=OFFSETS_S, win_s=1.0, n_win=60):
    """Median residuals per IMU time offset: rotation [deg], velocity [m/s], position [m] after win_s."""
    starts = np.linspace(gt.t[0] + 2.0, gt.t[-1] - win_s - 2.0, n_win)
    out = {}
    for d in offsets:
        er, ev, ep = [], [], []
        for s in starts:
            p0, R0 = gt.at(np.array([s]))
            v0 = gt.vel(np.array([s]))[0]
            R, v, p = integrate_imu(t, gyro, acc, s + d, s + d + win_s, R0[0], v0, p0[0])
            p1, R1 = gt.at(np.array([s + win_s]))
            v1 = gt.vel(np.array([s + win_s]))[0]
            er.append(_ang(R.inv() * R1[0]))
            ev.append(np.linalg.norm(v - v1))
            ep.append(np.linalg.norm(p - p1[0]))
        out[round(float(d), 3)] = (float(np.median(er)), float(np.median(ev)), float(np.median(ep)))
    return out


def read_bench(bench: Path, n_pairs: int, max_gap_s: float):
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    info = json.loads((bench / "convert_info.json").read_text())
    reader = rosbag2_py.SequentialReader()
    meta = (bench / "bench_bag" / "metadata.yaml").read_text()
    sid = next((ln.split(":")[1].strip() for ln in meta.splitlines() if "storage_identifier" in ln), "sqlite3")
    reader.open(rosbag2_py.StorageOptions(uri=str(bench / "bench_bag"), storage_id=sid), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = {n: get_message(t) for n, t in types.items()}
    imu, left, right = [], {}, {}
    while reader.has_next():
        name, raw, _ = reader.read_next()
        if name == info["imu_topic"]:
            m = deserialize_message(raw, cls[name])
            imu.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, *[getattr(m.angular_velocity, c) for c in "xyz"],
                        *[getattr(m.linear_acceleration, c) for c in "xyz"]))
        elif name in (info["left_topic"], info["right_topic"]):
            m = deserialize_message(raw, cls[name])
            t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
            img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_GRAYSCALE) if hasattr(m, "format") else \
                np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width]
            (left if name == info["left_topic"] else right)[round(t, 6)] = img
    imu = np.asarray(imu)
    ts = sorted(set(left) & set(right))
    idx = [i for i in range(len(ts) - 1) if ts[i + 1] - ts[i] <= max_gap_s]
    idx = idx[:: max(1, len(idx) // n_pairs)][:n_pairs]
    return imu, [(ts[i], ts[i + 1], left[ts[i]], right[ts[i]], left[ts[i + 1]]) for i in idx], ts


def cam_checks(gt: GtBody, imu, pairs, K, baseline, T_cam_imu, offsets=OFFSETS_S):
    import cv2

    R_C_B = Rotation.from_matrix(T_cam_imu[:3, :3])
    res = {"rot_img_vs_gt": [], "rot_gyro_vs_gt": [], "rot_img_vs_gyro": {round(float(d), 3): [] for d in offsets},
           "reproj_px": {round(float(d), 3): [] for d in offsets}, "stereo_disp_px": [], "n_corners": []}
    for t0, t1, L0, R0img, L1 in pairs:
        pts = cv2.goodFeaturesToTrack(L0, 400, 0.01, 8)
        if pts is None or len(pts) < 30:
            continue
        pr, st, _ = cv2.calcOpticalFlowPyrLK(L0, R0img, pts, None, winSize=(21, 21), maxLevel=3)
        disp = (pts - pr)[:, 0, 0]
        dy = np.abs((pts - pr)[:, 0, 1])
        ok_s = (st[:, 0] == 1) & (disp > 1.0) & (dy < 1.0)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(L0, L1, pts, None, winSize=(21, 21), maxLevel=3)
        ok = ok_s & (st1[:, 0] == 1)
        if ok.sum() < 30:
            continue
        res["stereo_disp_px"].append(float(np.median(disp[ok])))
        res["n_corners"].append(int(ok.sum()))
        a, b = pts[ok, 0], p1[ok, 0]
        Z = K[0, 0] * baseline / disp[ok]
        keep = (Z > 0.8) & (Z < 20.0)
        a, b, Z = a[keep], b[keep], Z[keep]
        if len(a) < 20:
            continue
        X = np.stack([(a[:, 0] - K[0, 2]) / K[0, 0] * Z, (a[:, 1] - K[1, 2]) / K[1, 1] * Z, Z], 1)
        E, mask = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            continue
        _, Rr, _, _ = cv2.recoverPose(E, a, b, K, mask=mask)
        R_img = Rotation.from_matrix(Rr.T)  # recoverPose: x1 = R x0 + t, so camera 1 orientation in camera 0 is R^T
        # reference relative rotation of the camera from t0 to t1 (cam0 <- cam1): R_C0^T R_C1
        for d in offsets:
            pa, Ra = gt.at(np.array([t0 + d]))
            pb, Rb = gt.at(np.array([t1 + d]))
            Rc_a, Rc_b = Ra[0] * R_C_B.inv(), Rb[0] * R_C_B.inv()
            rel = Rc_a.inv() * Rc_b
            pc_a = pa[0] + Ra[0].apply(rig.inv_t(T_cam_imu)[:3, 3])
            pc_b = pb[0] + Rb[0].apply(rig.inv_t(T_cam_imu)[:3, 3])
            t_rel = Rc_b.inv().apply(pc_a - pc_b)  # camera-a origin expressed in camera b
            Xb = Rc_b.inv().apply(Rc_a.apply(X)) + t_rel
            uv = (K @ Xb.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            res["reproj_px"][round(float(d), 3)].append(float(np.median(np.linalg.norm(uv - b, axis=1))))
            if abs(d) < 1e-9:
                res["rot_img_vs_gt"].append(float(_ang(R_img.inv() * rel)))
            # gyro-integrated camera rotation over [t0+d, t1+d]
            m = (imu[:, 0] >= t0 + d) & (imu[:, 0] <= t1 + d)
            Rb_int = Rotation.identity()
            tt, w = imu[m, 0], imu[m, 1:4]
            for k in range(len(tt) - 1):
                Rb_int = Rb_int * Rotation.from_rotvec(0.5 * (w[k] + w[k + 1]) * (tt[k + 1] - tt[k]))
            rel_gyro = R_C_B * Rb_int * R_C_B.inv()
            res["rot_img_vs_gyro"][round(float(d), 3)].append(float(_ang(R_img.inv() * rel_gyro)))
            if abs(d) < 1e-9:
                res["rot_gyro_vs_gt"].append(float(_ang(rel_gyro.inv() * rel)))
    return res


def summarise(res):
    med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
    out = {k: med(v) for k, v in res.items() if isinstance(v, list)}
    for k in ("reproj_px", "rot_img_vs_gyro"):
        out[k + "_by_offset_s"] = {d: round(med(v), 4) for d, v in res[k].items()}
        best = min(res[k], key=lambda d: med(res[k][d]))
        out[k + "_best_offset_s"] = best
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--max-gap-s", type=float, default=0.2)
    a = ap.parse_args()
    cc = yaml.safe_load((HERE / "calib" / "camchain.yaml").read_text())
    ic = yaml.safe_load((HERE / "calib" / "imucam.yaml").read_text())
    fx, fy, cx, cy = cc["cam0"]["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    baseline = -np.array(cc["cam1"]["T_cn_cnm1"])[0, 3]
    T_cam_imu = np.array(ic["cam0"]["T_cam_imu"], float)
    t, p, rot = load_tum(a.bench / "gt_leftcam.tum")
    gt = GtBody(t, p, rot, T_cam_imu)
    imu, pairs, stamps = read_bench(a.bench, a.pairs, a.max_gap_s)
    print(f"GT poses {len(t)}, IMU samples {len(imu)}, image pairs {len(stamps)}, using {len(pairs)} consecutive pairs", flush=True)
    r = imu_window_residuals(gt, imu[:, 0], imu[:, 1:4], imu[:, 4:7])
    best = min(r, key=lambda d: r[d][0])
    print("IMU windows (1 s), offset s -> median rot deg / vel m/s / pos m")
    for d, v in r.items():
        print(f"  {d:+.2f}  {v[0]:.4f}  {v[1]:.4f}  {v[2]:.4f}")
    print(f"IMU best offset by rotation: {best:+.3f} s")
    s = summarise(cam_checks(gt, imu, pairs, K, baseline, T_cam_imu))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
