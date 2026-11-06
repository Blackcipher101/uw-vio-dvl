#!/usr/bin/env python3
"""Write sim/calib/*.yaml (Kalibr format, as parsed by uw_ct_vio/app/run_bag.cpp) from rig.py.

  camchain.yaml        --intrinsics   cam0/cam1: pinhole, radtan (zero distortion), resolution, T_cn_cnm1 on cam1
  imucam.yaml          --imucam       cam0/cam1: T_cam_imu = inv(T_B_C) (IMU = body B), timeshift_cam_imu
  imu.yaml             --imu-yaml     top-level noise densities / random walks (run_bag reads them at the top level)
  dvl_extrinsic.yaml   --dvl-extrinsic  T_B_D (DVL frame in the body = IMU frame)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import rig

HERE = Path(__file__).resolve().parent
CAM_TOPICS = ("/sim/cam0/compressed", "/sim/cam1/compressed")


def mat(name: str, m: np.ndarray, indent: int) -> str:
    pad = " " * indent
    clean = np.where(np.abs(m) < 1e-12, 0.0, m)  # drop 1e-17 trigonometric residue
    return f"{pad}{name}:\n" + "".join(f"{pad}  - [" + ", ".join(f"{x:.12g}" for x in row) + "]\n" for row in clean)


def camchain_yaml(cam_topics=CAM_TOPICS) -> str:
    fx, fy, cx, cy = rig.intrinsics()
    rg = rig.build_rig()
    out = ""
    for i in (0, 1):
        out += f"cam{i}:\n"
        if i == 1:
            out += mat("T_cn_cnm1", rg.T_C1_C0, 2)
        out += f"  cam_overlaps: [{1 - i}]\n  camera_model: pinhole\n  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        out += f"  distortion_model: radtan\n  intrinsics: [{fx:.9g}, {fy:.9g}, {cx:.9g}, {cy:.9g}]\n"
        out += f"  resolution: [{rig.IMG_W}, {rig.IMG_H}]\n  rostopic: {cam_topics[i]}\n"
    return out


def imucam_yaml(cam_topics=CAM_TOPICS) -> str:
    rg = rig.build_rig()
    out = ""
    for i, T_B_C in enumerate((rg.T_B_C0, rg.T_B_C1)):
        out += f"cam{i}:\n" + mat("T_cam_imu", rig.inv_t(T_B_C), 2)
        out += f"  rostopic: {cam_topics[i]}\n  timeshift_cam_imu: 0.0\n"
    return out


def imu_yaml() -> str:
    return (f"# synthetic IMU, Kalibr continuous-time noise model (sim/rig.py); rate {rig.IMU_RATE_HZ:g} Hz\n"
            f"accelerometer_noise_density: {rig.ACC_NOISE_DENSITY}\naccelerometer_random_walk: {rig.ACC_RANDOM_WALK}\n"
            f"gyroscope_noise_density: {rig.GYR_NOISE_DENSITY}\ngyroscope_random_walk: {rig.GYR_RANDOM_WALK}\n"
            f"rostopic: /sim/imu\nupdate_rate: {rig.IMU_RATE_HZ:g}\n")


def dvl_yaml() -> str:
    return mat("T_B_D", rig.build_rig().T_B_D, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "calib")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, text in (("camchain.yaml", camchain_yaml()), ("imucam.yaml", imucam_yaml()), ("imu.yaml", imu_yaml()),
                       ("dvl_extrinsic.yaml", dvl_yaml())):
        (a.out / name).write_text(text)
        print("wrote", a.out / name)


if __name__ == "__main__":
    main()
