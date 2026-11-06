"""diag.py IMU-vs-GT check on consistent synthetic data: ~zero residual and best offset 0 (and it flags a shifted IMU)."""
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import convert
import diag
import gen_calib
import rig
import trajectory as tj


def _gt(traj):
    t = np.arange(tj.HOLD_START_S + 1.0, tj.HOLD_START_S + 60.0, 0.05)  # on the move: offsets are observable
    rows = convert.gt_leftcam(traj, t)
    T = np.array(yaml.safe_load(gen_calib.imucam_yaml())["cam0"]["T_cam_imu"], float)
    return diag.GtBody(rows[:, 0], rows[:, 1:4], Rotation.from_quat(rows[:, 4:8]), T)


def test_imu_consistent_with_gt(traj):
    gt = _gt(traj)
    imu = convert.synth_imu(traj, noise=False)
    r = diag.imu_window_residuals(gt, imu["t"] + rig.BAG_T0_S, imu["gyro"], imu["acc"], offsets=np.array([-0.05, 0.0, 0.05]), n_win=10)
    assert r[0.0][0] < 0.1 and r[0.0][2] < 0.02
    assert r[0.0][0] < r[0.05][0] and r[0.0][0] < r[-0.05][0]
