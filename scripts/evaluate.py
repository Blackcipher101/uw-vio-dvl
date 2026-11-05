"""Score TUM trajectories against the dataset's COLMAP ground truth.

Metrics (positions of the left camera):
  ate_se3_m   RMSE after rigid alignment (scale must already be right: stereo / VIO)
  ate_sim3_m  RMSE after similarity alignment (fair to monocular methods), plus the scale
  rpe_trans_pct / rpe_rot_deg_per_m  over all pose pairs RPE_DIST_M apart along the GT path
  coverage    fraction of GT stamps that have an estimate within ASSOC_TOL_S

The GT file does not state its pose convention. Both T_world_cam and its inverse are
tried; the one with the lower Sim(3) ATE is used and reported as ``gt_convention``.

--hand-eye: the GT tracks another rigid body (e.g. motion-capture markers on the vehicle,
T_world_body). The fixed T_body_cam is estimated jointly with the world alignment (see
hand_eye), the GT is moved onto the camera (T_world_body @ T_body_cam) and scored as above.
Needs a metric estimate. The fit absorbs a constant offset, so it slightly flatters ATE.

Usage: evaluate.py <gt.txt> <est.tum> [<est.tum> ...] [--hand-eye] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geometry import mat_to_quat, quat_to_mat, umeyama

ASSOC_TOL_S = 0.02  # frames are ~80 ms apart; GT stamps are image stamps
RPE_DIST_M = 5.0
RPE_MAX_GAP_S = 1.0  # COLMAP GT skips a few frames; motion capture drops out for seconds


def load_tum(path: Path) -> np.ndarray:
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.size == 0:
        return np.zeros((0, 8))
    if a.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns (t x y z qx qy qz qw)")
    a = a[np.all(np.isfinite(a), axis=1)]
    return a[np.argsort(a[:, 0])]


def to_mats(tum: np.ndarray) -> np.ndarray:
    m = np.tile(np.eye(4), (len(tum), 1, 1))
    for i, row in enumerate(tum):
        m[i, :3, :3] = quat_to_mat(*row[4:8])
        m[i, :3, 3] = row[1:4]
    return m


def associate(t_gt: np.ndarray, t_est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.clip(np.searchsorted(t_est, t_gt), 1, len(t_est) - 1)
    j = np.where(np.abs(t_est[j - 1] - t_gt) < np.abs(t_est[j] - t_gt), j - 1, j)
    ok = np.abs(t_est[j] - t_gt) <= ASSOC_TOL_S
    return np.nonzero(ok)[0], j[ok]


def rot_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def rpe(gt: np.ndarray, est: np.ndarray, scale: float, t: np.ndarray) -> tuple[float, float]:
    """Mean translational error (% of segment) and rotation drift (deg/m) over RPE_DIST_M segments.

    Segments spanning a GT dropout (stamp gap > RPE_MAX_GAP_S, e.g. motion capture losing the
    vehicle) are skipped: the GT path across the gap is a jump, not the travelled distance."""
    p = gt[:, :3, 3]
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    gap_before = np.concatenate([[0], np.cumsum(np.diff(t) > RPE_MAX_GAP_S)])  # gaps up to index
    te, re = [], []
    for i in range(len(gt)):
        k = np.searchsorted(dist, dist[i] + RPE_DIST_M)
        if k >= len(gt):
            break
        if gap_before[k] != gap_before[i]:
            continue
        d_gt = np.linalg.inv(gt[i]) @ gt[k]
        d_est = np.linalg.inv(est[i]) @ est[k]
        d_est[:3, 3] *= scale
        err = np.linalg.inv(d_gt) @ d_est
        seg = dist[k] - dist[i]
        te.append(np.linalg.norm(err[:3, 3]) / seg)
        re.append(rot_angle_deg(err[:3, :3]) / seg)
    if not te:
        return float("nan"), float("nan")
    return 100 * float(np.mean(te)), float(np.mean(re))


def score(gt_tum: np.ndarray, est_tum: np.ndarray) -> dict:
    if len(est_tum) < 2:
        return {"matched": 0, "error": f"estimate has {len(est_tum)} poses"}
    ig, ie = associate(gt_tum[:, 0], est_tum[:, 0])
    if len(ig) < 10:
        return {"matched": int(len(ig)), "error": "too few associated poses"}
    est = to_mats(est_tum[ie])
    best = None
    for conv in ("T_world_cam", "T_cam_world"):
        gt = to_mats(gt_tum[ig])
        if conv == "T_cam_world":
            gt = np.linalg.inv(gt)
        s, r, t = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale=True)
        e = np.linalg.norm((s * (r @ est[:, :3, 3].T)).T + t - gt[:, :3, 3], axis=1)
        cand = (float(np.sqrt(np.mean(e ** 2))), conv, gt, s)
        if best is None or cand[0] < best[0]:
            best = cand
    ate_sim3, conv, gt, s = best
    _, r1, t1 = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale=False)
    e1 = np.linalg.norm((r1 @ est[:, :3, 3].T).T + t1 - gt[:, :3, 3], axis=1)
    rpe_t, rpe_r = rpe(gt, est, s, gt_tum[ig, 0])
    path = float(np.sum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)))
    return {"matched": int(len(ig)), "coverage": float(len(ig) / len(gt_tum)),
            "gt_convention": conv, "gt_path_m": path,
            "ate_se3_m": float(np.sqrt(np.mean(e1 ** 2))), "ate_se3_max_m": float(e1.max()),
            "ate_sim3_m": ate_sim3, "sim3_scale": float(s),
            f"rpe_trans_pct_{RPE_DIST_M:g}m": rpe_t, f"rpe_rot_deg_per_m_{RPE_DIST_M:g}m": rpe_r}


HAND_EYE_MAX_ITERS = 100
HAND_EYE_TOL_M = 1e-7


def log_so3(r: np.ndarray) -> np.ndarray:
    ang = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    if ang < 1e-9:
        return np.zeros(3)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return ang / (2 * np.sin(ang)) * w


def project_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt


def hand_eye(gt_tum: np.ndarray, est_tum: np.ndarray, first_s: float = 0.0) -> tuple[np.ndarray, dict]:
    """T_body_cam for a GT that tracks another body, from a metric estimate.

    Relative-motion hand-eye (AX = XB) is degenerate when the vehicle mostly yaws (rotation about
    one axis leaves R_X free about it). Instead, alternate: world alignment W (Umeyama, no scale)
    of estimated camera positions onto GT camera positions p_body + R_body t_X, then t_X by least
    squares; R_X is the chordal mean of R_body^T R_W R_est over all poses."""
    ig, ie = associate(gt_tum[:, 0], est_tum[:, 0])
    if first_s > 0:  # fit on the start only, so later drift is not absorbed into the lever arm
        keep = gt_tum[ig, 0] <= gt_tum[ig[0], 0] + first_s
        ig, ie = ig[keep], ie[keep]
    if len(ig) < 10:
        raise ValueError("hand-eye: too few associated poses")
    G, E = to_mats(gt_tum[ig]), to_mats(est_tum[ie])
    Rg, pg, pe = G[:, :3, :3], G[:, :3, 3], E[:, :3, 3]
    tx = np.zeros(3)
    for _ in range(HAND_EYE_MAX_ITERS):
        _, rw, tw = umeyama(pe, pg + Rg @ tx, with_scale=False)
        # R_g t_X = R_W p_e + t_W - p_g, stacked over all poses
        rhs = (rw @ pe.T).T + tw - pg
        tx_new, *_ = np.linalg.lstsq(Rg.reshape(-1, 3), rhs.reshape(-1), rcond=None)
        done = np.linalg.norm(tx_new - tx) < HAND_EYE_TOL_M
        tx = tx_new
        if done:
            break
    R = project_so3(np.einsum("nji,jk,nkl->il", Rg, rw, E[:, :3, :3]))  # mean of R_g^T R_W R_e
    rot_res = np.degrees(np.median([np.linalg.norm(log_so3((rg @ R).T @ rw @ re))
                                    for rg, re in zip(Rg, E[:, :3, :3])]))
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, tx
    info = {"hand_eye_poses": int(len(ig)), "hand_eye_t_m": [round(float(x), 4) for x in tx],
            "hand_eye_rot_deg": round(float(np.degrees(np.linalg.norm(log_so3(R)))), 3),
            "hand_eye_rot_residual_deg": round(float(rot_res), 3)}
    return X, info


def move_gt(gt_tum: np.ndarray, X: np.ndarray) -> np.ndarray:
    out = gt_tum.copy()
    for n, T in enumerate(to_mats(gt_tum)):
        T = T @ X
        out[n, 1:4] = T[:3, 3]
        out[n, 4:8] = mat_to_quat(T[:3, :3])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("est", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--hand-eye", action="store_true", help="GT tracks another body; estimate T_body_cam")
    ap.add_argument("--hand-eye-first-s", type=float, default=0.0,
                    help="fit T_body_cam on the first S seconds only (0 = whole run)")
    a = ap.parse_args()
    gt = load_tum(a.gt)
    out = {}
    for p in a.est:
        est = load_tum(p)
        if a.hand_eye:
            X, info = hand_eye(gt, est, a.hand_eye_first_s)
            res = {**score(move_gt(gt, X), est), **info}
        else:
            res = score(gt, est)
        out[p.name] = res
        print(p.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}),
              flush=True)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
