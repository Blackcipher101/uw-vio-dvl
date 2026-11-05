"""Small SE(3)/Sim(3) helpers shared by the runners and the evaluator (numpy only)."""
from __future__ import annotations

import numpy as np


def quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError(f"invalid quaternion ({x},{y},{z},{w})")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(r: np.ndarray) -> list[float]:
    """Rotation matrix -> [x, y, z, w]."""
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares (s, R, t) with dst ~= s * R @ src + t. src, dst: (N, 3), N >= 3."""
    if len(src) < 3:
        raise ValueError("umeyama needs at least 3 points")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2, 2] = -1
    r = u @ sgn @ vt
    var_s = (xs ** 2).sum() / len(src)
    if with_scale:
        if var_s < 1e-12:
            raise ValueError("degenerate source trajectory (no motion)")
        s = float(np.trace(np.diag(d) @ sgn) / var_s)
    else:
        s = 1.0
    t = mu_d - s * r @ mu_s
    return s, r, t
