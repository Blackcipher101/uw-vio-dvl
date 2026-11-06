"""Render the camera-blackout comparison video for a converted Stonefish run.

Left: the simulated left camera (dimmed, with a banner, while the estimator is blind). Right: top-down map of the
ground-truth left camera and of two estimates (vision-only and vision + DVL) through the same camera blackout, plus
their position error over time.

Both estimates are aligned to the ground truth (SE(3), positions) using ONLY the samples before the blackout starts,
so the drift during the blackout is shown as the robot would experience it, not hidden by a whole-run fit.

    python3 sim/make_video.py --bench <run>/bench --results <runs>/results --variant clear --out blackout.mp4

Needs ROS 2 (rosbag2_py) for the bench bag, OpenCV, matplotlib and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from geometry import umeyama  # noqa: E402

ASSOC_TOL_S = 0.02
W_PX, H_PX, DPI = 1280, 720, 100
C_GT, C_VIS, C_DVL, C_BLK = "#f2efe6", "#ff6b5a", "#d7f25a", "#6a5cf0"


def load_tum(path: Path) -> np.ndarray:
    d = np.loadtxt(path, comments="#")
    if d.ndim != 2 or d.shape[1] < 4 or len(d) < 3:
        raise SystemExit(f"{path}: expected a TUM file with >= 3 rows")
    return d[:, :4]


def associate(t_ref: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index into t of the nearest sample to each t_ref (-1 if further than ASSOC_TOL_S)."""
    j = np.clip(np.searchsorted(t, t_ref), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - t_ref) < np.abs(t[j] - t_ref), j - 1, j)
    return np.where(np.abs(t[j] - t_ref) <= ASSOC_TOL_S, j, -1)


def aligned(gt: np.ndarray, est: np.ndarray, t_fit_end: float) -> np.ndarray:
    """Estimate positions at the GT times, SE(3)-aligned on GT samples before t_fit_end. NaN where unmatched."""
    j = associate(gt[:, 0], est[:, 0])
    ok = j >= 0
    fit = ok & (gt[:, 0] < t_fit_end)
    if fit.sum() < 10:
        raise SystemExit("fewer than 10 matched samples before the blackout; cannot align")
    _, r, t = umeyama(est[j[fit], 1:4], gt[fit, 1:4], with_scale=False)
    out = np.full((len(gt), 3), np.nan)
    out[ok] = est[j[ok], 1:4] @ r.T + t
    return out


def read_left_images(bag: Path, topic: str) -> tuple[np.ndarray, list[bytes]]:
    import rosbag2_py  # noqa: PLC0415  (ROS only needed here)
    from rclpy.serialization import deserialize_message  # noqa: PLC0415
    from sensor_msgs.msg import CompressedImage  # noqa: PLC0415

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    stamps, blobs = [], []
    while reader.has_next():
        _, raw, _ = reader.read_next()
        m = deserialize_message(raw, CompressedImage)
        stamps.append(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec)
        blobs.append(bytes(m.data))
    if not blobs:
        raise SystemExit(f"no images on {topic} in {bag}")
    return np.asarray(stamps), blobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True, help="converted run dir (bench_bag, gt_leftcam.tum, convert_info.json)")
    ap.add_argument("--results", type=Path, required=True, help="bench_sim.sh output dir with <variant>_blackout*.tum")
    ap.add_argument("--variant", default="clear")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=3, help="render every n-th camera frame")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    info = json.loads((a.bench / "convert_info.json").read_text())
    gt = load_tum(a.bench / "gt_leftcam.tum")
    blk0, blk_dur = (float(v) for v in info["suggested_blackout"].split(":"))
    t_blk0 = gt[0, 0] + blk0  # the run_bag blackout is relative to the first image
    t_blk1 = t_blk0 + blk_dur
    vis = aligned(gt, load_tum(a.results / f"{a.variant}_blackout.tum"), t_blk0)
    dvl = aligned(gt, load_tum(a.results / f"{a.variant}_blackout_dvl.tum"), t_blk0)
    e_vis = np.linalg.norm(vis - gt[:, 1:4], axis=1)
    e_dvl = np.linalg.norm(dvl - gt[:, 1:4], axis=1)
    t_img, blobs = read_left_images(a.bench / "bench_bag", info["left_topic"])
    k_img = associate(gt[:, 0], t_img)
    print(f"{len(gt)} GT samples, {len(blobs)} images, blackout {t_blk0 - gt[0, 0]:.0f}-{t_blk1 - gt[0, 0]:.0f} s; "
          f"max error in blackout: vision {np.nanmax(e_vis[(gt[:, 0] >= t_blk0) & (gt[:, 0] <= t_blk1)]):.2f} m, "
          f"vision+DVL {np.nanmax(e_dvl[(gt[:, 0] >= t_blk0) & (gt[:, 0] <= t_blk1)]):.2f} m", flush=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": C_GT, "axes.labelcolor": "#9a97a6",
                         "xtick.color": "#9a97a6", "ytick.color": "#9a97a6", "axes.edgecolor": "#3a3946"})
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor="#0c0b10")
    ax_img = fig.add_axes((0.02, 0.17, 0.50, 0.70))
    ax_map = fig.add_axes((0.57, 0.33, 0.40, 0.56), facecolor="#14131b")
    ax_err = fig.add_axes((0.57, 0.08, 0.40, 0.17), facecolor="#14131b")
    fig.text(0.02, 0.93, "Stonefish seabed survey  ·  20 s camera blackout", fontsize=17, weight="bold")
    fig.text(0.02, 0.89, "uw_ct_vio: stereo + IMU + pressure, with and without a tightly coupled DVL", fontsize=10.5, color="#9a97a6")
    fig.text(0.02, 0.04, "Simulated (Stonefish); estimates aligned on the pre-blackout segment only. github.com/Blackcipher101/uw-vio-dvl",
             fontsize=8.5, color="#6f6c7c")
    ax_img.set_axis_off()
    # map: NED -> east right, north up
    ax_map.plot(gt[:, 2], gt[:, 1], color=C_GT, lw=1, alpha=0.18)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("east [m]", fontsize=8)
    ax_map.set_ylabel("north [m]", fontsize=8)
    ax_map.tick_params(labelsize=7)
    l_gt, = ax_map.plot([], [], color=C_GT, lw=1.6, label="ground truth")
    l_vis, = ax_map.plot([], [], color=C_VIS, lw=1.6, label="vision only")
    l_dvl, = ax_map.plot([], [], color=C_DVL, lw=1.6, label="vision + DVL")
    p_gt, = ax_map.plot([], [], "o", color=C_GT, ms=5)
    p_vis, = ax_map.plot([], [], "o", color=C_VIS, ms=5)
    p_dvl, = ax_map.plot([], [], "o", color=C_DVL, ms=5)
    ax_map.legend(loc="upper right", fontsize=8, frameon=False)
    tr = gt[:, 0] - gt[0, 0]
    ax_err.axvspan(t_blk0 - gt[0, 0], t_blk1 - gt[0, 0], color=C_BLK, alpha=0.35, lw=0)
    ax_err.set_xlim(0, tr[-1])
    ax_err.set_ylim(0, max(1.0, np.nanmax(e_vis) * 1.1))
    ax_err.set_ylabel("error [m]", fontsize=8)
    ax_err.set_xlabel("time [s]", fontsize=8)
    ax_err.tick_params(labelsize=7)
    le_vis, = ax_err.plot([], [], color=C_VIS, lw=1.3)
    le_dvl, = ax_err.plot([], [], color=C_DVL, lw=1.3)
    txt = fig.text(0.57, 0.92, "", fontsize=11, family="DejaVu Sans Mono")
    im = ax_img.imshow(np.zeros((540, 720, 3), np.uint8))
    banner = ax_img.text(0.5, 0.5, "CAMERA BLACKOUT\nestimator is blind", transform=ax_img.transAxes, ha="center",
                         va="center", fontsize=20, weight="bold", color="#0c0b10",
                         bbox={"boxstyle": "round,pad=0.6", "fc": C_DVL, "ec": "none"}, visible=False)

    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W_PX}x{H_PX}", "-r", str(a.fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", "-movflags", "+faststart", str(a.out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last_img = None
    frames = range(0, len(gt), a.stride)
    for n, i in enumerate(frames):
        t = gt[i, 0]
        blind = t_blk0 <= t <= t_blk1
        if k_img[i] >= 0:
            last_img = cv2.cvtColor(cv2.imdecode(np.frombuffer(blobs[k_img[i]], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if last_img is not None:
            im.set_data((last_img * 0.35).astype(np.uint8) if blind else last_img)
        banner.set_visible(blind)
        s = slice(0, i + 1)
        for line, xy in ((l_gt, gt[s, 1:3]), (l_vis, vis[s, :2]), (l_dvl, dvl[s, :2])):
            line.set_data(xy[:, 1], xy[:, 0])
        for pt, xy in ((p_gt, gt[i, 1:3]), (p_vis, vis[i, :2]), (p_dvl, dvl[i, :2])):
            pt.set_data([xy[1]], [xy[0]])
        le_vis.set_data(tr[s], e_vis[s])
        le_dvl.set_data(tr[s], e_dvl[s])
        txt.set_text(f"t {tr[i]:6.1f} s   error: vision {e_vis[i]:5.2f} m   +DVL {e_dvl[i]:5.2f} m")
        txt.set_color(C_DVL if blind else C_GT)
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if n % 100 == 0:
            print(f"frame {n}/{len(frames)}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
