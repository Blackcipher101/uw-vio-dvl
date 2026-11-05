#!/usr/bin/env python3
"""Regenerate the README figures of uw-vio-dvl from run outputs.

Usage: python3 scripts/make_figures.py --data <uw_figdata> --mocap-root <dir with traj_0X/traj_0X_mocap.tum> \
           [--out docs/figures]

<data> holds ntnu_all/<tag>/traj_0X.{tum,eval}, ntnu_all.tsv and dvl_sim/traj_0X/<scenario>.csv.
Estimates are aligned exactly as scripts/evaluate.py --hand-eye --hand-eye-first-s 60 does
(hand-eye on the first 60 s, then SE3 Umeyama, no scale); the script prints its ATE next to the
ATE stored in the .eval file so the plots can be checked against the reported numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate import associate, hand_eye, load_tum, move_gt, to_mats  # noqa: E402
from geometry import umeyama  # noqa: E402

GT, BASE, NODVL, DVL, GAP = "#222222", "#4C72B0", "#DD5A2B", "#2E8B57", "#8E5BA8"
BLACKOUT_START_S, BLACKOUT_DUR_S, GAP_START_S, GAP_DUR_S = 60.0, 20.0, 65.0, 7.0
HAND_EYE_FIRST_S = 60.0
GT_SEGMENT_GAP_S = 0.2
TRAJS = [f"traj_0{i}" for i in range(1, 7)]
SCENARIOS = [("dvl_none_off", "DVL off"), ("dvl_full", "full"), ("dvl_periodic", "periodic"),
             ("dvl_long", "long"), ("dvl_bursts", "bursts"), ("dvl_outliers", "outliers")]
DATASET = "NTNU underwater 2023 (subset-3)"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150})


def align(gt: np.ndarray, est: np.ndarray):
    """Return (est xy z aligned into GT frame, est times, moved GT, matched idx pair, ATE_se3)."""
    X, _ = hand_eye(gt, est, HAND_EYE_FIRST_S)
    gtm = move_gt(gt, X)
    ig, ie = associate(gtm[:, 0], est[:, 0])
    e = to_mats(est[ie])[:, :3, 3]
    best = None
    for conv in ("T_world_cam", "T_cam_world"):
        g = to_mats(gtm[ig])
        if conv == "T_cam_world":
            g = np.linalg.inv(g)
        _, r, t = umeyama(e, g[:, :3, 3], with_scale=False)
        err = np.linalg.norm((r @ e.T).T + t - g[:, :3, 3], axis=1)
        ate = float(np.sqrt(np.mean(err ** 2)))
        if best is None or ate < best[0]:
            best = (ate, r, t, err)
    ate, r, t, err = best
    all_e = to_mats(est)[:, :3, 3]
    return (r @ all_e.T).T + t, gtm, ig, ie, err, ate


def segments(t: np.ndarray, gap: float):
    cuts = np.nonzero(np.diff(t) > gap)[0] + 1
    return np.split(np.arange(len(t)), cuts)


def load_run(data: Path, tag: str, traj: str, gt: np.ndarray) -> dict:
    est = load_tum(data / "ntnu_all" / tag / f"{traj}.tum")
    pos, gtm, ig, ie, err, ate = align(gt, est)
    stored = json.loads((data / "ntnu_all" / tag / f"{traj}.eval").read_text().strip().splitlines()[-1]
                        .split(" ", 1)[1])["ate_se3_m"]
    print(f"{tag:22s} {traj}: plotted ATE {ate:.4f} m, stored .eval ATE {stored:.4f} m, "
          f"diff {abs(ate - stored) * 100:.2f} cm")
    return {"est": est, "pos": pos, "gtm": gtm, "ig": ig, "ie": ie, "err": err, "ate": ate}


def shade(ax, t0: float, vertical: bool = True):
    ax.axvspan(t0 + BLACKOUT_START_S, t0 + BLACKOUT_START_S + BLACKOUT_DUR_S, color="#BBBBBB", alpha=0.45,
               lw=0, label="20 s camera blackout")


def read_tsv(path: Path) -> dict:
    d: dict = {}
    for line in path.read_text().splitlines():
        tag, traj, ate, *_ = line.split("\t")
        if not traj.startswith("MEAN"):
            d.setdefault(tag, {})[traj] = float(ate)
    return d


def fig_hero_and_error(data: Path, mocap: Path, out: Path, ates: dict) -> None:
    worst = max(("traj_02", "traj_03", "traj_06"), key=lambda k: ates["blackout20_nodvl"][k])
    gt = load_tum(mocap / worst / f"{worst}_mocap.tum")
    runs = {tag: load_run(data, tag, worst, gt) for tag in
            ("dvl_none_off", "blackout20_nodvl", "blackout20_dvl", "blackout20_dvlgap7")}
    t0 = runs["dvl_none_off"]["est"][0, 0]
    ref = runs["blackout20_nodvl"]
    gtm = ref["gtm"]
    style = {"blackout20_nodvl": (NODVL, "blackout, no DVL"), "blackout20_dvl": (DVL, "blackout + DVL (simulated)")}

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for seg in segments(gtm[:, 0], GT_SEGMENT_GAP_S):
        if len(seg) > 1:
            ax.plot(gtm[seg, 1], gtm[seg, 2], color=GT, lw=2.2, label="mocap ground truth" if seg[0] == 0 else None)
    for tag, (c, lab) in style.items():
        r = runs[tag]
        p, t = r["pos"], r["est"][:, 0] - t0
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.4, label=f"{lab}  (ATE {r['ate']:.2f} m)")
        m = (t >= BLACKOUT_START_S) & (t <= BLACKOUT_START_S + BLACKOUT_DUR_S)
        ax.plot(p[m, 0], p[m, 1], color=c, lw=4.5, alpha=0.4, solid_capstyle="round")
    ax.plot([], [], color="0.5", lw=4.5, alpha=0.4, label="blackout segment (thick)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{worst}: a 20 s camera blackout sends the no-DVL estimate far off;\n"
                 "with a (simulated) DVL it stays close to the true path", fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "hero_trajectory.png", pil_kwargs={"optimize": True})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    shade(ax, 0.0)
    ax.axvspan(GAP_START_S, GAP_START_S + GAP_DUR_S, ymin=0.0, ymax=0.04, color=GAP, lw=0, label="DVL also invalid (gap7 run)")
    for tag, c, lab in (("dvl_none_off", BASE, "cameras OK, no DVL (baseline)"),
                        ("blackout20_nodvl", NODVL, "blackout, no DVL"),
                        ("blackout20_dvl", DVL, "blackout + DVL (simulated)"),
                        ("blackout20_dvlgap7", GAP, "blackout + DVL, DVL lost 7 s")):
        r = runs[tag]
        t = r["est"][r["ie"], 0] - t0
        for seg in segments(t, GT_SEGMENT_GAP_S):
            ax.plot(t[seg], r["err"][seg], color=c, lw=1.2, label=lab if seg[0] == segments(t, GT_SEGMENT_GAP_S)[0][0] else None)
    ax.set_yscale("log")
    ax.set_xlabel("time since first image (s)")
    ax.set_ylabel("position error (m, log)")
    ax.set_title(f"{worst}: error grows during the blackout without DVL and stays bounded with it", fontsize=11)
    ax.set_ylim(0.02, 40)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "error_vs_time.png", pil_kwargs={"optimize": True})
    plt.close(fig)


def fig_bars(ates: dict, out: Path) -> None:
    sets = [("dvl_none_off", BASE, "baseline (cameras OK, no DVL)"), ("blackout20_nodvl", NODVL, "blackout, no DVL"),
            ("blackout20_dvl", DVL, "blackout + DVL (simulated)")]
    x = np.arange(len(TRAJS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for i, (tag, c, lab) in enumerate(sets):
        v = [ates[tag][t] for t in TRAJS]
        b = ax.bar(x + (i - 1) * w, v, w, color=c, label=lab)
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val * 1.06, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(x, [t.replace("traj_", "traj ") for t in TRAJS])
    ax.set_ylabel("ATE SE3 (m, log)")
    ax.set_ylim(0.1, 25)
    ax.set_title("A 20 s camera blackout costs up to 10.7 m without DVL; with DVL it stays below 1 m", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ate_blackout_bars.png", pil_kwargs={"optimize": True})
    plt.close(fig)


def fig_dropouts(ates: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    base = np.mean([ates["dvl_none_off"][t] for t in TRAJS])
    for i, (tag, lab) in enumerate(SCENARIOS):
        v = np.array([ates[tag][t] for t in TRAJS])
        c = BASE if i == 0 else DVL
        ax.bar(i, v.mean(), 0.6, color=c, alpha=0.55)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, s=14, color=GT, zorder=3,
                   label="per-trajectory ATE" if i == 0 else None)
        ax.text(i, v.max() + 0.03, f"mean {v.mean():.3f}", ha="center", fontsize=8.5)
    ax.axhline(base, color=BASE, ls="--", lw=1, label=f"DVL-off mean ({base:.3f} m)")
    ax.set_xticks(range(len(SCENARIOS)), [s[1] for s in SCENARIOS])
    ax.set_xlabel("simulated DVL scenario")
    ax.set_ylabel("ATE SE3 (m)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Simulated DVL dropouts and outliers do not raise the mean ATE (cameras OK)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "dvl_dropout_robustness.png", pil_kwargs={"optimize": True})
    plt.close(fig)


def fig_dvl_sim(data: Path, out: Path, traj: str = "traj_02") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.6), sharex=True)
    for ax, sc in zip(axes, ("periodic", "outliers")):
        a = np.genfromtxt(data / "dvl_sim" / traj / f"{sc}.csv", delimiter=",", names=True)
        t = a["t"] - a["t"][0]
        valid = a["valid"] > 0.5
        # the simulator only emits samples where mocap exists: do not draw lines across those holes
        hole = np.concatenate([[False], np.diff(t) > 0.5])
        for comp, c in zip(("vx", "vy", "vz"), (BASE, NODVL, DVL)):
            ax.plot(t, np.where(valid & ~hole, a[comp], np.nan), lw=0.7, color=c, label=comp)
        for seg in segments(t[~valid], 1.0) if (~valid).any() else []:
            tt = t[~valid][seg]
            ax.axvspan(tt[0], tt[-1] + 0.12, color="#BBBBBB", alpha=0.6, lw=0)
        ax.axvspan(0, 0, color="#BBBBBB", alpha=0.6, label="valid = 0")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title(f"simulated DVL, scenario '{sc}' ({traj}); valid = 0 stretches shaded", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper right")
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].set_xlabel("time since first DVL sample (s)")
    fig.tight_layout()
    fig.savefig(out / "simulated_dvl_example.png", pil_kwargs={"optimize": True})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mocap-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ates = read_tsv(a.data / "ntnu_all.tsv")
    fig_hero_and_error(a.data, a.mocap_root, a.out, ates)
    fig_bars(ates, a.out)
    fig_dropouts(ates, a.out)
    fig_dvl_sim(a.data, a.out)


if __name__ == "__main__":
    main()
