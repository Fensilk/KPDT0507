"""
Phase 10 §5.3 — qualitative failure mining: E1-full vs the 7 reference baselines.

Answers the question the planning doc (§1.2) asks for: on the same 1,200 test videos,
**where does E1 get it right while most generic video models get it wrong, and what exactly
do they get wrong** ("who turns lying into fallen").

Definitions
  * frame error rate of one model on one video = fraction of covered frames with pred != gt
  * "consensus failure" video = E1 error <= --e1_ok AND at least --min_wrong of the 7 baselines
    have error >= --wrong_err
  * each baseline's error on such a video is decomposed into
        over-trigger : gt normal      -> pred fall/fallen
        miss         : gt fall/fallen -> pred normal
        swap         : gt fall <-> fallen (state confusion, the lying<->fallen axis)
    and the dominant component is the video's failure type.

Inputs (all local)
  * E1-full : logs/phase9/e1_full/pv_best.pt          (per-video per-frame merged predictions)
  * baselines: logs/phase10/<tag>/test_dense_preds.npz (paths + (N,80) int8, -1 = uncovered)

Outputs (default logs/phase10/_consensus/)
  * consensus_failures.json  per-video breakdown
  * summary.json             aggregates: type counts, per-model "wrong" ranking, representatives
  * fig_traj_<type>.png      per-frame prediction timeline (gt band + E1 + worst baselines)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# the local py310 env links two OpenMP runtimes (torch + MKL); must be set before numpy/torch load
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "experiments"))

from eval_p10_compare_frames import load_e1, load_ref  # noqa: E402

MODELS = ["tsm_fair", "slowfast_r50", "x3d_s", "timesformer", "i3d_r50", "c3d", "vivit"]
LABEL = {"tsm_fair": "TSM", "slowfast_r50": "SlowFast-R50", "x3d_s": "X3D-S",
         "timesformer": "TimeSformer", "i3d_r50": "I3D-R50 (8f)", "c3d": "C3D",
         "vivit": "ViViT-T1"}
COLORS = ["#d62728", "#ff7f0e", "#2ca02c"]      # fall / fallen / normal
CLASSES = ["fall", "fallen", "normal"]


def breakdown(pred, gt):
    """Error decomposition of one model on one video (frames covered by both)."""
    n = min(len(pred), len(gt))
    p, g = np.asarray(pred[:n]), np.asarray(gt[:n])
    valid = (p >= 0) & (g >= 0)
    p, g = p[valid], g[valid]
    total = len(p)
    if total == 0:
        return None
    over = int(((g == 2) & (p != 2)).sum())          # normal reported as fall/fallen
    miss = int(((g != 2) & (p == 2)).sum())          # fall/fallen reported as normal
    swap = int(((g == 0) & (p == 1)).sum() + ((g == 1) & (p == 0)).sum())
    err = int((p != g).sum())
    dom = max([("over", over), ("miss", miss), ("swap", swap)], key=lambda kv: kv[1])[0]
    return {"frames": total, "errors": err, "err_rate": err / total,
            "over": over, "miss": miss, "swap": swap,
            "over_rate": over / total, "miss_rate": miss / total, "swap_rate": swap / total,
            "dominant": dom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=os.path.join(PROJ_ROOT, "logs", "phase9", "e1_full", "pv_best.pt"))
    ap.add_argument("--logs_root", default=os.path.join(PROJ_ROOT, "logs", "phase10"))
    ap.add_argument("--out_dir", default=os.path.join(PROJ_ROOT, "logs", "phase10", "_consensus"))
    ap.add_argument("--e1_ok", type=float, default=0.10, help="E1 must be at least this accurate")
    ap.add_argument("--wrong_err", type=float, default=0.30, help="a baseline counts as 'wrong' above this error")
    ap.add_argument("--min_wrong", type=int, default=5, help="how many baselines must be wrong")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    e1_pred, e1_gt = load_e1(args.e1)
    refs, missing = {}, []
    for m in MODELS:
        path = os.path.join(args.logs_root, m, "test_dense_preds.npz")
        if os.path.exists(path):
            refs[m] = load_ref(path)
        else:
            missing.append(m)
    if missing:
        print("[warn] missing baselines: %s" % missing)

    videos = sorted(set(e1_pred) & set.intersection(*[set(r) for r in refs.values()]))
    print("[data] %d common videos, %d baselines" % (len(videos), len(refs)))

    per_video, type_counter, model_wrong = {}, {"over": 0, "miss": 0, "swap": 0}, {m: 0 for m in refs}
    for v in videos:
        e1 = breakdown(e1_pred[v], e1_gt[v])
        if e1 is None or e1["err_rate"] > args.e1_ok:
            continue
        stats = {m: breakdown(refs[m][v], e1_gt[v]) for m in refs}
        wrong = [m for m, s in stats.items() if s and s["err_rate"] >= args.wrong_err]
        if len(wrong) < args.min_wrong:
            continue
        # the video's failure type = the dominant error across the wrong baselines
        tally = {"over": 0, "miss": 0, "swap": 0}
        for m in wrong:
            tally[stats[m]["dominant"]] += stats[m]["errors"]
        vtype = max(tally, key=tally.get)
        type_counter[vtype] += 1
        for m in wrong:
            model_wrong[m] += 1
        per_video[v] = {"type": vtype, "e1_err_rate": e1["err_rate"], "n_wrong": len(wrong),
                        "wrong_models": wrong,
                        "models": {m: stats[m] for m in refs}}

    ranked = sorted(per_video.items(), key=lambda kv: (-kv[1]["n_wrong"], kv[1]["e1_err_rate"]))
    # per failure type: which baselines are the worst offenders, and what the videos look like
    by_type = {}
    for vtype in ("over", "miss", "swap"):
        group = {v: d for v, d in ranked if d["type"] == vtype}
        if not group:
            continue
        stats = {}
        for m in refs:
            errs = [d["models"][m]["err_rate"] for d in group.values()]
            comp = {"over": sum(d["models"][m]["over"] for d in group.values()),
                    "miss": sum(d["models"][m]["miss"] for d in group.values()),
                    "swap": sum(d["models"][m]["swap"] for d in group.values())}
            stats[m] = {"mean_err": float(np.mean(errs)),
                        "err_rate_on_type": comp[vtype] / max(1, sum(comp.values())),
                        "abs_frames": comp[vtype]}
        by_type[vtype] = {
            "n_videos": len(group),
            "videos": list(group.keys()),
            "model_mean_err": {m: round(s["mean_err"], 4) for m, s in
                               sorted(stats.items(), key=lambda kv: -kv[1]["mean_err"])},
        }
    summary = {
        "n_videos_checked": len(videos),
        "n_consensus_failures": len(per_video),
        "criteria": {"e1_ok": args.e1_ok, "wrong_err": args.wrong_err, "min_wrong": args.min_wrong},
        "type_counts": type_counter,
        "model_wrong_counts": model_wrong,
        "representatives": {},
        "by_type": by_type,
    }

    # a representative video per failure type: most wrong baselines, then largest error margin
    for vtype in ("over", "miss", "swap"):
        cands = [(v, d) for v, d in ranked if d["type"] == vtype]
        if not cands:
            continue
        cands.sort(key=lambda kv: (-kv[1]["n_wrong"],
                                   -(np.mean([kv[1]["models"][m]["err_rate"] for m in kv[1]["wrong_models"]]))))
        v, d = cands[0]
        worst = sorted(d["wrong_models"], key=lambda m: -d["models"][m]["err_rate"])[:2]
        summary["representatives"][vtype] = {"video": v, "n_wrong": d["n_wrong"], "worst_models": worst}
        make_figure(v, e1_pred[v], e1_gt[v], {m: refs[m][v] for m in worst}, args.out_dir, vtype)

    with open(os.path.join(args.out_dir, "consensus_failures.json"), "w", encoding="utf-8") as fh:
        json.dump({v: d for v, d in ranked}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n=== 共识失败：E1 对、>=%d 个基线错 的视频共 %d / %d ===" % (args.min_wrong, len(per_video), len(videos)))
    for k, n in type_counter.items():
        print("  %-5s %d 个视频" % (k, n))
    print("\n=== 各基线在这些视频里被判“错”的次数 ===")
    for m, n in sorted(model_wrong.items(), key=lambda kv: -kv[1]):
        print("  %-14s %d" % (LABEL.get(m, m), n))
    print("\n=== 按失败类型：基线的平均帧错误率（谁最严重） ===")
    for vtype, d in by_type.items():
        top = list(d["model_mean_err"].items())[:3]
        print("  %-5s (%2d 视频) 最差: %s" % (vtype, d["n_videos"],
                                             ", ".join("%s %.3f" % (LABEL.get(m, m), r) for m, r in top)))
        print("        视频: %s" % ", ".join(d["videos"][:6]))
    print("\n=== 代表视频 ===")
    for k, d in summary["representatives"].items():
        print("  %-5s %s  (错基线 %d 个，最差：%s)" % (k, d["video"], d["n_wrong"],
                                                    ", ".join(LABEL.get(m, m) for m in d["worst_models"])))
    print("\n[out] %s" % args.out_dir)


def make_figure(video, e1_pred, gt, preds, out_dir, tag):
    """Per-frame prediction timeline: gt band + E1 + the two worst baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(COLORS)
    rows = [("ground truth", np.asarray(gt[:80]), "GT")]
    rows.append(("E1-full", np.asarray(e1_pred[:80]), "E1"))
    for m, p in preds.items():
        rows.append((LABEL.get(m, m), np.asarray(p[:80]), m))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.55 * n + 1.1))
    mat = np.vstack([np.where(r[1] < 0, 2, r[1]) for r in rows])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xticks(range(0, 81, 10))
    ax.set_xticklabels([str(i) for i in range(0, 81, 10)], fontsize=8)
    ax.set_xlabel("frame index (0-79)", fontsize=9)
    ax.set_title("failure type: %s   |   %s" % (tag, video.split("/")[-1]), fontsize=10)
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS, CLASSES)],
              loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_traj_%s.png" % tag)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig] %s" % path)


if __name__ == "__main__":
    main()
