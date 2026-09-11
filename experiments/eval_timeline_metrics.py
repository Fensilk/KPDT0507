"""真·事件级指标评估（替代名实不符的 frame-flattened "event_f1"）。

为什么换指标（详见 docs/0902实验第九阶段总结.md §5.1）：
- 旧口径 `eval_p9e1_dual.binary_event_metrics` 的 "event_f1" 是**逐帧二值 F1**
  （GT: fall|fallen 帧；Pred: ≠normal 帧），无事件匹配/IoU/延迟/容差，且把
  fall（动态瞬态）与 fallen（静态状态）压成一轴 → 名实不符、掩盖轴分工、
  对 lying 误报不敏感、且在 6000 数据即饱和。
- 本脚本复用 `utils/timeline_metrics.py`（per-video 时间线质量指标），它通过
  **GT 自检**（GT 当预测喂入 → F1 全 = 1.0；非事件视频 F1 记 None 被排除而非记 0）。

产出：逐视频 F1 分布、检测覆盖率、误报率、检测延迟（**单位=帧**）、翻转次数、
时序正确率。**注意 order_correct 在"全漏"时也为 1.0（模块口径）→ 必须与
detection_rate 联读，不可单独引用。**

输入：per-video 预测缓存（{path 或 idx: {ternary_gt, bridge_pred, ...}}），
格式同 `logs/phase9/e3_rulev2/cache/*.pt`（可带 {'pv': ...} 外壳）与
`logs/phase9/e1_rulev2/pv_cache.pt`。

用法：
    python experiments/eval_timeline_metrics.py --out logs/phase9/timeline_metrics.json \
        baseline=logs/phase9/e3_rulev2/cache/baseline.pt:pv \
        e3full=logs/phase9/e3_rulev2/cache/E3-full.pt:pv \
        e1_2000=logs/phase9/e1_rulev2/pv_cache.pt
    # 加 --gt_ceiling 追加"GT 自检"行（天花板，应恒为 1.0）
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils.timeline_metrics import compute_video_timeline_metrics, aggregate_timeline_metrics  # noqa: E402


def load_spec(spec):
    """'name=path[:wrap]' → (name, {path: {...}})"""
    name, rest = spec.split("=", 1)
    wrap = None
    if ":" in rest and not rest.endswith(".pt"):
        rest, wrap = rest.rsplit(":", 1)
    elif rest.count(":") == 1 and rest.split(":")[1].strip():
        rest, wrap = rest.split(":", 1)
    d = torch.load(rest, weights_only=False, map_location="cpu")
    if wrap:
        d = d[wrap]
    return name, d


def tern_binaries(tern):
    """三元 (0=fall,1=fallen,2=normal) → (fall_ind, fallen_ind) 两个 0/1 序列。"""
    t = np.asarray(tern)
    return (t == 0).astype(int), (t == 1).astype(int)


def eval_pv(pv):
    per = []
    for _k, v in pv.items():
        gt = np.asarray(v["ternary_gt"])
        pr = np.asarray(v["bridge_pred"])
        gF, gN = tern_binaries(gt)
        pF, pN = tern_binaries(pr)
        per.append(compute_video_timeline_metrics(pF, pN, gF, gN))
    return per, aggregate_timeline_metrics(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="name=path[:wrap] ...")
    ap.add_argument("--out", default=os.path.join(BASE, "logs", "phase9", "timeline_metrics.json"))
    ap.add_argument("--gt_ceiling", action="store_true", help="追加 GT 自检行（GT 当预测）")
    args = ap.parse_args()

    results = {}
    rows = []

    def add(name, pv):
        per, agg = eval_pv(pv)
        f, n, av = agg["per_video_fall_f1"], agg["per_video_fallen_f1"], agg["per_video_avg_f1"]
        results[name] = agg
        rows.append(dict(name=name, agg=agg, f=f, n=n, av=av))

    for spec in args.specs:
        name, pv = load_spec(spec)
        add(name, pv)
    if args.gt_ceiling:
        # 用第一个 spec 的 GT 造"完美预测"
        name0, pv0 = load_spec(args.specs[0])
        add("GT(自检天花板)", {k: {**v, "bridge_pred": v["ternary_gt"]} for k, v in pv0.items()})

    hdr = ("%-18s %6s %8s %9s %8s %9s %10s %9s %11s %8s %8s" %
           ("模型", "n事件", "fall_f1", "fallen_f1", "avg_f1", "覆盖率F/N", "误报率F/N",
            "延迟中位", "翻转(video)", "order", "满分比"))
    print(hdr); print("-" * len(hdr))
    for r in rows:
        a, f, n = r["agg"], r["f"], r["n"]
        full = "%d/%d" % (f["1.0"], f["n"]) if f["n"] else "-"
        print("%-18s %6d %8s %9s %8s %9s %10s %9s %11s %8s %8s" % (
            r["name"], f["n"],
            ("%.4f" % f["mean"]) if f["mean"] is not None else "-",
            ("%.4f" % n["mean"]) if n["mean"] is not None else "-",
            ("%.4f" % r["av"]["mean"]) if r["av"]["mean"] is not None else "-",
            "%.3f/%.3f" % (a["fall_detection_rate"], a["fallen_detection_rate"]),
            "%.3f/%.3f" % (a["fall_false_alarm_rate"], a["fallen_false_alarm_rate"]),
            "%s" % (round(a["median_fall_delay"], 1) if a["median_fall_delay"] is not None else "-"),
            "%.2f" % a["mean_total_flips"],
            "%.3f" % a["order_correct_rate"],
            full))

    print("\n== 逐视频 fall_f1 分布（事件视频内）==")
    for r in rows:
        f = r["f"]
        print("  %-18s n=%3d | =1.0:%3d  0.8-1.0:%3d  0.5-0.8:%3d  0-0.5:%3d  =0:%3d" %
              (r["name"], f["n"], f["1.0"], f["0.8-1.0"], f["0.5-0.8"], f["0.0-0.5"], f["0.0"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=lambda o: float(o))
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
