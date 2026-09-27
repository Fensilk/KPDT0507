# -*- coding: utf-8 -*-
"""验证「val 只看每视频前 64 帧」到底带来多大偏差。

背景：Phase 9 的 val 协议是 150 视频 × stride 80 → 每视频仅 [0:64] 一个窗口，
而 test 是 1200 视频 × stride 8 → 覆盖 0-79 全帧。两者 val→test 的 fallen 轴
落差高达 0.12-0.36。此前把它归因于「窗口截断」，但该落差里混了三个因子：

  (a) 窗口截断：val 看不到 64-79 帧
  (b) 视频集不同：val 是 150 个 val.csv 视频，test 是 1200 个 test.csv 视频
  (c) 选型偏差：best ckpt 是按 val 指标挑的 → val 数字是"跨 ckpt 取最大"，天然乐观

本脚本只隔离 (a)，用它来回答「尾部 16 帧是否无关紧要」：
固定同一批预测（E1-full 的 1200 个 test 视频逐视频预测），
把评测窗口从 0-79 截到 0-63，指标的变化量 = 纯截断偏差，不含 (b)(c)。

顺带给出标签层的分布证据（尾部 16 帧里有多少 fall/fallen 帧、多少状态切换）。

用法：python -u analysis/verify_val_truncation_bias.py
"""

import os
import sys
import json
import csv

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils.timeline_metrics import (  # noqa: E402
    compute_video_timeline_metrics,
    aggregate_timeline_metrics,
)

NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
SPLITS = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random")
PV = os.path.join(BASE, "logs", "phase9", "e1_full", "pv_best.pt")


def load_split(name):
    with open(os.path.join(SPLITS, f"{name}.csv"), newline="", encoding="utf-8") as fh:
        return [r["path"] for r in csv.DictReader(fh)]


def resolve_val150():
    """val 监控用的 150 个视频：从 data/p9e1_frames_val 的文件名还原。

    文件名格式 {idx}_{label16}_{videoname}.pt，videoname 是 split CSV 路径的 basename。
    """
    d = os.path.join(BASE, "data", "p9e1_frames_val")
    if not os.path.isdir(d):
        return None
    all_paths = []
    for s in ("train", "val", "test"):
        all_paths += load_split(s)
    by_base = {norm(p).split("/")[-1]: norm(p) for p in all_paths}
    picked, missed = [], []
    for fn in sorted(os.listdir(d)):
        stem = fn[:-3] if fn.endswith(".pt") else fn
        hit = next((b for b in by_base if stem.endswith(b)), None)
        (picked if hit else missed).append(by_base[hit] if hit else stem)
    if missed:
        print(f"  ⚠ 有 {len(missed)} 个 val 文件名无法匹配到 split，例：{missed[:3]}")
    return picked


def norm(p):
    p = str(p).replace("\\", "/")
    for ext in (".mp4", ".avi", ".mkv"):
        if p.lower().endswith(ext):
            p = p[: -len(ext)]
    return p


# ---------------------------------------------------------------- 标签层
def label_stats(split_name, index_of, ternary_all, starts, paths=None):
    """尾部 16 帧（64-79）里 fall/fallen 的质量占比、状态切换数、首发位置。"""
    if paths is None:
        paths = load_split(split_name)
    n_fall = n_fallen = 0
    tail_fall = tail_fallen = 0
    first_fall_pos, first_fallen_pos = [], []
    tail_trans, all_trans = 0, 0
    n_vid = 0
    for p in paths:
        i = index_of.get(norm(p))
        if i is None:
            continue
        n_vid += 1
        t = ternary_all[starts[i]: starts[i] + 80]
        n_fall += int((t == 0).sum())
        n_fallen += int((t == 1).sum())
        tail_fall += int((t[64:] == 0).sum())
        tail_fallen += int((t[64:] == 1).sum())
        fi = np.flatnonzero(t == 0)
        ni = np.flatnonzero(t == 1)
        if len(fi):
            first_fall_pos.append(int(fi[0]))
        if len(ni):
            first_fallen_pos.append(int(ni[0]))
        tr = (t[1:] != t[:-1])
        all_trans += int(tr.sum())
        tail_trans += int(tr[63:].sum())  # t[64] vs t[63] 起算 → 落在 64-79 的切换
    return dict(
        split=split_name, n_videos=n_vid,
        fall_frames=n_fall, fallen_frames=n_fallen,
        tail_fall=tail_fall, tail_fallen=tail_fallen,
        pct_fall_in_tail=100.0 * tail_fall / max(n_fall, 1),
        pct_fallen_in_tail=100.0 * tail_fallen / max(n_fallen, 1),
        n_first_fall_ge64=sum(1 for v in first_fall_pos if v >= 64),
        n_first_fallen_ge64=sum(1 for v in first_fallen_pos if v >= 64),
        n_has_fall=len(first_fall_pos), n_has_fallen=len(first_fallen_pos),
        transitions_total=all_trans, transitions_in_tail=tail_trans,
        pct_trans_in_tail=100.0 * tail_trans / max(all_trans, 1),
        median_first_fall=float(np.median(first_fall_pos)) if first_fall_pos else None,
        median_first_fallen=float(np.median(first_fallen_pos)) if first_fallen_pos else None,
    )


# ---------------------------------------------------------------- 预测层
def tern_bin(t):
    t = np.asarray(t)
    return (t == 0).astype(int), (t == 1).astype(int)


def eval_trunc(pv, n_frames):
    """n_frames=80 → 全帧；=64 → 只评测前 64 帧（模拟 val 协议）。"""
    per = []
    for _k, v in pv.items():
        gt = np.asarray(v["ternary_gt"])[:n_frames]
        pr = np.asarray(v["bridge_pred"])[:n_frames]
        gF, gN = tern_bin(gt)
        pF, pN = tern_bin(pr)
        per.append(compute_video_timeline_metrics(pF, pN, gF, gN))
    return aggregate_timeline_metrics(per)


def instance_metrics(pv, n_frames):
    """instance（逐帧实例）口径的 P/R/F1，用于与 val 的 val_fallen_prec 对齐。

    逐视频指标（Q2）与 val 报告的 fallen_prec 不是同一口径，必须两者都测。
    """
    TP = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)
    for _k, v in pv.items():
        gt = np.asarray(v["ternary_gt"])[:n_frames]
        pr = np.asarray(v["bridge_pred"])[:n_frames]
        for c in range(3):
            TP[c] += int(((pr == c) & (gt == c)).sum())
            FP[c] += int(((pr == c) & (gt != c)).sum())
            FN[c] += int(((pr != c) & (gt == c)).sum())
    P = TP / np.maximum(TP + FP, 1)
    R = TP / np.maximum(TP + FN, 1)
    F = 2 * P * R / np.maximum(P + R, 1e-12)
    return P, R, F, int(TP.sum() + FP.sum() + FN.sum())


def main():
    print("=" * 78)
    print("Q2  预测层：固定同一批预测，全帧(0-79) vs 截断(0-63) 的指标差")
    print("=" * 78)
    pv = torch.load(PV, weights_only=False, map_location="cpu")
    if "pv" in pv:
        pv = pv["pv"]
    keys = list(pv.keys())
    lens = {len(np.asarray(v["ternary_gt"])) for v in pv.values()}
    print(f"pv 条目数={len(keys)}  序列长度集合={lens}  样例 key={keys[0]!r}")

    full = eval_trunc(pv, 80)
    tr = eval_trunc(pv, 64)
    rows = [
        ("per_video_fall_f1", "mean"),
        ("per_video_fallen_f1", "mean"),
        ("per_video_avg_f1", "mean"),
        ("fall_detection_rate", None),
        ("fallen_detection_rate", None),
        ("fall_false_alarm_rate", None),
        ("fallen_false_alarm_rate", None),
        ("mean_fall_flips", None),
        ("mean_fallen_flips", None),
        ("mean_total_flips", None),
        ("mean_fall_delay", None),
        ("mean_fallen_delay", None),
        ("order_correct_rate", None),
        ("pass_rate", None),
    ]
    print(f"\n{'指标':<28}{'全帧0-79':>12}{'截断0-63':>12}{'Δ(截断-全帧)':>16}")
    print("-" * 68)
    for name, sub in rows:
        a, b = full.get(name), tr.get(name)
        if isinstance(a, dict):
            a, b = a.get(sub), b.get(sub)
        if a is None or b is None:
            continue
        print(f"{name:<28}{a:>12.4f}{b:>12.4f}{b - a:>+16.4f}")

    # --- 同一件事的 instance 口径（与 val 报告的 fallen_prec 同口径）---
    print("\n" + "-" * 68)
    print("Q2b instance（逐帧实例）口径 —— 这一栏才与 val 的 val_fallen_prec 同口径")
    print("-" * 68)
    P80, R80, F80, _ = instance_metrics(pv, 80)
    P64, R64, F64, _ = instance_metrics(pv, 64)
    print(f"{'':<14}{'全帧0-79':>26}{'截断0-63':>26}")
    for c, nm in enumerate(("fall", "fallen")):
        print(f"  {nm:<12}P {P80[c]:.4f}  R {R80[c]:.4f}  F1 {F80[c]:.4f}"
              f"   P {P64[c]:.4f}  R {R64[c]:.4f}  F1 {F64[c]:.4f}")
    print(f"  avg_f1      {F80.mean():.4f}{'':>21}{F64.mean():.4f}")
    print(f"\n  instance Δ(截断−全帧): "
          f"fall_prec {P64[0]-P80[0]:+.4f}  fall_rec {R64[0]-R80[0]:+.4f}  |  "
          f"fallen_prec {P64[1]-P80[1]:+.4f}  fallen_rec {R64[1]-R80[1]:+.4f}  |  "
          f"avg_f1 {F64.mean()-F80.mean():+.4f}")
    print("  对照：E1-full best ckpt 的 val fallen_prec=0.9020 vs test 0.7551 → 落差 0.147")
    print("  ⇒ 截断只能解释该落差的约 11%，其余来自 val 集类分布差异与 150 视频小样本/选型偏差")

    print("\n" + "=" * 78)
    print("Q1  标签层：尾部 16 帧(64-79) 占多少 fall/fallen 帧、多少状态切换")
    print("=" * 78)
    npz = np.load(NPZ, mmap_mode="r", allow_pickle=True)
    labels16 = np.asarray(npz["labels_16"])
    starts = np.asarray(npz["video_start_indices"])
    vpaths = npz["video_paths"]
    ternary_all = np.full(len(labels16), 2, dtype=np.int8)
    ternary_all[labels16 == 1] = 0
    ternary_all[labels16 == 2] = 1
    index_of = {norm(p): i for i, p in enumerate(vpaths)}
    print(f"npz 视频数={len(vpaths)}  帧数={len(labels16)}")

    stats = [label_stats(s, index_of, ternary_all, starts) for s in ("val", "test")]

    v150 = resolve_val150()
    if v150:
        print(f"\n还原出 val 监控用的 {len(v150)} 个视频（data/p9e1_frames_val）")
        stats.append(label_stats("val150(实际用于监控)", index_of, ternary_all, starts,
                                 paths=v150))
    for st in stats:
        print(f"\n--- {st['split']}.csv ({st['n_videos']} 视频) ---")
        print(f"  fall 帧 {st['fall_frames']}，其中尾部(64-79) {st['tail_fall']} "
              f"= {st['pct_fall_in_tail']:.2f}%")
        print(f"  fallen 帧 {st['fallen_frames']}，其中尾部(64-79) {st['tail_fallen']} "
              f"= {st['pct_fallen_in_tail']:.2f}%")
        print(f"  首个 fall 帧 >=64 的视频数: {st['n_first_fall_ge64']} / {st['n_has_fall']}")
        print(f"  首个 fallen 帧 >=64 的视频数: {st['n_first_fallen_ge64']} / {st['n_has_fallen']}")
        print(f"  状态切换总数 {st['transitions_total']}，落在尾部 {st['transitions_in_tail']} "
              f"= {st['pct_trans_in_tail']:.2f}%")
        print(f"  首个 fall 位置中位数 {st['median_first_fall']}；"
              f"首个 fallen 位置中位数 {st['median_first_fallen']}")

    print("\n--- 视频集构成对比（因子②：val 监控集 vs test 评测集）---")
    keys = ("n_videos", "n_has_fall", "n_has_fallen",
            "pct_fall_in_tail", "pct_fallen_in_tail", "pct_trans_in_tail",
            "median_first_fall", "median_first_fallen")
    print("  %-22s" % "指标" + "".join("%18s" % st["split"][:16] for st in stats))
    for k in keys:
        line = "  %-22s" % k
        for st in stats:
            val = st[k]
            line += "%18s" % (f"{val:.2f}" if isinstance(val, float) else str(val))
        print(line)

    out = os.path.join(BASE, "logs", "phase9", "val_truncation_bias.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "full_80": full,
            "trunc_64": tr,
            "instance_full_80": {"fall_prec": P80[0], "fall_rec": R80[0], "fall_f1": F80[0],
                                 "fallen_prec": P80[1], "fallen_rec": R80[1], "fallen_f1": F80[1],
                                 "avg_f1": float(F80.mean())},
            "instance_trunc_64": {"fall_prec": P64[0], "fall_rec": R64[0], "fall_f1": F64[0],
                                  "fallen_prec": P64[1], "fallen_rec": R64[1], "fallen_f1": F64[1],
                                  "avg_f1": float(F64.mean())},
            "label_stats": stats,
        }, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\n已写出: {out}")


if __name__ == "__main__":
    main()
