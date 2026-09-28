# -*- coding: utf-8 -*-
"""边界指标的**两道验收** + E1-full 锚点的边界指标（规划 §5.2）。

CLAUDE.md 强制："新指标必须过两道验收才可采用：① GT 自检——真值当预测喂入须得满分
（非事件样本应排除而非记 0）；② 负对照——故意注入错误（全漏/平移/抖动）须单调劣化。
两者都要跑出数并写进文档。"

本脚本跑出这些数，并把 E1-full 的边界指标一并算出（它是 W1 的对照基准）。

用法：python -u analysis/verify_boundary_metrics.py
"""

import os
import sys
import json

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils.boundary_metrics import (  # noqa: E402
    compute_video_boundary_metrics,
    compute_video_boundary_metrics_by_type,
    aggregate_boundary_metrics,
    find_boundaries,
    match_boundaries,
)

PV = os.path.join(BASE, "logs", "phase9", "e1_full", "pv_best.pt")
TAUS = (1, 3, 5)
NORMAL = 2


def load_pv():
    pv = torch.load(PV, weights_only=False, map_location="cpu")
    if "pv" in pv:
        pv = pv["pv"]
    out = {}
    for k, v in pv.items():
        out[k] = (np.asarray(v["ternary_gt"]), np.asarray(v["bridge_pred"]))
    return out


def shift_seq(seq, d):
    """把序列右移 d 帧（模拟"边界整体滞后 d 帧"），左侧用首值填充。"""
    seq = np.asarray(seq)
    if d == 0:
        return seq.copy()
    out = np.empty_like(seq)
    out[:d] = seq[0]
    out[d:] = seq[:-d]
    return out


def jitter_seq(seq, k, rng):
    """注入 k 个单帧"毛刺"（翻转一帧的类别）→ 每处产生 2 个假边界。"""
    seq = np.asarray(seq).copy()
    T = len(seq)
    pos = rng.choice(np.arange(1, T - 1), size=min(k, T - 2), replace=False)
    for t in pos:
        # 翻成与左右都不同的类别，保证 t 与 t+1 都成为边界
        others = [c for c in (0, 1, 2) if c != seq[t - 1] and c != seq[t + 1]]
        if others:
            seq[t] = others[0]
    return seq


def eval_all(pairs, tau):
    per = {k: compute_video_boundary_metrics(g, p, tau) for k, (g, p) in pairs.items()}
    return aggregate_boundary_metrics(per, taus=(tau,))[f"tau{tau}"]


def matched_offsets(gt_seq, pred_seq, tau):
    """匹配上的边界对的有符号偏移 (pred - gt)，用于验证"延迟被准确读出"。"""
    gt, pred = find_boundaries(gt_seq), find_boundaries(pred_seq)
    i = j = 0
    offs = []
    while i < len(gt) and j < len(pred):
        if abs(gt[i] - pred[j]) <= tau:
            offs.append(pred[j] - gt[i])
            i += 1
            j += 1
        elif pred[j] < gt[i]:
            j += 1
        else:
            i += 1
    return offs


def main():
    pairs = load_pv()
    print(f"载入 {len(pairs)} 个视频的逐视频预测（E1-full best，已过投票后处理）")

    # ---------------------------------------------------------------- ① GT 自检
    print("\n" + "=" * 84)
    print("① GT 自检 —— 真值当预测喂入，必须满分（非事件视频排除而非记 0）")
    print("=" * 84)
    gt_as_pred = {k: (g, g.copy()) for k, (g, _p) in pairs.items()}
    ok = True
    for tau in TAUS:
        r = eval_all(gt_as_pred, tau)
        good = (abs(r["micro_f1"] - 1.0) < 1e-9 and abs(r["macro_f1"] - 1.0) < 1e-9)
        ok &= good
        print(f"  τ={tau}: micro P/R/F1 = {r['micro_precision']:.4f}/{r['micro_recall']:.4f}/"
              f"{r['micro_f1']:.4f} | macro F1 {r['macro_f1']:.4f} | "
              f"含边界视频 {r['n_videos']} | 边界总数 {r['n_gt']}  {'✅' if good else '❌'}")
    # 排除机制核查：含边界视频数必须 < 总视频数
    n_videos_with_b = sum(1 for g, _ in pairs.values() if len(find_boundaries(g)) > 0)
    print(f"  排除机制核查：含边界视频 {n_videos_with_b} / 总 {len(pairs)} → "
          f"被排除 {len(pairs) - n_videos_with_b} 个（若记 0 则自检必不满分）")
    print(f"  ⇒ GT 自检 {'通过 ✅' if ok else '失败 ❌'}")

    # ---------------------------------------------------------------- ② 负对照
    print("\n" + "=" * 84)
    print("② 负对照 a —— 全漏（预测恒为 normal，不报任何边界）")
    print("=" * 84)
    all_normal = {k: (g, np.full_like(g, NORMAL)) for k, (g, _p) in pairs.items()}
    for tau in TAUS:
        r = eval_all(all_normal, tau)
        print(f"  τ={tau}: micro P/R/F1 = {r['micro_precision']:.4f}/{r['micro_recall']:.4f}/"
              f"{r['micro_f1']:.4f} | 含边界视频 {r['n_videos']} | 边界总数 {r['n_gt']}")

    print("\n" + "=" * 84)
    print("② 负对照 b —— 真值平移 d 帧（预测=平移后的真值），τ=3")
    print("=" * 84)
    print("  期望行为：容差内（d<=τ）F1≈1 且偏移被准确读出；超出容差后坍缩到 ~0。")
    print("  ⚠ 注意：这是**阶跃**而非渐变——容差型指标对「整体固定延迟」不敏感，")
    print("     超出 τ 后的残余匹配纯属巧合（~0.02-0.03），故「单调」不是正确的期望，")
    print("     该性质须作为已知缺陷随指标声明。延迟程度需另看 timeline_metrics 的 delay。")
    print(f"\n  {'d(帧)':>6}{'真值边界':>10}{'预测边界':>10}{'micro_P':>10}{'micro_R':>10}"
          f"{'micro_F1':>10}{'匹配对偏移均值':>16}")
    shift_rows = []
    for d in (0, 1, 2, 3, 4, 6, 10):
        shifted = {k: (g, shift_seq(g, d)) for k, (g, _p) in pairs.items()}
        r = eval_all(shifted, 3)
        offs = []
        for g, p in shifted.values():
            offs += matched_offsets(g, p, 3)
        mean_off = float(np.mean(offs)) if offs else float("nan")
        shift_rows.append((d, r["micro_f1"], mean_off))
        print(f"  {d:>6}{r['n_gt']:>10}{r['n_pred']:>10}{r['micro_precision']:>10.4f}"
              f"{r['micro_recall']:>10.4f}{r['micro_f1']:>10.4f}{mean_off:>16.3f}")
    in_tol = [x for x in shift_rows if x[0] <= 3]
    out_tol = [x for x in shift_rows if x[0] > 3]
    ok_in = all(abs(x[1] - 1.0) < 0.01 for x in in_tol)
    ok_off = all(abs(x[2] - x[0]) < 1e-6 for x in in_tol)
    ok_out = all(x[1] < 0.05 for x in out_tol)
    print(f"\n  ⇒ 容差内 F1≈1（d<=3）: {'✅' if ok_in else '❌'} ；"
          f"容差内偏移被准确读出（d=1/2/3→1/2/3）: {'✅' if ok_off else '❌'} ；"
          f"超容差坍缩至 <0.05: {'✅' if ok_out else '❌'}")
    print(f"  ⇒ 残余水平 {min(x[1] for x in out_tol):.4f}~{max(x[1] for x in out_tol):.4f}"
          f"（巧合匹配，非信号）→ 判读时 <0.05 的边界 F1 差异不可当真")

    print("\n" + "=" * 84)
    print("② 负对照 c —— 抖动（注入 k 处单帧毛刺 → 每处约 2 个假边界），τ=3")
    print("=" * 84)
    print(f"  {'k':>4}{'n_pred':>10}{'micro_P':>10}{'micro_R':>10}{'micro_F1':>10}")
    rng = np.random.default_rng(42)
    jit_rows = []
    for k in (0, 1, 2, 5, 10, 20):
        jit = {key: (g, jitter_seq(g, k, rng)) for key, (g, _p) in pairs.items()}
        r = eval_all(jit, 3)
        jit_rows.append((k, r["micro_precision"]))
        print(f"  {k:>4}{r['n_pred']:>10}{r['micro_precision']:>10.4f}"
              f"{r['micro_recall']:>10.4f}{r['micro_f1']:>10.4f}")
    ps = [x[1] for x in jit_rows]
    mono_p = all(ps[i] >= ps[i + 1] - 1e-9 for i in range(len(ps) - 1))
    print(f"  ⇒ 精确率随 k 单调不增: {'✅' if mono_p else '❌'}")

    # ---------------------------------------------------------------- 锚点数字
    print("\n" + "=" * 84)
    print("E1-full best 的边界指标（W1 对照基准）")
    print("=" * 84)
    per = {k: {tau: compute_video_boundary_metrics(g, p, tau) for tau in TAUS}
           for k, (g, p) in pairs.items()}
    agg = aggregate_boundary_metrics(per, taus=TAUS)
    print(f"  {'τ':>4}{'含边界视频':>11}{'真值边界':>10}{'预测边界':>10}"
          f"{'micro_P':>10}{'micro_R':>10}{'micro_F1':>10}{'macro_F1':>10}")
    for tau in TAUS:
        r = agg[f"tau{tau}"]
        print(f"  {tau:>4}{r['n_videos']:>11}{r['n_gt']:>10}{r['n_pred']:>10}"
              f"{r['micro_precision']:>10.4f}{r['micro_recall']:>10.4f}"
              f"{r['micro_f1']:>10.4f}{r['macro_f1']:>10.4f}")

    print("\n  按切换类型分列（τ=3，micro 口径）")
    by_type = {}
    for k, (g, p) in pairs.items():
        d = compute_video_boundary_metrics_by_type(g, p, 3)
        if not d:
            continue
        for tn, m in d.items():
            by_type.setdefault(tn, []).append(m)
    print(f"  {'类型':<18}{'视频数':>8}{'真值':>8}{'预测':>8}{'P':>10}{'R':>10}{'F1':>10}")
    for tn, ms in by_type.items():
        tp = sum(m["tp"] for m in ms)
        fp = sum(m["fp"] for m in ms)
        fn = sum(m["fn"] for m in ms)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        print(f"  {tn:<18}{len(ms):>8}{tp+fn:>8}{tp+fp:>8}{P:>10.4f}{R:>10.4f}{F:>10.4f}")

    out = os.path.join(BASE, "logs", "phase9", "boundary_metrics_e1full.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"aggregate": agg,
                   "shift_control": [{"d": d, "micro_f1": f, "mean_offset": o}
                                     for d, f, o in shift_rows],
                   "jitter_control": [{"k": k, "micro_precision": p} for k, p in jit_rows]},
                  fh, indent=2, ensure_ascii=False, default=float)
    print(f"\n已写出: {out}")


if __name__ == "__main__":
    main()
