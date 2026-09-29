# -*- coding: utf-8 -*-
"""prec-rec **前沿曲线**比较 —— 替代"单点比大小"。

**为什么需要它**（2026-09-30 实测结论）：
同一模型的各 checkpoint 在 fallen 轴上的 (prec, rec) 几乎完全落在一条陡峭前沿上
（二者相关系数 r = −0.955 / −0.988），而 `avg_f1` 极差仅 0.035 —— 即 checkpoint 之间
的差异**主要是沿前沿滑动，不是能力差异**。位置由训练步数/种子/微小扰动决定。

因此"拿 A 的落点比 B 的落点"等于比较两次抽签。**曲线对"落在哪一点"不变**，
测的是前沿本身（能力），而不是这次的落点（表现）。

用法：
  python -u analysis/prec_rec_curves.py A0=logs/phase11/a0_2000/pv_best.pt \
                                       1a=logs/phase11/w1_1a/pv_best.pt \
                                       a0cos=logs/phase11/w1_a0cos/pv_best.pt
  可选 --axis fallen|fall|normal（默认两轴都出）  --out 前缀

前提：pv 里必须有 `probs`（逐帧概率）。旧 pv 只存 argmax 标签，**无法事后补**——
需用新版 eval_p9e1_dual.py 重跑一次才会带上。
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AXES = {0: "fall", 1: "fallen", 2: "normal"}
# 报告曲线时取这些召回水平比较精度（覆盖我们关心的区间）
REC_LEVELS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)


def load_pv(spec):
    """'name=path' → (name, {rel: {...}})"""
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        name, path = os.path.basename(os.path.dirname(spec)), spec
    d = torch.load(path, weights_only=False, map_location="cpu")
    if "pv" in d:
        d = d["pv"]
    return name, d


def curves_for_axis(pv, cls):
    """扫阈值 → (recall[], precision[], f1[]) 曲线。

    分数 = 逐帧 softmax 概率中该类的值；真值正类 = 该帧三元标签等于 cls。
    逐帧计算，覆盖全部视频的全部 80 帧（去重口径，与 merged_raw 一致）。
    """
    scores, gts = [], []
    for v in pv.values():
        if "probs" not in v:
            raise KeyError(
                "pv 缺少 'probs'（逐帧概率）——需用新版 eval_p9e1_dual.py 重跑评测。"
                "旧 pv 只存 argmax 标签，无法事后扫阈值。")
        scores.append(np.asarray(v["probs"])[:, cls])
        gts.append((np.asarray(v["ternary_gt"]) == cls).astype(np.int8))
    s = np.concatenate(scores)
    g = np.concatenate(gts)
    n_pos = int(g.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-s)
    g_sorted = g[order]
    tp_cum = np.cumsum(g_sorted)
    # 在排序后的位置上，前 k 个预测为正 → prec = tp/k, rec = tp/n_pos
    k = np.arange(1, len(g) + 1)
    prec = tp_cum / k
    rec = tp_cum / n_pos
    # 保持 recall 单调不减（同 recall 取最大 prec）
    keep = np.concatenate([[True], np.diff(rec) > 0])
    prec_u, rec_u = prec[keep], rec[keep]
    f1 = 2 * prec_u * rec_u / np.maximum(prec_u + rec_u, 1e-12)
    return {"recall": rec_u, "precision": prec_u, "f1": f1,
            "n_frames": int(len(g)), "n_pos": n_pos,
            "best_f1": float(f1.max()), "best_f1_at_rec": float(rec_u[f1.argmax()])}


def prec_at_recall(cur, target):
    """在给定召回水平上取精度（曲线 recall 单调不减，用插值）。"""
    r, p = cur["recall"], cur["precision"]
    if target > r[-1]:
        return None
    return float(np.interp(target, r, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="name=path 或 path")
    ap.add_argument("--axes", default="fall,fallen",
                    help="逗号分隔：fall,fallen,normal（默认前两个）")
    ap.add_argument("--out", default=None, help="曲线图输出前缀（默认 logs/phase11/prec_rec）")
    ap.add_argument("--json", default=None, help="把数值写成 JSON")
    args = ap.parse_args()

    arms = [load_pv(s) for s in args.specs]
    axes = [AXES[int(a)] if a.isdigit() else a.strip() for a in args.axes.split(",")]
    cls_of = {v: k for k, v in AXES.items()}

    out = {}
    for ax in axes:
        cls = cls_of[ax]
        print("=" * 92)
        print(f"轴 = {ax}（类 {cls}）   帧级 prec-rec 前沿；每臂为该 ckpt 的整条曲线")
        print("=" * 92)
        curs = {}
        for name, pv in arms:
            c = curves_for_axis(pv, cls)
            if c is None:
                print(f"  {name}: 无正样本，跳过"); continue
            curs[name] = c
        # 表 1：各臂的曲线极值
        print(f"  {'臂':<12}{'最优 F1':>10}{'@recall':>10}   " +
              "".join(f"{'P@R=%.2f' % t:>11}" for t in REC_LEVELS))
        for name, c in curs.items():
            row = f"  {name:<12}{c['best_f1']:>10.4f}{c['best_f1_at_rec']:>10.3f}   "
            for t in REC_LEVELS:
                v = prec_at_recall(c, t)
                row += f"{(v if v is not None else float('nan')):>11.4f}"
            print(row)
        out[ax] = {n: {t: prec_at_recall(c, t) for t in REC_LEVELS} for n, c in curs.items()}
        out[ax]["_best_f1"] = {n: c["best_f1"] for n, c in curs.items()}

        # 表 2：两两支配关系（在共同覆盖的召回区间上逐点比）
        names = list(curs)
        if len(names) > 1:
            print(f"\n  ── 支配关系（在共同召回区间上，谁在每一点都不低）")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = curs[names[i]], curs[names[j]]
                    lo = max(a["recall"][0], b["recall"][0])
                    hi = min(a["recall"][-1], b["recall"][-1])
                    if hi <= lo:
                        print(f"    {names[i]} vs {names[j]}: 召回区间不重叠，无法比"); continue
                    grid = np.linspace(lo, hi, 40)
                    pa = np.interp(grid, a["recall"], a["precision"])
                    pb = np.interp(grid, b["recall"], b["precision"])
                    d = pa - pb
                    if (d >= -1e-9).all():
                        print(f"    {names[i]} 支配 {names[j]}（同召回下精度处处不低，"
                              f"均值高 {d.mean():+.4f}）")
                    elif (d <= 1e-9).all():
                        print(f"    {names[j]} 支配 {names[i]}（均值高 {-d.mean():+.4f}）")
                    else:
                        print(f"    交叉：{names[i]} 在低召回端更好，{names[j]} 在高召回端更好"
                              f"（均值差 {d.mean():+.4f}）")
            print("\n  ⚠ 若两条曲线交叉或均值差很小，则两臂**不可分辨**——"
                  "不要用某个单一落点判胜负。")
        print()

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(axes)
            fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
            for k, ax_name in enumerate(axes):
                a = axs[0][k]
                cls = cls_of[ax_name]
                for name, pv in arms:
                    c = curves_for_axis(pv, cls)
                    if c is None:
                        continue
                    a.plot(c["recall"], c["precision"], lw=2, label=name)
                a.set_title(f"{ax_name}: frame-level prec-recall frontier")
                a.set_xlabel("recall"); a.set_ylabel("precision")
                a.grid(alpha=.3); a.legend(fontsize=9); a.set_xlim(0, 1); a.set_ylim(0, 1)
            plt.tight_layout()
            for suf in ("png",):
                fig.savefig(f"{args.out}.{suf}", dpi=130)
            print(f"曲线图已保存: {args.out}.png")
        except Exception as e:
            print(f"[WARN] 绘图失败（数值结果不受影响）: {e}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
        print(f"数值已写出: {args.json}")


if __name__ == "__main__":
    main()
