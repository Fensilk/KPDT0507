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


def curves_for_axis(pv, cls, protocol="voted"):
    """扫阈值 → (recall[], precision[], f1[]) 曲线。

    分数 = 逐帧 softmax 概率中该类的值；真值正类 = 该帧三元标签等于 cls。
    逐帧计算，覆盖全部视频的全部 80 帧（去重口径，与 merged_raw 一致）。
    """
    # 协议：
    #   voted —— 多窗口在重叠处取均值后的 80 帧/视频（E1 侧 pv 的口径，跨阶段可比的默认）
    #   flat  —— 所有窗口**原样拼接**（Phase 7/8 侧报告 test_results 用的口径）
    #   ⚠ 复核某阶段的结论时**必须用该阶段自己的协议**，否则是跨口径比较。
    key_s = "probs" if protocol == "voted" else "flat_probs"
    key_g = "ternary_gt" if protocol == "voted" else "flat_gt"
    scores, gts = [], []
    for v in pv.values():
        if key_s not in v:
            raise KeyError(
                f"pv 缺少 '{key_s}' —— voted 口径需新版 eval 重跑；"
                f"flat 口径需 eval_p8_probs.py 的产物。旧 pv 只存 argmax 标签，无法事后扫阈值。")
        scores.append(np.asarray(v[key_s])[:, cls])
        gts.append((np.asarray(v[key_g]) == cls).astype(np.int8))
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


# 支配关系比较的网格：只用**共同支撑且操作相关**的召回区间。
# 曲线在 R≈0 只由 1 个最自信的预测点支撑、在 R≈1 由全部预测支撑，两端的比较没有意义
# （np.interp 在区间外是**端点钳制**，会拿钳制值去比，结论失真）。
DOM_LO, DOM_HI, DOM_N = 0.50, 0.95, 40


def dominance(a, b):
    """在共同召回区间上逐点比 a 与 b；d = a 的精度 − b 的精度（>0 表示 a 更好）。

    ⚠ 返回值**必须落盘**（`_dominance`）：本函数算的是 40 点插值均值，
    与 `REC_LEVELS` 那 7 个采样点的均值**不是同一个数**（实测差约 0.003）。
    2026-09-30 曾因此出现"文档写了 +0.0608、但归档 JSON 只能算出 +0.0640"的
    不可复现缺口——文档数字来自这里，所以这里必须被持久化。

    None 表示共同支撑区间过窄、无法比。
    """
    lo = max(a["recall"][0], b["recall"][0], DOM_LO)
    hi = min(a["recall"][-1], b["recall"][-1], DOM_HI)
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, DOM_N)
    d = (np.interp(grid, a["recall"], a["precision"])
         - np.interp(grid, b["recall"], b["precision"]))
    return {
        "lo": float(grid[0]), "hi": float(grid[-1]), "n_grid": int(DOM_N),
        "mean": float(d.mean()), "median": float(np.median(d)),
        "min": float(d.min()), "max": float(d.max()),
        "delta": [float(x) for x in d],
        # ⚠ 支配 = 处处 **≥** 且**至少一处 >**（标准 Pareto 支配）。
        #   只写"处处 ≥"是错的：d 全为 0 时（同一模型自己比自己）两个方向会**同时**成立，
        #   于是"支配"与"被支配"都返回 True —— judge_w1_arm.py 的"臂=基准"自测
        #   当场把 A0 判成了【硬否决】。加 `any(>)` 后，全等曲线两个方向都为 False（不可分辨）。
        "a_dominates_b": bool((d >= -1e-9).all() and (d > 1e-9).any()),
        "b_dominates_a": bool((d <= 1e-9).all() and (d < -1e-9).any()),
        "identical": bool(np.abs(d).max() <= 1e-9),
    }


def dominance_line(na, nb, dm):
    """把 dominance() 的结果写成人读的一行。"""
    if dm["a_dominates_b"]:
        return f"{na} 支配 {nb}（同召回下精度处处不低，均值高 {dm['mean']:+.4f}）"
    if dm["b_dominates_a"]:
        return f"{nb} 支配 {na}（同召回下精度处处不低，均值高 {-dm['mean']:+.4f}）"
    # ⚠ 交叉方向**必须由数据判定**，不能写死。
    #   本块曾硬编码"a 在低召回端更好、b 在高召回端更好"，
    #   结果与 mean(d) 的符号自相矛盾（提示说 a 好，均值说 b 好）。
    d = dm["delta"]
    mean_txt = f"全区间均值 {dm['mean']:+.4f}（正 = {na} 更好）"
    span = f"逐点差区间 [{dm['min']:+.4f}, {dm['max']:+.4f}]"
    # ⚠ 两端同向 ≠ 交叉。未构成支配但两端都偏向同一臂，只可能是**中段回落**。
    #   2026-10-01 曾把 1b1 的 fallen 轴印成"交叉（两端都是本臂更好）"——
    #   两端都说同一臂好却叫"交叉"，会让人以为曲线交叉，实为中间下沉。
    if d[0] > 0 and d[-1] > 0:
        return f"{na} 两端更好但**中段回落**（{span}）；{mean_txt}"
    if d[0] < 0 and d[-1] < 0:
        return f"{nb} 两端更好但**中段回落**（{span}）；{mean_txt}"
    lo_who = na if d[0] > 0 else nb
    hi_who = na if d[-1] > 0 else nb
    return (f"交叉：低召回端（R≈{dm['lo']:.2f}）{lo_who} 更好；"
            f"高召回端（R≈{dm['hi']:.2f}）{hi_who} 更好；{mean_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="name=path 或 path")
    ap.add_argument("--axes", default="fall,fallen",
                    help="逗号分隔：fall,fallen,normal（默认前两个）")
    ap.add_argument("--protocol", default="voted", choices=["voted", "flat"],
                    help="voted=多窗口均值（默认，E1 侧口径）；flat=窗口拼接（Phase 7/8 侧口径）。"
                         "复核某阶段结论时必须用该阶段自己的协议。")
    ap.add_argument("--out", default=os.path.join(BASE, "logs", "phase11", "prec_rec_reeval", "curves"),
                    help="曲线图输出前缀（默认 logs/phase11/prec_rec_reeval/curves）")
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
            c = curves_for_axis(pv, cls, args.protocol)
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
            doms = {}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    na, nb = names[i], names[j]
                    dm = dominance(curs[na], curs[nb])
                    if dm is None:
                        print(f"    {na} vs {nb}: 共同支撑区间过窄，无法比")
                        continue
                    doms[f"{na}|{nb}"] = dm
                    print("    " + dominance_line(na, nb, dm))
            # ⚠ 必须落盘：文档里的支配均值来自 dominance()，不写下来就不可复现。
            out[ax]["_dominance"] = doms
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
                    c = curves_for_axis(pv, cls, args.protocol)
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
