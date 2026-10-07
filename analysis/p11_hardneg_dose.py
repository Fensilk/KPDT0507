# -*- coding: utf-8 -*-
"""Exp-1b 难例采样的**剂量诊断** —— 复算 sampler 权重，扫 α，并算"有效梯度权重"。

为什么要这个脚本（2026-10-01）：
Exp-1b-1（`--hard_neg_alpha 2.0`）实测只把难例窗口的采样占比从 6.65% 抬到 7.48%（×1.125），
远低于 `(1+α)=3` 的设计意图。直觉会以为"把 ×3 再乘一遍就好了"——但代码里那个 ×3
**本来就是乘在逆频权重之上的**（`w = inv_freq[ter] × (1+α)`），所以那样改等于没改。

真正的原因是**两个机制在打架**：
  ① sampler 按 `逆频三元权重 × (1+α)` 抽窗口；
  ② 损失用**三元** CE 类权重 `[1.336, 1.532, 0.132]`（`compute_class_weights`）。
而 `lie_down` 是 **normal** 子类 → 它的逆频权重与 CE 权重都取 normal 那一档（最小）。
normal 窗口占 ~84%，其单窗逆频权重比 fall 窗口小约 1 个数量级；
把一个本来就小的权重乘 3，在"总权重"里依然是小量。

本脚本把这件事算清楚，并给出：**要让难例占到 X% 的采样，α 该取多少**，
以及**采样占比 × CE 权重 = 有效梯度权重**（后者才是真正影响学习的东西）。

用法（项目根运行，需 data/e1_2000_manifest.csv 与主 NPZ）：
    python -u analysis/p11_hardneg_dose.py
    python -u analysis/p11_hardneg_dose.py --manifest data/e1_6000_manifest.csv --pool lie_down,other
"""

import os
import sys
import csv
import argparse

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from models.dataset import CLASS_NAMES  # noqa: E402

DEFAULT_NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
DEFAULT_MANIFEST = os.path.join(BASE, "data", "e1_2000_manifest.csv")

WINDOW, STRIDE = 64, 16


def build_window_labels(manifest, npz):
    """复刻 train_p9e1.py 的 Mp4WindowCache.meta 与 sampler 用的中心帧类别。

    meta 为 (视频, st, ws)，ws ∈ range(0, 80-window+1, stride)；
    sampler 取中心帧 cls16 = labels16[st + ws + window//2]（见 build_hard_neg_sampler）。
    """
    d = np.load(npz, mmap_mode="r", allow_pickle=True)
    labels16 = d["labels_16"]
    paths = [str(p) for p in d["video_paths"]]
    vsi = d["video_start_indices"]
    idx = {p: i for i, p in enumerate(paths)}

    with open(manifest, newline="", encoding="utf-8") as fh:
        rels = [r["path"] for r in csv.DictReader(fh)]

    half = WINDOW // 2
    cls16, starts = [], []
    n_skip = 0
    for rel in rels:
        if rel not in idx:
            n_skip += 1
            continue
        st = int(vsi[idx[rel]])
        starts.append(st)
        for ws in range(0, 80 - WINDOW + 1, STRIDE):
            cls16.append(int(labels16[st + ws + half]))
    return np.array(cls16, dtype=np.int64), np.array(starts, dtype=np.int64), labels16, n_skip


def ternary_of_cls16(c):
    """与 build_hard_neg_sampler 完全同式：16 类的 1/2 → 三元 0/1，其余 → 2。"""
    return np.where(c == 1, 0, np.where(c == 2, 1, 2)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--pool", default="lie_down",
                    help="难例池，逗号分隔的 16 类名（默认 lie_down）")
    ap.add_argument("--targets", default="5,10,15,20,30",
                    help="想知道'难例占到 X%% 采样'需要多大 α（逗号分隔）")
    args = ap.parse_args()

    cls16, starts, labels16, n_skip = build_window_labels(args.manifest, args.npz)
    ter = ternary_of_cls16(cls16)
    pool_names = [s.strip() for s in args.pool.split(",") if s.strip()]
    pool_ids = [CLASS_NAMES[n] for n in pool_names]
    mask = np.isin(cls16, pool_ids)

    n = len(cls16)
    print(f"清单: {os.path.basename(args.manifest)}  窗口 {n}（跳过 {n_skip} 个不在 NPZ 的视频）")
    print(f"难例池: {{{', '.join(pool_names)}}}  命中窗口 {int(mask.sum())}/{n} ({100*mask.mean():.2f}%)")
    print()

    # ---- 复刻 sampler 权重 ----
    counts = np.bincount(ter, minlength=3).astype(np.float64)
    inv = 1.0 / (counts + 1e-6)
    base = inv[ter]
    print("三元窗口计数: fall=%d  fallen=%d  normal=%d" % tuple(counts))
    print("单窗逆频权重: fall=%.6f  fallen=%.6f  normal=%.6f  （normal/fall = 1:%.1f）"
          % (inv[0], inv[1], inv[2], inv[0] / inv[2]))
    print()

    # ---- 复刻 CE 类权重（compute_class_weights：全 80 帧逆频，归一化到和为 3）----
    ce_counts = np.zeros(3)
    for st in starts:
        ce_counts += np.bincount(ternary_of_cls16(labels16[st: st + 80]), minlength=3)
    ce = ce_counts.sum() / (3.0 * ce_counts + 1e-9)
    ce = ce / ce.sum() * 3.0
    print("CE 类权重（全 80 帧）: fall=%.3f  fallen=%.3f  normal=%.3f" % tuple(ce))
    print("  → 与训练日志的 [1.336, 1.532, 0.132] 对照（应一致）")
    print()

    # ---- 扫 α ----
    print("α   (1+α)   难例窗口采样占比   对 fall 窗口的『有效梯度权重』之比")
    print("-" * 78)
    rows = []
    for alpha in (0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 12.0, 19.0):
        w = base * np.where(mask, 1.0 + alpha, 1.0)
        share = w[mask].sum() / w.sum()
        # 每类窗口的采样概率（按权重归一）
        p = w / w.sum()
        # 有效梯度权重 = 采样概率 × 该类 CE 类权重
        eff_hard = p[mask].mean() * ce[2]                      # 难例窗口都是 normal
        eff_fall = p[ter == 0].mean() * ce[0]
        eff_norm = p[(ter == 2) & ~mask].mean() * ce[2]
        rows.append((alpha, share, eff_hard, eff_fall, eff_norm))
        print("  %-5.1f %-8.1f %8.2f%%          hard/fall = %6.3f   (hard=%.2e fall=%.2e norm=%.2e)"
              % (alpha, 1 + alpha, 100 * share, eff_hard / eff_fall, eff_hard, eff_fall, eff_norm))
    print()
    print("⚠ α=2 那行应与 1b1 启动日志的『采样后占比 7.48%』一致 —— 对得上才说明本复刻可信。")
    print()

    # ---- 反解：要让难例占到 X%，α 需要多大 ----
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        w = base * np.where(mask, 1.0 + mid, 1.0)
        if w[mask].sum() / w.sum() < 0.5:
            lo = mid
        else:
            hi = mid
    A = base[mask].sum()
    B = base.sum() - A
    print("要让难例窗口占到目标采样比例，α 需取：")
    for t in [float(x) / 100 for x in args.targets.split(",")]:
        # share = m·A/(m·A+B) = t  →  m = t·B/((1-t)·A)
        m = t * B / ((1 - t) * A)
        print("   目标 %5.1f%%  →  (1+α) = %6.2f   即 α ≈ %.1f" % (100 * t, m, m - 1))
    print()
    print("参考：A = Σ难例窗口逆频权重 = %.4f，B = Σ其余 = %.4f，A/B = %.5f" % (A, B, A / B))
    print("     （A/B 才是剂量弱的根源：难例窗口占**计数**的 {:.2f}%，"
          "却只占总**权重**的 {:.2f}%）".format(100 * mask.mean(), 100 * A / (A + B)))


if __name__ == "__main__":
    main()
