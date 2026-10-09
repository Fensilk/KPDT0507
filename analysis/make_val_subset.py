# -*- coding: utf-8 -*-
"""从 val.csv 抽一个 val 子集（供 Phase 11 P2 的离线选点第 ① 段）。

═══ 为什么需要它 ═══

P2 的离线选点是两段式：① 48 个密集 ckpt 先在**小 val** 上评分 → 取 top-8；② top-8 在**全量 1200** 上重评。
第 ① 段需要一个够便宜又够有代表性的子集。

═══ 两条实测依据（见 1006 附加分析）═══

1. **不需要人为"配平"**：`val.csv` 全量与 `test.csv` 的**帧级三元分布差 ≤0.63pp**
   （fall 6.35/6.78%，fallen 8.31/7.69%，normal 85.34/85.53%）——两者本就是同一池子的 i.i.d. 两个 1200。
   当前旧 150 缓存的 fallen 密度是 test 的 **1.63×**（正好复现规划 §6.2 的 1.62×），
   那是 **150 的抽样噪声（≈1.9σ）**，不是设计缺陷。**取大样本即可，不必对齐。**
2. ⚠ **绝不可取"前 N 行"**：`val.csv` **按类文件夹排序**
   （fall → fallen → lie_down → lying → other → sit_down → …）。
   直接取前 600 行会**完全没有** standing / stand_up / walking。**必须打散。**

用法：
    python analysis/make_val_subset.py --n 300 --out data/val_subset300.csv
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "val.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="子集视频数")
    ap.add_argument("--seed", type=int, default=42, help="打散种子（固定，保证可复现）")
    ap.add_argument("--val_csv", default=VAL_CSV)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.val_csv)
    rels = df["path"].astype(str).str.strip().tolist()
    if args.n > len(rels):
        raise SystemExit(f"--n {args.n} > val.csv 的 {len(rels)} 条")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rels))
    sel = [rels[i] for i in perm[: args.n]]

    # 自检：子集的顶层类别分布必须覆盖全部类别（若取"前 N 行"会缺一半）
    top = lambda p: p.split("/")[0]
    got = {}
    for p in sel:
        got[top(p)] = got.get(top(p), 0) + 1
    all_cls = sorted({top(p) for p in rels})
    missing = [c for c in all_cls if c not in got]
    print(f"抽 {len(sel)}/{len(rels)}（seed={args.seed}）")
    print(f"  类别覆盖 {len(got)}/{len(all_cls)}" + (f"  ❌ 缺 {missing}" if missing else "  ✅ 全覆盖"))
    print("  分布:", {k: got[k] for k in all_cls if k in got})
    if missing:
        raise SystemExit("子集类别覆盖不全 —— 抽取方式有问题，停。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame({"path": sel}).to_csv(args.out, index=False)
    print(f"已写出 {args.out}")


if __name__ == "__main__":
    main()
