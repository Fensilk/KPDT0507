# -*- coding: utf-8 -*-
"""从基线模型的实际误报里反推「难例类别清单」——Phase 11 Exp-1b 的输入。

**为什么不先验指定**：初稿说的"快速坐下、蹲下、弯腰、主动躺下"是直觉清单。
若直接照用，无从判断选得对不对、也漏掉可能更严重的类别。改成章程式做法：

  章程 = 用基线模型 E1-full 在 test 集上的逐帧预测，统计**类条件误报率**
         `rate_fall(c) = P(预测为 fall | GT 类别 = c)`
  取 rate 显著高于全局基准的类别作为难例池。

步骤：
  1. 读 `logs/phase9/e1_full/pv_best.pt`（逐视频 GT 与预测）与 NPZ 的 `labels_16`
  2. 按 16 类逐类累计 n / 被预测为 fall 数 / 被预测为 fallen 数
  3. 输出类条件误报率排名 + 全局基准（用于定阈值）
  4. 给出建议的难例池与"应排除"的类别（真事件类，不是负样本）

用法：python -u analysis/derive_hard_negatives.py
"""

import os
import sys
import json

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
PV = os.path.join(BASE, "logs", "phase9", "e1_full", "pv_best.pt")

# 16 类名（DATASET-omnifall/LABELS.md）
CLASSES = {
    0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
    5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other",
    10: "kneel_down", 11: "kneeling", 12: "squat_down", 13: "squatting",
    14: "crawl", 15: "jump",
}
# GT 里属于「真事件」的类别 —— 它们不该进负样本池
POSITIVE_CLASSES = {1, 2}          # fall, fallen
# 三元映射与 models/dataset.py 一致
FALL_16, FALLEN_16 = 1, 2


def norm(p):
    p = str(p).replace("\\", "/")
    for ext in (".mp4", ".avi", ".mkv"):
        if p.lower().endswith(ext):
            p = p[: -len(ext)]
    return p


def main():
    pv = torch.load(PV, weights_only=False, map_location="cpu")
    if "pv" in pv:
        pv = pv["pv"]

    npz = np.load(NPZ, mmap_mode="r", allow_pickle=True)
    labels16 = np.asarray(npz["labels_16"])
    starts = np.asarray(npz["video_start_indices"])
    vpaths = npz["video_paths"]
    index_of = {norm(p): i for i, p in enumerate(vpaths)}

    n_cls = 16
    n_frames = np.zeros(n_cls, dtype=np.int64)
    n_pred_fall = np.zeros(n_cls, dtype=np.int64)
    n_pred_fallen = np.zeros(n_cls, dtype=np.int64)
    n_pred_normal = np.zeros(n_cls, dtype=np.int64)
    n_videos_with_cls = np.zeros(n_cls, dtype=np.int64)
    matched = 0

    for key, v in pv.items():
        i = index_of.get(norm(key))
        if i is None:
            continue
        matched += 1
        gt16 = labels16[starts[i]: starts[i] + 80]
        pr = np.asarray(v["bridge_pred"])
        n = min(len(gt16), len(pr))
        gt16, pr = gt16[:n], pr[:n]
        for c in np.unique(gt16):
            m = gt16 == c
            n_frames[c] += int(m.sum())
            n_pred_fall[c] += int((pr[m] == 0).sum())
            n_pred_fallen[c] += int((pr[m] == 1).sum())
            n_pred_normal[c] += int((pr[m] == 2).sum())
            n_videos_with_cls[c] += 1

    print(f"匹配到 {matched} / {len(pv)} 个视频")
    print()
    print("=" * 96)
    print("类条件误报率 —— rate_fall(c) = P(预测为 fall | GT=c)，分母 = 该类在 test 集的总帧数")
    print("=" * 96)
    neg = [c for c in range(n_cls) if c not in POSITIVE_CLASSES]
    tot = sum(n_frames[c] for c in neg)
    tot_fp_fall = sum(n_pred_fall[c] for c in neg)
    base = tot_fp_fall / max(tot, 1) if tot else 0.0

    rows = []
    for c in range(n_cls):
        if n_frames[c] == 0:
            continue
        rf = n_pred_fall[c] / n_frames[c]
        rn = n_pred_fallen[c] / n_frames[c]
        fp = int(n_pred_fall[c])
        share = fp / max(tot_fp_fall, 1)
        rows.append((c, int(n_frames[c]), int(n_videos_with_cls[c]), rf, rn, fp, share))

    print("表 1  按「难度」排序 —— rate_fall(c) = P(预测为 fall | GT=c)")
    print(f"{'id':>3} {'类别':<12}{'帧数':>9}{'含该类视频':>11}"
          f"{'P(pred=fall|c)':>17}{'倍数':>8}{'P(pred=fallen|c)':>18}")
    for c, nf, nv, rf, rn, fp, share in sorted(rows, key=lambda r: -r[3]):
        mark = " ← 真事件" if c in POSITIVE_CLASSES else ""
        mult = f"{rf/base:.1f}x" if base else "n/a"
        print(f"{c:>3} {CLASSES[c]:<12}{nf:>9}{nv:>11}{rf:>17.4f}{mult:>8}{rn:>18.4f}{mark}")

    print(f"\n全局基准（全部非事件帧）: P(pred=fall|normal-ish) = {base:.4f}  (分母 {tot} 帧)")
    print(f"全部 fall 误报帧数（非事件类合计）: {tot_fp_fall}")

    print("\n" + "=" * 96)
    print("表 2  按「对全部 fall 误报的贡献占比」排序 —— 这才是「影响」的正确度量")
    print("（rate 只说「这类有多难」，贡献占比才说「这类贡献了多少错误」）")
    print("=" * 96)
    print(f"{'id':>3} {'类别':<12}{'误报帧数':>11}{'占全部 fall 误报':>19}"
          f"{'难度倍数':>10}{'含该类视频':>12}")
    for c, nf, nv, rf, rn, fp, share in sorted(rows, key=lambda r: -r[6]):
        if c in POSITIVE_CLASSES:
            continue
        print(f"{c:>3} {CLASSES[c]:<12}{fp:>11}{share*100:>18.1f}%"
              f"{rf/base:>9.1f}x{nv:>12}")

    print("\n" + "=" * 96)
    print("难例池筛选：难度倍数 >= 2x 且贡献占比 >= 5%（两条都要过）")
    print("=" * 96)
    pool = [(c, rf, share) for c, nf, nv, rf, rn, fp, share in rows
            if c not in POSITIVE_CLASSES and rf >= 2 * base and share >= 0.05]
    pool.sort(key=lambda t: -t[2])
    for c, rf, share in pool:
        print(f"  {CLASSES[c]:<12} 难度 {rf/base:>4.1f}x   贡献占比 {share*100:>5.1f}%   "
              f"n_frames={n_frames[c]:>6}  n_videos={n_videos_with_cls[c]}")
    rejected = [(c, rf, share) for c, nf, nv, rf, rn, fp, share in rows
                if c not in POSITIVE_CLASSES and not (rf >= 2 * base and share >= 0.05)]
    if rejected:
        print("\n  未入选（说明其「难」或「少」不足以影响全局）：")
        for c, rf, share in sorted(rejected, key=lambda t: -t[2]):
            why = []
            if rf < 2 * base:
                why.append(f"难度不足({rf/base:.1f}x)")
            if share < 0.05:
                why.append(f"占比不足({share*100:.1f}%)")
            print(f"    {CLASSES[c]:<12} {', '.join(why)}")

    print("\n应排除（真事件类，是正样本不是负样本）："
          + ", ".join(CLASSES[c] for c in sorted(POSITIVE_CLASSES)))

    out = os.path.join(BASE, "logs", "phase9", "hard_negatives.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "global_baseline_rate_fall": base,
            "total_fall_fp_frames": int(tot_fp_fall),
            "per_class": [{"id": c, "name": CLASSES[c], "n_frames": nf,
                           "n_videos": nv, "rate_fall": rf, "rate_fallen": rn,
                           "fall_fp_frames": fp, "share_of_fall_fp": share}
                          for c, nf, nv, rf, rn, fp, share in rows],
            "suggested_pool": [CLASSES[c] for c, _, _ in pool],
            "pool_criterion": "rate_fall >= 2x baseline AND share_of_fall_fp >= 5%",
        }, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\n已写出: {out}")


if __name__ == "__main__":
    main()
