"""
E1@2000 vs E3@2000 匹配对 · fallen 轴窗口级归因 —— 检验 方案 D 的前提。

方案 D 前提：端到端(E1)相对两阶段(E3@2000)的静态轴增益，是否集中在
"窗口内含 fall→fallen 因果链"的那部分 fallen 帧上？若集中 → 事件窗采样保留信号；
若分散到"前因超窗/无前因"的 fallen 帧 → 端到端学到的是不依赖窗口内容的静态判别，
砍纯 normal 窗口反而会删掉承载它的反证。

定义（逐视频帧级）：
  - causal   fallen 帧：最近一次 GT-fall 距它 ≤63 帧（存在一个 64 帧窗口同时含 fall 与 fallen）
  - noncausal fallen 帧：前因超窗或整段无 GT-fall（GT 判 fallen 但无窗口内前因）
  - normal-FP：GT-normal 帧被判 fallen；按"所在视频是否含 GT-fall"与 lying(16类=6)细分

两个 pv 均须为同口径（stride8 / 逐帧投票 merge / 每视频 80 帧）：
  - E1@2000: logs/phase9/e1_rulev2/pv_cache.pt（eval_p9e1_rulev2.py 产物）
  - E3@2000: 本脚本用 load_and_infer 现建并缓存 logs/phase9/e3_rulev2/cache/E3-2000.pt

Usage:
    python experiments/analyze_e1_e3_attribution.py          # E3 缓存缺则先建；E1 不全则提示等待
"""
import os
import sys
import json

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from eval_rule_c import load_and_infer  # noqa: E402

E1_CACHE = os.path.join(BASE, "logs", "phase9", "e1_rulev2", "pv_cache.pt")
E3_CACHE = os.path.join(BASE, "logs", "phase9", "e3_rulev2", "cache", "E3-2000.pt")
OUT = os.path.join(BASE, "logs", "phase9", "e1_rulev2", "attribution_e1_vs_e3.json")

CTX = 63  # 前因距离 ≤63 → 存在含两者的 64 帧窗口


def build_e3_pv():
    os.makedirs(os.path.dirname(E3_CACHE), exist_ok=True)
    if os.path.exists(E3_CACHE):
        pv = torch.load(E3_CACHE, map_location="cpu", weights_only=False)
        print(f"[E3@2000] load cached {len(pv)} videos <- {E3_CACHE}", flush=True)
        return pv
    pv_vi = load_and_infer(
        os.path.join(BASE, "logs", "phase9", "e3_stage2_2000", "best_model.pt"),
        os.path.join(BASE, "data", "omnifall_dinov2_finetuned_frame.npz"),
        "DATASET-omnifall/splits/syn/random", use_diff=True, T=64, stride=8, bs=32)
    d = np.load(os.path.join(BASE, "data", "omnifall_dinov2_finetuned_frame.npz"),
                allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    pv = {vp[vi]: val for vi, val in pv_vi.items()}
    torch.save(pv, E3_CACHE)
    print(f"[E3@2000] inferred & cached {len(pv)} videos -> {E3_CACHE}", flush=True)
    return pv


def per_video_masks(gt, pred, labels16):
    """返回该视频的两个结构：fallen 帧按因果分组；normal→fallen FP 标记。"""
    out = {"causal_fallen": [], "noncausal_fallen": [],          # 元素 (is_pred_fallen)
           "fp_fallen": [],                                      # normal 帧被判 fallen
           "video_has_fall": bool((gt == 0).any())}
    n = len(gt)
    last_fall = -10**9
    for f in range(n):
        if gt[f] == 0:
            last_fall = f
        elif gt[f] == 1:
            causal = (f - last_fall) <= CTX
            out["causal_fallen" if causal else "noncausal_fallen"].append(int(pred[f] == 1))
        else:  # normal
            if pred[f] == 1:
                is_lying = int(labels16[f] == 6)
                out["fp_fallen"].append({"has_fall": bool((gt == 0).any()), "lying": is_lying})
    return out


def summarize(name, pv_e1, pv_e3, paths):
    agg = {m: {"causal_n": 0, "causal_rec": 0.0, "noncausal_n": 0, "noncausal_rec": 0.0,
               "fp_total": 0, "fp_vid_fall": 0, "fp_vid_nofall": 0,
               "fp_lying_vid_fall": 0, "fp_lying_vid_nofall": 0,
               "fallen_tp": 0, "fallen_fp": 0, "fallen_fn": 0}
           for m in ("E1@2000", "E3@2000")}
    pvs = {"E1@2000": pv_e1, "E3@2000": pv_e3}
    for rel in paths:
        pv1 = pvs["E1@2000"][rel]; pv3 = pvs["E3@2000"][rel]
        labels16 = pv3["labels_16"]
        m1 = per_video_masks(pv1["ternary_gt"], pv1["bridge_pred"], labels16)
        m3 = per_video_masks(pv3["ternary_gt"], pv3["bridge_pred"], labels16)
        for model, m in (("E1@2000", m1), ("E3@2000", m3)):
            a = agg[model]
            for group_key, counter in (("causal_fallen", "causal"), ("noncausal_fallen", "noncausal")):
                lst = m[group_key]
                if lst:
                    agg[model][counter + "_n"] += len(lst)
                    agg[model][counter + "_rec"] += sum(lst)
            for fp in m["fp_fallen"]:
                a["fp_total"] += 1
                if fp["has_fall"]:
                    a["fp_vid_fall"] += 1
                    a["fp_lying_vid_fall"] += fp["lying"]
                else:
                    a["fp_vid_nofall"] += 1
                    a["fp_lying_vid_nofall"] += fp["lying"]
    # rec 汇总
    for model in ("E1@2000", "E3@2000"):
        a = agg[model]
        for c in ("causal", "noncausal"):
            a[c + "_rec"] = a[c + "_rec"] / a[c + "_n"] if a[c + "_n"] else None
    return agg


def main():
    os.chdir(BASE)
    pv_e3 = build_e3_pv()

    if not os.path.exists(E1_CACHE):
        sys.exit("[E1@2000] pv 缓存尚不存在（eval_p9e1_rulev2.py 未跑完）")
    pv_e1 = torch.load(E1_CACHE, map_location="cpu", weights_only=False)
    n_e1 = len(pv_e1)
    print(f"[E1@2000] cache videos = {n_e1}", flush=True)
    if n_e1 < 1200:
        sys.exit(f"[E1@2000] pv 缓存不完整 ({n_e1}/1200)——等全量 eval 完成后重跑本脚本")

    paths = sorted(set(pv_e1) & set(pv_e3))
    print(f"对齐视频数 = {len(paths)}")
    agg = summarize("attribution", pv_e1, pv_e3, paths)

    print("\n" + "=" * 74)
    print("fallen 轴归因：E1@2000 vs E3@2000（同 2000 训练数据 · 全量 test · merged 口径）")
    print("=" * 74)
    print("\n── GT-fallen 帧召回，按前因分组 ──")
    print(f"{'组':<10s} {'帧数':>7s} {'E3@2000':>9s} {'E1@2000':>9s} {'Δ(E1-E3)':>10s}")
    for c, label in (("causal", "前因在窗(≤63)"), ("noncausal", "前因超窗/无")):
        n = agg["E1@2000"][c + "_n"]
        r3 = agg["E3@2000"][c + "_rec"]; r1 = agg["E1@2000"][c + "_rec"]
        d = (r1 - r3) if (r1 is not None and r3 is not None) else None
        print(f"{label:<10s} {n:>7,d} {r3:>9.4f} {r1:>9.4f} "
              f"{(f'+{d:.4f}' if d and d > 0 else (f'{d:.4f}' if d else '--')):>10s}")

    print("\n── normal→fallen 误报（FP），按视频是否含 GT-fall 细分 ──")
    print(f"{'子集':<22s} {'E3@2000':>9s} {'E1@2000':>9s} {'Δ(E1-E3)':>10s}")
    for key, label in (("fp_vid_fall", "含fall视频·normal FP"),
                       ("fp_vid_nofall", "无fall视频·normal FP"),
                       ("fp_lying_vid_fall", " 其中 lying·含fall"),
                       ("fp_lying_vid_nofall", " 其中 lying·无fall")):
        v3 = agg["E3@2000"][key]; v1 = agg["E1@2000"][key]
        print(f"{label:<22s} {v3:>9,d} {v1:>9,d} {v1 - v3:>+10,d}")
    print(f"\nnormal→fallen FP 总数:  E3@2000 {agg['E3@2000']['fp_total']:,d}  "
          f"E1@2000 {agg['E1@2000']['fp_total']:,d}  Δ {agg['E1@2000']['fp_total']-agg['E3@2000']['fp_total']:+,d}")

    print("\n判读（方案 D 前提检验）：")
    print(" 1) E1 的 fallen 召回增益若集中在'前因在窗'组 → 端到端信号确实在事件窗内，D 方向对；")
    print("    若'前因超窗/无'组也有增益 → E1 学到不依赖窗口的静态判别，砍 normal 会直接伤它。")
    print(" 2) E1 的 fallen_prec 提升（FP 减少）若集中在'无fall视频'（keep-lying，P9 集）→ E1 的")
    print("    保护来自纯 normal/lying 负样本，D 删它们 = 删 E1 的优势来源，最坏情形。")

    with open(OUT, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
