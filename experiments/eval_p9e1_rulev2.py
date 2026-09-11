"""
E1@2000 全量 test + RuleV2 事件后处理（与 baseline/E3-full 同一规则口径）。

与 eval_p9e1（官方实例级）的关键差异：规则需要**每视频合并时间线**，故这里
按两阶段管线（eval_rule_c / analyze_p7d_full）的 merge 语义构造 pv——
窗口起点 ws∈{0,8,16}（window=64/stride=8）逐帧预测 → 每帧跨覆盖窗口投票
argmax → 每视频 80 帧三元序列 bp；GT = ternary(labels16[st:st+80])。此后
apply_rulev2 + 指标与 experiments/sweep_rulev2.py 完全一致，可直接对比
e3_rulev2 的 baseline / E3-full 读数（同 merged 口径）。

运行（须 CUDA；giant 需 ~>10GB 或看实测，本地 4060-8G 可能 OOM→autodl）:
    python experiments/eval_p9e1_rulev2.py                                  # 全量 test
    python experiments/eval_p9e1_rulev2.py --limit 30                        # 冒烟/测速
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from experiments.eval_p9e1 import decode_frames, N_FRAMES, MEAN_CANARY  # noqa: E402
from experiments.train_p9e1 import preprocess_windows, ternary           # noqa: E402
from models.phase9_e1_model import E1Model                               # noqa: E402
from eval_rule_c import apply_rulev2                                     # noqa: E402

VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
DEFAULT_OUT = os.path.join(BASE, "logs", "phase9", "e1_rulev2")

K1 = 2
K2_GRID = [3, 5, 7, 10, 15, 20]
W_GRID = [32, 64, 128]
CANONICAL = (7, 64)


# ── 指标（与 sweep_rulev2.py 同款）──
def ternary_metrics(gt3, pred):
    return {
        "ternary_acc": float((pred == gt3).mean()),
        "fall_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
        "fall_precision": float(precision_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
        "fall_recall": float(recall_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
        "fallen_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
        "fallen_precision": float(precision_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
        "fallen_recall": float(recall_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
        "avg_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0).mean()),
        "confusion_matrix": confusion_matrix(gt3, pred, labels=[0, 1, 2]).tolist(),
    }


def binary_event_metrics(gt3, pred):
    binary_gt = ((gt3 == 0) | (gt3 == 1)).astype(int)
    binary_pred = (pred != 2).astype(int)
    cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    ep = tp / max(tp + fp, 1); er = tp / max(tp + fn, 1)
    return {
        "event_f1": 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0,
        "event_precision": float(ep), "event_recall": float(er),
        "event_fp": fp, "event_fn": fn,
    }


# ── 逐视频合并推理（支持断点续跑：每 save_every 个新视频存一次 cache）──
def build_pv(model, test_paths, npz_path, window, stride, device, limit=None,
             pv=None, cache_path=None, save_every=100):
    d = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    idx = {p: i for i, p in enumerate(vp)}
    labels16 = d["labels_16"]; vsi = d["video_start_indices"]

    videos = test_paths[:limit] if limit else test_paths
    pv = dict(pv) if pv else {}
    done0 = len(pv)
    n_skip, n_new, n_windows = 0, 0, 0
    for i, rel in enumerate(videos):
        if rel in pv:
            continue
        if rel not in idx:
            print(f"  [警告] 跳过 {rel}: 不在 NPZ 索引中", flush=True); n_skip += 1; continue
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if not os.path.exists(mp4):
            print(f"  [警告] 跳过 {rel}: 视频缺失", flush=True); n_skip += 1; continue
        frames = decode_frames(mp4)
        if frames is None:
            print(f"  [警告] 跳过 {rel}: 解码空帧", flush=True); n_skip += 1; continue
        if float(frames.mean()) <= MEAN_CANARY:
            print(f"  [警告] 跳过 {rel}: 黑帧", flush=True); n_skip += 1; continue

        st = int(vsi[idx[rel]])
        nf = N_FRAMES
        votes = np.zeros((nf, 3), dtype=int)
        for ws in range(0, nf - window + 1, stride):
            fr = torch.from_numpy(frames[ws:ws + window])               # (T,H,W,3)
            x = preprocess_windows(fr.unsqueeze(0), device)             # (1,T,3,H,W)
            with torch.no_grad():
                logits = model(x)                                       # (1,T,3)
            p = logits.argmax(-1).cpu().numpy().ravel()                 # (T,)
            for t in range(window):
                votes[ws + t, p[t]] += 1
            n_windows += 1
        bp = votes.argmax(1)                                            # (nf,)
        gt = ternary(labels16[st:st + nf])
        pv[rel] = {"ternary_gt": gt, "bridge_pred": bp,
                   "labels_16": labels16[st:st + nf]}
        n_new += 1
        if cache_path and ((n_new) % save_every == 0 or i == len(videos) - 1):
            torch.save(pv, cache_path)
            print(f"  [checkpoint] cached {len(pv)} videos -> {cache_path}", flush=True)
        if (i + 1) % 100 == 0 or i == len(videos) - 1:
            print(f"  [{i + 1}/{len(videos)}] 完成 {done0 + n_new} 视频 / {n_windows} 新窗口, "
                  f"跳过 {n_skip}", flush=True)
    meta = {"n_skip": n_skip, "n_windows": n_windows, "resumed": done0, "n_videos": len(pv)}
    if n_new == 0:
        print("  [注意] 无可处理的新视频（可能全部已缓存）", flush=True)
    return pv, meta


def flatten_pv(pv):
    gts, raws = [], []
    for rel in pv:
        gts.append(pv[rel]["ternary_gt"]); raws.append(pv[rel]["bridge_pred"])
    return np.concatenate(gts), np.concatenate(raws)


def run_grid(pv):
    gt3, bp0 = flatten_pv(pv)
    raw = {"ternary": ternary_metrics(gt3, bp0), "binary_event": binary_event_metrics(gt3, bp0)}
    rows = []
    for K2 in K2_GRID:
        for W in W_GRID:
            corr, stats = apply_rulev2(pv, K1=K1, K2=K2, W=W)
            cps = [corr[rel][: len(pv[rel]["ternary_gt"])] for rel in pv]
            cp = np.concatenate(cps)
            rows.append({"K1": K1, "K2": K2, "W": W,
                         "ternary": ternary_metrics(gt3, cp),
                         "binary_event": binary_event_metrics(gt3, cp),
                         "rule_stats": stats})
    best = max(rows, key=lambda r: r["binary_event"]["event_f1"])
    return raw, rows, best


def fmt(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/dinov2-giant")
    ap.add_argument("--ckpt", default=os.path.join(BASE, "logs", "phase9", "e1_5ep", "best_model.pt"))
    ap.add_argument("--npz", default=os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz"))
    ap.add_argument("--test_csv", default=os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "test.csv"))
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--cache", default=None,
                    help="pv 断点缓存路径（默认 {out_dir}/pv_cache.pt；存在则续跑跳过已完成视频）")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(BASE)
    if not torch.cuda.is_available():
        sys.exit("[error] E1 推理需 CUDA（vit fp16 × LoRA fp32）")
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1] device={device} | window={args.window} stride={args.stride} | model={args.model_name}")
    print(f"[2] 重建 E1Model from {args.ckpt}", flush=True)
    model = E1Model.from_checkpoint(args.ckpt, device=device, model_name=args.model_name)
    model.eval()

    paths = list(pd.read_csv(args.test_csv)["path"].str.strip())
    if args.limit:
        paths = paths[: args.limit]
    print(f"[3] test 视频数 = {len(paths)}（limit={args.limit}）", flush=True)

    cache_path = args.cache or os.path.join(args.out_dir, "pv_cache.pt")
    pv = {}
    if os.path.exists(cache_path):
        pv = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[3b] 续跑: 载入已有缓存 {len(pv)} 视频 <- {cache_path}", flush=True)

    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    pv, meta = build_pv(model, paths, args.npz, args.window, args.stride, device,
                        pv=pv, cache_path=cache_path, save_every=100)
    t1.record(); torch.cuda.synchronize()
    el = t0.elapsed_time(t1) / 1000.0
    print(f"[4] 推理完成 {len(pv)} 视频 / {meta['n_windows']} 窗口, 墙钟 {el:.1f}s"
          f" ({el/max(len(pv),1):.1f}s/视频)", flush=True)

    raw, rows, best = run_grid(pv)
    b7 = next(r for r in rows if (r["K2"], r["W"]) == CANONICAL)

    print(f"\n  raw  : ev_f1={fmt(raw['binary_event']['event_f1'])} "
          f"prec={fmt(raw['binary_event']['event_precision'])} rec={fmt(raw['binary_event']['event_recall'])} "
          f"| fallen_prec={fmt(raw['ternary']['fallen_precision'])} avg_f1={fmt(raw['ternary']['avg_f1'])}")
    print(f"  canonical(K2=7,W=64): ev_f1={fmt(b7['binary_event']['event_f1'])} "
          f"prec={fmt(b7['binary_event']['event_precision'])} rec={fmt(b7['binary_event']['event_recall'])} "
          f"| fallen_prec={fmt(b7['ternary']['fallen_precision'])} avg_f1={fmt(b7['ternary']['avg_f1'])}")
    print(f"  best(K2={best['K2']},W={best['W']}): ev_f1={fmt(best['binary_event']['event_f1'])} "
          f"prec={fmt(best['binary_event']['event_precision'])} rec={fmt(best['binary_event']['event_recall'])} "
          f"| fallen_prec={fmt(best['ternary']['fallen_precision'])} avg_f1={fmt(best['ternary']['avg_f1'])}")

    # grid 表
    print("\n  event_f1 grid（* = best）:")
    hdr = f"{'K2':>4s} | " + "   ".join(f"W={w:<3d}" for w in W_GRID)
    print("  " + hdr)
    for K2 in K2_GRID:
        cells = []
        for W in W_GRID:
            r = next(r for r in rows if r["K2"] == K2 and r["W"] == W)
            cells.append(fmt(r["binary_event"]["event_f1"]) + ("*" if (K2, W) == (best["K2"], best["W"]) else " "))
        print(f"  K2={K2:<3d} | " + "   ".join(f"{c:<9s}" for c in cells))

    # 保存
    out = {"meta": meta, "raw": raw, "canonical": b7, "best": best, "grid": rows,
           "args": vars(args)}
    out_path = os.path.join(args.out_dir, "rulev2_e1.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # 与两阶段对照（若本地已有 e3_rulev2 sweep json）
    ref = os.path.join(BASE, "logs", "phase9", "e3_rulev2", "rulev2_sweep.json")
    if os.path.exists(ref):
        d = json.load(open(ref, encoding="utf-8"))
        print("\n  ── 对照（同 merged 口径）──")
        for tag in ["baseline", "E3-full"]:
            b = d[tag]["best"]; ra = d[tag]["raw"]["binary_event"]
            print(f"  {tag:<10s} raw {fmt(ra['event_f1'])}  |  best(K2={b['K2']},W={b['W']}) "
                  f"{fmt(b['binary_event']['event_f1'])} (prec {fmt(b['binary_event']['event_precision'])}, "
                  f"rec {fmt(b['binary_event']['event_recall'])}, fallen_prec {fmt(b['ternary']['fallen_precision'])})")
        b = best
        print(f"  {'E1@2000':<10s} raw {fmt(raw['binary_event']['event_f1'])}  |  best(K2={b['K2']},W={b['W']}) "
              f"{fmt(b['binary_event']['event_f1'])} (prec {fmt(b['binary_event']['event_precision'])}, "
              f"rec {fmt(b['binary_event']['event_recall'])}, fallen_prec {fmt(b['ternary']['fallen_precision'])})")


if __name__ == "__main__":
    main()
