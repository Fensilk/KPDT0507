"""
E1@6000 (H) 全量 test 双口径评估：单次推理同时产出 instance 与 merged/RuleV2。

- instance 口径 = 每窗口每帧独立计（与 eval_p9e1/E1@2000 官方 test_results 一致，
  用于对 E3-full 目标线：fall_rec/fallen_prec 等）。
- merged 口径 = 逐视频投票 merge → 每视频 80 帧 bp → RuleV2(K1=2,K2=7,W=64) + K2/W
  网格（与 e3_rulev2/E1 规则读数一致，用于对 baseline+RuleV2 0.7735 / E1@2000 规则数）。

仅一次前向：instance 累计 + 逐视频 votes 同时攒。

用法（autodl）:
    python experiments/eval_p9e1_dual.py --ckpt logs/phase9/e1_6000/best_model.pt \
        --out logs/phase9/e1_6000/eval_best.json
"""
import os, sys, json, argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from experiments.train_p9e1 import preprocess_windows, ternary, decode_mp4_frames  # noqa: E402
from models.phase9_e1_model import E1Model                                        # noqa: E402
from eval_rule_c import apply_rulev2                                              # noqa: E402

VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
K2_GRID = [3, 5, 7, 10, 15, 20]
W_GRID = [32, 64, 128]
CANONICAL = (7, 64)


def ternary_metrics(y, p):
    return {
        "ternary_acc": float((p == y).mean()),
        "fall_f1": float(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[0]),
        "fall_precision": float(precision_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[0]),
        "fall_recall": float(recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[0]),
        "fallen_f1": float(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[1]),
        "fallen_precision": float(precision_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[1]),
        "fallen_recall": float(recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[1]),
        "normal_f1": float(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[2]),
        "avg_f1": float(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0).mean()),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def binary_event_metrics(y, p):
    bg = ((y == 0) | (y == 1)).astype(int)
    bp = (p != 2).astype(int)
    cm = confusion_matrix(bg, bp, labels=[0, 1])
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    ep = tp / max(tp + fp, 1); er = tp / max(tp + fn, 1)
    return {"event_f1": 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0,
            "event_precision": float(ep), "event_recall": float(er),
            "event_fp": fp, "event_fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="facebook/dinov2-giant")
    ap.add_argument("--npz", default=os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz"))
    ap.add_argument("--test_csv", default=os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "test.csv"))
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--win_batch", type=int, default=0,
                    help="同一视频内多少个窗口合并成一次前向（0=全部合并，默认）。"
                         "显存不足时调小；结果与逐窗口前向等价。")
    ap.add_argument("--limit_videos", type=int, default=None,
                    help="只评测 N 个视频（**确定性随机子集**，非前 N 个——见下）")
    ap.add_argument("--subset_seed", type=int, default=42,
                    help="--limit_videos 的抽样种子。⚠ 不可用「前 N 个」：test.csv 按路径"
                         "排序、fall/ 打头，前缀样本会严重偏向跌倒视频（本项目已两次踩坑）。")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump_pv", default=None,
                    help="把逐视频预测 pv 存到该路径（供 experiments/eval_timeline_metrics.py 复用）")
    args = ap.parse_args()
    os.chdir(BASE)

    model = E1Model.from_checkpoint(args.ckpt, device=args.device, model_name=args.model_name)
    model.eval()

    d = np.load(args.npz, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    idx = {p: i for i, p in enumerate(vp)}
    labels16 = d["labels_16"]; vsi = d["video_start_indices"]
    paths = list(pd.read_csv(args.test_csv)["path"].str.strip())
    if args.limit_videos and args.limit_videos < len(paths):
        # 确定性**随机**子集（不用前缀：test.csv 按路径排序、fall/ 打头，前缀有偏）
        rng = np.random.default_rng(args.subset_seed)
        keep = np.sort(rng.choice(len(paths), size=args.limit_videos, replace=False))
        paths = [paths[i] for i in keep]
        print(f"[subset] 从全集随机抽取 {len(paths)} 个视频评测"
              f"（seed={args.subset_seed}，确定性可复现）", flush=True)

    # instance 累计 + merged 累计
    iy, ip = [], []              # instance 口径 (flatten all window positions)
    pv = {}                      # merged 口径 per-video bp
    n_skip = 0
    for k, rel in enumerate(paths):
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if rel not in idx or not os.path.exists(mp4):
            n_skip += 1; continue
        frames = decode_mp4_frames(mp4)
        if frames is None or float(frames.mean()) <= 20:
            print(f"  [警告] 跳过 {rel}: 解码失败/黑帧"); n_skip += 1; continue
        st = int(vsi[idx[rel]])
        gt80 = ternary(labels16[st:st + 80])
        votes = np.zeros((80, 3), dtype=int)
        # 把同一视频的多个窗口**合并成一次批量前向**（原实现是每窗口一次 batch=1，
        # GPU 只用到 ~4GB/24GB，严重欠载）。B=窗口数，默认全并；显存紧张时用
        # --win_batch 切块。
        wins = list(range(0, 80 - args.window + 1, args.stride))
        step = args.win_batch if args.win_batch and args.win_batch > 0 else len(wins)
        for c0 in range(0, len(wins), step):
            chunk = wins[c0:c0 + step]
            batch = torch.stack(
                [torch.from_numpy(frames[ws:ws + args.window]) for ws in chunk], dim=0
            )                                                             # (B,T,H,W,3) uint8
            x = preprocess_windows(batch, args.device)                    # (B,T,3,H,W)
            with torch.no_grad():
                logits = model(x)                                        # (B,T,3)
            # Phase 11 Exp-1a：开了 boundary head 时 forward 返回 (logits, boundary_logits)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            preds = logits.argmax(-1).cpu().numpy()                       # (B,T)
            for i, ws in enumerate(chunk):
                p = preds[i]
                y = gt80[ws:ws + args.window]
                iy.append(y); ip.append(p)
                for t in range(args.window):
                    votes[ws + t, p[t]] += 1
        pv[rel] = {"ternary_gt": gt80, "bridge_pred": votes.argmax(1), "labels_16": labels16[st:st + 80]}
        if (k + 1) % 200 == 0:
            print(f"  [{k+1}/{len(paths)}] videos done, skip {n_skip}", flush=True)

    # ---- instance 口径 ----
    y = np.concatenate(iy); p = np.concatenate(ip)
    inst = ternary_metrics(y, p)
    inst["n_instances"] = int(len(y))

    # ---- merged 口径 raw + RuleV2 ----
    def flatten_pv():
        g = np.concatenate([pv[r]["ternary_gt"] for r in pv])
        b = np.concatenate([pv[r]["bridge_pred"] for r in pv])
        return g, b
    mg, mb = flatten_pv()
    merged_raw = {"ternary": ternary_metrics(mg, mb), "event": binary_event_metrics(mg, mb)}
    rows = {}
    for K2 in K2_GRID:
        for W in W_GRID:
            corr, _ = apply_rulev2(pv, K1=2, K2=K2, W=W)
            c = np.concatenate([corr[r][:80] for r in pv])
            rows[f"K2={K2},W={W}"] = {"ternary": ternary_metrics(mg, c), "event": binary_event_metrics(mg, c)}
    canonical = rows["K2=7,W=64"]
    best_key = max(rows, key=lambda k: rows[k]["event"]["event_f1"])

    out = {
        "ckpt": args.ckpt,
        "n_videos": len(pv), "n_skip": n_skip,
        "instance": inst,
        "merged_raw": merged_raw,
        "merged_canonical_RuleV2": canonical,
        "merged_best_RuleV2": {"key": best_key, **rows[best_key]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out}")
    if args.dump_pv:
        os.makedirs(os.path.dirname(args.dump_pv) or ".", exist_ok=True)
        torch.save(pv, args.dump_pv)
        print(f"Dumped pv: {args.dump_pv} ({len(pv)} videos)")
    print(f"[instance] acc={inst['ternary_acc']:.4f} fall_rec={inst['fall_recall']:.4f} "
          f"fallen_prec={inst['fallen_precision']:.4f} fallen_rec={inst['fallen_recall']:.4f} avg={inst['avg_f1']:.4f}")
    print(f"[merged] raw ev_f1={merged_raw['event']['event_f1']:.4f} | canonical={canonical['event']['event_f1']:.4f} "
          f"| best({best_key})={rows[best_key]['event']['event_f1']:.4f} "
          f"fallen_prec={rows[best_key]['ternary']['fallen_precision']:.4f}")


if __name__ == "__main__":
    main()
