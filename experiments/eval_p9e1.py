"""E1 全量 test 诊断矩阵评估：端到端 LoRA 微调模型 vs baseline(p7d_delta) vs E3。

Phase 9 E1（docs/0902实验第九阶段总结.md §E1）vs E3 的唯一差异是训练范式
（端到端反传 LoRA vs 两段式微调特征）。因此评估口径必须与 E3 完全一致，才能
给出 apples-to-apples 的三列对比表。E3 Stage2 test 混淆矩阵总帧数
230,400 = 1,200 视频 × 每视频 3 窗口 × 64 帧（window=64 / stride=8 滑窗，
实例级：每窗口每帧独立判一次，跨窗口不投票不合并）。

流程（逐 test 视频，直接解码 mp4——train/val 帧缓存只覆盖 train+val 视频）:
    mp4 (PyAV→cv2 兜底) → 80 帧 (H,W,3) uint8 → cv2.resize 224×224 (INTER_LINEAR)
    → 窗口 ws ∈ {0,8,16} (range(0, 80-window+1, stride)) 取连续 64 帧
    → preprocess_windows(与训练完全一致的 /255 + permute + DINOv2 mean/std)
    → E1Model.from_checkpoint 重建 → no_grad forward → argmax → 逐帧三分预测
    真值: NPZ labels_16[st+ws : st+ws+window] → ternary (1→0 fall, 2→1 fallen, else→2)
    → 累计全部窗口/视频 → compute_ternary_metrics（与 utils/metrics.py / E3 同款）

解码失败（黑帧/缺视频/解码返回空）的视频打警告并跳过、计数，不静默喂黑帧。

用法（项目根目录，py310）:
    python experiments/eval_p9e1.py                                # 全量 1,200 视频
    python experiments/eval_p9e1.py --limit 5                      # 冒烟（前 5 个 test 视频）
    python experiments/eval_p9e1.py --ckpt logs/phase9/e1/best_model.pt \
        --out logs/phase9/e1/test_results.json
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))   # 供 av_decode / step7 导入

# 训练侧预处理 / 标签映射 / 紧凑 checkpoint 重建。train_p9e1 的 main() 由
# `if __name__ == "__main__"` 守卫，模块导入无副作用 → 直接复用保证归一化逐字节对齐训练。
from experiments.train_p9e1 import preprocess_windows, ternary   # noqa: E402
from models.phase9_e1_model import E1Model                       # noqa: E402
from av_decode import read_rgb_frames                            # noqa: E402
from utils.metrics import compute_ternary_metrics                # noqa: E402

VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
BASELINE_JSON = os.path.join(BASE, "logs", "phase7", "p7d_delta", "test_results.json")
E3_JSON = os.path.join(BASE, "logs", "phase9", "e3_stage2", "test_results.json")

SIZE = (224, 224)     # DINOv2 输入尺寸；训练缓存帧已 resize，此处为兜底安全
N_FRAMES = 80         # 每视频帧数（OmniFall 81 → read_rgb_frames 截/补到 80）
MEAN_CANARY = 20.0    # 解码成功判据：真实视频帧均值远 > 20，黑帧/失败 ≈ 0

# 三列 Δ 表的行（键名与 baseline / E3 test_results.json 完全一致）
TOP_KEYS = ["fall_precision", "fall_recall", "fall_f1",
            "fallen_precision", "fallen_recall", "fallen_f1",
            "avg_f1", "ternary_acc"]


def decode_frames(mp4: str, n: int = N_FRAMES):
    """解码 mp4 → (n,224,224,3) uint8。失败返回 None（黑帧判据由调用方把关）。"""
    frames = read_rgb_frames(mp4, n)
    if not frames:
        return None
    # prep_p9e1_frames.py 同款：逐帧 INTER_LINEAR resize（视频原生非 224，此步必需）
    imgs = np.stack([cv2.resize(f, SIZE, interpolation=cv2.INTER_LINEAR)
                     for f in frames])
    return np.ascontiguousarray(imgs).astype(np.uint8)


def load_metrics_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
# 全量评估
# ═══════════════════════════════════════════════════════════
def run_test(model, test_paths, npz_path, window, stride, device, limit=None):
    """逐视频解码 + 滑窗推理，累计实例级帧预测 → metrics dict。

    metrics 内含 compute_ternary_metrics 全部主指标，外加 n_videos_evaluated /
    n_windows / n_decode_skips / n_frames。窗口起点 ws ∈ range(0, 80-window+1,
    stride)：window=64/stride=8 → {0,8,16} = 每视频 3 窗口（与 E3 混淆总数
    230,400 同口径）。跨窗口不投票、不合并，每窗口每帧独立判一次后直接累计。
    """
    d = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    idx = {p: i for i, p in enumerate(vp)}
    labels16 = d["labels_16"]
    vsi = d["video_start_indices"]

    videos = test_paths[:limit] if limit else test_paths
    all_y, all_p = [], []
    n_skip, n_done, n_windows = 0, 0, 0
    for i, rel in enumerate(videos):
        if rel not in idx:
            print(f"  [警告] 跳过 {rel}: 不在 NPZ 索引中")
            n_skip += 1
            continue
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if not os.path.exists(mp4):
            print(f"  [警告] 跳过 {rel}: 视频缺失 ({mp4})")
            n_skip += 1
            continue
        frames = decode_frames(mp4)
        if frames is None:
            print(f"  [警告] 跳过 {rel}: 解码返回空帧")
            n_skip += 1
            continue
        fmean = float(frames.mean())
        if fmean <= MEAN_CANARY:
            # 解码失败/黑帧金丝雀（prep_p9e1_frames 用 >20 判真实解码帧）
            print(f"  [警告] 跳过 {rel}: 帧均值 {fmean:.1f} ≤ {MEAN_CANARY}"
                  f"（解码失败/黑帧）")
            n_skip += 1
            continue

        st = int(vsi[idx[rel]])
        n_win_this = 0
        for ws in range(0, N_FRAMES - window + 1, stride):
            fr = torch.from_numpy(frames[ws:ws + window])          # (T,H,W,3) uint8
            y = ternary(labels16[st + ws: st + ws + window])       # 帧级三分真值 (T,)
            x = preprocess_windows(fr.unsqueeze(0), device)        # (1,T,3,H,W)
            with torch.no_grad():
                logits = model(x)                                  # (1,T,3)
            p = logits.argmax(-1).cpu().numpy().ravel()            # (T,)
            all_y.append(y)
            all_p.append(p)
            n_win_this += 1
        n_windows += n_win_this
        n_done += 1
        if (i + 1) % 200 == 0 or i == len(videos) - 1:
            print(f"  [{i + 1}/{len(videos)}] 已完成 {n_done} 视频 / "
                  f"{n_windows} 窗口, 跳过 {n_skip}", flush=True)

    if not all_y:
        raise RuntimeError("无任何可评估视频（全部解码失败或缺失）")

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    metrics = compute_ternary_metrics(p, y)
    metrics["n_videos_evaluated"] = n_done
    metrics["n_windows"] = n_windows
    metrics["n_decode_skips"] = n_skip
    metrics["n_frames"] = int(len(y))
    return metrics


# ═══════════════════════════════════════════════════════════
# 三列对比表打印（镜像 eval_p9e3.py，但加 E1 列与 Δ(E1−E3)）
# ═══════════════════════════════════════════════════════════
def print_table(b, e3, e1):
    print(f"{'指标':<16s} {'baseline':>10s} {'E3微调':>10s} {'E1':>10s} {'Δ(E1-E3)':>10s}")
    print("-" * 62)
    for k in TOP_KEYS:
        bv = b.get(k)
        ev = e3.get(k) if e3 else None
        e1v = e1.get(k)
        if bv is None or e1v is None:
            continue
        if ev is None:
            print(f"{k:<16s} {bv:>10.4f} {'--':>10s} {e1v:>10.4f} {'--':>10s}")
        else:
            print(f"{k:<16s} {bv:>10.4f} {ev:>10.4f} {e1v:>10.4f} {e1v - ev:>+10.4f}")

    print()
    print("诊断矩阵观察（主指标视角）:")
    if e3 is None:
        print(f"  动态轴 lie_down vs fall: fall_rec={e1.get('fall_recall', 0):.3f} "
              f"(baseline {b.get('fall_recall', 0):.3f}) "
              f"fall_prec={e1.get('fall_precision', 0):.3f} "
              f"(baseline {b.get('fall_precision', 0):.3f})")
        print(f"  静态轴 lying vs fallen: fallen_rec={e1.get('fallen_recall', 0):.3f} "
              f"(baseline {b.get('fallen_recall', 0):.3f}) "
              f"fallen_prec={e1.get('fallen_precision', 0):.3f} "
              f"(baseline {b.get('fallen_precision', 0):.3f})")
    else:
        print(f"  动态轴 lie_down vs fall: fall_rec={e1.get('fall_recall', 0):.3f} "
              f"(baseline {b.get('fall_recall', 0):.3f}, E3 {e3.get('fall_recall', 0):.3f}) "
              f"fall_prec={e1.get('fall_precision', 0):.3f} "
              f"(baseline {b.get('fall_precision', 0):.3f}, E3 {e3.get('fall_precision', 0):.3f})")
        print(f"  静态轴 lying vs fallen: fallen_rec={e1.get('fallen_recall', 0):.3f} "
              f"(baseline {b.get('fallen_recall', 0):.3f}, E3 {e3.get('fallen_recall', 0):.3f}) "
              f"fallen_prec={e1.get('fallen_precision', 0):.3f} "
              f"(baseline {b.get('fallen_precision', 0):.3f}, E3 {e3.get('fallen_precision', 0):.3f})")
    print()
    print("注：E1 与 E3 同口径（window-64/stride-8 实例级）。两者唯一差异是训练范式")
    print("    （端到端 LoRA 反传 vs 两段式微调）。若只有动态轴改善而静态轴持平/有限，")
    print("    即支持帧级端到端仍不足以解因果静态区分的判断。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/dinov2-giant",
                    help="重建用的 ViT 骨干（方案E 小骨干: facebook/dinov2-base/small）")
    ap.add_argument("--ckpt", default=os.path.join(BASE, "logs", "phase9", "e1",
                                                   "best_model.pt"))
    ap.add_argument("--out", default=os.path.join(BASE, "logs", "phase9", "e1",
                                                  "test_results.json"))
    ap.add_argument("--npz", default=os.path.join(BASE, "data",
                                                  "omnifall_dinov2_giant_frame.npz"))
    ap.add_argument("--test_csv", default=os.path.join(
        BASE, "DATASET-omnifall", "splits", "syn", "random", "test.csv"))
    ap.add_argument("--baseline", default=BASELINE_JSON)
    ap.add_argument("--e3", default=E3_JSON)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟：只评估前 N 个 test 视频")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.chdir(BASE)

    if not os.path.exists(args.ckpt):
        print(f"[error] checkpoint 不存在: {args.ckpt}（先跑 train_p9e1.py）")
        sys.exit(1)

    device = args.device
    # E1Model.forward 在 fp16 autocast("cuda") 下跑 fp16 vit 主干 × fp32 LoRA A/B，
    # CPU 上 autocast 失效会触发 Half/float dtype 错配 → 需 GPU（py310 cu121 / autodl）
    if not device.startswith("cuda"):
        print(f"[error] E1 推理需 CUDA（vit fp16 × LoRA fp32 在 autocast(\"cuda\") 下）; "
              f"当前 device={device} 不受支持")
        sys.exit(1)
    print(f"[0] device={device} | window={args.window} stride={args.stride}")
    print(f"[1] 重建 E1Model: {args.ckpt}")
    model = E1Model.from_checkpoint(args.ckpt, device=device, model_name=args.model_name)
    model.eval()

    if not os.path.exists(args.test_csv):
        print(f"[error] test_csv 不存在: {args.test_csv}")
        sys.exit(1)
    paths = list(pd.read_csv(args.test_csv)["path"].str.strip())
    print(f"[2] test 视频: {len(paths)}（limit={args.limit} → 评估前 {args.limit}）")

    print(f"[3] 解码 + 滑窗推理（逐视频 PyAV→cv2）…")
    metrics = run_test(model, paths, args.npz, args.window, args.stride,
                       device, limit=args.limit)

    print(f"\n{'=' * 60}")
    print("E1 TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  三元 acc:        {metrics['ternary_acc']:.4f}  "
          f"(random: {metrics['random_baseline']:.4f})")
    print(f"  帧数: {metrics['n_frames']} = {metrics['n_videos_evaluated']} 视频 × "
          f"{metrics['n_windows'] // max(metrics['n_videos_evaluated'], 1)} 窗口 × "
          f"{args.window} 帧（解码跳过 {metrics['n_decode_skips']}）")
    for k in TOP_KEYS:
        print(f"  {k:<17s} {metrics[k]:.4f}")
    print(f"\n  --- Classification Report (3-class) ---")
    print(metrics["cls_report"])

    # ---- 三列对比表 ----
    print("=" * 62)
    b = load_metrics_json(args.baseline)
    e3 = None
    e3_note = ""
    if os.path.exists(args.e3):
        e3 = load_metrics_json(args.e3)
    else:
        e3_note = f"（E3 json 缺失: {args.e3} → 仅打印 baseline | E1）"
    print(f"对比: E1 vs baseline(p7d_delta) vs E3(stage2){e3_note}")
    print_table(b, e3, metrics)

    # ---- 写 test_results.json（键 schema 与 baseline/E3 对齐）----
    config = {
        "model": "E1 lora16 dinov2-giant",
        "paradigm": "end-to-end LoRA finetune (window-64/stride-8 instance-based, mirrors E3)",
        "lora_rank": int(getattr(model, "lora_rank", 16)),
        "window": args.window,
        "stride": args.stride,
        "ckpt": args.ckpt,
        "npz": args.npz,
        "test_csv": args.test_csv,
        "n_test_videos_requested": len(paths) if args.limit is None else args.limit,
        "device": str(device),
    }
    results = {k: v for k, v in metrics.items()
               if k not in ("cls_report", "confusion_matrix")}
    results["cls_report"] = metrics["cls_report"]
    results["confusion_matrix"] = metrics["confusion_matrix"]
    results["config"] = config
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
