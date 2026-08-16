"""
在服务器上运行：模型对 test 集推理，导出逐样本预测（小文件）。

导出结果只含预测标签，不含大特征，方便传回本地做诊断/抽帧。

用法（服务器上，GPU 环境）：
    # p7d_delta baseline（ViT-g + Δframe）
    python experiments/dump_test_preds.py

    # Phase 8 E1（ViT-g + Δframe + 二阶差分）
    python experiments/dump_test_preds.py \
        --ckpt logs/phase8/p8a_accel/best_model.pt \
        --use_diff --use_accel \
        --out analysis/diagnose_lying_event/test_preds_accel.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_rule_c import load_and_infer

NPZ    = "data/omnifall_dinov2_giant_frame.npz"
SPLITS = "DATASET-omnifall/splits/syn/random"


def parse_args():
    parser = argparse.ArgumentParser(description="Dump per-video test predictions")
    parser.add_argument("--ckpt", type=str,
                        default="logs/phase7/p7d_delta/best_model.pt",
                        help="Model checkpoint path")
    parser.add_argument("--use_diff", action="store_true",
                        help="Checkpoint trained with Δframe features")
    parser.add_argument("--use_accel", action="store_true",
                        help="Checkpoint trained with second-order diff (accel) features")
    parser.add_argument("--out", type=str,
                        default="analysis/diagnose_lying_event/test_preds.json",
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[1] Running inference ({args.ckpt}, use_diff={args.use_diff}, "
          f"use_accel={args.use_accel})...")
    pv = load_and_infer(args.ckpt, NPZ, SPLITS,
                        use_diff=args.use_diff, use_accel=args.use_accel)
    print(f"    Inferred {len(pv)} test videos")

    # 从 NPZ 取 video_paths 映射
    vp = np.load(NPZ, allow_pickle=True, mmap_mode='r')["video_paths"]

    out = {}
    for vi, d in pv.items():
        out[str(vp[vi])] = {
            "bridge_pred": [int(x) for x in d["bridge_pred"]],
            "ternary_gt":  [int(x) for x in d["ternary_gt"]],
            "labels_16":   [int(x) for x in d["labels_16"]],
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[2] Saved {len(out)} videos' predictions to {args.out}")
    print("    把这个文件传回本地，再跑 experiments/extract_false_positives.py")


if __name__ == "__main__":
    main()
