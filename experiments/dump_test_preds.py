"""
在服务器上运行：p7d_delta 模型对 test 集推理，导出逐样本预测（小文件）。

导出结果只含预测标签，不含大特征，方便传回本地做诊断/抽帧。

用法（服务器上，GPU 环境）：
    python experiments/dump_test_preds.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_rule_c import load_and_infer

CKPT   = "logs/phase7/p7d_delta/best_model.pt"
NPZ    = "data/omnifall_dinov2_giant_frame.npz"
SPLITS = "DATASET-omnifall/splits/syn/random"
OUT    = "analysis/diagnose_lying_event/test_preds.json"


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("[1] Running inference (p7d_delta, use_diff=True)...")
    pv = load_and_infer(CKPT, NPZ, SPLITS, use_diff=True)
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

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[2] Saved {len(out)} videos' predictions to {OUT}")
    print("    把这个文件传回本地，再跑 experiments/extract_false_positives.py")


if __name__ == "__main__":
    main()
