"""
对 cross-attention 姿态融合模型在 test 集上做视频级投票推理 + RuleV2，
输出与 baseline 同口径的 event_f1 / ternary avg_f1（与 eval_rule_c 一致）。

用法：
    python experiments/eval_crossattn_pose.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cross_attention_fusion_model import CrossAttentionFusionModel
from experiments.eval_rule_c import apply_rulev2, evaluate

CKPT = "logs/phase8/p8_crossattn_pose/best_model.pt"
NPZ = "data/omnifall_dinov2_giant_frame.npz"
POSE_NPZ = "data/omnifall_pose_frame.npz"
SPLITS = "DATASET-omnifall/splits/syn/random"
T, STRIDE, BS = 64, 8, 32


def load_and_infer_crossattn(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    d = np.load(NPZ, allow_pickle=True, mmap_mode='r')
    pose = np.load(POSE_NPZ, allow_pickle=True, mmap_mode='r')

    paths = set()
    with open(os.path.join(SPLITS, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p:
                paths.add(p)

    pose_feats = pose["pose_features"]

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = CrossAttentionFusionModel(
        main_dim=1536 * 2, aux_dim=99, hidden_dim=384, num_heads=6,
        dropout=0.3, causal=False, max_len=T + 10, num_classes=3,
    ).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()

    vp = d["video_paths"]; vsi = d["video_start_indices"]; vcc = d["video_clip_counts"]
    feats = d["features"]; lb16 = d["labels_16"]; fl = d["fall_labels"]; fdl = d["fallen_labels"]

    wins, wins_pose, meta, vinfo = [], [], [], {}
    for vi in range(len(vp)):
        if str(vp[vi]) not in paths:
            continue
        ns = int(vsi[vi]); nf = int(vcc[vi]); nw = (nf - T) // STRIDE + 1
        vinfo[vi] = (ns, nf, nw)
        vf = feats[ns:ns + nf]
        pf = pose_feats[ns:ns + nf]
        for w in range(nw):
            off = w * STRIDE
            wins.append(vf[off:off + T])
            wins_pose.append(pf[off:off + T])
            meta.append((vi, off))

    preds = []
    for b0 in range(0, len(wins), BS):
        b1 = min(b0 + BS, len(wins))
        batch = torch.from_numpy(np.stack(wins[b0:b1])).float().to(device)
        diff = torch.zeros_like(batch); diff[:, 1:] = batch[:, 1:] - batch[:, :-1]
        main = torch.cat([batch, diff], dim=-1)               # (B,T,3072)
        aux = torch.from_numpy(np.stack(wins_pose[b0:b1])).float().to(device)  # (B,T,99)
        with torch.no_grad():
            preds.append(m(main, aux).argmax(dim=-1).cpu().numpy())
        if (b0 // BS + 1) % 50 == 0:
            print(f"  [{b1}/{len(wins)}]")

    preds = np.concatenate(preds, axis=0)
    pv = {}
    for vi, (ns, nf, nw) in vinfo.items():
        votes = np.zeros((nf, 3), dtype=int)
        for wi, (vj, off) in enumerate(meta):
            if vj != vi:
                continue
            for t in range(T):
                f = off + t
                if f < nf:
                    votes[f, preds[wi, t]] += 1
        bp = np.argmax(votes, axis=1)
        tg = np.full(nf, 2, dtype=int)
        tg[fl[ns:ns + nf] == 1] = 0
        tg[fdl[ns:ns + nf] == 1] = 1
        pv[vi] = {"labels_16": lb16[ns:ns + nf], "ternary_gt": tg,
                   "bridge_pred": bp,
                   "binary_fall_gt": fl[ns:ns + nf],
                   "binary_fallen_gt": fdl[ns:ns + nf]}
    return pv


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)

    print("=" * 72)
    print("Cross-Attn Pose Fusion — video-level inference + RuleV2")
    print("=" * 72)

    pv = load_and_infer_crossattn(CKPT)
    print(f"  Loaded {len(pv)} videos")

    # RuleV2（与 baseline 同参数 K1=2, K2=7, W=64）
    corr, stats = apply_rulev2(pv, K1=2, K2=7, W=64)
    r = evaluate(pv, corr)

    tb = r["ternary_bridge"]
    bb = r["binary_bridge"]
    bc = r["binary_corrected"]
    tc = r["ternary_corrected"]

    print("\n  ┌─ Ternary (video-voted bridge) ─────────────────────")
    print(f"  │  avg_f1={tb['avg_f1']:.4f}  fall_f1={tb['fall_f1']:.4f}  "
          f"fallen_f1={tb['fallen_f1']:.4f}  normal_f1={tb['normal_f1']:.4f}")
    print(f"  └──────────────────────────────────────────────────")
    print("\n  ┌─ RuleV2 event metrics ────────────────────────────")
    print(f"  │  event_f1={bc['event_f1']:.4f}  prec={bc['event_precision']:.4f}  "
          f"recall={bc['event_recall']:.4f}  false_events={bc['false_events']}")
    print(f"  │  (baseline 参考: event_f1=0.774, false_events=4,492)")
    print(f"  └──────────────────────────────────────────────────")

    print("\n[RuleV2 stats]", stats)


if __name__ == "__main__":
    main()
