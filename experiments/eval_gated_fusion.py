"""
评估门控融合模型（光流 aux）的 event_f1，对比 baseline p7d_delta。

用同一条规则 RuleV2（apply_rule_c, W=64, K=7）处理两个模型的预测，公平对比。

用法：
    python experiments/eval_gated_fusion.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gated_fusion_model import GatedFusionModel
from experiments.eval_rule_c import load_and_infer, apply_rulev2, evaluate


def load_and_infer_gated(ckpt, main_npz, aux_npz, splits, use_diff,
                         main_dim=3072, aux_dim=128, T=64, stride=8, bs=32):
    """门控融合模型推理：main (ViT-g+diff) + aux (光流) 分别喂给模型。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer-Gated] Device: {device}")

    d = np.load(main_npz, allow_pickle=True, mmap_mode='r')
    paths = set()
    with open(os.path.join(splits, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p: paths.add(p)

    aux_d = np.load(aux_npz, allow_pickle=True, mmap_mode='r')
    aux_feats = aux_d["pose_features"]  # (960000, aux_dim)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = GatedFusionModel(
        main_dim=main_dim, aux_dim=aux_dim, hidden_dim=384,
        decoder_type="transformer", causal=False, num_layers=1,
        num_heads=6, dropout=0.3, max_len=T + 10, num_classes=3,
    ).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()

    vp = d["video_paths"]; vsi = d["video_start_indices"]; vcc = d["video_clip_counts"]
    feats = d["features"]; lb16 = d["labels_16"]; fl = d["fall_labels"]; fdl = d["fallen_labels"]

    wins, wins_aux, meta, vinfo = [], [], [], {}
    for vi in range(len(vp)):
        if str(vp[vi]) not in paths: continue
        ns = int(vsi[vi]); nf = int(vcc[vi]); nw = (nf - T) // stride + 1
        vinfo[vi] = (ns, nf, nw)
        vf = feats[ns:ns + nf]
        af = aux_feats[ns:ns + nf]
        for w in range(nw):
            off = w * stride
            wins.append(vf[off:off + T]); wins_aux.append(af[off:off + T])
            meta.append((vi, off))

    preds = []
    for b0 in range(0, len(wins), bs):
        b1 = min(b0 + bs, len(wins))
        batch = torch.from_numpy(np.stack(wins[b0:b1])).float().to(device)
        aux_batch = torch.from_numpy(np.stack(wins_aux[b0:b1])).float().to(device)
        if use_diff:
            diff = torch.zeros_like(batch); diff[:, 1:] = batch[:, 1:] - batch[:, :-1]
            batch = torch.cat([batch, diff], dim=-1)
        with torch.no_grad(): preds.append(m(batch, aux_batch).argmax(dim=-1).cpu().numpy())
        if (b0 // bs + 1) % 50 == 0: print(f"  [{b1}/{len(wins)}]")

    preds = np.concatenate(preds, axis=0)
    pv = {}
    for vi, (ns, nf, nw) in vinfo.items():
        votes = np.zeros((nf, 3), dtype=int)
        for wi, (vj, off) in enumerate(meta):
            if vj != vi: continue
            for t in range(T):
                f = off + t
                if f < nf: votes[f, preds[wi, t]] += 1
        bp = np.argmax(votes, axis=1)
        tg = np.full(nf, 2, dtype=int)
        tg[fl[ns:ns + nf] == 1] = 0
        tg[fdl[ns:ns + nf] == 1] = 1
        pv[vi] = {"labels_16": lb16[ns:ns + nf], "ternary_gt": tg,
                  "bridge_pred": bp, "binary_fall_gt": fl[ns:ns + nf],
                  "binary_fallen_gt": fdl[ns:ns + nf]}
    return pv


def report(name, pv):
    """用 RuleV2 (K1=2, K2=7, W=64) 处理，报告 event_f1。"""
    corr, stats = apply_rulev2(pv, K1=2, K2=7, W=64)
    ev = evaluate(pv, corr)
    b = ev["binary_corrected"]
    t = ev["ternary_corrected"]
    print(f"\n{name}:")
    print(f"  event_f1      = {b['event_f1']:.4f}")
    print(f"  event_prec    = {b['event_precision']:.4f}")
    print(f"  event_rec     = {b['event_recall']:.4f}")
    print(f"  false_events  = {b['false_events']:,}")
    print(f"  avg_f1 (3cls) = {t['avg_f1']:.4f}")
    print(f"  events        = {stats['events']:,}")
    return b, t


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    NPZ = "data/omnifall_dinov2_giant_frame.npz"
    SPLITS = "DATASET-omnifall/splits/syn/random"
    FLOW = "data/omnifall_optical_flow.npz"

    # 1. baseline p7d_delta
    print("=" * 70)
    print("  [1] baseline p7d_delta")
    print("=" * 70)
    pv_base = load_and_infer("logs/phase7/p7d_delta/best_model.pt", NPZ, SPLITS, use_diff=True)
    report("baseline (p7d_delta)", pv_base)

    # 2. gated flow
    print("\n" + "=" * 70)
    print("  [2] gated flow (p8_gated_flow)")
    print("=" * 70)
    pv_gated = load_and_infer_gated(
        "logs/phase8/p8_gated_flow/best_model.pt", NPZ, FLOW, SPLITS,
        use_diff=True, main_dim=3072, aux_dim=128,
    )
    report("gated flow (p8_gated_flow)", pv_gated)

    print("\nDone.")


if __name__ == "__main__":
    main()
