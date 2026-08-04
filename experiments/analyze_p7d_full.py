"""
P7d full analysis: p7d_delta + RuleV1 + binary event view.
Compares with p6_bridge+R1, p7c_binary, and Oracle.

Usage:
    python experiments/analyze_p7d_full.py
"""

import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.phase6_model import Phase6TernaryModel

CLASS_NAMES_16 = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]


def load_and_infer(checkpoint_path, npz_path, splits_dir, T=64, stride=8, bs=32):
    """Load model checkpoint and run per-video inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading: {checkpoint_path}")

    d = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    paths = set()
    with open(os.path.join(splits_dir, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p: paths.add(p)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    use_diff = cfg.get("use_diff", False)
    input_dim = 3072 if use_diff else 1536
    print(f"[INFO] use_diff={use_diff}, input_dim={input_dim}")

    model = Phase6TernaryModel(
        input_dim=input_dim, hidden_dim=384, decoder_type="transformer",
        causal=False, num_layers=1, num_heads=6, dropout=0.3,
        max_len=T + 10, num_classes=3,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    vp = d["video_paths"]; vsi = d["video_start_indices"]
    vcc = d["video_clip_counts"]; feats = d["features"]
    lb16 = d["labels_16"]; fl = d["fall_labels"]; fdl = d["fallen_labels"]

    wins, meta, vinfo = [], [], {}
    for vi in range(len(vp)):
        if str(vp[vi]) not in paths: continue
        ns = int(vsi[vi]); nf = int(vcc[vi]); nw = (nf - T) // stride + 1
        vinfo[vi] = (ns, nf, nw)
        vf = feats[ns:ns + nf]
        for w in range(nw):
            off = w * stride; wins.append(vf[off:off + T]); meta.append((vi, off))

    preds = []
    for b0 in range(0, len(wins), bs):
        b1 = min(b0 + bs, len(wins))
        batch = torch.from_numpy(np.stack(wins[b0:b1])).float().to(device)
        if use_diff:
            diff = torch.zeros_like(batch)
            diff[:, 1:] = batch[:, 1:] - batch[:, :-1]
            batch = torch.cat([batch, diff], dim=-1)
        with torch.no_grad(): preds.append(model(batch).argmax(dim=-1).cpu().numpy())
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


def apply_rulev1(per_video, event_timeout=32):
    """P7a RuleV1: event-level unidirectional suppression."""
    corrected = {}
    stats = {"events": 0, "fallen_kept": 0, "fallen_suppressed": 0}
    for vi, data in per_video.items():
        bp = data["bridge_pred"].copy(); nf = len(bp); cp = bp.copy()
        event_active = False; timer = 0; cons_fall = 0
        for f in range(nf):
            p = bp[f]
            cons_fall = cons_fall + 1 if p == 0 else 0
            if not event_active and cons_fall >= 3:
                event_active = True; timer = 0; stats["events"] += 1
            if event_active:
                if p == 1: stats["fallen_kept"] += 1; timer = 0
                elif p == 0: timer = 0
                else:
                    timer += 1
                    if timer > event_timeout: event_active = False; timer = 0
            else:
                if p == 1: cp[f] = 2; stats["fallen_suppressed"] += 1
        corrected[vi] = cp
    return corrected, stats


def compute_all_metrics(per_video, corrected):
    """Ternary + binary event metrics + 16-class breakdown."""
    all_gt3, all_bp, all_cp, all_l16, all_fall_g, all_fallen_g = [], [], [], [], [], []
    for vi, d in per_video.items():
        n = len(d["ternary_gt"])
        all_gt3.append(d["ternary_gt"]); all_bp.append(d["bridge_pred"])
        all_cp.append(corrected[vi][:n]); all_l16.append(d["labels_16"])
        all_fall_g.append(d["binary_fall_gt"]); all_fallen_g.append(d["binary_fallen_gt"])
    gt3 = np.concatenate(all_gt3); bp = np.concatenate(all_bp); cp = np.concatenate(all_cp)
    l16 = np.concatenate(all_l16); fg = np.concatenate(all_fall_g); flg = np.concatenate(all_fallen_g)

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    def ternary_metrics(pred):
        cm = confusion_matrix(gt3, pred, labels=[0, 1, 2])
        fall_p = cm[0, 0] / max(cm[0].sum(), 1); fall_r = cm[0, 0] / max(cm[:, 0].sum(), 1)
        fall_f1 = 2 * fall_p * fall_r / (fall_p + fall_r) if (fall_p + fall_r) > 0 else 0
        # Actually sklearn gives per-class OvR. Let me use sklearn.
        return {
            "ternary_acc": float((pred == gt3).mean()),
            "fall_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
            "fall_precision": float(precision_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
            "fall_recall": float(recall_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
            "fallen_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
            "fallen_precision": float(precision_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
            "fallen_recall": float(recall_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
            "normal_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[2]),
            "avg_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0).mean()),
            "confusion_matrix": cm.tolist(),
        }

    def binary_event_metrics(pred):
        binary_gt = ((fg == 1) | (flg == 1)).astype(int)  # event=1, normal=0
        binary_pred = (pred != 2).astype(int)  # fall/fallen=1(event), normal=0
        cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
        return {
            "event_f1": float(f1_score(binary_gt, binary_pred, pos_label=1, zero_division=0)),
            "event_precision": float(precision_score(binary_gt, binary_pred, pos_label=1, zero_division=0)),
            "event_recall": float(recall_score(binary_gt, binary_pred, pos_label=1, zero_division=0)),
            "binary_cm": cm.tolist(),
        }

    def breakdown_16(pred):
        out = {}
        binary_pred = (pred != 2).astype(int)
        for c in range(16):
            m = l16 == c; n = int(m.sum())
            if n == 0: continue
            out[CLASS_NAMES_16[c]] = {
                "total": n,
                "pred_fall": int((pred[m] == 0).sum()),
                "pred_fallen": int((pred[m] == 1).sum()),
                "pred_normal": int((pred[m] == 2).sum()),
                "pred_event": int(binary_pred[m].sum()),
            }
        return out

    return {
        "bridge_ternary": ternary_metrics(bp),
        "bridge_binary": binary_event_metrics(bp),
        "rulev1_ternary": ternary_metrics(cp),
        "rulev1_binary": binary_event_metrics(cp),
        "breakdown_bridge": breakdown_16(bp),
        "breakdown_rulev1": breakdown_16(cp),
    }


def print_comparison(results, label):
    bt = results["bridge_ternary"]; bb = results["bridge_binary"]
    rt = results["rulev1_ternary"]; rb = results["rulev1_binary"]
    bdb = results["breakdown_bridge"]; bdr = results["breakdown_rulev1"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n--- Ternary ---")
    for k in ["fall_f1", "fall_precision", "fall_recall", "fallen_f1",
              "fallen_precision", "fallen_recall", "avg_f1"]:
        print(f"  {k:<20s}: Bridge={bt[k]:.4f}  +R1={rt[k]:.4f}  Δ={rt[k]-bt[k]:+.4f}")

    print(f"\n--- Binary Event ---")
    for k in ["event_f1", "event_precision", "event_recall"]:
        print(f"  {k:<20s}: Bridge={bb[k]:.4f}  +R1={rb[k]:.4f}  Δ={rb[k]-bb[k]:+.4f}")

    print(f"\n--- Lying / Lie Down → Event ---")
    for cls in ["lying", "lie_down"]:
        if cls not in bdb: continue
        b_ev = bdb[cls]["pred_event"]; r_ev = bdr[cls]["pred_event"]
        t = bdb[cls]["total"]
        print(f"  {cls:>12s}: {b_ev:>6,d} ({b_ev/t*100:.1f}%) → {r_ev:>6,d} ({r_ev/t*100:.1f}%) Δ={r_ev-b_ev:+d}")

    # Miss rates
    cm_b = np.array(bt["confusion_matrix"]); cm_r = np.array(rt["confusion_matrix"])
    print(f"\n--- Safety: GT→Normal miss ---")
    for i, name in enumerate(["fall", "fallen"]):
        rsb = cm_b[i].sum() or 1; rsr = cm_r[i].sum() or 1
        print(f"  {name}→normal: {cm_b[i,2]/rsb:.1%} → {cm_r[i,2]/rsr:.1%}")


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)

    print("=" * 70)
    print("  P7d Δframe + RuleV1 + Binary Event — Full Analysis")
    print("=" * 70)

    # ── p7d_delta ──
    pv_delta = load_and_infer(
        "logs/phase7/p7d_delta/best_model.pt",
        "data/omnifall_dinov2_giant_frame.npz",
        "DATASET-omnifall/splits/syn/random",
    )
    corrected_delta, stats_delta = apply_rulev1(pv_delta)
    results_delta = compute_all_metrics(pv_delta, corrected_delta)
    print_comparison(results_delta, "P7d Δframe + RuleV1")

    # Save results
    out = {
        "bridge_ternary": results_delta["bridge_ternary"],
        "rulev1_ternary": results_delta["rulev1_ternary"],
        "bridge_binary": results_delta["bridge_binary"],
        "rulev1_binary": results_delta["rulev1_binary"],
        "breakdown_bridge": results_delta["breakdown_bridge"],
        "breakdown_rulev1": results_delta["breakdown_rulev1"],
        "rule_stats": stats_delta,
    }
    out_path = "logs/phase7/p7d_delta/analysis_rulev1.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
