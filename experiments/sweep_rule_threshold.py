"""
Sweep RuleV1 fall trigger threshold k (1-12) for p6_bridge and p7d_delta.

Usage:
    python experiments/sweep_rule_threshold.py
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

NORMAL_CLASSES = ["lying", "lie_down", "sit_down", "sitting", "stand_up",
                   "other", "walk", "standing", "jump"]


def load_and_infer(checkpoint_path, npz_path, splits_dir, use_diff, T=64, stride=8, bs=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    paths = set()
    with open(os.path.join(splits_dir, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p: paths.add(p)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Phase6TernaryModel(
        input_dim=3072 if use_diff else 1536, hidden_dim=384,
        decoder_type="transformer", causal=False, num_layers=1,
        num_heads=6, dropout=0.3, max_len=T + 10, num_classes=3,
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


def apply_rulev1(per_video, k=3, event_timeout=32):
    """Apply RuleV1 with fall trigger threshold k."""
    corrected = {}
    stats = {"events": 0, "fallen_kept": 0, "fallen_suppressed": 0}
    for vi, data in per_video.items():
        bp = data["bridge_pred"].copy(); nf = len(bp); cp = bp.copy()
        event_active = False; timer = 0; cons_fall = 0
        for f in range(nf):
            p = bp[f]
            cons_fall = cons_fall + 1 if p == 0 else 0
            if not event_active and cons_fall >= k:
                event_active = True; timer = 0; stats["events"] += 1
            if event_active:
                if p == 1: stats["fallen_kept"] += 1; timer = 0
                elif p == 0: timer = 0
                else:
                    timer += 1
                    if timer > event_timeout: event_active = False
            else:
                if p == 1: cp[f] = 2; stats["fallen_suppressed"] += 1
        corrected[vi] = cp
    return corrected, stats


def evaluate(pv, corrected):
    all_l16, all_bp, all_cp = [], [], []
    for vi, d in pv.items():
        n = len(d["ternary_gt"])
        all_l16.append(d["labels_16"]); all_bp.append(d["bridge_pred"])
        all_cp.append(corrected[vi][:n])
    l16 = np.concatenate(all_l16); bp = np.concatenate(all_bp)
    cp = np.concatenate(all_cp)

    results = {}
    for cls in NORMAL_CLASSES:
        mask = l16 == CLASS_NAMES_16.index(cls)
        b_ev = int(((bp[mask] == 0) | (bp[mask] == 1)).sum())
        c_ev = int(((cp[mask] == 0) | (cp[mask] == 1)).sum())
        results[cls] = {"bridge_event": b_ev, "rulev1_event": c_ev,
                         "total": int(mask.sum())}

    # Total normal→event
    results["_total_bridge"] = sum(results[c]["bridge_event"] for c in NORMAL_CLASSES)
    results["_total_rulev1"] = sum(results[c]["rulev1_event"] for c in NORMAL_CLASSES)
    results["_total_frames"] = sum(results[c]["total"] for c in NORMAL_CLASSES)
    return results


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    K_VALUES = [1, 2, 3, 5, 8, 12]

    results_bridge = {}
    results_delta = {}

    print("=" * 70)
    print("  RuleV1 Threshold Sweep: k ∈ {1,2,3,5,8,12}")
    print("=" * 70)

    # ── p6_bridge ──
    print("\n[1/2] Loading p6_bridge...")
    pv_bridge = load_and_infer(
        "logs/phase6/p6_bridge/best_model.pt",
        "data/omnifall_dinov2_giant_frame.npz",
        "DATASET-omnifall/splits/syn/random", use_diff=False)

    for k in K_VALUES:
        corrected, stats = apply_rulev1(pv_bridge, k=k)
        results_bridge[k] = evaluate(pv_bridge, corrected)
        results_bridge[k]["_stats"] = stats

    # ── p7d_delta ──
    print("\n[2/2] Loading p7d_delta...")
    pv_delta = load_and_infer(
        "logs/phase7/p7d_delta/best_model.pt",
        "data/omnifall_dinov2_giant_frame.npz",
        "DATASET-omnifall/splits/syn/random", use_diff=True)

    for k in K_VALUES:
        corrected, stats = apply_rulev1(pv_delta, k=k)
        results_delta[k] = evaluate(pv_delta, corrected)
        results_delta[k]["_stats"] = stats

    # ── Print results ──
    for model_name, results in [("p6_bridge", results_bridge), ("p7d_delta", results_delta)]:
        print(f"\n{'='*70}")
        print(f"  {model_name} — Sweep Results")
        print(f"{'='*70}")

        # Header
        header = f"{'k':>3s}  {'events':>6s}  {'suppr':>6s}"
        for cls in NORMAL_CLASSES:
            header += f"  {cls[:6]:>6s}"
        header += f"  {'TOTAL':>6s}  {'rate':>6s}  {'Δ%':>6s}"
        print(header)
        print("-" * len(header))

        base_total = results[K_VALUES[0]]["_total_bridge"]
        base_rate = base_total / results[K_VALUES[0]]["_total_frames"] * 100

        for k in K_VALUES:
            r = results[k]
            st = r["_stats"]
            total = r["_total_rulev1"]
            rate = total / r["_total_frames"] * 100
            delta = (total - base_total) / base_total * 100
            row = f"{k:>3d}  {st['events']:>6d}  {st['fallen_suppressed']:>6d}"
            for cls in NORMAL_CLASSES:
                row += f"  {r[cls]['rulev1_event']:>6d}"
            row += f"  {total:>6d}  {rate:>5.1f}%  {delta:>+5.1f}%"
            print(row)

        print(f"\n  Baseline total normal→event: {base_total} ({base_rate:.1f}%)")

    # ── Optimal k comparison ──
    print(f"\n{'='*70}")
    print(f"  OPTIMAL K — lying→event comparison")
    print(f"{'='*70}")
    print(f"  {'k':>3s}  {'p6_bridge lying→event':>20s}  {'p7d_delta lying→event':>20s}")
    print(f"  {'-'*50}")
    for k in K_VALUES:
        b_ly = results_bridge[k]["lying"]["rulev1_event"]
        d_ly = results_delta[k]["lying"]["rulev1_event"]
        print(f"  {k:>3d}  {b_ly:>20,d}  {d_ly:>20,d}")

    # Save
    out = {
        "K_VALUES": K_VALUES,
        "p6_bridge": {str(k): v for k, v in results_bridge.items()},
        "p7d_delta": {str(k): v for k, v in results_delta.items()},
    }
    out_path = "logs/phase7/threshold_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
