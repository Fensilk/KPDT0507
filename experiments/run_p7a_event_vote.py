"""
P7a: Event-level rule — scan Bridge predictions for fall→fallen sequences.

Test 3 (RuleV3): Bidirectional event cleaning.
  1. Detect event: >=3 consecutive fall preds → activate event window
  2. Within event, fill small gaps to produce clean fall, fallen, normal segments
  3. Outside event: isolated fallen → normal, isolated fall → normal
  4. Event expires after timeout without fall/fallen

Tests also include:
  Test 1: Cross-window boundary vote (original, negative result)
  Test 2: Oracle — GT fall query (upper bound)

Usage:
    python experiments/run_p7a_event_vote.py
    python experiments/run_p7a_event_vote.py --event_timeout 32 --gap_fill 5
"""

import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.phase6_model import Phase6TernaryModel
from utils.metrics import compute_ternary_metrics

TERNARY_NAMES = ["fall", "fallen", "normal"]
CLASS_NAMES_16 = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]


# ═══════════════════════════════════════════════════════════════
# Data loading & inference
# ═══════════════════════════════════════════════════════════════

def load_data(npz_path, splits_dir):
    d = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    paths = set()
    with open(os.path.join(splits_dir, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p: paths.add(p)
    return d, paths


def run_inference(model, npz_data, test_paths, T, stride, device, bs=32):
    vp = npz_data["video_paths"]; vsi = npz_data["video_start_indices"]
    vcc = npz_data["video_clip_counts"]; feats = npz_data["features"]
    lb16 = npz_data["labels_16"]; fl = npz_data["fall_labels"]
    fdl = npz_data["fallen_labels"]

    wins, meta, vinfo = [], [], {}
    for vi in range(len(vp)):
        if str(vp[vi]) not in test_paths: continue
        ns = int(vsi[vi]); nf = int(vcc[vi]); nw = (nf-T)//stride+1
        vinfo[vi] = (ns, nf, nw)
        vf = feats[ns:ns+nf]
        for w in range(nw):
            off = w*stride; wins.append(vf[off:off+T]); meta.append((vi, off))

    preds = []
    for b0 in range(0, len(wins), bs):
        b1 = min(b0+bs, len(wins))
        batch = torch.from_numpy(np.stack(wins[b0:b1])).float().to(device)
        with torch.no_grad(): preds.append(model(batch).argmax(dim=-1).cpu().numpy())
        if (b0//bs+1)%50==0: print(f"  [{b1}/{len(wins)}]")
    preds = np.concatenate(preds, axis=0)

    pv = {}
    for vi, (ns, nf, nw) in vinfo.items():
        votes = np.zeros((nf, 3), dtype=int)
        for wi, (vj, off) in enumerate(meta):
            if vj != vi: continue
            for t in range(T):
                f = off+t
                if f < nf: votes[f, preds[wi, t]] += 1
        bp = np.argmax(votes, axis=1)
        tg = np.full(nf, 2, dtype=int)
        tg[fl[ns:ns+nf]==1]=0; tg[fdl[ns:ns+nf]==1]=1
        pv[vi] = {"labels_16": lb16[ns:ns+nf], "ternary_gt": tg,
                   "bridge_pred": bp, "binary_fall_gt": fl[ns:ns+nf],
                   "n_windows": nw}
    return pv


# ═══════════════════════════════════════════════════════════════
# RuleV3: Bidirectional event cleaning
# ═══════════════════════════════════════════════════════════════

def apply_event_rule_v3(per_video, event_timeout=32, gap_fill=5):
    """
    Bidirectional event-level cleaning.

    Phase 1 — Detect events:
      Scan for >=3 consecutive fall → activate event.
      Event stays active while fall/fallen appear within timeout frames.
      Event expires after timeout consecutive non-fall/fallen frames.

    Phase 2 — Clean within each event:
      For frames inside [event_start, event_end]:
        - Fill small gaps: if <=gap_fill normal frames between fall/fallen → re-label
          to the surrounding class (fall near fall, fallen near fallen)
        - Bridge fall→fallen transition: the gap between last fall cluster and
          first fallen cluster → fill as fallen

    Phase 3 — Clean outside events:
      Isolated fallen → normal
      Isolated fall (single-frame or <3 consecutive) → normal
    """
    corrected = {}
    stats = {"events_detected": 0, "fallen_suppressed": 0, "fall_suppressed": 0,
             "fallen_filled": 0, "fall_filled": 0, "frames_in_event": 0}

    for vi, data in per_video.items():
        bp = data["bridge_pred"].copy()
        nf = len(bp)
        cp = bp.copy()

        # ── Phase 1: Find event boundaries ──
        events = []  # list of (start, end) indices
        in_event = False
        event_start = 0
        consecutive_fall = 0
        timer = 0

        for f in range(nf):
            p = bp[f]
            if p == 0: consecutive_fall += 1
            else: consecutive_fall = 0

            if not in_event and consecutive_fall >= 3:
                in_event = True
                event_start = max(0, f - 2)  # include the first fall frame
                timer = 0

            if in_event:
                if p in (0, 1):
                    timer = 0
                else:
                    timer += 1
                    if timer > event_timeout:
                        events.append((event_start, f - timer))
                        in_event = False
                        timer = 0

        if in_event:
            events.append((event_start, nf - 1))

        stats["events_detected"] += len(events)

        # ── Phase 2: Clean within each event ──
        event_mask = np.zeros(nf, dtype=bool)
        for es, ee in events:
            event_mask[es:ee+1] = True
            stats["frames_in_event"] += (ee - es + 1)

            # Fill gaps within the event
            for f in range(es, ee + 1):
                if bp[f] != 2:  # not normal — keep
                    continue

                # Check forward and backward within gap_fill for fall/fallen
                is_in_fall_context = False
                is_in_fallen_context = False

                # Look backward
                for b in range(max(es, f - gap_fill), f):
                    if bp[b] == 0: is_in_fall_context = True
                    if bp[b] == 1: is_in_fallen_context = True

                # Look forward
                for a in range(f + 1, min(ee + 1, f + gap_fill + 1)):
                    if bp[a] == 0: is_in_fall_context = True
                    if bp[a] == 1: is_in_fallen_context = True

                # Also check corrected predictions (cumulative fill)
                for b in range(max(es, f - gap_fill), f):
                    if cp[b] == 0: is_in_fall_context = True
                    if cp[b] == 1: is_in_fallen_context = True

                if is_in_fallen_context:
                    cp[f] = 1
                    stats["fallen_filled"] += 1
                elif is_in_fall_context:
                    cp[f] = 0
                    stats["fall_filled"] += 1
                # else: keep as normal (gap too wide)

        # ── Phase 3: Suppress isolated fall/fallen outside events ──
        for f in range(nf):
            if event_mask[f]:
                continue
            p = bp[f]
            if p == 1:  # isolated fallen
                cp[f] = 2
                stats["fallen_suppressed"] += 1
            elif p == 0:  # isolated fall
                cp[f] = 2
                stats["fall_suppressed"] += 1

        corrected[vi] = cp

    return corrected, stats


# ═══════════════════════════════════════════════════════════════
# Metrics & reporting
# ═══════════════════════════════════════════════════════════════

def compute_metrics(per_video, corrected):
    all_gt, all_bridge, all_corrected, all_l16 = [], [], [], []
    for vi, data in per_video.items():
        n = len(data["ternary_gt"])
        all_gt.append(data["ternary_gt"])
        all_bridge.append(data["bridge_pred"])
        all_corrected.append(corrected[vi][:n])
        all_l16.append(data["labels_16"])
    gt = np.concatenate(all_gt); br = np.concatenate(all_bridge)
    co = np.concatenate(all_corrected); l16 = np.concatenate(all_l16)

    def breakdown(pred):
        out = {}
        for c in range(16):
            m = l16 == c; n = int(m.sum())
            if n == 0: continue
            out[CLASS_NAMES_16[c]] = {
                "total": n, "pred_fall": int((pred[m]==0).sum()),
                "pred_fallen": int((pred[m]==1).sum()),
                "pred_normal": int((pred[m]==2).sum()),
            }
        return out

    return {
        "bridge": compute_ternary_metrics(br, gt),
        "corrected": compute_ternary_metrics(co, gt),
        "breakdown_bridge": breakdown(br),
        "breakdown_corrected": breakdown(co),
        "total_frames": int(len(gt)),
        "gt_dist": {"fall": int((gt==0).sum()), "fallen": int((gt==1).sum()),
                     "normal": int((gt==2).sum())},
    }


def print_report(metrics, rule_stats, label):
    b = metrics["bridge"]; c = metrics["corrected"]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Events detected: {rule_stats['events_detected']}")
    print(f"  Frames in events: {rule_stats['frames_in_event']:,}")
    print(f"  Fall filled (gap): {rule_stats['fall_filled']}")
    print(f"  Fallen filled (gap): {rule_stats['fallen_filled']}")
    print(f"  Fall suppressed (isolated): {rule_stats['fall_suppressed']}")
    print(f"  Fallen suppressed (isolated): {rule_stats['fallen_suppressed']}")

    keys = [("fall_f1","Fall F1"),("fall_precision","Fall Prec"),("fall_recall","Fall Rec"),
            ("fallen_f1","Fallen F1"),("fallen_precision","Fallen Prec"),("fallen_recall","Fallen Rec"),
            ("normal_f1","Normal F1"),("avg_f1","Avg F1"),("ternary_acc","Accuracy")]
    print(f"\n{'Metric':<22} {'Bridge':>10} {'RuleV3':>10} {'Delta':>10}")
    print("-"*54)
    for key, name in keys:
        bv, cv = b.get(key, 0), c.get(key, 0)
        print(f"{name:<22} {bv:>10.4f} {cv:>10.4f} {cv-bv:>+10.4f}")

    print(f"\n--- LYING / LIE_DOWN → FALLEN ---")
    bb, cb = metrics["breakdown_bridge"], metrics["breakdown_corrected"]
    for cls in ["lying", "lie_down"]:
        if cls not in bb: continue
        bt, ct = bb[cls]["total"], cb[cls]["total"]
        bf, cf = bb[cls]["pred_fallen"], cb[cls]["pred_fallen"]
        print(f"  {cls:>12s}: {bf:>6,d} ({bf/bt*100:.1f}%) → {cf:>6,d} ({cf/ct*100:.1f}%) Δ={cf-bf:+d}")

    print(f"\n--- CONFUSION (row-normalized) ---")
    cm_b = np.array(b["confusion_matrix"]); cm_c = np.array(c["confusion_matrix"])
    for i, name in enumerate(TERNARY_NAMES):
        rsb, rsc = cm_b[i].sum() or 1, cm_c[i].sum() or 1
        print(f"  {name:>8s}: Bridge [{cm_b[i,0]/rsb:.1%}f {cm_b[i,1]/rsb:.1%}fd {cm_b[i,2]/rsb:.1%}n]  →  "
              f"RuleV3 [{cm_c[i,0]/rsc:.1%}f {cm_c[i,1]/rsc:.1%}fd {cm_c[i,2]/rsc:.1%}n]")

    print(f"\n--- SAFETY: GT→normal miss rate ---")
    for i, name in enumerate(TERNARY_NAMES):
        if name == "normal": continue
        rsb, rsc = cm_b[i].sum() or 1, cm_c[i].sum() or 1
        mb, mc = cm_b[i,2]/rsb, cm_c[i,2]/rsc
        print(f"  {name:>8s}→normal: {mb:.1%} → {mc:.1%} (Δ={mc-mb:+.1%}) {'⚠' if mc>mb+0.001 else '✓'}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="P7a RuleV3: Bidirectional event cleaning")
    p.add_argument("--exp_dir", default="logs/phase6/p6_bridge")
    p.add_argument("--npz_path", default="data/omnifall_dinov2_giant_frame.npz")
    p.add_argument("--splits_dir", default="DATASET-omnifall/splits/syn/random")
    p.add_argument("--output_dir", default="logs/phase7/p7a_event_rule_v3")
    p.add_argument("--event_timeout", type=int, default=32)
    p.add_argument("--gap_fill", type=int, default=5)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device=="auto" else args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] P7a RuleV3: Bidirectional Event Cleaning")
    print(f"[INFO] Event timeout: {args.event_timeout}, Gap fill: {args.gap_fill}")
    print(f"[INFO] Device: {device}")

    # Load model
    results_path = os.path.join(BASE, args.exp_dir, "test_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f: cfg = json.load(f)["config"]
        hd, dt, cs, nl, nh, dr = (cfg["hidden_dim"], cfg["decoder_type"],
            cfg["causal"], cfg["num_layers"], cfg["num_heads"], cfg["dropout"])
    else:
        hd, dt, cs, nl, nh, dr = 384, "transformer", False, 1, 6, 0.3

    model = Phase6TernaryModel(
        input_dim=1536, hidden_dim=hd, decoder_type=dt,
        causal=cs, num_layers=nl, num_heads=nh,
        dropout=dr, max_len=args.window_size+10,
    ).to(device)
    ckpt = torch.load(os.path.join(BASE, args.exp_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Inference
    npz_data, test_paths = load_data(
        os.path.join(BASE, args.npz_path), os.path.join(BASE, args.splits_dir))
    per_video = run_inference(model, npz_data, test_paths,
                              T=args.window_size, stride=args.stride,
                              device=device, bs=args.batch_size)

    # ── RuleV3 ──
    print(f"\n[INFO] Applying RuleV3...")
    corrected, rule_stats = apply_event_rule_v3(
        per_video, event_timeout=args.event_timeout, gap_fill=args.gap_fill)

    metrics = compute_metrics(per_video, corrected)
    print_report(metrics, rule_stats,
                 f"RuleV3 (timeout={args.event_timeout}, gap_fill={args.gap_fill})")

    # Save
    results = {
        "config": {"experiment": "p7a_event_rule_v3", "bridge_exp": args.exp_dir,
                   "event_timeout": args.event_timeout, "gap_fill": args.gap_fill,
                   "window_size": args.window_size, "stride": args.stride},
        "rule_stats": rule_stats,
        "metrics_bridge": metrics["bridge"],
        "metrics_corrected": metrics["corrected"],
        "breakdown_bridge": metrics["breakdown_bridge"],
        "breakdown_corrected": metrics["breakdown_corrected"],
        "total_frames": metrics["total_frames"],
        "gt_dist": metrics["gt_dist"],
    }
    out = os.path.join(args.output_dir, "test_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved: {out}")

    return metrics


if __name__ == "__main__":
    main()
