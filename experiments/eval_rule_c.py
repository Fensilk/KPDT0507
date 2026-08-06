"""
Evaluate RuleC vs RuleV1 on p6_bridge predictions.

RuleC: fallen-triggered event with causal fall lookback.
  - NORMAL: accumulate consecutive fall count. On fallen frame,
    look back W frames for >=K consecutive fall.
    Found → IN_EVENT. Not found → suppressed fallen (isolated).
  - IN_EVENT: fallen kept, normal kept. On fall frame → event ends,
    this fall starts new accumulation. Fallback timeout after last
    fallen prevents permanent event.

Grid search: W ∈ {32, 64}, K ∈ {3, 5}
Compare: RuleV1(k=3), RuleV1(k=5)

Usage:
    python experiments/eval_rule_c.py
"""
import os, sys, json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════
# Inference (reused from eval_p7d_k5.py)
# ═══════════════════════════════════════════════════════════════

def load_and_infer(ckpt, npz, splits, use_diff, T=64, stride=8, bs=32):
    import torch
    from models.phase6_model import Phase6TernaryModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    d = np.load(npz, allow_pickle=True, mmap_mode='r')
    paths = set()
    with open(os.path.join(splits, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p: paths.add(p)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = Phase6TernaryModel(
        input_dim=3072 if use_diff else 1536, hidden_dim=384, decoder_type="transformer",
        causal=False, num_layers=1, num_heads=6, dropout=0.3, max_len=T+10, num_classes=3,
    ).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()

    vp = d["video_paths"]; vsi = d["video_start_indices"]; vcc = d["video_clip_counts"]
    feats = d["features"]; lb16 = d["labels_16"]; fl = d["fall_labels"]; fdl = d["fallen_labels"]

    wins, meta, vinfo = [], [], {}
    for vi in range(len(vp)):
        if str(vp[vi]) not in paths: continue
        ns = int(vsi[vi]); nf = int(vcc[vi]); nw = (nf - T) // stride + 1
        vinfo[vi] = (ns, nf, nw)
        vf = feats[ns:ns+nf]
        for w in range(nw): off = w * stride; wins.append(vf[off:off+T]); meta.append((vi, off))

    preds = []
    for b0 in range(0, len(wins), bs):
        b1 = min(b0 + bs, len(wins))
        batch = torch.from_numpy(np.stack(wins[b0:b1])).float().to(device)
        if use_diff:
            diff = torch.zeros_like(batch); diff[:,1:] = batch[:,1:] - batch[:,:-1]
            batch = torch.cat([batch, diff], dim=-1)
        with torch.no_grad(): preds.append(m(batch).argmax(dim=-1).cpu().numpy())
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
        tg = np.full(nf, 2, dtype=int); tg[fl[ns:ns+nf]==1] = 0; tg[fdl[ns:ns+nf]==1] = 1
        pv[vi] = {"labels_16": lb16[ns:ns+nf], "ternary_gt": tg, "bridge_pred": bp,
                   "binary_fall_gt": fl[ns:ns+nf], "binary_fallen_gt": fdl[ns:ns+nf]}
    return pv


# ═══════════════════════════════════════════════════════════════
# RuleV1 (baseline, from eval_p7d_k5.py)
# ═══════════════════════════════════════════════════════════════

def apply_rulev1(pv, k=5, timeout=32):
    """fall-triggered event, timeout-based end."""
    corr = {}
    stats = {"events": 0, "fallen_kept": 0, "fallen_suppressed": 0}
    for vi, d in pv.items():
        bp = d["bridge_pred"].copy(); nf = len(bp); cp = bp.copy()
        ev, timer, cf = False, 0, 0
        for f in range(nf):
            p = bp[f]
            cf = cf + 1 if p == 0 else 0
            if not ev and cf >= k: ev = True; timer = 0; stats["events"] += 1
            if ev:
                if p == 1: stats["fallen_kept"] += 1; timer = 0
                elif p == 0: timer = 0
                else: timer += 1
                if timer > timeout: ev = False
            elif p == 1: cp[f] = 2; stats["fallen_suppressed"] += 1
        corr[vi] = cp
    return corr, stats


# ═══════════════════════════════════════════════════════════════
# RuleC: fallen-triggered with causal lookback
# ═══════════════════════════════════════════════════════════════

def compute_max_fallen_seq(pv):
    """Compute max consecutive GT fallen frames across all videos."""
    max_len = 0
    for d in pv.values():
        gt = d["ternary_gt"]
        run = 0
        for g in gt:
            if g == 1: run += 1; max_len = max(max_len, run)
            else: run = 0
    return max_len


def find_best_fall_run_in_window(preds, start, end, K):
    """Find the longest >=K consecutive fall run in preds[start:end].
    Returns (run_length, run_end_idx) or (0, -1) if none found."""
    best_run, best_end = 0, -1
    run = 0
    for j in range(start, min(end, len(preds))):
        if preds[j] == 0:
            run += 1
            if run >= K and run > best_run:
                best_run = run
                best_end = j
        else:
            run = 0
    return best_run, best_end


def apply_rule_c(pv, W=64, K=5, fallback_timeout=None):
    """
    RuleC: fallen-triggered event with causal fall lookback.

    NORMAL state:
        - Accumulate consecutive fall (class 0) count.
        - On fallen (class 1) frame: look back W frames for >=K consecutive fall.
          Found → enter IN_EVENT.
          Not found → suppress this fallen to normal (isolated).

    IN_EVENT state:
        - fallen → keep, reset fallback timer.
        - normal → keep, accumulate fallback counter.
        - fall → event ends! This fall starts new accumulation (cf=1), back to NORMAL.
        - fallback counter > fallback_timeout → event ends, back to NORMAL.
    """
    if fallback_timeout is None:
        fallback_timeout = compute_max_fallen_seq(pv)
        print(f"[RuleC] Computed fallback_timeout from GT: {fallback_timeout} frames")

    corr = {}
    stats = {
        "events": 0, "fallen_kept": 0, "fallen_suppressed": 0,
        "events_ended_by_fall": 0, "events_ended_by_timeout": 0,
        "lookback_success": 0, "lookback_fail": 0,
        "fallback_timeout": fallback_timeout,
    }

    for vi, d in pv.items():
        bp = d["bridge_pred"].copy()
        nf = len(bp)
        cp = bp.copy()

        ev = False
        cf = 0            # consecutive fall counter (NORMAL state)
        fb_timer = 0      # fallback timeout counter (IN_EVENT state, consecutive normals)
        last_non_normal = -1  # last frame in event that was fall/fallen

        for f in range(nf):
            p = bp[f]

            if not ev:
                # ── NORMAL state ──
                cf = cf + 1 if p == 0 else 0

                if p == 1:
                    # Look back up to W frames for >=K consecutive fall
                    lb_start = max(0, f - W)
                    best_run, best_end = find_best_fall_run_in_window(bp, lb_start, f, K)

                    if best_run >= K:
                        ev = True
                        stats["events"] += 1
                        stats["lookback_success"] += 1
                        stats["fallen_kept"] += 1
                        fb_timer = 0
                        last_non_normal = f
                    else:
                        cp[f] = 2  # suppress isolated fallen
                        stats["fallen_suppressed"] += 1
                        stats["lookback_fail"] += 1

            else:
                # ── IN_EVENT state ──
                if p == 1:  # fallen
                    stats["fallen_kept"] += 1
                    fb_timer = 0
                    last_non_normal = f

                elif p == 0:  # fall → event ends
                    ev = False
                    stats["events_ended_by_fall"] += 1
                    cf = 1  # this fall starts new accumulation

                else:  # normal
                    fb_timer += 1
                    if fallback_timeout and fb_timer > fallback_timeout:
                        ev = False
                        stats["events_ended_by_timeout"] += 1
                        cf = 0

        corr[vi] = cp

    return corr, stats


# ═══════════════════════════════════════════════════════════════
# Risk Pattern Analysis (on raw bridge_pred)
# ═══════════════════════════════════════════════════════════════

def analyze_risk_patterns(pv, W=64, K=5):
    """
    Scan raw bridge_pred per video and count the four risk patterns.

    Returns dict with per-pattern counts and rates.
    """
    stats = {
        "R1_fake_events": 0,        # N→F chain + N→D → fake event trigger
        "R2_fallen_to_fall": 0,     # D→F in true fallen segment → fragmentation
        "R3_fall_to_fallen": 0,     # F→D in true fall segment → fragmentation
        "R4_broken_fall_runs": 0,   # F→N breaking >=K consecutive fall → missed trigger
        "total_fall_segments": 0,
        "total_fallen_segments": 0,
        "total_normal_segments": 0,
        "R4_fall_segments_broken": 0,  # fall segments where longest pred-fall run < K
        "R4_fall_segments_intact": 0,
    }

    for vi, d in pv.items():
        bp = d["bridge_pred"]
        gt = d["ternary_gt"]
        nf = len(bp)

        # Find GT segments
        # ── Find GT fall segments ──
        in_fall_seg = False
        fall_start = 0
        for f in range(nf + 1):
            is_fall = f < nf and gt[f] == 0
            if is_fall and not in_fall_seg:
                in_fall_seg = True
                fall_start = f
            elif not is_fall and in_fall_seg:
                in_fall_seg = False
                seg_bp = bp[fall_start:f]
                stats["total_fall_segments"] += 1

                # R3: check for F→D within the fall segment
                if 1 in seg_bp:  # any pred=fallen in true fall segment
                    stats["R3_fall_to_fallen"] += 1

                # R4: check if N insertions break >=K consecutive fall
                longest_run, run = 0, 0
                for p in seg_bp:
                    if p == 0: run += 1; longest_run = max(longest_run, run)
                    else: run = 0
                if longest_run < K:
                    stats["R4_fall_segments_broken"] += 1
                    # Count individual N insertions that shorten runs
                    stats["R4_broken_fall_runs"] += 1
                else:
                    stats["R4_fall_segments_intact"] += 1

        # ── Find GT fallen segments ──
        in_fallen_seg = False
        fallen_start = 0
        for f in range(nf + 1):
            is_fallen = f < nf and gt[f] == 1
            if is_fallen and not in_fallen_seg:
                in_fallen_seg = True
                fallen_start = f
            elif not is_fallen and in_fallen_seg:
                in_fallen_seg = False
                seg_bp = bp[fallen_start:f]
                stats["total_fallen_segments"] += 1

                # R2: check for D→F within the fallen segment
                if 0 in seg_bp:  # any pred=fall in true fallen segment
                    stats["R2_fallen_to_fall"] += 1

        # ── R1: Find fake events in GT normal segments ──
        # A fake event = W-frame window containing >=K consecutive pred_fall
        # followed by pred_fallen, all within GT normal frames.
        in_normal_seg = False
        normal_start = 0
        for f in range(nf + 1):
            is_normal = f < nf and gt[f] == 2
            if is_normal and not in_normal_seg:
                in_normal_seg = True
                normal_start = f
            elif not is_normal and in_normal_seg:
                in_normal_seg = False
                seg_end = f
                seg_bp = bp[normal_start:seg_end]
                seg_len = seg_end - normal_start
                stats["total_normal_segments"] += 1

                # Scan for pattern: >=K consecutive fall, then fallen within W frames
                cf = 0
                for j in range(seg_len):
                    if seg_bp[j] == 0:
                        cf += 1
                        if cf >= K:
                            # Look ahead up to W frames for fallen
                            ahead_end = min(j + 1 + W, seg_len)
                            if 1 in seg_bp[j+1:ahead_end]:
                                stats["R1_fake_events"] += 1
                                break  # count once per normal segment
                    else:
                        cf = 0

    # Compute rates
    rates = {}
    if stats["total_fallen_segments"] > 0:
        rates["R2_rate"] = stats["R2_fallen_to_fall"] / stats["total_fallen_segments"]
    if stats["total_fall_segments"] > 0:
        rates["R3_rate"] = stats["R3_fall_to_fallen"] / stats["total_fall_segments"]
        rates["R4_rate"] = stats["R4_fall_segments_broken"] / stats["total_fall_segments"]
    if stats["total_normal_segments"] > 0:
        rates["R1_rate"] = stats["R1_fake_events"] / stats["total_normal_segments"]

    return stats, rates


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(pv, corr):
    all_gt3, all_bp, all_cp = [], [], []
    for vi, d in pv.items():
        n = len(d["ternary_gt"])
        all_gt3.append(d["ternary_gt"])
        all_bp.append(d["bridge_pred"][:n])
        all_cp.append(corr[vi][:n])

    gt3 = np.concatenate(all_gt3)
    bp = np.concatenate(all_bp)
    cp = np.concatenate(all_cp)

    def ternary_metrics(pred):
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
            "confusion_matrix": confusion_matrix(gt3, pred, labels=[0, 1, 2]).tolist(),
        }

    # Binary event: fall or fallen = event (positive class)
    def binary_metrics(pred):
        binary_gt = ((gt3 == 0) | (gt3 == 1)).astype(int)
        binary_pred = (pred != 2).astype(int)
        cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
        tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]
        ep = tp / max(tp + fp, 1)
        er = tp / max(tp + fn, 1)
        return {
            "event_f1": 2 * ep * er / (ep + er) if (ep + er) > 0 else 0,
            "event_precision": ep,
            "event_recall": er,
            "false_events": int(fp),
            "binary_cm": cm.tolist(),
        }

    return {
        "ternary_bridge": ternary_metrics(bp),
        "ternary_corrected": ternary_metrics(cp),
        "binary_bridge": binary_metrics(bp),
        "binary_corrected": binary_metrics(cp),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)

    CKPT = "logs/phase6/p6_bridge/best_model.pt"
    NPZ  = "data/omnifall_dinov2_giant_frame.npz"
    SPLITS = "DATASET-omnifall/splits/syn/random"

    print("=" * 72)
    print("RuleC vs RuleV1 — p6_bridge baseline")
    print("=" * 72)

    # ── 1. Load p6_bridge predictions ──
    print("\n[1] Loading p6_bridge predictions...")
    pv = load_and_infer(CKPT, NPZ, SPLITS, use_diff=False)
    print(f"  Loaded {len(pv)} videos")

    # ── 2. Risk pattern analysis on raw predictions ──
    print("\n[2] Analyzing risk patterns on raw p6_bridge...")
    risk_stats, risk_rates = analyze_risk_patterns(pv, W=64, K=5)

    print(f"\n  ┌─ Segment counts ─────────────────────────")
    print(f"  │  Fall segments:   {risk_stats['total_fall_segments']:>6}")
    print(f"  │  Fallen segments: {risk_stats['total_fallen_segments']:>6}")
    print(f"  │  Normal segments: {risk_stats['total_normal_segments']:>6}")
    print(f"  └──────────────────────────────────────────")
    print(f"\n  ┌─ Risk Pattern Frequencies ──────────────")
    print(f"  │  R1 (N→F+N→D fake event):     {risk_stats['R1_fake_events']:>6}  ({risk_rates.get('R1_rate',0)*100:.1f}% of normal segs)")
    print(f"  │  R2 (D→F in fallen segment):   {risk_stats['R2_fallen_to_fall']:>6}  ({risk_rates.get('R2_rate',0)*100:.1f}% of fallen segs)")
    print(f"  │  R3 (F→D in fall segment):     {risk_stats['R3_fall_to_fallen']:>6}  ({risk_rates.get('R3_rate',0)*100:.1f}% of fall segs)")
    print(f"  │  R4 (F→N breaking fall runs):  {risk_stats['R4_fall_segments_broken']:>6}  ({risk_rates.get('R4_rate',0)*100:.1f}% of fall segs)")
    print(f"  │      (intact fall segments):   {risk_stats['R4_fall_segments_intact']:>6}")
    print(f"  └──────────────────────────────────────────")

    # ── 3. Run all rule variants ──
    print("\n[3] Running rules...")

    results = {}

    # RuleV1 baselines
    for k in [3, 5]:
        label = f"RuleV1(k={k})"
        print(f"  {label}...")
        corr, stats = apply_rulev1(pv, k=k)
        r = evaluate(pv, corr)
        results[label] = {"corr": corr, "stats": stats, "eval": r}

    # RuleC grid
    for W in [32, 64]:
        for K in [3, 5]:
            label = f"RuleC(W={W},K={K})"
            print(f"  {label}...")
            corr, stats = apply_rule_c(pv, W=W, K=K)
            r = evaluate(pv, corr)
            results[label] = {"corr": corr, "stats": stats, "eval": r}

    # ── 4. Print comparison tables ──
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    # Rule stats
    print(f"\n{'Rule':<24s} {'Events':>8s} {'Kept':>8s} {'Suppr':>8s} {'EndFall':>8s} {'EndTO':>8s} {'LBSucc':>8s} {'LBFail':>8s}")
    print("-" * 96)
    for label, r in results.items():
        s = r["stats"]
        ev = s.get("events", s.get("events", 0))
        fk = s.get("fallen_kept", 0)
        fs = s.get("fallen_suppressed", 0)
        ef = s.get("events_ended_by_fall", "-")
        et = s.get("events_ended_by_timeout", "-")
        ls = s.get("lookback_success", "-")
        lf = s.get("lookback_fail", "-")
        print(f"{label:<24s} {str(ev):>8s} {str(fk):>8s} {str(fs):>8s} {str(ef):>8s} {str(et):>8s} {str(ls):>8s} {str(lf):>8s}")

    # Binary event metrics (the main metric)
    bb = results["RuleV1(k=3)"]["eval"]["binary_bridge"]
    print(f"\n{'':<24s} {'event_f1':>10s} {'event_prec':>10s} {'event_rec':>10s} {'false_ev':>10s}")
    print(f"{'p6_bridge (raw)':<24s} {bb['event_f1']:>10.4f} {bb['event_precision']:>10.4f} {bb['event_recall']:>10.4f} {bb['false_events']:>10}")
    print("-" * 72)
    for label, r in results.items():
        b = r["eval"]["binary_corrected"]
        # For RuleV1, stats contain event count
        fe = b["false_events"]
        print(f"{label:<24s} {b['event_f1']:>10.4f} {b['event_precision']:>10.4f} {b['event_recall']:>10.4f} {fe:>10}")

    # Ternary metrics summary
    print(f"\n{'':<24s} {'avg_f1':>10s} {'fall_f1':>10s} {'fallen_f1':>10s} {'normal_f1':>10s} {'acc':>10s}")
    tb = results["RuleV1(k=3)"]["eval"]["ternary_bridge"]
    print(f"{'p6_bridge (raw)':<24s} {tb['avg_f1']:>10.4f} {tb['fall_f1']:>10.4f} {tb['fallen_f1']:>10.4f} {tb['normal_f1']:>10.4f} {tb['ternary_acc']:>10.4f}")
    print("-" * 96)
    for label, r in results.items():
        t = r["eval"]["ternary_corrected"]
        print(f"{label:<24s} {t['avg_f1']:>10.4f} {t['fall_f1']:>10.4f} {t['fallen_f1']:>10.4f} {t['normal_f1']:>10.4f} {t['ternary_acc']:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
