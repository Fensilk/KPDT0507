"""
Analyze error patterns inside RuleC events and identify safe corrections.

Questions:
1. How many GT fall events vs RuleC vs RuleV1 events?
2. Inside RuleC events, what prediction errors exist?
3. Which errors can be safely corrected?
"""
import os, sys, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_rule_c import (
    load_and_infer, apply_rule_c, apply_rulev1, find_best_fall_run_in_window
)


def apply_rule_c_with_state(pv, W=64, K=5, fallback_timeout=None):
    """
    RuleC that also returns per-frame event state.
    event_state[f] = True if frame f is inside a confirmed event.
    """
    if fallback_timeout is None:
        from experiments.eval_rule_c import compute_max_fallen_seq
        fallback_timeout = compute_max_fallen_seq(pv)

    corr = {}
    event_masks = {}  # per-video boolean array
    stats = {"events": 0, "events_ended_by_fall": 0, "events_ended_by_timeout": 0,
             "lookback_success": 0, "lookback_fail": 0}

    for vi, d in pv.items():
        bp = d["bridge_pred"].copy()
        nf = len(bp)
        cp = bp.copy()
        em = np.zeros(nf, dtype=bool)

        ev = False
        cf = 0
        fb_timer = 0
        event_fall_start = -1  # frame index where the confirmed fall run starts

        for f in range(nf):
            p = bp[f]

            if not ev:
                cf = cf + 1 if p == 0 else 0

                if p == 1:  # fallen
                    lb_start = max(0, f - W)
                    best_run, best_end = find_best_fall_run_in_window(bp, lb_start, f, K)

                    if best_run >= K:
                        ev = True
                        stats["events"] += 1
                        stats["lookback_success"] += 1
                        fb_timer = 0
                        # Mark from fall run start to current frame as in-event
                        event_fall_start = best_end - best_run + 1
                        em[event_fall_start:f+1] = True
                    else:
                        cp[f] = 2
                        stats["lookback_fail"] += 1
            else:
                em[f] = True  # this frame is in event

                if p == 1:  # fallen
                    fb_timer = 0
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
        event_masks[vi] = em

    return corr, event_masks, stats


def analyze():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)

    CKPT = "logs/phase6/p6_bridge/best_model.pt"
    NPZ  = "data/omnifall_dinov2_giant_frame.npz"
    SPLITS = "DATASET-omnifall/splits/syn/random"

    print("Loading p6_bridge...")
    pv = load_and_infer(CKPT, NPZ, SPLITS, use_diff=False)

    # ── Q1: GT event count ──
    gt_events = 0
    gt_fall_lens, gt_fallen_lens = [], []
    for vi, d in pv.items():
        gt = d["ternary_gt"]; nf = len(gt); f = 0
        while f < nf:
            if gt[f] == 0:
                fs = f
                while f < nf and gt[f] == 0: f += 1
                fe = f
                while f < nf and gt[f] == 2: f += 1  # skip normal gap
                if f < nf and gt[f] == 1:
                    fds = f
                    while f < nf and gt[f] == 1: f += 1
                    fde = f
                    gt_events += 1
                    gt_fall_lens.append(fe - fs)
                    gt_fallen_lens.append(fde - fds)
                # else: fall without fallen — not counted as event
            else:
                f += 1

    print(f"\n{'='*60}")
    print(f"Q1: EVENT COUNTS")
    print(f"{'='*60}")
    print(f"  GT fall events:              {gt_events:>6}")
    print(f"  GT fall len:  mean={np.mean(gt_fall_lens):.1f}, "
          f"median={np.median(gt_fall_lens):.0f}, max={max(gt_fall_lens)}")
    print(f"  GT fallen len: mean={np.mean(gt_fallen_lens):.1f}, "
          f"median={np.median(gt_fallen_lens):.0f}, max={max(gt_fallen_lens)}")

    # RuleV1 events
    _, stats_v1_3 = apply_rulev1(pv, k=3)
    _, stats_v1_5 = apply_rulev1(pv, k=5)

    # RuleC events
    _, _, stats_c_64_5 = apply_rule_c_with_state(pv, W=64, K=5)
    _, _, stats_c_32_5 = apply_rule_c_with_state(pv, W=32, K=5)

    print(f"\n  {'Method':<24s} {'Events':>8s} {'vs GT':>10s}")
    print(f"  {'-'*42}")
    print(f"  {'GT (ground truth)':<24s} {gt_events:>8}")
    print(f"  {'RuleV1(k=3)':<24s} {stats_v1_3['events']:>8}  +{stats_v1_3['events']-gt_events:>5} (+{(stats_v1_3['events']-gt_events)/gt_events*100:.0f}%)")
    print(f"  {'RuleV1(k=5)':<24s} {stats_v1_5['events']:>8}  +{stats_v1_5['events']-gt_events:>5} (+{(stats_v1_5['events']-gt_events)/gt_events*100:.0f}%)")
    print(f"  {'RuleC(W=64,K=5)':<24s} {stats_c_64_5['events']:>8}  +{stats_c_64_5['events']-gt_events:>5} (+{(stats_c_64_5['events']-gt_events)/gt_events*100:.0f}%)")
    print(f"  {'RuleC(W=32,K=5)':<24s} {stats_c_32_5['events']:>8}  +{stats_c_32_5['events']-gt_events:>5} (+{(stats_c_32_5['events']-gt_events)/gt_events*100:.0f}%)")

    # ── Q2: Error patterns inside RuleC events ──
    _, em, _ = apply_rule_c_with_state(pv, W=64, K=5)

    # Per-position error analysis inside events
    total_in_event = 0
    # Count: (gt, pred) pairs inside events
    confusion_in_event = np.zeros((3, 3), dtype=int)
    # Count: isolated errors inside events
    isolated_stats = {
        "N_in_fallen_cluster": 0,   # isolated N within a run of D
        "F_in_fallen_cluster": 0,   # isolated F within a run of D
        "N_in_fall_gap": 0,         # N in the transition between F cluster and D cluster
        "D_in_fall_cluster": 0,     # D within a run of F
    }

    for vi, d in pv.items():
        bp = d["bridge_pred"]
        gt = d["ternary_gt"]
        em_v = em[vi]
        nf = len(bp)

        for f in range(nf):
            if em_v[f]:
                total_in_event += 1
                confusion_in_event[gt[f], bp[f]] += 1

        # Find isolated errors within event
        # Approach: within event masks, find runs and single-frame deviations
        in_ev = False
        f = 0
        while f < nf:
            if em_v[f]:
                if not in_ev:
                    in_ev = True
                    ev_start = f
                # Process when event ends
                if f == nf - 1 or not em_v[f+1]:
                    ev_end = f + 1
                    # Analyze this event segment
                    seg_bp = bp[ev_start:ev_end]
                    seg_gt = gt[ev_start:ev_end]
                    seg_len = ev_end - ev_start

                    # Find isolated N within D runs (≥3 D on both sides)
                    for j in range(2, seg_len - 2):
                        if seg_bp[j] == 2:  # normal
                            # Check if surrounded by fallen
                            left_d = all(seg_bp[max(0,j-3):j] == 1)
                            right_d = all(seg_bp[j+1:min(seg_len,j+4)] == 1)
                            if left_d and right_d:
                                isolated_stats["N_in_fallen_cluster"] += 1

                    # Find isolated F within D runs
                    for j in range(2, seg_len - 2):
                        if seg_bp[j] == 0:  # fall
                            left_d = all(seg_bp[max(0,j-3):j] == 1)
                            right_d = all(seg_bp[j+1:min(seg_len,j+4)] == 1)
                            if left_d and right_d:
                                isolated_stats["F_in_fallen_cluster"] += 1

                    # Find D within F runs
                    for j in range(2, seg_len - 2):
                        if seg_bp[j] == 1:  # fallen
                            left_f = all(seg_bp[max(0,j-3):j] == 0)
                            right_f = all(seg_bp[j+1:min(seg_len,j+4)] == 0)
                            if left_f and right_f:
                                isolated_stats["D_in_fall_cluster"] += 1
            else:
                in_ev = False
            f += 1

    print(f"\n{'='*60}")
    print(f"Q2: ERRORS INSIDE RuleC(W=64,K=5) EVENTS")
    print(f"{'='*60}")
    print(f"  Total frames inside events: {total_in_event:,}")
    print(f"\n  Confusion matrix (GT vs Pred) inside events:")
    print(f"                pred_fall  pred_fallen  pred_normal")
    for i, name in enumerate(["GT fall  ", "GT fallen", "GT normal"]):
        row = confusion_in_event[i]
        print(f"  {name:<12s} {row[0]:>10,}  {row[1]:>10,}  {row[2]:>10,}")

    # Accuracy inside vs outside events
    in_event_mask = np.concatenate([em[vi] for vi in sorted(em.keys())])
    all_gt = np.concatenate([pv[vi]["ternary_gt"] for vi in sorted(pv.keys())])
    all_bp = np.concatenate([pv[vi]["bridge_pred"] for vi in sorted(pv.keys())])

    in_correct = (all_gt[in_event_mask] == all_bp[in_event_mask]).sum()
    in_total = in_event_mask.sum()
    out_correct = (all_gt[~in_event_mask] == all_bp[~in_event_mask]).sum()
    out_total = (~in_event_mask).sum()

    print(f"\n  Accuracy inside events:  {in_correct/in_total*100:.1f}% ({in_correct:,}/{in_total:,})")
    print(f"  Accuracy outside events: {out_correct/out_total*100:.1f}% ({out_correct:,}/{out_total:,})")

    # ── Q3: Safe corrections ──
    print(f"\n{'='*60}")
    print(f"Q3: CANDIDATES FOR SAFE CORRECTION")
    print(f"{'='*60}")
    print(f"  (inside RuleC events only)")
    print(f"")
    print(f"  Isolated N within D cluster:  {isolated_stats['N_in_fallen_cluster']:>6}")
    print(f"  Isolated F within D cluster:  {isolated_stats['F_in_fallen_cluster']:>6}")
    print(f"  Isolated D within F cluster:  {isolated_stats['D_in_fall_cluster']:>6}")

    # Count how many corrections would flip to correct vs incorrect
    # For each isolated N in D cluster: GT is likely fallen → correction would help
    # For each isolated F in D cluster: GT is likely fallen → correction would help

    # Now: count total frames that are mispredicted inside events
    total_errors_in_event = int((all_gt[in_event_mask] != all_bp[in_event_mask]).sum())
    print(f"\n  Total prediction errors inside events: {total_errors_in_event:,}")
    print(f"  Of which:")
    for gt_i, gt_name in enumerate(["fall", "fallen", "normal"]):
        for pred_j, pred_name in enumerate(["fall", "fallen", "normal"]):
            if gt_i != pred_j:
                n = confusion_in_event[gt_i, pred_j]
                if n > 0:
                    print(f"    {gt_name} → {pred_name}: {n:,}")

    # What % of errors inside events are isolated (correctable)?
    isolatable = sum(isolated_stats.values())
    print(f"\n  Isolated errors (correctable by smoothing): {isolatable:,}")
    print(f"  = {isolatable/total_errors_in_event*100:.1f}% of all in-event errors")

    # ── Per-correction impact analysis ──
    print(f"\n  ┌─ Correction impact (RuleC W=64,K=5) ─────────────────")

    # Correction 1: Isolated N in fallen cluster → fallen
    n_corrections = isolated_stats['N_in_fallen_cluster']
    # How many would be correct?
    # We need to actually check each isolated N's GT
    # For a rough estimate: in fallen clusters, GT is mostly fallen
    print(f"  │ C1: Isolated N in D cluster → D        {n_corrections:>6} frames")

    # Correction 2: Isolated F in fallen cluster → fallen
    f_corrections = isolated_stats['F_in_fallen_cluster']
    print(f"  │ C2: Isolated F in D cluster → D        {f_corrections:>6} frames")

    # Correction 3: Isolated D in fall cluster → fall
    d_corrections = isolated_stats['D_in_fall_cluster']
    print(f"  │ C3: Isolated D in F cluster → F        {d_corrections:>6} frames")

    total_correctable = n_corrections + f_corrections + d_corrections
    print(f"  │ Total correctable isolated errors:     {total_correctable:>6} frames")
    print(f"  └──────────────────────────────────────────────────")

    # ── More detailed analysis: verify each correction ──
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Check GT for each correction candidate")
    print(f"{'='*60}")

    # Re-scan with GT verification
    verify = {
        "N_in_D→D": {"total": 0, "gt_fallen": 0, "gt_fall": 0, "gt_normal": 0},
        "F_in_D→D": {"total": 0, "gt_fallen": 0, "gt_fall": 0, "gt_normal": 0},
        "D_in_F→F": {"total": 0, "gt_fallen": 0, "gt_fall": 0, "gt_normal": 0},
    }

    for vi, d in pv.items():
        bp = d["bridge_pred"]
        gt = d["ternary_gt"]
        em_v = em[vi]
        nf = len(bp)
        f = 0
        while f < nf:
            if em_v[f]:
                ev_start = f
                while f < nf and em_v[f]: f += 1
                ev_end = f

                seg_bp = bp[ev_start:ev_end]
                seg_gt = gt[ev_start:ev_end]
                seg_len = ev_end - ev_start

                for j in range(2, seg_len - 2):
                    abs_j = ev_start + j

                    # N in D cluster
                    if seg_bp[j] == 2:
                        left_d = all(seg_bp[max(0,j-3):j] == 1)
                        right_d = all(seg_bp[j+1:min(seg_len,j+4)] == 1)
                        if left_d and right_d and seg_gt[j] != 2:  # only count if GT is not normal
                            verify["N_in_D→D"]["total"] += 1
                            if seg_gt[j] == 1: verify["N_in_D→D"]["gt_fallen"] += 1
                            elif seg_gt[j] == 0: verify["N_in_D→D"]["gt_fall"] += 1
                            elif seg_gt[j] == 2: verify["N_in_D→D"]["gt_normal"] += 1

                    # F in D cluster
                    if seg_bp[j] == 0:
                        left_d = all(seg_bp[max(0,j-3):j] == 1)
                        right_d = all(seg_bp[j+1:min(seg_len,j+4)] == 1)
                        if left_d and right_d and seg_gt[j] != 0:  # only if GT is not fall
                            verify["F_in_D→D"]["total"] += 1
                            if seg_gt[j] == 1: verify["F_in_D→D"]["gt_fallen"] += 1
                            elif seg_gt[j] == 0: verify["F_in_D→D"]["gt_fall"] += 1
                            elif seg_gt[j] == 2: verify["F_in_D→D"]["gt_normal"] += 1

                    # D in F cluster
                    if seg_bp[j] == 1:
                        left_f = all(seg_bp[max(0,j-3):j] == 0)
                        right_f = all(seg_bp[j+1:min(seg_len,j+4)] == 0)
                        if left_f and right_f and seg_gt[j] != 1:  # only if GT is not fallen
                            verify["D_in_F→F"]["total"] += 1
                            if seg_gt[j] == 0: verify["D_in_F→F"]["gt_fall"] += 1
                            elif seg_gt[j] == 1: verify["D_in_F→F"]["gt_fallen"] += 1
                            elif seg_gt[j] == 2: verify["D_in_F→F"]["gt_normal"] += 1
            else:
                f += 1

    for corr_name, v in verify.items():
        t = v["total"]
        if t > 0:
            print(f"\n  {corr_name}: {t} candidates")
            print(f"    GT=fall:   {v['gt_fall']:>4} ({v['gt_fall']/t*100:.1f}%)")
            print(f"    GT=fallen: {v['gt_fallen']:>4} ({v['gt_fallen']/t*100:.1f}%)")
            print(f"    GT=normal: {v['gt_normal']:>4} ({v['gt_normal']/t*100:.1f}%)")
            correct = v["gt_fallen"] if "→D" in corr_name else v["gt_fall"]
            wrong = t - correct
            print(f"    → correction {'HELPS' if correct > wrong else 'HURTS'} "
                  f"({correct} correct vs {wrong} wrong)")


if __name__ == "__main__":
    analyze()
