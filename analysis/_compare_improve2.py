"""
Compare improvement-2 experiment results against baseline.

Usage:
    python _compare_improve2.py
    python _compare_improve2.py --logs logs/phase2/imp2_*
"""

import os, sys, json, glob, argparse
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(log_dir):
    """Load test_results.json from a log directory."""
    path = os.path.join(log_dir, "test_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare improvement-2 experiments")
    parser.add_argument("--logs", type=str, nargs="*", default=None,
                        help="Specific log directories to compare")
    args = parser.parse_args()

    if args.logs:
        log_dirs = args.logs
    else:
        # Auto-discover imp2 experiment logs + baseline references
        log_dirs = sorted(glob.glob("logs/phase2/imp2_*"))
        # Also include reference baselines
        for ref in ["logs/phase1/tcn_full", "logs/phase1/tcn_syn_001"]:
            if os.path.exists(ref) and ref not in log_dirs:
                log_dirs.append(ref)

    if not log_dirs:
        print("No experiment logs found!")
        return

    # Collect results
    rows = []
    for ld in log_dirs:
        r = load_results(ld)
        if r is None:
            print(f"[SKIP] {ld}: no test_results.json")
            continue

        name = os.path.basename(ld)
        rows.append({
            "name": name,
            "fall_f1": r.get("fall_f1", 0),
            "fallen_f1": r.get("fallen_f1", 0),
            "avg_f1": r.get("avg_f1", 0),
            "seg_acc": r.get("seg_acc", 0),
            "loss": r.get("loss", 0),
        })

    if not rows:
        print("No valid results found!")
        return

    # Sort: put baseline-like names first, then by avg_f1
    def sort_key(row):
        is_baseline = "baseline" in row["name"].lower() or \
                      row["name"] in ("tcn_full", "tcn_syn_001")
        return (not is_baseline, -row["avg_f1"])

    rows.sort(key=sort_key)

    # Find baseline for delta computation
    baseline = None
    for r in rows:
        if r["name"] in ("tcn_full", "imp2_baseline"):
            baseline = r
            break
    if baseline is None:
        baseline = rows[0]  # use first as reference

    # Print table
    sep = "-" * 85
    print(f"\n{'='*85}")
    print("IMPROVEMENT 2 — EXPERIMENT COMPARISON")
    print(f"{'='*85}")
    print(f"Baseline: {baseline['name']}")
    print(sep)
    print(f"{'Experiment':<25} {'fall_f1':>8} {'fallen_f1':>8} {'avg_f1':>8} {'seg_acc':>8} {'Δ avg_f1':>10}")
    print(sep)

    for r in rows:
        delta = r["avg_f1"] - baseline["avg_f1"]
        delta_str = f"{delta:+.4f}" if delta != 0 else "  (baseline)"
        marker = ""
        if r["name"] == baseline["name"]:
            marker = " <-- baseline"
        print(f"{r['name']:<25} {r['fall_f1']:>8.4f} {r['fallen_f1']:>8.4f} "
              f"{r['avg_f1']:>8.4f} {r['seg_acc']:>8.4f} {delta_str:>10}{marker}")

    print(sep)

    # Best improvement
    non_baseline = [r for r in rows if r["name"] != baseline["name"]]
    if non_baseline:
        best = max(non_baseline, key=lambda r: r["avg_f1"])
        best_fall = max(non_baseline, key=lambda r: r["fall_f1"])
        print(f"Best avg_f1:  {best['name']} ({best['avg_f1']:.4f}, "
              f"Δ={best['avg_f1'] - baseline['avg_f1']:+.4f})")
        print(f"Best fall_f1: {best_fall['name']} ({best_fall['fall_f1']:.4f}, "
              f"Δ={best_fall['fall_f1'] - baseline['fall_f1']:+.4f})")

    print(f"{'='*85}\n")


if __name__ == "__main__":
    main()
