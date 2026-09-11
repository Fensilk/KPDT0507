"""
Sweep RuleV2 (K1=2, K2, W) params over E3-full vs baseline per-video predictions.

推理只做一次（结果缓存到 logs/phase9/e3_rulev2/cache/*.pt），规则网格全在缓存上跑，
秒级完成。指标口径与 eval_rule_c.py / eval_p9_rulev2.py 一致：
  - per-video 窗口推理 → 帧级投票 merge → 每帧三元预测
  - ternary + binary event（event=fall|fallen）帧展平

注意：K2/W 在 test 上选参 → 报"best"属 test-selected 上界（乐观），作诊断用。

Usage:
    python experiments/sweep_rulev2.py
"""
import os, sys, json
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_rule_c import load_and_infer, apply_rulev2

OUT_DIR = "logs/phase9/e3_rulev2"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
K1 = 2                     # 固定（用户指定只调 K2/W）
K2_GRID = [3, 5, 7, 10, 15, 20]
W_GRID = [32, 64, 128]

MODELS = [
    ("baseline", "logs/phase7/p7d_delta/best_model.pt", "data/omnifall_dinov2_giant_frame.npz"),
    ("E3-full", "logs/phase9/e3_stage2/best_model.pt", "data/omnifall_dinov2_finetuned_frame.npz"),
]


def ternary_metrics(gt3, pred):
    return {
        "ternary_acc": float((pred == gt3).mean()),
        "fall_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
        "fall_precision": float(precision_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
        "fall_recall": float(recall_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[0]),
        "fallen_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
        "fallen_precision": float(precision_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
        "fallen_recall": float(recall_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0)[1]),
        "avg_f1": float(f1_score(gt3, pred, average=None, labels=[0, 1, 2], zero_division=0).mean()),
    }


def binary_event_metrics(gt3, pred):
    binary_gt = ((gt3 == 0) | (gt3 == 1)).astype(int)
    binary_pred = (pred != 2).astype(int)
    cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    ep = tp / max(tp + fp, 1); er = tp / max(tp + fn, 1)
    return {
        "event_f1": 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0,
        "event_precision": float(ep),
        "event_recall": float(er),
        "event_fp": fp, "event_fn": fn,
    }


def flatten_pv(pv):
    gts, raws = [], []
    for d in pv.values():
        gts.append(d["ternary_gt"]); raws.append(d["bridge_pred"])
    return np.concatenate(gts), np.concatenate(raws)


def get_pv(tag, ckpt, npz):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{tag}.pt")
    if os.path.exists(cache):
        c = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[{tag}] cached pv from {cache} (n={len(c['pv'])})", flush=True)
        return c["pv"]
    pv = load_and_infer(ckpt, npz, "DATASET-omnifall/splits/syn/random", use_diff=True,
                        T=64, stride=8, bs=32)
    torch.save({"pv": pv}, cache)
    print(f"[{tag}] inferred & cached -> {cache} (n={len(pv)})", flush=True)
    return pv


def run_grid(pv):
    gt3, bp0 = flatten_pv(pv)
    raw = {"ternary": ternary_metrics(gt3, bp0), "binary_event": binary_event_metrics(gt3, bp0)}
    rows = []
    for K2 in K2_GRID:
        for W in W_GRID:
            corr, stats = apply_rulev2(pv, K1=K1, K2=K2, W=W)
            cps = [corr[vi][: len(pv[vi]["ternary_gt"])] for vi in pv]
            cp = np.concatenate(cps)
            rows.append({
                "K1": K1, "K2": K2, "W": W,
                "ternary": ternary_metrics(gt3, cp),
                "binary_event": binary_event_metrics(gt3, cp),
                "rule_stats": stats,
            })
    best = max(rows, key=lambda r: r["binary_event"]["event_f1"])
    return raw, rows, best


def fmt(x):
    return f"{x:.4f}"


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for tag, ckpt, npz in MODELS:
        print(f"\n{'='*66}\n[{tag}] sweep K2 x W\n{'='*66}", flush=True)
        pv = get_pv(tag, ckpt, npz)
        raw, rows, best = run_grid(pv)
        results[tag] = {"raw": raw, "grid": rows, "best": best}

        # grid 表 (event_f1 / fallen_prec)
        print(f"\n  raw  : ev_f1={fmt(raw['binary_event']['event_f1'])} "
              f"prec={fmt(raw['binary_event']['event_precision'])} rec={fmt(raw['binary_event']['event_recall'])}")
        print(f"\n  K2 row = event_f1（* = 本模型 best）")
        print(f"  {'K2':>4s} | " + "   ".join(f"W={w:<3d}" for w in W_GRID))
        for K2 in K2_GRID:
            cells = []
            for W in W_GRID:
                r = next(r for r in rows if r["K2"] == K2 and r["W"] == W)
                mark = "*" if (r["K2"], r["W"]) == (best["K2"], best["W"]) else " "
                cells.append(fmt(r["binary_event"]["event_f1"]) + mark)
            print(f"  K2={K2:<3d} | " + "   ".join(f"{c:<9s}" for c in cells))
        b = best
        print(f"\n  best: K2={b['K2']} W={b['W']}  ev_f1={fmt(b['binary_event']['event_f1'])} "
              f"prec={fmt(b['binary_event']['event_precision'])} rec={fmt(b['binary_event']['event_recall'])} "
              f"| fallen_prec={fmt(b['ternary']['fallen_precision'])} avg_f1={fmt(b['ternary']['avg_f1'])} "
              f"| fp={b['binary_event']['event_fp']} fn={b['binary_event']['event_fn']}", flush=True)

    # 汇总对比
    print("\n" + "=" * 66)
    print("SUMMARY  (test-selected 上界；K1=2 固定)")
    print("=" * 66)
    print(f"{'':<12s} {'ev_f1':>8s} {'ev_prec':>8s} {'ev_rec':>8s} {'fallen_prec':>12s} {'avg_f1':>8s}")
    for tag in results:
        raw = results[tag]["raw"]; b = results[tag]["best"]
        lab_r = f"{tag} raw"
        lab_b = f"{tag} best(K2={b['K2']},W={b['W']})"
        print(f"  {lab_r:<20s} {fmt(raw['binary_event']['event_f1']):>8s} "
              f"{fmt(raw['binary_event']['event_precision']):>8s} {fmt(raw['binary_event']['event_recall']):>8s} "
              f"{fmt(raw['ternary']['fallen_precision']):>12s} {fmt(raw['ternary']['avg_f1']):>8s}")
        print(f"  {lab_b:<20s} {fmt(b['binary_event']['event_f1']):>8s} "
              f"{fmt(b['binary_event']['event_precision']):>8s} {fmt(b['binary_event']['event_recall']):>8s} "
              f"{fmt(b['ternary']['fallen_precision']):>12s} {fmt(b['ternary']['avg_f1']):>8s}")

    out_path = os.path.join(OUT_DIR, "rulev2_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
