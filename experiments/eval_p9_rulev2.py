"""
Compare baseline (p7d_delta) vs E3-full under RuleV2 (event-level post-processing).

RuleV2 (canonical, Phase 7 最优): K1=2 / K2=7 / W=64 — fallen 触发 + 前向回查
≥K2 连续 fall + 视频级上下文跳过（纯 fallen / 无 fallen 视频不处理）。实现与
experiments/eval_rule_c.py::apply_rulev2 同一份，此处直接 import，保证口径一致。

指标口径与 eval_rule_c.py::evaluate 同一套：
  - 逐视频窗口推理 → 帧级投票 merge 成每帧三元预测
  - ternary: sklearn 每类 P/R/F1（flatten 全 test 帧）
  - binary event: event=fall|fallen 帧（GT），pred event = pred!=2（帧展平二元）

运行两个模型（同评估口径 stride8 / 全量 test），输出 raw vs +RuleV2 两组，
并验证 baseline 复现历史 event_f1≈0.774。

Usage:
    python experiments/eval_p9_rulev2.py
"""
import os, sys, json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 复用 eval_rule_c 的推理 / RuleV2 / 基础指标函数（同一份实现，避免漂移）
from eval_rule_c import load_and_infer, apply_rulev2

CLASS_NAMES_16 = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]

OUT_DIR = "logs/phase9/e3_rulev2"


def ternary_metrics(gt3, pred):
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


def binary_event_metrics(gt3, pred):
    """event=fall|fallen（帧展平二元），pred_event = pred!=2."""
    binary_gt = ((gt3 == 0) | (gt3 == 1)).astype(int)
    binary_pred = (pred != 2).astype(int)
    cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    ep = tp / max(tp + fp, 1); er = tp / max(tp + fn, 1)
    return {
        "event_f1": 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0,
        "event_precision": float(ep),
        "event_recall": float(er),
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
        "binary_cm": cm.tolist(),
    }


def breakdown_event_fp(pv, corr):
    """16 类真值中，哪些帧被判成 event（pred!=2）；raw vs 规则后各 16 类的 event 帧数.

    用于看: lying / lie_down（静态误报源）在 RuleV2 下被抑制掉多少。
    corr 是 dict[vi]→修正后三元序列，与 pv 逐 vi 对齐。
    """
    raw, corrected = {}, {}
    for vi, d in pv.items():
        l16 = d["labels_16"]; bp = d["bridge_pred"]; cp = corr[vi]
        ev_raw = (bp != 2); ev_corr = (cp != 2)
        for c in np.unique(l16):
            m = l16 == c
            raw.setdefault(int(c), {"total": 0, "event": 0})
            corrected.setdefault(int(c), {"total": 0, "event": 0})
            raw[int(c)]["total"] += int(m.sum())
            corrected[int(c)]["total"] += int(m.sum())
            raw[int(c)]["event"] += int((ev_raw & m).sum())
            corrected[int(c)]["event"] += int((ev_corr & m).sum())
    return {"raw": {CLASS_NAMES_16[k]: v for k, v in sorted(raw.items())},
            "corrected": {CLASS_NAMES_16[k]: v for k, v in sorted(corrected.items())}}


def run_one(tag, ckpt, npz, use_diff=True, T=64, stride=8, bs=32):
    print(f"\n{'='*70}\n[{tag}] infer + RuleV2 (K1=2,K2=7,W=64)\n{'='*70}", flush=True)
    pv = load_and_infer(ckpt, npz, "DATASET-omnifall/splits/syn/random", use_diff=use_diff,
                        T=T, stride=stride, bs=bs)
    print(f"[{tag}] videos={len(pv)}", flush=True)
    corr, stats = apply_rulev2(pv)
    print(f"[{tag}] rule_stats: {json.dumps(stats)}", flush=True)

    # flatten
    gts, raws, corrs = [], [], []
    for vi, d in pv.items():
        n = len(d["ternary_gt"])
        gts.append(d["ternary_gt"]); raws.append(d["bridge_pred"][:n]); corrs.append(corr[vi][:n])
    gt3 = np.concatenate(gts); bp = np.concatenate(raws); cp = np.concatenate(corrs)

    bd = breakdown_event_fp(pv, corr)
    return {
        "tag": tag,
        "n_videos": len(pv),
        "rule_stats": stats,
        "raw": {"ternary": ternary_metrics(gt3, bp), "binary_event": binary_event_metrics(gt3, bp)},
        "rulev2": {"ternary": ternary_metrics(gt3, cp), "binary_event": binary_event_metrics(gt3, cp)},
        "event_fp_by_class": bd,
    }


def fmt(x):
    return f"{x:.4f}"


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)
    os.makedirs(OUT_DIR, exist_ok=True)

    models = [
        ("baseline(p7d_delta)", "logs/phase7/p7d_delta/best_model.pt",
         "data/omnifall_dinov2_giant_frame.npz"),
        ("E3-full(e3_stage2)", "logs/phase9/e3_stage2/best_model.pt",
         "data/omnifall_dinov2_finetuned_frame.npz"),
    ]

    results = []
    for tag, ckpt, npz in models:
        results.append(run_one(tag, ckpt, npz))

    # ── 打印对比表 ──
    print("\n" + "=" * 78)
    print("COMPARISON  (全量 test, stride8 同口径, 帧展平)")
    print("=" * 78)

    cols = ["fall_prec", "fall_rec", "fallen_prec", "fallen_rec", "avg_f1", "ternary_acc"]
    print("\n--- Ternary: raw vs +RuleV2 ---")
    print(f"{'model':<16s} " + "  ".join(f"{c:<20s}" for c in ["fall_rec", "fallen_prec", "avg_f1"]))
    for r in results:
        raw = r["raw"]["ternary"]; r2 = r["rulev2"]["ternary"]
        row = "  ".join(f"{fmt(raw[c])} / {fmt(r2[c])}" for c in ["fall_recall", "fallen_precision", "avg_f1"])
        print(f"  {r['tag']:<16s} {row}")

    print("\n--- Binary event (主指标 event_f1) ---")
    print(f"{'model':<24s} {'ev_f1':>10s}{'+R2':>10s} {'ev_prec':>10s}{'+R2':>10s} {'ev_rec':>10s}{'+R2':>10s} | {'R2后 FP帧':>10s} FN帧")
    for r in results:
        b0 = r["raw"]["binary_event"]; b2 = r["rulev2"]["binary_event"]
        print(f"  {r['tag']:<22s} {fmt(b0['event_f1']):>10s}{fmt(b2['event_f1']):>10s} "
              f"{fmt(b0['event_precision']):>10s}{fmt(b2['event_precision']):>10s} "
              f"{fmt(b0['event_recall']):>10s}{fmt(b2['event_recall']):>10s} | "
              f"{b2['event_fp']:>10,d} {b2['event_fn']:,d}")

    print("\n--- RuleV2 对 lying/lie_down 帧误报抑制 (event 帧数: raw → +R2) ---")
    for r in results:
        bd = r["event_fp_by_class"]
        parts = []
        for cls in ["lying", "lie_down"]:
            if cls in bd["raw"]:
                a = bd["raw"][cls]["event"]; b = bd["corrected"][cls]["event"]
                tot = bd["raw"][cls]["total"]
                parts.append(f"{cls}: {a:,d}({a/tot*100:.1f}%)→{b:,d}({b/tot*100:.1f}%)")
        print(f"  {r['tag']:<22s} " + "  ".join(parts))

    # ── 规则统计对比 ──
    print("\n--- RuleV2 stats ---")
    print(f"{'model':<24s} {'videos_proc':>12s} {'kept':>8s} {'suppressed':>10s} {'events_trig':>11s}")
    for r in results:
        s = r["rule_stats"]
        print(f"  {r['tag']:<22s} {s['videos_processed']:>12,d} {s['fallen_kept']:>8,d} "
              f"{s['fallen_suppressed']:>10,d} {s['events']:>11,d}")

    # ── 存 json ──
    out = {"comparison": results}
    out_path = os.path.join(OUT_DIR, "rulev2_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
