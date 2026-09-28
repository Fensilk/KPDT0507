# -*- coding: utf-8 -*-
"""按规划 §7.2 对 W1 各臂机械化判定 —— 判据冻结在代码里，不由人现场解读。

为什么要有这个脚本：§7.2 的判据是**事先登记**的（吸取 Phase 9"判据没定死、
事后只过 1/3"的教训）。把冻结的阈值写进代码，可以避免两件事：
  ① 事后凭印象挑对自己有利的指标；
  ② 手工比对 6+ 个数字时的抄录错误。

对照基准是 **A0**（`logs/phase11/a0_2000/`），阈值由 A0 的实测值现算，
不写死常数——这样若 A0 因故重跑，判据自动跟随。

用法：
  python -u experiments/judge_w1_arm.py logs/phase11/w1_1a
  python -u experiments/judge_w1_arm.py logs/phase11/w1_1a --a0 logs/phase11/a0_2000

判据（规划 §7.2，2000 数据档）：
  强通过   fall_prec  >= A0 + 0.05        （≈2σ，超出 §3.1 的跨尺度噪声地板）
        或 边界 F1(τ=3) >= A0 + 0.05
  硬否决   fall_rec    <  A0 - 0.03       （重演 Phase 8"压召回换精度"）
        或 fallen_prec <  A0 - 0.02       （拆静态轴）
        或 逐视频 avg_f1 < A0 - 0.02      （事件侧退化）
  不确定带 (+0.02 ~ +0.05) → 判"不确定"，不得据此择优
"""

import os
import sys
import json
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 判据余量（与规划 §7.2 一致；改这里等于改判据，需同步改文档）
PASS_MARGIN = 0.05      # 强通过：fall_prec 或 边界 F1 需高出 A0 多少
VETO_REC = 0.03         # 否决：fall_rec 低于 A0 多少
VETO_FALLEN_PREC = 0.02  # 否决：fallen_prec 低于 A0 多少
VETO_EVENT = 0.02       # 否决：逐视频 avg_f1 低于 A0 多少
UNCERTAIN_LO = 0.02     # 不确定带下沿


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ternary_of(d):
    """从 eval_*.json 取 merged_raw 的 ternary 指标（与 A0 基准同口径）。"""
    if not d:
        return None
    m = d.get("merged_raw", {})
    return m.get("ternary", m) if m else None


def collect(out_dir, a0_dir):
    """收集一条臂与 A0 的各项指标。缺失项返回 None 并在报告中显式标注。"""
    a = {}
    a["eval"] = ternary_of(load_json(os.path.join(out_dir, "eval_best_model.json")))
    tl = load_json(os.path.join(out_dir, "timeline_metrics.json"))
    if tl is None:
        # eval_timeline_metrics.py 的输出是 {name: agg}，取第一个非 GT 项
        p = os.path.join(out_dir, "timeline_metrics_a0.json")
        tl = load_json(p)
    a["timeline"] = _pick_timeline(tl)
    bm = load_json(os.path.join(out_dir, "boundary_metrics.json"))
    if bm is None:
        bm = load_json(os.path.join(out_dir, "boundary_metrics_a0.json"))
    a["boundary"] = _pick_boundary(bm)

    b = {}
    b["eval"] = ternary_of(load_json(os.path.join(a0_dir, "eval_best_model.json")))
    b["timeline"] = _pick_timeline(load_json(os.path.join(a0_dir, "timeline_metrics_a0.json")))
    b["boundary"] = _pick_boundary(load_json(os.path.join(a0_dir, "boundary_metrics_a0.json")))
    return a, b


def _pick_timeline(tl):
    if not tl:
        return None
    for k, v in tl.items():
        if k.startswith("GT"):
            continue
        return v
    return None


def _pick_boundary(bm):
    """boundary_metrics_*.json 的结构是 {'aggregate': {'tau1':..,'tau3':..}}。"""
    if not bm:
        return None
    agg = bm.get("aggregate", bm)
    return agg.get("tau3")


def fmt(v, spec=".4f"):
    return "  n/a  " if v is None else format(v, spec)


def scalar(v):
    """timeline 的 per_video_* 是嵌套 dict（{n, mean, median, std}）——取 mean。

    （这个坑值得记：直接拿 dict 去相减会 TypeError，而不是静默出错，算是好事。）
    """
    if isinstance(v, dict):
        return v.get("mean")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="臂的输出目录，如 logs/phase11/w1_1a")
    ap.add_argument("--a0", default=os.path.join(BASE, "logs", "phase11", "a0_2000"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or os.path.basename(args.out_dir.rstrip("/\\"))

    a, b = collect(args.out_dir, args.a0)
    if not b["eval"]:
        print(f"[ERROR] 找不到 A0 基准：{args.a0}/eval_best_model.json"); sys.exit(1)
    if not a["eval"]:
        print(f"[ERROR] 找不到 {args.out_dir}/eval_best_model.json（训练/评测还没跑完？）")
        sys.exit(1)

    print("=" * 78)
    print(f"W1 判定  arm = {tag}   （对照 A0 = {os.path.basename(args.a0)}）")
    print("=" * 78)
    print(f"{'指标':<26}{'A0':>10}{'本臂':>10}{'Δ':>10}   阈值")
    print("-" * 78)

    rows = [
        ("fall_prec", a["eval"], b["eval"], "fall_precision"),
        ("fall_rec", a["eval"], b["eval"], "fall_recall"),
        ("fallen_prec", a["eval"], b["eval"], "fallen_precision"),
        ("fallen_rec", a["eval"], b["eval"], "fallen_recall"),
        ("avg_f1 (merged_raw)", a["eval"], b["eval"], "avg_f1"),
        ("逐视频 avg_f1", a["timeline"], b["timeline"], "per_video_avg_f1"),
        ("边界 F1 (τ=3)", a["boundary"], b["boundary"], "micro_f1"),
    ]
    vals = {}
    for name, av, bv, key in rows:
        x = scalar((av or {}).get(key))
        y = scalar((bv or {}).get(key))
        d = (x - y) if (x is not None and y is not None) else None
        vals[name] = (x, y, d)
        thr = ""
        if name == "fall_prec":
            thr = f"强通过 ≥ {y + PASS_MARGIN:.4f}" if y is not None else ""
        elif name == "fall_rec":
            thr = f"否决 < {y - VETO_REC:.4f}" if y is not None else ""
        elif name == "fallen_prec":
            thr = f"否决 < {y - VETO_FALLEN_PREC:.4f}" if y is not None else ""
        elif name == "逐视频 avg_f1":
            thr = f"否决 < {y - VETO_EVENT:.4f}" if y is not None else ""
        elif name == "边界 F1 (τ=3)":
            thr = f"强通过 ≥ {y + PASS_MARGIN:.4f}" if y is not None else ""
        print(f"{name:<26}{fmt(x):>10}{fmt(y):>10}{fmt(d, '+.4f') if d is not None else '  n/a  ':>10}   {thr}")

    # ---- 判定 ----
    print("\n" + "-" * 78)
    vetoes = []
    _, a0_rec, d_rec = vals["fall_rec"]
    _, a0_fp, d_fp = vals["fallen_prec"]
    _, a0_ev, d_ev = vals["逐视频 avg_f1"]
    if d_rec is not None and d_rec < -VETO_REC:
        vetoes.append(f"fall_rec {d_rec:+.4f} < -{VETO_REC}（压召回换精度）")
    if d_fp is not None and d_fp < -VETO_FALLEN_PREC:
        vetoes.append(f"fallen_prec {d_fp:+.4f} < -{VETO_FALLEN_PREC}（拆静态轴）")
    if d_ev is not None and d_ev < -VETO_EVENT:
        vetoes.append(f"逐视频 avg_f1 {d_ev:+.4f} < -{VETO_EVENT}（事件侧退化）")

    if vetoes:
        print("判定：【硬否决 ❌】")
        for v in vetoes:
            print(f"    - {v}")
        print("  → 无论其它指标多好，本臂判失败（规划 §7.2）")
    else:
        cands = []
        _, _, d_fprec = vals["fall_prec"]
        _, _, d_bnd = vals["边界 F1 (τ=3)"]
        if d_fprec is not None and d_fprec >= PASS_MARGIN:
            cands.append(f"fall_prec {d_fprec:+.4f} ≥ +{PASS_MARGIN}")
        if d_bnd is not None and d_bnd >= PASS_MARGIN:
            cands.append(f"边界 F1 {d_bnd:+.4f} ≥ +{PASS_MARGIN}")
        if cands:
            print("判定：【强通过 ✅】")
            for c in cands:
                print(f"    - {c}")
            print("  → 进入 W2 的候选，仍需 9600 确认")
        else:
            un = [d for d in (d_fprec, d_bnd) if d is not None]
            band = any(UNCERTAIN_LO <= d < PASS_MARGIN for d in un)
            if band:
                print("判定：【不确定带 ⚠】")
                print(f"  → 效应落在 +{UNCERTAIN_LO} ~ +{PASS_MARGIN}（§3.1 噪声地板内），"
                      "**不得据此择优**")
                print("     若多臂都落在这一带，按'机制最可信+实现最简'选 1 臂进 9600，"
                      "并在文档中标注该选择未经 W1 证据支持")
            else:
                print("判定：【未达强通过，未被否决】")
                print("  → 不构成证据；按 §7.2 不得择优")

    print("\n" + "-" * 78)
    print("⚠ 提醒（规划 §3.1 / §7.4）：单点比较不可用——同一臂内跨 ckpt 的极差"
          "（fall_prec 0.119 / fall_rec 0.166）已超过本判据的阈值。")
    print("   本判定只是**止损闸门**；最终结论须看 epoch 轨迹与 9600 确认。")


if __name__ == "__main__":
    main()
