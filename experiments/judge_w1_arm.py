# -*- coding: utf-8 -*-
"""按规划 §7.2（**2026-09-30 勘误版**）对 W1 各臂机械化判定 —— 判据冻结在代码里。

为什么要有这个脚本：§7.2 的判据是**事先登记**的（吸取 Phase 9"判据没定死、
事后只过 1/3"的教训）。把冻结的判据写进代码，可以避免两件事：
  ① 事后凭印象挑对自己有利的指标；
  ② 手工比对 6+ 个数字时的抄录错误。

═══ 2026-09-30 判据变更（规划 §7.2 勘误，见附录 A）═══

原判据用**单点** `fall_prec` / `fallen_prec` 比大小。**该做法已被实测证伪**：

  同一臂内跨 ckpt 的摆动（A0：fall_prec 0.119 / fall_rec 0.166 / fallen_prec 0.179）
  **全都超过原判据的阈值 0.05**；且这些 ckpt 的 (prec, rec) 几乎完全落在一条陡峭
  前沿上（相关系数 r = −0.955 ~ −0.988），avg_f1 极差仅 0.035。
  → ckpt 之间的差异**主要是沿前沿滑动（落点），不是能力差异**。
  → "拿 A 的落点比 B 的落点"等于比较两次抽签；单点法据此把 Exp-1a 误判为"硬否决"。

新判据改用**前沿曲线口径**（对落点不变，测的是前沿本身）。

═══ 现判据（voted 协议，全量 test 1200，逐帧）═══

  主判据（fall / fallen **两轴并列**，取本臂与 A0 的 prec-rec 曲线）：
    ① **支配关系**：在共同召回区间上逐点比，本臂处处不低于 A0 → 本臂支配 A0
    ② **P@R=0.80**：匹配召回 0.80 下的精度

  强通过   任一轴**支配** A0，且另一轴**未被** A0 支配
  不确定带 两轴均未达支配，但两轴 P@R=0.80 均 ≥ A0（有优势但非处处不低）
  硬否决   ① 两轴**均被** A0 支配（即在任何工作点都不优于 A0）
           ② 任一轴 P@R=0.80 < A0 − MARGIN（"低出"，MARGIN 默认 0.05，与强通过余量对称）

  支配关系的计算**不在这里重写**，而是复用 `analysis/prec_rec_curves.py` 的
  `dominance()` —— 判据与画图必须同源，否则两处会漂移。

  辅助读数（**不作判定**，仅供人判读，保留历史可比性）：
  §7.2 原有的单点表（fall_prec/fall_rec/fallen_prec/fallen_rec/avg_f1/逐视频 avg_f1/边界 F1）。

用法：
  python -u experiments/judge_w1_arm.py logs/phase11/w1_1a
  python -u experiments/judge_w1_arm.py logs/phase11/w1_1a --a0 logs/phase11/a0_2000
  可选：--protocol voted|flat   --pr80 0.80   --margin 0.05
"""

import os
import sys
import json
import argparse
import importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# ── 判据常数（改这里等于改判据，需同步改文档）──
PR80 = 0.80          # 主判据的召回水平：P@R=0.80
MARGIN = 0.05        # 硬否决②的"低出"余量；与原 §7.2 强通过余量对称

# ── 辅助读数用的旧阈值（仅用于在单点表上标注，不再触发判定）──
PASS_MARGIN_OLD = 0.05
VETO_REC_OLD = 0.03
VETO_FALLEN_PREC_OLD = 0.02
VETO_EVENT_OLD = 0.02

AXES = {"fall": 0, "fallen": 1}


def _load_curves_module():
    """载入 analysis/prec_rec_curves.py（它是脚本不是包，故用 importlib 按路径载入）。

    为什么不复制一份支配逻辑：判据与画图必须同源——本项目已有"副本间漂移"的教训。
    """
    path = os.path.join(BASE, "analysis", "prec_rec_curves.py")
    spec = importlib.util.spec_from_file_location("prec_rec_curves", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def collect(out_dir, a0_dir):
    """收集一条臂与 A0 的**单点**指标（辅助读数用）。缺失项返回 None。"""
    def _side(d, tl_names, bm_names):
        s = {}
        s["eval"] = ternary_of(load_json(os.path.join(d, "eval_best_model.json")))
        tl = None
        for n in tl_names:
            tl = load_json(os.path.join(d, n))
            if tl is not None:
                break
        s["timeline"] = _pick_timeline(tl)
        bm = None
        for n in bm_names:
            bm = load_json(os.path.join(d, n))
            if bm is not None:
                break
        s["boundary"] = _pick_boundary(bm)
        return s

    a = _side(out_dir, ["timeline_metrics.json", "timeline_metrics_a0.json"],
              ["boundary_metrics.json", "boundary_metrics_a0.json"])
    b = _side(a0_dir, ["timeline_metrics_a0.json"], ["boundary_metrics_a0.json"])
    return a, b


def scalar(v):
    """timeline 的 per_video_* 是嵌套 dict（{n, mean, median, std}）——取 mean。

    （这个坑值得记：直接拿 dict 去相减会 TypeError，而不是静默出错，算是好事。）
    """
    if isinstance(v, dict):
        return v.get("mean")
    return v


def fmt(v, spec=".4f"):
    return "  n/a  " if v is None else format(v, spec)


def find_pv(d):
    """优先用带逐帧概率的 pv（--dump_pv 版），否则返回 None 并由调用方报错。"""
    for n in ("pv_best_probs.pt", "pv_best.pt"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p, n
    return None, None


def curve_side(pv_path, name, cls, protocol, mod):
    _, pv = mod.load_pv(f"{name}={pv_path}")
    return mod.curves_for_axis(pv, cls, protocol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="臂的输出目录，如 logs/phase11/w1_1a")
    ap.add_argument("--a0", default=os.path.join(BASE, "logs", "phase11", "a0_2000"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--protocol", default="voted", choices=["voted", "flat"])
    ap.add_argument("--pr80", type=float, default=PR80)
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()
    tag = args.tag or os.path.basename(args.out_dir.rstrip("/\\"))
    # 基准名只在这里算一次。**不可写成 f-string 内联表达式**：
    # `f"{os.path.basename(p.rstrip('/\\'))}"` 里的反斜杠在 Python 3.12 之前是语法错误
    # （PEP 701 才放开）——本地是 3.13 能跑，远端 py310 直接 SyntaxError。
    # 本项目目标环境是 py310（CLAUDE.md），故一律先算好再用。
    bl = os.path.basename(args.a0.rstrip("/\\"))

    mod = _load_curves_module()

    pv_arm, pv_arm_name = find_pv(args.out_dir)
    pv_a0, pv_a0_name = find_pv(args.a0)
    if pv_arm is None:
        print(f"[ERROR] 找不到 {args.out_dir}/pv_best_probs.pt（需 eval 时 --dump_pv）")
        sys.exit(1)
    if pv_a0 is None:
        print(f"[ERROR] 找不到 A0 的 pv：{args.a0}")
        sys.exit(1)
    # ⚠ 必须**按内容**判断 pv 是否带逐帧概率，**不能按文件名**：
    #   新版 eval_p9e1_dual.py 已把 probs 直接写进 pv_best.pt，
    #   按文件名判断会误报（2026-10-01 对 1b1 就这样误报过一次，
    #   险些让人以为算出来的曲线不可信）。
    key_s = "probs" if args.protocol == "voted" else "flat_probs"
    _, _probe = mod.load_pv(f"probe={pv_arm}")
    if not _probe or key_s not in next(iter(_probe.values())):
        print(f"[ERROR] {os.path.basename(pv_arm)} 缺少逐帧概率键 '{key_s}'，"
              f"无法扫阈值——请用 eval_p9e1_dual.py --dump_pv 重跑评测。")
        sys.exit(1)

    print("=" * 86)
    print(f"W1 判定（§7.2 勘误版 · 曲线口径）  arm = {tag}   （对照 A0 = "
          f"{os.path.basename(args.a0)}）")
    print(f"  协议 = {args.protocol}   主判据 = P@R={args.pr80:.2f} + 支配关系   "
          f"否决余量 = {args.margin}")
    print("=" * 86)

    # ── 主判据：逐轴曲线 ──
    res = {}
    for ax, cls in AXES.items():
        try:
            ca = curve_side(pv_arm, tag, cls, args.protocol, mod)
            cb = curve_side(pv_a0, "A0", cls, args.protocol, mod)
        except KeyError as e:
            print(f"[ERROR] {ax} 轴：{e}")
            sys.exit(1)
        dm = mod.dominance(ca, cb)          # d = 本臂 − A0
        p_arm = mod.prec_at_recall(ca, args.pr80)
        p_a0 = mod.prec_at_recall(cb, args.pr80)
        res[ax] = {"dm": dm, "arm": p_arm, "a0": p_a0,
                   "d": (p_arm - p_a0) if (p_arm is not None and p_a0 is not None) else None,
                   "curve_arm": ca, "curve_a0": cb}

    # ⚠ 基准列名必须**取自 --a0 实参**（bl，见上），不能写死 "A0"：负对照时 --a0 会指向别的臂，
    #   写死会让读表人以为基准仍是 A0（数值与 Δ 都对，只有列名骗人——最难被发现的一种错）。
    print(f"\n【主判据】逐轴 P@R 与支配关系（全量 test 1200，{args.protocol} 协议）")
    print(f"  {'轴':<8}{bl + ' P@R':>12}{'本臂 P@R':>11}{'Δ':>10}   支配关系")
    print("  " + "-" * 82)
    for ax, r in res.items():
        dm = r["dm"]
        if dm is None:
            rel = "共同召回区间过窄，无法比"
        elif dm.get("identical"):
            rel = "曲线逐点全等（无可判定差异）"
        else:
            # ⚠ 关系文案**复用 prec_rec_curves.dominance_line()**，绝不在这里重写一份：
            #   本块曾经自己实现过一遍（只复用计算、没复用呈现），结果那边修好了
            #   "两端同向 ≠ 交叉"，这边仍印旧的"交叉（两端都是本臂更好）"——
            #   同一判定在两处说不同的话，正是"副本漂移"。
            #   把本臂命名为"本臂"传入，即可与基准名 bl 区分。
            rel = mod.dominance_line("本臂", bl, dm)
        print(f"  {ax:<8}{fmt(r['a0']):>10}{fmt(r['arm']):>11}"
              f"{fmt(r['d'], '+.4f') if r['d'] is not None else '  n/a  ':>10}   {rel}")

    # ── 判定 ──
    print("\n" + "-" * 86)
    vetoes, passes = [], []
    for ax, r in res.items():
        dm = r["dm"]
        if dm is not None and dm["a_dominates_b"]:
            passes.append(f"{ax} 轴：本臂支配 A0（均值 {dm['mean']:+.4f}）")
        if r["d"] is not None and r["d"] < -args.margin:
            vetoes.append(f"{ax} 轴：P@R={args.pr80:.2f} 低出 {r['d']:+.4f} < -{args.margin}")
    both_dominated = all((r["dm"] is not None and r["dm"]["b_dominates_a"])
                         for r in res.values())
    if both_dominated:
        vetoes.insert(0, f"两轴均被 {bl} 支配（在任何工作点都不优于基准）")

    # 退化情形：臂与 A0 的曲线逐点全等（"臂=基准"自测即走这条）。
    # **必须是独立分支**，不能只是清空 vetoes/passes —— 那样会继续往下掉进
    # "不确定带"，而那里印的"有优势而非处处不低"对全等曲线是**假话**。
    all_identical = all((r["dm"] is not None and r["dm"].get("identical"))
                        for r in res.values())

    if all_identical:
        print("判定：【臂 = 基准，无可判定差异】（两轴曲线逐点全等）")
        print("  → 自测应得此结果；若真实臂落在这里，说明它与 A0 无差异")
    elif vetoes:
        print("判定：【硬否决 ❌】")
        for v in vetoes:
            print(f"    - {v}")
        print("  → 判失败（§7.2 勘误版）；原'单点 fallen_prec 否决'已废除")
    else:
        if passes:
            undominated = all(not (r["dm"] is not None and r["dm"]["b_dominates_a"])
                              for r in res.values())
            if undominated:
                print("判定：【强通过 ✅】")
                for c in passes:
                    print(f"    - {c}")
                print("  → 进入 W2 的候选，仍需 9600 确认（§7.2 是止损闸门，不是择优器）")
            else:
                print("判定：【不确定带 ⚠】")
                for c in passes:
                    print(f"    - {c}")
                print("  → 有轴支配，但另一轴被 A0 支配；不得据此择优")
        else:
            ds = [r["d"] for r in res.values() if r["d"] is not None]
            if ds and all(d >= 0 for d in ds):
                print("判定：【不确定带 ⚠】")
                print(f"  → 两轴 P@R={args.pr80:.2f} 均不低于 A0，但均未达支配"
                      "（有优势而非处处不低）；**不得据此择优**")
            else:
                print("判定：【未达强通过，未被否决】")
                print("  → 不构成证据；按 §7.2b 不得择优。"
                      "**注意区分「未达」与「否决」**：本臂只是没拿到证据，"
                      "不等于该方向被证伪（若探索变量本身未到位，结论是「实验无效」）。")

    # ── 辅助读数：旧单点表（不作判定）──
    a, b = collect(args.out_dir, args.a0)
    if a["eval"] and b["eval"]:
        print("\n" + "-" * 86)
        print("【辅助读数】旧 §7.2 单点表 —— **不参与判定**，仅保留历史可比性")
        print("⚠ 单点差异里混着'落点'成分，其摆动幅度已实测超过原判据阈值，不可据以判胜负")
        print(f"{'指标':<26}{'A0':>10}{'本臂':>10}{'Δ':>10}   原阈值（已废）")
        print("-" * 86)
        rows = [
            ("fall_prec", a["eval"], b["eval"], "fall_precision"),
            ("fall_rec", a["eval"], b["eval"], "fall_recall"),
            ("fallen_prec", a["eval"], b["eval"], "fallen_precision"),
            ("fallen_rec", a["eval"], b["eval"], "fallen_recall"),
            ("avg_f1 (merged_raw)", a["eval"], b["eval"], "avg_f1"),
            ("逐视频 avg_f1", a["timeline"], b["timeline"], "per_video_avg_f1"),
            ("边界 F1 (τ=3)", a["boundary"], b["boundary"], "micro_f1"),
        ]
        for name, av, bv, key in rows:
            x = scalar((av or {}).get(key))
            y = scalar((bv or {}).get(key))
            d = (x - y) if (x is not None and y is not None) else None
            thr = ""
            if name == "fall_prec":
                thr = f"强通过 ≥ {y + PASS_MARGIN_OLD:.4f}" if y is not None else ""
            elif name == "fall_rec":
                thr = f"否决 < {y - VETO_REC_OLD:.4f}" if y is not None else ""
            elif name == "fallen_prec":
                thr = f"否决 < {y - VETO_FALLEN_PREC_OLD:.4f}" if y is not None else ""
            elif name == "逐视频 avg_f1":
                thr = f"否决 < {y - VETO_EVENT_OLD:.4f}" if y is not None else ""
            elif name == "边界 F1 (τ=3)":
                thr = f"强通过 ≥ {y + PASS_MARGIN_OLD:.4f}" if y is not None else ""
            # ⚠ 列序：表头是「A0 / 本臂」，而 x=本臂、y=A0 —— 必须印 y 再印 x。
            #   曾经印反过：数值与 Δ 都对，但列名对调，读表的人会以为 A0 是另一个数。
            print(f"{name:<26}{fmt(y):>10}{fmt(x):>10}"
                  f"{fmt(d, '+.4f') if d is not None else '  n/a  ':>10}   {thr}")

    print("\n" + "-" * 86)
    print("⚠ 提醒（§3.1 / §7.4）：单点比较不可用——同一臂内跨 ckpt 的极差"
          "（fall_prec 0.119 / fall_rec 0.166）已超过原判据阈值。")
    print("   本判定是**止损闸门**；最终择优在 §7.3 的 9600 上做。")


if __name__ == "__main__":
    main()
