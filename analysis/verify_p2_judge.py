# -*- coding: utf-8 -*-
"""P2 判读的**独立复算** —— 证明 `p2_judge.out` 里的数字不是工具自身的产物。

为什么需要它
------------
`p2_judge.py` 在 2026-10-09 首次对着真数据跑，一次爆出三个 bug（一个崩溃 +
两个静默读错字段）。修好之后只能说"改对了"，不能说"**验过了**"——因为
`p2_judge` 与它调用的 `judge_w1_arm`/`prec_rec_curves` 是同一批作者、同一批假设，
自己验自己证明不了什么。

本脚本的做法
------------
**绕开 `p2_judge.py`、`judge_w1_arm.py`、`prec_rec_curves.py` 三者**：

* 曲线：不 import 上面任何模块，用**自己的 numpy 实现**从 pv 逐帧概率重扫阈值
  （排序 → 累计 TP → prec/rec 包络 → 在 R=0.80 插值），并**额外**用"直接阈值搜索"
  （在 rec≥0.80 的阈值里取最大 prec）做**语义交叉验证**——这一路连"包络+插值"
  这个算法本身都换掉了，能抓到算法级错误。
* §7.3：直接从 `eval_*.json` 与 `timeline_*.json` 读原始字段求五项。
* R1：直接从两份 `monitor.json` 重算 val 曲线差。
* 判定：按 `judge_w1_arm.py` **源码里写明**的规则（支配/否决/不确定带）独立重推。

最后把 `logs/phase11/p2_judge/p2_judge.out` 里打印的数字解析出来逐项对照。

用法（项目根目录）：
    KMP_DUPLICATE_LIB_OK=TRUE python -u analysis/verify_p2_judge.py
"""
import os
import re
import sys
import json

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(BASE, *a)

PR80 = 0.80
MARGIN = 0.05          # judge_w1_arm --margin 默认值
DOM_LO, DOM_HI, DOM_N = 0.50, 0.95, 40     # 与 prec_rec_curves 一致（规则的一部分，非实现）
AXES = {0: "fall", 1: "fallen"}

ARMS = {"anchor": "logs/phase11/p2_anchor",
        "1a": "logs/phase11/p2_1a",
        "4": "logs/phase11/p2_4"}
SEL = {"anchor": "step_13800", "1a": "step_12000", "4": "step_13800"}
TH = {"fall_prec": ("gt", 0.6951), "fall_rec": ("ge", 0.8205),
      "timeline_fall_f1": ("gt", 0.7827), "fallen_prec": ("ge", 0.7351),
      "timeline_avg_f1": ("ge", 0.7213)}

OK, BAD = [], []


def chk(name, mine, theirs, tol=5e-4):
    """对照一项；theirs 为 p2_judge.out 解析值。"""
    if theirs is None:
        OK.append("%-44s 我的=%s  (p2_judge 未解析到，跳过对照)" % (name, mine))
        return
    good = (mine is not None and abs(mine - theirs) <= tol)
    (OK if good else BAD).append(
        "%-44s 我的=%.4f  p2_judge=%.4f  %s" % (name, mine, theirs, "✅" if good else "❌差 %.4f" % abs(mine - theirs)))


# ───────────────────────── 独立曲线实现 ─────────────────────────
def load_raw_pv(path):
    d = torch.load(path, weights_only=False, map_location="cpu")
    return d["pv"] if "pv" in d else d


def my_curve(pv, cls):
    """自己的包络实现：排序扫阈值 → prec/rec（recall 单调不减，同 recall 取最大 prec）。"""
    s = np.concatenate([np.asarray(v["probs"])[:, cls] for v in pv.values()])
    g = np.concatenate([(np.asarray(v["ternary_gt"]) == cls).astype(np.int64) for v in pv.values()])
    npos = int(g.sum())
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(g[o])
    k = np.arange(1, len(g) + 1)
    prec, rec = tp / k, tp / npos
    keep = np.concatenate([[True], np.diff(rec) > 0])
    return rec[keep], prec[keep], npos, len(g), s, g


def my_p_at_r(rec, prec, target=PR80):
    if target > rec[-1]:
        return None
    return float(np.interp(target, rec, prec))


def my_p_at_r_direct(s, g, target=PR80):
    """**语义交叉验证**：直接扫阈值——在 rec>=target 的阈值里取最大 prec。

    与"包络+插值"是两条不同的算法路径；两者在 R=0.80 处应彼此吻合
    （插值是在离散包络上取样，直接搜索取得是离 R=0.80 最近的可达阈值）。
    """
    npos = int(g.sum())
    o = np.argsort(-s, kind="stable")
    gs = g[o]
    tp = np.cumsum(gs)
    k = np.arange(1, len(g) + 1)
    rec = tp / npos
    ok = rec >= target
    if not ok.any():
        return None
    prec = (tp / k)[ok]
    return float(prec.max())


def my_dominance(ra, pa, rb, pb):
    lo = max(ra[0], rb[0], DOM_LO)
    hi = min(ra[-1], rb[-1], DOM_HI)
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, DOM_N)
    d = np.interp(grid, ra, pa) - np.interp(grid, rb, pb)
    return {"mean": float(d.mean()),
            "a_dom": bool((d >= -1e-9).all() and (d > 1e-9).any()),
            "b_dom": bool((d <= 1e-9).all() and (d < -1e-9).any()),
            "identical": bool(np.abs(d).max() <= 1e-9)}


def my_verdict(res):
    """按 judge_w1_arm.py 源码写明的规则独立重推。"""
    vetoes, passes = [], []
    for ax, r in res.items():
        if r["dm"] is not None and r["dm"]["a_dom"]:
            passes.append(ax)
        if r["d"] is not None and r["d"] < -MARGIN:
            vetoes.append(ax)
    both_dom = all(r["dm"] is not None and r["dm"]["b_dom"] for r in res.values())
    if both_dom:
        vetoes.insert(0, "both")
    all_id = all(r["dm"] is not None and r["dm"]["identical"] for r in res.values())
    if all_id:
        return "臂=基准"
    if vetoes:
        return "硬否决 ❌"
    if passes:
        undom = all(not (r["dm"] is not None and r["dm"]["b_dom"]) for r in res.values())
        return "强通过 ✅" if undom else "不确定带 ⚠"
    ds = [r["d"] for r in res.values() if r["d"] is not None]
    if ds and all(d >= 0 for d in ds):
        return "不确定带 ⚠"
    return "未达强通过，未被否决"


# ───────────────────────── 解析 p2_judge.out ─────────────────────────
def parse_out():
    p = P("logs", "phase11", "p2_judge", "p2_judge.out")
    txt = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
    curve, sev = {}, {}
    for m in re.finditer(r"^  ── (\S+): 判定：【([^】]+)】", txt, re.M):
        curve.setdefault("verdict", {})[m.group(1)] = m.group(2).strip()
    # 曲线表行：  fall        0.7187     0.7455   +0.0268   ...
    for m in re.finditer(r"^     (fall|fallen)\s+([\d.]+|n/a)\s+([\d.]+|n/a)\s+([+-]?[\d.]+|n/a)",
                         txt, re.M):
        curve.setdefault("rows", []).append((m.group(1), m.group(2), m.group(3)))
    # §7.3： 臂块 + 五项实测
    cur = None
    for ln in txt.splitlines():
        m = re.match(r"^  ── (anchor|1a|4)（选中 \S+）：\*\*(\d)/5\*\*", ln)
        if m:
            cur = m.group(1)
            sev.setdefault(cur, {"n": int(m.group(2)), "vals": {}})
            continue
        m = re.match(r"^    (\w+)\s+阈值 \S+ [\d.]+\s+实测 ([\d.]+|缺)", ln)
        if m and cur:
            sev[cur]["vals"][m.group(1)] = None if m.group(2) == "缺" else float(m.group(2))
    # R1 val 曲线
    r1 = {}
    m = re.search(r"step 9600 \(ep1 末\)\s+新 ([\d.]+)\s+旧 ([\d.]+)", txt)
    if m:
        r1["ep1"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"step 19200 \(ep2 末\)\s+新 ([\d.]+)\s+旧 ([\d.]+)", txt)
    if m:
        r1["ep2"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"step 28800 \(ep3 末\)\s+新 ([\d.]+)\s+旧 ([\d.]+)", txt)
    if m:
        r1["ep3"] = (float(m.group(1)), float(m.group(2)))
    return curve, sev, r1


def main():
    curve, sev, r1 = parse_out()
    print("=" * 100)
    print("P2 判读 · 独立复算（绕开 p2_judge / judge_w1_arm / prec_rec_curves）")
    print("=" * 100)

    # ═══ A. 曲线口径 ═══
    print("\n【A】曲线口径 —— 自写 numpy 实现 vs p2_judge.out")
    nav = load_raw_pv(P(ARMS["anchor"], "pv_best.pt"))
    anchor_cur = {cls: my_curve(nav, cls) for cls in AXES}
    rows = curve.get("rows", [])
    ri = 0
    for arm in ("1a", "4"):
        av = load_raw_pv(P(ARMS[arm], "pv_best.pt"))
        res = {}
        for cls, ax in AXES.items():
            ra, pa, npos_a, nfr, sa, ga = my_curve(av, cls)
            rb, pb, npos_b, _, sb, gb = my_curve(nav, cls)
            p_arm, p_a0 = my_p_at_r(ra, pa), my_p_at_r(rb, pb)
            res[ax] = {"d": (p_arm - p_a0) if (p_arm is not None and p_a0 is not None) else None,
                       "dm": my_dominance(ra, pa, rb, pb),
                       "p_direct": my_p_at_r_direct(sa, ga)}
            if ri < len(rows):          # 对照 p2_judge 打印的 (A0 P@R, 本臂 P@R)
                _, t_a0, t_arm = rows[ri]
                chk("  %s/%s  P@R 本臂" % (arm, ax), p_arm, float(t_arm) if t_arm != "n/a" else None)
                chk("  %s/%s  P@R A0" % (arm, ax), p_a0, float(t_a0) if t_a0 != "n/a" else None)
                ri += 1
            print("    %-3s %-7s 直接阈值搜索 P@R=%.4f vs 包络插值 %.4f  (差 %.5f)"
                  % (arm, ax, res[ax]["p_direct"] or float("nan"), p_arm or float("nan"),
                     abs((res[ax]["p_direct"] or 0) - (p_arm or 0))))
        v = my_verdict(res)
        t = curve.get("verdict", {}).get(arm)
        (OK if (t and t.replace(" ", "") in v.replace(" ", "")) else BAD).append(
            "  %s 判定：我=%s  p2_judge=%s" % (arm, v, t))
        for ax in AXES.values():
            if res[ax]["dm"]:
                print("    %s/%s 支配均值=%+.4f  a支配=%s b支配=%s"
                      % (arm, ax, res[ax]["dm"]["mean"], res[ax]["dm"]["a_dom"], res[ax]["dm"]["b_dom"]))

    # ═══ B. §7.3 ═══
    print("\n【B】§7.3 五项 —— 直接读原始 JSON vs p2_judge.out")
    for arm in ("anchor", "1a", "4"):
        d = ARMS[arm]
        ck = SEL[arm]
        ev = json.load(open(P(d, "eval_%s.json" % ck), encoding="utf-8"))
        tl = json.load(open(P(d, "timeline_%s.json" % ck), encoding="utf-8"))
        inner = list(tl.values())[0]
        ins = ev.get("instance") or {}
        got = {"fall_prec": ins.get("fall_precision"), "fall_rec": ins.get("fall_recall"),
               "fallen_prec": ins.get("fallen_precision"),
               "timeline_fall_f1": (inner.get("per_video_fall_f1") or {}).get("mean"),
               "timeline_avg_f1": (inner.get("per_video_avg_f1") or {}).get("mean")}
        npass = 0
        for k, (op, th) in TH.items():
            v = got[k]
            ok = (v > th) if op == "gt" else (v >= th)
            npass += ok
            chk("  %s/%s" % (arm, k), v, sev.get(arm, {}).get("vals", {}).get(k))
        tn = sev.get(arm, {}).get("n")
        (OK if npass == tn else BAD).append("  %s 通过数：我=%d/5  p2_judge=%s/5" % (arm, npass, tn))

    # ═══ C. R1 val 曲线 ═══
    print("\n【C】R1 val 曲线 —— 直接读 monitor.json vs p2_judge.out")
    def mc(d):
        m = json.load(open(P(d, "monitor.json"), encoding="utf-8"))
        return {x["step"]: x for x in m}
    ca, co = mc(ARMS["anchor"]), mc("logs/phase9/e1_full")
    for tag, s in (("ep1", 9600), ("ep2", 19200), ("ep3", 28800)):
        if s in ca and s in co and tag in r1:
            mine = ca[s]["val_avg_f1"] - co[s]["val_avg_f1"]
            chk("  R1 %s Δ(新−旧) @step%d" % (tag, s), mine, r1[tag][0] - r1[tag][1])
            chk("  R1 %s 新值" % tag, ca[s]["val_avg_f1"], r1[tag][0])
            chk("  R1 %s 旧值" % tag, co[s]["val_avg_f1"], r1[tag][1])

    # ═══ 汇总 ═══
    print("\n" + "=" * 100)
    for l in OK:
        print("  " + l)
    for l in BAD:
        print("  " + l)
    print("=" * 100)
    print("  对照通过 %d 项，不一致 %d 项" % (len(OK), len(BAD)))
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
