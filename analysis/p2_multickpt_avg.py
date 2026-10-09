# -*- coding: utf-8 -*-
"""P2 的**多 ckpt 平均前沿**读数。

⚠⚠ 这不是预登记 §2.4 ⚠⚠
------------------------
§2.4 冻结的是「每臂 **48 个**密集 ckpt 的**全部**前沿平均」。本次**无法交付**：
48 个 ckpt 那一池（`p2_val_probe/<arm>.pv.pt`，val）**只存 argmax 标签、没有逐帧概率**
（元素键 = `frames/p/rel/y`），而"前沿"必须扫阈值 → 做不出来；有概率的那一池
（stage② 的 test `pv_*.pt`）**每臂只有 5 个候选**，而 §2.4 明文"不得事后缩小到
对自己有利的子集"。

本脚本因此是**降级后的补充读数**：对**每臂 5 个 test 候选**（选中者 + epoch_1/2/3 + 次优）
把各自的前沿插值到公共召回网格上取均值。它
  · **不是** §2.4，不满足"48 个"；
  · **不提供** §2.4 承诺的降方差（1/√48 ≈ 7×）——只有 1/√5 ≈ 2.2×；
  · 只能当**描述性补充**，不得当作主判据。

用法（项目根目录）：
    KMP_DUPLICATE_LIB_OK=TRUE python -u analysis/p2_multickpt_avg.py
"""
import os
import sys
import json

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "analysis"))
import prec_rec_curves as mod          # 判据实现**不重写**，用冻结的那份

PR80 = 0.80
MARGIN = 0.05
DOM_LO, DOM_HI = 0.50, 0.95
AXES = {0: "fall", 1: "fallen"}
ARMS = {"anchor": "logs/phase11/p2_anchor", "1a": "logs/phase11/p2_1a", "4": "logs/phase11/p2_4"}
GRID_N = 200


def ckpts_of(arm):
    """该臂 stage② 的 5 个候选（顺序与离线脚本一致）。"""
    sel = json.load(open(os.path.join(BASE, "logs", "phase11", "p2_val_probe", "selection.json"),
                         encoding="utf-8"))
    cands = [sel[arm], "epoch_1", "epoch_2", "epoch_3", sel[arm + "_runner_up"]]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def curves(run_dir, ck, cls):
    _, pv = mod.load_pv("%s=%s" % (ck, os.path.join(BASE, run_dir, "pv_%s.pt" % ck)))
    return mod.curves_for_axis(pv, cls, "voted")


def avg_frontier(cs):
    """把 k 条前沿插值到公共召回网格上取均值；网格取**所有曲线支撑的交集**。"""
    lo = max(c["recall"][0] for c in cs)
    hi = min(c["recall"][-1] for c in cs)
    if hi <= lo:
        return None
    g = np.linspace(lo, hi, GRID_N)
    P = np.vstack([np.interp(g, c["recall"], c["precision"]) for c in cs])
    return {"recall": g, "precision": P.mean(axis=0), "n": len(cs)}


def pat(rec, prec, target=PR80):
    return None if target > rec[-1] else float(np.interp(target, rec, prec))


def dom(a, b):
    lo = max(a["recall"][0], b["recall"][0], DOM_LO)
    hi = min(a["recall"][-1], b["recall"][-1], DOM_HI)
    if hi <= lo:
        return None
    g = np.linspace(lo, hi, 40)
    d = np.interp(g, a["recall"], a["precision"]) - np.interp(g, b["recall"], b["precision"])
    return {"mean": float(d.mean()),
            "a_dom": bool((d >= -1e-9).all() and (d > 1e-9).any()),
            "b_dom": bool((d <= 1e-9).all() and (d < -1e-9).any())}


def verdict(res):
    vetoes = [ax for ax, r in res.items() if r["d"] is not None and r["d"] < -MARGIN]
    passes = [ax for ax, r in res.items() if r["dm"] and r["dm"]["a_dom"]]
    if all(r["dm"] and r["dm"]["b_dom"] for r in res.values()):
        vetoes.insert(0, "both")
    if vetoes:
        return "硬否决 ❌"
    if passes:
        return "强通过 ✅" if all(not (r["dm"] and r["dm"]["b_dom"]) for r in res.values()) else "不确定带 ⚠"
    ds = [r["d"] for r in res.values() if r["d"] is not None]
    return "不确定带 ⚠" if ds and all(d >= 0 for d in ds) else "未达强通过，未被否决"


def main():
    print("=" * 92)
    print("多 ckpt 平均前沿 —— **降级补充读数（5 个 ckpt，非 §2.4 的 48 个）**")
    print("=" * 92)

    data = {}
    for arm, d in ARMS.items():
        cks = ckpts_of(arm)
        data[arm] = {"cks": cks, "cur": {}}
        for cls, ax in AXES.items():
            cs = [curves(d, c, cls) for c in cks]
            data[arm]["cur"][ax] = {"cs": cs, "avg": avg_frontier(cs)}
        print("  %-7s 候选 %d 个: %s" % (arm, len(cks), " ".join(cks)))

    print("\n【1】平均前沿的 P@R=0.80（对照参照 = anchor）")
    for arm in ("1a", "4"):
        res = {}
        for cls, ax in AXES.items():
            a = data[arm]["cur"][ax]["avg"]
            b = data["anchor"]["cur"][ax]["avg"]
            pa, pb = pat(a["recall"], a["precision"]), pat(b["recall"], b["precision"])
            res[ax] = {"d": (pa - pb) if (pa is not None and pb is not None) else None,
                       "dm": dom(a, b), "pa": pa, "pb": pb}
            print("   %-3s %-7s 平均前沿 P@R: 本臂 %.4f  anchor %.4f  Δ %+.4f   支配均值 %+.4f (a支配=%s)"
                  % (arm, ax, pa, pb, res[ax]["d"], res[ax]["dm"]["mean"], res[ax]["dm"]["a_dom"]))
        print("       → 降级读数判定：【%s】\n" % verdict(res))

    print("【2】同一臂内 5 个 ckpt 的**单点摆幅**（落点噪声的直接度量）")
    print("     %-7s %-7s %s" % ("臂", "轴", "5 个 ckpt 的 P@R（越小越靠前=选中者）"))
    for arm in ("anchor", "1a", "4"):
        for cls, ax in AXES.items():
            vals = [pat(c["recall"], c["precision"]) for c in data[arm]["cur"][ax]["cs"]]
            v = [x for x in vals if x is not None]
            print("     %-7s %-7s [%s]   摆幅 %.4f（%.4f~%.4f）"
                  % (arm, ax, " ".join("%.4f" % x if x is not None else " n/a " for x in vals),
                     max(v) - min(v), min(v), max(v)))

    print("\n⚠ 本读数：k=5 → 降方差仅 1/√5≈2.2×（§2.4 承诺的是 1/√48≈6.9×）；")
    print("   且候选集含 epoch_1/2/3，它们是**固定网格**（跨臂可比），不是随机子集。")
    print("   → 只能作描述性补充，**不得充当主判据**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
