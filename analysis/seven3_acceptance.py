# -*- coding: utf-8 -*-
"""§7.3 五项的**两道验收** —— CLAUDE.md 强制项，此前从未对本读数做过。

CLAUDE.md：「新指标必须过两道验收才可采用：① **GT 自检** —— 真值当预测喂入须得满分
（非事件样本应**排除**而非记 0）；② **负对照** —— 故意注入错误（全漏/平移/抖动）须单调劣化。
两者都要**跑出数**并写进文档。」

背景：§7.3 的五项此前被当作"现成读数"直接使用，从未验收。2026-10-09 首次对着真数据跑时，
它在一个**静默**缺陷上暴露（读错键名 `fall_f1`/`avg_f1`，真名 `per_video_*` 且需取 `.mean`），
两项恒"缺"、通过数上限被压到 3/5。**本脚本就是它当初该做的那两道验收。**

指标实现**不重写**：instance 三项调 `experiments/eval_p9e1_dual.ternary_metrics`，
逐视频两项调 `utils/timeline_metrics` 的 `compute_video_timeline_metrics` +
`aggregate_timeline_metrics`（与 `experiments/eval_timeline_metrics.py` 同一条路径）。

用法（项目根目录）：
    KMP_DUPLICATE_LIB_OK=TRUE python -u analysis/seven3_acceptance.py
"""
import os
import sys
import copy

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "experiments"))

from experiments.eval_p9e1_dual import ternary_metrics                 # noqa: E402
from utils.timeline_metrics import (compute_video_timeline_metrics,    # noqa: E402
                                    aggregate_timeline_metrics)

PV = os.path.join(BASE, "logs", "phase11", "p2_anchor", "pv_step_13800.pt")

# §7.3 冻结阈值（取自已发表 E1-full，**不得重算**）
TH = {"fall_prec": ("gt", 0.6951), "fall_rec": ("ge", 0.8205),
      "timeline_fall_f1": ("gt", 0.7827), "fallen_prec": ("ge", 0.7351),
      "timeline_avg_f1": ("ge", 0.7213)}


def tern_binaries(t):
    t = np.asarray(t)
    return (t == 0).astype(int), (t == 1).astype(int)


def load():
    d = torch.load(PV, weights_only=False, map_location="cpu")
    return d["pv"] if "pv" in d else d


def eval_variant(pv):
    """返回 §7.3 五项实测值（instance 三项 pooled；逐视频两项取 mean）。"""
    y = np.concatenate([np.asarray(v["ternary_gt"]) for v in pv.values()])
    p = np.concatenate([np.asarray(v["bridge_pred"]) for v in pv.values()])
    inst = ternary_metrics(y, p)
    per = []
    for v in pv.values():
        gt, pr = np.asarray(v["ternary_gt"]), np.asarray(v["bridge_pred"])
        gF, gN = tern_binaries(gt)
        pF, pN = tern_binaries(pr)
        per.append(compute_video_timeline_metrics(pF, pN, gF, gN))
    agg = aggregate_timeline_metrics(per)
    return {"fall_prec": inst["fall_precision"], "fall_rec": inst["fall_recall"],
            "fallen_prec": inst["fallen_precision"],
            "timeline_fall_f1": agg["per_video_fall_f1"]["mean"],
            "timeline_avg_f1": agg["per_video_avg_f1"]["mean"]}


def passes(got):
    n, rows = 0, []
    for k, (op, th) in TH.items():
        v = got.get(k)
        ok = v is not None and ((v > th) if op == "gt" else (v >= th))
        n += ok
        rows.append((k, v, ok))
    return n, rows


def build(base, kind, **kw):
    """按 kind 造一个变体 pv（只改 bridge_pred）。"""
    out = {}
    rng = np.random.RandomState(20261009)          # 固定种子，可复现
    for k, v in base.items():
        pv = copy.copy(v)
        gt = np.asarray(v["ternary_gt"])
        if kind == "gt":
            pr = gt.copy()
        elif kind == "real":
            pr = np.asarray(v["bridge_pred"]).copy()
        elif kind == "all_normal":                  # 全漏：一律报 normal
            pr = np.full_like(gt, 2)
        elif kind == "all_fall":                    # 全报：一律报 fall
            pr = np.zeros_like(gt)
        elif kind == "shift":                       # 完美但整体平移 k 帧
            pr = np.roll(gt, kw["k"])
        elif kind == "jitter":                      # 在真值上随机翻转 p 比例
            pr = gt.copy()
            m = rng.rand(len(gt)) < kw["p"]
            pr[m] = rng.randint(0, 3, size=int(m.sum()))
        pv["bridge_pred"] = pr
        out[k] = pv
    return out


def main():
    base = load()
    print("=" * 100)
    print("§7.3 五项的两道验收（CLI 实现不重写；阈值冻结自 E1-full）")
    print("=" * 100)
    print("  数据：%s（%d 视频）\n" % (os.path.relpath(PV, BASE), len(base)))
    print("  %-22s %-9s %-9s %-11s %-11s %-11s %s" %
          ("条件", "fall_prec", "fall_rec", "逐视频f1", "fallen_prec", "逐视频avg", "通过"))

    cases = [("①GT 自检", "gt", {}), ("（真实预测）", "real", {}),
             ("②全漏(全 normal)", "all_normal", {}), ("②全报(全 fall)", "all_fall", {}),
             ("②平移 1 帧", "shift", {"k": 1}), ("②平移 5 帧", "shift", {"k": 5}),
             ("②平移 20 帧", "shift", {"k": 20}),
             ("②抖动 5%", "jitter", {"p": 0.05}), ("②抖动 20%", "jitter", {"p": 0.20}),
             ("②抖动 50%", "jitter", {"p": 0.50})]

    res = {}
    for label, kind, kw in cases:
        got = eval_variant(build(base, kind, **kw))
        n, _ = passes(got)
        res[label] = (got, n)
        print("  %-22s %-9.4f %-9.4f %-11.4f %-11.4f %-11.4f %d/5" %
              (label, got["fall_prec"], got["fall_rec"], got["timeline_fall_f1"],
               got["fallen_prec"], got["timeline_avg_f1"], n))

    print("\n" + "-" * 100)
    bad = []
    # ① GT 自检：须满分
    gt5 = res["①GT 自检"][1]
    print("  ① GT 自检：%d/5 %s" % (gt5, "✅ 满分（真值喂入应得满分）" if gt5 == 5 else "❌ 未满分——指标有缺陷"))
    if gt5 != 5:
        bad.append("GT 自检未满分")

    # ② 负对照：全漏/全报须劣化；平移、抖动须**单调**劣化
    def seq(labels, key):
        return [res[l][0][key] for l in labels]

    for key in TH:
        sh = seq(["②平移 1 帧", "②平移 5 帧", "②平移 20 帧"], key)
        ji = seq(["②抖动 5%", "②抖动 20%", "②抖动 50%"], key)
        if not all(sh[i] >= sh[i + 1] - 1e-9 for i in range(len(sh) - 1)):
            bad.append("平移非单调劣化 [%s]: %s" % (key, ["%.4f" % x for x in sh]))
        if not all(ji[i] >= ji[i + 1] - 1e-9 for i in range(len(ji) - 1)):
            bad.append("抖动非单调劣化 [%s]: %s" % (key, ["%.4f" % x for x in ji]))
    print("  ② 负对照：全漏 %d/5 ・ 全报 %d/5（均应显著劣于真实预测 %d/5）%s"
          % (res["②全漏(全 normal)"][1], res["②全报(全 fall)"][1], res["（真实预测）"][1],
             " ✅" if res["②全漏(全 normal)"][1] <= res["（真实预测）"][1] else " ❌"))
    print("  ② 平移单调性 / 抖动单调性：%s" % ("✅ 全部单调劣化" if not [b for b in bad if "单调" in b] else "❌"))

    print("\n" + "=" * 100)
    if bad:
        print("  验收**未通过**：")
        for b in bad:
            print("    - " + b)
    else:
        print("  验收**通过**：① GT 自检满分；② 负对照全漏/全报显著劣化，平移与抖动均单调劣化。")
    print("=" * 100)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
