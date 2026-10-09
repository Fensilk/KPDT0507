# -*- coding: utf-8 -*-
"""复跑判读的**曲线口径取值**（预登记 §4b 的 ①「曲线口径值」）。

给 `analysis/p2_judge.py` 的复跑段消费：对**固定网格** epoch_1/2/3，逐个给出
每个 run 的 P@R=0.80 与相对参照 run 的支配余量，供 run 间**描述性**比较。

为什么单独一个脚本、而不直接复用 `judge_w1_arm.py`
---------------------------------------------------
`judge_w1_arm.py` 是**单点判定器**：它按「一个目录一个 `pv_best.pt`」取 pv，
输出的是"支配/否决"的**结论**，且只认 `pv_best_probs.pt` / `pv_best.pt` 两个文件名。
而复跑要的是**逐 epoch 的标量**（ep1/2/3 各一组），且 §4b 明写「**只报描述、不设阈值**」。
两者消费同一批 pv、判定实现同样来自 `analysis/prec_rec_curves.py`（**不重写**，
避免"副本漂移"）；差别只在"取哪些点、以什么形态输出"。

⚗ 为什么必须放在**子进程**里跑
------------------------------
`prec_rec_curves.py` 顶部 `import torch`（pv 是 torch 张量），而本机 `p2_judge.py`
直接 import torch 会报 OMP duplicate（CLAUDE.md 坑 #7）——这与 p2_judge 用 subprocess
调 judge_w1_arm 是**同一个理由**。

⚠ 复跑用的 pv 文件名是 `pv_<ckpt>.pt`（见 run_p11_p2_offline.sh 的 RERUN/主臂统一 dump），
  **不是** `pv_best.pt`。

用法（项目根目录）：
    python -u analysis/rerun_curves.py \
        --ref logs/phase11/p2_anchor \
        --runs logs/phase11/p2_anchor_seed43 logs/phase11/p2_1a logs/phase11/p2_4 \
        --out logs/phase11/p2_judge/rerun_curves.json
"""
import os
import sys
import json
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRID = ("epoch_1", "epoch_2", "epoch_3")
# 轴 cls：0=fall、1=fallen（与 prec_rec_curves.AXES 一致）；normal 不进曲线口径
AXES = {0: "fall", 1: "fallen"}


def load_curves_module():
    """从**唯一那份实现**加载，绝不在这里重写曲线/支配算法。"""
    sys.path.insert(0, os.path.join(BASE, "analysis"))
    import prec_rec_curves as mod
    return mod


def pv_path(run_dir, ckpt):
    return os.path.join(BASE, run_dir, f"pv_{ckpt}.pt")


PR80_DEFAULT = 0.80


def curve_of(mod, run_dir, ckpt, protocol, pr80):
    """返回 {轴名: P@R 或 None}；pv 缺失返回 None。"""
    p = pv_path(run_dir, ckpt)
    if not os.path.exists(p):
        return None
    _, pv = mod.load_pv(f"{ckpt}={p}")
    out = {}
    for cls, ax in AXES.items():
        try:
            cur = mod.curves_for_axis(pv, cls, protocol)
        except KeyError as e:
            # 该轴无可用类别（数据里没有）——如实记为 None，不猜
            print(f"    [warn] {run_dir}/{ckpt} {ax} 轴取曲线失败：{e}")
            out[ax] = None
            continue
        out[ax] = mod.prec_at_recall(cur, pr80)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参照 run 的输出目录（同一批次的 anchor）")
    ap.add_argument("--runs", nargs="+", required=True, help="要比的各 run 输出目录")
    ap.add_argument("--out", required=True, help="JSON 输出路径")
    ap.add_argument("--protocol", default="voted", choices=["voted", "flat"])
    ap.add_argument("--pr80", type=float, default=PR80_DEFAULT)
    ap.add_argument("--ckpts", nargs="+", default=list(GRID),
                    help="固定网格（默认 epoch_1 epoch_2 epoch_3，跨 run 可比）")
    args = ap.parse_args()

    mod = load_curves_module()

    def tag_of(d):
        return os.path.basename(d.rstrip("/\\"))

    result = {"ref": args.ref, "ref_tag": tag_of(args.ref), "pr80": args.pr80,
              "protocol": args.protocol, "grid": list(args.ckpts), "runs": {}}

    # 参照 run 自己也要逐 epoch 取值（协同性检验的另一侧就是它）
    all_runs = [args.ref] + [r for r in args.runs if r != args.ref]
    ref_curves = {}

    print("=" * 86)
    print(f"复跑曲线口径取值  P@R={args.pr80:.2f}  协议={args.protocol}  "
          f"网格={' '.join(args.ckpts)}")
    print(f"  参照 run = {args.ref}")
    print("=" * 86)

    for run in all_runs:
        tag = tag_of(run)
        per_ckpt = {}
        for ck in args.ckpts:
            c = curve_of(mod, run, ck, args.protocol, args.pr80)
            if c is None:
                print(f"  [缺] {tag}/{ck}: 无 pv_{ck}.pt —— 该 run 的离线未按新 dump 规则跑？")
                per_ckpt[ck] = None
                continue
            # 与**同一 epoch** 的参照比（跨 epoch 比没有意义）
            refc = ref_curves.get(ck)
            row = {"p_at_r": c, "ref_p_at_r": refc, "delta": None, "dominance": None}
            if refc is not None:
                row["delta"] = {ax: (None if (c[ax] is None or refc[ax] is None)
                                     else round(c[ax] - refc[ax], 6)) for ax in c}
                _, pv_a = mod.load_pv(f"a={pv_path(run, ck)}")
                _, pv_b = mod.load_pv(f"b={pv_path(args.ref, ck)}")
                dm = {}
                for cls, ax in AXES.items():
                    try:
                        dm[ax] = mod.dominance(mod.curves_for_axis(pv_a, cls, args.protocol),
                                               mod.curves_for_axis(pv_b, cls, args.protocol))
                    except KeyError:
                        dm[ax] = None
                row["dominance"] = dm
            per_ckpt[ck] = row
            if run == args.ref:
                ref_curves[ck] = c
            d = row["delta"]
            dtxt = " ".join(f"{ax} {('n/a' if d[ax] is None else format(d[ax], '+.4f'))}"
                            for ax in d) if d else "（参照本身）"
            print(f"  {tag:22s} {ck:10s} P@R " +
                  " ".join(f"{ax}={('n/a' if c[ax] is None else format(c[ax], '.4f'))}"
                           for ax in c) + f"   Δ(本run−参照) {dtxt}")
        result["runs"][tag] = per_ckpt

    os.makedirs(os.path.dirname(os.path.join(BASE, args.out)), exist_ok=True)
    with open(os.path.join(BASE, args.out), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已写出 {args.out}")


if __name__ == "__main__":
    main()
