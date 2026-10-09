# -*- coding: utf-8 -*-
"""Phase 11 增补 step 1 的四道检验 + 选点试算（见 docs/1006…增补-选点规则冻结.md §2.4）。

输入（全部由 run_p11_valprobe.sh 产出，或既存）：
    logs/phase11/val_probe/reval.json     新 val 的每 ckpt 分数
    logs/phase11/val_probe/reval.pv.pt    逐视频逐帧 preds/labels（剂量-响应用）
    logs/phase11/w2_*/monitor.json        旧 val 曲线（对照 + F5 平滑源）
    logs/phase9/e1_full/monitor.json
    logs/**/eval_*.json                   test 侧数字（外部效度用）

产出：一份可读报告（stdout），四节对应四道检验。

⚠ 选点试算同时给出两种读法（见冻结文档 F5 的适用性说明）：
    A（推荐）：直接按**新 val 分数**在池内取最大 —— F5 的平滑是为"150 视频逐检查曲线"的
               噪声设计的；新 val 已把噪声降下来，且池成员相隔 9600 步，跨成员平滑无意义
    C（对照）：按 F5 字面，用**旧 val 曲线**的末 k=3 次均值取最大
    两读法都报，谁都不用挑。
"""
import os
import sys
import json
import glob

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"E1-full(锚点)": "logs/phase9/e1_full",
        "w2_1a": "logs/phase11/w2_1a",
        "w2_4": "logs/phase11/w2_4"}
EPOCH_STEP = 9600            # 9600 视频 × 2 窗 / batch 2 = 9600 步/epoch

# ckpt 文件名 → test 侧 eval JSON（用于外部效度）
TEST_JSON = {
    "best_model.pt": ["eval_best_model.json", "eval_best.json"],
    "epoch_1_model.pt": ["eval_epoch_1.full.json", "eval_ep1.json"],
    "epoch_2_model.pt": ["eval_epoch_2.full.json", "eval_ep2.json"],
    "epoch_3_model.pt": ["eval_epoch_3.full.json", "eval_ep3.json"],
}


def load_monitor(run_dir):
    p = os.path.join(BASE, run_dir, "monitor.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def ckpt_step(run_dir, name, mon):
    """ckpt 文件名 → 它对应的训练步。best 的步 = 旧 val 的原始极大点。"""
    if name.startswith("epoch_"):
        return int(name.split("_")[1]) * EPOCH_STEP
    if name.startswith("last"):
        return 3 * EPOCH_STEP
    if name.startswith("best") and mon:
        i = max(range(len(mon)), key=lambda k: mon[k]["val_avg_f1"])
        return mon[i]["step"]
    return None


def old_val_at(mon, step):
    if not mon or step is None:
        return None
    for x in mon:
        if x["step"] == step:
            return x["val_avg_f1"]
    return None


def old_val_smoothed(mon, step, k=3):
    """F5 字面：以该步为终点、向前取 k 次 val 检查的均值。"""
    if not mon or step is None:
        return None
    idx = [i for i, x in enumerate(mon) if x["step"] == step]
    if not idx:
        return None
    i = idx[0]
    lo = max(0, i - k + 1)
    return float(np.mean([mon[j]["val_avg_f1"] for j in range(lo, i + 1)]))


def test_metrics(run_dir, name):
    for cand in TEST_JSON.get(name, []):
        p = os.path.join(BASE, run_dir, cand)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            t = (d.get("merged_raw") or {}).get("ternary") or {}
            ins = d.get("instance") or {}
            out = {}
            if t:
                out["merged_avg_f1"] = float(np.mean([t["fall_f1"], t["fallen_f1"], t["normal_f1"]]))
                out["merged_fall_prec"] = t["fall_precision"]
            if ins:
                out["inst_avg_f1"] = float(np.mean([ins["fall_f1"], ins["fallen_f1"], ins["normal_f1"]]))
                out["inst_fall_prec"] = ins["fall_precision"]
                out["inst_fallen_prec"] = ins["fallen_precision"]
            out["_src"] = cand
            return out
    return None


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pv_avg_f1(pv_entry, n_videos=None):
    """从 per-video preds/labels 复算 macro F1（供剂量-响应）。"""
    from sklearn.metrics import f1_score
    vids = pv_entry if n_videos is None else pv_entry[:n_videos]
    y = np.concatenate([v["y"] for v in vids])
    p = np.concatenate([v["p"] for v in vids])
    return float(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0).mean())


def pv_avg_f1_cov(pv_entry, max_frame=None):
    """只保留帧号 ≤ max_frame 的预测复算 macro F1（max_frame=None = 全帧）。

    用于把"val 变大"与"val 看到尾部帧"两个改动拆开：旧 val 是 stride 80，
    每视频只看帧 0–63；本函数 max_frame=63 即复现那个覆盖。
    """
    from sklearn.metrics import f1_score
    ys, ps = [], []
    for v in pv_entry:
        fr = np.asarray(v["frames"])
        m = np.ones(len(fr), bool) if max_frame is None else (fr <= max_frame)
        if m.any():
            ys.append(v["y"][m]); ps.append(v["p"][m])
    if not ys:
        return float("nan")
    y = np.concatenate(ys); p = np.concatenate(ps)
    return float(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0).mean())


def main():
    reval_p = os.path.join(BASE, "logs", "phase11", "val_probe", "reval.json")
    if not os.path.exists(reval_p):
        print("还没有 reval.json —— 作业尚未产出"); return
    reval = json.load(open(reval_p, encoding="utf-8"))
    spec = reval["spec"]
    got = {r["ckpt"].replace("\\", "/"): r for r in reval["results"]}
    print(f"reval.json: 完成 {spec.get('done')}/{spec.get('total')}  "
          f"val=val.csv 全量 stride={spec['stride']}  聚合={spec['agg']}\n")

    # ── ① 规则自测：构造显然的池 ──
    print("=" * 78)
    print("① 规则自测（构造显然的池，规则必须给出显然答案）")
    fake = {"a": 0.50, "b": 0.70, "c": 0.60}
    pick = max(fake, key=fake.get)
    print(f"   池 {fake} → 取最大 = '{pick}'  期望 'b'  → {'✅ 通过' if pick == 'b' else '❌ 失败'}")
    tie = {"x": 0.60, "y": 0.60}
    print(f"   并列 {tie} → 按 F7 取 step 更小者（实现层保证）")

    # ── ② 负对照 + 选点试算 ──
    print("=" * 78)
    print("② 负对照（已知最差的 1a-ep3 不得被选中） + 选点试算")
    table = []
    for label, run_dir in RUNS.items():
        mon = load_monitor(run_dir)
        pool = []
        for name in ("best_model.pt", "epoch_1_model.pt", "epoch_2_model.pt", "epoch_3_model.pt"):
            key = f"{run_dir}/{name}"
            if key not in got:
                continue
            step = ckpt_step(run_dir, name, mon)
            pool.append({
                "ckpt": name, "step": step,
                "new_val": got[key]["val_avg_f1"],
                "old_val": old_val_at(mon, step),
                "old_smoothed": old_val_smoothed(mon, step),
            })
        if not pool:
            continue
        pool.sort(key=lambda d: (d["step"] if d["step"] is not None else 1 << 30))
        print(f"\n  ── {label} ──")
        print(f"     {'ckpt':18s}{'step':>7s}{'新val':>9s}{'旧val':>9s}{'旧val平滑':>11s}")
        for d in pool:
            print(f"     {d['ckpt']:18s}{d['step'] if d['step'] else 0:>7d}"
                  f"{d['new_val']:>9.4f}"
                  f"{d['old_val'] if d['old_val'] is not None else float('nan'):>9.4f}"
                  f"{d['old_smoothed'] if d['old_smoothed'] is not None else float('nan'):>11.4f}")
        pickA = max(pool, key=lambda d: d["new_val"])["ckpt"]
        pickC = max(pool, key=lambda d: (d["old_smoothed"] if d["old_smoothed"] is not None else -1))["ckpt"]
        cur = "best_model.pt"
        print(f"     读法 A（新 val 取最大） : {pickA}   {'= 现状' if pickA == cur else '⚠ 与现状不同'}")
        print(f"     读法 C（旧 val 平滑）   : {pickC}   {'= 现状' if pickC == cur else '⚠ 与现状不同'}")
        table.append((label, pool, pickA, pickC))

    neg = [t for t in table if t[0] == "w2_1a"]
    if neg:
        picked = neg[0][2]
        print(f"\n   负对照：1a 的池里被选中 = {picked}"
              f"  → {'❌ 选中了已知最差的 ep3，该 val 无判别力' if picked == 'epoch_3_model.pt' else '✅ 未选中 ep3'}")

    # ── ③ 外部效度：新 val 排序 vs test 排序 ──
    print("=" * 78)
    print("③ 外部效度：新 val 分数 vs test 指标（对照旧 val 的 ρ = −0.14）")
    rows = []
    for label, run_dir in RUNS.items():
        for name in ("best_model.pt", "epoch_1_model.pt", "epoch_2_model.pt", "epoch_3_model.pt"):
            key = f"{run_dir}/{name}"
            if key not in got:
                continue
            tm = test_metrics(run_dir, name)
            if not tm:
                continue
            rows.append((f"{label.split('(')[0]}/{name.replace('_model.pt','')}",
                         got[key]["val_avg_f1"], tm))
    if len(rows) >= 3:
        print(f"     {'ckpt':22s}{'新val':>9s}{'test avg_f1(merged)':>21s}{'test fall_prec':>16s}")
        for n, v, tm in rows:
            print(f"     {n:22s}{v:>9.4f}{tm.get('merged_avg_f1', float('nan')):>21.4f}"
                  f"{tm.get('merged_fall_prec', float('nan')):>16.4f}")
        for key in ("merged_avg_f1", "inst_avg_f1", "inst_fall_prec", "inst_fallen_prec"):
            vv = [tm[key] for _, _, tm in rows if key in tm]
            if len(vv) >= 3:
                rho = spearman([v for _, v, _ in rows], vv)
                print(f"     ρ(新 val, test {key:20s}) = {rho:+.3f}   n={len(vv)}")
        print("     ⚠ n 很小（每臂 4 个 ckpt、且同臂内高度相关），ρ 只作数量级参考")
    else:
        print(f"     可配对 ckpt 只有 {len(rows)} 个（test 侧 eval JSON 不全），跳过")

    # ── ④ 剂量-响应 ──
    print("=" * 78)
    print("④ 剂量-响应：新 val 取前 n 个视频复算，看 ρ 随 n 怎么变")
    pv_p = os.path.join(BASE, "logs", "phase11", "val_probe", "reval.pv.pt")
    if not os.path.exists(pv_p):
        print("     reval.pv.pt 未产出，跳过"); return
    import torch
    pv = torch.load(pv_p, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(42)          # 固定置换：val.csv 按类分组，不能直接取前 n
    for n in (150, 300, 600, 1200):
        acc = {}
        for key, entry in pv.items():
            if len(entry) < n:
                continue
            order = rng.permutation(len(entry))
            sub = [entry[i] for i in order[:n]]
            acc[key.replace("\\", "/")] = pv_avg_f1(sub)
        pairs = []
        for label, run_dir in RUNS.items():
            for name in ("best_model.pt", "epoch_1_model.pt", "epoch_2_model.pt", "epoch_3_model.pt"):
                k = f"{run_dir}/{name}"
                tm = test_metrics(run_dir, name)
                if k in acc and tm and "merged_avg_f1" in tm:
                    pairs.append((acc[k], tm["merged_avg_f1"]))
        if len(pairs) >= 3:
            print(f"     n={n:5d}  可配对 {len(pairs)} 个 ckpt   ρ(val, test merged avg_f1) = "
                  f"{spearman([a for a, _ in pairs], [b for _, b in pairs]):+.3f}")

    # ── ④b 覆盖效应（把"更多视频"与"看到尾部帧"拆开）──
    # ⚠ ④ 只变视频数、不变帧覆盖，因此**无法区分**"val 变大"与"val 覆盖了帧 64–79"。
    #   而旧 val 的另一个缺陷恰是 stride 80 → 每视频只看帧 0–63。这里补一条：
    #   同样 n=1200，只保留帧 ≤63 复算 → 与全帧对照，即为**覆盖效应**。
    print("\n     ④b 覆盖效应：n=1200，只保留帧 ≤63（模拟旧 val 的覆盖）vs 全帧")
    for tag, mf in (("只帧 0–63", 63), ("全帧 0–79", None)):
        acc = {}
        for key, entry in pv.items():
            acc[key.replace("\\", "/")] = pv_avg_f1_cov(entry, mf)
        pairs = []
        for label, run_dir in RUNS.items():
            for name in ("best_model.pt", "epoch_1_model.pt", "epoch_2_model.pt", "epoch_3_model.pt"):
                k = f"{run_dir}/{name}"
                tm = test_metrics(run_dir, name)
                if k in acc and tm and "merged_avg_f1" in tm:
                    pairs.append((acc[k], tm["merged_avg_f1"]))
        if len(pairs) >= 3:
            print(f"        {tag:12s}  ρ(val, test merged avg_f1) = "
                  f"{spearman([a for a, _ in pairs], [b for _, b in pairs]):+.3f}   n={len(pairs)}")


if __name__ == "__main__":
    main()
