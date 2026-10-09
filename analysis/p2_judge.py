# -*- coding: utf-8 -*-
"""P2 增补的判读：**R1 / R3**（R2 已撤销，见预登记 §2.1）。

预登记：docs/1007实验第十一阶段增补-P2三臂重跑冻结.md

  R1 = p2_anchor vs **已发表 E1-full** → 环境效应（环境 + 一次重训，混合；兼作 band 的保守上界）
  R3 = p2_1a / p2_4 vs **p2_anchor**   → 本阶段模型调整的效果，逐条对照旧结论

⚠ 曲线判定**不在这里重写**，而是 subprocess 调 `experiments/judge_w1_arm.py`
（它调用 analysis/prec_rec_curves.py 的 dominance()）——本项目有"副本间漂移"的教训，
判据必须与画图同源。

用法（项目根目录）：
    python -u analysis/p2_judge.py
"""
import os
import sys
import json
import glob
import subprocess

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

OLD_ANCHOR = "logs/phase9/e1_full"
P2 = {"anchor": "logs/phase11/p2_anchor", "1a": "logs/phase11/p2_1a", "4": "logs/phase11/p2_4"}

# §7.3 的五个阈值（**冻结值，取自已发表 E1-full，不得重算**）
TH = {"fall_prec": ("gt", 0.6951), "fall_rec": ("ge", 0.8205),
      "timeline_fall_f1": ("gt", 0.7827), "fallen_prec": ("ge", 0.7351),
      "timeline_avg_f1": ("ge", 0.7213)}

# §7.3 五条相对锚点的余量（用于声明噪声敏感性）
MARGIN = {"fall_prec": 0.00, "fall_rec": 0.01, "timeline_fall_f1": 0.00,
          "fallen_prec": 0.02, "timeline_avg_f1": 0.02}
# 已实测的落点噪声（同一臂内仅换 ckpt 的摆幅），用于与余量对照
LANDING_NOISE = {"单点 (A0)": (0.076, 0.179), "曲线 (1a / 4)": (0.021, 0.135)}


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def monitor_curve(d):
    m = jload(os.path.join(d, "monitor.json"))
    return {x["step"]: x for x in m} if m else None


def eval_of(d, name):
    """返回该 ckpt 的 eval JSON —— **取 n_videos 最大的那一份**。

    ⚠ 不能按文件名顺序取：同一 ckpt 可能同时存在
        · `eval_epoch_N.full.json`（新臂，全量 1200）
        · `eval_epoch_N_model.subset.json`（当年只跑了 300 子集）
        · `eval_epN.json`（**锚点当年的命名**，全量 1200）
    按名字排序会误取 300 子集（分母不同 → 不可比）。按 n_videos 取最大最稳。
    也排除 `*_dump.json`（另一种格式）。
    """
    # 已知网格名走固定表；**其余任意 ckpt 名**（如离线选出的 step_13800）按名字直接匹配。
    # 之前这里写死成 [name]，于是 §7.3 一传进选点结果就 KeyError——那条路径在
    # selection.json 存在之前从未被执行过，属于潜伏 bug（2026-10-09 首次触发）。
    pats = {"best": ["eval_best_model.json", "eval_best.json"],
            "epoch_1": ["eval_epoch_1*.json", "eval_ep1.json"],
            "epoch_2": ["eval_epoch_2*.json", "eval_ep2.json"],
            "epoch_3": ["eval_epoch_3*.json", "eval_ep3.json"]}.get(name)
    if pats is None:
        pats = ["eval_%s*.json" % name, "eval_%s_model*.json" % name]
    best = None
    for pat in pats:
        for p in glob.glob(os.path.join(d, pat)):
            if p.endswith("_dump.json"):
                continue
            js = jload(p)
            if js and (best is None or (js.get("n_videos") or 0) > (best.get("n_videos") or 0)):
                best = js
    return best


def pick(d, js, *keys):
    for k in keys:
        if k in js:
            return js[k]
    return float("nan")


def line(tag, js):
    if not js:
        return f"  {tag:26s} （无评测）"
    ins, mr = js.get("instance") or {}, (js.get("merged_raw") or {}).get("ternary") or {}
    avg = lambda D: np.mean([D["fall_f1"], D["fallen_f1"], D["normal_f1"]]) if D else float("nan")
    return (f"  {tag:26s} n={js.get('n_videos')}  "
            f"avg_f1(inst) {avg(ins):.4f}  fall_prec {ins.get('fall_precision', float('nan')):.4f}  "
            f"fallen_prec {ins.get('fallen_precision', float('nan')):.4f}  "
            f"avg_f1(merged) {avg(mr):.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 复跑（同环境换 seed）判读 —— 预登记 §4b
#
# ⚠ 判读方式冻结为「**乙**：只报描述、不设阈值」。初稿曾拍过「符号一致 ≥2/3」，
#   **已撤**——零假设下该事件概率恰为 0.5，等于掷硬币，没有判别力；
#   「通过数差 ≤1 项」同样无依据、亦撤。**所以本节一个判定都不下，只并列数字。**
# ══════════════════════════════════════════════════════════════════════════
RERUN_DIR = "logs/phase11/p2_anchor_seed43"     # 复跑（seed 43）的输出目录
GRID = ("epoch_1", "epoch_2", "epoch_3")        # 固定网格：跨 run 可比（复跑不做选点）
CURVES_JSON = "logs/phase11/p2_judge/rerun_curves.json"


def _f(v, signed=False):
    return " n/a " if v is None else format(v, "+.4f" if signed else ".4f")


def _tlmean(v):
    """timeline 的 `per_video_*` 是**嵌套 dict**（{n, mean, median, std, 分桶...}）→ 取 mean。

    ⚠ 与 `judge_w1_arm.scalar()` 同一约定。此前本文件读的是 `fall_f1` / `avg_f1`
    两个**并不存在**的键名，导致 §7.3 的两项逐视频指标恒为"缺"、五项实际只算三项
    （2026-10-09 首次跑到 §7.3 时暴露）。
    字段与冻结阈值的同源已核对：e1_full/timeline_metrics.json 的 `E1-full_best`
    → per_video_fall_f1.mean=0.7827、per_video_avg_f1.mean=0.7413，正是 §7.3 的锚点值。
    """
    if isinstance(v, dict):
        return v.get("mean")
    return v


def seven3_of(d, ck):
    """某 ckpt 的 §7.3 五项**实测值**（阈值共用 TH，不重算）。"""
    js = eval_of(os.path.join(BASE, d), ck)
    if not js:
        return None
    ins = js.get("instance") or {}
    tl = jload(os.path.join(BASE, d, "timeline_%s.json" % ck)) or {}
    tl = next((v for v in tl.values() if isinstance(v, dict)), {}) if tl else {}
    return {"fall_prec": ins.get("fall_precision"),
            "fall_rec": ins.get("fall_recall"),
            "fallen_prec": ins.get("fallen_precision"),
            "timeline_fall_f1": _tlmean(tl.get("per_video_fall_f1")),
            "timeline_avg_f1": _tlmean(tl.get("per_video_avg_f1"))}


def seven3_pass(k, v):
    if v is None:
        return False
    op, th = TH[k]
    return (v > th) if op == "gt" else (v >= th)


def rerun_section():
    print("=" * 92)
    print("【复跑】同环境换 seed（p2_anchor_seed43）—— 预登记 §4b：**只报描述、不设阈值**")
    print("=" * 92)

    # 守卫看**产物**、不看目录：本地为了挂日志镜像会先建出同名的空目录，
    # 用 isdir 判断会误以为"已产出"，进而去调 rerun_curves 白跑一趟。
    if not os.path.exists(os.path.join(BASE, RERUN_DIR, "eval_epoch_1.json")):
        print("  ⚠ 复跑评测产物尚未产出（训练 ≈16.2 h + 离线 ≈1.8 h）—— 本节跳过")
        print("     需要 %s/ 下的 eval_epoch_*.json、timeline_epoch_*.json、pv_epoch_*.pt\n"
              % RERUN_DIR)
        return

    # ── ① 曲线口径值（协同性检验的输入）──
    # 走子进程：prec_rec_curves.py 顶部 import torch，本进程直接 import 会 OMP duplicate
    # （CLAUDE.md 坑 #7）——与 R3 调 judge_w1_arm 同一个理由。
    script = os.path.join(BASE, "analysis", "rerun_curves.py")
    try:
        r = subprocess.run([PY, "-u", script, "--ref", P2["anchor"],
                            "--runs", RERUN_DIR, P2["1a"], P2["4"],
                            "--out", CURVES_JSON],
                           cwd=BASE, capture_output=True, timeout=3600,
                           # ⚠ 必须显式 utf-8：子进程会打印中文，而 Windows 的
                           #   text=True 默认按**本地代码页(GBK)**解码 → UnicodeDecodeError
                           #   在读取线程里炸掉，主流程只看到空输出，极难归因。
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("  ⚠ rerun_curves.py 退出码 %d，协同性可能算不出：" % r.returncode)
            print("   ", (r.stderr or r.stdout)[-500:])
    except Exception as e:
        print("  ⚠ 调用 rerun_curves.py 异常：%s" % e)

    cv = jload(os.path.join(BASE, CURVES_JSON))
    if not cv:
        print("  ⚠ 无曲线取值产物，跳过协同性\n")
        return

    ref_tag = cv.get("ref_tag")
    grid = cv.get("grid", list(GRID))
    print("\n  ① 曲线口径值 P@R=%.2f（参照 run = %s）" % (cv.get("pr80", 0.8), ref_tag))
    for tag, per in cv.get("runs", {}).items():
        for ck in grid:
            row = per.get(ck) or {}
            pr = row.get("p_at_r")
            if not pr:
                print("    %-24s %-9s —— 无 pv（该 run 的离线未按 per-candidate dump 规则跑）"
                      % (tag, ck))
                continue
            dl = row.get("delta") or {}
            print("    %-24s %-9s P@R fall=%s fallen=%s   Δ(本run−参照) fall=%s fallen=%s"
                  % (tag, ck, _f(pr.get("fall")), _f(pr.get("fallen")),
                     _f(dl.get("fall"), True), _f(dl.get("fallen"), True)))

    # ── ② 协同性：A_i − B₁_i 与 A_i − B₂_i 的符号一致数 k/3 ──
    # rerun_curves 的 delta 一律是 (X − 参照=seed42)。故
    #   A − B₁ = delta(A)，  B₂ − B₁ = delta(seed43)，  A − B₂ = delta(A) − delta(seed43)
    print("\n  ② 协同性（§4b）：A−B₁ 与 A−B₂ 在 %d 个网格点上的**符号一致数 k/3**" % len(grid))
    print("     A = 臂 ｜ B₁ = %s（seed42）｜ B₂ = %s（seed43）"
          % (ref_tag, os.path.basename(RERUN_DIR)))
    per43 = (cv.get("runs") or {}).get(os.path.basename(RERUN_DIR)) or {}
    for arm in ("1a", "4"):
        per = (cv.get("runs") or {}).get(os.path.basename(P2[arm])) or {}
        if not per or not per43:
            print("     ── %s：缺数据，跳过（需该臂与复跑的 pv 都在）" % arm)
            continue
        for ax in ("fall", "fallen"):
            agree = n = 0
            detail = []
            for ck in grid:
                ra, r43 = per.get(ck) or {}, per43.get(ck) or {}
                da = (ra.get("delta") or {}).get(ax)      # A − B₁
                d43 = (r43.get("delta") or {}).get(ax)    # B₂ − B₁
                if da is None or d43 is None:
                    continue
                db = da - d43                              # A − B₂
                n += 1
                if da == 0 or db == 0:
                    detail.append("%s=0" % ck.replace("epoch_", "ep"))
                elif (da > 0) == (db > 0):
                    agree += 1
                    detail.append("%s同" % ck.replace("epoch_", "ep"))
                else:
                    detail.append("%s异" % ck.replace("epoch_", "ep"))
            print("       %-3s / %-7s k=%d/%d   [%s]" % (arm, ax, agree, n, " ".join(detail)))

    # ── ③ §7.3 的模式稳定性：两次 anchor 各一条通过数序列，并列原样 ──
    print("\n  ③ §7.3 五项通过数（ep1/2/3）—— 两次 anchor **并列原样**，不下判定")
    print("     ⚠ 阈值冻结自 E1-full（不得重算）；五条余量 0–0.02 ≪ 落点噪声 0.076–0.179")
    print("       → 逐项「过/不过」基本是抽签（与 §7.2 被废同因），故只作模式描述")
    for tag, d in ((os.path.basename(P2["anchor"]) + "（seed42）", P2["anchor"]),
                   (os.path.basename(RERUN_DIR) + "（seed43）", RERUN_DIR)):
        seq = []
        for ck in GRID:
            got = seven3_of(d, ck)
            if got is None:
                seq.append("  --")
                continue
            seq.append("%d/5" % sum(1 for k in TH if seven3_pass(k, got[k])))
        print("     %-28s ep1=%-5s ep2=%-5s ep3=%-5s" % (tag, seq[0], seq[1], seq[2]))
    print("     （R2 已撤销 → 本复跑**不产出阈值判定**，只提供上述描述；见预登记 §4b）")
    print()


def main():
    print("=" * 92)
    print("P2 判读（R1 / R3）—— R2 已撤销，见预登记 §2.1")
    print("=" * 92)

    # ── 文件就绪检查 ──
    ready = {}
    for k, d in P2.items():
        ready[k] = os.path.isdir(os.path.join(BASE, d))
    if not all(ready.values()):
        miss = [k for k, v in ready.items() if not v]
        print(f"⚠ 尚未产出：{miss}（训练未结束或离线阶段未跑）—— 下面能算多少算多少\n")

    # ── R1：p2_anchor vs 已发表 E1-full ──
    print("【R1】环境效应：p2_anchor vs 已发表 E1-full（环境 + 一次重训，混合）")
    ca = monitor_curve(os.path.join(BASE, P2["anchor"]))
    co = monitor_curve(os.path.join(BASE, OLD_ANCHOR))
    if ca and co:
        common = sorted(set(ca) & set(co))
        d1 = [ca[s]["val_avg_f1"] - co[s]["val_avg_f1"] for s in common]
        print(f"  val 曲线：可比 {len(common)}/{len(co)} 点（每 600 步）")
        print(f"    val_avg_f1 差（新 − 旧）：均值 {np.mean(d1):+.4f}  中位 {np.median(d1):+.4f}  "
              f"极差 {min(d1):+.4f}~{max(d1):+.4f}  |差| 中位 {np.median(np.abs(d1)):.4f}")
        for tag, s in (("step 9600 (ep1 末)", 9600), ("step 19200 (ep2 末)", 19200),
                       ("step 28800 (ep3 末)", 28800)):
            if s in ca and s in co:
                print(f"    {tag:22s} 新 {ca[s]['val_avg_f1']:.4f}  旧 {co[s]['val_avg_f1']:.4f}  "
                      f"Δ {ca[s]['val_avg_f1'] - co[s]['val_avg_f1']:+.4f}")
    else:
        print("  （monitor.json 未齐）")
    print("  test 逐 ckpt：")
    for nm in ("best", "epoch_1", "epoch_2", "epoch_3"):
        a = eval_of(os.path.join(BASE, P2["anchor"]), nm)
        o = eval_of(os.path.join(BASE, OLD_ANCHOR), nm)
        print(line(f"新 anchor/{nm}", a))
        print(line(f"旧 E1-full/{nm}", o))
    print()

    # ── R3：曲线判定（调冻结实现，不重写）──
    print("【R3】曲线判定：p2_<臂> vs p2_anchor（调用冻结的 judge_w1_arm.py，与画图同源）")
    judge = os.path.join(BASE, "experiments", "judge_w1_arm.py")
    for arm in ("1a", "4"):
        d = os.path.join(BASE, P2[arm])
        if not os.path.isdir(d):
            print(f"  ── {arm}: 未产出，跳过"); continue
        try:
            r = subprocess.run([PY, "-u", judge, os.path.relpath(d, BASE),
                                "--a0", os.path.relpath(os.path.join(BASE, P2["anchor"]), BASE),
                                "--tag", f"p2-{arm}"],
                               cwd=BASE, capture_output=True, timeout=900,
                               # ⚠ 同上：judge_w1_arm 会打印中文表格，text=True 走 GBK 会炸。
                               encoding="utf-8", errors="replace")
            txt = r.stdout
            verdict = [l for l in txt.splitlines() if l.startswith("判定：")]
            print(f"  ── {arm}: {verdict[0] if verdict else '（未解析到判定，见下）'}")
            for l in txt.splitlines():
                if l.strip().startswith(("fall ", "fallen ")):
                    print("     " + l.strip())
            if not verdict:
                print("     " + (r.stderr or txt)[-400:])
        except Exception as e:
            print(f"  ── {arm}: 调用失败 {e}")
    print()

    # ── R3：§7.3 五项（冻结阈值，不得重算）──
    print("【R3】§7.3 五项（阈值取自已发表 E1-full，**冻结**）")
    sel_p = os.path.join(BASE, "logs", "phase11", "p2_val_probe", "selection.json")
    sel = jload(sel_p) or {}
    for arm in ("anchor", "1a", "4"):
        ck = sel.get(arm, "best")
        js = eval_of(os.path.join(BASE, P2[arm]), ck) if os.path.isdir(os.path.join(BASE, P2[arm])) else None
        if not js:
            print(f"  ── {arm}: （{ck} 无评测／未产出）"); continue
        ins = js.get("instance") or {}
        got = {"fall_prec": ins.get("fall_precision"), "fall_rec": ins.get("fall_recall"),
               "fallen_prec": ins.get("fallen_precision")}
        # 逐视频口径：runner 每个候选各写一份 `timeline_<ckpt>.json`（单键），取选中者那份
        tl = jload(os.path.join(BASE, P2[arm], f"timeline_{ck}.json")) or {}
        tl = next((v for v in tl.values() if isinstance(v, dict)), {}) if tl else {}
        got["timeline_fall_f1"] = _tlmean(tl.get("per_video_fall_f1"))
        got["timeline_avg_f1"] = _tlmean(tl.get("per_video_avg_f1"))
        npass, rows = 0, []
        for k, (op, th) in TH.items():
            v = got.get(k)
            if v is None:
                rows.append(f"    {k:18s} 阈值 {op} {th:.4f}   实测 缺")
                continue
            ok = (v > th) if op == "gt" else (v >= th)
            npass += ok
            rows.append(f"    {k:18s} 阈值 {op} {th:.4f}   实测 {v:.4f}   {'✅' if ok else '❌'}")
        print(f"  ── {arm}（选中 {ck}）：**{npass}/5**")
        print("\n".join(rows))
    print()

    # ── 复跑（同环境换 seed）判读 —— 预登记 §4b ──
    rerun_section()

    # ── 判据优先级 + 噪声敏感性（预登记 §2.2 / §2.3）──
    print("【判据优先级】预登记 §2.2：**曲线为主，§7.3 为辅**（自成一体，不引用任何旧结论）")
    print("  §7.3 五条相对锚点余量：", {k: f"{v:.2f}" for k, v in MARGIN.items()})
    print("  已测落点噪声（同臂换 ckpt 摆幅）：", LANDING_NOISE)
    print("  → **余量 0–0.02 ≪ 噪声 0.02–0.18**：§7.3 各项的「过/不过」基本是抽签（与 §7.2 被废同因）")
    print("  → 因此 §7.3 **只作辅助读数**，判定以曲线口径为准；两者分歧时**如实并列**并说明是'能力 vs 落点'两个量")
    print("  ⚠ 事件轴的洞照旧登记（总结 §8.2 第 1 条），**不靠 §7.3 假装补上**")
    print()
    print("⚠ 撤 R2 的后果：本实验**不产出纯 run 间 band**。R3 给的是效应量的**点估计**（同环境单变量，效度好），")
    print("   但**没有误差棒**；R1 **不能**充当 band（R1 ≈ δ环境 + ε重跑，单次实现里两者可反号相消，")
    print("   幅度既非上界也非点估计；只有方差口径才有 Var(R1) ≥ Var(ε)，那是分布参数的界）。")
    print("   → R3 对「效应 vs 噪声」**不作答**（没有误差棒）；该问题由**换 seed 复跑**（§4b）另行描述， 见上【复跑】一节——那里同样**只报描述、不设阈值**。。")


if __name__ == "__main__":
    main()
