"""
逐视频错误分析：最优模型 p7d_delta 在 test 集上错在哪些视频、什么错误类型。

输入: analysis/diagnose_lying_event/test_preds.json (dump_test_preds.py 导出, 含逐视频 bridge_pred)
输出: analysis/per_video_errors.csv + 终端 Top 列表

错误类型（帧级，GT=labels_16, pred=bridge_pred 三分类）:
  fall_missed      : GT=fall(1),   pred=normal → 漏检跌倒
  fall_as_fallen   : GT=fall(1),   pred=fallen
  fallen_missed    : GT=fallen(2), pred=normal → 漏检倒地
  fallen_as_fall   : GT=fallen(2), pred=fall
  lying_event_fp   : GT∈{lying,lie_down}(6,5), pred∈{fall,fallen} → 躺→事件假阳性 (Phase7/8 已知瓶颈)
  other_normal_fp  : GT=其他 normal, pred∈{fall,fallen}

同时给出原始输出与 RuleV2 校正后两种口径，看规则在每个视频上是救了还是误伤。

用法:
    python analysis/per_video_errors.py [--top 30] [--out analysis/per_video_errors.csv]
"""
import os, sys, json, csv, argparse
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.eval_rule_c import apply_rulev2

PREDS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "diagnose_lying_event/test_preds.json")

FALL, FALLEN = 1, 2
LIE_DOWN, LYING = 5, 6
ERR_TYPES = ["fall_missed", "fall_as_fallen", "fallen_missed",
             "fallen_as_fall", "lying_event_fp", "other_normal_fp"]


def per_frame_errors(l16, pred):
    """返回该视频各错误类型的帧数计数。"""
    c = Counter()
    for l, p in zip(l16, pred):
        if l == FALL:
            if p == 2:      c["fall_missed"] += 1
            elif p == 1:    c["fall_as_fallen"] += 1
        elif l == FALLEN:
            if p == 2:      c["fallen_missed"] += 1
            elif p == 0:    c["fallen_as_fall"] += 1
        elif l in (LIE_DOWN, LYING):
            if p in (0, 1): c["lying_event_fp"] += 1
        else:
            if p in (0, 1): c["other_normal_fp"] += 1
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "per_video_errors.csv"))
    args = parser.parse_args()

    with open(PREDS, "r", encoding="utf-8") as f:
        preds = json.load(f)
    print(f"[1] 加载 {len(preds)} 个 test 视频的预测")

    # 重建 RuleV2 需要的 pv 格式
    pv = {}
    for vi, d in enumerate(preds.values()):
        bp = np.asarray(d["bridge_pred"])  # eval_rule_c 需要 numpy 数组
        pv[vi] = {"bridge_pred": bp,
                  "labels_16": d["labels_16"],
                  "ternary_gt": d["ternary_gt"]}
    corr, stats = apply_rulev2(pv)
    print(f"[2] RuleV2 应用: 抑制孤立 fallen {stats['fallen_suppressed']} 帧, "
          f"保留 {stats['fallen_kept']} 帧, 事件 {stats['events']} 个")

    rows = []
    for vi, vp in enumerate(preds):
        d = preds[vp]
        l16, bp = d["labels_16"], d["bridge_pred"]
        cp = corr[vi]
        e_raw = per_frame_errors(l16, bp)
        e_corr = per_frame_errors(l16, cp)
        gt_cnt = Counter(l16)
        has_event = (gt_cnt.get(FALL, 0) + gt_cnt.get(FALLEN, 0)) > 0
        rows.append({
            "path": vp,
            "n_frames": len(bp),
            "has_event_gt": has_event,
            "wrong_raw": sum(e_raw.values()),
            "wrong_corr": sum(e_corr.values()),
            "rule_saved": sum(e_raw.values()) - sum(e_corr.values()),
            "gt_16dist": " ".join(f"{k}:{n}" for k, n in sorted(gt_cnt.items())),
            **{f"raw_{t}": e_raw[t] for t in ERR_TYPES},
            **{f"corr_{t}": e_corr[t] for t in ERR_TYPES},
        })

    rows.sort(key=lambda r: -r["wrong_corr"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[3] 已保存全量 {len(rows)} 个视频到 {args.out}")

    # ── 汇总统计 ──
    tot_raw = sum(r["wrong_raw"] for r in rows)
    tot_corr = sum(r["wrong_corr"] for r in rows)
    print(f"\n[汇总] 全 test 集错误帧: raw={tot_raw:,}  RuleV2后={tot_corr:,}"
          f"  规则净救回 {tot_raw - tot_corr:,} 帧")
    for t in ERR_TYPES:
        print(f"  {t:16s} raw={sum(r[f'raw_{t}'] for r in rows):6,}  "
              f"corr={sum(r[f'corr_{t}'] for r in rows):6,}")

    # ── 按错误类型分别 Top 列表 ──
    print(f"\n[Top {args.top} 错误最严重视频（按 RuleV2 后错帧数）]")
    print(f"{'#':>3} {'path':40s} {'错帧':>4} {'raw':>4} {'类型细目(corr)':s}")
    for i, r in enumerate(rows[:args.top], 1):
        det = " ".join(f"{t[:6]}={r[f'corr_{t}']}"
                       for t in ERR_TYPES if r[f"corr_{t}"])
        print(f"{i:>3} {r['path']:40s} {r['wrong_corr']:>4} {r['wrong_raw']:>4}  {det}")

    # ── 规则净效果最差（规则误伤）视频 ──
    hurt = [r for r in rows if r["rule_saved"] < 0]
    hurt.sort(key=lambda r: r["rule_saved"])
    print(f"\n[规则反而变差（误伤真事件）的视频: {len(hurt)} 个]")
    for r in hurt[:10]:
        print(f"  {r['path']:40s} raw错{r['wrong_raw']:>3} → RuleV2后{r['wrong_corr']:>3}"
              f"  净{r['rule_saved']:+d}")


if __name__ == "__main__":
    main()
