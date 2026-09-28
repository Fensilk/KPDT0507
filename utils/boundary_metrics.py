# -*- coding: utf-8 -*-
"""边界定位指标（**边界级**，单位 = 帧）。

定义冻结于 `docs/0925实验第十一阶段规划.md` §5.1，本模块是其实现。

与既有指标的分工（**不要混用**）：
  - `utils/metrics.py`        逐帧类别指标（P/R/F1、acc、混淆矩阵）
  - `utils/timeline_metrics.py` 事件级（覆盖率/误报/翻转/延迟/逐视频 F1）
  - **本模块**                  边界级（切换点定位，带容差匹配）

关键定义（分子 / 分母 / 口径 / 单位）：
  - 真值边界集合 `B_gt = { t : y[t] != y[t-1], t ∈ [1, T-1] }`，在整段视频上算
  - 预测边界集合 `B_pred` 同定义，由**预测序列**派生。调用方须先施加与评测一致的
    后处理（多窗投票等），否则预测抖动会膨胀 |B_pred| 从而压低精确率
  - 匹配：按时间序**贪心一对一**，命中条件 `|b_pred - b_gt| <= tau`
  - 边界精确率 = 匹配上的 pred 数 / |B_pred|      （分母 = 模型报出的边界数）
  - 边界召回率 = 匹配上的 gt 数   / |B_gt|        （分母 = 真值边界数）
  - 边界 F1   = 2PR/(P+R)
  - **适用人群**：全部视频，但**不含任何真值边界的视频被排除（记 None），而非记 0**

已知缺陷（随指标一起声明，勿单独引用）：
  1. `tau` 影响可比性 → 必须 τ=1/3/5 三档并列报告
  2. 抖动惩罚偏重精确率（见上）
  3. 样本量小（1200 视频仅 249 个切换点）→ 跨配置 <0.03 的差异不可当真

CLAUDE.md 强制：本模块必须先通过①GT 自检满分 ②负对照单调劣化 才可采用。
验收脚本：`analysis/verify_boundary_metrics.py`
"""

from __future__ import annotations

# 三元类别名（0=fall, 1=fallen, 2=normal），与 models/dataset.py 一致
NAMES = {0: "fall", 1: "fallen", 2: "normal"}

# 切换类型：以 (前态, 后态) 命名
TYPE_ONSET = "normal->fall"      # 跌倒起始
TYPE_SETTLE = "fall->fallen"     # 倒地确认
TYPE_RECOVER = "fallen->normal"  # 起身
TYPE_OTHER = "other"             # 其余（罕见）


def find_boundaries(seq):
    """相邻取值变化的位置集合。

    seq: 逐帧类别序列（int）。返回 sorted list，t ∈ [1, len-1]，
    满足 seq[t] != seq[t-1]（即 t 是"变化后的第一帧"）。
    """
    s = list(seq)
    return [t for t in range(1, len(s)) if s[t] != s[t - 1]]


def boundary_type(seq, t):
    """位置 t 处的切换类型（用前态/后态命名）。"""
    a, b = seq[t - 1], seq[t]
    if a == 2 and b == 0:
        return TYPE_ONSET
    if a == 0 and b == 1:
        return TYPE_SETTLE
    if a == 1 and b == 2:
        return TYPE_RECOVER
    return TYPE_OTHER


def match_boundaries(gt, pred, tau):
    """贪心一对一匹配（按时间序双指针），命中条件 |gt - pred| <= tau。

    返回 (TP, FP, FN)。双指针贪心是"带容差区间匹配"的标准解：
    两侧任一超出容差就推进较靠前的那个（早到的 pred 记 FP、早到的 gt 记 FN）。
    """
    i = j = 0
    tp = 0
    while i < len(gt) and j < len(pred):
        if abs(gt[i] - pred[j]) <= tau:
            tp += 1
            i += 1
            j += 1
        elif pred[j] < gt[i]:
            j += 1
        else:
            i += 1
    return tp, len(pred) - tp, len(gt) - tp


def _prf(tp, fp, fn):
    """由 TP/FP/FN 得 P/R/F1。无预测时 P 记 0（而非 0/0），以保"全漏 → F1=0"。"""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_video_boundary_metrics(gt_seq, pred_seq, tau=3):
    """单条视频的边界指标。

    **无真值边界的视频返回 None**（排除而非记 0）——这是 GT 自检能满分的关键：
    若记 0，则标签恒定的视频永远拿不到满分，自检本身会暴露该实现错误。
    """
    gt = find_boundaries(gt_seq)
    if not gt:
        return None
    pred = find_boundaries(pred_seq)
    tp, fp, fn = match_boundaries(gt, pred, tau)
    p, r, f1 = _prf(tp, fp, fn)
    return {
        "n_gt": len(gt), "n_pred": len(pred),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
    }


def compute_video_boundary_metrics_by_type(gt_seq, pred_seq, tau=3):
    """按切换类型分列（同类型才允许匹配）。仅统计真值中出现过的类型。"""
    gt_all = find_boundaries(gt_seq)
    if not gt_all:
        return None
    pred_all = find_boundaries(pred_seq)
    out = {}
    for tname in (TYPE_ONSET, TYPE_SETTLE, TYPE_RECOVER, TYPE_OTHER):
        gt = [t for t in gt_all if boundary_type(gt_seq, t) == tname]
        if not gt:
            continue
        pred = [t for t in pred_all if boundary_type(pred_seq, t) == tname]
        tp, fp, fn = match_boundaries(gt, pred, tau)
        p, r, f1 = _prf(tp, fp, fn)
        out[tname] = {"n_gt": len(gt), "n_pred": len(pred),
                      "tp": tp, "fp": fp, "fn": fn,
                      "precision": p, "recall": r, "f1": f1}
    return out


def aggregate_boundary_metrics(per_video, taus=(1, 3, 5)):
    """聚合逐视频结果。

    per_video: `{key: {tau: metrics_dict|None}}`（每个 tau 一份）或
               `{key: metrics_dict|None}`（单 tau，等价于 taus=(tau,)）。

    同时给两个口径，**禁止只报一个**：
      - `micro_*`：把全部视频的 TP/FP/FN 相加再算 P/R/F1（按边界数聚合）
      - `macro_*`：逐视频 F1 的等权均值（每视频等权，非事件视频已排除）
    """
    res = {}
    for tau in taus:
        agg = {k: 0 for k in ("tp", "fp", "fn")}
        f1s, prs, rcs = [], [], []
        n_videos = 0
        keys = getattr(per_video, "keys", lambda: range(len(per_video)))()
        for k in keys:
            v = per_video[k]
            if v is None:
                continue
            m = v[tau] if (isinstance(v, dict) and tau in v) else v
            if m is None:
                continue
            n_videos += 1
            for kk in agg:
                agg[kk] += m[kk]
            f1s.append(m["f1"])
            prs.append(m["precision"])
            rcs.append(m["recall"])
        p, r, f1 = _prf(agg["tp"], agg["fp"], agg["fn"])
        res[f"tau{tau}"] = {
            "n_videos": n_videos,
            "n_gt": agg["tp"] + agg["fn"],
            "n_pred": agg["tp"] + agg["fp"],
            "micro_tp": agg["tp"], "micro_fp": agg["fp"], "micro_fn": agg["fn"],
            "micro_precision": p, "micro_recall": r, "micro_f1": f1,
            "macro_precision": sum(prs) / len(prs) if prs else 0.0,
            "macro_recall": sum(rcs) / len(rcs) if rcs else 0.0,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    return res
