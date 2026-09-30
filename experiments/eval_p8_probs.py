# -*- coding: utf-8 -*-
"""Phase 8 各臂的**逐帧概率**重评 —— 供 prec-rec 前沿曲线比较。

为什么需要（2026-09-30）：Phase 8 的核心结论「增补特征要么用误报换召回、要么压召回换精度」
是**单点 prec/rec 比较**得出的。而本项目已实测：同一模型的各 checkpoint 几乎完全落在一条
陡峭前沿上（r≈-0.96~-0.99），**单点位置基本随机**。故 Phase 8 的结论需按前沿曲线复核。

与 E1 重评的关键差别：**Phase 8 全部是冻结特征**（无 ViT 前向），所以本脚本**秒级**完成，
10 个臂 + 基线全做也只要几分钟。

架构判别（由 state_dict 的独特键，不靠猜）：
    alpha_raw        → LateFusionModel        forward(main, aux)
    gate_proj        → GatedFusionModel       forward(main, aux)
    cross_attn       → CrossAttentionFusionModel  forward(main, aux)
    input_proj (仅)  → Phase6TernaryModel      forward(features)   ← concat 路线

维度全部由权重形状推导（main_proj/aux_proj/input_proj），**并断言与数据集的
feature_dim 一致** —— 不一致会当场报错，而不是静默跑出错结果。

用法：
  python -u experiments/eval_p8_probs.py --ckpt logs/phase8/p8_gated_pose/best_model.pt \
                                         --out logs/phase8/probs/p8_gated_pose.pt
  python -u experiments/eval_p8_probs.py --all      # 基线 + 10 个臂一次跑完
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "experiments"))

from models.dataset import LongSequenceDataset, ternary_from_labels16   # noqa: E402
from models.phase6_model import Phase6TernaryModel                 # noqa: E402
from models.gated_fusion_model import GatedFusionModel             # noqa: E402
from models.late_fusion_model import LateFusionModel               # noqa: E402
from models.cross_attention_fusion_model import CrossAttentionFusionModel  # noqa: E402

TEST_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "test.csv")

# Phase 8 的全部臂；基线 p7d_delta 在 phase7 下
ARMS = [
    ("baseline_p7d_delta", "logs/phase7/p7d_delta/best_model.pt"),
    ("p8_pose",            "logs/phase8/p8_pose/best_model.pt"),
    ("p8_pose_semantic",   "logs/phase8/p8_pose_semantic/best_model.pt"),
    ("p8_pose_accel",      "logs/phase8/p8_pose_accel/best_model.pt"),
    ("p8_flow",            "logs/phase8/p8_flow/best_model.pt"),
    ("p8_gated_pose",      "logs/phase8/p8_gated_pose/best_model.pt"),
    ("p8_gated_flow",      "logs/phase8/p8_gated_flow/best_model.pt"),
    ("p8_crossattn_pose",  "logs/phase8/p8_crossattn_pose/best_model.pt"),
    ("p8_late_flow",       "logs/phase8/p8_late_flow/best_model.pt"),
    ("p8a_accel",          "logs/phase8/p8a_accel/best_model.pt"),
    ("p8a_accel_reg",      "logs/phase8/p8a_accel_reg/best_model.pt"),
]


def classify(sd):
    ks = set(sd.keys())
    if "alpha_raw" in ks:
        return "late"
    if "gate_proj.weight" in ks:
        return "gated"
    if "cross_attn.in_proj_weight" in ks:
        return "crossattn"
    if "input_proj.weight" in ks:
        return "concat"
    raise ValueError(f"无法判别架构，键样例: {sorted(ks)[:8]}")


def build_model(kind, sd, args, win):
    """由 state_dict 形状 + args 推出维度并构建模型。

    维度**不能靠类默认值**：本项目已两次因此报错——
      · hidden 必须从 head.weight 的 **shape[1]** 取（shape[0] 是 num_classes=3，
        会让 PositionalEncoding 因奇数 d_model 报错）；
      · max_len 必须 = window_size + 10（训练时的实际值，默认 100 会与
        ckpt 里的 pos_encoder.pe 形状不符 → size mismatch）。
    """
    hw = sd["head.weight"] if "head.weight" in sd else sd["main_head.weight"]
    hidden = hw.shape[1]
    max_len = int(args.get("max_len") or (win + 10))
    # 也优先信 ckpt 里 pe 的真实形状（最可靠）
    for k in ("pos_encoder.pe", "main_pos.pe"):
        if k in sd:
            max_len = sd[k].shape[1]
            break
    if kind == "concat":
        in_dim = sd["input_proj.weight"].shape[1]
        m = Phase6TernaryModel(input_dim=in_dim, hidden_dim=hidden,
                               decoder_type=args.get("decoder_type", "transformer"),
                               causal=False, num_classes=3, max_len=max_len)
        return m, {"input_dim": in_dim, "hidden": hidden, "max_len": max_len}
    main_dim = sd["main_proj.weight"].shape[1]
    aux_dim = sd["aux_proj.weight"].shape[1]
    if kind == "gated":
        m = GatedFusionModel(main_dim=main_dim, aux_dim=aux_dim, hidden_dim=hidden,
                             max_len=max_len)
    elif kind == "crossattn":
        # ⚠ CrossAttentionFusionModel 内部做 `Linear(aux_dim*2, hidden)` —— 它自己在
        #   forward 里把 aux 与 Δaux 拼起来（见 train_cross_attention_fusion.py 首行注释：
        #   "aux 在模型内显式追加 Δ姿态（速度），不 concat"）。
        #   故 ckpt 里 aux_proj 的 in_features 是 **2×aux_dim**，要除以 2 传回构造函数。
        #   gated / late 无此约定（分别是 Linear(aux_dim,·) 与 Linear(aux_dim,aux_hidden)）。
        assert aux_dim % 2 == 0, f"crossattn 的 aux_proj in_features 应为偶数，实得 {aux_dim}"
        m = CrossAttentionFusionModel(main_dim=main_dim, aux_dim=aux_dim // 2,
                                      hidden_dim=hidden, max_len=max_len)
    else:
        aux_hidden = sd["aux_proj.weight"].shape[0]
        m = LateFusionModel(main_dim=main_dim, aux_dim=aux_dim,
                            hidden_dim=hidden, aux_hidden=aux_hidden, max_len=max_len)
    return m, {"main_dim": main_dim, "aux_dim": aux_dim, "hidden": hidden, "max_len": max_len}


def eval_one(name, ckpt_path, device, dump_windows=True):
    ck = torch.load(os.path.join(BASE, ckpt_path), weights_only=False, map_location="cpu")
    sd = ck["model_state_dict"]
    raw_args = ck.get("args", {})
    if not isinstance(raw_args, dict):
        raw_args = vars(raw_args)

    win = int(raw_args.get("window_size", 64))
    stride = int(raw_args.get("stride", 8))
    kind = classify(sd)
    model, dims = build_model(kind, sd, raw_args, win)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(f"{name}: state_dict 有 {len(unexpected)} 个意外键 {list(unexpected)[:5]}")
    model = model.to(device).eval()

    use_diff = bool(raw_args.get("use_diff", False))
    use_accel = bool(raw_args.get("use_accel", False))
    pose_npz = raw_args.get("pose_npz")
    pose_path = os.path.join(BASE, pose_npz) if pose_npz else None
    if pose_path and not os.path.exists(pose_path):
        raise FileNotFoundError(f"{name}: 找不到 {pose_path}")

    ds = LongSequenceDataset(
        npz_path=os.path.join(BASE, raw_args.get("frame_npz", "data/omnifall_dinov2_giant_frame.npz")),
        split_csv_path=TEST_CSV, window_size=win, stride=stride,
        use_diff=use_diff, use_accel=use_accel, pose_npz_path=pose_path,
        return_aux_separately=(kind != "concat"),   # 融合模型要单独的 aux
    )
    # 维度自检：不一致会当场报错，而不是静默跑出错误结果
    if kind == "concat":
        assert ds.feature_dim == dims["input_dim"], \
            f"{name}: 数据集 feature_dim={ds.feature_dim} 与 ckpt input_dim={dims['input_dim']} 不符"
    else:
        assert ds.base_feature_dim * (2 if use_diff else 1) + (ds.base_feature_dim if use_accel else 0) == dims["main_dim"], \
            f"{name}: main_dim 不符（数据集 {ds.base_feature_dim}×(1+diff+accel) vs ckpt {dims['main_dim']}）"
    print(f"  [{name}] 架构={kind} {dims} 窗口={len(ds)} (T={win}, stride={stride}, "
          f"diff={use_diff}, accel={use_accel}, pose={os.path.basename(pose_path) if pose_path else '无'})")

    # 逐视频汇总：按窗口收集，供两种协议复算
    #
    # ⚠ 这里有个曾把结果搞错的坑：dataset 返回的 `ternary_labels` 是**窗口的 T 帧**标签，
    #   不是整片 80 帧的。若拿它当整片 gt，再去切 [0:T]/[8:8+T]/[16:16+T]，
    #   会得到 64/56/48 帧（长度 168 而非 192），**静默产出错误的 gt**。
    #   整片标签必须从 NPZ 按视频起始位置取（window_offset==0 即新视频的第一个窗口）。
    pv = {}
    n_win = (80 - win) // stride + 1
    cur = None
    for i in range(len(ds)):
        npz_start, off = ds.window_index[i]
        if off == 0:                      # 新视频的第一个窗口
            full16 = ds.labels_16_all[npz_start: npz_start + 80]
            rel = str(ds.video_windows[i // n_win][0]) if hasattr(ds, "video_windows")                 else f"vid{npz_start}"
            cur = rel
            pv[cur] = {"ternary_gt": ternary_from_labels16(full16).astype(np.int64),
                       "win_probs": []}
        item = ds[i]
        feats = item["features"].unsqueeze(0).to(device)
        with torch.no_grad():
            if kind == "concat":
                logits = model(feats)
            else:
                aux = item["aux_features"].unsqueeze(0).to(device)
                logits = model(feats, aux)
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()   # (T,3)
        pv[cur]["win_probs"].append((off // stride, probs.astype(np.float32)))

    # 两种协议都保留：
    #   voted  —— 按窗口位置铺开、重叠处取均值 + argmax（与 E1 侧的 pv 口径一致，供跨阶段比较）
    #   flat   —— 所有窗口**原样拼接**的 gt/pred/probs（Phase 8 自己报告 test_results 用的协议，
    #             保留它才能把本脚本的结果与 Phase 8 已发表的数字**对齐验证**）
    out = {}
    for key, v in pv.items():
        sums = np.zeros((80, 3), dtype=np.float64)
        cnts = np.zeros(80, dtype=np.float64)
        for w, pr in v["win_probs"]:
            s = w * stride
            sums[s:s + win] += pr
            cnts[s:s + win] += 1
        voted = (sums / np.maximum(cnts[:, None], 1)).astype(np.float32)
        fgt = np.concatenate([v["ternary_gt"][w * stride: w * stride + win]
                              for w, _ in v["win_probs"]]).astype(np.int64)
        fpr = np.concatenate([pr for _, pr in v["win_probs"]], axis=0).astype(np.float32)
        out[key] = {
            "ternary_gt": v["ternary_gt"].astype(np.int64),
            "probs": voted,                                  # voted 口径（80×3）
            "flat_gt": fgt,                                  # flatten 口径（3T,）
            "flat_pred": fpr.argmax(-1).astype(np.int64),
            "flat_probs": fpr,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="单个 ckpt 路径")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true", help="基线 + 全部 Phase 8 臂")
    ap.add_argument("--out_dir", default=os.path.join(BASE, "logs", "phase8", "probs"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.all:
        todo = [(n, os.path.join(BASE, p)) for n, p in ARMS
                if os.path.exists(os.path.join(BASE, p))]
        missing = [n for n, p in ARMS if not os.path.exists(os.path.join(BASE, p))]
        if missing:
            print(f"[WARN] 以下臂的 ckpt 不存在，跳过: {missing}")
    else:
        assert args.ckpt, "给 --ckpt 或 --all"
        todo = [(args.name or os.path.basename(os.path.dirname(args.ckpt)), args.ckpt)]

    print(f"设备 {args.device}；待评 {len(todo)} 个臂")
    for name, path in todo:
        out_pv = eval_one(name, os.path.relpath(path, BASE), args.device)
        dst = args.out or os.path.join(args.out_dir, f"{name}.pt")
        torch.save(out_pv, dst)
        print(f"    → {os.path.relpath(dst, BASE)}（{len(out_pv)} 视频）")


if __name__ == "__main__":
    main()
