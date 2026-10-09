# -*- coding: utf-8 -*-
"""对一批 ckpt 在**新 val**（val.csv，stride 16）上重评，供 Phase 11 选点规则补执行。

═══ 为什么有这个脚本 ═══

规划 §6.3 的选点规则（val 平滑值 top-k → 全量 test → 择一）**从未被执行**；
W2 的判定用的是 `best_model.pt` = **val 单点极大值**，而那正是 §6.3 第 1 条明文禁止的做法。
要把规则补执行到"两侧对称"，需要锚点与各臂在**同一 val 口径**下可比 —— 本脚本就是给
每个 ckpt 算出这个分数，**零重训**（三个 run 的已存 ckpt 都落在 val 检查步上）。

═══ 冻结的口径（改动即作废，见 docs/1006…增补-选点规则冻结.md）═══

  1. val 集   = `splits/syn/random/val.csv` 全量 1200（不是原先的 150 子集）
  2. 窗口     = window 64 / **stride 16** → 每视频 2 窗（[0:64] 与 [16:80]），覆盖帧 0–79
                （旧 val 用 stride 80 → 每视频 1 窗，**只覆盖帧 0–63**，尾部 16 帧从不被评）
  3. 聚合     = 每帧**只计一次**，重叠区（帧 16–63）归**第一个窗口**
  4. 指标     = 三类 macro F1（与 train_p9e1.val_check 同口径），另存 per-class P/R

═══ 三个刻意的设计 ═══

  * **整批 val 只在内存里解码一次**，12 个 ckpt 共用。若按 Dataset 逐窗口取，
    每个视频会被解码 `窗口数` 次 —— 实测这一步占了单 ckpt 耗时的一半（26.6 s/20 视频里
    解码 13.2 s），而本机 RAM 754 GB，14.5 GB 的 val 帧完全放得下。5.3 h → ≈2.75 h。
  * **开关从 ckpt 的 args 里读**（`use_boundary_head` / `pose_npz`），不靠命令行传。
    附录 A.3 的教训（pose 通路"三处漏接"，连爆两次各赔一轮 8.8 h）正是手工传参造成的。
    ⚠ 且 `Mp4WindowCache` 的第 3 个返回值是**互斥槽位**（boundary 标签 **或** pose 特征），
    不能一律当 aux —— 冒烟当场抓到过这个错。
  * 复用 `train_p9e1` 的 `decode_mp4_frames` / `ternary` / `preprocess_windows` 与
    `Mp4WindowCache` 的元数据，**不写第二份副本**（本项目有"副本间漂移"的教训）。

═══ 附带产物 ═══

  另存 `*.pv.pt`（逐视频逐帧 preds/labels）。有它之后，
  **"val 扩到多大才够"的剂量-响应**（n=150/300/600/1200）可离线复算，不必重跑 GPU。

用法（项目根目录，py310）：
    python -u analysis/reeval_val_ckpts.py --out logs/phase11/val_probe/reval.json \\
        logs/phase9/e1_full/best_model.pt logs/phase11/w2_1a/best_model.pt ...
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

# Windows 控制台是 GBK，直接 print ✅/❌ 会 UnicodeEncodeError（附录 A.2 #3）。
# 进程启动后再设 PYTHONIOENCODING 无效，必须 reconfigure 已建好的 stdout。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "experiments"))
sys.path.insert(0, os.path.join(BASE, "models"))

VAL_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "random", "val.csv")
NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")


def load_ckpt_switches(path):
    """从 ckpt 的 args 里取回训练时用的开关 —— 不靠命令行传，避免 A.3 的漏接。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    if not isinstance(a, dict):
        a = vars(a)
    return {
        "use_boundary": bool(a.get("use_boundary_head", False)),
        "pose_npz": a.get("pose_npz") or None,
    }


def build_val_cache(ds, T, log_every=50):
    """把 val 全部解码进内存 + 取好标签。返回 list，每项 {rel, frames, wins:[(ws, y, aux)]}。

    只做一次；后续每个 ckpt 复用同一份，避免"每 ckpt 每窗口各解码一次"。
    """
    cache, t0 = [], time.time()
    for k, (rel, st) in enumerate(ds.videos):
        mp4 = os.path.join(ds.video_root, rel + ".mp4")
        frames = T.decode_mp4_frames(mp4)
        if frames is None:
            raise RuntimeError(f"[val cache] 解码失败: {rel}")
        wins = []
        for ws in range(0, 80 - ds.window + 1, ds.stride):
            y = np.asarray(T.ternary(ds.labels16[st + ws: st + ws + ds.window]), dtype=np.int64)
            aux = None
            if ds.pose_all is not None:
                aux = np.ascontiguousarray(ds.pose_all[st + ws: st + ws + ds.window]).astype(np.float32)
            wins.append((ws, y, aux))
        cache.append({"rel": rel, "frames": frames, "wins": wins})
        if (k + 1) % log_every == 0:
            print(f"    [decode] {k+1}/{len(ds.videos)} 视频  {time.time()-t0:.0f}s", flush=True)
    print(f"  val 解码完成：{len(cache)} 视频 × {len(cache[0]['wins'])} 窗  "
          f"{time.time()-t0:.0f}s", flush=True)
    return cache


@torch.no_grad()
def score_ckpt(path, cache, device, T, window=64, batch=8, log_every=100):
    """返回 (per_video, summary)。按 F3：每帧只计一次，重叠区归第一个窗口。"""
    sw = load_ckpt_switches(path)
    model = T.E1Model.from_checkpoint(path, device=device)
    model.eval()

    # 展平成窗口列表（frames 切片是视图，不复制）
    items = []
    for v in cache:
        for ws, y, aux in v["wins"]:
            items.append((v["rel"], ws, y, v["frames"][ws:ws + window], aux))

    per_frame, t0 = {}, time.time()
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        x = torch.stack([torch.from_numpy(it[3]) for it in chunk]).to(device)
        x = T.preprocess_windows(x, device)                 # (B,T,3,H,W) fp32
        aux = None
        # ⚠ 必须按**本 ckpt** 的开关决定是否传 aux，不能按 dataset 是否带 pose：
        #   val cache 是按"任一臂用 pose"建的，对没有 pose 的锚点传 aux 会把 input_dim
        #   从 3072 变成 3084 → mat1/mat2 形状不符（冒烟第二次抓到的同类漏接，见 A.3）。
        if sw["pose_npz"] and chunk[0][4] is not None:
            aux = torch.from_numpy(np.stack([it[4] for it in chunk])).to(device)
        out = model(x, aux)
        if sw["use_boundary"]:
            out = out[0]                                     # 丢掉 boundary logits
        pred = out.argmax(-1).cpu().numpy()                  # (B,T)
        for k, (rel, ws, y, _fr, _aux) in enumerate(chunk):
            slot = per_frame.setdefault(rel, {})
            for t in range(pred.shape[1]):
                fidx = ws + t
                if fidx not in slot:                         # ★ 冻结规则：第一个窗口优先
                    slot[fidx] = (int(y[t]), int(pred[k, t]))
        if (s // batch) % log_every == 0:
            print(f"    [{os.path.basename(os.path.dirname(path))}] "
                  f"{min(s+batch, len(items))}/{len(items)} 窗  {time.time()-t0:.0f}s", flush=True)

    per_video = []
    for rel, slot in per_frame.items():
        fr = sorted(slot)
        per_video.append({"rel": rel,
                          "y": np.array([slot[f][0] for f in fr], dtype=np.int64),
                          "p": np.array([slot[f][1] for f in fr], dtype=np.int64),
                          "frames": fr})
    per_video.sort(key=lambda d: d["rel"])

    y = np.concatenate([d["y"] for d in per_video])
    p = np.concatenate([d["p"] for d in per_video])
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    pr = precision_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    rc = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    summary = {
        "ckpt": path,
        "switches": {"use_boundary": sw["use_boundary"], "pose_npz": sw["pose_npz"]},
        "n_videos": len(per_video), "n_frames": int(len(y)),
        "frames_per_video": sorted({len(d["y"]) for d in per_video}),
        "val_avg_f1": float(f1.mean()),
        "val_fall_prec": float(pr[0]), "val_fall_rec": float(rc[0]),
        "val_fallen_prec": float(pr[1]), "val_fallen_rec": float(rc[1]),
        "val_normal_prec": float(pr[2]), "val_normal_rec": float(rc[2]),
        "secs": round(time.time() - t0, 1),
    }
    return per_video, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", default=os.path.join(BASE, "logs", "phase11", "val_probe", "reval.json"))
    ap.add_argument("--stride", type=int, default=16, help="冻结=16（2 窗，覆盖 0–79）")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit_videos", type=int, default=None)
    # P2 离线选点第 ① 段要用打散后的子集 CSV（不是"前 N 行"——val.csv 按类排序，
    # 取前 N 会缺 standing/stand_up/walking）。由 analysis/make_val_subset.py 生成。
    ap.add_argument("--val_csv", default=VAL_CSV, help="默认全量 val.csv；子集阶段传子集 CSV")
    args = ap.parse_args()
    val_csv = args.val_csv

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pv_path = os.path.splitext(args.out)[0] + ".pv.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  val={val_csv}  stride={args.stride}  ckpt 数={len(args.ckpts)}",
          flush=True)

    import train_p9e1 as T
    from phase9_e1_model import E1Model
    T.E1Model = E1Model

    # val cache 建一次给所有 ckpt 共用，所以只要**任一** ckpt 用 pose 就把 pose 读进来
    # （peek 哪个窗口需要 aux 由 score_ckpt 按各 ckpt 自己的开关决定，见那里的注释）。
    all_sw = {p: load_ckpt_switches(p) for p in args.ckpts}      # 每个 ckpt 只读一次
    any_pose = next((s["pose_npz"] for s in all_sw.values() if s["pose_npz"]), None)
    ds = T.Mp4WindowCache(val_csv, NPZ, window=args.window, stride=args.stride,
                          limit_videos=args.limit_videos, pose_npz=any_pose)
    print(f"窗口总数={len(ds)}  视频数={len(ds.videos)}  pose={'有' if any_pose else '无'}",
          flush=True)

    cache = build_val_cache(ds, T)

    results, pv_all = [], {}
    for path in args.ckpts:
        sw = load_ckpt_switches(path)
        print(f"\n=== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}  "
              f"boundary={sw['use_boundary']} pose={'有' if sw['pose_npz'] else '无'}", flush=True)
        pv, summ = score_ckpt(path, cache, device, T, window=args.window, batch=args.batch)
        results.append(summ)
        pv_all[path.replace(os.sep, "/")] = pv
        print(f"  → val_avg_f1={summ['val_avg_f1']:.4f}  帧数={summ['n_frames']}  "
              f"每视频帧数={summ['frames_per_video']}  耗时={summ['secs']}s", flush=True)
        torch.cuda.empty_cache()      # score_ckpt 返回后模型已出作用域，清一次显存碎片

        # ★ 每完成一个 ckpt 就落盘（终止预案）：全程 ~2.5 h，中途崩掉不能把已算的全丢
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"spec": {"val_csv": val_csv, "stride": args.stride,
                                "window": args.window,
                                "agg": "每帧只计一次，重叠区归第一个窗口",
                                "done": len(results), "total": len(args.ckpts)},
                       "results": results}, f, ensure_ascii=False, indent=2)
        torch.save(pv_all, pv_path)
        print(f"  [checkpoint] 已落盘 {len(results)}/{len(args.ckpts)} → {args.out}", flush=True)

    print(f"\nALL_CKPTS_DONE 已写出 {args.out}\n                {pv_path}", flush=True)


if __name__ == "__main__":
    main()
