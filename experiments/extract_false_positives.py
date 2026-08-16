"""
在本地运行：加载服务器导出的预测，找出 lying→event 假阳性，抽取帧供人工检查。

按「帧级」标准：GT 16类 ∈ {lying(6), lie_down(5)}，但模型预测 ∈ {fall(0), fallen(1)} 的帧，
按每视频误判帧数降序排序，抽取最严重的 N 个视频。

依赖：本地有 DATASET-omnifall/data_files/extracted/ 下的 mp4 视频，以及 ffmpeg。

用法（本地，无需 GPU）：
    python experiments/extract_false_positives.py
"""
import os, sys, json, subprocess, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREDS      = "analysis/diagnose_lying_event/test_preds.json"
VIDEO_ROOT = "DATASET-omnifall/data_files/extracted"
OUT_DIR    = "analysis/diagnose_lying_event"
SAMPLE_N   = 25        # 抽取视频数
FRAME_EVERY = 4        # 每 N 帧抽 1 帧（81 帧 → ~20 帧）
SEED       = 42

LIE_DOWN, LYING = 5, 6


def fp_frame_count(d):
    """GT ∈ {lying, lie_down} 但被预测为 fall/fallen 的帧数。"""
    lab = d["labels_16"]
    bp = d["bridge_pred"]
    return sum(1 for l, p in zip(lab, bp) if l in (LIE_DOWN, LYING) and p in (0, 1))


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE)

    with open(PREDS, "r", encoding="utf-8") as f:
        preds = json.load(f)
    print(f"[1] Loaded predictions for {len(preds)} videos")

    # 帧级假阳性计数，按降序排序
    candidates = []
    for vp, d in preds.items():
        n = fp_frame_count(d)
        if n > 0:
            mp4 = os.path.join(VIDEO_ROOT, vp + ".mp4")
            if os.path.exists(mp4):
                candidates.append((n, vp, mp4, d))
    candidates.sort(key=lambda x: -x[0])
    print(f"[2] 含 lying→event 假阳性的视频（本地有 mp4）: {len(candidates)}")

    # 取最严重的 N 个
    sampled = candidates[:SAMPLE_N]

    # 抽帧
    print(f"[3] 抽取误判最严重的 {len(sampled)} 个视频的帧到 {OUT_DIR}/...")
    summary = []
    for rank, (fp_n, vp, mp4, d) in enumerate(sampled, 1):
        video_dir = os.path.join(OUT_DIR, f"{rank:02d}_{vp.replace('/', '_')}")
        os.makedirs(video_dir, exist_ok=True)

        out_pattern = os.path.join(video_dir, "frame_%02d.png")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", mp4,
            "-vf", f"select=not(mod(n\\,{FRAME_EVERY}))",
            "-vsync", "vfr",
            out_pattern,
        ]
        subprocess.run(cmd, check=False)

        bp = d["bridge_pred"]
        lab = d["labels_16"]
        # 统计该视频的 GT 标签分布（用于判断"躺着分不清"还是"动作快"）
        from collections import Counter
        gt_dist = Counter(lab)
        summary.append({
            "rank": rank,
            "video_path": vp,
            "fp_frames": fp_n,
            "total_frames": len(bp),
            "gt_16class_dist": {int(k): v for k, v in gt_dist.items()},
            "pred_fall_frames": bp.count(0),
            "pred_fallen_frames": bp.count(1),
            "pred_normal_frames": bp.count(2),
        })
        print(f"    [{rank:02d}] {vp}  误判帧 {fp_n}/{len(bp)}  "
              f"pred(fall={bp.count(0)}, fallen={bp.count(1)})")

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. 抽取 {len(sampled)} 个误判最严重的视频。")
    print(f"帧在 {OUT_DIR}/<序号_视频名>/，摘要见 summary.json")


if __name__ == "__main__":
    main()
