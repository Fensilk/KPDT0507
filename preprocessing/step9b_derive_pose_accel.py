"""
Step 9b: 在 8 维派生姿态特征（4 base + 4 velocity）基础上追加加速度。

从 data/omnifall_pose_semantic.npz 的 8 维（4 base + 4 vel）派生出 12 维：
  1-4.   base   （head_height_ratio / torso_angle / bbox_aspect / hip_height）
  5-8.   vel    （上述 4 量的帧间差分）
  9-12.  accel  （上述 4 量的二阶差分 = 真实加速度）

accel[t] = base[t+1] - 2·base[t] + base[t-1]，逐视频计算（首尾帧=0）。

输出 data/omnifall_pose_semantic_accel.npz (960000, 12)，与原始 NPZ 对齐。

用法：
    python preprocessing/step9b_derive_pose_accel.py
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "data", "omnifall_pose_semantic.npz")
OUT = os.path.join(ROOT, "data", "omnifall_pose_semantic_accel.npz")


def main():
    os.chdir(ROOT)
    d = np.load(IN, allow_pickle=True, mmap_mode="r")
    semantic = d["pose_features"]              # (960000, 8) = 4 base + 4 vel
    base = semantic[:, :4]                      # 位置量
    vel = semantic[:, 4:8]                      # 速度量
    starts = d["video_start_indices"].astype(np.int64)
    counts = d["video_clip_counts"].astype(np.int64)

    n_frames = base.shape[0]
    print(f"输入: {semantic.shape} (4 base + 4 vel)")

    # 逐视频中心二阶差分
    accel = np.zeros_like(base, dtype=np.float32)
    for s, c in zip(starts, counts):
        if c < 3:
            continue
        seg = base[s:s + c]
        accel[s + 1:s + c - 1] = seg[2:] - 2.0 * seg[1:-1] + seg[:-2]

    out = np.concatenate([base, vel, accel], axis=1).astype(np.float32)
    print(f"输出: {out.shape} (4 base + 4 vel + 4 accel)")

    np.savez_compressed(
        OUT,
        pose_features=out,
        video_paths=d["video_paths"],
        video_clip_counts=d["video_clip_counts"],
        video_start_indices=d["video_start_indices"],
    )
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"已保存: {OUT} ({size_mb:.1f} MB)")

    # 快速统计
    names = ["head_height_ratio", "torso_angle", "bbox_aspect", "hip_height"]
    print("\n各特征 accel 统计:")
    for i, name in enumerate(names):
        a = accel[:, i]
        print(f"  accel[{name:<18s}] mean={a.mean():.5f} std={a.std():.5f} "
              f"max|a|={np.abs(a).max():.5f}")


if __name__ == "__main__":
    main()
