"""
Step 9: 从原始 33 关键点姿态特征派生语义特征。

从 data/omnifall_pose_frame.npz 的 99 维（33关键点×xyz）派生 8 维语义特征：
  1. head_height_ratio: 头部相对髋部的高度（除以躯干长度，尺度不变）
  2. torso_angle: 躯干倾角（肩-髋连线 vs 垂线，弧度）
  3. bbox_aspect: 人体框高宽比（站立>1，躺卧<1）
  4. hip_height: 髋部高度（图像归一化 y，越低越接近倒地）
  5-8. 上述 4 个量的速度（帧间差分，捕获"如何到达地面"）

输出 data/omnifall_pose_semantic.npz (960000, 8)，与原始 NPZ 对齐。

用法：
    python preprocessing/step9_derive_pose.py
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_POSE = os.path.join(ROOT, "data", "omnifall_pose_frame.npz")
OUT = os.path.join(ROOT, "data", "omnifall_pose_semantic.npz")

# MediaPipe 关键点索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

EPS = 1e-6


def derive_frame(lm):
    """
    从一帧的 33 关键点 × (x,y,visibility) 派生 4 个语义特征。
    lm: (33, 3) float32
    返回: (4,) float32
    """
    nose = lm[NOSE]
    shoulder_mid = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2.0
    hip_mid = (lm[L_HIP] + lm[R_HIP]) / 2.0

    # 躯干向量：肩 -> 髋
    torso_vec = shoulder_mid[:2] - hip_mid[:2]  # (dx, dy)
    torso_length = np.linalg.norm(torso_vec)

    # 1. head_height_ratio = (髋 y - 鼻 y) / 躯干长度
    #    站立时头部在髋上方，比值为正（约 1.5~2）；躺卧时接近 0
    head_height = hip_mid[1] - nose[1]
    head_height_ratio = head_height / (torso_length + EPS)

    # 2. torso_angle = 躯干与垂线的夹角（弧度）
    #    atan2(水平分量, 垂直分量)；站立≈0，躺卧≈±π/2
    torso_angle = np.arctan2(torso_vec[0], -torso_vec[1])

    # 3. bbox_aspect = 人体框高/宽（y 是垂直高度，x 是水平宽度）
    xs = lm[:, 0]; ys = lm[:, 1]
    body_w = xs.max() - xs.min()  # 水平范围
    body_h = ys.max() - ys.min()  # 垂直范围
    bbox_aspect = body_h / (body_w + EPS)  # 站立>1（竖直长），躺卧<1（横向长）

    # 4. hip_height = 髋部 y（图像归一化，越小越接近地面）
    hip_height = hip_mid[1]

    feat = np.array([head_height_ratio, torso_angle, bbox_aspect, hip_height],
                    dtype=np.float32)

    # 处理退化帧（躯干长度≈0，即 MediaPipe 失败或关键点重合）
    if torso_length < EPS:
        feat = np.zeros(4, dtype=np.float32)

    # 裁剪到合理范围，抑制除零放大 / MediaPipe 跟踪失败的极端离群值
    feat[0] = np.clip(feat[0], -5.0, 5.0)     # head_height_ratio
    feat[1] = np.clip(feat[1], -np.pi, np.pi) # torso_angle（atan2 本就在此范围）
    feat[2] = np.clip(feat[2], 0.05, 10.0)    # bbox_aspect
    feat[3] = np.clip(feat[3], -0.5, 1.5)     # hip_height

    return feat


def main():
    os.chdir(ROOT)
    raw = np.load(RAW_POSE, allow_pickle=True, mmap_mode='r')
    pose_feats = raw["pose_features"]  # (960000, 99)
    video_clip_counts = raw["video_clip_counts"]  # 每个视频 80 帧

    n_frames = pose_feats.shape[0]
    print(f"原始姿态: {pose_feats.shape}")

    # 逐帧派生 base 特征
    base = np.zeros((n_frames, 4), dtype=np.float32)
    for i in range(n_frames):
        lm = pose_feats[i].reshape(33, 3)
        base[i] = derive_frame(lm)
        if (i + 1) % 100000 == 0:
            print(f"  base 派生 {i+1:,}/{n_frames:,}")

    # 逐视频计算速度（帧间差分，首帧=0）
    vel = np.zeros_like(base)
    for vi in range(len(video_clip_counts)):
        nf = int(video_clip_counts[vi])
        ns = int(raw["video_start_indices"][vi])
        if nf > 1:
            vel[ns + 1:ns + nf] = base[ns + 1:ns + nf] - base[ns:ns + nf - 1]

    # 拼接 base + vel = 8 维
    semantic = np.concatenate([base, vel], axis=1).astype(np.float32)
    print(f"语义特征: {semantic.shape} (4 base + 4 velocity)")

    # 保存
    np.savez_compressed(
        OUT,
        pose_features=semantic,
        video_paths=raw["video_paths"],
        video_clip_counts=raw["video_clip_counts"],
        video_start_indices=raw["video_start_indices"],
    )
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"已保存: {OUT} ({size_mb:.1f} MB)")

    # 快速统计
    print("\n各特征统计（base 前 4 维）:")
    for name, idx in [("head_height_ratio", 0), ("torso_angle", 1),
                      ("bbox_aspect", 2), ("hip_height", 3)]:
        vals = semantic[:, idx]
        print(f"  {name:<18s} mean={vals.mean():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")


if __name__ == "__main__":
    main()
