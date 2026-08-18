"""
冗余探针（predictability probe）：冻结的 ViT-g 特征是否已隐式编码姿态？

背景：Phase 8 结论把"姿态集成失败"归因于"姿态信号与 ViT-g 冗余（ViT-g 已隐式编码姿态）"，
但这是一条**推断**——由"集成没帮助"反推而来，从未直接测量。本脚本把这条推断变成可测量的 R²。

判据：从冻结的 ViT-g 1536d 特征线性（Ridge）回归预测姿态，在留出的 test 视频帧上报告 R²
（R²=0 为"预测均值"的无技能基线，可负）。
  - R² 高 → ViT-g 已编码姿态 → "冗余"被直接证实 → 终止姿态集成合理。
  - R² 低 → 姿态信号并不冗余 → 集成失败是"机制"而非"信号"，Phase 8 总结论需改写。

关键细分（8d 语义特征不是同质信号，必须分开测）：
  - 静态 4d（头高比/躯干角/高宽比/髋高）：单帧量，可用单帧 ViT-g 线性探针测。
  - 速度 4d（上述量的帧间差分）：跨帧量，单帧 ViT-g 在构造上无法编码；
    冗余问题应问"ViT-g 的 Δframe 是否已编码姿态速度"（baseline 已含 Δframe），
    故单独用"ViT-g Δframe → 姿态速度"探针测。

对照（校准 R² 可信度）：
  - ceiling：原始 99d 姿态 → 语义静态 4d（应高，证明探针容量足够）。
  - 非线性 MLP：排除"姿态被 ViT-g 非线性编码、线性探针低估"的可能。

用法：
    python experiments/probe_redundancy.py
"""
import os, sys, json
# Windows 控制台默认 GBK，无法编码 R²/≈ 等字符；统一转 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

VITG_NPZ = "data/omnifall_dinov2_giant_frame.npz"
POSE_NPZ = "data/omnifall_pose_frame.npz"
SEMANTIC_NPZ = "data/omnifall_pose_semantic.npz"
SPLITS = "DATASET-omnifall/splits/syn/random"
OUT_JSON = "logs/phase8/probe_redundancy.json"

SEED = 0
TRAIN_SUBSAMPLE = 80000   # Ridge 训练帧数
TEST_SUBSAMPLE = 20000    # Ridge 测试帧数
DIFF_TRAIN = 50000        # Δframe 探针训练帧数
DIFF_TEST = 15000
MLP_TRAIN = 30000         # MLP 训练帧数
MLP_TEST = 10000
RIDGE_ALPHA = 1.0

STATIC_NAMES = ["head_h_ratio", "torso_angle", "bbox_aspect", "hip_height"]
VEL_NAMES = ["head_h_ratio_v", "torso_angle_v", "bbox_aspect_v", "hip_height_v"]


def collect_frames(split, vp, vsi, vcc, skip_first=False):
    """按 split CSV（视频级）收集帧索引；skip_first=True 时排除每视频首帧
    （首帧无帧间差分 / 速度=0）。"""
    df = pd.read_csv(os.path.join(SPLITS, f"{split}.csv"))
    paths = set(df["path"].str.strip())
    frames = []
    for vi in range(len(vp)):
        base = str(vp[vi]).replace(".mp4", "")
        if base not in paths:
            continue
        ns = int(vsi[vi]); nf = int(vcc[vi])
        lo = 1 if skip_first else 0
        frames.extend(range(ns + lo, ns + nf))
    return np.asarray(frames, dtype=np.int64)


def subsample(idx, n, rng):
    return rng.choice(idx, size=min(n, len(idx)), replace=False)


def ridge_fit(x_tr, y_tr):
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(x_tr, y_tr)


def report(names, r2, tag):
    print(f"\n[{tag}]  test R²")
    for name, r in zip(names, r2):
        print(f"    {name:<16s} R²={r:+.3f}")
    print(f"    {'mean':<16s} R²={r2.mean():+.3f}")
    return dict(zip(names, r2.tolist()))


def main():
    rng = np.random.default_rng(SEED)

    print("[Probe] Loading NPZ (mmap)...")
    vitg = np.load(VITG_NPZ, allow_pickle=True, mmap_mode='r')
    pose = np.load(POSE_NPZ, allow_pickle=True, mmap_mode='r')
    sem = np.load(SEMANTIC_NPZ, allow_pickle=True, mmap_mode='r')

    vp = vitg["video_paths"]; vsi = vitg["video_start_indices"]; vcc = vitg["video_clip_counts"]
    V = vitg["features"]; P = pose["pose_features"]; S = sem["pose_features"]
    print(f"[Probe] ViT-g {V.shape} {V.dtype}, pose {P.shape}, semantic {S.shape}")

    train_frames = collect_frames("train", vp, vsi, vcc)
    test_frames = collect_frames("test", vp, vsi, vcc)
    print(f"[Probe] frames — train={len(train_frames):,}, test={len(test_frames):,}")

    tr = subsample(train_frames, TRAIN_SUBSAMPLE, rng)
    te = subsample(test_frames, TEST_SUBSAMPLE, rng)
    dtr = subsample(collect_frames("train", vp, vsi, vcc, skip_first=True), DIFF_TRAIN, rng)
    dte = subsample(collect_frames("test", vp, vsi, vcc, skip_first=True), DIFF_TEST, rng)
    mtr = subsample(train_frames, MLP_TRAIN, rng)
    mte = subsample(test_frames, MLP_TEST, rng)

    X_tr = V[tr].astype(np.float32); X_te = V[te].astype(np.float32)
    P_tr = P[tr].astype(np.float32); P_te = P[te].astype(np.float32)
    S_te = S[te].astype(np.float32)
    Sstatic_tr = S[tr][:, :4].astype(np.float32); Sstatic_te = S_te[:, :4]
    results = {}

    # ── 1. 静态冗余：ViT-g(单帧) → 语义静态 4d ──
    model = ridge_fit(X_tr, Sstatic_tr)
    r2 = r2_score(Sstatic_te, model.predict(X_te), multioutput="raw_values")
    results["vitg_to_static"] = report(STATIC_NAMES, r2, "1) 静态冗余  ViT-g(单帧1536d) -> 语义静态4d")
    results["vitg_to_static_mean"] = float(r2.mean())

    # ── 2. 静态 ceiling：原始姿态99d → 语义静态4d（探针容量校准，应高） ──
    ceil = ridge_fit(P_tr, Sstatic_tr)
    r2c = r2_score(Sstatic_te, ceil.predict(P_te), multioutput="raw_values")
    results["pose_to_static_ceiling"] = report(STATIC_NAMES, r2c, "2) ceiling  原始姿态99d -> 语义静态4d (应高)")
    results["pose_to_static_ceiling_mean"] = float(r2c.mean())

    # ── 3. 速度冗余：ViT-g Δframe → 姿态速度4d（baseline 已含 Δframe） ──
    Xd_tr = V[dtr].astype(np.float32) - V[dtr - 1].astype(np.float32)
    Xd_te = V[dte].astype(np.float32) - V[dte - 1].astype(np.float32)
    Svel_tr = S[dtr][:, 4:8].astype(np.float32)
    Svel_te = S[dte][:, 4:8].astype(np.float32)
    mvel = ridge_fit(Xd_tr, Svel_tr)
    r2v = r2_score(Svel_te, mvel.predict(Xd_te), multioutput="raw_values")
    results["vitg_diff_to_velocity"] = report(VEL_NAMES, r2v, "3) 速度冗余  ViT-g Δframe(1536d) -> 姿态速度4d")
    results["vitg_diff_to_velocity_mean"] = float(r2v.mean())

    # 速度 ceiling：原始姿态 Δframe99d → 姿态速度4d
    Pd_tr = P[dtr].astype(np.float32) - P[dtr - 1].astype(np.float32)
    Pd_te = P[dte].astype(np.float32) - P[dte - 1].astype(np.float32)
    cvel = ridge_fit(Pd_tr, Svel_tr)
    r2cv = r2_score(Svel_te, cvel.predict(Pd_te), multioutput="raw_values")
    results["pose_diff_to_velocity_ceiling"] = report(VEL_NAMES, r2cv, "3b) ceiling  原始姿态Δframe99d -> 姿态速度4d (应高)")
    results["pose_diff_to_velocity_ceiling_mean"] = float(r2cv.mean())

    # ── 4. 非线性探针：MLP ViT-g(单帧) → 语义静态4d ──
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256,), max_iter=30, batch_size=256,
                     early_stopping=True, validation_fraction=0.1,
                     random_state=SEED, verbose=False),
    )
    Xm_tr = V[mtr].astype(np.float32); Xm_te = V[mte].astype(np.float32)
    Sm_tr = S[mtr][:, :4].astype(np.float32); Sm_te = S[mte][:, :4].astype(np.float32)
    mlp.fit(Xm_tr, Sm_tr)
    r2m = r2_score(Sm_te, mlp.predict(Xm_te), multioutput="raw_values")
    results["vitg_mlp_to_static"] = report(STATIC_NAMES, r2m, "4) 非线性探针  MLP ViT-g -> 语义静态4d (排除非线性编码)")
    results["vitg_mlp_to_static_mean"] = float(r2m.mean())

    # ── 5. 结论 ──
    torso = results["vitg_to_static"]["torso_angle"]
    hip = results["vitg_to_static"]["hip_height"]
    static_mean = results["vitg_to_static_mean"]
    ceiling_mean = results["pose_to_static_ceiling_mean"]
    vel_mean = results["vitg_diff_to_velocity_mean"]
    vel_ceiling = results["pose_diff_to_velocity_ceiling_mean"]

    lines = []
    lines.append(f"静态冗余：ViT-g→静态语义 mean R²={static_mean:+.3f}，ceiling={ceiling_mean:+.3f} "
                 f"（ViT-g 达 ceiling 的 {static_mean/max(ceiling_mean,1e-6)*100:.0f}%）")
    lines.append(f"  判别性维度：torso_angle R²={torso:+.3f}（ViT-g 几乎不含躯干倾角），"
                 f"hip_height R²={hip:+.3f}")
    lines.append(f"速度冗余：ViT-g Δframe→姿态速度 mean R²={vel_mean:+.3f}，ceiling={vel_ceiling:+.3f}")

    if torso < 0.15 and static_mean < 0.5:
        verdict = ("冗余不成立：ViT-g 只编码粗粒度外观（bbox/头高比），"
                   "未编码判别性姿态（躯干角≈0、髋高弱）→ 姿态集成失败的根因不是冗余，"
                   "Phase 8 'ViT-g 已隐式编码姿态' 的推断被证伪，需改写")
    elif static_mean < 0.5:
        verdict = "部分冗余：静态姿态被部分编码，但判别性维度（躯干角/髋高）编码弱，冗余解释存疑"
    else:
        verdict = "冗余证实：ViT-g 已强编码姿态语义，终止姿态集成合理"
    results["verdict"] = verdict
    results["summary"] = lines

    print(f"\n{'='*72}")
    for ln in lines:
        print("  " + ln)
    print(f"结论：{verdict}")
    print(f"{'='*72}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Probe] 已保存 {OUT_JSON}")


if __name__ == "__main__":
    main()
