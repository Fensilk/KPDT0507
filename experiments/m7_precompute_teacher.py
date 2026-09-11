"""方法7(蒸馏) teacher 软标签预计算：E1@2000 在 e1_2000 训练视频的 stride16 窗口上前向。

窗口顺序 = [ (video, ws) for video in e1_2000/train.csv 顺序 for ws in (0,16) ]（4000 窗口）。
产出 data/m7_teacher_logits.npz: {logits:(4000,64,3) fp32, videos:(4000,) obj, ws:(4000,) int}
学生(本地)按同一枚举对齐取 teacher logits 做 KL 蒸馏。
"""
import os, sys
import numpy as np, torch, pandas as pd, cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))
sys.path.insert(0, os.path.join(BASE, "experiments"))
from av_decode import read_rgb_frames
from models.phase9_e1_model import E1Model
from experiments.train_p9e1 import preprocess_windows

VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")
TRAIN_CSV = os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "e1_2000", "train.csv")
CKPT = "/root/e1_out2/best_model.pt"
OUT = os.path.join(BASE, "data", "m7_teacher_logits.npz")
WINDOW, STRIDE, N_FRAMES, SIZE = 64, 16, 80, (224, 224)


def main():
    paths = list(pd.read_csv(TRAIN_CSV)["path"].str.strip())
    windows = [(v, ws) for v in paths for ws in range(0, N_FRAMES - WINDOW + 1, STRIDE)]
    n_win = len(windows)
    print(f"[0] 窗口数 {n_win} = {len(paths)} 视频 × 2 窗口")

    # 断点续跑：若已有部分 npz，读回并跳过已填充的视频（每个视频 2 个窗口都非零=已完成）
    logits_all = np.zeros((n_win, WINDOW, 3), dtype=np.float32)
    vids = [w[0] for w in windows]
    wss = [w[1] for w in windows]
    done_videos = set()
    if os.path.exists(OUT):
        old = np.load(OUT, allow_pickle=True)
        nz = (np.abs(old["logits"]).sum(axis=(1, 2)) > 0)
        # 视频级的 done：该视频两个窗口都非零
        for k in range(0, n_win, 2):
            if nz[k] and nz[k + 1]:
                done_videos.add(windows[k][0])
                logits_all[k] = old["logits"][k]; logits_all[k + 1] = old["logits"][k + 1]
        print(f"  [resume] 已从部分 npz 恢复 {len(done_videos)}/{len(paths)} 视频")

    print("[1] 加载 E1@2000 teacher ...")
    model = E1Model.from_checkpoint(CKPT, device="cuda")
    model.eval()

    n_ok = len(done_videos)
    n_fail = 0
    for vi, v in enumerate(paths):
        if v in done_videos:
            continue
        try:
            mp4 = os.path.join(VIDEO_ROOT, v + ".mp4")
            if not os.path.exists(mp4):
                print(f"  [skip] {v} mp4 缺失"); done_videos.add(v); continue
            fr = read_rgb_frames(mp4, N_FRAMES)
            imgs = np.stack([cv2.resize(f, SIZE, interpolation=cv2.INTER_LINEAR)
                             for f in fr]).astype(np.uint8)
            base = vi * 2  # 第 vi 个视频的窗口占 [vi*2, vi*2+2)
            for j, ws in enumerate(range(0, N_FRAMES - WINDOW + 1, STRIDE)):
                win = imgs[ws:ws + WINDOW]
                x = preprocess_windows(torch.from_numpy(win).float().unsqueeze(0), "cuda")
                with torch.no_grad():
                    lg = model(x)
                logits_all[base + j] = lg[0].float().cpu().numpy()
            done_videos.add(v)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  [WARN] 视频 {v} 失败: {type(e).__name__}: {e}（跳过，计入失败 {n_fail}）")
        if (vi + 1) % 50 == 0:
            print(f"  [{vi+1}/{len(paths)}] 完成 {n_ok} 失败 {n_fail}")
            np.savez(OUT, logits=logits_all, videos=np.array(vids, dtype=object), ws=np.array(wss))
    np.savez(OUT, logits=logits_all, videos=np.array(vids, dtype=object), ws=np.array(wss))
    print(f"[2] 完成: {n_ok} 视频, 失败 {n_fail} → {OUT}  logits {logits_all.shape}")
    print("  (若 n_ok<2000，重跑一次会断点续传跳过已完成视频)")


if __name__ == "__main__":
    main()
