"""
AV1 视频帧解码器：优先 PyAV（自带完整 ffmpeg + AV1 软件解码），cv2 兜底。

背景：autodl 的 cv2 无法解码 OmniFall 的 AV1 视频（"Missing Sequence Header"），
但 PyAV 可以。此模块统一帧读取，供帧准备与特征重抽使用。

返回：RGB uint8 帧列表，每帧 (H, W, 3)。长度不足 n 时用最后一帧/黑帧填充到 n。
"""
import numpy as np


def read_rgb_frames(mp4: str, n: int = 80):
    """
    读取视频前 n 帧（RGB uint8，shape (H,W,3)）。

    优先 PyAV，失败回退 cv2。两者都失败返回空列表（调用方决定是否跳过）。
    """
    frames = _read_pyav(mp4, n)
    if not frames:
        frames = _read_cv2(mp4, n)
    # pad/trim 到恰好 n 帧
    if len(frames) < n:
        while len(frames) < n:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    return frames[:n]


def _read_pyav(mp4: str, n: int):
    try:
        import av
    except ImportError:
        return []
    try:
        container = av.open(mp4)
        frames = []
        try:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))  # (H,W,3) uint8
                if len(frames) >= n:
                    break
        finally:
            container.close()
        return frames
    except Exception:
        return []


def _read_cv2(mp4: str, n: int):
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return []
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) >= n:
                break
    finally:
        cap.release()
    return frames
