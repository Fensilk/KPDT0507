"""
E3 特征重抽：用 Stage1 微调后的 ViT-g（LoRA 合并回权重）重抽逐帧 CLS 特征，
产出与 step7 相同格式的新 NPZ（labels 复用现有 NPZ，保证与 baseline 同标签）。

用法（项目根目录，py310）：
    # 跑通（限 N 个视频验证管线）
    python experiments/extract_p9_features.py --limit 100
    # 全量
    python experiments/extract_p9_features.py
产出：
    data/p9_features_finetuned/{path}.pt  +  data/omnifall_dinov2_finetuned_frame.npz
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "preprocessing"))

from step7_extract_dinov2 import extract_dinov2_features, aggregate_to_npz  # noqa: E402
from train_p9_e3_stage1 import inject_lora  # noqa: E402

OLD_NPZ = os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz")
CKPT = os.path.join(BASE, "logs", "phase9", "e3_stage1", "finetuned_lora.pt")
PT_OUT = os.path.join(BASE, "data", "p9_features_finetuned")
NPZ_OUT = os.path.join(BASE, "data", "omnifall_dinov2_finetuned_frame.npz")
VIDEO_ROOT = os.path.join(BASE, "DATASET-omnifall", "data_files", "extracted")


def merge_lora(model):
    """把 LoRALinear 的 A/B 合并回原权重，替换为普通 Linear。"""
    def _merge_linear(module, name):
        mod = getattr(module, name)
        if isinstance(mod, nn.Linear):
            return  # 未注入 LoRA
        delta = (mod.scaling * (mod.lora_B @ mod.lora_A)).to(mod.original.weight.dtype)
        new_w = mod.original.weight.data + delta
        new_lin = nn.Linear(mod.original.in_features, mod.original.out_features,
                            bias=mod.original.bias is not None)
        new_lin.weight.data = new_w
        if mod.original.bias is not None:
            new_lin.bias.data = mod.original.bias.data
        setattr(module, name, new_lin)

    for layer in model.encoder.layer:
        attn = layer.attention.attention
        for name in ("query", "key", "value"):
            _merge_linear(attn, name)
        _merge_linear(layer.attention.output, "dense")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="跑通时只抽前 N 个视频")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModel

    print(f"[1] 加载 facebook/dinov2-giant + 注入 LoRA + 合并 ...")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(args.device)
    inject_lora(model, r=16)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["lora_state"], strict=False)
    # 只加载 LoRA 参数
    merge_lora(model)
    model = model.half().eval()

    print(f"[2] 读取旧 NPZ（标签 + 视频列表）...")
    d = np.load(OLD_NPZ, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    vsi = d["video_start_indices"]
    labels16 = d["labels_16"]; fl = d["fall_labels"]; fdl = d["fallen_labels"]

    n_videos = len(vp) if args.limit is None else min(args.limit, len(vp))
    os.makedirs(PT_OUT, exist_ok=True)
    n_ok = 0
    print(f"[3] 重抽特征（{n_videos} 视频）...")
    for i in range(n_videos):
        rel = vp[i]
        mp4 = os.path.join(VIDEO_ROOT, rel + ".mp4")
        if not os.path.exists(mp4):
            print(f"  [skip] {rel} mp4 不存在")
            continue
        st = int(vsi[i])
        feats = extract_dinov2_features(mp4, model, args.device, args.batch_size)
        if feats is None:
            continue
        n = len(feats)
        out_path = os.path.join(PT_OUT, rel + ".pt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({
            "features": torch.from_numpy(feats),
            "labels": torch.from_numpy(labels16[st:st + n].astype(np.int64)),
            "fall_label": torch.from_numpy(fl[st:st + n].astype(np.int64)),
            "fallen_label": torch.from_numpy(fdl[st:st + n].astype(np.int64)),
        }, out_path)
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n_videos}] 已抽 {n_ok}")

    print(f"[4] 聚合 NPZ ...")
    aggregate_to_npz(PT_OUT, NPZ_OUT)
    print(f"完成: {n_ok} 视频 → {NPZ_OUT}")
    print("下一步: python experiments/train_phase6.py --frame_npz data/omnifall_dinov2_finetuned_frame.npz ...")


if __name__ == "__main__":
    main()
