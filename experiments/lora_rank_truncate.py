"""LoRA 秩截断探针：把已训好的 E1 紧凑 ckpt 的 LoRA 更新截断到 rank k（零训练）。

用途：回答"**r=16 是否过剩**"。LoRALinear 的贡献恰是 scaling·(B@A) —— 一个精确的
rank-r 矩阵，故可对它做 SVD、只保留前 k 个奇异值重建，其余（时序头、主干）不变，
再交给未改动的 `eval_p9e1_dual.py` 评估。**只能向下**（k<r）：r=16 训出的解活在
16 维子空间里，无法外推到 32 维——r=32 只能真训。

两种用法：
    --report_only   只算每层 Δ 的奇异值能量分布（CPU、秒级、不写文件）。
                    若 top-k 能量比已≈1，即"r=16 过剩"的强证据，可先看它再决定要不要跑评估。
    --out ...       写出截断后的紧凑 ckpt（lora_rank=k），交给 eval_p9e1_dual.py。

scaling 陷阱：inject_lora(alpha=16) → scaling = 16/r。存 lora_rank=k 后重建模型会用
scaling_k = 16/k，故写入的因子必须**预除** scaling_k，否则整体被缩放 16/k 倍。

用法:
    python experiments/lora_rank_truncate.py --ckpt logs/phase9/e1_full/best_model.pt --report_only
    python experiments/lora_rank_truncate.py --ckpt <in.pt> --out <out.pt> --k 8
    python experiments/lora_rank_truncate.py --ckpt <in.pt> --out <out.pt> --k 16 --verify
"""
import argparse
import os
import sys

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

ALPHA = 16.0  # inject_lora 默认 alpha（E1Model 未覆写）


def _pairs(lora_state):
    """按前缀配对 lora_A / lora_B —— 返回 [(prefix, A, B), ...]。"""
    out = []
    for key in sorted(lora_state):
        if key.endswith(".lora_A"):
            pre = key[: -len(".lora_A")]
            bkey = pre + ".lora_B"
            if bkey not in lora_state:
                raise KeyError(f"{bkey} 缺失，无法配对")
            out.append((pre, lora_state[key], lora_state[bkey]))
    return out


def _svd(delta):
    return torch.linalg.svd(delta, full_matrices=False)   # U, S, Vh


def report(ckpt, ks):
    """打印每层 Δ 的奇异值能量分布（top-k 捕获比），全层聚合。"""
    rank = ckpt.get("lora_rank", 16)
    scaling = ALPHA / rank
    ps = _pairs(ckpt["lora_state"])
    print(f"[report] rank={rank} scaling={scaling} 层数(投影对)={len(ps)}")
    ratios = {k: [] for k in ks}
    s_curves = []
    for pre, A, B in ps:
        delta = scaling * (B @ A)
        _, S, _ = _svd(delta)
        energy = (S ** 2).sum()
        s_curves.append(S)
        for k in ks:
            ratios[k].append(float((S[:k] ** 2).sum() / energy))
    print(f"\n{'k':>4} | {'能量捕获比 均值':>16} | {'最小':>8} | {'最大':>8}")
    print("-" * 46)
    for k in ks:
        r = torch.tensor(ratios[k])
        print(f"{k:>4} | {r.mean():>16.6f} | {r.min():>8.6f} | {r.max():>8.6f}")
    # 全层平均奇异值谱。Δ=B@A 的秩 ≤ rank，故只有前 rank 个非零，其余恒为 0。
    S_mean = torch.stack(s_curves).mean(dim=0)
    share = (S_mean ** 2) / (S_mean ** 2).sum()
    head = share[:rank]
    print(f"\n全层平均奇异值能量占比（前 {rank} 个 = 全部非零项，其余恒 0）:")
    print("  " + " ".join(f"{i+1}:{float(v):.4f}" for i, v in enumerate(head)))
    print(f"  累积: " + " ".join(f"k={k}:{float(head[:k].sum()):.4f}" for k in ks))


def truncate(ckpt, k, verify=False):
    """返回截断到 rank k 的新 ckpt（新 lora_state / lora_rank=k）。"""
    rank = ckpt.get("lora_rank", 16)
    scaling = ALPHA / rank
    if k > rank:
        raise ValueError(f"k={k} > rank={rank}：截断只能向下，向上需真训")
    ps = _pairs(ckpt["lora_state"])
    new_state = dict(ckpt["lora_state"])
    scaling_k = ALPHA / k
    worst = 0.0
    for pre, A, B in ps:
        delta = scaling * (B @ A)                     # 原始贡献（out,in）
        U, S, Vh = _svd(delta)
        Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
        delta_k = Uk @ torch.diag(Sk) @ Vhk           # 最优 rank-k 近似
        # 重建时模型用 scaling_k → 写入因子需预除，使 scaling_k·(B_k@A_k)=delta_k
        # .contiguous() 必需：Uk/Vhk 是 SVD 输出的切片视图，直接存会把底层整个
        # (out,out)/(in,in) storage 一起序列化（实测 42MB ckpt 膨胀到 1.5GB）。
        new_state[pre + ".lora_B"] = (Uk * (Sk / scaling_k).unsqueeze(0)).contiguous()
        new_state[pre + ".lora_A"] = Vhk.contiguous()
        # 回读校验：按目标模型的 scaling 复原
        back = scaling_k * (new_state[pre + ".lora_B"] @ new_state[pre + ".lora_A"])
        worst = max(worst, float((back - delta_k).abs().max()))
    new = dict(ckpt)
    new["lora_state"] = new_state
    new["lora_rank"] = k
    print(f"[truncate] k={k}（原 rank={rank}）→ {len(ps)} 对已截断；"
          f"回读重建最大绝对误差 {worst:.2e}（应≈0，浮点级）")
    if verify:
        if k != rank:
            print("[verify] 跳过：仅在 k=rank 时校验逐位保真")
        else:
            assert worst < 1e-4, f"k=rank 时应逐位保真，实测误差 {worst}"
            print("[verify] ✅ k=rank 逐位保真（截断路径本身无副作用）")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report_ks", default="2,4,6,8,10,12,16")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.report_only:
        ks = [int(x) for x in args.report_ks.split(",")]
        report(ckpt, ks)
        return
    if args.k is None or args.out is None:
        ap.error("非 --report_only 时必须给 --k 与 --out")
    new = truncate(ckpt, args.k, verify=args.verify)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # 原子写：先落 .tmp 再 replace。否则 build 中途被打断会留下半截文件，
    # 而调用方的 `[ ! -f $out ]` 会把它当成"已存在"跳过重建 → 续跑用坏 ckpt。
    tmp = args.out + ".tmp"
    torch.save(new, tmp)
    os.replace(tmp, args.out)
    print(f"[saved] {args.out}  (lora_rank={new['lora_rank']})")


if __name__ == "__main__":
    main()
