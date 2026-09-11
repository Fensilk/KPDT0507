"""方法7 蒸馏学生训练（本地 4060）：
学生 = 冻特征(3072 Δdiff) 上的 Phase6 bridge 时序头；
loss = KL(学生|teacher软标签) + λ·CE(硬标签)。teacher = E1@2000（m7_precompute_teacher.py 产出）。
按 val avg_f1 早停选 best，best 跑全量 test。

对比对象：E3@2000（同数据/同头/硬标签 CE，fallen_prec 0.632）、E1@2000(0.645)。

注意：finetuned 特征 NPZ(2.7G) 必须在 main 里整块载入内存一次（mmap 逐窗口随机读在
Windows 上极慢 ~120ms/次），否则一个 epoch 要十几分钟。
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from models.phase6_model import Phase6TernaryModel
from utils.metrics import compute_ternary_metrics

WINDOW, STRIDE, EVAL_STRIDE, N_FRAMES = 64, 16, 8, 80


def ternary(l16):
    return np.where(l16 == 1, 0, np.where(l16 == 2, 1, 2))


def _feat_window(feat_arr, st, ws, T):
    """feat_arr: 内存中的 (总帧,1536) fp16 数组。"""
    fw = feat_arr[st + ws: st + ws + T].astype(np.float32)
    diff = np.zeros_like(fw); diff[1:] = fw[1:] - fw[:-1]
    return np.concatenate([fw, diff], axis=-1)      # (T,3072)


class DistillSet(Dataset):
    """e1_2000 train stride16 窗口；枚举顺序与 teacher 预计算完全一致。feat_arr 已载入内存。"""

    def __init__(self, feat_arr, l16, vsi, idx, paths, teacher_npz):
        self.feat_arr = feat_arr
        self.l16, self.vsi, self.idx = l16, vsi, idx
        self.paths = paths
        self.meta = [(v, ws) for v in paths for ws in range(0, N_FRAMES - WINDOW + 1, STRIDE)]
        t = np.load(teacher_npz, allow_pickle=True)
        self.t_logits = t["logits"]
        assert len(self.meta) == len(self.t_logits), f"{len(self.meta)} != teacher {len(self.t_logits)}"
        tv, tws = t["videos"], t["ws"]
        for k in range(0, len(self.meta), max(1, len(self.meta) // 5)):
            assert tv[k] == self.meta[k][0] and int(tws[k]) == self.meta[k][1], f"顺序错位@{k}"
        self.st = np.array([int(vsi[idx[v]]) for v, _ in self.meta])
        self.lab = np.stack([ternary(l16[int(vsi[idx[v]]) + ws: int(vsi[idx[v]]) + ws + WINDOW])
                             for v, ws in self.meta]).astype(np.int64)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        v, ws = self.meta[i]
        x = _feat_window(self.feat_arr, int(self.st[i]), ws, WINDOW)
        return x, self.lab[i], self.t_logits[i].astype(np.float32)


def _load_npz_labels(lab_npz):
    d = np.load(lab_npz, allow_pickle=True, mmap_mode="r")
    vp = [str(p) for p in d["video_paths"]]
    return d["labels_16"], d["video_start_indices"], {p: i for i, p in enumerate(vp)}


def eval_split(model, feat_arr, l16, vsi, idx, split_csv, device, stride=EVAL_STRIDE):
    model.eval()
    paths = list(pd.read_csv(split_csv)["path"].str.strip())
    all_y, all_p = [], []
    with torch.no_grad():
        for v in paths:
            st = int(vsi[idx[v]])
            for ws in range(0, N_FRAMES - WINDOW + 1, stride):
                x = torch.from_numpy(_feat_window(feat_arr, st, ws, WINDOW)).float().unsqueeze(0).to(device)
                lg = model(x)[0]
                all_p.append(lg.argmax(-1).cpu().numpy())
                all_y.append(ternary(l16[st + ws: st + ws + WINDOW]))
    return np.concatenate(all_y), np.concatenate(all_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE, "data", "omnifall_dinov2_finetuned_frame.npz"))
    ap.add_argument("--labels_npz", default=os.path.join(BASE, "data", "omnifall_dinov2_giant_frame.npz"))
    ap.add_argument("--teacher", default=os.path.join(BASE, "data", "m7_teacher_logits.npz"))
    ap.add_argument("--splits", default=os.path.join(BASE, "DATASET-omnifall", "splits", "syn", "e1_2000"))
    ap.add_argument("--exp_tag", default="phase9/m7_distill")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_every", type=int, default=1, help="每 N epoch 做一次 val 评估（省时）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = f"logs/{args.exp_tag}"; os.makedirs(out_dir, exist_ok=True)
    device = args.device

    print("[load] 载入特征到内存(2.7G)...", flush=True)
    feat_arr = np.load(args.data)["features"]            # (960000,1536) fp16 整块入内存
    l16, vsi, idx = _load_npz_labels(args.labels_npz)
    train_paths = list(pd.read_csv(f"{args.splits}/train.csv")["path"].str.strip())
    ds = DistillSet(feat_arr, l16, vsi, idx, train_paths, args.teacher)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    print(f"[数据] {len(ds)} 窗口 | device {device}", flush=True)

    model = Phase6TernaryModel(input_dim=3072, hidden_dim=384, decoder_type="transformer",
                               causal=False, num_heads=6, dropout=0.3, max_len=WINDOW + 10,
                               num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[模型] {sum(p.numel() for p in model.parameters()):,} 参数 | temp={args.temp} lam={args.lam}", flush=True)

    best_f1, best_ep, no_improve = -1.0, 0, 0
    hist = []
    for ep in range(args.epochs):
        model.train(); tot, nk, nc, ntot = 0.0, 0.0, 0.0, 0
        for x, y, t in loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            opt.zero_grad()
            s = model(x)
            kl = F.kl_div(F.log_softmax(s / args.temp, -1), F.softmax(t / args.temp, -1),
                          reduction="batchmean") * args.temp ** 2
            ce = F.cross_entropy(s.reshape(-1, 3), y.reshape(-1))
            loss = kl + args.lam * ce
            loss.backward(); opt.step()
            n = y.shape[0]; tot += loss.item() * n; nk += kl.item() * n; nc += ce.item() * n
            ntot += n
        # 每 val_every epoch 做一次 val
        m = None
        if (ep + 1) % args.val_every == 0:
            yv, pv = eval_split(model, feat_arr, l16, vsi, idx, f"{args.splits}/val.csv", device)
            m = compute_ternary_metrics(pv, yv)
            hist.append({"ep": ep + 1, "loss": tot / ntot, "kl": nk / ntot, "ce": nc / ntot,
                         "val_f1": float(m["avg_f1"]), "val_fallen_prec": float(m["fallen_precision"])})
            print(f"ep {ep+1:3d} loss={tot/ntot:.4f} kl={nk/ntot:.4f} ce={nc/ntot:.4f} | "
                  f"val avgF1={m['avg_f1']:.4f} fallen_prec={m['fallen_precision']:.4f}", flush=True)
            if m["avg_f1"] > best_f1:
                best_f1, best_ep, no_improve = float(m["avg_f1"]), ep + 1, 0
                torch.save(model.state_dict(), f"{out_dir}/best_head.pt")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[early stop] ep {ep+1}（best ep {best_ep}, val avgF1 {best_f1:.4f}）", flush=True)
                    break
    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(hist, f, ensure_ascii=False)

    if os.path.exists(f"{out_dir}/best_head.pt"):
        model.load_state_dict(torch.load(f"{out_dir}/best_head.pt", weights_only=True))
    yt, pt = eval_split(model, feat_arr, l16, vsi, idx, f"{args.splits}/test.csv", device)
    m = compute_ternary_metrics(pt, yt)
    if hasattr(m["confusion_matrix"], "tolist"):      # 已是 list 则不动
        m["confusion_matrix"] = m["confusion_matrix"].tolist()
    m["cls_report"] = classification_report(yt, pt, labels=[0, 1, 2],
                                            target_names=["fall", "fallen", "normal"], zero_division=0)
    m["config"] = vars(args)
    with open(f"{out_dir}/test_results.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"[done] {out_dir}/test_results.json (best ep {best_ep})", flush=True)
    print(f"  fallen_prec={m['fallen_precision']:.4f} fallen_rec={m['fallen_recall']:.4f} "
          f"fall_rec={m['fall_recall']:.4f} avg_f1={m['avg_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
