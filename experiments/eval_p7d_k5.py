"""
Evaluate p7d_delta + RuleV1(k=5) — full ternary + binary event metrics.

Usage:
    python experiments/eval_p7d_k5.py
"""
import os, sys, json
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.phase6_model import Phase6TernaryModel

CLASS_NAMES_16 = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying",
    "stand_up", "standing", "other", "kneel_down", "kneeling",
    "squat_down", "squatting", "crawl", "jump",
]

def load_and_infer(ckpt, npz, splits, use_diff, T=64, stride=8, bs=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(npz, allow_pickle=True, mmap_mode='r')
    paths = set()
    with open(os.path.join(splits, "test.csv")) as f:
        for line in f:
            p = line.strip().split(",")[0]
            if p: paths.add(p)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = Phase6TernaryModel(
        input_dim=3072 if use_diff else 1536, hidden_dim=384, decoder_type="transformer",
        causal=False, num_layers=1, num_heads=6, dropout=0.3, max_len=T+10, num_classes=3,
    ).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    vp = d["video_paths"]; vsi = d["video_start_indices"]; vcc = d["video_clip_counts"]
    feats = d["features"]; lb16 = d["labels_16"]; fl = d["fall_labels"]; fdl = d["fallen_labels"]
    wins, meta, vinfo = [], [], {}
    for vi in range(len(vp)):
        if str(vp[vi]) not in paths: continue
        ns = int(vsi[vi]); nf = int(vcc[vi]); nw = (nf-T)//stride+1
        vinfo[vi] = (ns, nf, nw); vf = feats[ns:ns+nf]
        for w in range(nw): off = w*stride; wins.append(vf[off:off+T]); meta.append((vi,off))
    preds = []
    for b0 in range(0, len(wins), bs):
        b1 = min(b0+bs, len(wins))
        batch = torch.from_numpy(np.stack(wins[b0:b1])).float().to(device)
        if use_diff:
            diff = torch.zeros_like(batch); diff[:,1:] = batch[:,1:] - batch[:,:-1]
            batch = torch.cat([batch, diff], dim=-1)
        with torch.no_grad(): preds.append(m(batch).argmax(dim=-1).cpu().numpy())
        if (b0//bs+1)%50==0: print(f"  [{b1}/{len(wins)}]")
    preds = np.concatenate(preds, axis=0)
    pv = {}
    for vi, (ns, nf, nw) in vinfo.items():
        votes = np.zeros((nf,3), dtype=int)
        for wi, (vj, off) in enumerate(meta):
            if vj != vi: continue
            for t in range(T):
                f = off+t
                if f < nf: votes[f, preds[wi,t]] += 1
        bp = np.argmax(votes, axis=1)
        tg = np.full(nf, 2, dtype=int); tg[fl[ns:ns+nf]==1]=0; tg[fdl[ns:ns+nf]==1]=1
        pv[vi] = {"labels_16": lb16[ns:ns+nf], "ternary_gt": tg, "bridge_pred": bp,
                   "binary_fall_gt": fl[ns:ns+nf], "binary_fallen_gt": fdl[ns:ns+nf]}
    return pv

def apply_rulev1(pv, k=5, timeout=32):
    corr = {}; stats = {"events":0, "fallen_kept":0, "fallen_suppressed":0}
    for vi, d in pv.items():
        bp = d["bridge_pred"].copy(); nf = len(bp); cp = bp.copy()
        ev, timer, cf = False, 0, 0
        for f in range(nf):
            p = bp[f]; cf = cf+1 if p==0 else 0
            if not ev and cf>=k: ev=True; timer=0; stats["events"]+=1
            if ev:
                if p==1: stats["fallen_kept"]+=1; timer=0
                elif p==0: timer=0
                else: timer+=1
                if timer>timeout: ev=False
            elif p==1: cp[f]=2; stats["fallen_suppressed"]+=1
        corr[vi]=cp
    return corr, stats

def evaluate(pv, corr):
    all_gt3, all_bp, all_cp, all_l16, all_fg, all_flg = [], [], [], [], [], []
    for vi, d in pv.items():
        n = len(d["ternary_gt"])
        all_gt3.append(d["ternary_gt"]); all_bp.append(d["bridge_pred"])
        all_cp.append(corr[vi][:n]); all_l16.append(d["labels_16"])
        all_fg.append(d["binary_fall_gt"]); all_flg.append(d["binary_fallen_gt"])
    gt3 = np.concatenate(all_gt3); bp = np.concatenate(all_bp); cp = np.concatenate(all_cp)
    l16 = np.concatenate(all_l16); fg = np.concatenate(all_fg); flg = np.concatenate(all_flg)

    def ternary_metrics(pred):
        return {
            "ternary_acc": float((pred==gt3).mean()),
            "fall_f1": float(f1_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[0]),
            "fall_precision": float(precision_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[0]),
            "fall_recall": float(recall_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[0]),
            "fallen_f1": float(f1_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[1]),
            "fallen_precision": float(precision_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[1]),
            "fallen_recall": float(recall_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[1]),
            "normal_f1": float(f1_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0)[2]),
            "avg_f1": float(f1_score(gt3,pred,average=None,labels=[0,1,2],zero_division=0).mean()),
            "confusion_matrix": confusion_matrix(gt3,pred,labels=[0,1,2]).tolist(),
        }

    def binary_metrics(pred):
        binary_gt = ((fg==1)|(flg==1)).astype(int)
        binary_pred = (pred!=2).astype(int)
        cm = confusion_matrix(binary_gt, binary_pred, labels=[0,1])
        tp = cm[1,1]; fp = cm[0,1]; fn = cm[1,0]
        ep = tp/max(tp+fp,1); er = tp/max(tp+fn,1)
        return {
            "event_f1": 2*ep*er/(ep+er) if (ep+er)>0 else 0,
            "event_precision": ep,
            "event_recall": er,
            "binary_cm": cm.tolist(),
        }

    def breakdown(pred):
        bd = {}
        binary_pred = (pred!=2).astype(int)
        for c in range(16):
            m = l16==c; n = int(m.sum())
            if n==0: continue
            bd[CLASS_NAMES_16[c]] = {
                "total": n,
                "pred_fall": int((pred[m]==0).sum()),
                "pred_fallen": int((pred[m]==1).sum()),
                "pred_normal": int((pred[m]==2).sum()),
                "pred_event": int(binary_pred[m].sum()),
            }
        return bd

    return {
        "ternary": ternary_metrics(bp), "ternary_r1": ternary_metrics(cp),
        "binary": binary_metrics(bp), "binary_r1": binary_metrics(cp),
        "breakdown": breakdown(bp), "breakdown_r1": breakdown(cp),
    }


def main():
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(BASE)
    print("p7d_delta + RuleV1(k=5) — Full Evaluation")

    pv = load_and_infer("logs/phase7/p7d_delta/best_model.pt",
        "data/omnifall_dinov2_giant_frame.npz", "DATASET-omnifall/splits/syn/random", use_diff=True)
    corr, stats = apply_rulev1(pv, k=5)
    r = evaluate(pv, corr)

    print(f"\nRule Stats: events={stats['events']}, kept={stats['fallen_kept']:,}, "
          f"suppressed={stats['fallen_suppressed']:,}")

    # Ternary
    t, t1 = r["ternary"], r["ternary_r1"]
    print(f"\n=== Ternary ===")
    print(f"{'Metric':<22s} {'Bridge':>10s} {'+R1(k=5)':>10s} {'Delta':>10s}")
    for k in ["fall_f1","fall_precision","fall_recall","fallen_f1","fallen_precision",
              "fallen_recall","normal_f1","avg_f1","ternary_acc"]:
        print(f"{k:<22s} {t[k]:>10.4f} {t1[k]:>10.4f} {t1[k]-t[k]:>+10.4f}")

    # Binary event
    b, b1 = r["binary"], r["binary_r1"]
    print(f"\n=== Binary Event ===")
    print(f"{'Metric':<22s} {'Bridge':>10s} {'+R1(k=5)':>10s} {'Delta':>10s}")
    for k in ["event_f1","event_precision","event_recall"]:
        print(f"{k:<22s} {b[k]:>10.4f} {b1[k]:>10.4f} {b1[k]-b[k]:>+10.4f}")

    print(f"\nBinary CM (Bridge):  {b['binary_cm']}")
    print(f"Binary CM (+R1 k=5): {b1['binary_cm']}")

    # Save
    out = {"rule_stats": stats, "ternary": t, "ternary_r1": t1,
           "binary": b, "binary_r1": b1, "breakdown": r["breakdown"],
           "breakdown_r1": r["breakdown_r1"]}
    with open("logs/phase7/p7d_delta/eval_k5.json", "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved: logs/phase7/p7d_delta/eval_k5.json")

if __name__ == "__main__":
    main()
