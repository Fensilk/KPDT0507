"""
E3 Stage2 结果对比：微调特征 vs baseline（p7d_delta）。

对比两阶段后时序模型的 test_results.json 主指标，并打印诊断矩阵所需的两条轴观察：
  - 动态轴（lie_down vs fall）：fall_rec / fall_prec
  - 静态轴（lying vs fallen）：fallen_rec / fallen_prec

用法（项目根目录）：
    python experiments/eval_p9e3.py [--e3 logs/phase9/e3_stage2/test_results.json]
"""
import os
import sys
import json
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

BASELINE = "logs/phase7/p7d_delta/test_results.json"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", default="logs/phase9/e3_stage2/test_results.json")
    args = ap.parse_args()

    if not os.path.exists(args.e3):
        print(f"[error] {args.e3} 不存在，先跑 Stage2。")
        sys.exit(1)

    b = load(BASELINE)
    e = load(args.e3)

    keys = ["fall_precision", "fall_recall", "fall_f1",
            "fallen_precision", "fallen_recall", "fallen_f1",
            "avg_f1", "ternary_acc"]
    print(f"{'指标':<18s} {'baseline':>10s} {'E3微调':>10s} {'Δ':>8s}")
    print("-" * 50)
    for k in keys:
        bv = b.get(k); ev = e.get(k)
        if bv is None or ev is None:
            continue
        print(f"{k:<18s} {bv:>10.4f} {ev:>10.4f} {ev-bv:>+8.4f}")

    print()
    print("诊断矩阵观察（主指标视角）:")
    print(f"  动态轴 lie_down vs fall: fall_rec={e.get('fall_recall',0):.3f} (baseline {b.get('fall_recall',0):.3f}) "
          f"fall_prec={e.get('fall_precision',0):.3f} (baseline {b.get('fall_precision',0):.3f})")
    print(f"  静态轴 lying vs fallen: fallen_rec={e.get('fallen_recall',0):.3f} (baseline {b.get('fallen_recall',0):.3f}) "
          f"fallen_prec={e.get('fallen_precision',0):.3f} (baseline {b.get('fallen_precision',0):.3f})")
    print()
    print("注：E3 是帧级微调，预期动态轴（受控性，帧级可学）改善；")
    print("    静态轴（因果，需时序）改善有限——若只有动态轴改善，即证实端到端（E1）必要性。")


if __name__ == "__main__":
    main()
