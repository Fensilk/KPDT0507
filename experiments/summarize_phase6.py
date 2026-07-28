"""Summarize Phase 6 experiment results."""
import json, os, glob

# Phase 6 results summary
exps = ['p6_bridge','p6a_ct_g2','p6a_ct_g3','p6a_ct_g5',
        'p6b_lstm_g2','p6b_lstm_g3','p6b_lstm_g5',
        'p6c_mamba_g2','p6c_mamba_g3','p6c_mamba_g5']

print("="*90)
print("PHASE 6 TEST RESULTS SUMMARY")
print("="*90)
header = f"{'Exp':<18} {'Acc':>8} {'Fall_F1':>8} {'Fallen_F1':>8} {'Normal_F1':>8} {'Avg_F1':>8} {'BestEp':>7} {'Epochs':>7}"
print(header)
print("-"*90)

for exp in exps:
    path = f'logs/phase6/{exp}/test_results.json'
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        cfg = d['config']
        print(f"{exp:<18} {d['ternary_acc']:8.4f} {d['fall_f1']:8.4f} {d['fallen_f1']:8.4f} {d['normal_f1']:8.4f} {d['avg_f1']:8.4f} {cfg['best_epoch']:>7} {cfg['epochs_run']:>7}")

print()
print("="*90)
print("CONFIGURATION MATRIX")
print("="*90)
print(f"{'Exp':<18} {'Decoder':>12} {'Causal':>7} {'Layers':>7} {'Gamma':>6} {'Params':>8}")
print("-"*70)
for exp in exps:
    path = f'logs/phase6/{exp}/test_results.json'
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        cfg = d['config']
        print(f"{exp:<18} {cfg['decoder_type']:>12} {str(cfg['causal']):>7} {cfg['num_layers']:>7} {cfg['focal_gamma']:>6} {cfg['n_params']:>8}")

print()
print("="*90)
print("DETAILED PER-CLASS METRICS")
print("="*90)
for exp in exps:
    path = f'logs/phase6/{exp}/test_results.json'
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        print(f"\n--- {exp} ---")
        print(f"  Fall:     P={d['fall_precision']:.4f}  R={d['fall_recall']:.4f}  F1={d['fall_f1']:.4f}")
        print(f"  Fallen:   P={d['fallen_precision']:.4f}  R={d['fallen_recall']:.4f}  F1={d['fallen_f1']:.4f}")
        print(f"  Normal:   P={d['normal_precision']:.4f}  R={d['normal_recall']:.4f}  F1={d['normal_f1']:.4f}")
        cm = d['confusion_matrix']
        print(f"  CM:       [[{cm[0][0]}, {cm[0][1]}, {cm[0][2]}],")
        print(f"            [{cm[1][0]}, {cm[1][1]}, {cm[1][2]}],")
        print(f"            [{cm[2][0]}, {cm[2][1]}, {cm[2][2]}]]")
        # Row-normalized confusion matrix
        print(f"  CM(norm):")
        for i, row in enumerate(cm):
            total = sum(row)
            norm = [f"{x/total:.3f}" for x in row]
            print(f"            {['Fall','Fallen','Normal'][i]:>8}: {norm}")

print()
print("="*90)
print("MISS RATE ANALYSIS (key safety metric)")
print("="*90)
for exp in exps:
    path = f'logs/phase6/{exp}/test_results.json'
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        cm = d['confusion_matrix']
        # Fall -> Normal miss rate
        fall_to_normal = cm[0][2] / sum(cm[0]) * 100
        # Fallen -> Normal miss rate
        fallen_to_normal = cm[1][2] / sum(cm[1]) * 100
        # Fall -> Fallen confusion
        fall_to_fallen = cm[0][1] / sum(cm[0]) * 100
        fallen_to_fall = cm[1][0] / sum(cm[1]) * 100
        print(f"  {exp:<18} Fall->Norm: {fall_to_normal:5.1f}%  Fallen->Norm: {fallen_to_normal:5.1f}%  Fall->Fallen: {fall_to_fallen:5.1f}%  Fallen->Fall: {fallen_to_fall:5.1f}%")

print()
print("="*90)
print("TRAINING DYNAMICS")
print("="*90)
for exp in exps:
    path = f'logs/phase6/{exp}/training_history.json'
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        best_idx = max(enumerate(d['val_fall_f1']), key=lambda x: x[1])[0]
        print(f"\n{exp}:")
        print(f"  Best val_fall_f1 @ epoch {best_idx+1}: {d['val_fall_f1'][best_idx]:.4f}")
        print(f"  Peak val_fallen_f1: {max(d['val_fallen_f1']):.4f}")
        print(f"  Peak val_normal_f1: {max(d['val_normal_f1']):.4f}")
        print(f"  Train loss: {d['train_loss'][0]:.4f} -> {d['train_loss'][-1]:.4f}")
        print(f"  Val loss:   {d['val_loss'][0]:.4f} -> {d['val_loss'][-1]:.4f}")
        print(f"  Stopped at epoch {len(d['train_loss'])}/{d['train_loss'].__len__()} (early stopping)")

# Phase 4 reference
print()
print("="*90)
print("PHASE 4 REFERENCE (Multi-task, T=32, bidirectional Transformer)")
print("="*90)
for f in sorted(glob.glob('logs/phase4/*/test_results.json')):
    exp = os.path.basename(os.path.dirname(f))
    d = json.load(open(f, encoding='utf-8'))
    cfg = d.get('config', {})
    print(f"\n{exp}:")
    keys = ['fall_f1','fallen_f1','ternary_acc','avg_f1']
    for k in keys:
        if k in d:
            print(f"  {k}: {d[k]}")
    if 'cls_report' in d:
        print(f"  cls_report: {d['cls_report'][:300]}")
