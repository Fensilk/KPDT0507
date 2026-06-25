"""
Standalone training script for T=32 experiment.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('logs/tcn_t32_s1', exist_ok=True)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from models.tcn_model import TCNModel
from models.dataset import create_longseq_dataloaders
from utils.metrics import compute_all_metrics
from utils.visualization import CLASS_NAMES

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda')
print(f'Device: {device}')

# Data
print('\n[1/5] Loading data...')
train_loader, val_loader, test_loader = create_longseq_dataloaders(
    npz_path='data/omnifall_frame_preprocessed.npz',
    splits_dir='DATASET-omnifall/splits/syn/random',
    window_size=32, stride=1, batch_size=16, num_workers=0,
    use_weighted_sampler=True,
)

# Model
print('[2/5] Building model...')
model = TCNModel(window_size=32).to(device)
print(f'  Params: {sum(p.numel() for p in model.parameters()):,}, RF={model.receptive_field}')

# Criterions
print('[3/5] Building criterions...')
train_dataset = train_loader.dataset
n_samples = min(len(train_dataset), 5000)
step = max(1, len(train_dataset) // n_samples)
labels_list, fall_list, fallen_list = [], [], []
for i in range(0, len(train_dataset), step):
    s = train_dataset[i]
    labels_list.append(s['labels_16'])
    fall_list.append(s['fall_labels'])
    fallen_list.append(s['fallen_labels'])
all_labels = torch.cat(labels_list).numpy()
all_fall = torch.cat(fall_list).numpy()
all_fallen = torch.cat(fallen_list).numpy()
class_counts = np.bincount(all_labels, minlength=16).astype(np.float32)
class_counts = np.where(class_counts == 0, 1.0, class_counts)
cls_w = torch.FloatTensor(len(all_labels)/(16*class_counts)).to(device)
fall_pw = torch.FloatTensor([max(len(all_fall)-all_fall.sum(),1)/max(all_fall.sum(),1)]).to(device)
fallen_pw = torch.FloatTensor([max(len(all_fallen)-all_fallen.sum(),1)/max(all_fallen.sum(),1)]).to(device)
criterions = {
    'cls': nn.CrossEntropyLoss(weight=cls_w),
    'fall': nn.BCEWithLogitsLoss(pos_weight=fall_pw),
    'fallen': nn.BCEWithLogitsLoss(pos_weight=fallen_pw),
}

# Training
print('[4/5] Training (10 epochs)...')
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)

history = {'train_loss':[], 'val_loss':[], 'seg_acc':[], 'fall_f1':[], 'fallen_f1':[]}
best_fall_f1, best_epoch, patience = 0, 0, 0

for epoch in range(1, 11):
    # Train
    model.train()
    t0 = time.time()
    train_loss = 0.0
    for batch in train_loader:
        features = batch['features'].to(device)
        labels_16 = batch['labels_16'].to(device)
        fall_gt = batch['fall_labels'].to(device)
        fallen_gt = batch['fallen_labels'].to(device)
        B, T = labels_16.shape
        logits_cls, logits_fall, logits_fallen = model(features)
        loss = criterions['cls'](logits_cls.reshape(B*T,-1), labels_16.reshape(B*T)) \
             + 0.5*criterions['fall'](logits_fall.reshape(B*T), fall_gt.reshape(B*T).float()) \
             + 0.5*criterions['fallen'](logits_fallen.reshape(B*T), fallen_gt.reshape(B*T).float())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    all_pred_cls, all_pred_fall, all_pred_fallen = [], [], []
    all_gt_cls, all_gt_fall, all_gt_fallen = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels_16 = batch['labels_16'].to(device)
            fall_gt = batch['fall_labels'].to(device)
            fallen_gt = batch['fallen_labels'].to(device)
            B, T = labels_16.shape
            logits_cls, logits_fall, logits_fallen = model(features)
            loss = criterions['cls'](logits_cls.reshape(B*T,-1), labels_16.reshape(B*T)) \
                 + 0.5*criterions['fall'](logits_fall.reshape(B*T), fall_gt.reshape(B*T).float()) \
                 + 0.5*criterions['fallen'](logits_fallen.reshape(B*T), fallen_gt.reshape(B*T).float())
            val_loss += loss.item()
            all_pred_cls.append(logits_cls.argmax(-1).cpu().numpy().ravel())
            all_pred_fall.append((torch.sigmoid(logits_fall)>0.5).long().squeeze(-1).cpu().numpy().ravel())
            all_pred_fallen.append((torch.sigmoid(logits_fallen)>0.5).long().squeeze(-1).cpu().numpy().ravel())
            all_gt_cls.append(labels_16.cpu().numpy().ravel())
            all_gt_fall.append(fall_gt.cpu().numpy().ravel())
            all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())
    val_loss /= len(val_loader)

    metrics = compute_all_metrics(
        np.concatenate(all_pred_cls), np.concatenate(all_pred_fall), np.concatenate(all_pred_fallen),
        np.concatenate(all_gt_cls), np.concatenate(all_gt_fall), np.concatenate(all_gt_fallen),
        CLASS_NAMES)

    scheduler.step(metrics['fall_f1'])

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['seg_acc'].append(metrics['seg_acc'])
    history['fall_f1'].append(metrics['fall_f1'])
    history['fallen_f1'].append(metrics['fallen_f1'])

    print(f'Epoch {epoch:2d}: train_loss={train_loss:.4f} ({time.time()-t0:.0f}s) | '
          f'val_loss={val_loss:.4f} | seg_acc={metrics["seg_acc"]:.4f} | '
          f'fall_f1={metrics["fall_f1"]:.4f} | fallen_f1={metrics["fallen_f1"]:.4f} | lr={optimizer.param_groups[0]["lr"]:.2e}')

    if metrics['fall_f1'] > best_fall_f1:
        best_fall_f1 = metrics['fall_f1']
        best_epoch = epoch
        patience = 0
        torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(), 'best_fall_f1':best_fall_f1, 'val_metrics':metrics},
                   'logs/tcn_t32_s1/tcn_best.pt')
        print(f'  [BEST] saved!')
    else:
        patience += 1
        if patience >= 10:
            print(f'Early stopping at epoch {epoch}')
            break

# Test
print(f'\n[5/5] Testing (best epoch={best_epoch})...')
ckpt = torch.load('logs/tcn_t32_s1/tcn_best.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

all_pred_cls, all_pred_fall, all_pred_fallen = [], [], []
all_gt_cls, all_gt_fall, all_gt_fallen = [], [], []
test_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        features = batch['features'].to(device)
        labels_16 = batch['labels_16'].to(device)
        fall_gt = batch['fall_labels'].to(device)
        fallen_gt = batch['fallen_labels'].to(device)
        B, T = labels_16.shape
        logits_cls, logits_fall, logits_fallen = model(features)
        loss = criterions['cls'](logits_cls.reshape(B*T,-1), labels_16.reshape(B*T)) \
             + 0.5*criterions['fall'](logits_fall.reshape(B*T), fall_gt.reshape(B*T).float()) \
             + 0.5*criterions['fallen'](logits_fallen.reshape(B*T), fallen_gt.reshape(B*T).float())
        test_loss += loss.item()
        all_pred_cls.append(logits_cls.argmax(-1).cpu().numpy().ravel())
        all_pred_fall.append((torch.sigmoid(logits_fall)>0.5).long().squeeze(-1).cpu().numpy().ravel())
        all_pred_fallen.append((torch.sigmoid(logits_fallen)>0.5).long().squeeze(-1).cpu().numpy().ravel())
        all_gt_cls.append(labels_16.cpu().numpy().ravel())
        all_gt_fall.append(fall_gt.cpu().numpy().ravel())
        all_gt_fallen.append(fallen_gt.cpu().numpy().ravel())

test_metrics = compute_all_metrics(
    np.concatenate(all_pred_cls), np.concatenate(all_pred_fall), np.concatenate(all_pred_fallen),
    np.concatenate(all_gt_cls), np.concatenate(all_gt_fall), np.concatenate(all_gt_fallen),
    CLASS_NAMES)
test_metrics['loss'] = test_loss / len(test_loader)

# Save
with open('logs/tcn_t32_s1/training_history.json','w') as f:
    json.dump(history, f, indent=2)
with open('logs/tcn_t32_s1/test_results.json','w') as f:
    r = {k:v for k,v in test_metrics.items() if k!='cls_report'}
    r['cls_report'] = test_metrics['cls_report']
    json.dump(r, f, indent=2)

print(f'\nResults:')
print(f'  Best epoch: {best_epoch}')
print(f'  seg_acc: {test_metrics["seg_acc"]:.4f}')
print(f'  fall_f1: {test_metrics["fall_f1"]:.4f}')
print(f'  fallen_f1: {test_metrics["fallen_f1"]:.4f}')
print(f'  avg_f1: {test_metrics["avg_f1"]:.4f}')
print(f'  Test loss: {test_metrics["loss"]:.4f}')
print('\nT=32 experiment COMPLETE!')
