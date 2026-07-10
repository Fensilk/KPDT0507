#!/bin/bash
# ================================================================
# AutoDL Deployment Script — Phase 4 DINOv2 + Transformer
# 
# 用法 (在 AutoDL 服务器上):
#   1. 上传本目录到 /root/autodl-tmp/KPDT0507/
#   2. bash autoDL_deploy.sh
# ================================================================
set -e

echo "=============================================="
echo "Phase 4: DINOv2 + Transformer Fall Detection"
echo "=============================================="

# 确认数据文件
echo ""
echo "Checking data files..."
for f in data/omnifall_dinov2_frame.npz data/omnifall_pose_frame.npz; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OK] $f ($SIZE)"
    else
        echo "  [MISSING] $f — please upload this file!"
        exit 1
    fi
done

# 安装依赖
echo ""
echo "Installing dependencies..."
pip install -q tensorboard scikit-learn matplotlib 2>&1 | tail -1
echo "  [OK] Dependencies installed"

# 快速验证
echo ""
echo "Quick verification..."
python -c "
from models.transformer_model import MultimodalFeatureTransformer
import torch
m = MultimodalFeatureTransformer(input_dim=1024, hidden_dim=384, num_layers=1, dropout=0.3)
x = torch.randn(2, 32, 1024)
c, f, fn = m(x)
print(f'  Model OK — Params: {sum(p.numel() for p in m.parameters()):,}')
print(f'  Input: {x.shape} -> cls: {c.shape}, fall: {f.shape}, fallen: {fn.shape}')
"

echo ""
echo "=============================================="
echo "Setup complete! Ready to run experiments."
echo ""
echo "Quick start:"
echo "  bash experiments/run_phase4.sh 4a    # ~2h on RTX 3090"
echo "  bash experiments/run_phase4.sh all   # full run"
echo "=============================================="
