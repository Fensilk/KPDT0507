"""
C3D (Tran et al., ICCV 2015) re-implemented in PyTorch for the Phase-10 baseline.

Why a re-implementation: the official repository vendored in `reference/C3D` is a Caffe fork and
ships **no trained weights** (`data/` contains only prototxt definitions and file lists), and
building that Caffe (CUDA-8 era) on a modern machine is impractical.

This follows the published architecture: 8 conv3d layers (64/128/256/256/512/512/512/512) with the
prototxt's pooling schedule, then two 4096-d FC layers and a 3-class head. Input 16 x 112 x 112.
It is trained from scratch (no Kinetics pre-training) -> documented as a limitation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class C3D(nn.Module):
    """C3D for 16 x 112 x 112 clips."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.5, in_frames: int = 16, in_size: int = 112):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64),                       # conv1a
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            _conv_block(64, 128),                     # conv2a
            nn.MaxPool3d(kernel_size=2, stride=2),
            _conv_block(128, 256),                    # conv3a
            _conv_block(256, 256),                    # conv3b
            nn.MaxPool3d(kernel_size=2, stride=2),
            _conv_block(256, 512),                    # conv4a
            _conv_block(512, 512),                    # conv4b
            nn.MaxPool3d(kernel_size=2, stride=2),
            _conv_block(512, 512),                    # conv5a
            _conv_block(512, 512),                    # conv5b
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # The published C3D feeds 512 x 1 x 4 x 4 = 8192 into FC6. With the prototxt pooling schedule
        # an 112x112 input yields 3x3 after pool5 (integer floor), so we normalise to the paper's
        # 1x4x4 shape instead of silently changing the FC dimensions (keeps ~78 M params as reported).
        self.spatial_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, in_frames, in_size, in_size)
            n_feat = int(self.spatial_pool(self.features(dummy)).numel())

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_feat, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.n_feat = n_feat

    def forward(self, x):          # x: (B, C, T, H, W)
        x = self.features(x)
        x = self.spatial_pool(x)
        x = x.flatten(1)
        return self.classifier(x)
