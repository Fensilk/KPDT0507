"""
TCN (Temporal Convolutional Network) 跌倒检测模型.

实现老师指定的架构:
    features [T, 512]
        ↓ Linear: 512 -> 256
        ↓ TCN: 因果膨胀卷积, 在时间维度建模 clip 关系
        ↓
    ├── Head 1: 16 类动作分类
    ├── Head 2: fall 概率 (二分类)
    └── Head 3: fallen 概率 (二分类)

参考: Bai et al. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" (2018)
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    TCN 的基本构建块.

    包含两层因果膨胀卷积 + 残差连接.
    因果性保证: 位置 t 的输出只依赖于 ≤t 的输入 (不看未来).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_channels: 输入通道数.
            out_channels: 输出通道数.
            kernel_size: 卷积核大小.
            dilation: 膨胀率, 控制感受野.
            dropout: Dropout 比例.
        """
        super().__init__()

        # 因果卷积需要的左侧 padding 量
        # 对于因果卷积, 我们只在左侧 pad, 右侧不 pad
        # 实现方式: 两侧对称 pad 后裁剪掉右侧多余的
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接: 如果维度不匹配, 用 1x1 卷积对齐
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) — 输入特征.
        Returns:
            (B, C_out, T) — 输出特征, 时间维度不变.
        """
        residual = self.downsample(x)

        # 第一层因果卷积
        out = self.conv1(x)           # (B, C_out, T + padding)
        out = out[:, :, : -self.padding]  # 裁剪右侧 → 保持因果性: (B, C_out, T)
        out = self.relu1(out)
        out = self.dropout1(out)

        # 第二层因果卷积
        out = self.conv2(out)
        out = out[:, :, : -self.padding]
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu2(out + residual)


class TCNModel(nn.Module):
    """
    多任务 TCN 跌倒检测模型.

    输入 ResNet18 提取的 clip 特征, 输出三个预测头:
    - 16 类动作分类
    - fall 二分类 (正在摔倒)
    - fallen 二分类 (已经倒地)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        kernel_size: int = 3,
        dilations: list | None = None,
        window_size: int | None = None,
        tcn_dropout: float = 0.2,
        num_classes: int = 16,
    ):
        """
        Args:
            input_dim: 输入特征维度 (ResNet18 = 512).
            hidden_dim: TCN 隐藏层维度 (老师指定 256).
            num_blocks: TCN 块数量 (当 dilations 和 window_size 都未指定时使用).
            kernel_size: 卷积核大小.
            dilations: 每层的膨胀率列表. 当为 None 且 window_size 指定时自动计算.
            window_size: 序列长度 T, 用于自动计算合适的 dilations.
                         T=16 → [1,2,4,8] (RF=31)
                         T=32 → [1,2,4,8,16] (RF=63)
            tcn_dropout: TCN 内部的 Dropout 比例.
            num_classes: 动作类别数 (16).
        """
        super().__init__()

        if dilations is None:
            if window_size is not None:
                dilations = self._compute_dilations(kernel_size, window_size)
            else:
                dilations = [1, 2]

        self.dilations = dilations

        # --- 降维层: 512 -> 256 (老师指定) ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- TCN 层: 在时间维度上建模 ---
        tcn_blocks = []
        for i, dilation in enumerate(dilations):
            block = TemporalBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=tcn_dropout,
            )
            tcn_blocks.append(block)
        self.tcn = nn.Sequential(*tcn_blocks)

        # --- 输出头 ---
        # Head 1: 16 类动作预测
        self.head_cls = nn.Linear(hidden_dim, num_classes)

        # Head 2: fall 概率 (二分类)
        self.head_fall = nn.Linear(hidden_dim, 1)

        # Head 3: fallen 概率 (二分类)
        self.head_fallen = nn.Linear(hidden_dim, 1)

        # 记录感受野用于调试
        self.receptive_field = self._compute_receptive_field(kernel_size, dilations)

    @staticmethod
    def _compute_dilations(kernel_size: int, window_size: int) -> list[int]:
        """
        Auto-compute dilations using doubling pattern until RF >= window_size.

        Uses the standard TCN exponential dilation pattern [1, 2, 4, 8, ...]
        until the total receptive field covers the target window size.

        Receptive field formula: RF = 1 + (kernel_size - 1) * sum(dilations)
        """
        dilations = []
        rf = 1
        d = 1
        while rf < window_size:
            dilations.append(d)
            rf += (kernel_size - 1) * d
            d *= 2
        return dilations

    def _compute_receptive_field(
        self, kernel_size: int, dilations: list[int]
    ) -> int:
        """计算 TCN 的总感受野."""
        rf = 1
        for d in dilations:
            rf += (kernel_size - 1) * d
        return rf

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, 512) — clip 特征序列.

        Returns:
            logits_cls:    (B, T, 16) — 16 类动作 logits.
            logits_fall:   (B, T, 1)  — fall 二分类 logits.
            logits_fallen: (B, T, 1)  — fallen 二分类 logits.
        """
        B, T, D = features.shape

        # 1. 降维: (B, T, 512) → (B, T, 256)
        x = self.input_proj(features)

        # 2. 转置为 Conv1d 需要的格式: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)  # (B, 256, T)

        # 3. TCN 时序建模
        x = self.tcn(x)  # (B, 256, T)

        # 4. 转回: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T, 256)

        # 5. 三个输出头分别预测每个时间步
        logits_cls = self.head_cls(x)        # (B, T, 16)
        logits_fall = self.head_fall(x)       # (B, T, 1)
        logits_fallen = self.head_fallen(x)   # (B, T, 1)

        return logits_cls, logits_fall, logits_fallen


def test_forward_shapes():
    """验证前向传播的形状正确性."""
    model = TCNModel(
        input_dim=512,
        hidden_dim=256,
        num_blocks=2,
        kernel_size=3,
        dilations=[1, 2],
    )
    model.eval()

    # 模拟一个 batch: 8 个视频, 每个 5 个 clip, 512 维特征
    batch_size, seq_len, feat_dim = 8, 5, 512
    x = torch.randn(batch_size, seq_len, feat_dim)

    with torch.no_grad():
        logits_cls, logits_fall, logits_fallen = model(x)

    assert logits_cls.shape == (batch_size, seq_len, 16), \
        f"Expected (8,5,16), got {logits_cls.shape}"
    assert logits_fall.shape == (batch_size, seq_len, 1), \
        f"Expected (8,5,1), got {logits_fall.shape}"
    assert logits_fallen.shape == (batch_size, seq_len, 1), \
        f"Expected (8,5,1), got {logits_fallen.shape}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] All shapes correct | Params: {total_params:,} | RF: {model.receptive_field}")

    return True


def test_causality():
    """
    验证因果性: 位置 t 的输出不受位置 t+1 及之后输入的影响.

    方法: 改变位置 t+1 的输入, 检查位置 t 的输出是否不变.
    """
    model = TCNModel(dilations=[1, 2])
    model.eval()

    x1 = torch.randn(1, 5, 512)
    x2 = x1.clone()
    # 改变最后一个时间步 (位置 4)
    x2[:, 4, :] = torch.randn(512)

    with torch.no_grad():
        out1_cls, out1_fall, out1_fallen = model(x1)
        out2_cls, out2_fall, out2_fallen = model(x2)

    # 位置 0-3 的输出应该完全相同 (因果性)
    for name, o1, o2 in [
        ("cls", out1_cls, out2_cls),
        ("fall", out1_fall, out2_fall),
        ("fallen", out1_fallen, out2_fallen),
    ]:
        diff = (o1[:, :4] - o2[:, :4]).abs().max().item()
        # 位置 4 可能不同 (因为感受野=7 但只有5个时间步, 位置4能看到所有输入)
        diff_t4 = (o1[:, 4:] - o2[:, 4:]).abs().max().item()
        assert diff < 1e-5, \
            f"{name}: 因果性失败! 位置 0-3 最大差异={diff:.2e}"
        print(f"[PASS] {name}: Causality OK | pos 0-3 max_diff={diff:.2e}")

    return True


def test_window_size_auto():
    """Verify auto-computed dilations for various window sizes."""
    # T=5 (baseline): dilations=[1,2], RF=7
    m5 = TCNModel(window_size=5)
    assert m5.dilations == [1, 2], f"T=5: expected [1,2], got {m5.dilations}"
    assert m5.receptive_field >= 5, f"T=5: RF={m5.receptive_field} < 5"
    print(f"[PASS] T=5: dilations={m5.dilations}, RF={m5.receptive_field}")

    # T=16: dilations=[1,2,4,8], RF=31
    m16 = TCNModel(window_size=16)
    assert m16.dilations == [1, 2, 4, 8], f"T=16: expected [1,2,4,8], got {m16.dilations}"
    assert m16.receptive_field >= 16, f"T=16: RF={m16.receptive_field} < 16"
    print(f"[PASS] T=16: dilations={m16.dilations}, RF={m16.receptive_field}")

    # T=32: dilations=[1,2,4,8,16], RF=63
    m32 = TCNModel(window_size=32)
    assert m32.dilations == [1, 2, 4, 8, 16], f"T=32: expected [1,2,4,8,16], got {m32.dilations}"
    assert m32.receptive_field >= 32, f"T=32: RF={m32.receptive_field} < 32"
    print(f"[PASS] T=32: dilations={m32.dilations}, RF={m32.receptive_field}")

    # T=16 with varying inputs: forward pass should work
    for T in [5, 8, 16, 32]:
        m = TCNModel(window_size=T)
        m.eval()
        x = torch.randn(2, T, 512)
        with torch.no_grad():
            c, f, fn = m(x)
        assert c.shape == (2, T, 16), f"T={T}: cls shape {c.shape}"
        assert f.shape == (2, T, 1), f"T={T}: fall shape {f.shape}"
        print(f"[PASS] T={T}: forward {x.shape} -> cls{c.shape} fall{f.shape}")

    print(f"\n[PASS] All window_size auto-config tests passed!")


if __name__ == "__main__":
    test_forward_shapes()
    test_causality()
    test_window_size_auto()
