"""
共享 LoRA：低秩适配注入 + 合并（不依赖 peft，autodl 直接用）。

从 train_p9_e3_stage1.py（E3，已在 Autodl 上跑通）抽出的手写实现：
冻结原始权重，在注意力 q/k/v 与 output.dense 投影上插入低秩 A/B
（B 置零、A kaiming 初始化），训练只更新 A/B；推理前用 merge_lora 把
A/B 合并回权重、替换成普通 Linear。

接口:
    LoRALinear(original: nn.Linear, r, alpha) -> nn.Module  带 A/B 的低秩投影包装
    inject_lora(model, r, alpha, targets) -> None           就地注入 q/k/v + output.dense
    merge_lora(model) -> None                                把 A/B 合并回权重，替换为普通 Linear（推理用）
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int = 16, alpha: int = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.original(x)
        return base + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())


def inject_lora(model, r=16, alpha=16, targets=("query", "key", "value")):
    """在注意力 q/k/v 和 output.dense 上注入 LoRA。返回可训练参数名列表。"""
    for layer in model.encoder.layer:
        attn = layer.attention.attention
        for name in targets:
            setattr(attn, name, LoRALinear(getattr(attn, name), r, alpha))
        layer.attention.output.dense = LoRALinear(layer.attention.output.dense, r, alpha)


def merge_lora(model):
    """LoRALinear 的 A/B 合并回权重，替换为普通 Linear（推理用）。"""
    def _merge(module, name):
        mod = getattr(module, name)
        if isinstance(mod, nn.Linear):
            return
        delta = (mod.scaling * (mod.lora_B @ mod.lora_A)).to(mod.original.weight.dtype)
        lin = nn.Linear(mod.original.in_features, mod.original.out_features,
                        bias=mod.original.bias is not None)
        lin.weight.data = mod.original.weight.data + delta
        if mod.original.bias is not None:
            lin.bias.data = mod.original.bias.data
        setattr(module, name, lin)

    for layer in model.encoder.layer:
        attn = layer.attention.attention
        for name in ("query", "key", "value"):
            _merge(attn, name)
        _merge(layer.attention.output, "dense")
