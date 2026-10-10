"""Time-domain SimCLR views; signal adaptations rather than original image transforms."""

import torch
from torch import nn


class SignalAugmenter(nn.Module):
    def __init__(self, jitter_std=0.02, scale_std=0.1, crop_min_ratio=0.8, mask_ratio=0.05):
        super().__init__()
        if jitter_std < 0 or scale_std < 0 or not 0 < crop_min_ratio <= 1 or not 0 <= mask_ratio < 1:
            raise ValueError("Invalid signal augmentation strengths")
        self.jitter_std, self.scale_std = jitter_std, scale_std
        self.crop_min_ratio, self.mask_ratio = crop_min_ratio, mask_ratio

    @torch.no_grad()
    def forward(self, signals, generator=None):
        if signals.ndim != 3 or signals.shape[1] < 2 or not signals.is_floating_point():
            raise ValueError("Augmentation expects floating-point (B,L,C), L >= 2")
        batch, length, channels = signals.shape
        device, dtype = signals.device, signals.dtype
        output = signals.clone()
        if self.crop_min_ratio < 1:
            # 每个样本独立随机裁剪，再线性插值回原长度，保持批次形状一致。
            sizes = (length * (self.crop_min_ratio + (1 - self.crop_min_ratio)
                               * torch.rand(batch, device=device, generator=generator))).long().clamp(min=2)
            starts = (torch.rand(batch, device=device, generator=generator) * (length - sizes + 1)).floor()
            positions = starts[:, None] + torch.linspace(0, 1, length, device=device)[None] * (sizes[:, None] - 1)
            lower = positions.long()
            upper = (lower + 1).clamp(max=length - 1)
            weight = (positions - lower).unsqueeze(-1).to(dtype)
            output = (output.gather(1, lower[..., None].expand(-1, -1, channels)) * (1 - weight)
                      + output.gather(1, upper[..., None].expand(-1, -1, channels)) * weight)
        if self.scale_std:
            # 整段幅值缩放与逐点噪声模拟测量扰动，不修改原始输入。
            output = output * (1 + self.scale_std * torch.randn((batch, 1, channels), device=device,
                                                                dtype=dtype, generator=generator))
        if self.jitter_std:
            output = output + self.jitter_std * torch.randn(output.shape, device=device, dtype=dtype,
                                                           generator=generator)
        count = int(length * self.mask_ratio)
        if count:
            # 屏蔽一段连续时间片，促使特征不过度依赖单个局部区域。
            starts = torch.randint(length - count + 1, (batch, 1), device=device, generator=generator)
            positions = torch.arange(length, device=device)[None]
            output = output.masked_fill(((positions >= starts) & (positions < starts + count))[..., None], 0)
        return output
