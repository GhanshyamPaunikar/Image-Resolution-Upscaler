"""
Fastv2 — lightweight residual super-resolution network.

Architecture:
  1. Shallow feature extraction (single conv)
  2. N residual channel-attention blocks (RCAB) for deep features
  3. Per-scale pixel-shuffle upsampler (2×/3×/4×/6×)
  4. Reconstruction conv → clamped RGB output

Accepts forward(x, upscale_factor) — matches the codebase convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.gap(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            ChannelAttention(channels, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class PixelShuffleUpsampler(nn.Module):
    """Sub-pixel convolution upsampler for a single integer scale."""

    def __init__(self, channels: int, scale: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, 1, 1, bias=True),
            nn.PixelShuffle(scale),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """
    Fastv2 SR model.  Supports upscale_factor ∈ {2, 3, 4, 6}.
    """

    SUPPORTED_SCALES = (2, 3, 4, 6)

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        ca_reduction: int = 16,
    ):
        super().__init__()
        self.shallow = nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features, ca_reduction) for _ in range(num_blocks)]
        )
        self.body_end = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

        self.upsamplers = nn.ModuleDict({
            str(s): PixelShuffleUpsampler(num_features, s)
            for s in self.SUPPORTED_SCALES
        })

        self.tail = nn.Conv2d(num_features, in_channels, 3, 1, 1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        upscale_factor: int = 4,
        **_kwargs,
    ) -> torch.Tensor:
        if upscale_factor not in self.SUPPORTED_SCALES:
            raise ValueError(
                f"upscale_factor must be one of {self.SUPPORTED_SCALES}, got {upscale_factor}"
            )

        shallow = self.shallow(x)
        deep = self.body_end(self.body(shallow)) + shallow  # global residual
        up = self.upsamplers[str(upscale_factor)](deep)
        return self.tail(up).clamp(0.0, 1.0)
