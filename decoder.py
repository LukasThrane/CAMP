"""
decoder.py — FingerFlex 1D U-Net (Lomtev et al. 2022, arXiv:2211.01960).

Off-chip decoder. Takes wavelet features (62*40=2480 features × T at 100 Hz),
predicts 5 finger trajectories. Architecture per Figure 2 of the paper:
  - Input: (B, 2480, T)
  - 1×1 input projection → 32
  - 6 encoder stages, widths (32, 32, 64, 64, 128, 128)
    Each: Conv1d → LayerNorm → GELU → Dropout → MaxPool(stride=2)
  - 6 decoder stages mirrored, with skip connections
    Each: Upsample(2×) → concat skip → Conv1d → GELU → Dropout
  - 1×1 head → 5 fingers
  - ~600k parameters total

Loss: 0.5 * MSE + 0.5 * (1 - cosine_similarity)
Optimizer: Adam, lr=8.4e-5
Target: ~0.66 Pearson r on Subject 1 (paper baseline)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dim of (B, C, T) — transpose, ln, transpose back."""
    def __init__(self, n_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class EncoderBlock(nn.Module):
    """Conv → LN → GELU → Dropout → MaxPool. Returns (skip, pooled)."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.norm = ChannelLayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        h = self.drop(self.act(self.norm(self.conv(x))))
        return h, self.pool(h)


class DecoderBlock(nn.Module):
    """Upsample → concat skip → Conv → GELU → Dropout."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.drop(self.act(self.conv(x)))


class FingerFlex(nn.Module):
    """Convolutional encoder-decoder for finger trajectory regression.

    Args:
        n_input_features: C * W. For our pipeline 62 * 40 = 2480.
        feature_widths: encoder widths. Paper: (32, 32, 64, 64, 128, 128).
        n_outputs: 5 fingers.
    """
    def __init__(
        self,
        n_input_features: int,
        feature_widths: tuple = (32, 32, 64, 64, 128, 128),
        n_outputs: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_input_features, feature_widths[0], kernel_size=1)

        self.encoders = nn.ModuleList()
        prev = feature_widths[0]
        for w in feature_widths:
            self.encoders.append(EncoderBlock(prev, w, dropout=dropout))
            prev = w

        self.decoders = nn.ModuleList()
        widths_rev = list(feature_widths)[::-1]
        cur = widths_rev[0]
        for i, w in enumerate(widths_rev):
            skip_w = w
            out_w = widths_rev[i + 1] if i + 1 < len(widths_rev) else w
            self.decoders.append(DecoderBlock(cur, skip_w, out_w, dropout=dropout))
            cur = out_w

        self.head = nn.Conv1d(cur, n_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


def fingerflex_loss(pred: torch.Tensor, target: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
    """0.5 * MSE + 0.5 * (1 - cosine over time, averaged over fingers).
    Paper ablation: cosine alone r=0.44, MSE alone r=0.64, combo r=0.66."""
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    return alpha * mse + (1.0 - alpha) * (1.0 - cos)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    n_features = 62 * 40
    model = FingerFlex(n_input_features=n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}  (paper: ~600k)")

    for T in [256, 200, 1000]:
        x = torch.randn(2, n_features, T)
        y = model(x)
        print(f"  T_in={T:4d} → y.shape={tuple(y.shape)}")
        assert y.shape == (2, 5, T)

    target = torch.randn(2, 5, 256)
    pred = model(torch.randn(2, n_features, 256))
    loss = fingerflex_loss(pred, target)
    print(f"  loss = {loss.item():.4f}")
    print("decoder.py OK")