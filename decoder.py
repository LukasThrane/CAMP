"""
decoder.py — exact FingerFlex AutoEncoder1D (Lomtev et al. 2022).

Reimplements the AutoEncoder1D from the Lightning_BCI-autoencoder.ipynb
notebook in their public repo (Irautak/FingerFlex). Architecture details
verified against their code:

  - 5 encoder stages, channel widths from `channels = [32, 32, 64, 64, 128, 128]`
    where the FIRST entry is the spatial-reduce output, and the remaining 5
    are the per-stage encoder widths.
  - kernel_sizes = [7, 7, 5, 5, 5]   (varied across stages, larger early)
  - strides = [2, 2, 2, 2, 2]
  - Each ConvBlock: Conv1d (no bias, padding='same') → LayerNorm → GELU
                   → Dropout(0.1) → MaxPool1d(stride=2)
  - LayerNorm normalizes the embedding axis (transpose, ln, transpose).
  - Decoder mirrors encoder, with skip connections concatenated AFTER upsample.
  - Final 1x1 conv projects to 5 fingers.
  - ~600k parameters total.

Loss (in the training notebook):
    F.mse_loss(y_hat, y) returned, then 0.5*loss + 0.5*(1 - cosine_similarity)
    where cosine_similarity is along the time axis, averaged over batch+fingers.

Optimizer: Adam, lr=8.42e-5, weight_decay=1e-6.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d → LayerNorm(over channels) → GELU → Dropout → MaxPool.

    Matches FingerFlex's ConvBlock exactly:
      - bias=False on the conv
      - padding='same'
      - LayerNorm applied to the channel axis (transpose-norm-transpose)
      - MaxPool1d(kernel_size=stride, stride=stride) for downsampling.
    """
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, dilation: int = 1, p_drop: float = 0.1,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, bias=False, padding="same",
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=p_drop)
        # MaxPool with kernel=stride downsamples by `stride`. When stride=1,
        # this is a no-op (kernel=1 means the pool sees one sample at a time).
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        # LayerNorm over the channel dim — transpose so channels are last.
        x = x.transpose(-2, -1)
        x = self.norm(x)
        x = x.transpose(-2, -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.downsample(x)
        return x


class UpConvBlock(nn.Module):
    """Decoder block: ConvBlock then linear upsample by `scale`.

    FingerFlex applies the conv FIRST (with the post-concat channel count)
    then upsamples — matches their UpConvBlock exactly.
    """
    def __init__(self, scale: int, **conv_args):
        super().__init__()
        self.conv_block = ConvBlock(**conv_args)
        self.upsample = nn.Upsample(
            scale_factor=scale, mode="linear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class AutoEncoder1D(nn.Module):
    """FingerFlex encoder-decoder.

    Args (all defaults match FingerFlex):
        n_electrodes: 62 for BCI Comp IV (number of ECoG channels)
        n_freqs:      40 (number of Morlet wavelets)
        n_channels_out: 5 (fingers)
        channels:     [32, 32, 64, 64, 128, 128]   (1 spatial_reduce + 5 stages)
        kernel_sizes: [7, 7, 5, 5, 5]              (per encoder stage)
        strides:      [2, 2, 2, 2, 2]              (per encoder stage)
    """
    def __init__(
        self,
        n_electrodes: int = 62,
        n_freqs: int = 40,
        n_channels_out: int = 5,
        channels: list = (32, 32, 64, 64, 128, 128),
        kernel_sizes: list = (7, 7, 5, 5, 5),
        strides: list = (2, 2, 2, 2, 2),
        dilation: list = (1, 1, 1, 1, 1),
    ):
        super().__init__()
        channels = list(channels)
        kernel_sizes = list(kernel_sizes)
        strides = list(strides)
        dilation = list(dilation)

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_electrodes * n_freqs
        self.n_channels_out = n_channels_out
        self.model_depth = len(channels) - 1  # 5

        # Spatial-reduce: one ConvBlock with kernel_size=3, stride=1.
        # Reduces 62*40=2480 features down to channels[0]=32.
        self.spatial_reduce = ConvBlock(
            self.n_inp_features, channels[0], kernel_size=3,
        )

        # Encoder: 5 ConvBlocks, each downsampling by stride[i].
        self.downsample_blocks = nn.ModuleList([
            ConvBlock(
                channels[i], channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                dilation=dilation[i],
            )
            for i in range(self.model_depth)
        ])

        # Decoder: 5 UpConvBlocks (in reverse). The input to each block is
        # `channels[i+1]` from the previous decoder stage if it's the first
        # decoder block (deepest in U), else `channels[i+1]*2` because of
        # the skip-connection concat from the matching encoder stage.
        self.upsample_blocks = nn.ModuleList([
            UpConvBlock(
                scale=strides[i],
                in_channels=(channels[i + 1] if i == self.model_depth - 1
                             else channels[i + 1] * 2),
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
            )
            for i in range(self.model_depth - 1, -1, -1)
        ])

        # Final 1x1 conv: takes channels[0]*2 (because of the FIRST skip
        # concat that happens AFTER the last decoder block) → 5 fingers.
        self.conv1x1 = nn.Conv1d(
            channels[0] * 2, n_channels_out, kernel_size=1, padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: (batch, n_electrodes, n_freqs, time).

        FingerFlex flattens the (electrodes, freqs) dims into the channel
        dimension internally. We support both:
          - (B, n_electrodes, n_freqs, T) → reshape to (B, n_electrodes*n_freqs, T)
          - (B, n_electrodes*n_freqs, T)  → use as-is
        """
        if x.ndim == 4:
            B, _, _, T = x.shape
            x = x.reshape(B, -1, T)

        x = self.spatial_reduce(x)

        # Encoder pass with skip connections saved BEFORE downsampling.
        skips = []
        for i in range(self.model_depth):
            skips.append(x)
            x = self.downsample_blocks[i](x)

        # Decoder pass — upsample, then concat skip from matching encoder
        # level. (FingerFlex's order: upsample first, then concat after.)
        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat([x, skips[-1 - i]], dim=1)

        x = self.conv1x1(x)
        return x


# ============================================================================
# Loss
# ============================================================================

def fingerflex_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * MSE + 0.5 * (1 - cosine_similarity).

    Matches FingerFlex's training_step exactly:
        loss = F.mse_loss(y_hat, y)
        corr = mean(cosine_similarity(y_hat, y, dim=-1))
        return 0.5*loss + 0.5*(1. - corr)
    """
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    return 0.5 * mse + 0.5 * (1.0 - cos)


# Backward compat — older code imports `FingerFlex`.
FingerFlex = AutoEncoder1D


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    model = AutoEncoder1D(n_electrodes=62, n_freqs=40, n_channels_out=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}  (paper: ~600k)")

    # Test 4D input: (B, n_electrodes, n_freqs, T)
    x4 = torch.randn(2, 62, 40, 256)
    y = model(x4)
    print(f"  (B,62,40,256) → {tuple(y.shape)}  expected (2, 5, 256)")
    assert y.shape == (2, 5, 256)

    # Test 3D input: (B, n_electrodes*n_freqs, T)
    x3 = torch.randn(2, 62 * 40, 256)
    y = model(x3)
    print(f"  (B,2480,256)  → {tuple(y.shape)}")
    assert y.shape == (2, 5, 256)

    # Loss runs.
    target = torch.randn(2, 5, 256)
    loss = fingerflex_loss(y, target)
    print(f"  loss = {loss.item():.4f}")
    print("decoder.py OK")