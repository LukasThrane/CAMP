"""
saliency.py — compute per-channel saliency from a trained FingerFlex decoder.

What this does
--------------
Loads a trained AutoEncoder1D, computes ∂(prediction)/∂(input) on training
windows, aggregates over wavelets and time and finger outputs, and emits a
single (62,) saliency vector that says "channel c contributes this fraction
to finger decoding."

This is what Neuralite computes (gradient of downstream output w.r.t. input)
applied at the channel level rather than the sample-position level — because
our scheduler operates on whole channels, not individual sample positions.

Output: results/saliency_subject_<N>.npz containing:
  - 'saliency': (62,) float, sums to 1.0
  - 'saliency_raw': (62,) float, unnormalized magnitudes
  - 'per_finger': (5, 62) float, per-finger saliency contributions
  - 'meta': training info (epochs, r_mean) of the decoder used

Usage:
    python saliency.py \\
        --checkpoint /content/drive/MyDrive/CAMP-results/smoke_pA/checkpoints/A_full_decoder.pt \\
        --out-dir /content/drive/MyDrive/CAMP-results/saliency \\
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loader import load_subject_split
from preprocess import preprocess_train_test
from decoder import AutoEncoder1D


# ============================================================================
# Saliency computation
# ============================================================================

def compute_saliency(
    model: AutoEncoder1D,
    features_train: np.ndarray,   # (62, 40, T) float32
    *,
    n_windows: int = 200,
    window_len: int = 256,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """
    Vanilla gradient saliency averaged over many input windows.

    For each window x of shape (62, 40, 256):
      1. Forward: y = model(x).            shape (5, 256)
      2. Aggregate y to a scalar: s = sum(|y|) over fingers and time.
         (Using L1 of the prediction magnitude. We take absolute value before
          summing so positive and negative finger predictions don't cancel.)
      3. Backward: grad = ds/dx.           shape (62, 40, 256)
      4. Channel saliency for this window: |grad| summed over wavelets, time.
                                                                shape (62,)
    Average over n_windows.

    Why this aggregation? The chip schedules at the *channel* level. Wavelets
    and time samples within a channel are bundled together — chip can't drop
    "only channel 17's 80 Hz component." So we collapse those dims by taking
    the L1 magnitude of the gradient.

    Returns a dict with the saliency vector plus per-finger breakdown.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    C, W, T = features_train.shape

    saliency_sum = np.zeros(C, dtype=np.float64)
    per_finger_sum = np.zeros((5, C), dtype=np.float64)

    # Sample n_windows random window starts uniformly over the recording.
    valid_starts = T - window_len
    starts = rng.integers(0, valid_starts, size=n_windows)

    feats_t = torch.from_numpy(features_train).float()

    for i, s in enumerate(starts):
        x = feats_t[..., s : s + window_len].clone().to(device)
        x = x.unsqueeze(0)              # (1, 62, 40, 256)
        x.requires_grad_(True)

        # Forward.
        pred = model(x)                 # (1, 5, 256)

        # Aggregate to scalar. L1 magnitude — avoids cancellation between
        # positive and negative predictions and works for both fingers that
        # are moving and fingers that aren't.
        scalar = pred.abs().sum()

        # Backward — populates x.grad.
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        scalar.backward()

        # Channel saliency for this window: L1 over (wavelet, time).
        g = x.grad.detach().abs()       # (1, 62, 40, 256)
        saliency_window = g.sum(dim=(0, 2, 3)).cpu().numpy()  # (62,)
        saliency_sum += saliency_window

        # Per-finger breakdown — re-run with each finger output isolated.
        # This is more expensive (5 backward passes per window) but tells
        # us which channels contribute to which fingers.
        for f in range(5):
            x_f = feats_t[..., s : s + window_len].clone().to(device)
            x_f = x_f.unsqueeze(0)
            x_f.requires_grad_(True)
            pred_f = model(x_f)
            scalar_f = pred_f[0, f, :].abs().sum()
            model.zero_grad()
            if x_f.grad is not None:
                x_f.grad.zero_()
            scalar_f.backward()
            g_f = x_f.grad.detach().abs().sum(dim=(0, 2, 3)).cpu().numpy()
            per_finger_sum[f] += g_f

        if (i + 1) % 50 == 0:
            print(f"  saliency window {i + 1}/{n_windows}")

    saliency_raw = saliency_sum / n_windows
    per_finger = per_finger_sum / n_windows

    # Normalize so saliency sums to 1 (matches scheduler's weight expectation).
    total = saliency_raw.sum()
    saliency = saliency_raw / total if total > 0 else np.full(C, 1.0 / C)

    return {
        "saliency": saliency,
        "saliency_raw": saliency_raw,
        "per_finger": per_finger,
    }


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_saliency(saliency: np.ndarray, per_finger: np.ndarray,
                  out_path: Path) -> None:
    """Bar chart of per-channel saliency, plus per-finger heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [2, 3]})

    # Top: bar chart of saliency per channel.
    ax = axes[0]
    n_ch = len(saliency)
    colors = plt.cm.viridis(saliency / saliency.max())
    ax.bar(range(n_ch), saliency, color=colors)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Saliency (normalized)")
    ax.set_title("Per-channel saliency for finger decoding")
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.grid(True, alpha=0.3)

    # Highlight the top-K channels.
    K = max(1, n_ch // 10)
    top_k = np.argsort(-saliency)[:K]
    for i in top_k:
        ax.axvline(i, color="red", alpha=0.2, linewidth=1)
    ax.text(0.02, 0.95,
            f"Top {K} channels: {sorted(top_k.tolist())}",
            transform=ax.transAxes, va="top",
            fontsize=9, family="monospace")

    # Bottom: heatmap, fingers × channels.
    ax = axes[1]
    # Normalize each finger row to its own max for visibility.
    pf_norm = per_finger / (per_finger.max(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(pf_norm, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["thumb", "index", "middle", "ring", "little"])
    ax.set_xlabel("Channel index")
    ax.set_title("Per-finger saliency (each row normalized to its own max)")
    plt.colorbar(im, ax=ax, label="Relative saliency")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained AutoEncoder1D checkpoint (.pt file).")
    ap.add_argument("--n-windows", type=int, default=200,
                    help="Number of random training windows for averaging.")
    ap.add_argument("--window-len", type=int, default=256)
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--out-dir", type=str, default="./results/saliency")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint.
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    print(f"  trained to epoch {ckpt.get('epoch', '?')}, "
          f"r_mean = {ckpt.get('r_mean', '?')}")

    model = AutoEncoder1D().to(args.device)
    model.load_state_dict(ckpt["model_state"])

    # Load + preprocess data so we can sample training windows.
    print("Loading data + preprocessing (full-resolution, no chip)…")
    split = load_subject_split(args.subject, cache_dir=Path(args.cache_dir))
    chunk_size = 30
    T_tr = (split["ecog_train"].shape[1] // chunk_size) * chunk_size
    T_te = (split["ecog_test"].shape[1] // chunk_size) * chunk_size
    pre = preprocess_train_test(
        split["ecog_train"][:, :T_tr], split["fingers_train"][:, :T_tr],
        split["ecog_test"][:, :T_te],  split["fingers_test"][:, :T_te],
        sfreq=split["sfreq"], line_freq=args.line_freq, verbose=False,
    )
    feats_train = pre["features_train"]   # (62, 40, T_train)
    print(f"  features_train shape: {feats_train.shape}")

    # Compute saliency.
    print(f"\nComputing saliency over {args.n_windows} random windows…")
    result = compute_saliency(
        model, feats_train,
        n_windows=args.n_windows,
        window_len=args.window_len,
        device=args.device,
    )

    # Report.
    s = result["saliency"]
    K = 10
    top_k_idx = np.argsort(-s)[:K]
    print(f"\nTop {K} channels by saliency:")
    for rank, c in enumerate(top_k_idx, 1):
        print(f"  {rank:2d}. channel {c:2d}  saliency={s[c]:.4f} "
              f"({s[c] * 100:.2f}% of total)")

    # Concentration metric.
    cumulative = np.cumsum(np.sort(s)[::-1])
    print(f"\nCumulative saliency: top 5 = {cumulative[4]*100:.1f}%, "
          f"top 10 = {cumulative[9]*100:.1f}%, "
          f"top 20 = {cumulative[19]*100:.1f}%")
    print(f"  (uniform would give 8.1% / 16.1% / 32.3%)")

    # Save.
    out_npz = out_dir / f"saliency_subject_{args.subject}.npz"
    np.savez(
        out_npz,
        saliency=result["saliency"],
        saliency_raw=result["saliency_raw"],
        per_finger=result["per_finger"],
        meta_epoch=ckpt.get("epoch", -1),
        meta_r_mean=ckpt.get("r_mean", -1.0),
    )
    print(f"\nSaved {out_npz}")

    # Plot.
    plot_saliency(
        result["saliency"], result["per_finger"],
        out_dir / f"saliency_subject_{args.subject}.png",
    )


if __name__ == "__main__":
    main()