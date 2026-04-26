"""
train.py — FingerFlex-faithful training loop.

Key choices (matching their Lightning_BCI-autoencoder.ipynb):
  - Window length 256 samples (2.56 s at 100 Hz).
  - Stride-1 windowing: every time index produces a window. With ~39000
    train timesteps after the 200ms delay shift, that's ~38744 windows
    per epoch.
  - Batch size 128 (their default).
  - Adam, lr=8.42e-5, weight_decay=1e-6.
  - Loss: 0.5*MSE + 0.5*(1 - cosine_similarity).
  - Validation: predict the FULL test sequence in one forward pass
    (chunked to avoid OOM), then Pearson r per finger over the whole
    sequence (matches FingerFlex's ValidationCallback).
  - Save best model by mean Pearson r.

We deliberately DO NOT use:
  - random window sampling (we use deterministic stride-1)
  - val_loss for checkpointing (we use Pearson r, like the paper)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from decoder import AutoEncoder1D, fingerflex_loss


# ============================================================================
# Dataset — stride-1 windows
# ============================================================================

class StridedWindowDataset(Dataset):
    """Stride-1 sliding windows over the recording.

    features: (n_electrodes, n_freqs, T) — wavelet spectrograms
    fingers:  (5, T)
    sample_len: window size in samples (256 in paper)

    Returns (x, y) where x has shape (n_electrodes, n_freqs, sample_len)
    and y has shape (5, sample_len).
    """
    def __init__(
        self, features: np.ndarray, fingers: np.ndarray, sample_len: int = 256,
    ):
        assert features.ndim == 3, f"features should be (C, W, T), got {features.shape}"
        assert fingers.ndim == 2, f"fingers should be (5, T), got {fingers.shape}"
        assert features.shape[-1] == fingers.shape[-1]
        self.features = torch.from_numpy(features).float()
        self.fingers = torch.from_numpy(fingers).float()
        self.sample_len = sample_len
        self.duration = features.shape[-1]
        # Stride 1, so each window starts at index 0..duration-sample_len.
        self.n_windows = self.duration - sample_len

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        s = idx
        e = idx + self.sample_len
        return self.features[..., s:e], self.fingers[..., s:e]


# ============================================================================
# Pearson r evaluation
# ============================================================================

def pearson_r_per_finger(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(5, T) → (5,) per-finger Pearson r."""
    assert pred.shape == true.shape
    r = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        r[i] = np.corrcoef(pred[i], true[i])[0, 1]
    return r


@torch.no_grad()
def predict_full_sequence(
    model: AutoEncoder1D, features: np.ndarray, device: str,
) -> np.ndarray:
    """Run the model over the entire test sequence in one forward pass.

    Matches FingerFlex's ValidationCallback: feed the whole sequence as
    one batch, get predictions, evaluate Pearson r over the whole thing.

    features: (n_electrodes, n_freqs, T) numpy array.
    Returns: (5, T) predictions.
    """
    model.eval()
    # FingerFlex truncates to a multiple of 64 (their SIZE constant). We
    # do the same to match — and so the forward pass through 5 stride-2
    # downsamples doesn't leave a fractional sample at the end.
    SIZE = 64
    T = features.shape[-1]
    bound = (T // SIZE) * SIZE

    x = torch.from_numpy(features[..., :bound]).float().to(device)
    # Add batch dim: (C, W, T) → (1, C, W, T)
    x = x.unsqueeze(0)

    y = model(x)[0]                    # (5, bound)
    return y.cpu().numpy()


# ============================================================================
# Training
# ============================================================================

def train_from_arrays(
    features_train: np.ndarray,
    fingers_train: np.ndarray,
    features_test: np.ndarray,
    fingers_test: np.ndarray,
    sample_len: int = 256,
    batch_size: int = 128,
    n_epochs: int = 30,
    lr: float = 8.42e-5,
    weight_decay: float = 1e-6,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
    ckpt_path: Optional[Path] = None,
    num_workers: int = 2,
) -> tuple[AutoEncoder1D, list]:
    """Train one FingerFlex decoder. Returns (best_model, history).

    features_train: (62, 40, T_train) float32
    fingers_train:  (5, T_train) float32 (MinMax scaled)
    features_test:  (62, 40, T_test)  float32  (validation set)
    fingers_test:   (5, T_test) float32 (MinMax scaled)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_electrodes, n_freqs, _ = features_train.shape

    train_ds = StridedWindowDataset(features_train, fingers_train, sample_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=(device == "cuda"),
    )

    model = AutoEncoder1D(
        n_electrodes=n_electrodes,
        n_freqs=n_freqs,
        n_channels_out=fingers_train.shape[0],
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model: {n_params:,} parameters")
        print(f"  train windows: {len(train_ds)}, batches/epoch: {len(train_loader)}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_r = -np.inf
    best_state = None
    history = []

    for epoch in range(1, n_epochs + 1):
        # --- train ---
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = fingerflex_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        # --- validate: full-sequence Pearson r on test set ---
        pred_full = predict_full_sequence(model, features_test, device)
        # Match the time length of the prediction (truncated to SIZE multiple).
        T_pred = pred_full.shape[-1]
        true = fingers_test[..., :T_pred]
        r_per = pearson_r_per_finger(pred_full, true)
        r_mean = float(np.mean(r_per))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "r_per_finger": r_per.tolist(),
            "r_mean": r_mean,
        })
        if verbose:
            print(f"  epoch {epoch:2d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"r_mean={r_mean:.3f}  "
                  f"per=[{', '.join(f'{r:.2f}' for r in r_per)}]")

        if r_mean > best_r:
            best_r = r_mean
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save({
                    "model_state": best_state,
                    "epoch": epoch, "r_mean": r_mean,
                    "r_per_finger": r_per.tolist(),
                }, ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    from data_loader import load_subject_split
    from preprocess import preprocess_train_test

    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    args = ap.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading subject {args.subject}…")
    split = load_subject_split(args.subject, cache_dir=Path(args.cache_dir))

    print("Preprocessing…")
    pre = preprocess_train_test(
        split["ecog_train"], split["fingers_train"],
        split["ecog_test"],  split["fingers_test"],
        sfreq=split["sfreq"], line_freq=args.line_freq, verbose=True,
    )
    print(f"  features_train: {pre['features_train'].shape}")
    print(f"  fingers_train:  {pre['fingers_train'].shape}")
    print(f"  features_test:  {pre['features_test'].shape}")
    print(f"  fingers_test:   {pre['fingers_test'].shape}")

    ckpt = Path(args.ckpt_dir) / f"subject_{args.subject}_best.pt"
    model, history = train_from_arrays(
        features_train=pre["features_train"],
        fingers_train=pre["fingers_train"],
        features_test=pre["features_test"],
        fingers_test=pre["fingers_test"],
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=ckpt,
    )
    best = max(history, key=lambda h: h["r_mean"])
    print(f"\nBest r_mean: {best['r_mean']:.3f} at epoch {best['epoch']}")
    print(f"Per finger:  {best['r_per_finger']}")
    print(f"Paper baseline for Subject 1: 0.66")


if __name__ == "__main__":
    main()