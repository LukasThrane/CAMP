"""
train.py — train one FingerFlex decoder on preprocessed data.

Usage as standalone:
    python preprocess.py --subject 1
    python train.py --subject 1

Or imported by the experiment runner:
    from train import train_from_arrays
    history = train_from_arrays(features, fingers, n_train, ...)

Per the paper:
  - lr = 8.4e-5 (fixed)
  - Adam optimizer
  - 30 epochs
  - Window length 256 samples (2.56s @ 100Hz)
  - Loss: 0.5*MSE + 0.5*(1 - cosine)
  - Save best by validation Pearson r
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from decoder import FingerFlex, fingerflex_loss


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class WindowDataset(Dataset):
    """Slice a long recording into windows.

    features: (F, T) flattened — already shape (C*W, T).
    fingers:  (5, T)
    mode='random':   sample windows uniformly at random from [0, T - window_len]
    mode='tiled':    non-overlapping windows for deterministic validation loss
    """
    def __init__(
        self,
        features: np.ndarray,
        fingers: np.ndarray,
        window_len: int,
        mode: str = "random",
        n_windows: Optional[int] = None,
    ):
        assert features.ndim == 2 and fingers.ndim == 2
        assert features.shape[1] == fingers.shape[1]
        assert mode in ("random", "tiled")
        self.features = torch.from_numpy(features).float()
        self.fingers = torch.from_numpy(fingers).float()
        self.window_len = window_len
        self.mode = mode
        self.T = features.shape[1]

        if mode == "random":
            self.n_windows = n_windows or max(1, self.T // window_len * 4)
        else:
            self.n_windows = max(1, self.T // window_len)
            self.starts = np.arange(self.n_windows) * window_len

    def __len__(self): return self.n_windows

    def __getitem__(self, idx: int):
        if self.mode == "random":
            start = np.random.randint(0, self.T - self.window_len + 1)
        else:
            start = self.starts[idx]
        end = start + self.window_len
        return self.features[:, start:end], self.fingers[:, start:end]


# --------------------------------------------------------------------------- #
# Pearson r
# --------------------------------------------------------------------------- #

def pearson_r_per_finger(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(5, T) → (5,) per-finger Pearson r."""
    assert pred.shape == true.shape
    r = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        r[i] = np.corrcoef(pred[i], true[i])[0, 1]
    return r


@torch.no_grad()
def predict_full_sequence(
    model: FingerFlex, features: np.ndarray, chunk_len: int, device: str
) -> np.ndarray:
    """Run the model over a full sequence in non-overlapping chunks. Fully
    convolutional → output time matches input."""
    model.eval()
    T = features.shape[1]
    out = np.zeros((5, T), dtype=np.float32)
    x_full = torch.from_numpy(features).float().to(device)
    for s in range(0, T, chunk_len):
        e = min(s + chunk_len, T)
        x = x_full[:, s:e].unsqueeze(0)
        out[:, s:e] = model(x).squeeze(0).cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_from_arrays(
    features: np.ndarray,   # (C, W, T) float32
    fingers: np.ndarray,    # (5, T) float32
    n_train: int,
    window_len: int = 256,
    batch_size: int = 16,
    n_epochs: int = 30,
    lr: float = 8.4e-5,
    n_windows_per_epoch: int = 2048,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
    ckpt_path: Optional[Path] = None,
) -> tuple[FingerFlex, list]:
    """Core training routine. Returns (best_model, history).

    The experiment runner calls this for each (strategy, bandwidth) cell.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    C, W, T = features.shape
    feats_flat = features.reshape(C * W, T)

    train_feats = feats_flat[:, :n_train]
    train_fings = fingers[:, :n_train]
    val_feats = feats_flat[:, n_train:]
    val_fings = fingers[:, n_train:]

    train_ds = WindowDataset(train_feats, train_fings, window_len,
                             mode="random", n_windows=n_windows_per_epoch)
    val_ds = WindowDataset(val_feats, val_fings, window_len, mode="tiled")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    model = FingerFlex(n_input_features=C * W).to(device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_r = -np.inf
    best_state = None
    history = []

    for epoch in range(1, n_epochs + 1):
        # Train.
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = fingerflex_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        # Val loss.
        model.eval()
        v_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_losses.append(fingerflex_loss(model(x), y).item())
        val_loss = float(np.mean(v_losses))

        # Pearson r over full val sequence.
        pred_full = predict_full_sequence(model, val_feats, 1024, device)
        r_per = pearson_r_per_finger(pred_full, val_fings)
        r_mean = float(np.mean(r_per))

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "r_per_finger": r_per.tolist(), "r_mean": r_mean,
        })
        if verbose:
            print(f"  epoch {epoch:2d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--preproc-dir", type=str, default="./preprocessed")
    ap.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    args = ap.parse_args()

    pkl = Path(args.preproc_dir) / f"subject_{args.subject}.pkl"
    print(f"Loading {pkl}…")
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir) / f"subject_{args.subject}_best.pt"

    model, history = train_from_arrays(
        features=data["features"],
        fingers=data["fingers"],
        n_train=data["n_train"],
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=ckpt,
    )
    best = max(history, key=lambda h: h["r_mean"])
    print(f"\nBest r_mean: {best['r_mean']:.3f} at epoch {best['epoch']}")
    print(f"Paper baseline for Subject 1: 0.66")


if __name__ == "__main__":
    main()