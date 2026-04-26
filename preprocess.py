"""
preprocess.py — SERVER-SIDE preprocessing for the off-chip decoder.

Runs FingerFlex-style preprocessing (Lomtev et al. 2022) on whatever the
chip transmitted: bandpass + 40 Morlet wavelets + decimate to 100 Hz +
RobustScaler. None of this runs on the implant — it's the server taking
the (held + reconstructed) stream and turning it into decoder features.

Inputs:
  - ECoG: (62, T) at 1 kHz — possibly chip-simulated with sample-and-hold
  - Fingers: (5, T) at 1 kHz — ground truth, NaN-interpolated

Outputs:
  - features: (62, 40, T_100hz) float32 — wavelet magnitudes, scaled
  - fingers:  (5, T_100hz) float32     — interpolated, MinMax scaled
  - scalers fit on train split only

This module exposes two functions:
  - run_preprocessing_from_arrays(ecog, fingers, sfreq, ...) — for the
    experiment runner, which feeds it chip-simulated streams
  - main()  — CLI for one-off preprocessing of the raw dataset
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, decimate, fftconvolve
from scipy.interpolate import CubicSpline


# --------------------------------------------------------------------------- #
# Morlet wavelets
# --------------------------------------------------------------------------- #

def make_morlet_bank(
    sfreq: float,
    n_wavelets: int = 40,
    f_min: float = 40.0,
    f_max: float = 300.0,
    n_cycles: float = 7.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Complex Morlet wavelets, log-spaced. Returns (freqs, kernels)."""
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_wavelets)
    kernels = []
    for f in freqs:
        sigma_t = n_cycles / (2 * np.pi * f)
        half_len = int(np.ceil(3.0 * sigma_t * sfreq))
        t = np.arange(-half_len, half_len + 1) / sfreq
        env = np.exp(-(t ** 2) / (2 * sigma_t ** 2))
        env /= np.sqrt(np.sum(env ** 2))
        carrier = np.exp(2j * np.pi * f * t)
        kernels.append((env * carrier).astype(np.complex64))
    return freqs, kernels


def convolve_morlet_bank(
    ecog: np.ndarray, kernels: list[np.ndarray], verbose: bool = True
) -> np.ndarray:
    """(C, T) → (C, n_wavelets, T) magnitudes."""
    C, T = ecog.shape
    n_wav = len(kernels)
    out = np.empty((C, n_wav, T), dtype=np.float32)
    for c in range(C):
        for w, k in enumerate(kernels):
            conv = fftconvolve(ecog[c], k, mode="same")
            out[c, w] = np.abs(conv).astype(np.float32)
        if verbose and (c + 1) % 10 == 0:
            print(f"    wavelet conv: channel {c + 1}/{C}")
    return out


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def preprocess_ecog_raw(
    ecog: np.ndarray,
    sfreq: float,
    line_freq: float = 60.0,
    bandpass: tuple[float, float] = (40.0, 300.0),
) -> np.ndarray:
    """Standardize + median subtract + bandpass + notch line frequency.

    Subject 1 is from the US (UW) so line noise = 60 Hz. Pass 50 for European.
    """
    ecog = ecog.astype(np.float64)
    ecog = (ecog - ecog.mean(axis=1, keepdims=True)) / (
        ecog.std(axis=1, keepdims=True) + 1e-8
    )
    ecog = ecog - np.median(ecog, axis=1, keepdims=True)

    nyq = sfreq / 2.0
    b, a = butter(4, [bandpass[0] / nyq, bandpass[1] / nyq], btype="band")
    ecog = filtfilt(b, a, ecog, axis=1)

    h = line_freq
    while h < bandpass[1]:
        if h >= bandpass[0]:
            bn, an = iirnotch(h / nyq, Q=30.0)
            ecog = filtfilt(bn, an, ecog, axis=1)
        h += line_freq
    return ecog


# --------------------------------------------------------------------------- #
# Finger preprocessing
# --------------------------------------------------------------------------- #

def fill_nan_by_interpolation(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    idx = np.arange(x.shape[1])
    for ch in range(x.shape[0]):
        valid = ~np.isnan(x[ch])
        if valid.sum() == 0:
            out[ch] = 0.0
        else:
            out[ch] = np.interp(idx, idx[valid], x[ch, valid])
    return out


def upsample_fingers_bicubic(
    fingers: np.ndarray, sfreq_in: float, sfreq_out: float
) -> np.ndarray:
    n_out = int(np.round(fingers.shape[1] * sfreq_out / sfreq_in))
    t_in = np.arange(fingers.shape[1]) / sfreq_in
    t_out = np.arange(n_out) / sfreq_out
    out = np.empty((fingers.shape[0], n_out), dtype=np.float64)
    for ch in range(fingers.shape[0]):
        out[ch] = CubicSpline(t_in, fingers[ch])(t_out)
    return out


# --------------------------------------------------------------------------- #
# Scalers — fit on train only
# --------------------------------------------------------------------------- #

@dataclass
class RobustScaler1D:
    q_low: float = 0.1
    q_high: float = 0.9
    median_: np.ndarray = None
    low_: np.ndarray = None
    high_: np.ndarray = None
    scale_: np.ndarray = None

    def fit(self, x: np.ndarray):
        self.median_ = np.median(x, axis=-1, keepdims=True)
        self.low_ = np.quantile(x, self.q_low, axis=-1, keepdims=True)
        self.high_ = np.quantile(x, self.q_high, axis=-1, keepdims=True)
        self.scale_ = self.high_ - self.low_ + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self.low_, self.high_)
        return ((x - self.median_) / self.scale_).astype(np.float32)


@dataclass
class MinMaxScaler1D:
    min_: np.ndarray = None
    max_: np.ndarray = None

    def fit(self, x: np.ndarray):
        self.min_ = x.min(axis=-1, keepdims=True)
        self.max_ = x.max(axis=-1, keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.min_) / (self.max_ - self.min_ + 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #

def run_preprocessing_from_arrays(
    ecog: np.ndarray,
    fingers: np.ndarray,
    sfreq: float,
    n_wavelets: int = 40,
    target_sfreq: float = 100.0,
    delay_ms: float = 20.0,
    train_frac: float = 0.65,
    line_freq: float = 60.0,
    verbose: bool = True,
) -> dict:
    """Run the full preprocessing pipeline on already-loaded arrays.

    The experiment runner calls this on chip-simulated ECoG streams. The CLI
    `main()` calls it on the raw dataset.

    Returns a dict with features, fingers, scalers, train split index, etc.
    """
    if verbose:
        print(f"  [preprocess] ECoG: {ecog.shape}, fs={sfreq}")

    if verbose: print("  [preprocess] filtering (bandpass + notch)…")
    ecog = preprocess_ecog_raw(ecog, sfreq, line_freq=line_freq)

    if verbose: print(f"  [preprocess] building {n_wavelets} Morlet wavelets…")
    freqs, kernels = make_morlet_bank(sfreq, n_wavelets=n_wavelets)

    if verbose: print("  [preprocess] convolving (slow step)…")
    feats_full = convolve_morlet_bank(ecog, kernels, verbose=verbose)
    if verbose: print(f"    shape: {feats_full.shape}")

    if verbose: print(f"  [preprocess] decimating {sfreq:.0f} → {target_sfreq:.0f} Hz…")
    decim = int(sfreq // target_sfreq)
    feats = decimate(feats_full, decim, axis=-1, ftype="iir").astype(np.float32)
    del feats_full

    if verbose: print("  [preprocess] interpolating fingers…")
    fingers_clean = fill_nan_by_interpolation(fingers)
    fingers_resamp = upsample_fingers_bicubic(fingers_clean, sfreq, target_sfreq)

    # Align lengths.
    T_min = min(feats.shape[-1], fingers_resamp.shape[-1])
    feats = feats[..., :T_min]
    fingers_resamp = fingers_resamp[..., :T_min]

    if verbose: print(f"  [preprocess] applying {delay_ms:.0f}ms ECoG-fingers shift…")
    delay = int(round(delay_ms * target_sfreq / 1000.0))
    if delay > 0:
        feats = feats[..., :-delay]
        fingers_resamp = fingers_resamp[..., delay:]

    n_total = feats.shape[-1]
    n_train = int(n_total * train_frac)

    if verbose: print("  [preprocess] fitting scalers on train split…")
    fs = RobustScaler1D().fit(feats[..., :n_train])
    ms = MinMaxScaler1D().fit(fingers_resamp[..., :n_train])

    return {
        "features": fs.transform(feats),
        "fingers": ms.transform(fingers_resamp),
        "sfreq": target_sfreq,
        "n_train": n_train,
        "feat_scaler": fs,
        "finger_scaler": ms,
        "wavelet_freqs": freqs,
    }


def main():
    """CLI: preprocess the raw dataset (no chip simulation) and save."""
    from braindecode.datasets import BCICompetitionIVDataset4

    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="./preprocessed")
    ap.add_argument("--line-freq", type=float, default=60.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading subject {args.subject}…")
    ds = BCICompetitionIVDataset4(subject_ids=[args.subject])
    raw = ds.datasets[0].raw
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    assert sfreq == 1000.0
    ecog, fingers = data[:62], data[62:67]

    out = run_preprocessing_from_arrays(
        ecog, fingers, sfreq, line_freq=args.line_freq
    )
    out["subject_id"] = args.subject

    path = out_dir / f"subject_{args.subject}.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {path}")
    print(f"  features: {out['features'].shape}  fingers: {out['fingers'].shape}")
    print(f"  n_train: {out['n_train']}  n_val: {out['features'].shape[-1] - out['n_train']}")


if __name__ == "__main__":
    main()