"""
data_loader.py — official BCI Competition IV Dataset 4 split.

Returns the SEPARATE train and test recordings, not a 65/35 partition of a
concatenated stream. This matches what FingerFlex does and what the BCI
Competition IV evaluation protocol expects.

Subject 1: train = 400s, test = 200s (both at 1 kHz).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def fill_nan_by_interpolation(x: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN gaps in finger labels, per channel."""
    out = x.copy()
    idx = np.arange(x.shape[1])
    for ch in range(x.shape[0]):
        valid = ~np.isnan(x[ch])
        if valid.sum() == 0:
            out[ch] = 0.0
        else:
            out[ch] = np.interp(idx, idx[valid], x[ch, valid])
    return out


def load_subject_split(subject_id: int = 1, cache_dir: Path = Path("./cache")) -> dict:
    """
    Load a subject's train + test ECoG and fingers as separate arrays.

    braindecode exposes the dataset as a BaseConcatDataset of two RawDatasets
    — train and test, in that order. We pull both out separately.

    Returns:
        {
            'ecog_train':  (62, 400000)  float64,
            'fingers_train': (5, 400000) float64 (NaN-interpolated),
            'ecog_test':   (62, 200000)  float64,
            'fingers_test':  (5, 200000) float64 (NaN-interpolated),
            'sfreq':       1000.0,
        }
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"raw_split_subject_{subject_id}.pkl"
    if cache.exists():
        print(f"[cache] loading {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"[load] subject {subject_id} via braindecode (official split)…")
    from braindecode.datasets import BCICompetitionIVDataset4
    ds = BCICompetitionIVDataset4(subject_ids=[subject_id])

    # braindecode loads the dataset as two separate RawDatasets per subject:
    #   ds.datasets[0] = train  (description['session'] == 'train')
    #   ds.datasets[1] = test   (description['session'] == 'test')
    train_raw = None
    test_raw = None
    for d in ds.datasets:
        sess = d.description.get("session", None)
        if sess == "train":
            train_raw = d.raw
        elif sess == "test":
            test_raw = d.raw
    assert train_raw is not None and test_raw is not None, \
        "Expected both train and test RawDatasets in BCICompetitionIVDataset4"

    sfreq = float(train_raw.info["sfreq"])
    assert sfreq == 1000.0

    train_arr = train_raw.get_data()
    test_arr = test_raw.get_data()
    # First 62 channels = ECoG, last 5 = fingers (NaN-padded at non-25Hz samples).
    out = {
        "ecog_train":   train_arr[:62],
        "fingers_train": fill_nan_by_interpolation(train_arr[62:67]),
        "ecog_test":    test_arr[:62],
        "fingers_test":  fill_nan_by_interpolation(test_arr[62:67]),
        "sfreq": sfreq,
    }

    print(f"  train ECoG: {out['ecog_train'].shape}, fingers: {out['fingers_train'].shape}")
    print(f"  test  ECoG: {out['ecog_test'].shape},  fingers: {out['fingers_test'].shape}")

    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"  cached → {cache}")
    return out


if __name__ == "__main__":
    d = load_subject_split(1)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.shape}, range [{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"{k}: {v}")