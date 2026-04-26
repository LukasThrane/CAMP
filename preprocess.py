"""
preprocess.py — FingerFlex-faithful preprocessing pipeline.

Reimplements the FingerFlex prepare_data.ipynb pipeline as closely as
possible. Key choices that match the published code (and differ from
naive interpretations of the paper):

1. Standardization THEN common average reference (CAR) — subtract per-time
   median across channels, NOT per-channel median across time.
2. MNE's filter_data and notch_filter for bandpass 40-300 Hz and powerline
   harmonics removal. Use 60 Hz for US data (Subject 1, UW), 50 Hz for EU.
3. MNE's tfr_array_morlet with n_cycles=7 (default) and 40 log-spaced
   frequencies between 40 and 300 Hz, output='power' (not magnitude).
4. Decimation by simple slicing [::10] (NOT scipy.signal.decimate with
   anti-aliasing), exactly what FingerFlex does.
5. Time delay = 200 ms (ECoG leads fingers by 200ms, per their code).
6. Fingers: MinMaxScaler fit on TRAIN, applied to both train and test.
7. NO RobustScaler on features — FingerFlex doesn't scale features, despite
   the paper text saying it does. They concluded "scaling helps" but their
   final code doesn't apply it on the wavelet output (the train script
   converts to db only optionally, default off). We follow their CODE.

The chip simulator runs upstream of this — it produces the (62, T) raw
ECoG stream that we feed in here.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Step 1: standardization + CAR
# ============================================================================

def normalize_with_car(ecog: np.ndarray) -> np.ndarray:
    """Per-channel z-score, then subtract per-time median across channels (CAR).

    This matches FingerFlex's `normalize` function:
        means = np.mean(ecog, axis=1, keepdims=True)        # per-channel
        stds  = np.std(ecog, axis=1, keepdims=True)         # per-channel
        x = (ecog - means) / stds
        common_average = np.median(x, axis=0, keepdims=True) # per-time, across channels
        x = x - common_average

    CAR (common average reference) is standard ECoG denoising — it removes
    the spatially-uniform noise component (e.g., distant muscle, line
    interference that escaped the notch filter).
    """
    ecog = ecog.astype(np.float64)
    means = ecog.mean(axis=1, keepdims=True)
    stds = ecog.std(axis=1, keepdims=True) + 1e-12
    x = (ecog - means) / stds
    car = np.median(x, axis=0, keepdims=True)
    return x - car


# ============================================================================
# Step 2: bandpass + notch (using MNE)
# ============================================================================

def filter_ecog_mne(
    ecog: np.ndarray,
    sfreq: float = 1000.0,
    line_freq: float = 60.0,
    l_freq: float = 40.0,
    h_freq: float = 300.0,
) -> np.ndarray:
    """Bandpass + powerline harmonic notch using MNE.

    Matches FingerFlex's `filter_ecog_data`:
        signal_filtered = mne.filter.filter_data(x, sfreq, l_freq=40, h_freq=300)
        x = mne.filter.notch_filter(signal_filtered, sfreq, freqs=harmonics)

    where `harmonics = [k*line_freq for k in range(1, sfreq/2 // line_freq)]`.
    For 1kHz sample rate and 60Hz line: harmonics = [60, 120, 180, 240, 300, 360, 420, 480].
    """
    import mne

    ecog = ecog.astype(np.float64)
    # Build harmonics inside the bandpass region (MNE will silently ignore
    # ones outside, but explicit is fine).
    nyq = sfreq / 2.0
    n_harmonics = int(nyq // line_freq)
    harmonics = np.array([(k + 1) * line_freq for k in range(n_harmonics)])

    filtered = mne.filter.filter_data(
        ecog, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    )
    notched = mne.filter.notch_filter(
        filtered, sfreq, freqs=harmonics, verbose=False
    )
    return notched


# ============================================================================
# Step 3: Morlet wavelets via MNE
# ============================================================================

def compute_spectrogramms_mne(
    ecog: np.ndarray,
    sfreq: float = 1000.0,
    n_wavelets: int = 40,
    f_min: float = 40.0,
    f_max: float = 300.0,
    n_cycles: float = 7.0,
    output: str = "power",
) -> np.ndarray:
    """Compute Morlet spectrograms using MNE's tfr_array_morlet.

    Matches FingerFlex's `compute_spectrogramms` exactly. Returns
    (n_channels, n_wavelets, n_times). FingerFlex uses output='power'
    (the default).

    n_cycles=7 is MNE's typical default for gamma-band analysis.
    """
    import mne

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_wavelets)
    # MNE expects (n_epochs, n_channels, n_times) → reshape with n_epochs=1
    x = ecog[np.newaxis, ...]   # (1, C, T)
    spec = mne.time_frequency.tfr_array_morlet(
        x, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        output=output, verbose=False,
    )[0]   # → (C, n_wavelets, T)
    return spec.astype(np.float32)


# ============================================================================
# Step 4: decimation (FingerFlex uses simple slicing, NOT antialiased)
# ============================================================================

def downsample_spectrogramms(
    spec: np.ndarray, sfreq_in: float = 1000.0, sfreq_out: float = 100.0
) -> np.ndarray:
    """Decimate by simple slice [::10]. Matches FingerFlex.

    Note: this does NOT apply an anti-aliasing filter. The Morlet wavelet
    bank already smooths the time axis, so aliasing is small. We follow
    FingerFlex's choice for fidelity.
    """
    factor = int(sfreq_in // sfreq_out)
    assert factor > 1
    return spec[:, :, ::factor]


# ============================================================================
# Step 5: finger upsampling
# ============================================================================

def interpolate_fingers(
    fingers: np.ndarray,
    sfreq_in: float = 1000.0,
    target_sfreq: float = 100.0,
    true_finger_sfreq: float = 25.0,
    interp_kind: str = "cubic",
) -> np.ndarray:
    """Cubic-interpolate fingers to target sfreq.

    Matches FingerFlex's `interpolate_fingerflex`. Approach:
      - Fingers were originally sampled at 25 Hz, then NaN-padded to 1 kHz.
      - We don't fully trust the NaN structure (braindecode interpolated for
        us already), so we'll re-extract every 40th sample (1000/25 = 40)
        as the "true" 25 Hz timeseries, append the last value to extend
        the interpolation domain, then cubic-interp to 100 Hz.
    """
    from scipy.interpolate import interp1d

    fingers = fingers.astype(np.float64)
    downscale_ratio = int(sfreq_in // true_finger_sfreq)   # 40
    upscale_ratio = int(target_sfreq // true_finger_sfreq) # 4

    # Take every 40th sample as the canonical 25 Hz signal.
    fingers_25hz = fingers[:, ::downscale_ratio]
    # Extend the right edge by repeating the last sample (so cubic interp
    # has support at the right boundary).
    fingers_25hz = np.concatenate(
        [fingers_25hz, fingers_25hz[:, -1:]], axis=1
    )

    # Build interpolant on 25 Hz time axis (in units of 100 Hz samples).
    ts_in = np.arange(fingers_25hz.shape[1]) * upscale_ratio
    funcs = [interp1d(ts_in, fingers_25hz[ch], kind=interp_kind)
             for ch in range(fingers_25hz.shape[0])]

    # Evaluate at every 100 Hz sample, dropping the trailing duplicate region.
    n_out = (fingers_25hz.shape[1] - 1) * upscale_ratio
    ts_out = np.arange(n_out)

    out = np.array([f(ts_out) for f in funcs], dtype=np.float64)
    return out


# ============================================================================
# Step 6: time delay
# ============================================================================

def crop_for_time_delay(
    fingers: np.ndarray,
    spec: np.ndarray,
    delay_s: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop first `delay_s` of fingers and last `delay_s` of spectrograms,
    so that spec[t] predicts fingers[t + delay].

    FingerFlex uses delay_s = 0.2 in their CODE, despite the paper text
    saying 20 ms. Code wins.
    """
    delay = int(round(delay_s * sfreq))
    if delay <= 0:
        return fingers, spec
    return fingers[..., delay:], spec[..., : spec.shape[-1] - delay]


# ============================================================================
# Step 7: finger MinMax scaling (fit on train, applied to both)
# ============================================================================

class MinMaxScalerFingers:
    """sklearn-style MinMaxScaler fitting on (5, T) finger arrays.

    fit() takes shape (5, T_train); transform() handles either (5, T) or
    (5, T) — basically transforms each finger by its train-fitted min/max.
    Output is float32.
    """
    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, fingers: np.ndarray) -> "MinMaxScalerFingers":
        self.min_ = fingers.min(axis=-1, keepdims=True)
        self.max_ = fingers.max(axis=-1, keepdims=True)
        return self

    def transform(self, fingers: np.ndarray) -> np.ndarray:
        return ((fingers - self.min_) /
                (self.max_ - self.min_ + 1e-12)).astype(np.float32)


# ============================================================================
# End-to-end: process train+test together
# ============================================================================

def preprocess_train_test(
    ecog_train: np.ndarray,
    fingers_train: np.ndarray,
    ecog_test: np.ndarray,
    fingers_test: np.ndarray,
    sfreq: float = 1000.0,
    line_freq: float = 60.0,
    n_wavelets: int = 40,
    target_sfreq: float = 100.0,
    delay_s: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Run the full FingerFlex pipeline on already-split train and test.

    Returns:
      {
        'features_train': (62, 40, T_train) float32,
        'fingers_train':  (5, T_train) float32, MinMax scaled,
        'features_test':  (62, 40, T_test) float32,
        'fingers_test':   (5, T_test) float32, MinMax scaled,
        'sfreq': 100.0,
        'finger_scaler': MinMaxScalerFingers (fit on train),
      }

    The two streams are processed separately (filtering, CAR, wavelets,
    decimation) so there's no leakage between train and test through the
    filter state. Finger scaler is fit on train only.
    """
    def _process_ecog(x, label):
        if verbose:
            print(f"  [{label}] standardize + CAR…")
        x = normalize_with_car(x)
        if verbose:
            print(f"  [{label}] bandpass + notch (line={line_freq}Hz)…")
        x = filter_ecog_mne(x, sfreq=sfreq, line_freq=line_freq)
        if verbose:
            print(f"  [{label}] Morlet ({n_wavelets} wavelets)…")
        spec = compute_spectrogramms_mne(
            x, sfreq=sfreq, n_wavelets=n_wavelets,
        )
        if verbose:
            print(f"  [{label}] decimate {sfreq:.0f} → {target_sfreq:.0f} Hz…")
        spec = downsample_spectrogramms(spec, sfreq, target_sfreq)
        return spec

    # Process ECoG.
    spec_train = _process_ecog(ecog_train, "train")
    spec_test = _process_ecog(ecog_test, "test")

    # Process fingers — interpolate to target_sfreq.
    if verbose:
        print("  [fingers] cubic interpolation 1kHz → 100Hz…")
    fings_train_interp = interpolate_fingers(
        fingers_train, sfreq_in=sfreq, target_sfreq=target_sfreq,
    )
    fings_test_interp = interpolate_fingers(
        fingers_test, sfreq_in=sfreq, target_sfreq=target_sfreq,
    )

    # Align lengths.
    T_train = min(spec_train.shape[-1], fings_train_interp.shape[-1])
    T_test  = min(spec_test.shape[-1],  fings_test_interp.shape[-1])
    spec_train = spec_train[..., :T_train]
    spec_test  = spec_test[..., :T_test]
    fings_train_interp = fings_train_interp[..., :T_train]
    fings_test_interp  = fings_test_interp[..., :T_test]

    # Apply 200ms time delay.
    if verbose:
        print(f"  [delay] applying {delay_s*1000:.0f} ms ECoG-fingers shift…")
    fings_train_interp, spec_train = crop_for_time_delay(
        fings_train_interp, spec_train, delay_s, target_sfreq,
    )
    fings_test_interp, spec_test = crop_for_time_delay(
        fings_test_interp, spec_test, delay_s, target_sfreq,
    )

    # MinMax scale fingers (fit on train).
    if verbose:
        print("  [fingers] MinMax scale (fit on train)…")
    scaler = MinMaxScalerFingers().fit(fings_train_interp)
    fings_train_scaled = scaler.transform(fings_train_interp)
    fings_test_scaled  = scaler.transform(fings_test_interp)

    return {
        "features_train": spec_train,
        "fingers_train":  fings_train_scaled,
        "features_test":  spec_test,
        "fingers_test":   fings_test_scaled,
        "sfreq": target_sfreq,
        "finger_scaler": scaler,
    }


if __name__ == "__main__":
    # Smoke test the pipeline shapes on synthetic data.
    sfreq = 1000.0
    rng = np.random.default_rng(0)
    e_tr = rng.standard_normal((62, 10000))
    f_tr = rng.standard_normal((5, 10000))
    e_te = rng.standard_normal((62, 5000))
    f_te = rng.standard_normal((5, 5000))
    out = preprocess_train_test(
        e_tr, f_tr, e_te, f_te, sfreq=sfreq, verbose=True,
    )
    print()
    print(f"features_train: {out['features_train'].shape}")
    print(f"fingers_train:  {out['fingers_train'].shape}")
    print(f"features_test:  {out['features_test'].shape}")
    print(f"fingers_test:   {out['fingers_test'].shape}")
    assert out['features_train'].shape[-1] == out['fingers_train'].shape[-1]
    assert out['features_test'].shape[-1] == out['fingers_test'].shape[-1]
    print("OK")