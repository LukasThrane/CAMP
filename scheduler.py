"""
scheduler.py — on-chip simulation for CAMP.

Simulates what runs on the implant ASIC: bandpass filter, lightweight band-power
predictor, DRR scheduler, sample-and-hold. Everything in this file is meant to
be gate-budget plausible (~few thousand gates, sub-µW per channel). The wavelet
decomposition and decoder live OFF chip, in preprocess.py and decoder.py.

On-chip cost summary (target):
  - Bandpass FIR: ~33 MACs/sample/channel  (33 taps × 1 multiplier reused)
  - Squared MA:   1 multiply + 1 add per sample per channel
  - Baseline EMA: 1 multiply + 1 add per sample per channel
  - DRR step:     1 add + 1 sub per channel per window (every 30 ms)
  - Top-K:        partial sort of N=62 entries every 30 ms
  Total: O(few thousand gates), well under the 4 mW/cm² thermal ceiling.

This file has THREE entry points depending on use:
  1. ChipSimulator.process(ecog_array) — batch mode for offline experiments
  2. ChipSimulator.step(chunk, ...)    — streaming mode for live demo
  3. run_scheduler(in_q, out_q, ...)   — multiprocessing wrapper around streams.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import Queue
from queue import Empty
from typing import Optional

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi


# --------------------------------------------------------------------------- #
# Band specifications
# --------------------------------------------------------------------------- #

@dataclass
class BandSpec:
    """One frequency band for the on-chip FIR. Per-region predictor assignment
    is the caller's responsibility — this class describes ONE band."""
    low_hz: float
    high_hz: float
    name: str = ""


# Region-specific bands per the proposal §II.C. Only `motor` is used for
# Dataset 4 (finger flexion ≡ motor cortex). The others are defined here so
# heterogeneous-predictor experiments on visual/speech datasets later don't
# need to touch this file.
BAND_MOTOR  = BandSpec(13.0, 30.0,  "beta")        # beta desync
BAND_VISUAL = BandSpec(70.0, 150.0, "high_gamma")  # high-gamma
BAND_SPEECH = BandSpec(70.0, 150.0, "high_gamma")
# Prefrontal (spectral entropy) would be a separate predictor class; out of
# scope for the motor experiment.


# --------------------------------------------------------------------------- #
# On-chip predictor
# --------------------------------------------------------------------------- #

class OnChipPredictor:
    """
    Bandpass FIR + squared MA + baseline-relative weighting.

    Pipeline per channel:
        x → FIR(beta band) → square → fast EMA P_fast (50 ms)
                                   → slow EMA P_slow (5 s)
        weight = |P_fast - P_slow| / (P_slow + ε)

    Why baseline-relative? Motor cortex shows BETA DESYNCHRONIZATION during
    movement: beta power DROPS relative to rest. A predictor that weights
    channels by raw beta power would do the wrong thing — it would prioritize
    quiet-motor channels. The deviation-from-baseline form captures both
    desync (drop) and ERS (rise), which is what we want.

    Hardware cost per channel:
        - 33-tap FIR:      33 MACs/sample (1 multiplier reused across taps)
        - 1 square:        1 multiply
        - 2 EMAs:          2 multiplies + 2 adds per sample
        - Weight compute:  1 sub + 1 abs + 1 divide every update tick
                           (divide can be approximated by shift if baseline
                            is power-of-2 quantized — typical ASIC trick)
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        band: BandSpec = BAND_MOTOR,
        n_taps: int = 33,
        ma_window_s: float = 0.050,     # fast EMA — current activity
        baseline_window_s: float = 5.0, # slow EMA — resting baseline
        update_period_s: float = 0.010,
        eps: float = 1e-8,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.band = band
        self.update_period_s = update_period_s
        self.eps = eps

        # FIR bandpass — same taps across channels; on real silicon this is
        # one shared coefficient ROM and N parallel MAC trees.
        nyq = sfreq / 2.0
        self.taps = firwin(
            n_taps,
            [band.low_hz / nyq, band.high_hz / nyq],
            pass_zero=False,
        )
        zi = lfilter_zi(self.taps, 1.0)
        self._filter_state = np.tile(zi, (n_channels, 1))

        # Two EMAs.
        self._alpha_fast = 1.0 - np.exp(-1.0 / (ma_window_s * sfreq))
        self._alpha_slow = 1.0 - np.exp(-1.0 / (baseline_window_s * sfreq))
        self._p_fast = np.zeros(n_channels, dtype=np.float64)
        self._p_slow = np.zeros(n_channels, dtype=np.float64)

        # Weights are held between update ticks.
        self._weights = np.full(n_channels, 1.0 / n_channels, dtype=np.float64)
        self._t_last_update = -np.inf

    def process_chunk(
        self, ecog_chunk: np.ndarray, t_start: float
    ) -> np.ndarray:
        """Consume a (C, T) chunk, advance state, return current weight vector.
        Weights only update on the configured cadence; otherwise held."""
        # Bandpass FIR per channel.
        filtered = np.empty_like(ecog_chunk, dtype=np.float64)
        for c in range(self.n_channels):
            filtered[c], self._filter_state[c] = lfilter(
                self.taps, 1.0, ecog_chunk[c], zi=self._filter_state[c]
            )

        # Squared, then both EMAs sample-by-sample.
        squared = filtered * filtered
        a_f, a_s = self._alpha_fast, self._alpha_slow
        p_f, p_s = self._p_fast, self._p_slow
        for t in range(squared.shape[1]):
            x = squared[:, t]
            p_f = (1.0 - a_f) * p_f + a_f * x
            p_s = (1.0 - a_s) * p_s + a_s * x
        self._p_fast = p_f
        self._p_slow = p_s

        # Update weights on cadence tick.
        chunk_dur = ecog_chunk.shape[1] / self.sfreq
        if t_start + chunk_dur - self._t_last_update >= self.update_period_s:
            # Baseline-relative: |fast - slow| / (slow + eps). Captures both
            # desync (fast < slow) and ERS (fast > slow).
            deviation = np.abs(self._p_fast - self._p_slow) / (
                self._p_slow + self.eps
            )
            total = deviation.sum()
            if total > 0:
                self._weights = deviation / total
            else:
                self._weights = np.full(
                    self.n_channels, 1.0 / self.n_channels, dtype=np.float64
                )
            self._t_last_update = t_start + chunk_dur

        return self._weights.copy()

    def reset(self) -> None:
        zi = lfilter_zi(self.taps, 1.0)
        self._filter_state = np.tile(zi, (self.n_channels, 1))
        self._p_fast[:] = 0.0
        self._p_slow[:] = 0.0
        self._weights[:] = 1.0 / self.n_channels
        self._t_last_update = -np.inf


# --------------------------------------------------------------------------- #
# Schedulers — three strategies share an interface
# --------------------------------------------------------------------------- #

class _BaseScheduler:
    """Common interface. Subclasses implement select()."""

    def __init__(self, n_channels: int, bandwidth_frac: float):
        assert 0.0 < bandwidth_frac <= 1.0
        self.n_channels = n_channels
        self.bandwidth_frac = bandwidth_frac
        self.K = max(1, int(np.floor(bandwidth_frac * n_channels)))

    def select(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class FullResolutionScheduler(_BaseScheduler):
    """Upper bound: transmit every channel every window."""

    def select(self, weights: np.ndarray) -> np.ndarray:
        return np.ones(self.n_channels, dtype=bool)


class RoundRobinScheduler(_BaseScheduler):
    """Lower bound: cycle through channels in order, K per window. Ignores
    weights entirely — the dumb baseline that CAMP must beat."""

    def __init__(self, n_channels: int, bandwidth_frac: float):
        super().__init__(n_channels, bandwidth_frac)
        self._cursor = 0

    def select(self, weights: np.ndarray) -> np.ndarray:
        sel = np.zeros(self.n_channels, dtype=bool)
        for i in range(self.K):
            sel[(self._cursor + i) % self.n_channels] = True
        self._cursor = (self._cursor + self.K) % self.n_channels
        return sel

    def reset(self) -> None:
        self._cursor = 0


class CAMPScheduler(_BaseScheduler):
    """
    Deficit Round Robin with top-K-by-weight selection and floor guarantee.

    Per window:
      1. deficit[c] += weights[c] * K
      2. Forced set F = channels above floor threshold (starvation prevention)
      3. Remaining K - |F| slots → top by weight, excluding F
      4. Selected channels pay 1.0 from their deficit

    Hardware cost: N registers (deficit counters) + N adders + 1 partial sort
    over N=62 entries every 30 ms. Negligible vs the 4 mW/cm² ceiling.
    """

    def __init__(
        self,
        n_channels: int,
        bandwidth_frac: float,
        floor_frac: float = 0.1,
    ):
        super().__init__(n_channels, bandwidth_frac)
        assert 0.0 <= floor_frac <= 1.0
        self.floor_frac = floor_frac
        self._deficit = np.zeros(n_channels, dtype=np.float64)
        self._floor_quota = floor_frac * (self.K / n_channels)

    def select(self, weights: np.ndarray) -> np.ndarray:
        # 1. Accumulate.
        self._deficit += weights * self.K

        # 2. Forced set.
        forced = self._deficit >= (1.0 + self._floor_quota)
        n_forced = int(forced.sum())

        sel = np.zeros(self.n_channels, dtype=bool)
        if n_forced >= self.K:
            # Too many starving — pick K most starving. Right failure mode.
            top_forced = np.argpartition(-self._deficit, self.K - 1)[: self.K]
            sel[top_forced] = True
        else:
            sel[forced] = True
            remaining = self.K - n_forced
            if remaining > 0:
                cand = np.where(sel, -np.inf, weights)
                top_w = np.argpartition(-cand, remaining - 1)[:remaining]
                sel[top_w] = True

        # 4. Pay.
        self._deficit[sel] -= 1.0
        return sel

    def reset(self) -> None:
        self._deficit[:] = 0.0


def make_scheduler(strategy: str, n_channels: int, bandwidth_frac: float,
                   floor_frac: float = 0.1) -> _BaseScheduler:
    """Factory: 'full' | 'round_robin' | 'camp'."""
    if strategy == "full":
        return FullResolutionScheduler(n_channels, bandwidth_frac)
    if strategy == "round_robin":
        return RoundRobinScheduler(n_channels, bandwidth_frac)
    if strategy == "camp":
        return CAMPScheduler(n_channels, bandwidth_frac, floor_frac)
    raise ValueError(f"unknown strategy: {strategy!r}")


# --------------------------------------------------------------------------- #
# Sample-and-hold
# --------------------------------------------------------------------------- #

class SampleAndHold:
    """Reconstructs full (C, T) stream from selected-channel transmissions.

    On-chip this is FREE: the chip simply doesn't transmit non-selected
    channels. The receiver (server) holds the last-known value of each
    channel and broadcasts it forward until a new sample arrives.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self._hold = np.zeros(n_channels, dtype=np.float64)
        self._initialized = False

    def step(self, chunk: np.ndarray, selected: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self._hold = chunk[:, 0].astype(np.float64).copy()
            self._initialized = True

        T = chunk.shape[1]
        out = np.empty_like(chunk, dtype=np.float64)
        out[selected] = chunk[selected]
        self._hold[selected] = chunk[selected, -1]
        out[~selected] = self._hold[~selected, None].repeat(T, axis=1)
        return out

    def reset(self) -> None:
        self._hold[:] = 0.0
        self._initialized = False


# --------------------------------------------------------------------------- #
# Chip simulator — bundles everything for offline use
# --------------------------------------------------------------------------- #

class ChipSimulator:
    """End-to-end chip simulation. Use process() for offline batch (the
    experiment runner uses this). Use step() for streaming (the multiprocessing
    wrapper uses this)."""

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        strategy: str,
        bandwidth_frac: float,
        band: BandSpec = BAND_MOTOR,
        floor_frac: float = 0.1,
        chunk_size: int = 30,  # 30 ms at 1 kHz
        update_period_s: float = 0.010,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        self.predictor = OnChipPredictor(
            n_channels=n_channels,
            sfreq=sfreq,
            band=band,
            update_period_s=update_period_s,
        )
        self.scheduler = make_scheduler(
            strategy, n_channels, bandwidth_frac, floor_frac
        )
        self.sah = SampleAndHold(n_channels)

    def step(self, chunk: np.ndarray, t_start: float):
        """One chunk. Returns (transmitted_with_hold, weights, selected)."""
        weights = self.predictor.process_chunk(chunk, t_start)
        selected = self.scheduler.select(weights)
        transmitted = self.sah.step(chunk, selected)
        return transmitted, weights, selected

    def process(self, ecog: np.ndarray) -> np.ndarray:
        """Batch: simulate the chip on a full recording, return the
        reconstructed (C, T) stream the decoder receives."""
        C, T = ecog.shape
        assert C == self.n_channels
        n_chunks = T // self.chunk_size
        T_out = n_chunks * self.chunk_size
        out = np.empty((C, T_out), dtype=np.float64)
        for i in range(n_chunks):
            s, e = i * self.chunk_size, (i + 1) * self.chunk_size
            out[:, s:e], _, _ = self.step(ecog[:, s:e], s / self.sfreq)
        return out

    def reset(self) -> None:
        self.predictor.reset()
        self.scheduler.reset()
        self.sah.reset()


# --------------------------------------------------------------------------- #
# Multiprocessing wrapper — for the live-stream demo
# --------------------------------------------------------------------------- #

@dataclass
class TransmissionChannel:
    """Models uplink latency by stamping arrival time. No wall-clock sleep
    so simulations stay deterministic."""
    uplink_latency_s: float = 0.005

    def send(self, payload: dict, t_send: float) -> dict:
        payload = dict(payload)
        payload["t_arrive"] = t_send + self.uplink_latency_s
        return payload


def run_scheduler(
    in_queue: Queue,
    out_queue: Queue,
    n_channels: int,
    sfreq: float,
    strategy: str,
    bandwidth_frac: float,
    band: BandSpec = BAND_MOTOR,
    floor_frac: float = 0.1,
    uplink_latency_s: float = 0.005,
    update_period_s: float = 0.010,
    stop_after_s: Optional[float] = None,
) -> None:
    """Streaming consumer for streams.py-produced queue."""
    chip = ChipSimulator(
        n_channels=n_channels,
        sfreq=sfreq,
        strategy=strategy,
        bandwidth_frac=bandwidth_frac,
        band=band,
        floor_frac=floor_frac,
        update_period_s=update_period_s,
    )
    channel = TransmissionChannel(uplink_latency_s=uplink_latency_s)

    t0 = time.monotonic()
    while True:
        if stop_after_s is not None and time.monotonic() - t0 > stop_after_s:
            break
        try:
            item = in_queue.get(timeout=1.0)
        except Empty:
            continue
        if item is None:
            out_queue.put(None)
            break

        ecog = item["ecog"]
        t_start = item["t_start"]
        transmitted, weights, selected = chip.step(ecog, t_start)

        payload = {
            "ecog": transmitted,
            "fingers": item["fingers"],
            "weights": weights,
            "selected": selected,
            "t_start": t_start,
            "sfreq": item["sfreq"],
        }
        out_queue.put(channel.send(payload, t_send=t_start))


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    """5 channels carry a 20 Hz beta carrier whose amplitude is MODULATED
    (rises and falls); the rest are noise. Baseline-relative predictor
    should catch the channels showing deviation."""
    rng = np.random.default_rng(0)
    C, T_chunk = 62, 30
    sfreq = 1000.0
    n_chunks = 400  # 12 s

    total_T = n_chunks * T_chunk
    t = np.arange(total_T) / sfreq
    envelope = 1.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)  # 0.3 Hz modulation
    signal = rng.standard_normal((C, total_T)) * 1.0
    for c in range(5):
        signal[c] += envelope * np.sin(2 * np.pi * 20.0 * t)

    chip = ChipSimulator(
        n_channels=C, sfreq=sfreq, strategy="camp",
        bandwidth_frac=0.2, floor_frac=0.1, chunk_size=T_chunk,
    )

    selections = np.zeros(C, dtype=np.int64)
    for i in range(n_chunks):
        chunk = signal[:, i * T_chunk : (i + 1) * T_chunk]
        _, _, sel = chip.step(chunk, t_start=i * T_chunk / sfreq)
        selections += sel.astype(np.int64)

    print(f"top weights:    {np.argsort(-chip.predictor._weights)[:8].tolist()}")
    print(f"beta channels (0-4) selections: {selections[:5].tolist()}")
    print(f"noise sample (10-14):           {selections[10:15].tolist()}")
    print(f"min/max selections: {selections.min()} / {selections.max()}")
    print(f"total: {selections.sum()} (expected ~{n_chunks * chip.scheduler.K})")
    assert selections[:5].mean() > selections[10:].mean(), \
        "modulated beta channels should dominate"
    assert selections.min() > 0, "floor should prevent zero-selection"
    print("OK")


if __name__ == "__main__":
    _smoke_test()