"""
scheduler.py — on-chip simulation for CAMP.

Simulates what runs on the implant ASIC: bandpass FIR, lightweight band-power
predictor (baseline-relative beta desync detector), DRR scheduler, and
sample-and-hold reconstruction at the receiver.

Everything in this file is meant to be gate-budget plausible (~few thousand
gates, sub-µW per channel). The wavelet decomposition and decoder live OFF
chip — see preprocess.py and decoder.py.

On-chip cost summary (target):
  - Bandpass FIR: ~33 MACs/sample/channel
  - Squared MA:   1 mul + 1 add per sample/channel
  - Baseline EMA: 1 mul + 1 add per sample/channel
  - DRR step:     1 add + 1 sub per channel per window (every 30 ms)
  - Top-K:        partial sort of N=62 entries every 30 ms
  Total: O(few thousand gates), well under the 4 mW/cm² thermal ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi


# ============================================================================
# Band specifications
# ============================================================================

@dataclass
class BandSpec:
    low_hz: float
    high_hz: float
    name: str = ""


BAND_MOTOR  = BandSpec(13.0, 30.0,  "beta")          # finger flexion
BAND_VISUAL = BandSpec(70.0, 150.0, "high_gamma")
BAND_SPEECH = BandSpec(70.0, 150.0, "high_gamma")


# ============================================================================
# On-chip predictor — baseline-relative beta desync detector
# ============================================================================

class OnChipPredictor:
    """Bandpass FIR + squared MA + baseline-relative weighting.

    Pipeline per channel:
        x → FIR(beta) → square → fast EMA P_fast (50 ms)
                              → slow EMA P_slow (5 s)
        weight = |P_fast - P_slow| / (P_slow + ε)

    Why baseline-relative? Motor cortex shows beta DESYNCHRONIZATION during
    movement: power DROPS relative to rest. Weighting by raw beta would
    prioritize quiet channels — backwards. Deviation from baseline captures
    both ERD (drop) and ERS (rise).
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        band: BandSpec = BAND_MOTOR,
        n_taps: int = 33,
        ma_window_s: float = 0.050,
        baseline_window_s: float = 5.0,
        update_period_s: float = 0.010,
        eps: float = 1e-8,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.update_period_s = update_period_s
        self.eps = eps

        nyq = sfreq / 2.0
        self.taps = firwin(
            n_taps, [band.low_hz / nyq, band.high_hz / nyq], pass_zero=False,
        )
        zi = lfilter_zi(self.taps, 1.0)
        self._filter_state = np.tile(zi, (n_channels, 1))

        self._alpha_fast = 1.0 - np.exp(-1.0 / (ma_window_s * sfreq))
        self._alpha_slow = 1.0 - np.exp(-1.0 / (baseline_window_s * sfreq))
        self._p_fast = np.zeros(n_channels)
        self._p_slow = np.zeros(n_channels)

        self._weights = np.full(n_channels, 1.0 / n_channels)
        self._t_last_update = -np.inf

    def process_chunk(self, ecog_chunk: np.ndarray, t_start: float) -> np.ndarray:
        # Bandpass FIR per channel.
        filtered = np.empty_like(ecog_chunk, dtype=np.float64)
        for c in range(self.n_channels):
            filtered[c], self._filter_state[c] = lfilter(
                self.taps, 1.0, ecog_chunk[c], zi=self._filter_state[c],
            )

        # Squared, then EMAs.
        squared = filtered * filtered
        a_f, a_s = self._alpha_fast, self._alpha_slow
        p_f, p_s = self._p_fast, self._p_slow
        for t in range(squared.shape[1]):
            x = squared[:, t]
            p_f = (1.0 - a_f) * p_f + a_f * x
            p_s = (1.0 - a_s) * p_s + a_s * x
        self._p_fast = p_f
        self._p_slow = p_s

        # Update weights on cadence.
        chunk_dur = ecog_chunk.shape[1] / self.sfreq
        if t_start + chunk_dur - self._t_last_update >= self.update_period_s:
            dev = np.abs(self._p_fast - self._p_slow) / (self._p_slow + self.eps)
            total = dev.sum()
            if total > 0:
                self._weights = dev / total
            else:
                self._weights = np.full(self.n_channels, 1.0 / self.n_channels)
            self._t_last_update = t_start + chunk_dur
        return self._weights.copy()

    def reset(self) -> None:
        zi = lfilter_zi(self.taps, 1.0)
        self._filter_state = np.tile(zi, (self.n_channels, 1))
        self._p_fast[:] = 0.0
        self._p_slow[:] = 0.0
        self._weights[:] = 1.0 / self.n_channels
        self._t_last_update = -np.inf


# ============================================================================
# Schedulers
# ============================================================================

class _BaseScheduler:
    def __init__(self, n_channels: int, bandwidth_frac: float):
        assert 0.0 < bandwidth_frac <= 1.0
        self.n_channels = n_channels
        self.bandwidth_frac = bandwidth_frac
        self.K = max(1, int(np.floor(bandwidth_frac * n_channels)))

    def select(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None: pass


class FullResolutionScheduler(_BaseScheduler):
    def select(self, weights):
        return np.ones(self.n_channels, dtype=bool)


class RoundRobinScheduler(_BaseScheduler):
    def __init__(self, n_channels, bandwidth_frac):
        super().__init__(n_channels, bandwidth_frac)
        self._cursor = 0

    def select(self, weights):
        sel = np.zeros(self.n_channels, dtype=bool)
        for i in range(self.K):
            sel[(self._cursor + i) % self.n_channels] = True
        self._cursor = (self._cursor + self.K) % self.n_channels
        return sel

    def reset(self):
        self._cursor = 0


class CAMPScheduler(_BaseScheduler):
    """DRR with top-K-by-weight + floor guarantee."""

    def __init__(self, n_channels, bandwidth_frac, floor_frac=0.1):
        super().__init__(n_channels, bandwidth_frac)
        assert 0.0 <= floor_frac <= 1.0
        self.floor_frac = floor_frac
        self._deficit = np.zeros(n_channels)
        self._floor_quota = floor_frac * (self.K / n_channels)

    def select(self, weights):
        self._deficit += weights * self.K

        forced = self._deficit >= (1.0 + self._floor_quota)
        n_forced = int(forced.sum())

        sel = np.zeros(self.n_channels, dtype=bool)
        if n_forced >= self.K:
            top_forced = np.argpartition(-self._deficit, self.K - 1)[: self.K]
            sel[top_forced] = True
        else:
            sel[forced] = True
            remaining = self.K - n_forced
            if remaining > 0:
                cand = np.where(sel, -np.inf, weights)
                top_w = np.argpartition(-cand, remaining - 1)[:remaining]
                sel[top_w] = True

        self._deficit[sel] -= 1.0
        return sel

    def reset(self):
        self._deficit[:] = 0.0


class StaticSaliencyScheduler(_BaseScheduler):
    """DRR with FIXED weights from offline saliency analysis.

    Same DRR mechanics as CAMPScheduler, but the weight vector is
    constant — set once at instantiation from a saliency NPZ — and never
    updates. This is the Neuralite-equivalent baseline: channel selection
    determined offline by analyzing the trained decoder.

    The floor mechanism still applies, so even very-low-saliency channels
    transmit occasionally. This prevents brittleness if the saliency
    estimate misses a channel that turns out to matter.
    """

    def __init__(self, n_channels, bandwidth_frac, saliency_weights,
                 floor_frac=0.1):
        super().__init__(n_channels, bandwidth_frac)
        assert 0.0 <= floor_frac <= 1.0
        assert saliency_weights.shape == (n_channels,), \
            f"expected ({n_channels},), got {saliency_weights.shape}"
        # Normalize defensively in case saliency vector doesn't quite sum to 1.
        s = saliency_weights.sum()
        self.weights = (saliency_weights / s if s > 0
                        else np.full(n_channels, 1.0 / n_channels))
        self.floor_frac = floor_frac
        self._deficit = np.zeros(n_channels)
        self._floor_quota = floor_frac * (self.K / n_channels)

    def select(self, _ignored_weights):
        # Use our static saliency weights regardless of what's passed in.
        # OnChipPredictor still runs upstream (we don't disable it) but its
        # output is ignored here.
        self._deficit += self.weights * self.K

        forced = self._deficit >= (1.0 + self._floor_quota)
        n_forced = int(forced.sum())

        sel = np.zeros(self.n_channels, dtype=bool)
        if n_forced >= self.K:
            top_forced = np.argpartition(-self._deficit, self.K - 1)[: self.K]
            sel[top_forced] = True
        else:
            sel[forced] = True
            remaining = self.K - n_forced
            if remaining > 0:
                cand = np.where(sel, -np.inf, self.weights)
                top_w = np.argpartition(-cand, remaining - 1)[:remaining]
                sel[top_w] = True

        self._deficit[sel] -= 1.0
        return sel

    def reset(self):
        self._deficit[:] = 0.0


class HybridScheduler(_BaseScheduler):
    """DRR with weights = saliency × activity, normalized.

    The on-chip predictor still produces dynamic activity weights based on
    beta desync. We multiply those by the offline saliency prior — favoring
    channels that are BOTH decoder-relevant AND currently active. A channel
    that's active but not decoder-relevant gets suppressed; a channel that's
    decoder-relevant but quiescent gets some bandwidth but less than during
    a flexion burst.

    This is the proposal's hybrid, with the offline saliency prior playing
    the role of the proposal's runtime server-side correction.
    """

    def __init__(self, n_channels, bandwidth_frac, saliency_weights,
                 floor_frac=0.1):
        super().__init__(n_channels, bandwidth_frac)
        assert 0.0 <= floor_frac <= 1.0
        assert saliency_weights.shape == (n_channels,)
        s = saliency_weights.sum()
        self.saliency = (saliency_weights / s if s > 0
                         else np.full(n_channels, 1.0 / n_channels))
        self.floor_frac = floor_frac
        self._deficit = np.zeros(n_channels)
        self._floor_quota = floor_frac * (self.K / n_channels)

    def select(self, activity_weights):
        # Element-wise multiply, then renormalize.
        combined = self.saliency * activity_weights
        total = combined.sum()
        if total > 0:
            combined = combined / total
        else:
            # Fall back to saliency only if activity is all zero.
            combined = self.saliency.copy()

        self._deficit += combined * self.K

        forced = self._deficit >= (1.0 + self._floor_quota)
        n_forced = int(forced.sum())

        sel = np.zeros(self.n_channels, dtype=bool)
        if n_forced >= self.K:
            top_forced = np.argpartition(-self._deficit, self.K - 1)[: self.K]
            sel[top_forced] = True
        else:
            sel[forced] = True
            remaining = self.K - n_forced
            if remaining > 0:
                cand = np.where(sel, -np.inf, combined)
                top_w = np.argpartition(-cand, remaining - 1)[:remaining]
                sel[top_w] = True

        self._deficit[sel] -= 1.0
        return sel

    def reset(self):
        self._deficit[:] = 0.0


def make_scheduler(strategy: str, n_channels: int, bandwidth_frac: float,
                   floor_frac: float = 0.1,
                   saliency_weights: np.ndarray | None = None) -> _BaseScheduler:
    """Strategy → scheduler. The strategy name encodes both the scheduler
    class and (where applicable) the predictor band, but the band itself is
    consumed by the ChipSimulator's OnChipPredictor, not by the scheduler.

    Strategies:
      'full'              — transmit every channel every chunk
      'round_robin'       — cyclic, ignores predictor weights
      'camp_beta'         — DRR top-K with beta-desync activity predictor
      'camp_gamma'        — DRR top-K with high-gamma activity predictor
      'camp_saliency'     — DRR top-K with FIXED offline saliency weights
                            (predictor still runs but its output is ignored)
      'camp_hybrid_beta'  — DRR top-K with saliency × beta-activity
      'camp_hybrid_gamma' — DRR top-K with saliency × gamma-activity
    """
    if strategy == "full":
        return FullResolutionScheduler(n_channels, bandwidth_frac)
    if strategy == "round_robin":
        return RoundRobinScheduler(n_channels, bandwidth_frac)
    if strategy in ("camp_beta", "camp_gamma"):
        # Same scheduler class — the difference is which band the
        # ChipSimulator's predictor is configured with.
        return CAMPScheduler(n_channels, bandwidth_frac, floor_frac)
    if strategy == "camp_saliency":
        if saliency_weights is None:
            raise ValueError(f"{strategy} requires saliency_weights")
        return StaticSaliencyScheduler(
            n_channels, bandwidth_frac, saliency_weights, floor_frac,
        )
    if strategy in ("camp_hybrid_beta", "camp_hybrid_gamma"):
        if saliency_weights is None:
            raise ValueError(f"{strategy} requires saliency_weights")
        return HybridScheduler(
            n_channels, bandwidth_frac, saliency_weights, floor_frac,
        )
    raise ValueError(f"unknown strategy: {strategy!r}")


def band_for_strategy(strategy: str) -> "BandSpec":
    """Return the band the on-chip predictor should run for this strategy.

    Even strategies that don't *use* the predictor's output (full,
    round_robin, camp_saliency) still get one — it's harmless and keeps
    the chip wiring uniform. We pick BAND_MOTOR (beta) as a no-op default.
    """
    if strategy in ("camp_gamma", "camp_hybrid_gamma"):
        return BAND_VISUAL   # high-gamma 70-150 Hz
    return BAND_MOTOR        # beta 13-30 Hz (or no-op for non-beta strategies)


# ============================================================================
# Sample-and-hold (server-side reconstruction)
# ============================================================================

class SampleAndHold:
    """Reconstructs full (C, T) stream from selected-channel transmissions.
    On-chip this is FREE — chip just doesn't transmit non-selected channels.
    Receiver holds last-known value and broadcasts forward."""

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self._hold = np.zeros(n_channels)
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

    def reset(self):
        self._hold[:] = 0.0
        self._initialized = False


# ============================================================================
# Chip simulator — bundles predictor + scheduler + S&H
# ============================================================================

class ChipSimulator:
    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        strategy: str,
        bandwidth_frac: float,
        band: BandSpec | None = None,
        floor_frac: float = 0.1,
        chunk_size: int = 30,
        update_period_s: float = 0.010,
        saliency_weights: np.ndarray | None = None,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        # If the caller doesn't override, pick the band based on strategy.
        # camp_gamma / camp_hybrid_gamma need the high-gamma band; everything
        # else defaults to beta (which is a no-op for non-CAMP strategies).
        if band is None:
            band = band_for_strategy(strategy)
        self.band = band
        self.predictor = OnChipPredictor(
            n_channels=n_channels, sfreq=sfreq, band=band,
            update_period_s=update_period_s,
        )
        self.scheduler = make_scheduler(
            strategy, n_channels, bandwidth_frac, floor_frac,
            saliency_weights=saliency_weights,
        )
        self.sah = SampleAndHold(n_channels)

    def step(self, chunk, t_start):
        weights = self.predictor.process_chunk(chunk, t_start)
        selected = self.scheduler.select(weights)
        transmitted = self.sah.step(chunk, selected)
        return transmitted, weights, selected

    def process(self, ecog: np.ndarray) -> np.ndarray:
        """Batch: process the whole recording, return chip-output (C, T)."""
        C, T = ecog.shape
        assert C == self.n_channels
        n_chunks = T // self.chunk_size
        T_out = n_chunks * self.chunk_size
        out = np.empty((C, T_out), dtype=np.float64)
        for i in range(n_chunks):
            s, e = i * self.chunk_size, (i + 1) * self.chunk_size
            out[:, s:e], _, _ = self.step(ecog[:, s:e], s / self.sfreq)
        return out

    def reset(self):
        self.predictor.reset()
        self.scheduler.reset()
        self.sah.reset()


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    C, T_chunk = 62, 30
    sfreq = 1000.0
    n_chunks = 400

    total_T = n_chunks * T_chunk
    t = np.arange(total_T) / sfreq
    envelope = 1.0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)
    signal = rng.standard_normal((C, total_T))
    for c in range(5):
        signal[c] += envelope * np.sin(2 * np.pi * 20.0 * t)

    chip = ChipSimulator(
        n_channels=C, sfreq=sfreq, strategy="camp_beta",
        bandwidth_frac=0.2, floor_frac=0.1, chunk_size=T_chunk,
    )
    selections = np.zeros(C, dtype=np.int64)
    for i in range(n_chunks):
        chunk = signal[:, i * T_chunk : (i + 1) * T_chunk]
        _, _, sel = chip.step(chunk, t_start=i * T_chunk / sfreq)
        selections += sel.astype(np.int64)

    print(f"top weights:   {np.argsort(-chip.predictor._weights)[:8].tolist()}")
    print(f"beta (0-4):    {selections[:5].tolist()}")
    print(f"noise (10-14): {selections[10:15].tolist()}")
    print(f"min/max:       {selections.min()} / {selections.max()}")
    assert selections[:5].mean() > selections[10:].mean()
    assert selections.min() > 0
    print("scheduler.py OK")