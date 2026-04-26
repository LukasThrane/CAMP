"""
streams.py

Loads BCI Competition IV Dataset 4 and streams ECoG + finger data
into a shared multiprocessing Queue, simulating live channel output.
"""

import time
import numpy as np
from multiprocessing import Queue

from braindecode.datasets import BCICompetitionIVDataset4


def fill_nan_by_interpolation(x: np.ndarray) -> np.ndarray:
    x_filled = x.copy()
    time_idx = np.arange(x.shape[1])
    for ch in range(x.shape[0]):
        y = x[ch]
        valid = ~np.isnan(y)
        if valid.sum() == 0:
            x_filled[ch] = 0.0
            continue
        x_filled[ch] = np.interp(time_idx, time_idx[valid], y[valid])
    return x_filled


def load_data():
    dataset = BCICompetitionIVDataset4(subject_ids=[1])
    raw = dataset.datasets[0].raw
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])

    ecog    = data[:62]
    fingers = fill_nan_by_interpolation(data[62:67])

    print(f"Loaded data — ECoG: {ecog.shape}, Fingers: {fingers.shape}")
    return ecog, fingers, sfreq


def stream(queue: Queue,
           ecog: np.ndarray,
           fingers: np.ndarray,
           sfreq: float,
           step_ms: int = 30,
           loop: bool = True) -> None:
    """
    Pushes chunks of (ecog_chunk, finger_chunk, t_start) into the queue
    at the real-time rate defined by step_ms.

    Each queue item is a dict:
        {
            'ecog':     np.ndarray (62, step_samples),
            'fingers':  np.ndarray (5,  step_samples),
            't_start':  float  — time in seconds of first sample in chunk,
            'sfreq':    float  — sampling rate
        }
    """
    step_samples = int(sfreq * step_ms / 1000.0)
    n_times      = ecog.shape[1]
    cursor       = 0

    print(f"Streaming — step: {step_ms}ms ({step_samples} samples), "
          f"loop: {loop}")

    while True:
        start = cursor
        end   = cursor + step_samples

        if end > n_times:
            if loop:
                cursor = 0
                continue
            else:
                print("Stream finished.")
                break

        chunk = {
            'ecog':    ecog[:, start:end],
            'fingers': fingers[:, start:end],
            't_start': start / sfreq,
            'sfreq':   sfreq,
        }

        # Block if queue is full — backpressure so consumer keeps up
        queue.put(chunk)

        cursor += step_samples
        time.sleep(step_ms / 1000.0)  # simulate real time


if __name__ == "__main__":
    from multiprocessing import Queue

    q = Queue(maxsize=50)
    ecog, fingers, sfreq = load_data()

    print("Streaming to queue — press Ctrl+C to stop.")
    try:
        stream(q, ecog, fingers, sfreq, step_ms=30, loop=True)
    except KeyboardInterrupt:
        print("Stopped.")