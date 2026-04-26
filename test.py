import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from braindecode.datasets import BCICompetitionIVDataset4

# ---------- settings ----------
CHANNELS_TO_SHOW = 16          # how many channels to plot
SECONDS_ON_SCREEN = 5.0        # rolling window size
UPDATE_MS = 50                 # update interval (ms)
BANDWIDTH_FRACTION = 1.0       # 1.0 = full, 0.5 = keep half, etc.
SEED = 0
# ----------------------------

rng = np.random.default_rng(SEED)

dataset = BCICompetitionIVDataset4(subject_ids=[1])
raw = dataset.datasets[0].raw  # mne.io.Raw-like
sfreq = float(raw.info["sfreq"])
data = raw.get_data()          # shape: (n_channels, n_times)
n_ch, n_times = data.shape

ch_idx = np.arange(min(CHANNELS_TO_SHOW, n_ch))
data = data[ch_idx]

chunk = int(round(sfreq * (UPDATE_MS / 1000.0)))
buf_len = int(round(sfreq * SECONDS_ON_SCREEN))
buf_raw = np.zeros((len(ch_idx), buf_len), dtype=np.float64)
buf_tx  = np.zeros((len(ch_idx), buf_len), dtype=np.float64)

# sample-and-hold state per channel
last = np.zeros((len(ch_idx),), dtype=np.float64)

def apply_bandwidth_and_hold(x, keep_frac):
    """x: (C, T) -> (C, T) with random drops + sample-and-hold."""
    if keep_frac >= 1.0:
        return x.copy()

    C, T = x.shape
    keep = rng.random((C, T)) < keep_frac
    y = x.copy()

    # set dropped positions to NaN, then forward-fill with last value
    y[~keep] = np.nan
    for c in range(C):
        row = y[c]
        # seed with last held value
        if np.isnan(row[0]):
            row[0] = last[c]
        # forward fill
        for t in range(1, T):
            if np.isnan(row[t]):
                row[t] = row[t - 1]
        last[c] = row[-1]
    return y

app = pg.mkQApp("CAMP streaming sim")
pg.setConfigOptions(antialias=True)

win = pg.GraphicsLayoutWidget(show=True, title="Channel streaming simulation")
win.resize(1200, 800)

p1 = win.addPlot(title="Raw (selected channels)")
p1.showGrid(x=True, y=True, alpha=0.3)
win.nextRow()
p2 = win.addPlot(title=f"Transmitted (bandwidth_fraction={BANDWIDTH_FRACTION}, sample-and-hold)")
p2.showGrid(x=True, y=True, alpha=0.3)

curves1, curves2 = [], []
offset = 0.0
for i in range(len(ch_idx)):
    c1 = p1.plot(pen=pg.intColor(i, hues=len(ch_idx)))
    c2 = p2.plot(pen=pg.intColor(i, hues=len(ch_idx)))
    curves1.append(c1)
    curves2.append(c2)

# simple vertical stacking so channels don't overlap
STACK = 200e-6  # adjust for visibility (dataset units are Volts)
t_axis = np.arange(buf_len) / sfreq

t0 = 0  # time index into recording

def update():
    global buf_raw, buf_tx, t0

    if t0 + chunk >= n_times:
        t0 = 0  # loop

    x = data[:, t0:t0 + chunk]  # (C, chunk)
    y = apply_bandwidth_and_hold(x, BANDWIDTH_FRACTION)

    # roll buffers and append new chunk
    buf_raw = np.roll(buf_raw, -chunk, axis=1)
    buf_tx  = np.roll(buf_tx,  -chunk, axis=1)
    buf_raw[:, -chunk:] = x
    buf_tx[:,  -chunk:] = y

    for i in range(len(ch_idx)):
        curves1[i].setData(t_axis, buf_raw[i] + i * STACK)
        curves2[i].setData(t_axis, buf_tx[i]  + i * STACK)

    t0 += chunk

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_MS)

pg.exec()