# CAMP — Context-Aware MAC Protocol (simulation)

This repo contains a **Python simulation** of CAMP: a Layer-2 style channel scheduling scheme for high-density implantable BCIs. The core idea is to treat channel forwarding as **weighted fair queuing** over electrodes, where weights come from a lightweight activity predictor (or a hybrid of activity + offline saliency), and bandwidth is allocated with a **Deficit Round Robin (DRR)**-style scheduler.

This project’s evaluation is **simulation-only** and **preliminary** (single dataset subject/task/decoder, limited compute). The main downstream task is finger-flexion decoding on **BCI Competition IV Dataset 4** using a reproduction of the **FingerFlex** pipeline and model.

## What’s implemented

- **On-chip side (simulated)**: activity predictor → DRR scheduler → “transmit K of N channels per window”.
- **Receiver side (simulated)**: **sample-and-hold** reconstruction for channels not transmitted in a window.
- **Downstream decoder (off-chip)**: FingerFlex-style Morlet spectrogram features + 1D conv encoder/decoder trained to predict 5 finger trajectories.

## Repo structure

- `run_experiment.py`: main grid runner (strategy × bandwidth). Supports two paradigms:
  - **Paradigm B (co-trained, default)**: train a fresh decoder per (strategy, bandwidth).
  - **Paradigm A (robustness)**: train one full-resolution decoder, then evaluate it on compressed streams.
- `scheduler.py`: “on-chip” simulation:
  - `OnChipPredictor`: FIR bandpass + fast/slow EMA power tracking → baseline-relative weights
  - schedulers: `round_robin`, DRR-based CAMP variants, static saliency, and hybrid (saliency × activity)
  - `SampleAndHold`: receiver reconstruction model
  - `ChipSimulator`: bundles predictor + scheduler + sample-and-hold
- `preprocess.py`: FingerFlex-faithful preprocessing:
  - per-channel z-score, CAR (median across channels per time)
  - MNE bandpass + notch (40–300 Hz with 50/60 Hz harmonics)
  - Morlet power spectrograms (40 log-spaced freqs 40–300 Hz), decimate to 100 Hz
  - cubic interpolate fingers to 100 Hz, apply 200 ms lead, MinMax scale fingers (fit on train)
- `decoder.py`: FingerFlex `AutoEncoder1D` reproduction + the training loss
- `train.py`: stride-1 window training loop + full-sequence Pearson \(r\) validation
- `data_loader.py`: loads the **official** train/test split via `braindecode` and caches it under `cache/`
- `saliency.py`: computes **offline per-channel saliency** from a trained decoder (gradient-based) and writes an `.npz` vector used by `camp_saliency` / `camp_hybrid_*`
- `streams.py`: small “live streaming” helper (not used by the grid runner)
- `docs/`: background PDFs you referenced while writing
- `results/`: default output directory (grid results + checkpoints)
- `CAMP_ds4_colab.ipynb`: the Colab notebook used to train/test with a GPU

## Setup

### Environment

You’ll need Python + scientific stack + PyTorch. A minimal setup is:

```bash
conda env create -f environment.yml
conda activate camp
```

Notes:
- The first dataset load will download BCI Comp IV DS4 via MNE/braindecode and then cache `cache/raw_split_subject_*.pkl`.
- GPU is optional but strongly recommended for training speed (`--device cuda`).
- If you have an NVIDIA GPU, enable CUDA in the env (example):

```bash
conda install -n camp pytorch-cuda=12.1 -c nvidia
```

## Running experiments (local)

### 1) (Optional but recommended) Train full-res decoder and compute saliency

Saliency is required if you want to run `camp_saliency`, `camp_hybrid_beta`, or `camp_hybrid_gamma`.

1. Train a full-resolution decoder (one-time). This writes `A_full_decoder.pt` under `--out-dir/checkpoints/`:

```bash
python run_experiment.py \
  --paradigm A \
  --strategies full \
  --bandwidths 1.0 \
  --epochs 15 \
  --device cuda \
  --out-dir results/fullres_for_saliency
```

2. Compute saliency from that checkpoint:

```bash
python saliency.py \
  --checkpoint results/fullres_for_saliency/checkpoints/A_full_decoder.pt \
  --out-dir results/saliency \
  --device cuda
```

This produces `results/saliency/saliency_subject_1.npz` and a plot PNG.

### 2) Run the strategy × bandwidth grid

Paradigm **B** (co-trained) is the default and matches the “information preservation” framing.

```bash
python run_experiment.py \
  --paradigm B \
  --strategies full round_robin camp_beta camp_gamma camp_saliency camp_hybrid_beta camp_hybrid_gamma \
  --bandwidths 1.0 0.9 0.3 0.1 \
  --epochs 15 \
  --device cuda \
  --saliency-path results/saliency/saliency_subject_1.npz \
  --out-dir results/grid_v2 \
  --resume
```

Outputs (in `--out-dir`):
- `grid.json`: one record per evaluated cell, including training history
- `grid.csv`: compact summary table
- `curve.png`: accuracy-vs-bandwidth plot
- `checkpoints/`: best checkpoints per cell

### 3) Paradigm A (deployment-robustness)

```bash
python run_experiment.py \
  --paradigm A \
  --strategies full round_robin camp_beta camp_gamma camp_saliency camp_hybrid_beta camp_hybrid_gamma \
  --bandwidths 1.0 0.9 0.3 0.1 \
  --epochs 15 \
  --device cuda \
  --saliency-path results/saliency/saliency_subject_1.npz \
  --out-dir results/paradigm_A \
  --resume
```

## Strategies (what they mean)

All strategies operate on a 30 ms “window” (a `chunk_size=30` at 1 kHz), and a bandwidth fraction \(B \in (0,1]\) sets \(K=\lfloor B\cdot N\rfloor\) channels transmitted per window.

- `full`: transmit all channels every window (no bandwidth constraint)
- `round_robin`: transmit \(K\) channels cyclically (bandwidth-unaware baseline)
- `camp_beta`: DRR top-\(K\) using **beta-band** (13–30 Hz) baseline-relative activity weights
- `camp_gamma`: DRR top-\(K\) using **high-gamma** (70–150 Hz) baseline-relative activity weights
- `camp_saliency`: DRR top-\(K\) using **fixed offline saliency weights** (`saliency.py`)
- `camp_hybrid_beta`: DRR top-\(K\) using **saliency × beta-activity** (renormalized)
- `camp_hybrid_gamma`: DRR top-\(K\) using **saliency × gamma-activity** (renormalized)

## Colab notebook workflow (what `CAMP_ds4_colab.ipynb` does)

The notebook follows this pattern:

1. Install dependencies in Colab:
   - `pip install braindecode moabb`
2. Clone the repo.
3. Mount Google Drive and write outputs under:
   - `/content/drive/MyDrive/CAMP-results/...`
4. Run the grid (example from the notebook):

```bash
python run_experiment.py \
  --paradigm B \
  --strategies full round_robin camp_beta camp_gamma camp_saliency camp_hybrid_beta camp_hybrid_gamma \
  --bandwidths 1.0 0.9 0.3 0.1 \
  --epochs 15 \
  --device cuda \
  --saliency-path /content/drive/MyDrive/CAMP-results/saliency/saliency_subject_1.npz \
  --out-dir /content/drive/MyDrive/CAMP-results/grid_v2 \
  --resume
```

## Reproducibility notes / caveats

- **Compute & variance**: results can vary across epochs/seed; the grid reports the **best epoch** by mean Pearson \(r\).
- **No hardware claims**: the “on-chip” pipeline is simulated in Python; gate count/power remarks are conceptual.
- **Dataset**: electrode indices are scrambled in this dataset; channel “locations” are not meaningful here.

## License

See `LICENSE`.

