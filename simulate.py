"""
run_experiment.py — the headline figure of the CAMP paper.

For each (strategy, bandwidth_frac) cell:
  1. Load the raw ECoG + fingers
  2. Apply the chip simulator to get the bandwidth-shaped stream
     (sample-and-hold reconstruction, no server-side correction)
  3. Run server-side preprocessing (wavelets) on the simulated stream
  4. Train a FingerFlex decoder on the result
  5. Record best validation Pearson r

Output:
  - results/grid.json    — full grid of results
  - results/grid.csv     — one row per (strategy, bandwidth) cell
  - results/curve.png    — Pearson r vs bandwidth, one line per strategy
  - per-cell checkpoints in results/checkpoints/

KEY DESIGN POINT: The decoder is TRAINED FROM SCRATCH for each cell.
This is "Option B" from earlier — the decoder co-adapts to whatever
distortion the chip introduces. Justification: at low bandwidth, a clean-
trained decoder would just be confused by sample-and-held streams, which
would penalize all three strategies equally and tell us nothing about
their relative quality. Co-training isolates the question we care about:
"given fixed bandwidth, does CAMP route it more usefully than RR?"

Usage:
    python run_experiment.py --strategies full round_robin camp \\
                             --bandwidths 0.9 0.5 0.2 0.1 \\
                             --epochs 30
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from scheduler import ChipSimulator
from preprocess import run_preprocessing_from_arrays
from train import train_from_arrays


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #

def load_subject(subject_id: int, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load raw ECoG + fingers, with disk cache to avoid hitting braindecode
    on every run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"raw_subject_{subject_id}.pkl"
    if cache.exists():
        print(f"[cache] loading {cache}")
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return d["ecog"], d["fingers"], d["sfreq"]

    print(f"[load] subject {subject_id} via braindecode…")
    from braindecode.datasets import BCICompetitionIVDataset4
    ds = BCICompetitionIVDataset4(subject_ids=[subject_id])
    raw = ds.datasets[0].raw
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    ecog, fingers = data[:62], data[62:67]
    with open(cache, "wb") as f:
        pickle.dump({"ecog": ecog, "fingers": fingers, "sfreq": sfreq}, f)
    print(f"  cached → {cache}")
    return ecog, fingers, sfreq


# --------------------------------------------------------------------------- #
# Single cell
# --------------------------------------------------------------------------- #

def run_cell(
    strategy: str,
    bandwidth_frac: float,
    ecog: np.ndarray,
    fingers: np.ndarray,
    sfreq: float,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    floor_frac: float,
    ckpt_dir: Path,
    line_freq: float,
    skip_chip_sim_for_full: bool = True,
) -> dict:
    """Run one (strategy, B) experiment cell. Returns a dict of results."""
    print(f"\n{'=' * 60}")
    print(f"CELL: strategy={strategy}  B={bandwidth_frac}")
    print(f"{'=' * 60}")

    t0 = time.monotonic()

    # 1. Chip simulation. For 'full' at any B we can short-circuit: the
    # full-resolution stream is just the raw ECoG (rounded to multiples of
    # chunk_size). Skip the chip sim and the preprocessing-on-sim cost.
    if strategy == "full" and skip_chip_sim_for_full:
        print("  [chip] skipping (full resolution = raw ECoG)")
        # Trim to chunk-aligned length for consistency with other cells.
        chunk_size = 30
        T_aligned = (ecog.shape[1] // chunk_size) * chunk_size
        chip_out = ecog[:, :T_aligned]
    else:
        print(f"  [chip] simulating strategy={strategy} B={bandwidth_frac}…")
        chip = ChipSimulator(
            n_channels=ecog.shape[0],
            sfreq=sfreq,
            strategy=strategy,
            bandwidth_frac=bandwidth_frac,
            floor_frac=floor_frac,
        )
        chip_out = chip.process(ecog)

    fingers_aligned = fingers[:, :chip_out.shape[1]]
    print(f"    chip output: {chip_out.shape}, fingers: {fingers_aligned.shape}")

    # 2. Server-side preprocessing.
    print("  [preprocess] running wavelets + decimation…")
    preproc = run_preprocessing_from_arrays(
        chip_out, fingers_aligned, sfreq,
        line_freq=line_freq, verbose=False,
    )

    # 3. Train decoder.
    print(f"  [train] training decoder ({epochs} epochs)…")
    cell_ckpt = ckpt_dir / f"{strategy}_b{bandwidth_frac}.pt"
    model, history = train_from_arrays(
        features=preproc["features"],
        fingers=preproc["fingers"],
        n_train=preproc["n_train"],
        n_epochs=epochs,
        batch_size=batch_size,
        device=device,
        verbose=False,
        ckpt_path=cell_ckpt,
    )

    best = max(history, key=lambda h: h["r_mean"])
    elapsed = time.monotonic() - t0
    print(f"  → best r_mean = {best['r_mean']:.3f}  "
          f"(epoch {best['epoch']})  [{elapsed:.0f}s]")

    return {
        "strategy": strategy,
        "bandwidth_frac": bandwidth_frac,
        "best_r_mean": best["r_mean"],
        "best_r_per_finger": best["r_per_finger"],
        "best_epoch": best["epoch"],
        "history": history,
        "elapsed_s": elapsed,
    }


# --------------------------------------------------------------------------- #
# Grid runner
# --------------------------------------------------------------------------- #

def run_grid(args) -> list[dict]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache_dir)

    ecog, fingers, sfreq = load_subject(args.subject, cache_dir)
    print(f"Data: ECoG {ecog.shape}, fingers {fingers.shape}, fs={sfreq}")

    results = []
    grid_path = out_dir / "grid.json"

    # Resume support: if grid.json exists, skip cells already completed.
    done_keys = set()
    if args.resume and grid_path.exists():
        with open(grid_path) as f:
            results = json.load(f)
        done_keys = {(r["strategy"], r["bandwidth_frac"]) for r in results}
        print(f"[resume] {len(done_keys)} cells already done")

    for strategy in args.strategies:
        for B in args.bandwidths:
            if (strategy, B) in done_keys:
                print(f"  [skip] {strategy} B={B} (already done)")
                continue
            try:
                r = run_cell(
                    strategy, B, ecog, fingers, sfreq,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    floor_frac=args.floor_frac,
                    ckpt_dir=ckpt_dir,
                    line_freq=args.line_freq,
                )
                results.append(r)
            except Exception as e:
                print(f"  ERROR in cell ({strategy}, {B}): {e}")
                results.append({
                    "strategy": strategy, "bandwidth_frac": B,
                    "error": str(e),
                })

            # Save after every cell so a crash doesn't lose hours of work.
            with open(grid_path, "w") as f:
                json.dump(results, f, indent=2)

    # CSV summary.
    csv_path = out_dir / "grid.csv"
    with open(csv_path, "w") as f:
        f.write("strategy,bandwidth_frac,r_mean,r_thumb,r_index,r_middle,r_ring,r_little\n")
        for r in results:
            if "error" in r:
                f.write(f"{r['strategy']},{r['bandwidth_frac']},ERR,,,,,\n")
                continue
            rp = r["best_r_per_finger"]
            f.write(f"{r['strategy']},{r['bandwidth_frac']},{r['best_r_mean']:.4f},"
                    f"{rp[0]:.4f},{rp[1]:.4f},{rp[2]:.4f},{rp[3]:.4f},{rp[4]:.4f}\n")
    print(f"\nSaved {csv_path}")

    return results


def plot_grid(results: list[dict], out_path: Path) -> None:
    """Pearson r vs bandwidth, one line per strategy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping plot")
        return

    by_strategy = {}
    for r in results:
        if "error" in r: continue
        by_strategy.setdefault(r["strategy"], []).append(
            (r["bandwidth_frac"], r["best_r_mean"])
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"full": "s", "round_robin": "o", "camp": "^"}
    for strat, points in by_strategy.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker=markers.get(strat, "o"), linewidth=2,
                markersize=8, label=strat)

    ax.set_xlabel("Bandwidth fraction")
    ax.set_ylabel("Mean Pearson r (validation)")
    ax.set_title("CAMP: decoding accuracy vs available bandwidth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--strategies", nargs="+",
                    default=["full", "round_robin", "camp"])
    ap.add_argument("--bandwidths", nargs="+", type=float,
                    default=[0.9, 0.5, 0.2, 0.1])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--floor-frac", type=float, default=0.1)
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--out-dir", type=str, default="./results")
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already in results/grid.json")
    args = ap.parse_args()

    print(f"Strategies: {args.strategies}")
    print(f"Bandwidths: {args.bandwidths}")
    print(f"Total cells: {len(args.strategies) * len(args.bandwidths)}")

    t0 = time.monotonic()
    results = run_grid(args)
    elapsed = time.monotonic() - t0

    plot_grid(results, Path(args.out_dir) / "curve.png")

    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")
    print(f"\nResults summary:")
    print(f"{'strategy':<14}{'B':>6}{'r_mean':>10}")
    for r in sorted(results, key=lambda x: (x.get("strategy"), x.get("bandwidth_frac", 0))):
        if "error" in r:
            print(f"{r['strategy']:<14}{r['bandwidth_frac']:>6}  ERROR")
        else:
            print(f"{r['strategy']:<14}{r['bandwidth_frac']:>6}{r['best_r_mean']:>10.3f}")


if __name__ == "__main__":
    main()