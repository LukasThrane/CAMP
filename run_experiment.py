"""
run_experiment.py — the headline figure of the CAMP paper.

For each (strategy, bandwidth_frac) cell:
  1. Load subject's official train + test split
  2. Apply the chip simulator to BOTH train and test ECoG
     (the chip behaves identically — causal, online — on both streams)
  3. Run FingerFlex preprocessing on the chip-simulated streams
     (CAR, MNE bandpass + notch, Morlet wavelets, decimation)
  4. Train a FingerFlex AutoEncoder1D from scratch on the result
  5. Record best validation Pearson r (full-sequence prediction on test)

KEY DESIGN:
  - Decoder is trained PER CELL on chip-simulated data. This co-adapts the
    decoder to whatever distortion the chip introduces, isolating the
    question "given fixed bandwidth, does CAMP route it more usefully than
    round-robin?"
  - We use the OFFICIAL train/test split (sub1_comp.mat train_data vs
    test_data), not a 65/35 partition of the concatenated recording. This
    matches FingerFlex and the BCI Competition IV evaluation protocol.

Usage:
    python run_experiment.py --strategies full round_robin camp \\
                             --bandwidths 0.9 0.5 0.2 0.1 \\
                             --epochs 30 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_loader import load_subject_split
from scheduler import ChipSimulator
from preprocess import preprocess_train_test
from train import train_from_arrays


# ============================================================================
# Single cell
# ============================================================================

def run_cell(
    strategy: str,
    bandwidth_frac: float,
    split: dict,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    floor_frac: float,
    line_freq: float,
    ckpt_dir: Path,
    skip_chip_for_full: bool = True,
) -> dict:
    """Run one (strategy, B) cell. Returns a dict of results."""
    print(f"\n{'=' * 60}")
    print(f"CELL: strategy={strategy}  B={bandwidth_frac}")
    print(f"{'=' * 60}")
    t0 = time.monotonic()

    sfreq = split["sfreq"]
    ecog_train = split["ecog_train"]
    ecog_test  = split["ecog_test"]
    fingers_train = split["fingers_train"]
    fingers_test  = split["fingers_test"]

    # ----- 1. Chip simulation, separately on train and test -----
    if strategy == "full" and skip_chip_for_full:
        print("  [chip] skipping (full resolution = raw ECoG)")
        chunk_size = 30
        T_train_aligned = (ecog_train.shape[1] // chunk_size) * chunk_size
        T_test_aligned  = (ecog_test.shape[1] // chunk_size) * chunk_size
        chip_train = ecog_train[:, :T_train_aligned]
        chip_test  = ecog_test[:, :T_test_aligned]
    else:
        print(f"  [chip] simulating {strategy} B={bandwidth_frac}…")
        # Build a fresh chip per stream — they have independent state.
        chip_tr = ChipSimulator(
            n_channels=ecog_train.shape[0], sfreq=sfreq,
            strategy=strategy, bandwidth_frac=bandwidth_frac,
            floor_frac=floor_frac,
        )
        chip_train = chip_tr.process(ecog_train)
        chip_te = ChipSimulator(
            n_channels=ecog_test.shape[0], sfreq=sfreq,
            strategy=strategy, bandwidth_frac=bandwidth_frac,
            floor_frac=floor_frac,
        )
        chip_test = chip_te.process(ecog_test)

    # Trim fingers to match chip output (chip rounds down to chunk_size).
    fings_train = fingers_train[:, :chip_train.shape[1]]
    fings_test  = fingers_test[:, :chip_test.shape[1]]
    print(f"    train: {chip_train.shape}, test: {chip_test.shape}")

    # ----- 2. Server-side preprocessing -----
    print("  [preprocess] running FingerFlex pipeline…")
    pre = preprocess_train_test(
        chip_train, fings_train, chip_test, fings_test,
        sfreq=sfreq, line_freq=line_freq, verbose=False,
    )

    # ----- 3. Train decoder -----
    print(f"  [train] {epochs} epochs (batch={batch_size}, device={device})…")
    cell_ckpt = ckpt_dir / f"{strategy}_b{bandwidth_frac}.pt"
    model, history = train_from_arrays(
        features_train=pre["features_train"],
        fingers_train=pre["fingers_train"],
        features_test=pre["features_test"],
        fingers_test=pre["fingers_test"],
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


# ============================================================================
# Grid runner
# ============================================================================

def run_grid(args) -> list[dict]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache_dir)

    split = load_subject_split(args.subject, cache_dir=cache_dir)

    grid_path = out_dir / "grid.json"
    results = []
    done_keys = set()
    if args.resume and grid_path.exists():
        with open(grid_path) as f:
            results = json.load(f)
        done_keys = {(r["strategy"], r["bandwidth_frac"]) for r in results
                     if "error" not in r}
        print(f"[resume] {len(done_keys)} cells already done")

    for strategy in args.strategies:
        for B in args.bandwidths:
            if (strategy, B) in done_keys:
                print(f"  [skip] {strategy} B={B}")
                continue
            try:
                r = run_cell(
                    strategy, B, split,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    floor_frac=args.floor_frac,
                    line_freq=args.line_freq,
                    ckpt_dir=ckpt_dir,
                )
                results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({
                    "strategy": strategy, "bandwidth_frac": B,
                    "error": str(e),
                })
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
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--floor-frac", type=float, default=0.1)
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--out-dir", type=str, default="./results")
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print(f"Strategies: {args.strategies}")
    print(f"Bandwidths: {args.bandwidths}")
    print(f"Total cells: {len(args.strategies) * len(args.bandwidths)}")
    print(f"Device: {args.device}, batch_size: {args.batch_size}, epochs: {args.epochs}")

    t0 = time.monotonic()
    results = run_grid(args)
    elapsed = time.monotonic() - t0

    plot_grid(results, Path(args.out_dir) / "curve.png")

    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}\nResults summary:")
    print(f"{'strategy':<14}{'B':>6}{'r_mean':>10}")
    for r in sorted(results, key=lambda x: (x.get("strategy"), x.get("bandwidth_frac", 0))):
        if "error" in r:
            print(f"{r['strategy']:<14}{r['bandwidth_frac']:>6}  ERROR")
        else:
            print(f"{r['strategy']:<14}{r['bandwidth_frac']:>6}{r['best_r_mean']:>10.3f}")


if __name__ == "__main__":
    main()