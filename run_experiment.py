"""
run_experiment.py — CAMP grid runner with two paradigms and deduplication.

Two evaluation paradigms:

  PARADIGM B — co-trained decoders (default).
    For each (strategy, B), train a fresh decoder on the chip's compressed
    output and evaluate it on the chip's compressed test output. Measures
    "information preservation" — given the chip's bandwidth-shaped stream,
    what's the ceiling on decoder accuracy if we get to retrain?

  PARADIGM A — single full-resolution decoder, evaluated on compressed.
    Train ONE decoder on full-resolution training data. Then evaluate it
    on each (strategy, B) chip-output test stream. Measures "deployment
    robustness" — does a clean-trained decoder still work on bandwidth-
    shaped input without retraining?

Deduplication: the runner skips redundant cells.
  - 'full' strategy at any B is identical (chip transmits everything).
    We train ONE 'full' decoder and broadcast its result.
  - 'round_robin' and 'camp' at B=1.0 are identical to 'full' (every
    channel transmits). We skip them.

Output (in results/grid.json):
  - one record per (paradigm, strategy, B) cell
  - 'best_r_mean', 'best_r_per_finger', 'history', etc.

Usage:
    # Paradigm B (default), full grid:
    python run_experiment.py --paradigm B --epochs 30 --device cuda

    # Paradigm A: train once on full, eval on each (strategy, B):
    python run_experiment.py --paradigm A --epochs 30 --device cuda

    # Both paradigms in one run:
    python run_experiment.py --paradigm both --epochs 30 --device cuda
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
from train import (
    train_from_arrays,
    pearson_r_per_finger,
    predict_full_sequence,
)


# ============================================================================
# Chip simulation helper
# ============================================================================

def simulate_chip(
    ecog_train: np.ndarray,
    ecog_test: np.ndarray,
    sfreq: float,
    strategy: str,
    bandwidth_frac: float,
    floor_frac: float,
    saliency_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the chip simulator separately on train and test ECoG.

    For strategy='full' we short-circuit: the chip output equals the input,
    just trimmed to chunk-aligned length.
    """
    chunk_size = 30
    if strategy == "full":
        T_tr = (ecog_train.shape[1] // chunk_size) * chunk_size
        T_te = (ecog_test.shape[1] // chunk_size) * chunk_size
        return ecog_train[:, :T_tr], ecog_test[:, :T_te]

    chip_tr = ChipSimulator(
        n_channels=ecog_train.shape[0], sfreq=sfreq,
        strategy=strategy, bandwidth_frac=bandwidth_frac, floor_frac=floor_frac,
        saliency_weights=saliency_weights,
    )
    chip_train = chip_tr.process(ecog_train)

    chip_te = ChipSimulator(
        n_channels=ecog_test.shape[0], sfreq=sfreq,
        strategy=strategy, bandwidth_frac=bandwidth_frac, floor_frac=floor_frac,
        saliency_weights=saliency_weights,
    )
    chip_test = chip_te.process(ecog_test)
    return chip_train, chip_test


# ============================================================================
# Cell builders — Paradigm B
# ============================================================================

def cell_paradigm_b(
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
    saliency_weights: np.ndarray | None = None,
) -> dict:
    """Train a fresh decoder on chip-output train, eval on chip-output test."""
    print(f"\n{'=' * 60}")
    print(f"[Paradigm B] strategy={strategy}  B={bandwidth_frac}")
    print(f"{'=' * 60}")
    t0 = time.monotonic()

    sfreq = split["sfreq"]
    chip_train, chip_test = simulate_chip(
        split["ecog_train"], split["ecog_test"], sfreq,
        strategy, bandwidth_frac, floor_frac,
        saliency_weights=saliency_weights,
    )
    fings_tr = split["fingers_train"][:, :chip_train.shape[1]]
    fings_te = split["fingers_test"][:,  :chip_test.shape[1]]
    print(f"  chip output train: {chip_train.shape}, test: {chip_test.shape}")

    print("  preprocess (FingerFlex pipeline)…")
    pre = preprocess_train_test(
        chip_train, fings_tr, chip_test, fings_te,
        sfreq=sfreq, line_freq=line_freq, verbose=False,
    )

    print(f"  train ({epochs} epochs, batch={batch_size}, device={device})…")
    cell_ckpt = ckpt_dir / f"B_{strategy}_b{bandwidth_frac}.pt"
    _, history = train_from_arrays(
        features_train=pre["features_train"],
        fingers_train=pre["fingers_train"],
        features_test=pre["features_test"],
        fingers_test=pre["fingers_test"],
        n_epochs=epochs, batch_size=batch_size, device=device,
        verbose=False, ckpt_path=cell_ckpt,
    )

    best = max(history, key=lambda h: h["r_mean"])
    elapsed = time.monotonic() - t0
    print(f"  → best r_mean = {best['r_mean']:.3f} "
          f"(epoch {best['epoch']})  [{elapsed:.0f}s]")
    return {
        "paradigm": "B",
        "strategy": strategy,
        "bandwidth_frac": bandwidth_frac,
        "best_r_mean": best["r_mean"],
        "best_r_per_finger": best["r_per_finger"],
        "best_epoch": best["epoch"],
        "history": history,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Cell builders — Paradigm A
# ============================================================================

def train_full_resolution_decoder(
    split: dict,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    line_freq: float,
    ckpt_dir: Path,
) -> tuple[dict, dict]:
    """Train ONE decoder on full-resolution data. Returns (artifacts, record).

    The artifacts dict carries the trained model and the full-res
    finger_scaler we'll need to transform test fingers in subsequent eval
    cells (so they're on the same scale the decoder was trained for).
    """
    print(f"\n{'=' * 60}")
    print(f"[Paradigm A] training full-resolution decoder (one-time)")
    print(f"{'=' * 60}")
    t0 = time.monotonic()

    sfreq = split["sfreq"]
    chip_train, chip_test = simulate_chip(
        split["ecog_train"], split["ecog_test"], sfreq,
        "full", 1.0, 0.0,
    )
    fings_tr = split["fingers_train"][:, :chip_train.shape[1]]
    fings_te = split["fingers_test"][:,  :chip_test.shape[1]]

    print("  preprocess full-resolution train+test…")
    pre = preprocess_train_test(
        chip_train, fings_tr, chip_test, fings_te,
        sfreq=sfreq, line_freq=line_freq, verbose=False,
    )

    print(f"  train ({epochs} epochs, batch={batch_size}, device={device})…")
    cell_ckpt = ckpt_dir / "A_full_decoder.pt"
    model, history = train_from_arrays(
        features_train=pre["features_train"],
        fingers_train=pre["fingers_train"],
        features_test=pre["features_test"],
        fingers_test=pre["fingers_test"],
        n_epochs=epochs, batch_size=batch_size, device=device,
        verbose=False, ckpt_path=cell_ckpt,
    )

    best = max(history, key=lambda h: h["r_mean"])
    elapsed = time.monotonic() - t0
    print(f"  → full-res r_mean = {best['r_mean']:.3f} "
          f"(epoch {best['epoch']})  [{elapsed:.0f}s]")

    artifacts = {
        "model": model,
        "finger_scaler": pre["finger_scaler"],  # the scaler the decoder learned
    }
    record = {
        "paradigm": "A",
        "strategy": "full",
        "bandwidth_frac": 1.0,
        "best_r_mean": best["r_mean"],
        "best_r_per_finger": best["r_per_finger"],
        "best_epoch": best["epoch"],
        "history": history,
        "elapsed_s": elapsed,
    }
    return artifacts, record


def eval_paradigm_a_cell(
    artifacts: dict,
    strategy: str,
    bandwidth_frac: float,
    split: dict,
    *,
    line_freq: float,
    floor_frac: float,
    device: str,
    saliency_weights: np.ndarray | None = None,
) -> dict:
    """Evaluate the pre-trained full-res decoder on (strategy, B) chip output.

    No training. Just chip-sim → preprocess → forward pass → Pearson r.
    Uses the original full-res finger_scaler (not a per-cell refit) — that's
    the realistic deployment scenario.
    """
    print(f"\n[Paradigm A] eval: strategy={strategy}  B={bandwidth_frac}")
    t0 = time.monotonic()

    sfreq = split["sfreq"]
    chip_train, chip_test = simulate_chip(
        split["ecog_train"], split["ecog_test"], sfreq,
        strategy, bandwidth_frac, floor_frac,
        saliency_weights=saliency_weights,
    )
    fings_tr = split["fingers_train"][:, :chip_train.shape[1]]
    fings_te = split["fingers_test"][:,  :chip_test.shape[1]]

    pre = preprocess_train_test(
        chip_train, fings_tr, chip_test, fings_te,
        sfreq=sfreq, line_freq=line_freq, verbose=False,
    )
    # Re-scale the test fingers using the FULL-RES finger_scaler (the one
    # the decoder was trained for). preprocess_train_test gave us fingers
    # scaled by a fresh per-cell scaler; we undo that and re-apply the
    # full-res one.
    cell_scaler = pre["finger_scaler"]
    full_scaler = artifacts["finger_scaler"]
    cell_test_scaled = pre["fingers_test"]
    cell_range = cell_scaler.max_ - cell_scaler.min_
    raw_test = cell_test_scaled * (cell_range + 1e-12) + cell_scaler.min_
    fings_test_full_scale = full_scaler.transform(raw_test)

    # Forward pass on the test features.
    model = artifacts["model"]
    pred = predict_full_sequence(model, pre["features_test"], device)
    T_pred = pred.shape[-1]
    true = fings_test_full_scale[..., :T_pred]
    r_per = pearson_r_per_finger(pred, true)
    r_mean = float(np.mean(r_per))

    elapsed = time.monotonic() - t0
    print(f"  → r_mean = {r_mean:.3f}  "
          f"per=[{', '.join(f'{r:.2f}' for r in r_per)}]  [{elapsed:.0f}s]")
    return {
        "paradigm": "A",
        "strategy": strategy,
        "bandwidth_frac": bandwidth_frac,
        "best_r_mean": r_mean,
        "best_r_per_finger": r_per.tolist(),
        "best_epoch": None,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Deduplication — which cells to actually run
# ============================================================================

def planned_cells(
    strategies: list[str], bandwidths: list[float]
) -> list[tuple[str, float]]:
    """Drop redundant cells.

    Rules:
      - 'full' is identical at all B → keep only ('full', max(bandwidths)).
        That single cell's r_mean stands in for all 'full' results.
      - 'round_robin' and 'camp' at B=1.0 are identical to 'full' → skip.
    """
    keep = []
    seen_full = False
    for s in strategies:
        for B in bandwidths:
            if s == "full":
                if not seen_full:
                    keep.append((s, max(bandwidths)))
                    seen_full = True
            else:
                if B >= 1.0:
                    continue   # same as full
                keep.append((s, B))
    return keep


# ============================================================================
# Grid driver
# ============================================================================

def run_grid(args) -> list[dict]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache_dir)

    split = load_subject_split(args.subject, cache_dir=cache_dir)

    # Load saliency vector if provided. Required if static_saliency or hybrid
    # are in the strategy list.
    saliency_weights = None
    if args.saliency_path is not None:
        sal_path = Path(args.saliency_path)
        if not sal_path.exists():
            raise FileNotFoundError(f"saliency file not found: {sal_path}")
        sal_data = np.load(sal_path)
        saliency_weights = sal_data["saliency"].astype(np.float64)
        print(f"[saliency] loaded {sal_path.name}, "
              f"shape={saliency_weights.shape}, "
              f"sum={saliency_weights.sum():.4f}")
        # Sanity check: should sum to ~1.
        if not np.isclose(saliency_weights.sum(), 1.0, atol=0.01):
            print(f"  warning: saliency sum is {saliency_weights.sum():.4f}, "
                  "renormalizing.")
            saliency_weights = saliency_weights / saliency_weights.sum()

    needs_saliency = any(
        s in ("static_saliency", "hybrid") for s in args.strategies
    )
    if needs_saliency and saliency_weights is None:
        raise ValueError(
            "static_saliency and hybrid strategies require --saliency-path"
        )

    grid_path = out_dir / "grid.json"
    results: list[dict] = []
    done_keys: set = set()
    if args.resume and grid_path.exists():
        with open(grid_path) as f:
            results = json.load(f)
        done_keys = {(r["paradigm"], r["strategy"], r["bandwidth_frac"])
                     for r in results if "error" not in r}
        print(f"[resume] {len(done_keys)} cells already done")

    cells_to_run = planned_cells(args.strategies, args.bandwidths)
    print(f"\nUnique cells to evaluate: {len(cells_to_run)}")
    for s, B in cells_to_run:
        print(f"  {s}  B={B}")

    paradigms = ["B", "A"] if args.paradigm == "both" else [args.paradigm]

    for paradigm in paradigms:
        print(f"\n{'#' * 60}\n# PARADIGM {paradigm}\n{'#' * 60}")

        if paradigm == "A":
            # Train the full-res decoder ONCE per paradigm-A pass.
            full_key = ("A", "full", max(args.bandwidths))
            if full_key not in done_keys:
                artifacts, full_record = train_full_resolution_decoder(
                    split, epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, line_freq=args.line_freq,
                    ckpt_dir=ckpt_dir,
                )
                results.append(full_record)
                with open(grid_path, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                # On true resume, we still need the artifacts in memory to
                # evaluate the remaining cells. Retrain from cache if we hit
                # this — the cache makes the data load ~instant, only the
                # 18min training is repeated. (Future improvement: pickle
                # the model state to disk and reload here.)
                print("[resume] retraining full-res decoder to recover model "
                      "in memory (fast cache load)…")
                artifacts, _ = train_full_resolution_decoder(
                    split, epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, line_freq=args.line_freq,
                    ckpt_dir=ckpt_dir,
                )

            for s, B in cells_to_run:
                if s == "full":
                    continue   # already covered by full_record
                key = ("A", s, B)
                if key in done_keys:
                    print(f"  [skip] {s} B={B}")
                    continue
                try:
                    r = eval_paradigm_a_cell(
                        artifacts, s, B, split,
                        line_freq=args.line_freq,
                        floor_frac=args.floor_frac,
                        device=args.device,
                        saliency_weights=saliency_weights,
                    )
                    results.append(r)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results.append({
                        "paradigm": "A", "strategy": s, "bandwidth_frac": B,
                        "error": str(e),
                    })
                with open(grid_path, "w") as f:
                    json.dump(results, f, indent=2)

        elif paradigm == "B":
            for s, B in cells_to_run:
                key = ("B", s, B)
                if key in done_keys:
                    print(f"  [skip] {s} B={B}")
                    continue
                try:
                    r = cell_paradigm_b(
                        s, B, split,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=args.device,
                        floor_frac=args.floor_frac,
                        line_freq=args.line_freq,
                        ckpt_dir=ckpt_dir,
                        saliency_weights=saliency_weights,
                    )
                    results.append(r)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results.append({
                        "paradigm": "B", "strategy": s, "bandwidth_frac": B,
                        "error": str(e),
                    })
                with open(grid_path, "w") as f:
                    json.dump(results, f, indent=2)

    # CSV summary.
    csv_path = out_dir / "grid.csv"
    with open(csv_path, "w") as f:
        f.write("paradigm,strategy,bandwidth_frac,r_mean,"
                "r_thumb,r_index,r_middle,r_ring,r_little\n")
        for r in results:
            if "error" in r:
                f.write(f"{r['paradigm']},{r['strategy']},"
                        f"{r['bandwidth_frac']},ERR,,,,,\n")
                continue
            rp = r["best_r_per_finger"]
            f.write(f"{r['paradigm']},{r['strategy']},{r['bandwidth_frac']},"
                    f"{r['best_r_mean']:.4f},"
                    f"{rp[0]:.4f},{rp[1]:.4f},{rp[2]:.4f},{rp[3]:.4f},{rp[4]:.4f}\n")
    print(f"\nSaved {csv_path}")
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_grid(
    results: list[dict], out_path: Path, all_bandwidths: list[float],
) -> None:
    """Plot Pearson r vs bandwidth, separated by paradigm.

    For 'full' (B-independent), draws a dashed horizontal reference line
    at the trained decoder's r_mean.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available")
        return

    paradigms_present = sorted(
        {r["paradigm"] for r in results if "error" not in r}
    )
    if not paradigms_present:
        print("[plot] no successful results, skipping")
        return

    fig, axes = plt.subplots(
        1, len(paradigms_present),
        figsize=(7 * len(paradigms_present), 5),
        squeeze=False,
    )

    markers = {
        "full": "s", "round_robin": "o", "camp": "^",
        "static_saliency": "D", "hybrid": "*",
    }
    colors  = {
        "full": "tab:gray", "round_robin": "tab:red", "camp": "tab:blue",
        "static_saliency": "tab:green", "hybrid": "tab:purple",
    }

    for ax_i, p in enumerate(paradigms_present):
        ax = axes[0, ax_i]
        recs = [r for r in results if r["paradigm"] == p and "error" not in r]
        by_strategy: dict[str, list[tuple[float, float]]] = {}
        for r in recs:
            by_strategy.setdefault(r["strategy"], []).append(
                (r["bandwidth_frac"], r["best_r_mean"])
            )

        for strat, points in by_strategy.items():
            points.sort()
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            if strat == "full":
                B_range = sorted(all_bandwidths)
                ax.plot(B_range, [ys[0]] * len(B_range),
                        color=colors[strat], linestyle="--", linewidth=2,
                        label=f"full (r={ys[0]:.2f})")
                ax.scatter(xs, ys, marker=markers[strat], s=80,
                           color=colors[strat])
            else:
                ax.plot(xs, ys, marker=markers.get(strat, "o"),
                        linewidth=2, markersize=8, label=strat,
                        color=colors.get(strat))

        ax.set_xlabel("Bandwidth fraction")
        ax.set_ylabel("Mean Pearson r (validation)")
        ax.set_title(f"Paradigm {p}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle("CAMP: decoding accuracy vs bandwidth", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument(
        "--paradigm", choices=["A", "B", "both"], default="B",
        help="A=train-once-on-full, B=co-train-per-cell, both=run both."
    )
    ap.add_argument("--strategies", nargs="+",
                    default=["full", "round_robin", "camp"])
    ap.add_argument("--bandwidths", nargs="+", type=float,
                    default=[1.0, 0.9, 0.5, 0.2, 0.1])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--floor-frac", type=float, default=0.1)
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--out-dir", type=str, default="./results")
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--saliency-path", type=str, default=None,
        help="Path to saliency NPZ from saliency.py. Required if "
             "static_saliency or hybrid in --strategies.",
    )
    args = ap.parse_args()

    print(f"Paradigm:   {args.paradigm}")
    print(f"Strategies: {args.strategies}")
    print(f"Bandwidths: {args.bandwidths}")
    print(f"Device: {args.device}, batch={args.batch_size}, epochs={args.epochs}")

    t0 = time.monotonic()
    results = run_grid(args)
    elapsed = time.monotonic() - t0

    plot_grid(results, Path(args.out_dir) / "curve.png", args.bandwidths)

    print(f"\n{'=' * 60}\nDONE in {elapsed / 60:.1f} min\n{'=' * 60}")
    print(f"\nResults summary:")
    print(f"{'paradigm':<10}{'strategy':<14}{'B':>6}{'r_mean':>10}")
    for r in sorted(
        results,
        key=lambda x: (x.get("paradigm", ""), x.get("strategy", ""),
                       x.get("bandwidth_frac", 0)),
    ):
        if "error" in r:
            print(f"{r['paradigm']:<10}{r['strategy']:<14}"
                  f"{r['bandwidth_frac']:>6}  ERROR")
        else:
            print(f"{r['paradigm']:<10}{r['strategy']:<14}"
                  f"{r['bandwidth_frac']:>6}{r['best_r_mean']:>10.3f}")


if __name__ == "__main__":
    main()