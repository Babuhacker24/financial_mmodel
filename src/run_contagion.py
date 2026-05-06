"""
Entry point for the contagion experiments.

Reads the thresholded adjacency matrix produced by build_network.py, runs the
factorial sweep tau x strategy x k, and writes results + figures to outputs/.

Run from the repo root:

    python src/run_contagion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments import load_adjacency, run_grid
from visualize import (
    cascade_trajectories,
    heatmap_size,
    compare_strategies,
)


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # -------- 1. Load network ------------------------------------------------
    W, tickers = load_adjacency(OUTPUT_DIR / "adjacency_matrix_thresholded.csv")
    N = W.shape[0]
    print(f"Loaded thresholded adjacency: {N} nodes, "
          f"{int((W > 0).sum() // 2)} undirected edges, "
          f"density = {((W > 0).sum() / (N * (N - 1))):.4f}")

    # -------- 2. Factorial sweep --------------------------------------------
    df = run_grid(
        W,
        taus=(0.2, 0.3, 0.5),
        seed_sizes=(1, 3, 5, 10),
        n_random=200,
    )
    df = df.sort_values(["tau", "n_seeds", "strategy"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "contagion_results.csv", index=False)
    print("\n=== Cascade results ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # -------- 3. Figures -----------------------------------------------------
    cascade_trajectories(W, k=1,  out_path=FIG_DIR / "trajectories_k1.png")
    cascade_trajectories(W, k=5,  out_path=FIG_DIR / "trajectories_k5.png")
    cascade_trajectories(W, k=10, out_path=FIG_DIR / "trajectories_k10.png")

    for strat in ("high", "low", "random"):
        heatmap_size(df, strategy=strat,
                     out_path=FIG_DIR / f"heatmap_{strat}.png")

    compare_strategies(df, k=5,  out_path=FIG_DIR / "strategies_k5.png")
    compare_strategies(df, k=10, out_path=FIG_DIR / "strategies_k10.png")

    # -------- 4. Quick takeaway ---------------------------------------------
    pivot = df.pivot_table(
        index=["tau", "strategy"], columns="n_seeds",
        values="final_size", aggfunc="first",
    )
    print("\n=== Final cascade size (rows: tau, strategy; cols: k) ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Results table:    {OUTPUT_DIR / 'contagion_results.csv'}")


if __name__ == "__main__":
    main()
