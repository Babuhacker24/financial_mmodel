"""
Time-resolved dynamics of the weighted linear-threshold cascade.

Where run_phase_diagram.py answers "how big does the cascade get?", this
script answers "how fast does it unfold?"

We compute four dynamics descriptors per (tau, k, strategy) cell:

    duration   T*       -- total number of update steps
    t_half     T_{1/2}  -- step at which 50% of nodes have failed
    t_peak              -- step at which the largest burst of new failures
                            occurs
    peak_rate           -- size of that burst (new failures / step)

and produce three figure families:

    1. Trajectory overlays (failed fraction vs. t, several taus on one axes)
    2. Failure waterfalls (new failures per step + cumulative on twin axes)
    3. Heatmaps of t_half and peak_rate over the (tau, k) grid.

Run from the project root:

    python src/run_dynamics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments import load_adjacency, run_dynamics_grid
from visualize import (
    trajectory_overlay,
    failure_waterfall,
    heatmap_thalf,
    heatmap_peakrate,
)


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    W, _ = load_adjacency(OUTPUT_DIR / "adjacency_matrix_thresholded.csv")
    print(f"Loaded W: {W.shape[0]} nodes, "
          f"{int((W > 0).sum() // 2)} edges.")

    # ---- 1. Dynamics grid ---------------------------------------------------
    print("Sweeping tau x k x strategy and recording dynamics ...")
    df = run_dynamics_grid(
        W,
        taus=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
        seed_sizes=(1, 3, 5, 10, 15, 20, 30, 50, 75, 100),
        n_random=50,
    )
    df = df.sort_values(["tau", "n_seeds", "strategy"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "dynamics_results.csv", index=False)

    # ---- 2. Print compact summary -----------------------------------------
    print("\n=== Time to half-collapse t_{1/2} (strategy = high) ===")
    pivot = (df[df["strategy"] == "high"]
             .pivot(index="tau", columns="n_seeds", values="t_half")
             .sort_index(ascending=False))
    print(pivot.to_string(float_format=lambda x: "  -" if pd.isna(x)
                          else f"{x:3.0f}"))

    print("\n=== Peak failure rate (new failures / step), strategy=high ===")
    pivot = (df[df["strategy"] == "high"]
             .pivot(index="tau", columns="n_seeds", values="peak_rate")
             .sort_index(ascending=False))
    print(pivot.to_string(float_format=lambda x: f"{x:3.0f}"))

    print("\n=== Cascade duration T* (steps to absorption), strategy=high ===")
    pivot = (df[df["strategy"] == "high"]
             .pivot(index="tau", columns="n_seeds", values="duration")
             .sort_index(ascending=False))
    print(pivot.to_string(float_format=lambda x: f"{x:3.0f}"))

    # ---- 3. Figures --------------------------------------------------------
    # 3a. Trajectory overlays for several seed sizes (high-centrality)
    for k in (5, 10, 30):
        trajectory_overlay(
            W, k=k, taus=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
            strategy="high",
            out_path=FIG_DIR / f"trajectories_overlay_high_k{k}.png",
        )
    # And one for random targeting at k = 30
    trajectory_overlay(
        W, k=30, taus=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        strategy="random",
        out_path=FIG_DIR / "trajectories_overlay_random_k30.png",
    )

    # 3b. Waterfalls — pick canonical "ignition" cells from the phase diagram
    waterfall_cases = [
        ("high",   0.10, 10, "ignition_easy"),    # super-critical, fast
        ("high",   0.20, 30, "ignition_medium"),  # super-critical, slower
        ("high",   0.30, 50, "ignition_hard"),    # super-critical, latest
        ("random", 0.10, 30, "ignition_random"),  # random in super-crit
        ("high",   0.30, 10, "subcritical_demo"), # contained
    ]
    for strat, tau, k, tag in waterfall_cases:
        failure_waterfall(
            W, tau=tau, k=k, strategy=strat,
            out_path=FIG_DIR / f"waterfall_{tag}.png",
        )

    # 3c. Heatmaps of dynamic descriptors
    for strat in ("high", "random"):
        heatmap_thalf(df, strategy=strat,
                      out_path=FIG_DIR / f"thalf_{strat}.png")
        heatmap_peakrate(df, strategy=strat,
                         out_path=FIG_DIR / f"peakrate_{strat}.png")

    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Full table:       {OUTPUT_DIR / 'dynamics_results.csv'}")


if __name__ == "__main__":
    main()
