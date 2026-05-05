"""
Phase diagram for the weighted linear-threshold cascade.

Sweeps a fine grid (tau, k) and plots the final cascade size as a heatmap.
The boundary between "shock absorbed" (small final size) and "global cascade"
(large final size) is the systemic-risk frontier of the network.

Three heatmaps are produced, one per seed strategy:
    high   -- top-k by eigenvector centrality      (worst case)
    random -- mean over Monte Carlo replicates     (average case)
    low    -- bottom-k by eigenvector centrality   (best case)

Run from the project root:

    python src/run_phase_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contagion import simulate_cascade
from centrality import (
    eigenvector_centrality,
    weighted_degree,
    select_seeds,
)
from experiments import load_adjacency


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# --- grid -----------------------------------------------------------------
TAUS  = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
KS    = np.array([1, 3, 5, 10, 15, 20, 30, 50, 75, 100])
N_RND = 100                  # Monte Carlo replicates for random strategy
RNG_S = 20260430


# --------------------------------------------------------------------------- #

def sweep(W: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_S)
    eig = eigenvector_centrality(W)
    strength = weighted_degree(W)

    rows = []
    for tau in TAUS:
        for k in KS:
            # high-centrality (deterministic)
            seeds = select_seeds(eig, k, "high",
                                 exclude_isolated_strength=strength)
            r = simulate_cascade(W, seeds, float(tau))
            rows.append(dict(tau=float(tau), k=int(k), strategy="high",
                             final_size=r.final_size, duration=r.duration))

            # low-centrality (deterministic)
            seeds = select_seeds(eig, k, "low",
                                 exclude_isolated_strength=strength)
            r = simulate_cascade(W, seeds, float(tau))
            rows.append(dict(tau=float(tau), k=int(k), strategy="low",
                             final_size=r.final_size, duration=r.duration))

            # random (Monte Carlo)
            sizes, durs = [], []
            for _ in range(N_RND):
                seeds = select_seeds(eig, k, "random", rng=rng,
                                     exclude_isolated_strength=strength)
                r = simulate_cascade(W, seeds, float(tau))
                sizes.append(r.final_size)
                durs.append(r.duration)
            rows.append(dict(tau=float(tau), k=int(k), strategy="random",
                             final_size=float(np.mean(sizes)),
                             duration=float(np.mean(durs))))
    return pd.DataFrame(rows)


def heatmap(df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    pivot = (df[df["strategy"] == strategy]
             .pivot(index="tau", columns="k", values="final_size")
             .sort_index(ascending=False))   # tau decreasing top->bottom
    fig, ax = plt.subplots(figsize=(8.2, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)),
                  [f"{t:.2f}" for t in pivot.index])
    ax.set_xlabel("seed size  $k$")
    ax.set_ylabel(r"threshold  $\tau$")
    ax.set_title(f"Final cascade size — strategy: {strategy}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="final fraction failed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def critical_threshold_curve(df: pd.DataFrame, out_path: Path,
                              cascade_cutoff: float = 0.5) -> None:
    """
    For each strategy, plot the smallest k for which the cascade exceeds
    `cascade_cutoff` of the network, as a function of tau.  This is the
    systemic-risk frontier in (tau, k_c) coordinates.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, marker in [("high", "o"), ("random", "s"), ("low", "^")]:
        sub = df[df["strategy"] == strategy]
        kc = []
        for tau in TAUS:
            ssub = sub[sub["tau"] == tau].sort_values("k")
            big = ssub[ssub["final_size"] >= cascade_cutoff]
            kc.append(int(big["k"].min()) if not big.empty else np.nan)
        ax.plot(TAUS, kc, marker=marker, label=strategy)
    ax.set_xlabel(r"threshold  $\tau$")
    ax.set_ylabel(rf"critical seed size $k_c$"
                  rf"  (smallest $k$ with $\sigma \geq {cascade_cutoff}$)")
    ax.set_title("Systemic-risk frontier")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="seed strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    W, _ = load_adjacency(OUTPUT_DIR / "adjacency_matrix_thresholded.csv")
    print(f"Sweeping {len(TAUS)} thresholds x {len(KS)} seed sizes "
          f"x 3 strategies (random averaged over {N_RND} runs) ...")
    df = sweep(W)
    df.to_csv(OUTPUT_DIR / "phase_diagram.csv", index=False)

    for strat in ("high", "random", "low"):
        heatmap(df, strat, FIG_DIR / f"phase_{strat}.png")

    critical_threshold_curve(df, FIG_DIR / "critical_frontier.png")

    print("\n=== Final cascade size, strategy = high (worst case) ===")
    pivot = df[df["strategy"] == "high"].pivot(
        index="tau", columns="k", values="final_size",
    ).sort_index(ascending=False)
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Final cascade size, strategy = random (average case) ===")
    pivot = df[df["strategy"] == "random"].pivot(
        index="tau", columns="k", values="final_size",
    ).sort_index(ascending=False)
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Full table:       {OUTPUT_DIR / 'phase_diagram.csv'}")


if __name__ == "__main__":
    main()
