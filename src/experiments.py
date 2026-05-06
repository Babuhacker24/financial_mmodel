"""
Experimental harness for the weighted linear-threshold cascade.

Runs a factorial design over:

    * threshold      tau   in {0.2, 0.3, 0.5}
    * seed strategy        in {"high", "low", "random"}
    * seed size      k     in {1, 3, 5, 10}

For "random" seeds we average across many realisations so that the
result is statistical (mean +/- std).

All centrality-based selections use eigenvector centrality, which is the
natural systemic-importance proxy attached to lambda_max(W).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from contagion import simulate_cascade, cascade_summary, speed_metrics
from centrality import (
    eigenvector_centrality,
    weighted_degree,
    select_seeds,
)


# --------------------------------------------------------------------------- #

def load_adjacency(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a square symmetric adjacency CSV and return (W, ticker_list)."""
    df = pd.read_csv(csv_path, index_col=0)
    if list(df.index) != list(df.columns):
        # Different orderings shouldn't happen but force-align just in case.
        df = df.reindex(index=df.columns)
    W = df.to_numpy(dtype=float)
    np.fill_diagonal(W, 0.0)
    # Symmetrise defensively — round-trip CSV writes can introduce 1e-16 noise.
    W = 0.5 * (W + W.T)
    return W, list(df.columns)


# --------------------------------------------------------------------------- #

def run_grid(
    W: np.ndarray,
    taus: Iterable[float] = (0.2, 0.3, 0.5),
    seed_sizes: Iterable[int] = (1, 3, 5, 10),
    n_random: int = 200,
    rng_seed: int = 20260430,
) -> pd.DataFrame:
    """
    Sweep tau x strategy x k and return a tidy DataFrame.

    For "random" we report the mean over ``n_random`` Monte Carlo replicates
    along with the empirical standard deviation. For "high" / "low" the
    selection is deterministic so a single run per cell is enough.
    """
    rng = np.random.default_rng(rng_seed)
    eig = eigenvector_centrality(W)
    strength = weighted_degree(W)

    rows: list[dict] = []
    for tau in taus:
        for k in seed_sizes:
            # Targeted (high eigenvector centrality)
            seeds_hi = select_seeds(eig, k, mode="high",
                                    exclude_isolated_strength=strength)
            r = simulate_cascade(W, seeds_hi, tau)
            rows.append({**cascade_summary(r),
                         "strategy": "high",
                         "size_std": 0.0,
                         "duration_std": 0.0})

            # Anti-targeted (low eigenvector centrality, non-isolated)
            seeds_lo = select_seeds(eig, k, mode="low",
                                    exclude_isolated_strength=strength)
            r = simulate_cascade(W, seeds_lo, tau)
            rows.append({**cascade_summary(r),
                         "strategy": "low",
                         "size_std": 0.0,
                         "duration_std": 0.0})

            # Random — Monte Carlo
            sizes, durs, speeds = [], [], []
            for _ in range(n_random):
                seeds_rnd = select_seeds(
                    eig, k, mode="random", rng=rng,
                    exclude_isolated_strength=strength,
                )
                r = simulate_cascade(W, seeds_rnd, tau)
                sizes.append(r.final_size)
                durs.append(r.duration)
                speeds.append(r.speed)
            rows.append({
                "tau": tau, "n_seeds": k, "strategy": "random",
                "final_size": float(np.mean(sizes)),
                "size_std": float(np.std(sizes, ddof=1)),
                "duration": float(np.mean(durs)),
                "duration_std": float(np.std(durs, ddof=1)),
                "speed": float(np.mean(speeds)),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def trajectory_for_plot(
    W: np.ndarray,
    tau: float,
    k: int,
    strategy: str,
    rng_seed: int = 20260430,
) -> np.ndarray:
    """Return cascade trajectory (failed fraction at each step) for plotting."""
    eig = eigenvector_centrality(W)
    strength = weighted_degree(W)
    rng = np.random.default_rng(rng_seed)
    seeds = select_seeds(eig, k, mode=strategy, rng=rng,
                         exclude_isolated_strength=strength)
    res = simulate_cascade(W, seeds, tau)
    return res.state_history.mean(axis=1)


def trajectory_full(
    W: np.ndarray,
    tau: float,
    k: int,
    strategy: str,
    rng_seed: int = 20260430,
):
    """Return the full CascadeResult (for waterfall / per-step plots)."""
    eig = eigenvector_centrality(W)
    strength = weighted_degree(W)
    rng = np.random.default_rng(rng_seed)
    seeds = select_seeds(eig, k, mode=strategy, rng=rng,
                         exclude_isolated_strength=strength)
    return simulate_cascade(W, seeds, tau)


def run_dynamics_grid(
    W: np.ndarray,
    taus: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
    seed_sizes: Iterable[int] = (1, 3, 5, 10, 15, 20, 30, 50, 75, 100),
    n_random: int = 50,
    rng_seed: int = 20260430,
) -> pd.DataFrame:
    """
    Sweep tau x k x strategy and record the dynamics metrics
    (final size, duration, t_half, t_peak, peak_rate, avg_rate).

    For "random" the per-cell metrics are averaged over Monte Carlo replicates.
    t_half / t_double can be None; we encode them as NaN in the DataFrame so
    that pandas/matplotlib handle them naturally.
    """
    rng = np.random.default_rng(rng_seed)
    eig = eigenvector_centrality(W)
    strength = weighted_degree(W)

    rows: list[dict] = []

    def encode(d: dict) -> dict:
        out = dict(d)
        for key in ("t_half", "t_double"):
            if out.get(key) is None:
                out[key] = np.nan
        return out

    for tau in taus:
        for k in seed_sizes:
            for strat in ("high", "low"):
                seeds = select_seeds(eig, k, mode=strat,
                                     exclude_isolated_strength=strength)
                r = simulate_cascade(W, seeds, float(tau))
                row = {**cascade_summary(r), **encode(speed_metrics(r))}
                row["strategy"] = strat
                rows.append(row)

            # random — Monte Carlo
            agg = {"t_half": [], "t_double": [],
                   "t_peak": [], "peak_rate": [],
                   "avg_rate": [], "duration": [], "final_size": []}
            for _ in range(n_random):
                seeds = select_seeds(eig, k, mode="random", rng=rng,
                                     exclude_isolated_strength=strength)
                r = simulate_cascade(W, seeds, float(tau))
                m = encode(speed_metrics(r))
                agg["t_half"].append(m["t_half"])
                agg["t_double"].append(m["t_double"])
                agg["t_peak"].append(m["t_peak"])
                agg["peak_rate"].append(m["peak_rate"])
                agg["avg_rate"].append(m["avg_rate"])
                agg["duration"].append(r.duration)
                agg["final_size"].append(r.final_size)
            rows.append({
                "tau": float(tau), "n_seeds": int(k), "strategy": "random",
                "final_size": float(np.nanmean(agg["final_size"])),
                "duration": float(np.nanmean(agg["duration"])),
                "speed": float(np.nanmean(agg["avg_rate"])),
                "t_half": float(np.nanmean(agg["t_half"])),
                "t_peak": float(np.nanmean(agg["t_peak"])),
                "peak_rate": float(np.nanmean(agg["peak_rate"])),
                "t_double": float(np.nanmean(agg["t_double"])),
                "avg_rate": float(np.nanmean(agg["avg_rate"])),
            })

    return pd.DataFrame(rows)
