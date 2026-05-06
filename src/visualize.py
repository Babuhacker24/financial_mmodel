"""
Plots for the contagion experiments.

    * cascade_trajectories  -- failed-fraction vs. time per (tau, strategy)
    * heatmap_size          -- final cascade size as a function of (tau, k)
    * compare_strategies    -- bar chart of final size by strategy at fixed k
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments import trajectory_for_plot, trajectory_full


def cascade_trajectories(W: np.ndarray, k: int, out_path: Path) -> None:
    """Failed-fraction over time for each (tau, strategy) combination."""
    taus = [0.2, 0.3, 0.5]
    strategies = ["high", "low", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, tau in zip(axes, taus):
        for strat in strategies:
            traj = trajectory_for_plot(W, tau=tau, k=k, strategy=strat)
            ax.plot(np.arange(len(traj)), traj, marker="o", label=strat)
        ax.set_title(rf"$\tau = {tau}$,  $k = {k}$")
        ax.set_xlabel("step $t$")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("failed fraction")
    axes[0].legend(title="seed strategy")
    fig.suptitle("Cascade trajectories (eigenvector-centrality targeting)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap_size(df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    """Final cascade size as a function of (tau, k) for a given strategy."""
    sub = df[df["strategy"] == strategy].pivot(
        index="tau", columns="n_seeds", values="final_size",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(sub.values, aspect="auto", origin="lower",
                   vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(sub.columns)), sub.columns)
    ax.set_yticks(range(len(sub.index)), [f"{t:g}" for t in sub.index])
    ax.set_xlabel("number of seeds $k$")
    ax.set_ylabel(r"threshold $\tau$")
    ax.set_title(f"Final cascade size — strategy: {strategy}")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            ax.text(j, i, f"{sub.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if sub.values[i, j] < 0.6 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="final fraction failed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def trajectory_overlay(
    W: np.ndarray, k: int, taus: list[float], strategy: str,
    out_path: Path,
) -> None:
    """Several trajectories at fixed (k, strategy), one curve per tau."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis
    for i, tau in enumerate(sorted(taus)):
        traj = trajectory_for_plot(W, tau=tau, k=k, strategy=strategy)
        ax.plot(np.arange(len(traj)), traj, marker="o",
                color=cmap(i / max(1, len(taus) - 1)),
                label=rf"$\tau={tau:g}$")
    ax.set_xlabel("step  $t$")
    ax.set_ylabel("failed fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Cascade trajectories — {strategy} targeting, $k={k}$")
    ax.grid(alpha=0.3)
    ax.legend(title="threshold", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def failure_waterfall(
    W: np.ndarray, tau: float, k: int, strategy: str, out_path: Path,
) -> None:
    """Bar chart: number of *new* failures at each step (the cascade pulse)."""
    res = trajectory_full(W, tau=tau, k=k, strategy=strategy)
    nfps = res.new_failures_per_step
    if not nfps:
        # nothing happened; still draw an "empty" plot for visual continuity
        nfps = [0]

    steps = np.arange(1, len(nfps) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.bar(steps, nfps, color="#c83737", alpha=0.85,
            label="new failures")
    ax1.set_xlabel("step  $t$")
    ax1.set_ylabel("new failures at step  $t$", color="#c83737")
    ax1.tick_params(axis="y", labelcolor="#c83737")

    ax2 = ax1.twinx()
    failed_fraction = res.state_history.mean(axis=1)
    ax2.plot(np.arange(len(failed_fraction)), failed_fraction,
             marker="o", color="#1f4e79", label="cumulative")
    ax2.set_ylabel("cumulative failed fraction", color="#1f4e79")
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#1f4e79")

    ax1.set_title(rf"Cascade pulse: $\tau={tau:g}$,  $k={k}$,  "
                  f"strategy={strategy}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap_thalf(df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    """Time to half-collapse t_{1/2} as a function of (tau, k)."""
    pivot = (df[df["strategy"] == strategy]
             .pivot(index="tau", columns="n_seeds", values="t_half")
             .sort_index(ascending=False))
    fig, ax = plt.subplots(figsize=(8.2, 5))
    masked = np.ma.masked_invalid(pivot.values)
    cmap = plt.cm.viridis_r
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)),
                  [f"{t:.2f}" for t in pivot.index])
    ax.set_xlabel("seed size  $k$")
    ax.set_ylabel(r"threshold  $\tau$")
    ax.set_title(f"Time to half-collapse $t_{{1/2}}$ — strategy: {strategy}\n"
                 "(grey = cascade never reaches 50%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="steps to 50% failure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap_peakrate(df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    """Peak number of new failures per step over (tau, k)."""
    pivot = (df[df["strategy"] == strategy]
             .pivot(index="tau", columns="n_seeds", values="peak_rate")
             .sort_index(ascending=False))
    fig, ax = plt.subplots(figsize=(8.2, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="inferno")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)),
                  [f"{t:.2f}" for t in pivot.index])
    ax.set_xlabel("seed size  $k$")
    ax.set_ylabel(r"threshold  $\tau$")
    ax.set_title(f"Peak failure rate — strategy: {strategy}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v < pivot.values.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="max new failures in any single step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_strategies(df: pd.DataFrame, k: int, out_path: Path) -> None:
    """Grouped bar chart: final size by strategy, one cluster per tau."""
    sub = df[df["n_seeds"] == k].pivot(
        index="tau", columns="strategy", values="final_size",
    )[["high", "random", "low"]]
    err = (
        df[(df["n_seeds"] == k) & (df["strategy"] == "random")]
        .set_index("tau")["size_std"]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(sub.index))
    width = 0.27
    for i, strat in enumerate(sub.columns):
        yerr = err.reindex(sub.index).values if strat == "random" else None
        ax.bar(x + (i - 1) * width, sub[strat].values, width=width,
               label=strat, yerr=yerr, capsize=3)
    ax.set_xticks(x, [f"{t:g}" for t in sub.index])
    ax.set_xlabel(r"threshold $\tau$")
    ax.set_ylabel("final cascade size")
    ax.set_title(f"Strategy comparison at $k = {k}$")
    ax.set_ylim(0, 1.02)
    ax.legend(title="seed strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
