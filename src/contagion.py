"""
Weighted Linear Threshold Contagion Model
=========================================

Implements a discrete-time, deterministic, monotone cascade on a weighted
undirected network G = (V, E, W).

State:
    s_i(t) in {0, 1}, with 0 = healthy, 1 = failed.

Dynamics (for each healthy node i at time t):
    L_i(t) = sum_j w_ij * s_j(t) / sum_j w_ij             (weighted load)
    s_i(t+1) = 1   if  s_i(t) = 1                          (absorbing)
             = 1   if  s_i(t) = 0  and  L_i(t) >= tau
             = 0   otherwise

Because failures are absorbing, the dynamics converge in at most N steps.

This module is deliberately decoupled from the data: it takes a numpy
adjacency matrix and returns trajectories. Experiments live in experiments.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
#  Result container                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class CascadeResult:
    """Result of a single contagion run."""
    seeds: np.ndarray                    # indices of initially failed nodes
    tau: float                           # threshold used
    state_history: np.ndarray            # shape (T*+1, N), boolean
    final_size: float                    # fraction of failed nodes at T*
    duration: int                        # T* (number of update steps)
    new_failures_per_step: list[int] = field(default_factory=list)

    @property
    def speed(self) -> float:
        """Average new failures per step (excluding the initial seed)."""
        if self.duration == 0:
            return 0.0
        N = self.state_history.shape[1]
        return (self.final_size * N - len(self.seeds)) / self.duration


# --------------------------------------------------------------------------- #
#  Core simulator                                                              #
# --------------------------------------------------------------------------- #

def simulate_cascade(
    W: np.ndarray,
    seeds: Iterable[int],
    tau: float,
    max_steps: int | None = None,
    macro_pressure=None,
) -> CascadeResult:
    """
    Run the weighted linear-threshold cascade until a fixed point is reached.

    Parameters
    ----------
    W : (N, N) ndarray, non-negative, symmetric, zero diagonal
        Weighted adjacency matrix.
    seeds : iterable of int
        Indices of nodes that fail at t = 0.
    tau : float in (0, 1]
        Failure threshold on the weighted load.
    max_steps : int, optional
        Hard cap on iterations (default N, which is sufficient for monotone
        dynamics).

    Returns
    -------
    CascadeResult
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square 2-D array.")
    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("W must be symmetric.")
    if (W < 0).any():
        raise ValueError("W must be non-negative.")
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must be in (0, 1].")

    N = W.shape[0]
    if max_steps is None:
        max_steps = N
    if macro_pressure is None:
        macro_pressure = np.zeros(N, dtype=float)
    else:
        macro_pressure = np.asarray(macro_pressure, dtype=float)
        if macro_pressure.shape != (N,):
            raise ValueError("macro_pressure must be a vector of length N.")

    # Pre-compute node strengths (denominator of L_i).
    strength = W.sum(axis=1)
    # For isolated nodes the load is undefined; we set the denom to 1 to avoid
    # divide-by-zero, knowing that the numerator will also be 0 in that case
    # so isolated healthy nodes never fail through contagion.
    safe_strength = np.where(strength > 0, strength, 1.0)

    seeds = np.asarray(sorted(set(int(i) for i in seeds)), dtype=int)
    if seeds.size == 0:
        raise ValueError("Seed set must be non-empty.")
    if (seeds < 0).any() or (seeds >= N).any():
        raise ValueError("Seed indices out of range.")

    s = np.zeros(N, dtype=bool)
    s[seeds] = True

    history = [s.copy()]
    new_failures_per_step: list[int] = []

    for _ in range(max_steps):
        # Network pressure: weighted fraction of failed neighbours for every node.
        network_load = (W @ s.astype(np.float64)) / safe_strength

        # Total pressure = network pressure + macro pressure.
        load = network_load + macro_pressure

        # Healthy nodes whose total load crosses the threshold this step.
        new_fail = (~s) & (load >= tau)
        n_new = int(new_fail.sum())
        if n_new == 0:
            break
        s = s | new_fail
        history.append(s.copy())
        new_failures_per_step.append(n_new)

    state_history = np.vstack(history)
    final_size = float(s.mean())
    duration = state_history.shape[0] - 1   # number of update steps taken

    return CascadeResult(
        seeds=seeds,
        tau=tau,
        state_history=state_history,
        final_size=final_size,
        duration=duration,
        new_failures_per_step=new_failures_per_step,
    )


# --------------------------------------------------------------------------- #
#  Convenience helpers                                                         #
# --------------------------------------------------------------------------- #

def cascade_summary(result: CascadeResult) -> dict:
    """One-row dict suitable for assembling results into a DataFrame."""
    return {
        "tau": result.tau,
        "n_seeds": len(result.seeds),
        "final_size": result.final_size,
        "duration": result.duration,
        "speed": result.speed,
    }


# --------------------------------------------------------------------------- #
#  Speed / dynamics metrics                                                   #
# --------------------------------------------------------------------------- #

def speed_metrics(result: CascadeResult) -> dict:
    """
    Extract dynamic (time-resolved) descriptors from a finished cascade.

    Definitions
    -----------
    t_half       : smallest t such that the failed fraction exceeds 0.5.
                   None if the cascade never reaches half collapse.
    t_peak       : step at which the largest number of *new* failures occur.
    peak_rate    : that largest number of new failures per step.
    t_double     : smallest t at which the failed fraction is >= 2 x sigma_0
                   (doubling time of the cascade).  None if never reached.
    duration     : T*, total number of update steps.
    avg_rate     : mean new failures per step over the active phase.
    """
    h = result.state_history                 # (T*+1, N) bool
    N = h.shape[1]
    failed_fraction = h.mean(axis=1)         # length T*+1

    half_idx = np.where(failed_fraction >= 0.5)[0]
    t_half = int(half_idx[0]) if half_idx.size > 0 else None

    nfps = result.new_failures_per_step      # length T*
    if nfps:
        peak_rate = int(max(nfps))
        t_peak = int(np.argmax(nfps)) + 1    # +1: nfps[0] describes t=0 -> 1
        avg_rate = float(np.mean(nfps))
    else:
        peak_rate, t_peak, avg_rate = 0, 0, 0.0

    base = failed_fraction[0]
    if base > 0:
        d_idx = np.where(failed_fraction >= 2.0 * base)[0]
        t_double = int(d_idx[0]) if d_idx.size > 0 else None
    else:
        t_double = None

    return {
        "t_half": t_half,
        "t_peak": t_peak,
        "peak_rate": peak_rate,
        "t_double": t_double,
        "duration": result.duration,
        "avg_rate": avg_rate,
    }
