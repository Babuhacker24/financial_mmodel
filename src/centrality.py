"""
Centrality measures used to choose initial shock targets.

We expose three families:

    1. Weighted degree (strength)         -- s_i = sum_j w_ij
    2. Eigenvector centrality              -- principal eigenvector of W
    3. Betweenness centrality (unweighted) -- on the thresholded graph

For a financial-network paper the eigenvector centrality is the most
defensible "systemic importance" proxy: it is exactly the right-hand object
attached to the largest eigenvalue lambda_max that already appears in the
network_summary.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- #

def weighted_degree(W: np.ndarray) -> np.ndarray:
    """Strength s_i = sum_j w_ij."""
    return W.sum(axis=1)


def eigenvector_centrality(W: np.ndarray) -> np.ndarray:
    """
    Principal eigenvector of W (the one associated with lambda_max).

    Returned vector is non-negative (Perron–Frobenius) and L2-normalised.
    """
    # numpy eigh is for symmetric matrices and returns eigenvalues in
    # ascending order, so the principal eigenvector is the LAST column.
    eigvals, eigvecs = np.linalg.eigh(W)
    v = eigvecs[:, -1]
    # Sign convention: make the entries non-negative.
    if v.sum() < 0:
        v = -v
    # Numerical floor to kill tiny negative entries from round-off.
    v = np.where(v < 0, 0.0, v)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def betweenness_centrality(W: np.ndarray, weighted: bool = False) -> np.ndarray:
    """
    Betweenness centrality computed via networkx.

    If ``weighted`` is False the graph is binarised (edge present iff w_ij > 0)
    which is what one usually does for the thresholded financial network.
    If True, edge weights w_ij are interpreted as similarities and the shortest
    paths are computed on distances d_ij = 1 - w_ij so that *stronger*
    correlations correspond to *shorter* paths.
    """
    N = W.shape[0]
    if weighted:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        i, j = np.where(np.triu(W, k=1) > 0)
        for a, b in zip(i, j):
            G.add_edge(int(a), int(b), weight=1.0 - float(W[a, b]))
        bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    else:
        A = (W > 0).astype(int)
        G = nx.from_numpy_array(A)
        bc = nx.betweenness_centrality(G, normalized=True)
    return np.array([bc[i] for i in range(N)])


# --------------------------------------------------------------------------- #

def select_seeds(
    centrality: np.ndarray,
    k: int,
    mode: str = "high",
    rng: np.random.Generator | None = None,
    exclude_isolated_strength: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pick k seed nodes.

    Parameters
    ----------
    centrality : (N,) array
        Centrality score for each node (higher = more central).
    k : int
        Number of seeds.
    mode : {"high", "low", "random"}
    rng : numpy random generator (used only for mode="random").
    exclude_isolated_strength : (N,) array, optional
        If given, nodes whose entry is 0 are excluded from "low" and "random"
        seed pools (these nodes can never propagate or receive contagion).
    """
    N = centrality.shape[0]
    candidates = np.arange(N)
    if exclude_isolated_strength is not None:
        candidates = candidates[exclude_isolated_strength > 0]

    if mode == "high":
        order = np.argsort(centrality[candidates])[::-1]
        return candidates[order[:k]]
    if mode == "low":
        order = np.argsort(centrality[candidates])
        return candidates[order[:k]]
    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(candidates, size=k, replace=False)
    raise ValueError(f"Unknown mode: {mode}")
