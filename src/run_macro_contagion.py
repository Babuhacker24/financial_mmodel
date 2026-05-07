from __future__ import annotations

from pathlib import Path

import pandas as pd

from contagion import simulate_cascade, cascade_summary
from centrality import eigenvector_centrality, weighted_degree, select_seeds
from experiments import load_adjacency
from macro import (
    compute_market_proxy,
    compute_macro_betas,
    normalise_betas,
    macro_pressure_vector,
)


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    adjacency_path = OUTPUT_DIR / "adjacency_matrix_thresholded.csv"
    returns_path = OUTPUT_DIR / "clean_returns.csv"

    W, tickers = load_adjacency(adjacency_path)

    returns = pd.read_csv(
        returns_path,
        index_col=0,
        parse_dates=True,
    )

    returns = returns.reindex(columns=tickers)

    market_proxy = compute_market_proxy(returns)
    raw_betas = compute_macro_betas(returns, market_proxy)
    macro_betas = normalise_betas(raw_betas)

    raw_betas.to_csv(OUTPUT_DIR / "raw_macro_betas.csv")
    macro_betas.to_csv(OUTPUT_DIR / "normalised_macro_betas.csv")

    eig = eigenvector_centrality(W)
    strength = weighted_degree(W)

    scenarios = [
        ("calm", 0.00),
        ("mild_stress", 0.05),
        ("medium_stress", 0.10),
        ("heavy_stress", 0.20),
        ("crisis_stress", 0.30),
    ]

    taus = [0.20, 0.30, 0.40]
    seed_sizes = [1, 3, 5, 10, 20]

    rows = []

    for tau in taus:
        for k in seed_sizes:
            seeds = select_seeds(
                eig,
                k,
                mode="high",
                exclude_isolated_strength=strength,
            )

            for scenario_name, M in scenarios:
                pressure = macro_pressure_vector(
                    macro_betas=macro_betas,
                    tickers=tickers,
                    M=M,
                )

                result = simulate_cascade(
                    W=W,
                    seeds=seeds,
                    tau=tau,
                    macro_pressure=pressure,
                )

                row = cascade_summary(result)
                row["scenario"] = scenario_name
                row["M"] = M
                row["seed_strategy"] = "high"
                row["n_seeds"] = k

                rows.append(row)

    results = pd.DataFrame(rows)

    preferred_columns = [
        "scenario",
        "M",
        "tau",
        "seed_strategy",
        "n_seeds",
        "final_size",
        "duration",
        "speed",
    ]

    existing_columns = [col for col in preferred_columns if col in results.columns]
    remaining_columns = [col for col in results.columns if col not in existing_columns]
    results = results[existing_columns + remaining_columns]

    results.to_csv(OUTPUT_DIR / "macro_contagion_results.csv", index=False)

    print("Macro-pressure contagion experiments complete.")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
    