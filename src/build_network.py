from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

RETURNS_FILE = DATA_DIR / "log_returns_wide.csv"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    returns = pd.read_csv(RETURNS_FILE)

    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.set_index("date").sort_index()

    returns = returns.apply(pd.to_numeric, errors="coerce")

    print("Initial returns matrix:")
    print(returns.head())
    print("Shape before cleaning:", returns.shape)

    # Remove banks with too much missing data
    min_observations = int(0.7 * len(returns))
    returns_clean = returns.dropna(axis=1, thresh=min_observations)

    # Remove columns with zero variance
    std = returns_clean.std(skipna=True)
    returns_clean = returns_clean.loc[:, std > 1e-8]

    print("Shape after cleaning:", returns_clean.shape)

    # Correlation matrix
    corr_matrix = returns_clean.corr(min_periods=80)

    # Adjacency matrix: absolute correlations
    adjacency_matrix = corr_matrix.abs()

    # Remove self-links
    np.fill_diagonal(adjacency_matrix.values, 0)

    # Thresholded adjacency matrix for cleaner network visualisation
    threshold = 0.4
    adjacency_thresholded = adjacency_matrix.where(adjacency_matrix >= threshold, 0)
    np.fill_diagonal(adjacency_thresholded.values, 0)

    # Largest eigenvalue
    A = adjacency_matrix.fillna(0).values
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = max(eigenvalues.real)

    # Save outputs
    returns_clean.to_csv(OUTPUT_DIR / "clean_returns.csv")
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    adjacency_matrix.to_csv(OUTPUT_DIR / "adjacency_matrix_weighted.csv")
    adjacency_thresholded.to_csv(OUTPUT_DIR / "adjacency_matrix_thresholded.csv")

    summary = pd.DataFrame({
        "metric": [
            "dates_used",
            "banks_original",
            "banks_after_cleaning",
            "threshold_used",
            "largest_eigenvalue"
        ],
        "value": [
            len(returns),
            returns.shape[1],
            returns_clean.shape[1],
            threshold,
            lambda_max
        ]
    })

    summary.to_csv(OUTPUT_DIR / "network_summary.csv", index=False)

    print("Correlation and adjacency matrices created.")
    print(f"Largest eigenvalue: {lambda_max:.4f}")
    print(f"Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
