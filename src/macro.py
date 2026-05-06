from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_proxy(returns: pd.DataFrame) -> pd.Series:
    """
    Computes the average bank return on each date.

    This is used as a simple proxy for the common banking-sector factor.
    It is not a pure macroeconomic variable, but it captures broad
    banking-market stress using the data already available.
    """
    return returns.mean(axis=1, skipna=True)


def compute_macro_betas(
    returns: pd.DataFrame,
    market_proxy: pd.Series,
    min_obs: int = 60,
) -> pd.Series:
    """
    Estimate each bank's beta to the banking-market proxy.

    For each bank i:

        r_i(t) = alpha_i + beta_i * r_m(t) + error_i(t)

    beta_i is calculated as:

        beta_i = Cov(r_i, r_m) / Var(r_m)
    """
    betas = {}

    for ticker in returns.columns:
        data = pd.concat([returns[ticker], market_proxy], axis=1).dropna()
        data.columns = ["bank_return", "market_return"]

        if len(data) < min_obs:
            betas[ticker] = np.nan
            continue

        market_variance = data["market_return"].var()

        if market_variance == 0 or pd.isna(market_variance):
            betas[ticker] = np.nan
            continue

        covariance = data["bank_return"].cov(data["market_return"])
        betas[ticker] = covariance / market_variance

    return pd.Series(betas, name="raw_beta")


def normalise_betas(betas: pd.Series) -> pd.Series:
    """
    Normalise betas so that the average non-missing beta equals 1.

    This makes interpretation easier:
    - beta > 1 means more macro-sensitive than average
    - beta < 1 means less macro-sensitive than average
    """
    clean_betas = betas.dropna()
    mean_beta = clean_betas.mean()

    if mean_beta == 0 or pd.isna(mean_beta):
        raise ValueError("Cannot normalise betas because the mean beta is zero or missing.")

    normalised = betas / mean_beta
    normalised.name = "macro_beta"

    return normalised


def macro_pressure_vector(
    macro_betas: pd.Series,
    tickers: list[str],
    M: float,
) -> np.ndarray:
    """
    Builds the macro pressure vector beta_i * M.

    M is the macro stress scenario:
    - M = 0.00 means calm
    - M = 0.10 means mild stress
    - M = 0.20 means heavy stress
    - M = 0.30 means crisis stress
    """
    aligned_betas = macro_betas.reindex(tickers).fillna(1.0)
    return aligned_betas.to_numpy(dtype=float) * M
