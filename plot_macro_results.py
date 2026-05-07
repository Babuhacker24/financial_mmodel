from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

if CURRENT_FILE.parent.name == "src":
    BASE_DIR = CURRENT_FILE.parents[1]
else:
    BASE_DIR = CURRENT_FILE.parent

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_FILE = OUTPUT_DIR / "macro_contagion_results.csv"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_FILE)

    # Sort nicely
    df = df.sort_values(["tau", "n_seeds", "M"]).reset_index(drop=True)

    # Convert final size to percentage for nicer plotting
    df["final_size_pct"] = 100 * df["final_size"]

    return df


def setup_figure():
    plt.figure(figsize=(9, 6))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Figure 1: headline chart
# ---------------------------------------------------------------------
def plot_headline_final_size(df: pd.DataFrame, seed_size: int = 1):
    subset = df[df["n_seeds"] == seed_size]

    setup_figure()

    for tau in sorted(subset["tau"].unique()):
        data = subset[subset["tau"] == tau].sort_values("M")

        plt.plot(
            data["M"],
            data["final_size_pct"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=rf"$\tau = {tau:.1f}$"
        )

        # annotate points
        for _, row in data.iterrows():
            plt.annotate(
                f"{row['final_size_pct']:.1f}%",
                (row["M"], row["final_size_pct"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8
            )

    plt.xlabel("Macro stress level $M$")
    plt.ylabel("Final cascade size (%)")
    plt.title(f"Final cascade size vs macro stress ($n_{{seeds}} = {seed_size}$)")
    plt.ylim(-2, 105)
    plt.legend()
    plt.tight_layout()

    out = FIGURES_DIR / f"headline_final_size_seed_{seed_size}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Figure 2: final size heatmaps
# ---------------------------------------------------------------------
def plot_heatmaps_final_size(df: pd.DataFrame):
    for tau in sorted(df["tau"].unique()):
        subset = df[df["tau"] == tau]

        pivot = subset.pivot(
            index="n_seeds",
            columns="M",
            values="final_size_pct"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("Macro stress level $M$")
        ax.set_ylabel("Number of initial seeds")
        ax.set_title(rf"Final cascade size heatmap ($\tau = {tau:.1f}$)")

        # annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=8
                )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Final cascade size (%)")

        plt.tight_layout()
        out = FIGURES_DIR / f"heatmap_final_size_tau_{tau:.1f}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------
# Figure 3: duration vs macro stress
# ---------------------------------------------------------------------
def plot_duration_by_seed(df: pd.DataFrame):
    for seed_size in sorted(df["n_seeds"].unique()):
        subset = df[df["n_seeds"] == seed_size]

        setup_figure()

        for tau in sorted(subset["tau"].unique()):
            data = subset[subset["tau"] == tau].sort_values("M")

            plt.plot(
                data["M"],
                data["duration"],
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=rf"$\tau = {tau:.1f}$"
            )

        plt.xlabel("Macro stress level $M$")
        plt.ylabel("Cascade duration (steps)")
        plt.title(f"Cascade duration vs macro stress ($n_{{seeds}} = {seed_size}$)")
        plt.legend()
        plt.tight_layout()

        out = FIGURES_DIR / f"duration_seed_{seed_size}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------
# Figure 4: speed vs macro stress
# ---------------------------------------------------------------------
def plot_speed_by_seed(df: pd.DataFrame):
    for seed_size in sorted(df["n_seeds"].unique()):
        subset = df[df["n_seeds"] == seed_size]

        setup_figure()

        for tau in sorted(subset["tau"].unique()):
            data = subset[subset["tau"] == tau].sort_values("M")

            plt.plot(
                data["M"],
                data["speed"],
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=rf"$\tau = {tau:.1f}$"
            )

        plt.xlabel("Macro stress level $M$")
        plt.ylabel("Average new failures per step")
        plt.title(f"Cascade speed vs macro stress ($n_{{seeds}} = {seed_size}$)")
        plt.legend()
        plt.tight_layout()

        out = FIGURES_DIR / f"speed_seed_{seed_size}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------
# Figure 5: critical macro stress
# ---------------------------------------------------------------------
def plot_critical_macro_stress(df: pd.DataFrame, collapse_threshold_pct: float = 50.0):
    rows = []

    for tau in sorted(df["tau"].unique()):
        for k in sorted(df["n_seeds"].unique()):
            subset = df[(df["tau"] == tau) & (df["n_seeds"] == k)].sort_values("M")

            triggered = subset[subset["final_size_pct"] >= collapse_threshold_pct]

            if len(triggered) == 0:
                critical_M = np.nan
            else:
                critical_M = triggered["M"].iloc[0]

            rows.append({
                "tau": tau,
                "n_seeds": k,
                "critical_M": critical_M
            })

    crit = pd.DataFrame(rows)

    setup_figure()

    for tau in sorted(crit["tau"].unique()):
        data = crit[crit["tau"] == tau].sort_values("n_seeds")

        plt.plot(
            data["n_seeds"],
            data["critical_M"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=rf"$\tau = {tau:.1f}$"
        )

    plt.xlabel("Number of initial seeds")
    plt.ylabel(r"Critical macro stress $M_c$")
    plt.title(r"Minimum macro stress required for a large cascade (final size $\geq 50\%$)")
    plt.legend()
    plt.tight_layout()

    out = FIGURES_DIR / "critical_macro_stress.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    # Main headline figure
    plot_headline_final_size(df, seed_size=1)

    # Heatmaps
    plot_heatmaps_final_size(df)

    # Duration and speed charts
    plot_duration_by_seed(df)
    plot_speed_by_seed(df)

    # Critical stress summary chart
    plot_critical_macro_stress(df, collapse_threshold_pct=50.0)

    print(f"All figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()