from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_daily_metric_comparison(daily_metric_dict, metric_col, ylabel, title, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.2))

    for scenario_name, df in daily_metric_dict.items():
        if df.empty:
            continue

        use_grid = "grid_on" in scenario_name
        linestyle = "-" if use_grid else "--"

        plt.plot(
            df["simulation_day"],
            df[metric_col],
            marker="o",
            linestyle=linestyle,
            label=scenario_name.replace("_", " ")
        )

    plt.xlabel("Day")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(1, 15))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
def make_summary_plots(summary_df: pd.DataFrame, scenario_outputs: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["lambda_driver_score"], summary_df["match_rate"], marker="o")
    plt.xlabel("Driver score weight (lambda)")
    plt.ylabel("Match rate")
    plt.title("Efficiency: Match Rate by Lambda")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_match_rate_by_lambda.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["lambda_driver_score"], summary_df["gini_income"], marker="o")
    plt.xlabel("Driver score weight (lambda)")
    plt.ylabel("Gini income")
    plt.title("Fairness: Gini of Driver Income by Lambda")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_gini_by_lambda.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["lambda_driver_score"], summary_df["avg_pickup_distance_km"], marker="o")
    plt.xlabel("Driver score weight (lambda)")
    plt.ylabel("Average pickup distance (km)")
    plt.title("Efficiency: Avg Pickup Distance by Lambda")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_pickup_distance_by_lambda.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["lambda_driver_score"], summary_df["corr_driver_score_income"], marker="o")
    plt.xlabel("Driver score weight (lambda)")
    plt.ylabel("Correlation")
    plt.title("Fairness: Correlation Between Driver Score and Income")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_score_income_corr_by_lambda.png", dpi=180)
    plt.close()

    for lam, result in scenario_outputs.items():
        hourly = result["hourly_summary"].copy()
        lam_suffix = str(lam).replace(".", "_")

        plt.figure(figsize=(9, 4.8))
        plt.plot(hourly["hour"], hourly["total_orders"], marker="o", label="total orders")
        plt.plot(hourly["hour"], hourly["matched_orders"], marker="o", label="matched orders")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        plt.title(f"Orders vs Matches by Hour (lambda={lam})")
        plt.xticks(range(24))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"plot_hourly_orders_vs_matches_lambda_{lam_suffix}.png", dpi=180)
        plt.close()

        income = result["income_by_driver"]["total_income"].values
        plt.figure(figsize=(8, 4.8))
        plt.hist(income, bins=40)
        plt.xlabel("Driver income")
        plt.ylabel("Number of drivers")
        plt.title(f"Driver Income Distribution (lambda={lam})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"plot_income_distribution_lambda_{lam_suffix}.png", dpi=180)
        plt.close()


def plot_driver_start_distribution(driver_start_dist: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    plt.plot(driver_start_dist["hour"], driver_start_dist["total_driver_start"], marker="o")
    plt.xlabel("Hour")
    plt.ylabel("Number of drivers starting online")
    plt.title("Driver Start Online Distribution by Hour")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_driver_start_distribution.png", dpi=180)
    plt.close()