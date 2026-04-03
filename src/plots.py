from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


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
        plt.plot(hourly["hour"], hourly["total_orders"], marker="o")
        plt.plot(hourly["hour"], hourly["matched_orders"], marker="o")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        plt.title(f"Orders vs Matches by Hour (lambda={lam})")
        plt.xticks(range(24))
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