import pandas as pd

from .config import (
    ORDER_FILE,
    PING_FILE,
    DRIVER_PERF_FILE,
    GRID_FILE,
    DATA_OUTPUT_DIR,
    LAMBDA_SCENARIOS,
    GRID_SCENARIOS,
)
from .simulator import run_replay_simulation
from .plots import (
    make_summary_plots,
    plot_driver_start_distribution,
    plot_daily_metric_comparison,
)
from .metrics import get_driver_start_distribution


def main():
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    order_df = pd.read_csv(ORDER_FILE)
    ping_df = pd.read_csv(PING_FILE)
    driver_perf_df = pd.read_csv(DRIVER_PERF_FILE)
    grid_df = pd.read_csv(GRID_FILE)

    # driver start distribution from ping_dataset
    driver_start_dist = get_driver_start_distribution(ping_df)
    driver_start_dist.to_csv(DATA_OUTPUT_DIR / "driver_start_distribution.csv", index=False)
    plot_driver_start_distribution(driver_start_dist, DATA_OUTPUT_DIR)

    scenario_outputs = {}
    summary_rows = []

    daily_utility_outputs = {}
    daily_conversion_outputs = {}
    daily_pickup_outputs = {}
    daily_corr_outputs = {}

    total_scenarios = len(GRID_SCENARIOS) * len(LAMBDA_SCENARIOS)
    scenario_idx = 0

    for use_grid in GRID_SCENARIOS:
        for lam in LAMBDA_SCENARIOS:
            scenario_idx += 1
            grid_flag = "on" if use_grid else "off"
            lam_suffix = str(lam).replace(".", "_")
            scenario_key = f"grid_{grid_flag}_lambda_{lam_suffix}"

            print(f"\n[{scenario_idx}/{total_scenarios}] Running scenario: grid={grid_flag}, lambda={lam}")

            result = run_replay_simulation(
                order_df=order_df,
                ping_df=ping_df,
                driver_perf_df=driver_perf_df,
                grid_df=grid_df,
                lambda_driver_score=lam,
                use_grid=use_grid,
            )

            scenario_outputs[scenario_key] = result

            summary = result["summary"].copy()
            summary["use_grid"] = use_grid
            summary_rows.append(summary)

            result["matchdataset"].to_csv(
                DATA_OUTPUT_DIR / f"matchdataset_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )
            result["order_dataset_final"].to_csv(
                DATA_OUTPUT_DIR / f"order_dataset_final_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )
            result["income_by_driver"].to_csv(
                DATA_OUTPUT_DIR / f"income_by_driver_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )
            result["hourly_summary"].to_csv(
                DATA_OUTPUT_DIR / f"hourly_summary_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )

            result["daily_utility"].to_csv(
                DATA_OUTPUT_DIR / f"daily_utility_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )
            result["daily_conversion"].to_csv(
                DATA_OUTPUT_DIR / f"daily_conversion_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )
            result["daily_pickup"].to_csv(
                DATA_OUTPUT_DIR / f"daily_pickup_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )
            result["daily_corr_income_score"].to_csv(
                DATA_OUTPUT_DIR / f"daily_corr_income_score_grid_{grid_flag}_lambda_{lam_suffix}.csv",
                index=False,
            )

            daily_utility_outputs[scenario_key] = result["daily_utility"]
            daily_conversion_outputs[scenario_key] = result["daily_conversion"]
            daily_pickup_outputs[scenario_key] = result["daily_pickup"]
            daily_corr_outputs[scenario_key] = result["daily_corr_income_score"]

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["use_grid", "lambda_driver_score"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(DATA_OUTPUT_DIR / "simulation_summary_across_lambda_and_grid.csv", index=False)

    make_summary_plots(summary_df, scenario_outputs, DATA_OUTPUT_DIR)

    plot_daily_metric_comparison(
        daily_utility_outputs,
        metric_col="total_utility",
        ylabel="Total Utility",
        title="Daily Utility Comparison",
        output_path=DATA_OUTPUT_DIR / "plot_daily_utility_comparison.png",
    )

    plot_daily_metric_comparison(
        daily_conversion_outputs,
        metric_col="conversion_rate",
        ylabel="Conversion Rate",
        title="Daily Conversion Rate Comparison",
        output_path=DATA_OUTPUT_DIR / "plot_daily_conversion_comparison.png",
    )

    plot_daily_metric_comparison(
        daily_pickup_outputs,
        metric_col="avg_pickup_distance",
        ylabel="Average Pickup Distance (km)",
        title="Daily Pickup Distance Comparison",
        output_path=DATA_OUTPUT_DIR / "plot_daily_pickup_comparison.png",
    )

    plot_daily_metric_comparison(
        daily_corr_outputs,
        metric_col="corr_income_score",
        ylabel="Correlation (Income vs Score)",
        title="Daily Correlation Between Income and Driver Score",
        output_path=DATA_OUTPUT_DIR / "plot_daily_corr_income_score_comparison.png",
    )

    print("\nFinished.")
    print("\nDriver start distribution:")
    print(driver_start_dist.to_string(index=False))
    print("\nSimulation summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()