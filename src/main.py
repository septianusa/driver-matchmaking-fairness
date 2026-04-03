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
from .plots import make_summary_plots, plot_driver_start_distribution
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

    for use_grid in GRID_SCENARIOS:
        for lam in LAMBDA_SCENARIOS:
            print(f"Running scenario: use_grid={use_grid}, lambda={lam} ...")

            result = run_replay_simulation(
                order_df=order_df,
                ping_df=ping_df,
                driver_perf_df=driver_perf_df,
                grid_df=grid_df,
                lambda_driver_score=lam,
                use_grid=use_grid,
            )

            grid_flag = "on" if use_grid else "off"
            lam_suffix = str(lam).replace(".", "_")

            # unique key for plotting later
            scenario_key = f"grid_{grid_flag}_lambda_{lam_suffix}"
            scenario_outputs[scenario_key] = result

            # add metadata into summary row
            summary = result["summary"].copy()
            summary["use_grid"] = use_grid
            summary_rows.append(summary)

            # save outputs with distinct filenames
            result["matchdataset"].to_csv(
                DATA_OUTPUT_DIR / f"matchdataset_grid_{grid_flag}_lambda_{lam_suffix}.csv",
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

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["use_grid", "lambda_driver_score"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(DATA_OUTPUT_DIR / "simulation_summary_across_lambda_and_grid.csv", index=False)

    make_summary_plots(summary_df, scenario_outputs, DATA_OUTPUT_DIR)

    print("\nFinished.")
    print("\nDriver start distribution:")
    print(driver_start_dist.to_string(index=False))
    print("\nSimulation summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()