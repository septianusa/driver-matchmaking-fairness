import pandas as pd

from .config import (
    ORDER_FILE,
    PING_FILE,
    DRIVER_PERF_FILE,
    GRID_FILE,
    DATA_OUTPUT_DIR,
    LAMBDA_SCENARIOS,
)
from .simulator import run_replay_simulation
from .plots import make_summary_plots


def main():
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    order_df = pd.read_csv(ORDER_FILE)
    ping_df = pd.read_csv(PING_FILE)
    driver_perf_df = pd.read_csv(DRIVER_PERF_FILE)
    grid_df = pd.read_csv(GRID_FILE)

    scenario_outputs = {}
    summary_rows = []

    for lam in LAMBDA_SCENARIOS:
        print(f"Running scenario lambda={lam} ...")

        result = run_replay_simulation(
            order_df=order_df,
            ping_df=ping_df,
            driver_perf_df=driver_perf_df,
            grid_df=grid_df,
            lambda_driver_score=lam,
        )

        scenario_outputs[lam] = result
        summary_rows.append(result["summary"])

        lam_suffix = str(lam).replace(".", "_")
        result["matchdataset"].to_csv(DATA_OUTPUT_DIR / f"matchdataset_lambda_{lam_suffix}.csv", index=False)
        result["income_by_driver"].to_csv(DATA_OUTPUT_DIR / f"income_by_driver_lambda_{lam_suffix}.csv", index=False)
        result["hourly_summary"].to_csv(DATA_OUTPUT_DIR / f"hourly_summary_lambda_{lam_suffix}.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("lambda_driver_score").reset_index(drop=True)
    summary_df.to_csv(DATA_OUTPUT_DIR / "simulation_summary_across_lambda.csv", index=False)

    make_summary_plots(summary_df, scenario_outputs, DATA_OUTPUT_DIR)

    print("\nFinished.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()