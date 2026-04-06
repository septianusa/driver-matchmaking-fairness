from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "output"

PING_FILE = DATA_RAW_DIR / "ping_dataset_surabaya_14day.csv"
ORDER_FILE = DATA_RAW_DIR / "order_dataset_surabaya_14day.csv"

TOTAL_DAY = 2
TOTAL_BATCH_PER_DAY = 1440


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ping_df = pd.read_csv(PING_FILE)
    order_df = pd.read_csv(ORDER_FILE)

    experiment_groups = sorted(
        p.name.replace("order_dataset_final_", "").replace(".csv", "")
        for p in OUTPUT_DIR.glob("order_dataset_final_*.csv")
    )

    if not experiment_groups:
        raise FileNotFoundError(
            f"No order_dataset_final_*.csv files found in {OUTPUT_DIR}. "
            f"Run python -m src.main first."
        )

    # static supply from raw ping dataset
    driver_per_batch = (
        ping_df.groupby("batchStep", as_index=False)
        .agg(available_driver=("driverId", "nunique"))
    )

    driver_per_batch["day"] = ((driver_per_batch["batchStep"] - 1) // TOTAL_BATCH_PER_DAY) + 1
    driver_per_batch["batch_in_day"] = ((driver_per_batch["batchStep"] - 1) % TOTAL_BATCH_PER_DAY) + 1

    scenario_frames = []

    for experiment_group in experiment_groups:
        order_file = OUTPUT_DIR / f"order_dataset_final_{experiment_group}.csv"
        if not order_file.exists():
            continue

        order_df_exp = pd.read_csv(order_file)

        order_per_batch = (
            order_df_exp.groupby("batchStep", as_index=False)
            .agg(total_order=("orderId", "count"))
        )

        merged = driver_per_batch.merge(order_per_batch, on="batchStep", how="left")
        merged["total_order"] = merged["total_order"].fillna(0)

        merged["ratio_supply_demand"] = merged.apply(
            lambda x: x["available_driver"] / x["total_order"] if x["total_order"] > 0 else 0.0,
            axis=1,
        )

        merged["experiment_group"] = experiment_group
        scenario_frames.append(
            merged[
                [
                    "experiment_group",
                    "batchStep",
                    "day",
                    "batch_in_day",
                    "available_driver",
                    "total_order",
                    "ratio_supply_demand",
                ]
            ]
        )

    if not scenario_frames:
        raise ValueError("No experiment data could be constructed.")

    final_df = pd.concat(scenario_frames, ignore_index=True)
    final_df.to_csv(OUTPUT_DIR / "supply_demand_ratio_by_experiment.csv", index=False)

    # one graph per day, multiline per experiment
    for day in range(1, TOTAL_DAY + 1):
        temp = final_df[final_df["day"] == day].copy()
        if temp.empty:
            continue

        plt.figure(figsize=(12, 5))

        for experiment_group in sorted(temp["experiment_group"].unique()):
            sub = temp[temp["experiment_group"] == experiment_group].sort_values("batch_in_day")
            plt.plot(
                sub["batch_in_day"],
                sub["ratio_supply_demand"],
                label=experiment_group,
            )

        plt.xlabel("Batch Number in Day")
        plt.ylabel("Available Driver / Total Order")
        plt.title(f"Supply–Demand Ratio by Experiment - Day {day}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"plot_supply_demand_ratio_day_{day}.png", dpi=180)
        plt.close()

    print(f"Saved table: {OUTPUT_DIR / 'supply_demand_ratio_by_experiment.csv'}")
    print("Saved plots:")
    for day in range(1, TOTAL_DAY + 1):
        print(OUTPUT_DIR / f"plot_supply_demand_ratio_day_{day}.png")


if __name__ == "__main__":
    main()