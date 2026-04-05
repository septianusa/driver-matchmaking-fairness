from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "output"

COMBINE_EXPERIMENT_FILE = OUTPUT_DIR / "combine_experiment.csv"
INCOME_GROUP_COMPARISON_FILE = OUTPUT_DIR / "income_group_comparison.csv"
INCOME_GROUP_DISPARITY_FILE = OUTPUT_DIR / "income_group_disparity.csv"
DRIVER_GROUP_OPERATIONAL_FILE = OUTPUT_DIR / "driver_group_operational_metrics.csv"
COVERAGE_RATE_FILE = OUTPUT_DIR / "coverage_rate_by_experiment.csv"

TOTAL_BATCH = 14 * 1440


def extract_experiment_group(filename: str) -> str:
    return filename.replace("matchdataset_", "").replace(".csv", "")


def gini(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return np.nan

    if np.min(arr) < 0:
        arr = arr - np.min(arr)

    if np.sum(arr) == 0:
        return 0.0

    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)

    return float(np.sum((2 * idx - n - 1) * arr) / (n * np.sum(arr)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    match_files = sorted(OUTPUT_DIR.glob("matchdataset_*.csv"))
    order_files = sorted(OUTPUT_DIR.glob("order_dataset_final_*.csv"))
    perf_file = DATA_RAW_DIR / "driverPerformance_dataset_surabaya_14day.csv"

    if not match_files:
        raise FileNotFoundError(f"No matchdataset files found in: {OUTPUT_DIR}")

    if not order_files:
        raise FileNotFoundError(f"No order_dataset_final files found in: {OUTPUT_DIR}")

    if not perf_file.exists():
        raise FileNotFoundError(f"Driver performance file not found: {perf_file}")

    perf_df = pd.read_csv(perf_file)

    required_perf_cols = ["driverId", "driverScore"]
    missing_perf_cols = [c for c in required_perf_cols if c not in perf_df.columns]
    if missing_perf_cols:
        raise ValueError(f"Driver performance file missing columns: {missing_perf_cols}")

    if "driverQualityGroup" not in perf_df.columns:
        perf_df["driverQualityGroup"] = np.where(
            perf_df["driverScore"] >= 0.8,
            "HQD",
            np.where(perf_df["driverScore"] >= 0.6, "MQD", "LQD"),
        )

    all_results = []
    income_group_comparison_list = []
    income_group_disparity_list = []
    driver_group_operational_list = []
    coverage_list = []

    for file_path in match_files:
        print(f"Processing: {file_path.name}")

        df = pd.read_csv(file_path)
        experiment_group = extract_experiment_group(file_path.name)

        required_match_cols = [
            "batchStep",
            "orderId",
            "driverId",
            "fare",
            "cancelProbability",
            "distance_ji_km",
            "estimatedTimeArrivalInBatch",
            "h3Origin",
        ]
        missing_match_cols = [c for c in required_match_cols if c not in df.columns]
        if missing_match_cols:
            raise ValueError(f"{file_path.name} missing columns: {missing_match_cols}")

        order_file = OUTPUT_DIR / f"order_dataset_final_{experiment_group}.csv"
        if not order_file.exists():
            raise FileNotFoundError(f"Required file not found: {order_file}")

        order_df = pd.read_csv(order_file)

        required_order_cols = ["orderId", "batchStep", "h3Origin"]
        missing_order_cols = [c for c in required_order_cols if c not in order_df.columns]
        if missing_order_cols:
            raise ValueError(f"{order_file.name} missing columns: {missing_order_cols}")

        # =================================================
        # 1. combine_experiment.csv
        # =================================================
        df["experiment_group"] = experiment_group
        df["day"] = ((df["batchStep"] - 1) // 1440) + 1
        df["hour"] = ((df["batchStep"] - 1) % 1440) // 60
        df["economic_utility"] = (1 - df["cancelProbability"]) * df["fare"]

        agg = (
            df.groupby(["experiment_group", "day", "hour"], as_index=False)
            .agg(
                sum_economic_utility=("economic_utility", "sum"),
                sum_distance_ji_km=("distance_ji_km", "sum"),
                total_order=("orderId", "count"),
            )
        )
        all_results.append(agg)

        # =================================================
        # 2. income comparison + utilization base
        # =================================================
        income_by_driver = (
            df.groupby("driverId", as_index=False)
            .agg(
                total_income=("fare", "sum"),
                total_batch_on_booking=("estimatedTimeArrivalInBatch", "sum"),
                total_booking=("orderId", "count"),
            )
        )

        income_by_driver["is_active_driver"] = (income_by_driver["total_booking"] > 0).astype(int)

        income_by_driver = perf_df[["driverId", "driverScore", "driverQualityGroup"]].merge(
            income_by_driver,
            on="driverId",
            how="left",
        )

        income_by_driver["total_income"] = income_by_driver["total_income"].fillna(0.0)
        income_by_driver["total_batch_on_booking"] = income_by_driver["total_batch_on_booking"].fillna(0.0)
        income_by_driver["total_booking"] = income_by_driver["total_booking"].fillna(0).astype(int)
        income_by_driver["is_active_driver"] = income_by_driver["is_active_driver"].fillna(0).astype(int)
        income_by_driver["experiment_group"] = experiment_group
        income_by_driver["utilization_rate"] = income_by_driver["total_batch_on_booking"] / TOTAL_BATCH

        comparison = (
            income_by_driver.groupby(
                ["experiment_group", "driverQualityGroup"],
                as_index=False
            )
            .agg(
                total_driver=("driverId", "count"),
                total_income=("total_income", "sum"),
                avg_income=("total_income", "mean"),
                median_income=("total_income", "median"),
            )
        )
        income_group_comparison_list.append(comparison)

        for grp, sub in income_by_driver.groupby("driverQualityGroup"):
            incomes = sub["total_income"].values
            p10 = np.percentile(incomes, 10)
            p50 = np.percentile(incomes, 50)
            p90 = np.percentile(incomes, 90)

            income_group_disparity_list.append(
                {
                    "experiment_group": experiment_group,
                    "driverQualityGroup": grp,
                    "total_driver": int(len(sub)),
                    "mean_income": float(np.mean(incomes)),
                    "median_income": float(np.median(incomes)),
                    "std_income": float(np.std(incomes)),
                    "min_income": float(np.min(incomes)),
                    "p10_income": float(p10),
                    "p50_income": float(p50),
                    "p90_income": float(p90),
                    "max_income": float(np.max(incomes)),
                    "p90_p10_ratio": float(p90 / p10) if p10 > 0 else np.nan,
                    "gini_income": gini(incomes),
                }
            )

        # =================================================
        # 3. success rate got next booking after dropoff
        # =================================================
        trip_df = df[["driverId", "batchStep", "estimatedTimeArrivalInBatch", "orderId"]].copy()
        trip_df = trip_df.sort_values(["driverId", "batchStep"]).reset_index(drop=True)
        trip_df["dropoff_batch"] = trip_df["batchStep"] + trip_df["estimatedTimeArrivalInBatch"]
        trip_df["next_batch_step"] = trip_df.groupby("driverId")["batchStep"].shift(-1)
        trip_df["success_next_booking_after_dropoff"] = (
            trip_df["next_batch_step"].notna()
            & (trip_df["next_batch_step"] > trip_df["dropoff_batch"])
        ).astype(int)

        trip_df = trip_df.merge(
            perf_df[["driverId", "driverQualityGroup"]],
            on="driverId",
            how="left",
        )

        success_by_group = (
            trip_df.groupby("driverQualityGroup", as_index=False)
            .agg(
                total_completed_booking=("orderId", "count"),
                successful_next_booking=("success_next_booking_after_dropoff", "sum"),
            )
        )
        success_by_group["success_rate_next_booking_after_dropoff"] = (
            success_by_group["successful_next_booking"] / success_by_group["total_completed_booking"]
        )

        # =================================================
        # 4. utilization by driver group
        # =================================================
        util_by_group = (
            income_by_driver.groupby("driverQualityGroup", as_index=False)
            .agg(
                total_driver=("driverId", "count"),
                active_driver_with_match=("is_active_driver", "sum"),
                total_batch_on_booking=("total_batch_on_booking", "sum"),
                avg_batch_on_booking_per_driver=("total_batch_on_booking", "mean"),
                avg_utilization_rate=("utilization_rate", "mean"),
                median_utilization_rate=("utilization_rate", "median"),
            )
        )

        util_by_group["group_utilization_rate"] = (
            util_by_group["total_batch_on_booking"] / (util_by_group["total_driver"] * TOTAL_BATCH)
        )

        util_by_group["active_driver_ratio"] = (
            util_by_group["active_driver_with_match"] / util_by_group["total_driver"]
        )

        operational = util_by_group.merge(
            success_by_group,
            on="driverQualityGroup",
            how="left",
        )
        operational["experiment_group"] = experiment_group

        driver_group_operational_list.append(
            operational[
                [
                    "experiment_group",
                    "driverQualityGroup",
                    "total_driver",
                    "active_driver_with_match",
                    "active_driver_ratio",
                    "total_batch_on_booking",
                    "avg_batch_on_booking_per_driver",
                    "avg_utilization_rate",
                    "median_utilization_rate",
                    "group_utilization_rate",
                    "total_completed_booking",
                    "successful_next_booking",
                    "success_rate_next_booking_after_dropoff",
                ]
            ]
        )

        # =================================================
        # 5. coverage rate by experiment
        # =================================================
        order_h3 = (
            order_df.groupby("h3Origin", as_index=False)
            .agg(total_order=("orderId", "count"))
        )

        match_h3 = (
            df.groupby("h3Origin", as_index=False)
            .agg(matched_order=("orderId", "count"))
        )

        h3_perf = order_h3.merge(match_h3, on="h3Origin", how="left")
        h3_perf["matched_order"] = h3_perf["matched_order"].fillna(0)
        h3_perf["conversion_rate"] = h3_perf["matched_order"] / h3_perf["total_order"]

        total_h3 = h3_perf["h3Origin"].nunique()
        high_perf_h3 = h3_perf.loc[
            h3_perf["conversion_rate"] >= 0.8,
            "h3Origin"
        ].nunique()

        coverage_rate = high_perf_h3 / total_h3 if total_h3 > 0 else np.nan

        coverage_list.append(
            {
                "experiment_group": experiment_group,
                "total_h3": total_h3,
                "high_performance_h3": high_perf_h3,
                "coverage_rate": coverage_rate,
            }
        )

    # =====================================================
    # Save outputs
    # =====================================================
    combine_experiment = (
        pd.concat(all_results, ignore_index=True)
        .sort_values(["experiment_group", "day", "hour"])
        .reset_index(drop=True)
    )
    combine_experiment.to_csv(COMBINE_EXPERIMENT_FILE, index=False)

    income_group_comparison = (
        pd.concat(income_group_comparison_list, ignore_index=True)
        .sort_values(["experiment_group", "driverQualityGroup"])
        .reset_index(drop=True)
    )
    income_group_comparison.to_csv(INCOME_GROUP_COMPARISON_FILE, index=False)

    income_group_disparity = (
        pd.DataFrame(income_group_disparity_list)
        .sort_values(["experiment_group", "driverQualityGroup"])
        .reset_index(drop=True)
    )
    income_group_disparity.to_csv(INCOME_GROUP_DISPARITY_FILE, index=False)

    driver_group_operational = (
        pd.concat(driver_group_operational_list, ignore_index=True)
        .sort_values(["experiment_group", "driverQualityGroup"])
        .reset_index(drop=True)
    )
    driver_group_operational.to_csv(DRIVER_GROUP_OPERATIONAL_FILE, index=False)

    coverage_df = (
        pd.DataFrame(coverage_list)
        .sort_values("experiment_group")
        .reset_index(drop=True)
    )
    coverage_df.to_csv(COVERAGE_RATE_FILE, index=False)

    print(f"\nSaved: {COMBINE_EXPERIMENT_FILE}")
    print(f"Saved: {INCOME_GROUP_COMPARISON_FILE}")
    print(f"Saved: {INCOME_GROUP_DISPARITY_FILE}")
    print(f"Saved: {DRIVER_GROUP_OPERATIONAL_FILE}")
    print(f"Saved: {COVERAGE_RATE_FILE}")

    print("\nSample driver group operational metrics:")
    print(driver_group_operational.head(12).to_string(index=False))

    print("\nCoverage rate by experiment:")
    print(coverage_df.to_string(index=False))


if __name__ == "__main__":
    main()