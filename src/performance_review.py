from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "output"

COMBINE_EXPERIMENT_FILE = OUTPUT_DIR / "combine_experiment.csv"
INCOME_GROUP_COMPARISON_FILE = OUTPUT_DIR / "income_group_comparison.csv"
INCOME_GROUP_DISPARITY_FILE = OUTPUT_DIR / "income_group_disparity.csv"


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
    perf_file = DATA_RAW_DIR / "driverPerformance_dataset_surabaya_14day.csv"

    if not match_files:
        raise FileNotFoundError(f"No matchdataset files found in: {OUTPUT_DIR}")

    if not perf_file.exists():
        raise FileNotFoundError(f"Driver performance file not found: {perf_file}")

    # =====================================================
    # Part 1: combine_experiment.csv
    # =====================================================
    all_results = []

    for file_path in match_files:
        print(f"Processing experiment summary: {file_path.name}")

        df = pd.read_csv(file_path)

        required_cols = [
            "batchStep",
            "orderId",
            "fare",
            "cancelProbability",
            "distance_ji_km",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{file_path.name} missing columns: {missing_cols}")

        experiment_group = extract_experiment_group(file_path.name)
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

    combine_experiment = (
        pd.concat(all_results, ignore_index=True)
        .sort_values(["experiment_group", "day", "hour"])
        .reset_index(drop=True)
    )

    combine_experiment.to_csv(COMBINE_EXPERIMENT_FILE, index=False)

    # =====================================================
    # Part 2: income comparison by driver group
    # =====================================================
    perf_df = pd.read_csv(perf_file)

    required_perf_cols = ["driverId", "driverScore"]
    missing_perf_cols = [c for c in required_perf_cols if c not in perf_df.columns]
    if missing_perf_cols:
        raise ValueError(f"Driver performance file missing columns: {missing_perf_cols}")

    # use provided group if exists, otherwise derive from score
    if "driverQualityGroup" not in perf_df.columns:
        perf_df["driverQualityGroup"] = np.where(
            perf_df["driverScore"] >= 0.8,
            "HQD",
            np.where(perf_df["driverScore"] >= 0.6, "MQD", "LQD"),
        )

    income_group_comparison_list = []
    income_group_disparity_list = []

    for file_path in match_files:
        print(f"Processing income analysis: {file_path.name}")

        df = pd.read_csv(file_path)

        required_cols = ["driverId", "fare"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{file_path.name} missing columns: {missing_cols}")

        experiment_group = extract_experiment_group(file_path.name)

        income_by_driver = (
            df.groupby("driverId", as_index=False)
            .agg(total_income=("fare", "sum"))
        )

        income_by_driver = perf_df[["driverId", "driverScore", "driverQualityGroup"]].merge(
            income_by_driver,
            on="driverId",
            how="left",
        )

        income_by_driver["total_income"] = income_by_driver["total_income"].fillna(0.0)
        income_by_driver["experiment_group"] = experiment_group

        # comparison table
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

        # disparity table
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

    income_group_comparison = (
        pd.concat(income_group_comparison_list, ignore_index=True)
        .sort_values(["experiment_group", "driverQualityGroup"])
        .reset_index(drop=True)
    )

    income_group_disparity = (
        pd.DataFrame(income_group_disparity_list)
        .sort_values(["experiment_group", "driverQualityGroup"])
        .reset_index(drop=True)
    )

    income_group_comparison.to_csv(INCOME_GROUP_COMPARISON_FILE, index=False)
    income_group_disparity.to_csv(INCOME_GROUP_DISPARITY_FILE, index=False)

    print(f"\nSaved: {COMBINE_EXPERIMENT_FILE}")
    print(f"Saved: {INCOME_GROUP_COMPARISON_FILE}")
    print(f"Saved: {INCOME_GROUP_DISPARITY_FILE}")

    print("\nSample income comparison:")
    print(income_group_comparison.head(12).to_string(index=False))

    print("\nSample income disparity:")
    print(income_group_disparity.head(12).to_string(index=False))


if __name__ == "__main__":
    main()