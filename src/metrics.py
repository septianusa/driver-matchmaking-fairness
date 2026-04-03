import numpy as np
import pandas as pd


def gini(array_like) -> float:
    x = np.array(array_like, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    if np.allclose(x.sum(), 0):
        return 0.0

    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * np.sum(x)))


def build_summary_metrics(
    orders: pd.DataFrame,
    match_df: pd.DataFrame,
    driver_perf: pd.DataFrame,
    lambda_driver_score: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    total_orders = len(orders)
    total_matched = int(orders["isMatched"].sum())
    total_unmatched = int((orders["isMatched"] == False).sum())

    match_rate = total_matched / total_orders if total_orders else np.nan
    avg_pickup_distance = match_df["distance_ji_km"].mean() if not match_df.empty else np.nan
    avg_cancel_probability = match_df["cancelProbability"].mean() if not match_df.empty else np.nan
    avg_weight = match_df["weightDriverIdOrderId"].mean() if not match_df.empty else np.nan
    avg_utility_base = match_df["utilityBase"].mean() if not match_df.empty else np.nan

    if not match_df.empty:
        income_by_driver = (
            match_df.groupby("driverId", as_index=False)
            .agg(
                total_income=("fare", "sum"),
                total_orders=("orderId", "count"),
            )
        )
    else:
        income_by_driver = pd.DataFrame(columns=["driverId", "total_income", "total_orders"])

    income_by_driver_full = driver_perf[["driverId", "driverScore"]].merge(
        income_by_driver,
        on="driverId",
        how="left",
    )
    income_by_driver_full["total_income"] = income_by_driver_full["total_income"].fillna(0.0)
    income_by_driver_full["total_orders"] = income_by_driver_full["total_orders"].fillna(0).astype(int)

    fairness_gini_income = gini(income_by_driver_full["total_income"].values)
    fairness_p10_income = np.percentile(income_by_driver_full["total_income"], 10)
    fairness_p50_income = np.percentile(income_by_driver_full["total_income"], 50)
    fairness_p90_income = np.percentile(income_by_driver_full["total_income"], 90)

    corr_driver_score_income = np.nan
    if income_by_driver_full["total_income"].nunique() > 1 and income_by_driver_full["driverScore"].nunique() > 1:
        corr_driver_score_income = income_by_driver_full["driverScore"].corr(income_by_driver_full["total_income"])

    hourly_orders = (
        orders.assign(hour=lambda df: (df["batchStep"] - 1) // 60)
        .groupby("hour", as_index=False)
        .agg(total_orders=("orderId", "count"))
    )

    if not match_df.empty:
        hourly_matches = (
            match_df.assign(hour=lambda df: (df["batchStep"] - 1) // 60)
            .groupby("hour", as_index=False)
            .agg(
                matched_orders=("orderId", "count"),
                avg_pickup_distance=("distance_ji_km", "mean"),
                avg_income=("fare", "mean"),
            )
        )
    else:
        hourly_matches = pd.DataFrame(
            {"hour": list(range(24)), "matched_orders": 0, "avg_pickup_distance": np.nan, "avg_income": np.nan}
        )

    hourly_summary = hourly_orders.merge(hourly_matches, on="hour", how="left")
    hourly_summary["matched_orders"] = hourly_summary["matched_orders"].fillna(0).astype(int)
    hourly_summary["match_rate"] = hourly_summary["matched_orders"] / hourly_summary["total_orders"]

    summary = {
        "lambda_driver_score": lambda_driver_score,
        "total_orders": total_orders,
        "total_matched": total_matched,
        "total_unmatched_or_expired": total_unmatched,
        "match_rate": match_rate,
        "avg_pickup_distance_km": avg_pickup_distance,
        "avg_cancel_probability": avg_cancel_probability,
        "avg_weight": avg_weight,
        "avg_utility_base": avg_utility_base,
        "gini_income": fairness_gini_income,
        "p10_income": fairness_p10_income,
        "p50_income": fairness_p50_income,
        "p90_income": fairness_p90_income,
        "corr_driver_score_income": corr_driver_score_income,
    }

    return summary, income_by_driver_full, hourly_summary