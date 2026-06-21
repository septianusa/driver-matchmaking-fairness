from __future__ import annotations

import pandas as pd

from src.evaluation.fairness import gini, safe_correlation


def summarize_results(
    orders: pd.DataFrame,
    match_log: pd.DataFrame,
    driver_metrics: pd.DataFrame,
    batch_metrics: pd.DataFrame,
    runtime_seconds: float,
) -> dict:
    total_orders = int(len(orders))
    matched_orders = int(match_log["order_id"].nunique()) if not match_log.empty else 0
    expected_completed = (
        float(match_log["predicted_completion_probability"].sum()) if not match_log.empty else 0.0
    )
    median_pickup = float(match_log["pickup_distance_km"].median()) if not match_log.empty else 0.0
    total_utility = float(match_log["economic_utility"].sum()) if not match_log.empty else 0.0
    incomes = driver_metrics.get("total_expected_income", pd.Series(dtype=float))
    scores = driver_metrics.get("driver_score", pd.Series(dtype=float))
    completed = driver_metrics.get("expected_completed_orders", pd.Series(dtype=float))
    return {
        "total_orders": total_orders,
        "matched_orders": matched_orders,
        "match_rate": matched_orders / total_orders if total_orders else 0.0,
        "expected_completed_orders": expected_completed,
        "expected_conversion_rate": expected_completed / matched_orders if matched_orders else 0.0,
        "median_pickup_distance_km": median_pickup,
        "total_expected_economic_utility": total_utility,
        "total_expected_driver_income": float(incomes.sum()) if not driver_metrics.empty else 0.0,
        "pearson_score_income_correlation": safe_correlation(scores, incomes, "pearson"),
        "spearman_score_income_correlation": safe_correlation(scores, incomes, "spearman"),
        "pearson_score_completed_orders_correlation": safe_correlation(scores, completed, "pearson"),
        "gini_coefficient": gini(incomes),
        "runtime_seconds": float(runtime_seconds),
        "batches_processed": int(len(batch_metrics)),
    }

