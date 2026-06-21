from __future__ import annotations

import pandas as pd


def validate_driver_scores(scores: pd.DataFrame) -> list[str]:
    if scores.empty:
        return []
    invalid = scores[(scores["driver_score"] < 0) | (scores["driver_score"] > 1)]
    if not invalid.empty:
        return [f"Driver scores outside [0, 1]: {len(invalid)} rows."]
    return []


def calculate_score_from_components(df: pd.DataFrame) -> pd.Series:
    required = {"acceptance_rate", "completion_rate", "online_hours_7d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing score component columns: {', '.join(sorted(missing))}")
    return (
        df["acceptance_rate"].astype(float) / 3.0
        + df["completion_rate"].astype(float) / 3.0
        + df["online_hours_7d"].astype(float).clip(lower=0, upper=40) / 40.0 / 3.0
    ).clip(0, 1)


def select_latest_scores(
    scores: pd.DataFrame,
    simulation_date: str,
    score_mode: str = "provided_score",
) -> tuple[pd.DataFrame, int]:
    if scores.empty:
        return pd.DataFrame(columns=["driver_id", "driver_score", "score_reference_date"]), 0
    scored = scores.copy()
    scored["score_reference_date"] = pd.to_datetime(scored["score_reference_date"]).dt.date
    sim_date = pd.to_datetime(simulation_date).date()
    future_rows = int((scored["score_reference_date"] > sim_date).sum())
    valid = scored[scored["score_reference_date"] <= sim_date].copy()
    if score_mode == "calculate_from_components":
        valid["driver_score"] = calculate_score_from_components(valid)
    valid = valid.sort_values(["driver_id", "score_reference_date"])
    latest = valid.groupby("driver_id", as_index=False).tail(1)
    return latest[["driver_id", "driver_score", "score_reference_date"]].reset_index(drop=True), future_rows


def attach_scores_to_drivers(
    drivers: pd.DataFrame,
    latest_scores: pd.DataFrame,
    *,
    missing_score_policy: str = "median_imputation",
    default_score: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    result = drivers.copy()
    result["driver_id"] = result["driver_id"].astype(str)
    latest = latest_scores.copy()
    latest["driver_id"] = latest["driver_id"].astype(str)
    result = result.merge(latest[["driver_id", "driver_score"]], on="driver_id", how="left")
    missing_count = int(result["driver_score"].isna().sum())
    if missing_count:
        if missing_score_policy == "median_imputation":
            median = float(latest["driver_score"].median()) if not latest.empty else default_score
            result["driver_score"] = result["driver_score"].fillna(median)
        elif missing_score_policy == "default_value":
            result["driver_score"] = result["driver_score"].fillna(float(default_score))
        elif missing_score_policy == "drop_driver":
            result = result.dropna(subset=["driver_score"]).copy()
        elif missing_score_policy == "raise_error":
            raise ValueError(f"{missing_count} drivers do not have valid scores.")
        else:
            raise ValueError(f"Unsupported missing score policy: {missing_score_policy}")
    unique_drivers = drivers["driver_id"].astype(str).nunique() if not drivers.empty else 0
    coverage = 0.0 if unique_drivers == 0 else (unique_drivers - missing_count) / unique_drivers
    return result, {
        "drivers_with_valid_scores": int(unique_drivers - missing_count),
        "drivers_without_scores": missing_count,
        "score_coverage_percent": round(coverage * 100.0, 2),
    }

