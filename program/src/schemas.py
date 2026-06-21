from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ORDER_COLUMNS = {
    "order_id": "order_id",
    "customer_id": "customer_id",
    "pickup_lat": "origin_latitude",
    "pickup_lon": "origin_longitude",
    "dest_lat": "destination_latitude",
    "dest_lon": "destination_longitude",
    "fare": "fare",
    "jakarta_data_date": "simulation_date",
    "batch_step": "created_batch",
}

DRIVER_LOCATION_COLUMNS = {
    "driver_id": "driver_id",
    "bucket_step": "batch_id",
    "avg_latitude": "latitude",
    "avg_longitude": "longitude",
    "jakarta_data_date": "simulation_date",
}

DRIVER_SCORE_COLUMNS = {
    "driver_id": "driver_id",
    "jakarta_period_date": "score_reference_date",
    "calculated_performance_score": "driver_score",
}

GRID_VALUE_COLUMNS = {
    "h3_index": "h3_index",
    "grid_value": "grid_value",
    "latitude": "latitude",
    "longitude": "longitude",
}

ORDER_STATUSES = {
    "new",
    "active_unmatched",
    "matched",
    "completed_expected",
    "completed_realized",
    "cancelled_realized",
    "expired_unfulfilled",
}


@dataclass(frozen=True)
class BatchNormalizationResult:
    values: pd.Series
    source_convention: str
    minimum_before: int | None
    maximum_before: int | None
    minimum_after: int | None
    maximum_after: int | None


def normalize_batch_ids(batch_values: pd.Series, total_batches: int = 1440) -> BatchNormalizationResult:
    numeric = pd.to_numeric(batch_values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Batch ids contain non-numeric values.")
    values = numeric.astype(int)
    if values.empty:
        return BatchNormalizationResult(values, "empty", None, None, None, None)
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum < 0 or maximum > total_batches:
        raise ValueError(
            f"Unsupported batch range {minimum}..{maximum}; expected 0..{total_batches - 1} or 1..{total_batches}."
        )
    if minimum == 0:
        normalized = values + 1
        convention = f"zero_based_0_to_{total_batches - 1}"
    elif maximum == total_batches:
        normalized = values
        convention = f"one_based_1_to_{total_batches}"
    elif minimum >= 1 and maximum <= total_batches:
        normalized = values
        convention = "one_based_assumed_from_subset"
    elif minimum >= 0 and maximum <= total_batches - 1:
        normalized = values + 1
        convention = "zero_based_assumed_from_subset"
    else:
        raise ValueError(
            f"Unsupported batch range {minimum}..{maximum}; expected 0..{total_batches - 1} or 1..{total_batches}."
        )
    if normalized.min() < 1 or normalized.max() > total_batches:
        raise ValueError("Batch normalization produced values outside 1..total_batches.")
    return BatchNormalizationResult(
        values=normalized,
        source_convention=convention,
        minimum_before=minimum,
        maximum_before=maximum,
        minimum_after=int(normalized.min()),
        maximum_after=int(normalized.max()),
    )


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> list[str]:
    missing = sorted(required - set(df.columns))
    return [f"{label} missing required columns: {', '.join(missing)}"] if missing else []

