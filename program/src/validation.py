from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_project_path
from src.data.file_loader import read_table
from src.models.driver_score import select_latest_scores, validate_driver_scores
from src.schemas import (
    DRIVER_LOCATION_COLUMNS,
    DRIVER_SCORE_COLUMNS,
    ORDER_COLUMNS,
    normalize_batch_ids,
    require_columns,
)


@dataclass
class ValidationReport:
    selected_simulation_date: str
    input_file_existence: dict[str, bool] = field(default_factory=dict)
    required_columns: dict[str, list[str]] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    date_coverage: dict[str, list[str]] = field(default_factory=dict)
    missing_values: dict[str, dict[str, int]] = field(default_factory=dict)
    duplicate_rows: dict[str, int] = field(default_factory=dict)
    coordinate_validation: dict[str, int] = field(default_factory=dict)
    fare_validation: dict[str, int] = field(default_factory=dict)
    batch_index_convention: dict[str, str] = field(default_factory=dict)
    batch_index_normalization_results: dict[str, dict[str, int | None]] = field(default_factory=dict)
    unique_driver_count: int = 0
    unique_order_count: int = 0
    driver_score_coverage: dict[str, Any] = field(default_factory=dict)
    drivers_missing_scores: list[str] = field(default_factory=list)
    future_score_rows_rejected: int = 0
    orders_per_hour: dict[str, int] = field(default_factory=dict)
    active_drivers_per_hour: dict[str, int] = field(default_factory=dict)
    missing_location_batches_by_driver: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    blocking_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocking_errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _missing_values(df: pd.DataFrame) -> dict[str, int]:
    return {column: int(count) for column, count in df.isna().sum().items() if int(count) > 0}


def _date_coverage(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df:
        return []
    return sorted(pd.to_datetime(df[column], errors="coerce").dt.date.dropna().astype(str).unique())


def _hour_counts(batch_series: pd.Series) -> dict[str, int]:
    if batch_series.empty:
        return {}
    hours = ((batch_series.astype(int) - 1) // 60).clip(lower=0)
    return {str(int(hour)): int(count) for hour, count in hours.value_counts().sort_index().items()}


def validate_inputs(config: dict[str, Any]) -> ValidationReport:
    simulation = config.get("simulation", {})
    selected_date = str(simulation.get("simulation_date"))
    total_batches = int(simulation.get("total_batches", 1440))
    data_cfg = config.get("data", {})
    paths = {
        "orders": resolve_project_path(config, data_cfg.get("orders", {}).get("path")),
        "driver_locations": resolve_project_path(
            config, data_cfg.get("driver_locations", {}).get("path")
        ),
        "driver_scores": resolve_project_path(config, data_cfg.get("driver_scores", {}).get("path")),
        "grid_values": resolve_project_path(config, (data_cfg.get("grid_values") or {}).get("path")),
    }
    report = ValidationReport(selected_simulation_date=selected_date)
    frames: dict[str, pd.DataFrame] = {}
    required = {
        "orders": set(ORDER_COLUMNS),
        "driver_locations": set(DRIVER_LOCATION_COLUMNS),
        "driver_scores": set(DRIVER_SCORE_COLUMNS),
    }
    for name, path in paths.items():
        exists = bool(path and Path(path).exists())
        report.input_file_existence[name] = exists
        if name != "grid_values" and not exists:
            report.blocking_errors.append(f"Missing required input file: {name} ({path}).")
            frames[name] = pd.DataFrame()
            continue
        try:
            frames[name] = read_table(path, optional=name == "grid_values")
        except Exception as exc:
            report.blocking_errors.append(f"Could not read {name}: {exc}")
            frames[name] = pd.DataFrame()
        report.row_counts[name] = int(len(frames[name]))
    for name, columns in required.items():
        missing = require_columns(frames[name], columns, name)
        report.required_columns[name] = missing
        report.blocking_errors.extend(missing)
    if report.blocking_errors:
        return report

    orders = frames["orders"].rename(columns=ORDER_COLUMNS).copy()
    locations = frames["driver_locations"].rename(columns=DRIVER_LOCATION_COLUMNS).copy()
    scores = frames["driver_scores"].rename(columns=DRIVER_SCORE_COLUMNS).copy()

    report.date_coverage["orders"] = _date_coverage(orders, "simulation_date")
    report.date_coverage["driver_locations"] = _date_coverage(locations, "simulation_date")
    report.date_coverage["driver_scores"] = _date_coverage(scores, "score_reference_date")
    report.missing_values["orders"] = _missing_values(orders)
    report.missing_values["driver_locations"] = _missing_values(locations)
    report.missing_values["driver_scores"] = _missing_values(scores)
    if any(report.missing_values.values()):
        report.blocking_errors.append("Missing values detected in mandatory input columns.")

    orders["simulation_date"] = pd.to_datetime(orders["simulation_date"]).dt.date.astype(str)
    locations["simulation_date"] = pd.to_datetime(locations["simulation_date"]).dt.date.astype(str)
    scores["score_reference_date"] = pd.to_datetime(scores["score_reference_date"]).dt.date.astype(str)
    orders_for_date = orders[orders["simulation_date"] == selected_date].copy()
    locations_for_date = locations[locations["simulation_date"] == selected_date].copy()
    if orders_for_date.empty:
        report.blocking_errors.append(f"No orders found for simulation_date={selected_date}.")
    if locations_for_date.empty:
        report.blocking_errors.append(f"No driver locations found for simulation_date={selected_date}.")

    duplicate_order_ids = int(orders_for_date["order_id"].astype(str).duplicated().sum())
    report.duplicate_rows["orders_duplicate_order_ids"] = duplicate_order_ids
    if duplicate_order_ids:
        report.blocking_errors.append(f"Duplicate order_id values detected: {duplicate_order_ids}.")
    duplicate_location_rows = int(
        locations_for_date.duplicated(["driver_id", "simulation_date", "batch_id"]).sum()
    )
    report.duplicate_rows["driver_location_duplicate_keys"] = duplicate_location_rows
    if duplicate_location_rows:
        strategy = config.get("data_quality", {}).get(
            "driver_location_duplicate_strategy", "aggregate_mean"
        )
        report.warnings.append(
            f"Duplicate driver-location rows detected: {duplicate_location_rows}; strategy={strategy}."
        )
        if strategy == "raise_error":
            report.blocking_errors.append("Duplicate driver-location rows are configured as blocking.")

    for label, frame, lat_col, lon_col in [
        ("orders_origin", orders_for_date, "origin_latitude", "origin_longitude"),
        ("orders_destination", orders_for_date, "destination_latitude", "destination_longitude"),
        ("driver_locations", locations_for_date, "latitude", "longitude"),
    ]:
        lat = pd.to_numeric(frame[lat_col], errors="coerce")
        lon = pd.to_numeric(frame[lon_col], errors="coerce")
        invalid_lat = int(((lat < -90) | (lat > 90) | lat.isna()).sum())
        invalid_lon = int(((lon < -180) | (lon > 180) | lon.isna()).sum())
        report.coordinate_validation[f"{label}_invalid_latitude"] = invalid_lat
        report.coordinate_validation[f"{label}_invalid_longitude"] = invalid_lon
        if invalid_lat or invalid_lon:
            report.blocking_errors.append(f"Invalid coordinates detected for {label}.")
    invalid_fares = int((pd.to_numeric(orders_for_date["fare"], errors="coerce") <= 0).sum())
    report.fare_validation["non_positive_fare_rows"] = invalid_fares
    if invalid_fares:
        report.blocking_errors.append(f"Non-positive fare rows detected: {invalid_fares}.")

    try:
        order_batch = normalize_batch_ids(orders["created_batch"], total_batches)
        location_batch = normalize_batch_ids(locations["batch_id"], total_batches)
        report.batch_index_convention["orders"] = order_batch.source_convention
        report.batch_index_convention["driver_locations"] = location_batch.source_convention
        report.batch_index_normalization_results["orders"] = {
            "minimum_before": order_batch.minimum_before,
            "maximum_before": order_batch.maximum_before,
            "minimum_after": order_batch.minimum_after,
            "maximum_after": order_batch.maximum_after,
        }
        report.batch_index_normalization_results["driver_locations"] = {
            "minimum_before": location_batch.minimum_before,
            "maximum_before": location_batch.maximum_before,
            "minimum_after": location_batch.minimum_after,
            "maximum_after": location_batch.maximum_after,
        }
        orders_for_date["created_batch"] = normalize_batch_ids(
            orders_for_date["created_batch"], total_batches
        ).values
        locations_for_date["batch_id"] = normalize_batch_ids(
            locations_for_date["batch_id"], total_batches
        ).values
        report.orders_per_hour = _hour_counts(orders_for_date["created_batch"])
        report.active_drivers_per_hour = _hour_counts(locations_for_date["batch_id"])
    except Exception as exc:
        report.blocking_errors.append(str(exc))

    score_errors = validate_driver_scores(scores.assign(driver_score=pd.to_numeric(scores["driver_score"])))
    report.blocking_errors.extend(score_errors)
    latest_scores, future_rows = select_latest_scores(
        scores.assign(driver_score=pd.to_numeric(scores["driver_score"], errors="coerce")),
        selected_date,
        config.get("driver_scores", {}).get("score_mode", "provided_score"),
    )
    report.future_score_rows_rejected = int(future_rows)
    if future_rows:
        report.warnings.append(f"Future score rows rejected for leakage prevention: {future_rows}.")
    driver_ids = set(locations_for_date["driver_id"].astype(str).unique())
    scored_ids = set(latest_scores["driver_id"].astype(str).unique())
    missing_driver_ids = sorted(driver_ids - scored_ids)
    report.unique_driver_count = len(driver_ids)
    report.unique_order_count = int(orders_for_date["order_id"].astype(str).nunique())
    report.drivers_missing_scores = missing_driver_ids
    report.driver_score_coverage = {
        "drivers_with_valid_scores": len(driver_ids & scored_ids),
        "drivers_without_scores": len(missing_driver_ids),
        "score_coverage_percent": round(
            (len(driver_ids & scored_ids) / len(driver_ids) * 100.0) if driver_ids else 0.0, 2
        ),
    }
    if missing_driver_ids:
        policy = config.get("driver_scores", {}).get("missing_score_policy", "median_imputation")
        report.warnings.append(f"{len(missing_driver_ids)} drivers missing scores; policy={policy}.")
        if policy == "raise_error":
            report.blocking_errors.append("Drivers missing scores are configured as blocking.")

    missing_location: dict[str, int] = {}
    if not locations_for_date.empty:
        observed = locations_for_date.groupby("driver_id")["batch_id"].nunique()
        for driver_id, count in observed.items():
            missing_location[str(driver_id)] = max(0, total_batches - int(count))
    report.missing_location_batches_by_driver = missing_location
    return report


def write_validation_report(config: dict[str, Any], output_dir: str | Path) -> ValidationReport:
    report = validate_inputs(config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report.write(Path(output_dir) / "validation_report.json")
    return report

