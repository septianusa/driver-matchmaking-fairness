"""Dataset validation and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig, load_config
from simulation_data_generation.io_utils import read_partitioned_parquet
from simulation_data_generation.spatial import haversine_km, is_inside_boundary, timestamp_to_batch
from simulation_data_generation.temporal_demand import simulation_dates


@dataclass
class ValidationResult:
    """Validation result container."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _required_columns(frame: pd.DataFrame, required: list[str], dataset: str, result: ValidationResult) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        result.errors.append(f"{dataset}: missing required columns {missing}")


def _plot_series(path: Path, x, y, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="#1f4e79", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_hist(path: Path, values, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=30, color="#1f4e79", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate_dataset(
    data_dir: str | Path,
    *,
    config: GenerationConfig | str | Path,
    report_dir: str | Path | None = None,
) -> ValidationResult:
    """Validate generated datasets and write a Markdown report plus figures."""
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    data_path = Path(data_dir).resolve()
    report_path = Path(report_dir).resolve() if report_dir else data_path.parent.parent / "reports"
    figures = report_path / "figures"
    result = ValidationResult()

    pois = read_partitioned_parquet(data_path / "poi_reference.parquet")
    driver_scores = read_partitioned_parquet(data_path / "driver_scores.parquet")
    orders = read_partitioned_parquet(data_path / "orders")
    positions = read_partitioned_parquet(data_path / "driver_positions")
    grid_reference = read_partitioned_parquet(data_path / "h3_grid_reference.parquet")
    grid_values = read_partitioned_parquet(data_path / "h3_grid_values")

    _required_columns(orders, ["order_id", "simulation_date", "batch_index", "pickup_h3_index", "dropoff_h3_index"], "orders", result)
    _required_columns(positions, ["driver_id", "simulation_date", "batch_index", "online_session_id", "h3_index"], "driver_positions", result)
    _required_columns(driver_scores, ["driver_id", "driver_behavior_score", "score_segment"], "driver_scores", result)
    _required_columns(grid_values, ["simulation_date", "batch_index", "h3_index", "grid_value"], "h3_grid_values", result)

    expected_dates = simulation_dates(cfg)
    generated_dates = sorted(orders["simulation_date"].astype(str).unique().tolist()) if not orders.empty else []
    result.stats["simulation_dates"] = generated_dates
    if cfg.number_of_days == 7 and len(generated_dates) != 7:
        result.errors.append(f"Expected exactly seven simulation dates, found {len(generated_dates)}")
    if generated_dates != expected_dates:
        result.warnings.append(f"Generated dates differ from configured dates: {generated_dates} vs {expected_dates}")

    if not orders.empty:
        if orders["order_id"].duplicated().any():
            result.errors.append("orders: order_id must be unique")
        if not orders["batch_index"].between(1, 1440).all():
            result.errors.append("orders: batch_index outside 1..1440")
        if orders[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]].isna().any().any():
            result.errors.append("orders: unexpected null coordinates")
        daily = orders.groupby("simulation_date").size()
        result.stats["orders_by_day"] = {str(k): int(v) for k, v in daily.items()}
        if cfg.number_of_days == 7 and (daily < cfg.minimum_daily_orders).any():
            result.errors.append("orders: at least one default simulation day is not above the minimum order count")
        if cfg.number_of_days == 7 and cfg.target_daily_order_min is not None and cfg.target_daily_order_max is not None:
            outside_target_band = daily[
                (daily < int(cfg.target_daily_order_min)) | (daily > int(cfg.target_daily_order_max))
            ]
            if not outside_target_band.empty:
                result.errors.append(
                    "orders: daily order counts are outside the configured design band; "
                    f"outside band={ {str(k): int(v) for k, v in outside_target_band.items()} }"
                )
        result.stats["average_daily_orders"] = float(daily.mean())
        same_coord = (
            (orders["pickup_latitude"] == orders["dropoff_latitude"])
            & (orders["pickup_longitude"] == orders["dropoff_longitude"])
        ).sum()
        if int(same_coord) > 0:
            result.errors.append(f"orders: {same_coord} pickup/dropoff coordinate pairs are identical")
        if not orders["route_distance_km"].ge(orders["straight_line_distance_km"]).all():
            result.errors.append("orders: route_distance_km below straight_line_distance_km")
        if not orders["fare_amount"].gt(0).all():
            result.errors.append("orders: fare_amount must be positive")
        fare = cfg.fare_parameters
        minimum_fare = float(fare.get("minimum_fare", 0.0))
        rate_min = float(fare.get("distance_rate_per_km_min", 0.0))
        rate_max = float(fare.get("distance_rate_per_km_max", float("inf")))
        fare_multiplier = float(fare.get("fare_distance_multiplier", 1.4))
        if (orders["fare_amount"] < minimum_fare - 1e-6).any():
            result.errors.append("orders: fare_amount below configured minimum fare")
        if "fare_rate_per_km" in orders.columns and not orders["fare_rate_per_km"].between(rate_min, rate_max).all():
            result.errors.append("orders: fare_rate_per_km outside configured fare band")
        if "fare_distance_km" in orders.columns:
            expected_fare_distance = orders["straight_line_distance_km"] * fare_multiplier
            if not np.allclose(orders["fare_distance_km"], expected_fare_distance, atol=0.01):
                result.errors.append("orders: fare_distance_km does not match straight_line_distance_km times fare multiplier")
        outside = 0
        for row in orders.head(20000).itertuples(index=False):
            if not is_inside_boundary(row.pickup_latitude, row.pickup_longitude, cfg.study_boundary):
                outside += 1
            if not is_inside_boundary(row.dropoff_latitude, row.dropoff_longitude, cfg.study_boundary):
                outside += 1
        if outside:
            result.errors.append(f"orders: {outside} sampled coordinates outside boundary")

    if not positions.empty:
        if not positions["batch_index"].between(1, 1440).all():
            result.errors.append("driver_positions: batch_index outside 1..1440")
        unknown_drivers = set(positions["driver_id"]) - set(driver_scores["driver_id"])
        if unknown_drivers:
            result.errors.append(f"driver_positions: {len(unknown_drivers)} driver IDs not present in driver_scores")
        driver_day = positions.groupby(["driver_id", "simulation_date"]).size()
        if (driver_day > 480).any():
            result.errors.append("driver_positions: at least one driver-day exceeds 480 rows")
        by_session = positions.sort_values(["driver_id", "online_session_id", "batch_index"]).groupby("online_session_id")
        discontinuities = 0
        for _, group in by_session:
            diffs = group["batch_index"].diff().dropna()
            discontinuities += int((diffs != 1).sum())
        if discontinuities:
            result.errors.append(f"driver_positions: {discontinuities} non-consecutive in-session batch steps")
        if positions["speed_kph"].max() > cfg.road_network.maximum_speed_kph + 1e-6:
            result.errors.append("driver_positions: speed exceeds configured maximum")
        active_by_day = positions.groupby("simulation_date")["driver_id"].nunique()
        result.stats["active_drivers_by_day"] = {str(k): int(v) for k, v in active_by_day.items()}
        if cfg.number_of_days == 7 and cfg.active_driver_target_min is not None and cfg.active_driver_target_max is not None:
            outside_active_band = active_by_day[
                (active_by_day < int(cfg.active_driver_target_min)) | (active_by_day > int(cfg.active_driver_target_max))
            ]
            if not outside_active_band.empty:
                result.errors.append(
                    "driver_positions: active drivers per day are outside the configured design band; "
                    f"outside band={ {str(k): int(v) for k, v in outside_active_band.items()} }"
                )
        result.stats["position_rows"] = int(len(positions))
        result.stats["active_drivers"] = int(positions["driver_id"].nunique())

    if not driver_scores.empty:
        if not (driver_scores["completed_orders"] <= driver_scores["accepted_orders"]).all():
            result.errors.append("driver_scores: completed_orders exceeds accepted_orders")
        if not (driver_scores["accepted_orders"] <= driver_scores["total_offered_orders"]).all():
            result.errors.append("driver_scores: accepted_orders exceeds total_offered_orders")
        recomputed_ar = driver_scores["accepted_orders"] / driver_scores["total_offered_orders"].replace(0, np.nan)
        recomputed_cr = driver_scores["completed_orders"] / driver_scores["accepted_orders"].replace(0, np.nan)
        recomputed_ar = recomputed_ar.fillna(0.0)
        recomputed_cr = recomputed_cr.fillna(0.0)
        if not np.allclose(recomputed_ar, driver_scores["acceptance_rate"], atol=0.01):
            result.errors.append("driver_scores: acceptance_rate does not match counts")
        if not np.allclose(recomputed_cr, driver_scores["completion_rate"], atol=0.01):
            result.errors.append("driver_scores: completion_rate does not match counts")
        proportions = driver_scores["score_segment"].value_counts(normalize=True).to_dict()
        result.stats["score_segment_proportions"] = {str(k): round(float(v), 4) for k, v in proportions.items()}
        for segment in ("high", "medium", "low"):
            target = float(cfg.driver_score_targets.get(segment, 0.0))
            observed = float(proportions.get(segment, 0.0))
            if abs(observed - target) > float(cfg.driver_score_targets.get("tolerance", 0.05)) + 0.015:
                result.errors.append(f"driver_scores: {segment} proportion {observed:.3f} differs from target {target:.3f}")

    if not grid_values.empty:
        grid_cells = set(grid_reference["h3_index"].astype(str))
        order_cells = set(orders.get("pickup_h3_index", pd.Series(dtype=str)).astype(str)) | set(
            orders.get("dropoff_h3_index", pd.Series(dtype=str)).astype(str)
        )
        driver_cells = set(positions.get("h3_index", pd.Series(dtype=str)).astype(str))
        missing_cells = (order_cells | driver_cells) - grid_cells
        if missing_cells:
            result.errors.append(f"h3_grid_reference: missing {len(missing_cells)} order/driver H3 cells")
        if not np.isfinite(grid_values["grid_value"]).all():
            result.errors.append("h3_grid_values: non-finite grid_value found")
        result.stats["h3_grid_cells"] = int(len(grid_reference))
        result.stats["h3_grid_value_rows"] = int(len(grid_values))

    _write_validation_figures(figures, orders, positions, driver_scores, grid_values)
    _write_markdown_report(report_path / "validation_summary.md", result)
    return result


def _write_validation_figures(
    figures: Path,
    orders: pd.DataFrame,
    positions: pd.DataFrame,
    driver_scores: pd.DataFrame,
    grid_values: pd.DataFrame,
) -> None:
    if not orders.empty:
        by_minute = orders.groupby("batch_index").size().reindex(range(1, 1441), fill_value=0)
        _plot_series(figures / "orders_by_minute.png", by_minute.index, by_minute.values, "Orders by Minute", "Batch", "Orders")
        by_day = orders.groupby("simulation_date").size()
        _plot_series(figures / "orders_by_day.png", by_day.index.astype(str), by_day.values, "Orders by Day", "Date", "Orders")
        _plot_hist(figures / "trip_distance_distribution.png", orders["route_distance_km"], "Trip Distance Distribution", "Route distance (km)")
        _plot_hist(figures / "fare_distribution.png", orders["fare_amount"], "Fare Distribution", "Fare (Rp)")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(orders["pickup_longitude"].head(50000), orders["pickup_latitude"].head(50000), s=1, alpha=0.15)
        ax.set_title("Order Pickup Heatmap Proxy")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(figures / "order_pickup_heatmap.png", dpi=150)
        plt.close(fig)
    if not positions.empty:
        active = positions.groupby("batch_index")["driver_id"].nunique().reindex(range(1, 1441), fill_value=0)
        _plot_series(figures / "active_drivers_by_minute.png", active.index, active.values, "Active Drivers by Minute", "Batch", "Drivers")
        online_hours = positions.groupby(["driver_id", "simulation_date"]).size() / 60.0
        _plot_hist(figures / "driver_online_hours_distribution.png", online_hours, "Driver Online Hours", "Hours per driver-day")
    if not driver_scores.empty:
        _plot_hist(figures / "driver_score_distribution.png", driver_scores["driver_behavior_score"], "Driver Score Distribution", "Score")
        proportions = driver_scores["score_segment"].value_counts(normalize=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(proportions.index.astype(str), proportions.values, color="#1f4e79")
        ax.set_title("Score Segment Proportions")
        ax.set_ylabel("Share")
        fig.tight_layout()
        fig.savefig(figures / "score_segment_proportions.png", dpi=150)
        plt.close(fig)
    if not grid_values.empty:
        ratio = grid_values.groupby("batch_index")[["active_order_count", "available_driver_count"]].sum()
        supply = ratio["available_driver_count"].replace(0, np.nan)
        sd = (supply / ratio["active_order_count"].replace(0, np.nan)).fillna(0)
        _plot_series(figures / "supply_demand_ratio_by_time.png", sd.index, sd.values, "Supply-Demand Ratio by Time", "Batch", "Drivers / orders")


def _write_markdown_report(path: Path, result: ValidationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation Summary", "", f"Status: {'PASS' if result.ok else 'FAIL'}", ""]
    lines.append("## Errors")
    lines.extend([f"- {error}" for error in result.errors] or ["- None"])
    lines.append("")
    lines.append("## Warnings")
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- None"])
    lines.append("")
    lines.append("## Statistics")
    for key, value in result.stats.items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
