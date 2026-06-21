from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import resolve_project_path
from src.data.base_loader import BaseDataLoader, LoadedData
from src.models.driver_score import attach_scores_to_drivers, select_latest_scores
from src.schemas import (
    DRIVER_LOCATION_COLUMNS,
    DRIVER_SCORE_COLUMNS,
    GRID_VALUE_COLUMNS,
    ORDER_COLUMNS,
    normalize_batch_ids,
)
from src.spatial.distance import eta_minutes, haversine_km
from src.spatial.h3_utils import latlon_to_h3


def read_table(path: str | Path | None, optional: bool = False) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    table_path = Path(path)
    if not table_path.exists():
        if optional:
            return pd.DataFrame()
        raise FileNotFoundError(f"Input file does not exist: {table_path}")
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)
    raise ValueError(f"Unsupported file extension for {table_path}; use CSV or Parquet.")


class FileDataLoader(BaseDataLoader):
    def load_all(self) -> LoadedData:
        data_cfg = self.config.get("data", {})
        orders_raw = read_table(resolve_project_path(self.config, data_cfg["orders"]["path"]))
        locations_raw = read_table(resolve_project_path(self.config, data_cfg["driver_locations"]["path"]))
        scores_raw = read_table(resolve_project_path(self.config, data_cfg["driver_scores"]["path"]))
        grid_path = (data_cfg.get("grid_values") or {}).get("path")
        grid_raw = read_table(resolve_project_path(self.config, grid_path), optional=True)
        orders, order_batch = self._normalize_orders(orders_raw)
        locations, location_batch = self._normalize_driver_locations(locations_raw)
        scores, future_score_rows = self._normalize_scores(scores_raw, locations)
        grid_values = self._normalize_grid_values(grid_raw)
        return LoadedData(
            orders=orders,
            driver_locations=locations,
            driver_scores=scores,
            grid_values=grid_values,
            metadata={
                "order_batch_convention": order_batch.source_convention,
                "driver_location_batch_convention": location_batch.source_convention,
                "future_score_rows_rejected": future_score_rows,
            },
        )

    def _simulation_date(self) -> str:
        return str(self.config.get("simulation", {}).get("simulation_date"))

    def _batch_range(self) -> tuple[int | None, int | None]:
        experiment = self.config.get("experiment", {})
        start = experiment.get("batch_start")
        end = experiment.get("batch_end")
        return (int(start) if start is not None else None, int(end) if end is not None else None)

    def _normalize_orders(self, raw: pd.DataFrame):
        total_batches = int(self.config.get("simulation", {}).get("total_batches", 1440))
        resolution = int(self.config.get("spatial", {}).get("h3_resolution", 8))
        eta_cfg = self.config.get("eta", {})
        orders = raw.rename(columns=ORDER_COLUMNS).copy()
        orders["order_id"] = orders["order_id"].astype(str)
        orders["customer_id"] = orders["customer_id"].astype(str)
        orders["simulation_date"] = pd.to_datetime(orders["simulation_date"]).dt.date.astype(str)
        batch = normalize_batch_ids(orders["created_batch"], total_batches)
        orders["created_batch"] = batch.values
        orders = orders[orders["simulation_date"] == self._simulation_date()].copy()
        batch_start, batch_end = self._batch_range()
        if batch_start is not None:
            orders = orders[orders["created_batch"] >= batch_start].copy()
        if batch_end is not None:
            orders = orders[orders["created_batch"] <= batch_end].copy()
        for column in [
            "origin_latitude",
            "origin_longitude",
            "destination_latitude",
            "destination_longitude",
            "fare",
        ]:
            orders[column] = pd.to_numeric(orders[column], errors="raise")
        orders["origin_h3_index"] = [
            latlon_to_h3(lat, lon, resolution)
            for lat, lon in zip(orders["origin_latitude"], orders["origin_longitude"])
        ]
        orders["destination_h3_index"] = [
            latlon_to_h3(lat, lon, resolution)
            for lat, lon in zip(
                orders["destination_latitude"], orders["destination_longitude"]
            )
        ]
        orders["trip_distance_km"] = [
            haversine_km(o_lat, o_lon, d_lat, d_lon)
            for o_lat, o_lon, d_lat, d_lon in zip(
                orders["origin_latitude"],
                orders["origin_longitude"],
                orders["destination_latitude"],
                orders["destination_longitude"],
            )
        ]
        orders["trip_eta_minutes"] = [
            eta_minutes(
                distance,
                float(eta_cfg.get("assumed_trip_speed_km_per_hour", 20.0)),
                int(eta_cfg.get("minimum_trip_eta_minutes", 1)),
                float(eta_cfg.get("trip_buffer_minutes", 3)),
            )
            for distance in orders["trip_distance_km"]
        ]
        carry = int(self.config.get("orders", {}).get("carry_over_batches", 5))
        orders["order_expiry_batch"] = orders["created_batch"] + carry
        orders["order_status"] = "new"
        return orders.reset_index(drop=True), batch

    def _normalize_driver_locations(self, raw: pd.DataFrame):
        total_batches = int(self.config.get("simulation", {}).get("total_batches", 1440))
        resolution = int(self.config.get("spatial", {}).get("h3_resolution", 8))
        locations = raw.rename(columns=DRIVER_LOCATION_COLUMNS).copy()
        locations["driver_id"] = locations["driver_id"].astype(str)
        locations["simulation_date"] = pd.to_datetime(locations["simulation_date"]).dt.date.astype(str)
        batch = normalize_batch_ids(locations["batch_id"], total_batches)
        locations["batch_id"] = batch.values
        locations = locations[locations["simulation_date"] == self._simulation_date()].copy()
        batch_start, batch_end = self._batch_range()
        if batch_start is not None:
            locations = locations[locations["batch_id"] >= batch_start].copy()
        if batch_end is not None:
            locations = locations[locations["batch_id"] <= batch_end].copy()
        locations["latitude"] = pd.to_numeric(locations["latitude"], errors="raise")
        locations["longitude"] = pd.to_numeric(locations["longitude"], errors="raise")
        strategy = self.config.get("data_quality", {}).get(
            "driver_location_duplicate_strategy", "aggregate_mean"
        )
        duplicate_keys = ["driver_id", "simulation_date", "batch_id"]
        duplicates = locations.duplicated(duplicate_keys, keep=False)
        if duplicates.any():
            if strategy == "raise_error":
                raise ValueError("Duplicate driver location rows detected.")
            if strategy == "keep_last":
                locations = locations.drop_duplicates(duplicate_keys, keep="last")
            elif strategy == "aggregate_mean":
                locations = (
                    locations.groupby(duplicate_keys, as_index=False)
                    .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
                    .copy()
                )
            else:
                raise ValueError(f"Unsupported duplicate strategy: {strategy}")
        locations["h3_index"] = [
            latlon_to_h3(lat, lon, resolution)
            for lat, lon in zip(locations["latitude"], locations["longitude"])
        ]
        locations["is_initialized"] = True
        locations["availability_status"] = "available"
        locations["available_again_batch"] = 1
        locations["consecutive_idle_batches"] = 0
        locations["cumulative_active_minutes"] = 0
        locations["current_latitude"] = locations["latitude"]
        locations["current_longitude"] = locations["longitude"]
        locations["current_h3_index"] = locations["h3_index"]
        locations["position_source"] = "historical_bootstrap"
        return locations.sort_values(["batch_id", "driver_id"]).reset_index(drop=True), batch

    def _normalize_scores(self, raw: pd.DataFrame, locations: pd.DataFrame):
        scores = raw.rename(columns=DRIVER_SCORE_COLUMNS).copy()
        scores["driver_id"] = scores["driver_id"].astype(str)
        scores["score_reference_date"] = pd.to_datetime(scores["score_reference_date"]).dt.date.astype(str)
        scores["driver_score"] = pd.to_numeric(scores["driver_score"], errors="raise")
        latest, future_score_rows = select_latest_scores(
            scores,
            self._simulation_date(),
            self.config.get("driver_scores", {}).get("score_mode", "provided_score"),
        )
        drivers = pd.DataFrame({"driver_id": sorted(locations["driver_id"].astype(str).unique())})
        scored, _coverage = attach_scores_to_drivers(
            drivers,
            latest,
            missing_score_policy=self.config.get("driver_scores", {}).get(
                "missing_score_policy", "median_imputation"
            ),
            default_score=float(self.config.get("driver_scores", {}).get("default_score", 0.5)),
        )
        return scored.reset_index(drop=True), future_score_rows

    def _normalize_grid_values(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(columns=["h3_index", "grid_value", "latitude", "longitude"])
        grid = raw.rename(columns=GRID_VALUE_COLUMNS).copy()
        grid["h3_index"] = grid["h3_index"].astype(str)
        grid["grid_value"] = pd.to_numeric(grid["grid_value"], errors="raise")
        if "latitude" not in grid.columns:
            grid["latitude"] = pd.NA
        if "longitude" not in grid.columns:
            grid["longitude"] = pd.NA
        return grid[["h3_index", "grid_value", "latitude", "longitude"]]
