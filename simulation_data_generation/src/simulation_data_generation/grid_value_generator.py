"""H3 grid reference and dynamic grid-value generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.constants import MINUTES_PER_DAY
from simulation_data_generation.spatial import batch_to_timestamp, h3_to_latlon


def build_h3_reference(cells: set[str], config: GenerationConfig) -> pd.DataFrame:
    """Build the H3 grid reference from generated cells."""
    rows = []
    for index, cell in enumerate(sorted(cells), start=1):
        lat, lon = h3_to_latlon(cell)
        rows.append(
            {
                "location_id": f"LOC-{index:06d}",
                "h3_index": cell,
                "h3_resolution": int(config.h3_resolution),
                "centroid_latitude": round(lat, 7),
                "centroid_longitude": round(lon, 7),
            }
        )
    return pd.DataFrame(rows)


def aggregate_order_cells(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate active orders by date, batch, and pickup H3 cell."""
    if orders.empty:
        return pd.DataFrame(columns=["simulation_date", "batch_index", "h3_index", "active_order_count"])
    return (
        orders.groupby(["simulation_date", "batch_index", "pickup_h3_index"], as_index=False)
        .size()
        .rename(columns={"pickup_h3_index": "h3_index", "size": "active_order_count"})
    )


def aggregate_driver_cells(positions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate online drivers by date, batch, and H3 cell."""
    if positions.empty:
        return pd.DataFrame(columns=["simulation_date", "batch_index", "h3_index", "available_driver_count"])
    return (
        positions[positions["driver_status"].isin(["available", "moving_to_demand_area", "resting"])]
        .groupby(["simulation_date", "batch_index", "h3_index"], as_index=False)
        .agg(available_driver_count=("driver_id", "nunique"))
    )


def generate_grid_values(
    config: GenerationConfig,
    grid_reference: pd.DataFrame,
    order_counts: pd.DataFrame,
    driver_counts: pd.DataFrame,
    dates: list[str],
) -> pd.DataFrame:
    """Generate dynamic grid values with temporal persistence."""
    if grid_reference.empty:
        return pd.DataFrame()
    location_lookup = grid_reference.set_index("h3_index").to_dict(orient="index")
    order_lookup = {
        (row.simulation_date, int(row.batch_index), row.h3_index): int(row.active_order_count)
        for row in order_counts.itertuples(index=False)
    }
    driver_lookup = {
        (row.simulation_date, int(row.batch_index), row.h3_index): int(row.available_driver_count)
        for row in driver_counts.itertuples(index=False)
    }
    max_demand = max([1] + [int(value) for value in order_counts.get("active_order_count", pd.Series(dtype=int)).tolist()])
    values = {str(cell): float(config.cold_start_grid_value) for cell in grid_reference["h3_index"].astype(str)}
    rows: list[dict[str, object]] = []
    batches = range(1, MINUTES_PER_DAY + 1) if config.dynamic_grid_values else [1]

    for simulation_date in dates:
        for batch in batches:
            for cell in values:
                demand = order_lookup.get((simulation_date, batch, cell), 0)
                supply = driver_lookup.get((simulation_date, batch, cell), 0)
                demand_intensity = float(demand / max_demand)
                target = float(config.cold_start_grid_value) + 1.8 * demand_intensity - 0.05 * min(supply, 25)
                values[cell] = float(np.clip(0.88 * values[cell] + 0.12 * target, 0.05, 5.0))
                ref = location_lookup[cell]
                rows.append(
                    {
                        "simulation_date": simulation_date,
                        "batch_index": int(batch),
                        "batch_timestamp": batch_to_timestamp(simulation_date, batch, config.timezone).isoformat(),
                        "location_id": ref["location_id"],
                        "h3_index": cell,
                        "h3_resolution": int(config.h3_resolution),
                        "centroid_latitude": ref["centroid_latitude"],
                        "centroid_longitude": ref["centroid_longitude"],
                        "grid_value": round(values[cell], 6),
                        "demand_intensity": round(demand_intensity, 6),
                        "available_driver_count": int(supply),
                        "active_order_count": int(demand),
                    }
                )
    return pd.DataFrame(rows)
