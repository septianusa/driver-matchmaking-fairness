from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.file_loader import FileDataLoader


def _active_orders_by_batch(orders: pd.DataFrame, total_batches: int) -> pd.Series:
    counts = []
    for batch in range(1, total_batches + 1):
        active = orders[
            (orders["created_batch"].astype(int) <= batch)
            & (orders["order_expiry_batch"].astype(int) >= batch)
        ]
        counts.append(len(active))
    return pd.Series(counts, index=range(1, total_batches + 1), dtype=int)


def _available_drivers_by_batch(locations: pd.DataFrame, total_batches: int) -> pd.Series:
    counts = []
    observed: set[str] = set()
    by_batch = {
        int(batch): set(group["driver_id"].astype(str))
        for batch, group in locations.groupby("batch_id", sort=True)
    }
    for batch in range(1, total_batches + 1):
        observed.update(by_batch.get(batch, set()))
        counts.append(len(observed))
    return pd.Series(counts, index=range(1, total_batches + 1), dtype=int)


def build_dataset_profile(config: dict, *, data_mode: str, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded = FileDataLoader(config).load_all()
    orders = loaded.orders
    locations = loaded.driver_locations
    scores = loaded.driver_scores
    total_batches = int(config.get("simulation", {}).get("total_batches", 1440))
    simulation_date = str(config.get("simulation", {}).get("simulation_date"))
    batch_start = int(config.get("experiment", {}).get("batch_start", 1))
    batch_end = int(config.get("experiment", {}).get("batch_end", total_batches))

    active_orders = _active_orders_by_batch(orders, total_batches).loc[batch_start:batch_end]
    available_drivers = _available_drivers_by_batch(locations, total_batches).loc[batch_start:batch_end]
    raw_pairs = active_orders * available_drivers
    by_batch = pd.DataFrame(
        {
            "batch_id": active_orders.index,
            "active_orders": active_orders.values,
            "available_drivers": available_drivers.values,
            "raw_cartesian_pair_count": raw_pairs.values,
            "hour": ((active_orders.index - 1) // 60).astype(int),
        }
    )
    by_hour = (
        by_batch.groupby("hour", as_index=False)
        .agg(
            active_orders_mean=("active_orders", "mean"),
            available_drivers_mean=("available_drivers", "mean"),
            raw_cartesian_pair_count_total=("raw_cartesian_pair_count", "sum"),
            raw_cartesian_pair_count_mean=("raw_cartesian_pair_count", "mean"),
            batch_count=("batch_id", "count"),
        )
        .copy()
    )

    unique_drivers = set(locations["driver_id"].astype(str).unique())
    scored_drivers = set(scores["driver_id"].astype(str).unique()) if not scores.empty else set()
    coverage = (len(unique_drivers & scored_drivers) / len(unique_drivers) * 100.0) if unique_drivers else 0.0

    profile = {
        "data_mode": data_mode,
        "simulation_date": simulation_date,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "number_of_orders": int(len(orders)),
        "number_of_unique_customers": int(orders["customer_id"].astype(str).nunique()) if not orders.empty else 0,
        "number_of_driver_location_rows": int(len(locations)),
        "number_of_unique_drivers": int(len(unique_drivers)),
        "number_of_driver_scores": int(len(scores)),
        "driver_score_coverage_pct": round(float(coverage), 4),
        "number_of_batches": int(batch_end - batch_start + 1),
        "number_of_orders_by_batch_mean": float(active_orders.mean()) if not active_orders.empty else 0.0,
        "number_of_orders_by_batch_median": float(active_orders.median()) if not active_orders.empty else 0.0,
        "number_of_orders_by_batch_max": int(active_orders.max()) if not active_orders.empty else 0,
        "number_of_available_drivers_by_batch_mean": float(available_drivers.mean()) if not available_drivers.empty else 0.0,
        "number_of_available_drivers_by_batch_median": float(available_drivers.median()) if not available_drivers.empty else 0.0,
        "number_of_available_drivers_by_batch_max": int(available_drivers.max()) if not available_drivers.empty else 0,
        "raw_cartesian_pair_count_total": int(raw_pairs.sum()),
        "raw_cartesian_pair_count_mean_per_batch": float(raw_pairs.mean()) if not raw_pairs.empty else 0.0,
        "raw_cartesian_pair_count_max_per_batch": int(raw_pairs.max()) if not raw_pairs.empty else 0,
    }

    (output / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    by_batch.to_csv(output / "dataset_profile_by_batch.csv", index=False)
    by_hour.to_csv(output / "dataset_profile_by_hour.csv", index=False)
    return profile
