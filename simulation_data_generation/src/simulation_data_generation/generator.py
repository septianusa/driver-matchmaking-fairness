"""End-to-end dataset orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from simulation_data_generation.config import GenerationConfig, load_config
from simulation_data_generation.driver_movement import generate_driver_positions_for_day
from simulation_data_generation.driver_schedule import generate_driver_schedules_for_day
from simulation_data_generation.driver_score_generator import generate_driver_scores
from simulation_data_generation.grid_value_generator import (
    aggregate_driver_cells,
    aggregate_order_cells,
    build_h3_reference,
    generate_grid_values,
)
from simulation_data_generation.io_utils import (
    dataframe_digest,
    prepare_output_directory,
    write_json,
    write_parquet,
    write_partitioned_by_date,
    write_sample_csv,
)
from simulation_data_generation.order_generator import generate_orders_for_day
from simulation_data_generation.poi_generator import generate_pois
from simulation_data_generation.road_network import build_road_network
from simulation_data_generation.temporal_demand import generate_minute_order_counts, simulation_dates

LOGGER = logging.getLogger(__name__)


def _output_paths(config: GenerationConfig, project_root: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    data_dir = output_dir or (project_root / config.output.data_dir)
    sample_dir = project_root / config.output.sample_dir
    report_dir = project_root / config.output.report_dir
    return data_dir.resolve(), sample_dir.resolve(), report_dir.resolve()


def generate_dataset(
    config: GenerationConfig | str | Path,
    *,
    project_root: str | Path | None = None,
    scale: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate the complete dispatch simulation dataset."""
    if isinstance(config, (str, Path)):
        loaded_config = load_config(config, scale=scale)
        config_path = Path(config).resolve()
        root = Path(project_root).resolve() if project_root else config_path.parent.parent.resolve()
    else:
        loaded_config = config
        root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    config = loaded_config
    data_dir, sample_dir, report_dir = _output_paths(config, root, Path(output_dir).resolve() if output_dir else None)
    start_time = time.perf_counter()
    rng = np.random.default_rng(int(config.random_seed))

    LOGGER.info("Starting generation seed=%s data_dir=%s", config.random_seed, data_dir)
    prepare_output_directory(data_dir, overwrite=overwrite)
    sample_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    network = build_road_network(config)
    pois = generate_pois(config, network, np.random.default_rng(config.random_seed + 11))
    driver_scores = generate_driver_scores(config, np.random.default_rng(config.random_seed + 22))
    dates = simulation_dates(config)

    write_parquet(pois, data_dir / "poi_reference.parquet")
    write_parquet(driver_scores, data_dir / "driver_scores.parquet")
    write_sample_csv(pois, sample_dir / "poi_reference_sample.csv", config.output.sample_rows)
    write_sample_csv(driver_scores, sample_dir / "driver_scores_sample.csv", config.output.sample_rows)

    all_order_samples: list[pd.DataFrame] = []
    all_position_samples: list[pd.DataFrame] = []
    order_count_frames: list[pd.DataFrame] = []
    driver_count_frames: list[pd.DataFrame] = []
    observed_cells: set[str] = set(pois["h3_index"].astype(str).tolist())
    daily_counts: list[dict[str, object]] = []

    for offset, simulation_date in enumerate(tqdm(dates, desc="Generating days", leave=False)):
        day_rng = np.random.default_rng(config.random_seed + 1000 + offset)
        minute_counts = generate_minute_order_counts(config, simulation_date, day_rng)
        orders = generate_orders_for_day(config, simulation_date, minute_counts, pois, day_rng)
        schedules = generate_driver_schedules_for_day(config, simulation_date, driver_scores, minute_counts, day_rng)
        positions = generate_driver_positions_for_day(config, schedules, pois, network, day_rng)

        write_partitioned_by_date(orders, data_dir / "orders")
        write_partitioned_by_date(positions, data_dir / "driver_positions")

        all_order_samples.append(orders.head(config.output.sample_rows))
        all_position_samples.append(positions.head(config.output.sample_rows))
        order_count_frames.append(aggregate_order_cells(orders))
        driver_count_frames.append(aggregate_driver_cells(positions))
        observed_cells.update(orders["pickup_h3_index"].astype(str).tolist())
        observed_cells.update(orders["dropoff_h3_index"].astype(str).tolist())
        if not positions.empty:
            observed_cells.update(positions["h3_index"].astype(str).tolist())
        daily_counts.append(
            {
                "simulation_date": simulation_date,
                "orders": int(len(orders)),
                "driver_position_rows": int(len(positions)),
                "active_drivers": int(positions["driver_id"].nunique()) if not positions.empty else 0,
                "online_sessions": int(len(schedules)),
            }
        )
        LOGGER.info(
            "Generated %s orders=%s position_rows=%s active_drivers=%s",
            simulation_date,
            len(orders),
            len(positions),
            positions["driver_id"].nunique() if not positions.empty else 0,
        )

    order_counts = pd.concat(order_count_frames, ignore_index=True) if order_count_frames else pd.DataFrame()
    driver_counts = pd.concat(driver_count_frames, ignore_index=True) if driver_count_frames else pd.DataFrame()
    grid_reference = build_h3_reference(observed_cells, config)
    grid_values = generate_grid_values(config, grid_reference, order_counts, driver_counts, dates)
    write_parquet(grid_reference, data_dir / "h3_grid_reference.parquet")
    write_partitioned_by_date(grid_values, data_dir / "h3_grid_values")

    orders_sample = pd.concat(all_order_samples, ignore_index=True).head(config.output.sample_rows)
    positions_sample = pd.concat(all_position_samples, ignore_index=True).head(config.output.sample_rows)
    write_sample_csv(orders_sample, sample_dir / "orders_sample.csv", config.output.sample_rows)
    write_sample_csv(positions_sample, sample_dir / "driver_positions_sample.csv", config.output.sample_rows)
    write_sample_csv(grid_values, sample_dir / "h3_grid_values_sample.csv", config.output.sample_rows)

    elapsed = time.perf_counter() - start_time
    manifest = {
        "dataset_type": "generated_surabaya_dispatch_profile",
        "description": "Generated dispatch research dataset for simulation and assignment experiments.",
        "design_profile": {
            "daily_order_target": int(config.daily_order_target),
            "daily_order_band": [config.target_daily_order_min, config.target_daily_order_max],
            "registered_drivers": int(config.number_of_drivers),
            "active_driver_band": [config.active_driver_target_min, config.active_driver_target_max],
            "maximum_online_minutes_per_driver_day": int(config.maximum_online_hours_per_day * 60),
        },
        "seed": int(config.random_seed),
        "study_area_name": config.study_area_name,
        "timezone": config.timezone,
        "simulation_dates": dates,
        "road_network_source_type": network.source_type,
        "output_data_dir": str(data_dir),
        "sample_dir": str(sample_dir),
        "row_counts": {
            "poi_reference": int(len(pois)),
            "driver_scores": int(len(driver_scores)),
            "orders_total": int(sum(item["orders"] for item in daily_counts)),
            "driver_positions_total": int(sum(item["driver_position_rows"] for item in daily_counts)),
            "orders_sample": int(len(orders_sample)),
            "driver_positions_sample": int(len(positions_sample)),
            "h3_grid_reference": int(len(grid_reference)),
            "h3_grid_values": int(len(grid_values)),
        },
        "daily_counts": daily_counts,
        "digests": {
            "poi_reference": dataframe_digest(pois),
            "driver_scores": dataframe_digest(driver_scores),
            "orders_sample": dataframe_digest(orders_sample),
            "driver_positions_sample": dataframe_digest(positions_sample),
            "h3_grid_values_sample": dataframe_digest(grid_values),
        },
        "elapsed_seconds": round(float(elapsed), 3),
    }
    write_json(data_dir / "generation_manifest.json", manifest)
    return manifest
