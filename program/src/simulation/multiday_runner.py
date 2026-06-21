from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import load_config, resolve_project_path
from src.evaluation.research_package import build_research_package
from src.simulation.engine import SimulationEngine


@dataclass
class MultidayRunResult:
    output_dir: Path
    research_package_dir: Path
    dates: list[str]
    summary: pd.DataFrame


def available_actual_dates(config: dict) -> list[str]:
    data_cfg = config.get("data", {})
    orders_path = resolve_project_path(config, data_cfg.get("orders", {}).get("path"))
    locations_path = resolve_project_path(config, data_cfg.get("driver_locations", {}).get("path"))
    if orders_path is None or locations_path is None:
        return []
    orders = pd.read_csv(orders_path, usecols=["jakarta_data_date"])
    locations = pd.read_csv(locations_path, usecols=["jakarta_data_date"])
    order_dates = set(pd.to_datetime(orders["jakarta_data_date"], errors="coerce").dt.date.dropna().astype(str))
    location_dates = set(
        pd.to_datetime(locations["jakarta_data_date"], errors="coerce").dt.date.dropna().astype(str)
    )
    return sorted(order_dates & location_dates)


def run_multiday_actual(
    config_path: str | Path,
    *,
    output_directory: str | Path = "outputs/thesis_actual_multiday",
    start_date: str | None = None,
    end_date: str | None = None,
    max_days: int | None = None,
    scenario_prefix: str = "thesis_actual",
    enable_candidate_edge_logging: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
    progress_interval_batches: int = 60,
) -> MultidayRunResult:
    run_start = time.perf_counter()
    base_config = load_config(config_path)
    dates = available_actual_dates(base_config)
    if start_date:
        dates = [date for date in dates if date >= start_date]
    if end_date:
        dates = [date for date in dates if date <= end_date]
    if max_days is not None:
        dates = dates[: int(max_days)]
    if not dates:
        raise ValueError("No runnable dates found in actual orders and driver_locations files.")

    root = Path(base_config.get("_project_root", ".")).resolve()
    output_dir = resolve_project_path(base_config, str(output_directory))
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_grid_path = resolve_project_path(
        base_config, (base_config.get("data", {}).get("grid_values") or {}).get("path")
    )
    carried_grid_path = initial_grid_path if initial_grid_path and initial_grid_path.exists() else None
    summary_rows: list[dict] = []

    if progress_callback:
        progress_callback(
            {
                "event": "multiday_start",
                "date_count": len(dates),
                "start_date": dates[0],
                "end_date": dates[-1],
                "output_dir": str(output_dir),
                "initial_grid_values_path": str(carried_grid_path) if carried_grid_path else None,
                "candidate_edge_logging": bool(enable_candidate_edge_logging),
                "progress_interval_batches": int(progress_interval_batches),
            }
        )

    for day_index, date in enumerate(dates, start=1):
        day_start = time.perf_counter()
        day_config = copy.deepcopy(base_config)
        day_config["simulation"]["simulation_date"] = date
        day_config["simulation"]["enable_candidate_edge_logging"] = bool(enable_candidate_edge_logging)
        day_config.setdefault("output", {})["base_directory"] = str(output_dir)
        day_config.setdefault("data", {}).setdefault("grid_values", {})["path"] = (
            str(carried_grid_path) if carried_grid_path is not None else None
        )
        scenario_id = f"{scenario_prefix}_{date}"
        if progress_callback:
            progress_callback(
                {
                    "event": "day_start",
                    "day_index": day_index,
                    "day_count": len(dates),
                    "date": date,
                    "scenario_id": scenario_id,
                    "input_grid_values_path": str(day_config["data"]["grid_values"]["path"]),
                    "elapsed_total_seconds": time.perf_counter() - run_start,
                }
            )

        def _batch_progress(record: dict) -> None:
            if not progress_callback:
                return
            current_batch = int(record.get("current_batch") or 0)
            total_batches = int(record.get("total_batches") or 0)
            day_elapsed = time.perf_counter() - day_start
            day_eta = None
            if current_batch > 0 and total_batches > 0:
                day_eta = (day_elapsed / current_batch) * (total_batches - current_batch)
            progress_callback(
                record
                | {
                    "event": "batch_progress",
                    "day_index": day_index,
                    "day_count": len(dates),
                    "date": date,
                    "scenario_id": scenario_id,
                    "day_elapsed_seconds": day_elapsed,
                    "day_eta_seconds": day_eta,
                    "elapsed_total_seconds": time.perf_counter() - run_start,
                }
            )

        try:
            result = SimulationEngine(day_config).run(
                scenario_id=scenario_id,
                progress_callback=_batch_progress if progress_callback else None,
                progress_interval_batches=progress_interval_batches,
            )
        except Exception as exc:
            if progress_callback:
                progress_callback(
                    {
                        "event": "day_failed",
                        "day_index": day_index,
                        "day_count": len(dates),
                        "date": date,
                        "scenario_id": scenario_id,
                        "elapsed_total_seconds": time.perf_counter() - run_start,
                        "day_runtime_seconds": time.perf_counter() - day_start,
                        "error_message": str(exc),
                    }
                )
            raise
        final_grid_path = result.output_dir / "grid_values_final.csv"
        carried_grid_path = final_grid_path
        row = dict(result.summary)
        row["jakarta_data_date"] = date
        row["input_grid_values_path"] = str(day_config["data"]["grid_values"]["path"])
        row["output_grid_values_path"] = str(final_grid_path)
        row["output_dir"] = str(result.output_dir)
        summary_rows.append(row)
        if progress_callback:
            progress_callback(
                {
                    "event": "day_success",
                    "day_index": day_index,
                    "day_count": len(dates),
                    "date": date,
                    "scenario_id": scenario_id,
                    "elapsed_total_seconds": time.perf_counter() - run_start,
                    "day_runtime_seconds": time.perf_counter() - day_start,
                    "total_orders": row.get("total_orders"),
                    "matched_orders": row.get("matched_orders"),
                    "match_rate": row.get("match_rate"),
                    "gini_coefficient": row.get("gini_coefficient"),
                    "output_dir": str(result.output_dir),
                    "output_grid_values_path": str(final_grid_path),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "multiday_summary.csv", index=False)
    metadata = pd.DataFrame(
        [
            {
                "project_root": str(root),
                "config_path": str(config_path),
                "date_count": len(dates),
                "start_date": dates[0],
                "end_date": dates[-1],
                "grid_carryover": True,
                "candidate_edge_logging": bool(enable_candidate_edge_logging),
            }
        ]
    )
    metadata.to_csv(output_dir / "multiday_run_metadata.csv", index=False)
    if progress_callback:
        progress_callback(
            {
                "event": "research_package_start",
                "date_count": len(dates),
                "elapsed_total_seconds": time.perf_counter() - run_start,
                "output_dir": str(output_dir),
            }
        )
    package_start = time.perf_counter()
    research_package_dir = build_research_package(output_dir, base_config, dates)
    if progress_callback:
        progress_callback(
            {
                "event": "multiday_complete",
                "date_count": len(dates),
                "elapsed_total_seconds": time.perf_counter() - run_start,
                "research_package_runtime_seconds": time.perf_counter() - package_start,
                "output_dir": str(output_dir),
                "research_package_dir": str(research_package_dir),
            }
        )
    return MultidayRunResult(
        output_dir=output_dir,
        research_package_dir=research_package_dir,
        dates=dates,
        summary=summary,
    )
