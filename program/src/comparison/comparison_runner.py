from __future__ import annotations

import copy
import html
import json
import platform
import re
import shutil
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from src.comparison.comparison_case import ComparisonCase, expand_comparison_cases
from src.comparison.comparison_metrics import (
    algorithm_quality,
    baseline_relative,
    enrich_run_metrics,
    sparse_method_quality,
    summarize_runs,
)
from src.comparison.comparison_report import write_comparison_report
from src.comparison.dataset_profiler import build_dataset_profile
from src.config import deep_update, load_config, resolve_project_path, yaml_dump
from src.evaluation.report import export_zip
from src.simulation.engine import SimulationEngine
from src.sparse.a2gat_adapter import A2GATCandidateProvider
from src.validation import validate_inputs


@dataclass
class ComparisonRunResult:
    run_id: str
    output_dir: Path
    dataset_profile: dict
    summary: pd.DataFrame
    raw_runs: pd.DataFrame


@dataclass
class MultidayComparisonRunResult:
    run_id: str
    output_dir: Path
    dates: list[str]
    dataset_profiles: pd.DataFrame
    summary_by_day: pd.DataFrame
    summary_across_days: pd.DataFrame
    raw_runs: pd.DataFrame


def _now_run_id() -> str:
    return "model_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_run_label(label: str | None) -> str:
    if not label:
        return ""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip()).strip("_").lower()
    return safe


def _now_multiday_run_id(label: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = _safe_run_label(label)
    if safe_label:
        return f"model_comparison_days_{safe_label}_{stamp}"
    return f"model_comparison_days_{stamp}"


def _base_config(project_root: Path) -> dict:
    return load_config(project_root / "configs" / "default.yaml")


def _ensure_project_root(config: dict) -> dict:
    root = Path(config.get("_project_root", "."))
    if not (root / "configs" / "default.yaml").exists():
        config["_project_root"] = str(Path(__file__).resolve().parents[2])
    return config


def _comparison_output_dir(config: dict, run_id: str) -> Path:
    root = Path(config.get("_project_root", "."))
    return root / "outputs" / run_id / "comparison"


def _copy_or_empty_grid(source: Path | None, target: Path) -> None:
    if source and source.exists():
        shutil.copyfile(source, target)
    else:
        pd.DataFrame(columns=["h3_index", "grid_value", "latitude", "longitude"]).to_csv(target, index=False)


def _apply_runtime_overrides(base: dict, comparison_config: dict) -> dict:
    """Apply scenario-level runtime overrides from the comparison YAML."""
    override_sections = [
        "cancellation_model",
        "data_quality",
        "driver_scores",
        "drivers",
        "eta",
        "grid_values",
        "matching",
        "orders",
        "spatial",
        "sparse",
        "utility",
    ]
    resolved = copy.deepcopy(base)
    for section in override_sections:
        if section in comparison_config:
            resolved[section] = deep_update(resolved.get(section, {}), comparison_config[section])
    return resolved


def _available_dates_from_data_paths(config: dict) -> list[str]:
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


def _available_comparison_dates(comparison_config: dict, data_mode: str) -> list[str]:
    if data_mode != "actual":
        raise ValueError(f"Unsupported comparison data_mode: {data_mode}")
    project_root = Path(comparison_config.get("_project_root", ".")).resolve()
    base = _base_config(project_root)
    actual = comparison_config.get("data", {}).get("actual", {})
    date_config = copy.deepcopy(base)
    date_config["data"]["orders"]["path"] = actual.get("orders_path") or base["data"]["orders"]["path"]
    date_config["data"]["driver_locations"]["path"] = (
        actual.get("driver_locations_path") or base["data"]["driver_locations"]["path"]
    )
    return _available_dates_from_data_paths(date_config)


def _filter_dates(
    dates: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    max_days: int | None = None,
) -> list[str]:
    selected = list(dates)
    if start_date:
        selected = [date for date in selected if date >= start_date]
    if end_date:
        selected = [date for date in selected if date <= end_date]
    if max_days is not None:
        selected = selected[: int(max_days)]
    return selected


def _build_snapshot_config(
    comparison_config: dict,
    *,
    data_mode: str,
    run_id: str,
) -> tuple[dict, Path]:
    project_root = Path(comparison_config.get("_project_root", ".")).resolve()
    base = _base_config(project_root)
    base = _apply_runtime_overrides(base, comparison_config)
    base["simulation"]["random_seed"] = int(
        comparison_config.get("comparison", {}).get("random_seed", base["simulation"].get("random_seed", 42))
    )
    snapshot_raw = _comparison_output_dir(comparison_config, run_id) / "snapshot" / "raw"
    snapshot_raw.mkdir(parents=True, exist_ok=True)

    if data_mode == "actual":
        actual = comparison_config.get("data", {}).get("actual", {})
        if actual.get("simulation_date"):
            base["simulation"]["simulation_date"] = str(actual["simulation_date"])
        path_map = {
            "orders": actual.get("orders_path") or base["data"]["orders"]["path"],
            "driver_locations": actual.get("driver_locations_path") or base["data"]["driver_locations"]["path"],
            "driver_scores": actual.get("driver_scores_path") or base["data"]["driver_scores"]["path"],
        }
        for key, rel_path in path_map.items():
            source = resolve_project_path(base, rel_path)
            if source is None or not source.exists():
                raise FileNotFoundError(f"Actual data file not found for {key}: {rel_path}")
            target_name = {
                "orders": "orders.csv",
                "driver_locations": "driver_locations.csv",
                "driver_scores": "driver_scores.csv",
            }[key]
            shutil.copyfile(source, snapshot_raw / target_name)
        grid_source = resolve_project_path(base, actual.get("grid_values_path"))
        _copy_or_empty_grid(grid_source, snapshot_raw / "grid_values.csv")
    else:
        raise ValueError(f"Unsupported comparison data_mode: {data_mode}")

    base["data"]["orders"]["path"] = str(snapshot_raw / "orders.csv")
    base["data"]["driver_locations"]["path"] = str(snapshot_raw / "driver_locations.csv")
    base["data"]["driver_scores"]["path"] = str(snapshot_raw / "driver_scores.csv")
    base["data"]["grid_values"]["path"] = str(snapshot_raw / "grid_values.csv")
    base.setdefault("experiment", {})
    if "batch_start" in comparison_config.get("comparison", {}):
        base["experiment"]["batch_start"] = int(comparison_config["comparison"]["batch_start"])
    if "batch_end" in comparison_config.get("comparison", {}):
        base["experiment"]["batch_end"] = int(comparison_config["comparison"]["batch_end"])
    return base, snapshot_raw


def dry_run(config_path: str | Path, data_mode_override: str | None = None) -> dict:
    config = _ensure_project_root(load_config(config_path))
    data_mode = data_mode_override or config.get("comparison", {}).get("data_mode", "actual")
    cases = expand_comparison_cases(config)
    run_id = "dry_run"
    output_dir = _comparison_output_dir(config, run_id)
    a2gat_available, a2gat_reason = A2GATCandidateProvider().availability_status()
    return {
        "data_mode": data_mode,
        "variant_count": len(cases),
        "measured_runs": len(cases) * int(config.get("comparison", {}).get("repeats", 3)),
        "warmup_runs": len(cases) * int(config.get("comparison", {}).get("warmup_runs", 1)),
        "matching_algorithms": config.get("comparison", {}).get("matching_algorithms", []),
        "sparse_methods": config.get("comparison", {}).get("sparse_methods", []),
        "sparse_integration_status": {"a2gat": {"available": a2gat_available, "message": a2gat_reason}},
        "unavailable_optional_integrations": {} if a2gat_available else {"a2gat": a2gat_reason},
        "selected_dataset_paths": config.get("data", {}).get(data_mode, {}),
        "output_path": str(output_dir),
    }


def _variant_skip_reason(case: ComparisonCase, config: dict) -> str | None:
    if case.sparse_method == "a2gat":
        available, reason = A2GATCandidateProvider().availability_status()
        return None if available else reason
    if case.sparse_method != "bfs_h3":
        return f"Unsupported sparse method: {case.sparse_method}"
    if case.matching_algorithm == "auction" and not bool(
        config.get("matching", {}).get("auction", {}).get("enabled", True)
    ):
        return "Auction matching is disabled."
    return None


def _result_to_metrics(
    *,
    result,
    case: ComparisonCase,
    repeat_index: int,
    warmup_flag: bool,
    data_mode: str,
    dataset_profile: dict,
    runtime_seconds: float,
    peak_memory_mb: float,
    cpu_mean: float | None,
    cpu_peak: float | None,
) -> dict:
    batch = result.batch_metrics
    matches = result.match_log
    order_events = result.order_events
    expired_count = (
        int(order_events[order_events["final_order_status"] == "expired_unfulfilled"]["order_id"].nunique())
        if not order_events.empty and "final_order_status" in order_events
        else 0
    )
    row = {
        "variant_id": case.variant_id,
        "repeat_index": repeat_index,
        "warmup_flag": bool(warmup_flag),
        "lambda_driver_score": case.lambda_driver_score,
        "matching_algorithm": case.matching_algorithm,
        "sparse_method": case.sparse_method,
        "data_mode": data_mode,
        "simulation_date": dataset_profile["simulation_date"],
        "batch_start": dataset_profile["batch_start"],
        "batch_end": dataset_profile["batch_end"],
        "number_of_orders": dataset_profile["number_of_orders"],
        "number_of_unique_drivers": dataset_profile["number_of_unique_drivers"],
        "number_of_batches": dataset_profile["number_of_batches"],
        "raw_cartesian_pair_count_total": dataset_profile["raw_cartesian_pair_count_total"],
        "candidate_pair_count_before_filter_total": int(batch.get("candidate_pair_count_before_filter", pd.Series(dtype=float)).sum()) if not batch.empty else 0,
        "candidate_pair_count_after_filter_total": int(batch.get("candidate_pair_count_after_filter", pd.Series(dtype=float)).sum()) if not batch.empty else 0,
        "data_preparation_runtime_seconds": 0.0,
        "candidate_generation_runtime_seconds": float(batch.get("candidate_generation_runtime_seconds", pd.Series(dtype=float)).sum()) if not batch.empty else 0.0,
        "haversine_filter_runtime_seconds": float(batch.get("haversine_filter_runtime_seconds", pd.Series(dtype=float)).sum()) if not batch.empty else 0.0,
        "edge_feature_runtime_seconds": float(batch.get("edge_feature_runtime_seconds", pd.Series(dtype=float)).sum()) if not batch.empty else 0.0,
        "solver_runtime_seconds": float(batch.get("solver_runtime_seconds", pd.Series(dtype=float)).sum()) if not batch.empty else 0.0,
        "state_update_runtime_seconds": float(batch.get("state_update_runtime_seconds", pd.Series(dtype=float)).sum()) if not batch.empty else 0.0,
        "end_to_end_runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
        "cpu_percent_mean": cpu_mean,
        "cpu_percent_peak": cpu_peak,
        "matched_order_count": result.summary.get("matched_orders", 0),
        "expected_completed_order_count": result.summary.get("expected_completed_orders", 0.0),
        "match_rate": result.summary.get("match_rate", 0.0),
        "expected_conversion_rate": result.summary.get("expected_conversion_rate", 0.0),
        "expired_unfulfilled_order_count": expired_count,
        "expired_unfulfilled_rate": expired_count / dataset_profile["number_of_orders"] if dataset_profile["number_of_orders"] else 0.0,
        "assignment_weight_total": float(matches.get("final_matching_weight", pd.Series(dtype=float)).sum()) if not matches.empty else 0.0,
        "expected_economic_utility_total": result.summary.get("total_expected_economic_utility", 0.0),
        "median_pickup_distance_km": result.summary.get("median_pickup_distance_km", 0.0),
        "mean_pickup_distance_km": float(matches.get("pickup_distance_km", pd.Series(dtype=float)).mean()) if not matches.empty else 0.0,
        "mean_predicted_cancellation_probability": float(matches.get("predicted_cancellation_probability", pd.Series(dtype=float)).mean()) if not matches.empty else 0.0,
        "pearson_driver_score_income": result.summary.get("pearson_score_income_correlation", 0.0),
        "spearman_driver_score_income": result.summary.get("spearman_score_income_correlation", 0.0),
        "income_gini": result.summary.get("gini_coefficient", 0.0),
        "status": "success",
        "skip_reason": "",
        "error_message": "",
    }
    return enrich_run_metrics(row)


def _empty_run_metrics(
    *,
    case: ComparisonCase,
    repeat_index: int,
    warmup_flag: bool,
    data_mode: str,
    dataset_profile: dict,
    status: str,
    skip_reason: str = "",
    error_message: str = "",
) -> dict:
    return enrich_run_metrics(
        {
            "variant_id": case.variant_id,
            "repeat_index": repeat_index,
            "warmup_flag": bool(warmup_flag),
            "lambda_driver_score": case.lambda_driver_score,
            "matching_algorithm": case.matching_algorithm,
            "sparse_method": case.sparse_method,
            "data_mode": data_mode,
            "simulation_date": dataset_profile["simulation_date"],
            "batch_start": dataset_profile["batch_start"],
            "batch_end": dataset_profile["batch_end"],
            "number_of_orders": dataset_profile["number_of_orders"],
            "number_of_unique_drivers": dataset_profile["number_of_unique_drivers"],
            "number_of_batches": dataset_profile["number_of_batches"],
            "raw_cartesian_pair_count_total": dataset_profile["raw_cartesian_pair_count_total"],
            "candidate_pair_count_before_filter_total": 0,
            "candidate_pair_count_after_filter_total": 0,
            "status": status,
            "skip_reason": skip_reason,
            "error_message": error_message,
        }
    )


def _summarize_runs_by_date(raw_runs: pd.DataFrame) -> pd.DataFrame:
    if raw_runs.empty:
        return pd.DataFrame()
    frames = []
    for date, group in raw_runs.groupby("simulation_date", sort=True, dropna=False):
        summary = summarize_runs(group)
        if summary.empty:
            continue
        summary.insert(0, "simulation_date", date)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _baseline_relative_by_date(summary_by_day: pd.DataFrame) -> pd.DataFrame:
    if summary_by_day.empty:
        return pd.DataFrame()
    frames = []
    for date, group in summary_by_day.groupby("simulation_date", sort=True, dropna=False):
        comparison = baseline_relative(group.drop(columns=["simulation_date"]))
        if comparison.empty:
            continue
        comparison.insert(0, "simulation_date", date)
        frames.append(comparison)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _append_checkpoint_row(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_checkpoint_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_checkpoint_summaries(
    output_dir: Path,
    *,
    raw_rows: list[dict],
    run_counter: int,
    total_runs: int,
    last_event: dict,
) -> None:
    raw_runs = pd.DataFrame(raw_rows)
    if not raw_runs.empty:
        _summarize_runs_by_date(raw_runs).to_csv(
            output_dir / "comparison_summary_by_day_checkpoint.csv", index=False
        )
        summarize_runs(raw_runs).to_csv(
            output_dir / "comparison_summary_across_days_checkpoint.csv", index=False
        )
    state = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_counter": int(run_counter),
        "total_runs": int(total_runs),
        "completed_pct": (float(run_counter) / float(total_runs)) if total_runs else 0.0,
        "last_event": last_event,
        "checkpoint_files": {
            "raw_runs": str(output_dir / "comparison_raw_runs_all_days_checkpoint.csv"),
            "batch_metrics": str(output_dir / "comparison_batch_metrics_all_days_checkpoint.csv"),
            "grid_carryover_audit": str(output_dir / "comparison_grid_carryover_audit_checkpoint.csv"),
            "summary_by_day": str(output_dir / "comparison_summary_by_day_checkpoint.csv"),
            "summary_across_days": str(output_dir / "comparison_summary_across_days_checkpoint.csv"),
        },
    }
    (output_dir / "checkpoint_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _write_multiday_comparison_report(
    output_dir: Path,
    *,
    dates: list[str],
    dataset_profiles: pd.DataFrame,
    summary_by_day: pd.DataFrame,
    summary_across_days: pd.DataFrame,
    raw_runs: pd.DataFrame,
    grid_carryover: bool,
) -> None:
    def _table(df: pd.DataFrame, limit: int = 30) -> str:
        if df.empty:
            return "<p>No rows.</p>"
        return df.head(limit).to_html(index=False, escape=True, classes="data")

    total_runs = len(raw_runs)
    success = int((raw_runs.get("status", pd.Series(dtype=str)) == "success").sum()) if total_runs else 0
    failed = int((raw_runs.get("status", pd.Series(dtype=str)) == "failed").sum()) if total_runs else 0
    skipped = int((raw_runs.get("status", pd.Series(dtype=str)) == "skipped").sum()) if total_runs else 0
    best_cols = [
        "lambda_driver_score",
        "matching_algorithm",
        "sparse_method",
        "match_rate_mean",
        "expected_conversion_rate_mean",
        "median_pickup_distance_km_mean",
        "spearman_driver_score_income_mean",
        "income_gini_mean",
        "end_to_end_runtime_seconds_mean",
    ]
    best = summary_across_days[[col for col in best_cols if col in summary_across_days.columns]].copy()
    if not best.empty and "spearman_driver_score_income_mean" in best.columns:
        best = best.sort_values(
            ["spearman_driver_score_income_mean", "match_rate_mean"],
            ascending=[False, False],
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>All-days model comparison report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    .meta {{ color: #52606d; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; }}
    .metric {{ font-size: 24px; font-weight: 700; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    table.data th, table.data td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    table.data th:first-child, table.data td:first-child {{ text-align: left; }}
    code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>All-days model comparison report</h1>
  <p class="meta">Dates: {html.escape(', '.join(dates))}. Grid carry-over: {grid_carryover}.</p>
  <div class="cards">
    <div class="card"><div>Total runs</div><div class="metric">{total_runs}</div></div>
    <div class="card"><div>Successful</div><div class="metric">{success}</div></div>
    <div class="card"><div>Skipped</div><div class="metric">{skipped}</div></div>
    <div class="card"><div>Failed</div><div class="metric">{failed}</div></div>
  </div>
  <h2>Best variants by behavior-income alignment</h2>
  {_table(best)}
  <h2>Across-day scenario summary</h2>
  {_table(summary_across_days)}
  <h2>Dataset profile by day</h2>
  {_table(dataset_profiles)}
  <h2>Summary by day and scenario</h2>
  {_table(summary_by_day, limit=80)}
  <p class="meta">Full CSV outputs are in <code>{html.escape(str(output_dir))}</code>.</p>
</body>
</html>
"""
    (output_dir / "comparison_multiday_report.html").write_text(html_text, encoding="utf-8")


def run_model_comparison(
    config_path: str | Path,
    *,
    data_mode_override: str | None = None,
    dry_run_only: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
    run_id: str | None = None,
) -> ComparisonRunResult | dict:
    if dry_run_only:
        return dry_run(config_path, data_mode_override)
    comparison_config = _ensure_project_root(load_config(config_path))
    data_mode = data_mode_override or comparison_config.get("comparison", {}).get("data_mode", "actual")
    run_id = run_id or _now_run_id()
    output_dir = _comparison_output_dir(comparison_config, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = expand_comparison_cases(comparison_config)
    if progress_callback:
        progress_callback(
            {
                "event": "setup_start",
                "run_id": run_id,
                "data_mode": data_mode,
                "variant_count": len(cases),
                "output_dir": str(output_dir),
            }
        )
    snapshot_config, _snapshot_raw = _build_snapshot_config(comparison_config, data_mode=data_mode, run_id=run_id)
    validation = validate_inputs(snapshot_config)
    validation.write(output_dir / "validation_report.json")
    if not validation.ok:
        raise ValueError("Comparison snapshot validation failed: " + "; ".join(validation.blocking_errors))
    dataset_profile = build_dataset_profile(snapshot_config, data_mode=data_mode, output_dir=output_dir)
    if progress_callback:
        progress_callback(
            {
                "event": "dataset_profile_ready",
                "run_id": run_id,
                "number_of_orders": dataset_profile["number_of_orders"],
                "number_of_unique_drivers": dataset_profile["number_of_unique_drivers"],
                "raw_cartesian_pair_count_total": dataset_profile["raw_cartesian_pair_count_total"],
            }
        )
    (output_dir / "comparison_config_resolved.yaml").write_text(
        yaml_dump(comparison_config), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "comparison_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    repeats = int(comparison_config.get("comparison", {}).get("repeats", 3))
    warmups = int(comparison_config.get("comparison", {}).get("warmup_runs", 1))
    measure_memory = bool(comparison_config.get("comparison", {}).get("measure_memory", True))
    measure_cpu = bool(comparison_config.get("comparison", {}).get("measure_cpu", True))
    process = None
    if measure_cpu:
        try:
            import psutil  # type: ignore

            process = psutil.Process()
        except Exception:
            process = None
    raw_rows: list[dict] = []
    batch_rows: list[pd.DataFrame] = []
    total_runs = len(cases) * (repeats + warmups)
    run_counter = 0
    for case in cases:
        skip_reason = _variant_skip_reason(case, comparison_config)
        for repeat in range(warmups + repeats):
            warmup_flag = repeat < warmups
            run_counter += 1
            if progress_callback:
                progress_callback(
                    {
                        "event": "run_start",
                        "variant_id": case.variant_id,
                        "lambda_driver_score": case.lambda_driver_score,
                        "matching_algorithm": case.matching_algorithm,
                        "sparse_method": case.sparse_method,
                        "repeat_index": repeat,
                        "warmup_flag": warmup_flag,
                        "run_counter": run_counter,
                        "total_runs": total_runs,
                    }
                )
            if skip_reason:
                raw_rows.append(
                    enrich_run_metrics(
                        {
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "lambda_driver_score": case.lambda_driver_score,
                            "matching_algorithm": case.matching_algorithm,
                            "sparse_method": case.sparse_method,
                            "data_mode": data_mode,
                            "simulation_date": dataset_profile["simulation_date"],
                            "batch_start": dataset_profile["batch_start"],
                            "batch_end": dataset_profile["batch_end"],
                            "number_of_orders": dataset_profile["number_of_orders"],
                            "number_of_unique_drivers": dataset_profile["number_of_unique_drivers"],
                            "number_of_batches": dataset_profile["number_of_batches"],
                            "raw_cartesian_pair_count_total": dataset_profile["raw_cartesian_pair_count_total"],
                            "candidate_pair_count_before_filter_total": 0,
                            "candidate_pair_count_after_filter_total": 0,
                            "status": "skipped",
                            "skip_reason": skip_reason,
                            "error_message": "",
                        }
                    )
                )
                if progress_callback:
                    progress_callback(
                        {
                            "event": "run_skipped",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "run_counter": run_counter,
                            "total_runs": total_runs,
                            "skip_reason": skip_reason,
                        }
                    )
                continue
            variant_config = copy.deepcopy(snapshot_config)
            variant_config["utility"]["lambda_driver_score"] = case.lambda_driver_score
            variant_config["matching"]["strategy"] = case.matching_algorithm
            variant_config["matching"]["auction"] = comparison_config.get("matching", {}).get("auction", {})
            variant_config.setdefault("sparse", {})["method"] = case.sparse_method
            try:
                if measure_memory:
                    tracemalloc.start()
                if process is not None:
                    process.cpu_percent(interval=None)
                start = time.perf_counter()
                result = SimulationEngine(variant_config).run(
                    scenario_id=f"{run_id}_{case.variant_id}_repeat{repeat}",
                    write_outputs=False,
                )
                runtime = time.perf_counter() - start
                if measure_memory:
                    _current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mb = peak / (1024 * 1024)
                else:
                    peak_mb = 0.0
                cpu_value = process.cpu_percent(interval=None) if process is not None else None
                row = _result_to_metrics(
                    result=result,
                    case=case,
                    repeat_index=repeat,
                    warmup_flag=warmup_flag,
                    data_mode=data_mode,
                    dataset_profile=dataset_profile,
                    runtime_seconds=runtime,
                    peak_memory_mb=peak_mb,
                    cpu_mean=cpu_value,
                    cpu_peak=cpu_value,
                )
                raw_rows.append(row)
                if progress_callback:
                    progress_callback(
                        {
                            "event": "run_success",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "run_counter": run_counter,
                            "total_runs": total_runs,
                            "runtime_seconds": runtime,
                            "match_rate": row.get("match_rate"),
                            "candidate_pair_count_after_filter_total": row.get(
                                "candidate_pair_count_after_filter_total"
                            ),
                        }
                    )
                if not result.batch_metrics.empty:
                    batch_frame = result.batch_metrics.copy()
                    batch_frame["variant_id"] = case.variant_id
                    batch_frame["repeat_index"] = repeat
                    batch_frame["warmup_flag"] = warmup_flag
                    dataset_by_batch = pd.read_csv(output_dir / "dataset_profile_by_batch.csv")
                    batch_frame = batch_frame.merge(
                        dataset_by_batch[["batch_id", "raw_cartesian_pair_count"]],
                        on="batch_id",
                        how="left",
                    )
                    batch_rows.append(batch_frame)
            except Exception as exc:
                if measure_memory and tracemalloc.is_tracing():
                    tracemalloc.stop()
                raw_rows.append(
                    enrich_run_metrics(
                        {
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "lambda_driver_score": case.lambda_driver_score,
                            "matching_algorithm": case.matching_algorithm,
                            "sparse_method": case.sparse_method,
                            "data_mode": data_mode,
                            "simulation_date": dataset_profile["simulation_date"],
                            "batch_start": dataset_profile["batch_start"],
                            "batch_end": dataset_profile["batch_end"],
                            "number_of_orders": dataset_profile["number_of_orders"],
                            "number_of_unique_drivers": dataset_profile["number_of_unique_drivers"],
                            "number_of_batches": dataset_profile["number_of_batches"],
                            "raw_cartesian_pair_count_total": dataset_profile["raw_cartesian_pair_count_total"],
                            "candidate_pair_count_before_filter_total": 0,
                            "candidate_pair_count_after_filter_total": 0,
                            "status": "failed",
                            "skip_reason": "",
                            "error_message": str(exc),
                        }
                    )
                )
                if progress_callback:
                    progress_callback(
                        {
                            "event": "run_failed",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "run_counter": run_counter,
                            "total_runs": total_runs,
                            "error_message": str(exc),
                        }
                    )
    raw_runs = pd.DataFrame(raw_rows)
    batch_metrics = pd.concat(batch_rows, ignore_index=True) if batch_rows else pd.DataFrame()
    summary = summarize_runs(raw_runs)
    baseline = baseline_relative(summary)
    algorithm = algorithm_quality(summary)
    sparse = sparse_method_quality(summary)

    raw_runs.to_csv(output_dir / "comparison_raw_runs.csv", index=False)
    batch_metrics.to_csv(output_dir / "comparison_batch_metrics.csv", index=False)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    baseline.to_csv(output_dir / "comparison_baseline_relative.csv", index=False)
    algorithm.to_csv(output_dir / "comparison_algorithm_quality.csv", index=False)
    sparse.to_csv(output_dir / "comparison_sparse_method_quality.csv", index=False)
    raw_runs[raw_runs["status"] == "skipped"].to_csv(output_dir / "comparison_skipped_variants.csv", index=False)
    raw_runs[raw_runs["status"] == "failed"].to_csv(output_dir / "comparison_errors.csv", index=False)
    write_comparison_report(output_dir, dataset_profile=dataset_profile, summary=summary, raw_runs=raw_runs, batch_metrics=batch_metrics)
    if progress_callback:
        progress_callback(
            {
                "event": "comparison_complete",
                "run_id": run_id,
                "output_dir": str(output_dir),
                "successful_runs": int((raw_runs["status"] == "success").sum()),
                "skipped_runs": int((raw_runs["status"] == "skipped").sum()),
                "failed_runs": int((raw_runs["status"] == "failed").sum()),
            }
        )
    return ComparisonRunResult(run_id, output_dir, dataset_profile, summary, raw_runs)


def dry_run_model_comparison_days(
    config_path: str | Path,
    *,
    data_mode_override: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_days: int | None = None,
) -> dict:
    comparison_config = _ensure_project_root(load_config(config_path))
    data_mode = data_mode_override or comparison_config.get("comparison", {}).get("data_mode", "actual")
    cases = expand_comparison_cases(comparison_config)
    dates = _filter_dates(
        _available_comparison_dates(comparison_config, data_mode),
        start_date=start_date,
        end_date=end_date,
        max_days=max_days,
    )
    repeats = int(comparison_config.get("comparison", {}).get("repeats", 3))
    warmups = int(comparison_config.get("comparison", {}).get("warmup_runs", 1))
    return {
        "data_mode": data_mode,
        "date_count": len(dates),
        "dates": dates,
        "variant_count": len(cases),
        "runs_per_variant_day": repeats + warmups,
        "measured_runs": len(cases) * len(dates) * repeats,
        "warmup_runs": len(cases) * len(dates) * warmups,
        "total_runs": len(cases) * len(dates) * (repeats + warmups),
        "matching_algorithms": comparison_config.get("comparison", {}).get("matching_algorithms", []),
        "sparse_methods": comparison_config.get("comparison", {}).get("sparse_methods", []),
        "lambda_driver_score_values": comparison_config.get("comparison", {}).get(
            "lambda_driver_score_values", []
        ),
    }


def run_model_comparison_days(
    config_path: str | Path,
    *,
    data_mode_override: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_days: int | None = None,
    dry_run_only: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
    run_id: str | None = None,
    grid_carryover: bool = True,
    enable_candidate_edge_logging: bool = False,
    progress_interval_batches: int = 0,
) -> MultidayComparisonRunResult | dict:
    if dry_run_only:
        return dry_run_model_comparison_days(
            config_path,
            data_mode_override=data_mode_override,
            start_date=start_date,
            end_date=end_date,
            max_days=max_days,
        )

    comparison_config = _ensure_project_root(load_config(config_path))
    data_mode = data_mode_override or comparison_config.get("comparison", {}).get("data_mode", "actual")
    if data_mode != "actual":
        raise ValueError("compare-models-days is intended for actual data.")

    run_id = run_id or _now_multiday_run_id(comparison_config.get("comparison", {}).get("run_label"))
    output_dir = _comparison_output_dir(comparison_config, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = expand_comparison_cases(comparison_config)
    repeats = int(comparison_config.get("comparison", {}).get("repeats", 3))
    warmups = int(comparison_config.get("comparison", {}).get("warmup_runs", 1))
    measure_memory = bool(comparison_config.get("comparison", {}).get("measure_memory", True))
    measure_cpu = bool(comparison_config.get("comparison", {}).get("measure_cpu", True))

    if progress_callback:
        progress_callback(
            {
                "event": "multiday_comparison_setup_start",
                "run_id": run_id,
                "data_mode": data_mode,
                "variant_count": len(cases),
                "repeats": repeats,
                "warmup_runs": warmups,
                "output_dir": str(output_dir),
                "checkpoint_state_path": str(output_dir / "checkpoint_state.json"),
                "grid_carryover": bool(grid_carryover),
                "candidate_edge_logging": bool(enable_candidate_edge_logging),
            }
        )

    snapshot_config, _snapshot_raw = _build_snapshot_config(comparison_config, data_mode=data_mode, run_id=run_id)
    snapshot_config["simulation"]["enable_candidate_edge_logging"] = bool(enable_candidate_edge_logging)
    dates = _filter_dates(
        _available_dates_from_data_paths(snapshot_config),
        start_date=start_date,
        end_date=end_date,
        max_days=max_days,
    )
    if not dates:
        raise ValueError("No runnable dates found in actual orders and driver_locations files.")

    total_runs = len(cases) * len(dates) * (repeats + warmups)
    if progress_callback:
        progress_callback(
            {
                "event": "multiday_comparison_plan_ready",
                "run_id": run_id,
                "date_count": len(dates),
                "dates": dates,
                "variant_count": len(cases),
                "total_runs": total_runs,
            }
        )

    (output_dir / "comparison_config_resolved.yaml").write_text(
        yaml_dump(comparison_config), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "comparison_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "data_mode": data_mode,
                "dates": dates,
                "variant_count": len(cases),
                "repeats": repeats,
                "warmup_runs": warmups,
                "total_runs": total_runs,
                "grid_carryover": bool(grid_carryover),
                "candidate_edge_logging": bool(enable_candidate_edge_logging),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    day_configs: dict[str, dict] = {}
    dataset_profiles_by_date: dict[str, dict] = {}
    dataset_by_batch_by_date: dict[str, pd.DataFrame] = {}
    validation_dir = output_dir / "validation_by_day"
    validation_dir.mkdir(parents=True, exist_ok=True)
    profile_root = output_dir / "dataset_profiles_by_day"
    for date in dates:
        day_config = copy.deepcopy(snapshot_config)
        day_config["simulation"]["simulation_date"] = date
        day_configs[date] = day_config
        validation = validate_inputs(day_config)
        validation.write(validation_dir / f"validation_report_{date}.json")
        if not validation.ok:
            raise ValueError(
                f"Validation failed for {date}: " + "; ".join(validation.blocking_errors)
            )
        profile_dir = profile_root / date
        dataset_profile = build_dataset_profile(day_config, data_mode=data_mode, output_dir=profile_dir)
        dataset_profiles_by_date[date] = dataset_profile
        dataset_by_batch_by_date[date] = pd.read_csv(profile_dir / "dataset_profile_by_batch.csv")
        if progress_callback:
            progress_callback(
                {
                    "event": "day_profile_ready",
                    "date": date,
                    "number_of_orders": dataset_profile["number_of_orders"],
                    "number_of_unique_drivers": dataset_profile["number_of_unique_drivers"],
                    "raw_cartesian_pair_count_total": dataset_profile["raw_cartesian_pair_count_total"],
                }
            )

    process = None
    if measure_cpu:
        try:
            import psutil  # type: ignore

            process = psutil.Process()
        except Exception:
            process = None

    initial_grid_path = resolve_project_path(
        snapshot_config, (snapshot_config.get("data", {}).get("grid_values") or {}).get("path")
    )
    raw_rows: list[dict] = []
    batch_rows: list[pd.DataFrame] = []
    carryover_rows: list[dict] = []
    blocked_run_keys: set[str] = set()
    grid_root = output_dir / "grid_carryover"
    grid_root.mkdir(parents=True, exist_ok=True)
    raw_checkpoint_path = output_dir / "comparison_raw_runs_all_days_checkpoint.csv"
    batch_checkpoint_path = output_dir / "comparison_batch_metrics_all_days_checkpoint.csv"
    grid_checkpoint_path = output_dir / "comparison_grid_carryover_audit_checkpoint.csv"
    run_counter = 0
    run_start = time.perf_counter()

    for case in cases:
        skip_reason = _variant_skip_reason(case, comparison_config)
        for repeat in range(warmups + repeats):
            warmup_flag = repeat < warmups
            run_key = f"{case.variant_id}__repeat{repeat}__warmup{int(warmup_flag)}"
            carried_grid_path = initial_grid_path if initial_grid_path and initial_grid_path.exists() else None
            for date_index, date in enumerate(dates, start=1):
                dataset_profile = dataset_profiles_by_date[date]
                run_counter += 1
                elapsed = time.perf_counter() - run_start
                completed_before = max(run_counter - 1, 0)
                eta = (elapsed / completed_before) * (total_runs - completed_before) if completed_before else None
                if progress_callback:
                    progress_callback(
                        {
                            "event": "multiday_run_start",
                            "run_id": run_id,
                            "variant_id": case.variant_id,
                            "lambda_driver_score": case.lambda_driver_score,
                            "matching_algorithm": case.matching_algorithm,
                            "sparse_method": case.sparse_method,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "date_index": date_index,
                            "date_count": len(dates),
                            "run_counter": run_counter,
                            "total_runs": total_runs,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta,
                            "input_grid_values_path": str(carried_grid_path) if carried_grid_path else None,
                        }
                    )

                if skip_reason:
                    row = _empty_run_metrics(
                        case=case,
                        repeat_index=repeat,
                        warmup_flag=warmup_flag,
                        data_mode=data_mode,
                        dataset_profile=dataset_profile,
                        status="skipped",
                        skip_reason=skip_reason,
                    )
                    raw_rows.append(row)
                    _append_checkpoint_row(raw_checkpoint_path, row)
                    _write_checkpoint_summaries(
                        output_dir,
                        raw_rows=raw_rows,
                        run_counter=run_counter,
                        total_runs=total_runs,
                        last_event={
                            "status": "skipped",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "skip_reason": skip_reason,
                        },
                    )
                    if progress_callback:
                        progress_callback(
                            {
                                "event": "multiday_run_skipped",
                                "variant_id": case.variant_id,
                                "repeat_index": repeat,
                                "warmup_flag": warmup_flag,
                                "simulation_date": date,
                                "run_counter": run_counter,
                                "total_runs": total_runs,
                                "skip_reason": skip_reason,
                            }
                        )
                    continue

                if run_key in blocked_run_keys:
                    reason = "Previous date failed; grid carry-over chain unavailable."
                    row = _empty_run_metrics(
                        case=case,
                        repeat_index=repeat,
                        warmup_flag=warmup_flag,
                        data_mode=data_mode,
                        dataset_profile=dataset_profile,
                        status="skipped",
                        skip_reason=reason,
                    )
                    raw_rows.append(row)
                    _append_checkpoint_row(raw_checkpoint_path, row)
                    _write_checkpoint_summaries(
                        output_dir,
                        raw_rows=raw_rows,
                        run_counter=run_counter,
                        total_runs=total_runs,
                        last_event={
                            "status": "skipped",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "skip_reason": reason,
                        },
                    )
                    if progress_callback:
                        progress_callback(
                            {
                                "event": "multiday_run_skipped",
                                "variant_id": case.variant_id,
                                "repeat_index": repeat,
                                "warmup_flag": warmup_flag,
                                "simulation_date": date,
                                "run_counter": run_counter,
                                "total_runs": total_runs,
                                "skip_reason": reason,
                            }
                        )
                    continue

                variant_config = copy.deepcopy(day_configs[date])
                variant_config["utility"]["lambda_driver_score"] = case.lambda_driver_score
                variant_config["matching"]["strategy"] = case.matching_algorithm
                variant_config["matching"]["auction"] = comparison_config.get("matching", {}).get("auction", {})
                variant_config.setdefault("sparse", {})["method"] = case.sparse_method
                grid_input_path = carried_grid_path if grid_carryover else initial_grid_path
                variant_config.setdefault("data", {}).setdefault("grid_values", {})["path"] = (
                    str(grid_input_path) if grid_input_path is not None else None
                )

                def _batch_progress(record: dict) -> None:
                    if not progress_callback:
                        return
                    progress_callback(
                        record
                        | {
                            "event": "multiday_run_batch_progress",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "run_counter": run_counter,
                            "total_runs": total_runs,
                        }
                    )

                try:
                    if measure_memory:
                        tracemalloc.start()
                    if process is not None:
                        process.cpu_percent(interval=None)
                    start = time.perf_counter()
                    result = SimulationEngine(variant_config).run(
                        scenario_id=f"{run_id}_{date}_{case.variant_id}_repeat{repeat}",
                        write_outputs=False,
                        progress_callback=_batch_progress if progress_interval_batches > 0 else None,
                        progress_interval_batches=progress_interval_batches,
                    )
                    runtime = time.perf_counter() - start
                    if measure_memory:
                        _current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_mb = peak / (1024 * 1024)
                    else:
                        peak_mb = 0.0
                    cpu_value = process.cpu_percent(interval=None) if process is not None else None
                    row = _result_to_metrics(
                        result=result,
                        case=case,
                        repeat_index=repeat,
                        warmup_flag=warmup_flag,
                        data_mode=data_mode,
                        dataset_profile=dataset_profile,
                        runtime_seconds=runtime,
                        peak_memory_mb=peak_mb,
                        cpu_mean=cpu_value,
                        cpu_peak=cpu_value,
                    )
                    raw_rows.append(row)

                    output_grid_path = None
                    if grid_carryover:
                        grid_dir = grid_root / case.variant_id / f"repeat{repeat}" / date
                        grid_dir.mkdir(parents=True, exist_ok=True)
                        output_grid_path = grid_dir / "grid_values_final.csv"
                        result.grid_values.to_csv(output_grid_path, index=False)
                        carried_grid_path = output_grid_path
                    carryover_rows.append(
                        {
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "input_grid_values_path": str(variant_config["data"]["grid_values"]["path"]),
                            "output_grid_values_path": str(output_grid_path) if output_grid_path else "",
                            "status": "success",
                        }
                    )
                    _append_checkpoint_row(raw_checkpoint_path, row)
                    _append_checkpoint_row(grid_checkpoint_path, carryover_rows[-1])

                    if not result.batch_metrics.empty:
                        batch_frame = result.batch_metrics.copy()
                        batch_frame["simulation_date"] = date
                        batch_frame["variant_id"] = case.variant_id
                        batch_frame["repeat_index"] = repeat
                        batch_frame["warmup_flag"] = warmup_flag
                        batch_frame["lambda_driver_score"] = case.lambda_driver_score
                        batch_frame["matching_algorithm"] = case.matching_algorithm
                        batch_frame["sparse_method"] = case.sparse_method
                        batch_frame = batch_frame.merge(
                            dataset_by_batch_by_date[date][["batch_id", "raw_cartesian_pair_count"]],
                            on="batch_id",
                            how="left",
                        )
                        batch_rows.append(batch_frame)
                        _append_checkpoint_frame(batch_checkpoint_path, batch_frame)
                    _write_checkpoint_summaries(
                        output_dir,
                        raw_rows=raw_rows,
                        run_counter=run_counter,
                        total_runs=total_runs,
                        last_event={
                            "status": "success",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "runtime_seconds": runtime,
                            "match_rate": row.get("match_rate"),
                        },
                    )

                    if progress_callback:
                        progress_callback(
                            {
                                "event": "multiday_run_success",
                                "variant_id": case.variant_id,
                                "repeat_index": repeat,
                                "warmup_flag": warmup_flag,
                                "simulation_date": date,
                                "run_counter": run_counter,
                                "total_runs": total_runs,
                                "runtime_seconds": runtime,
                                "match_rate": row.get("match_rate"),
                                "candidate_pair_count_after_filter_total": row.get(
                                    "candidate_pair_count_after_filter_total"
                                ),
                            }
                        )
                except Exception as exc:
                    if measure_memory and tracemalloc.is_tracing():
                        tracemalloc.stop()
                    row = _empty_run_metrics(
                        case=case,
                        repeat_index=repeat,
                        warmup_flag=warmup_flag,
                        data_mode=data_mode,
                        dataset_profile=dataset_profile,
                        status="failed",
                        error_message=str(exc),
                    )
                    raw_rows.append(row)
                    _append_checkpoint_row(raw_checkpoint_path, row)
                    carryover_rows.append(
                        {
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "input_grid_values_path": str(variant_config["data"]["grid_values"]["path"]),
                            "output_grid_values_path": "",
                            "status": "failed",
                            "error_message": str(exc),
                        }
                    )
                    _append_checkpoint_row(grid_checkpoint_path, carryover_rows[-1])
                    _write_checkpoint_summaries(
                        output_dir,
                        raw_rows=raw_rows,
                        run_counter=run_counter,
                        total_runs=total_runs,
                        last_event={
                            "status": "failed",
                            "variant_id": case.variant_id,
                            "repeat_index": repeat,
                            "warmup_flag": warmup_flag,
                            "simulation_date": date,
                            "error_message": str(exc),
                        },
                    )
                    if grid_carryover:
                        blocked_run_keys.add(run_key)
                    if progress_callback:
                        progress_callback(
                            {
                                "event": "multiday_run_failed",
                                "variant_id": case.variant_id,
                                "repeat_index": repeat,
                                "warmup_flag": warmup_flag,
                                "simulation_date": date,
                                "run_counter": run_counter,
                                "total_runs": total_runs,
                                "error_message": str(exc),
                            }
                        )

    raw_runs = pd.DataFrame(raw_rows)
    batch_metrics = pd.concat(batch_rows, ignore_index=True) if batch_rows else pd.DataFrame()
    grid_audit = pd.DataFrame(carryover_rows)
    dataset_profiles = pd.DataFrame(list(dataset_profiles_by_date.values()))
    summary_by_day = _summarize_runs_by_date(raw_runs)
    summary_across_days = summarize_runs(raw_runs)
    baseline_by_day = _baseline_relative_by_date(summary_by_day)
    baseline_across_days = baseline_relative(summary_across_days)
    algorithm_across_days = algorithm_quality(summary_across_days)
    sparse_across_days = sparse_method_quality(summary_across_days)

    raw_runs.to_csv(output_dir / "comparison_raw_runs_all_days.csv", index=False)
    batch_metrics.to_csv(output_dir / "comparison_batch_metrics_all_days.csv", index=False)
    dataset_profiles.to_csv(output_dir / "comparison_dataset_profiles_by_day.csv", index=False)
    grid_audit.to_csv(output_dir / "comparison_grid_carryover_audit.csv", index=False)
    summary_by_day.to_csv(output_dir / "comparison_summary_by_day.csv", index=False)
    summary_across_days.to_csv(output_dir / "comparison_summary_across_days.csv", index=False)
    baseline_by_day.to_csv(output_dir / "comparison_baseline_relative_by_day.csv", index=False)
    baseline_across_days.to_csv(output_dir / "comparison_baseline_relative_across_days.csv", index=False)
    algorithm_across_days.to_csv(output_dir / "comparison_algorithm_quality_across_days.csv", index=False)
    sparse_across_days.to_csv(output_dir / "comparison_sparse_method_quality_across_days.csv", index=False)
    raw_runs[raw_runs["status"] == "skipped"].to_csv(
        output_dir / "comparison_skipped_variants_all_days.csv", index=False
    )
    raw_runs[raw_runs["status"] == "failed"].to_csv(
        output_dir / "comparison_errors_all_days.csv", index=False
    )
    _write_multiday_comparison_report(
        output_dir,
        dates=dates,
        dataset_profiles=dataset_profiles,
        summary_by_day=summary_by_day,
        summary_across_days=summary_across_days,
        raw_runs=raw_runs,
        grid_carryover=grid_carryover,
    )
    if progress_callback:
        progress_callback(
            {
                "event": "multiday_comparison_complete",
                "run_id": run_id,
                "output_dir": str(output_dir),
                "date_count": len(dates),
                "successful_runs": int((raw_runs["status"] == "success").sum()),
                "skipped_runs": int((raw_runs["status"] == "skipped").sum()),
                "failed_runs": int((raw_runs["status"] == "failed").sum()),
                "elapsed_seconds": time.perf_counter() - run_start,
            }
        )
    return MultidayComparisonRunResult(
        run_id=run_id,
        output_dir=output_dir,
        dates=dates,
        dataset_profiles=dataset_profiles,
        summary_by_day=summary_by_day,
        summary_across_days=summary_across_days,
        raw_runs=raw_runs,
    )
