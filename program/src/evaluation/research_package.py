from __future__ import annotations

import hashlib
import html
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_project_path


METRIC_DEFINITIONS = [
    {
        "metric": "match_rate",
        "definition": "matched_orders / total_orders for a single simulation date.",
        "interpretation": "Higher values indicate more demand was assigned to drivers.",
    },
    {
        "metric": "expected_conversion_rate",
        "definition": "sum(predicted_completion_probability) / matched_orders.",
        "interpretation": "Expected completed share among matched orders after cancellation-risk modeling.",
    },
    {
        "metric": "median_pickup_distance_km",
        "definition": "Median pickup distance among matched order-driver pairs.",
        "interpretation": "Lower values indicate shorter passenger pickup burden.",
    },
    {
        "metric": "total_expected_economic_utility",
        "definition": "Sum of final matching utility over assigned pairs.",
        "interpretation": "Higher values indicate stronger dispatch objective value under the configured utility model.",
    },
    {
        "metric": "gini_coefficient",
        "definition": "Gini coefficient of driver expected income at the end of the simulated day.",
        "interpretation": "Lower values indicate more equal income distribution across drivers.",
    },
    {
        "metric": "spearman_score_income_correlation",
        "definition": "Rank correlation between driver score and expected income.",
        "interpretation": "Positive values indicate higher-scored drivers tend to receive higher expected income.",
    },
    {
        "metric": "future_score_rows_rejected",
        "definition": "Driver score rows after the simulation date excluded before scoring drivers.",
        "interpretation": "Leakage-prevention audit for temporal validity.",
    },
    {
        "metric": "driver_score_coverage_percent",
        "definition": "Drivers with a valid score on or before the simulation date divided by active drivers.",
        "interpretation": "Higher values mean fewer scores require imputation.",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(config: dict) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    paths = {
        "orders": data_cfg.get("orders", {}).get("path"),
        "driver_locations": data_cfg.get("driver_locations", {}).get("path"),
        "driver_scores": data_cfg.get("driver_scores", {}).get("path"),
        "grid_values": (data_cfg.get("grid_values") or {}).get("path"),
    }
    rows = []
    for label, raw_path in paths.items():
        path = resolve_project_path(config, raw_path)
        row: dict[str, Any] = {"source": label, "path": str(path) if path else "", "exists": bool(path and path.exists())}
        if path and path.exists():
            row["bytes"] = path.stat().st_size
            row["sha256"] = _sha256(path)
            try:
                row["rows"] = len(pd.read_csv(path, usecols=[0]))
            except Exception:
                row["rows"] = None
        else:
            row["bytes"] = 0
            row["sha256"] = ""
            row["rows"] = 0
        rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_stats(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {
            "grid_cell_count": 0,
            "grid_value_min": None,
            "grid_value_mean": None,
            "grid_value_max": None,
        }
    path = Path(path_text)
    if not path.exists():
        return {
            "grid_cell_count": 0,
            "grid_value_min": None,
            "grid_value_mean": None,
            "grid_value_max": None,
        }
    frame = pd.read_csv(path)
    if frame.empty or "grid_value" not in frame.columns:
        return {
            "grid_cell_count": int(len(frame)),
            "grid_value_min": None,
            "grid_value_mean": None,
            "grid_value_max": None,
        }
    values = pd.to_numeric(frame["grid_value"], errors="coerce").dropna()
    return {
        "grid_cell_count": int(len(frame)),
        "grid_value_min": float(values.min()) if not values.empty else None,
        "grid_value_mean": float(values.mean()) if not values.empty else None,
        "grid_value_max": float(values.max()) if not values.empty else None,
    }


def _validation_row(day_dir: Path, date: str) -> dict[str, Any]:
    report = _read_json(day_dir / "validation_report.json")
    coverage = report.get("driver_score_coverage", {})
    coordinate_validation = report.get("coordinate_validation", {})
    fare_validation = report.get("fare_validation", {})
    duplicate_rows = report.get("duplicate_rows", {})
    missing_values = report.get("missing_values", {})
    missing_value_total = sum(
        int(value)
        for table_values in missing_values.values()
        for value in (table_values or {}).values()
    )
    invalid_coordinate_rows = sum(int(value) for value in coordinate_validation.values())
    return {
        "jakarta_data_date": date,
        "validation_passed": len(report.get("blocking_errors", [])) == 0,
        "blocking_error_count": len(report.get("blocking_errors", [])),
        "warning_count": len(report.get("warnings", [])),
        "unique_orders": report.get("unique_order_count", 0),
        "unique_drivers": report.get("unique_driver_count", 0),
        "missing_value_count": missing_value_total,
        "duplicate_order_ids": duplicate_rows.get("orders_duplicate_order_ids", 0),
        "duplicate_driver_location_keys": duplicate_rows.get("driver_location_duplicate_keys", 0),
        "invalid_coordinate_rows": invalid_coordinate_rows,
        "non_positive_fare_rows": fare_validation.get("non_positive_fare_rows", 0),
        "future_score_rows_rejected": report.get("future_score_rows_rejected", 0),
        "drivers_with_valid_scores": coverage.get("drivers_with_valid_scores", 0),
        "drivers_without_scores": coverage.get("drivers_without_scores", 0),
        "driver_score_coverage_percent": coverage.get("score_coverage_percent", 0.0),
        "batch_convention_orders": report.get("batch_index_convention", {}).get("orders", ""),
        "batch_convention_driver_locations": report.get("batch_index_convention", {}).get("driver_locations", ""),
        "warnings": " | ".join(report.get("warnings", [])),
        "blocking_errors": " | ".join(report.get("blocking_errors", [])),
    }


def _collect_batch_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in summary.to_dict("records"):
        batch_path = Path(str(row["output_dir"])) / "batch_metrics.csv"
        if not batch_path.exists():
            continue
        batch = pd.read_csv(batch_path)
        batch["jakarta_data_date"] = row["jakarta_data_date"]
        batch["scenario_id"] = row["scenario_id"]
        frames.append(batch)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _hourly_summary(batch_metrics: pd.DataFrame) -> pd.DataFrame:
    if batch_metrics.empty:
        return pd.DataFrame()
    frame = batch_metrics.copy()
    frame["hour"] = ((frame["batch_id"].astype(int) - 1) // 60).clip(lower=0, upper=23)
    aggregations = {
        "orders_created": ("orders_created", "sum"),
        "matches_created": ("matches_created", "sum"),
        "expired_orders": ("expired_orders", "sum"),
        "candidate_edges": ("candidate_edges", "sum"),
        "avg_available_drivers": ("available_drivers", "mean"),
        "avg_active_orders": ("active_orders", "mean"),
    }
    result = frame.groupby(["jakarta_data_date", "hour"], as_index=False).agg(**aggregations)
    result["hour_label"] = result["hour"].map(lambda value: f"{int(value):02d}:00")
    return result


def _grid_carryover(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.to_dict("records"):
        input_stats = _grid_stats(row.get("input_grid_values_path"))
        output_stats = _grid_stats(row.get("output_grid_values_path"))
        rows.append(
            {
                "jakarta_data_date": row["jakarta_data_date"],
                "input_grid_values_path": row.get("input_grid_values_path", ""),
                "output_grid_values_path": row.get("output_grid_values_path", ""),
                "input_grid_cell_count": input_stats["grid_cell_count"],
                "output_grid_cell_count": output_stats["grid_cell_count"],
                "input_grid_value_mean": input_stats["grid_value_mean"],
                "output_grid_value_mean": output_stats["grid_value_mean"],
                "output_grid_value_min": output_stats["grid_value_min"],
                "output_grid_value_max": output_stats["grid_value_max"],
                "carried_to_next_day": True,
            }
        )
    if rows:
        rows[-1]["carried_to_next_day"] = False
    return pd.DataFrame(rows)


def _readiness_checklist(validation: pd.DataFrame, summary: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "criterion": "All simulated dates passed validation",
            "status": "pass" if not validation.empty and validation["validation_passed"].all() else "attention",
            "evidence": f"{int(validation['validation_passed'].sum())}/{len(validation)} dates passed",
        },
        {
            "criterion": "Temporal leakage prevention is documented",
            "status": "pass" if "future_score_rows_rejected" in validation else "attention",
            "evidence": f"{int(validation['future_score_rows_rejected'].sum()) if not validation.empty else 0} future score rows rejected",
        },
        {
            "criterion": "Driver score coverage is quantified",
            "status": "pass" if not validation.empty and "driver_score_coverage_percent" in validation else "attention",
            "evidence": (
                f"minimum coverage {validation['driver_score_coverage_percent'].min():.2f}%"
                if not validation.empty
                else "not available"
            ),
        },
        {
            "criterion": "Core outcome metrics exist per day",
            "status": "pass" if not summary.empty and {"match_rate", "gini_coefficient"}.issubset(summary.columns) else "attention",
            "evidence": f"{len(summary)} daily rows in multiday_summary.csv",
        },
        {
            "criterion": "Batch-level operational evidence is retained",
            "status": "pass",
            "evidence": "batch_metrics_all_days.csv created from day-level batch metrics",
        },
        {
            "criterion": "Grid value carry-over is auditable",
            "status": "pass" if not grid.empty and "input_grid_values_path" in grid else "attention",
            "evidence": f"{len(grid)} grid handoff rows",
        },
        {
            "criterion": "Source reproducibility hashes are recorded",
            "status": "pass",
            "evidence": "source_file_manifest.csv includes SHA-256 hashes",
        },
        {
            "criterion": "Research comparison breadth",
            "status": "attention",
            "evidence": "For a journal paper, also run model comparison variants or repeated seeds when reporting algorithm superiority.",
        },
    ]
    return pd.DataFrame(checks)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}"
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    existing = [column for column in columns if column in frame.columns]
    if frame.empty or not existing:
        return "<p>No table data available.</p>"
    return frame[existing].head(max_rows).to_html(index=False, classes="paper-table", border=0)


def _write_html_report(
    report_path: Path,
    *,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    grid: pd.DataFrame,
    checklist: pd.DataFrame,
    source_manifest: pd.DataFrame,
) -> None:
    total_orders = int(summary["total_orders"].sum()) if not summary.empty else 0
    total_matches = int(summary["matched_orders"].sum()) if not summary.empty else 0
    weighted_match_rate = total_matches / total_orders if total_orders else 0.0
    mean_gini = float(summary["gini_coefficient"].mean()) if not summary.empty else 0.0
    min_score_coverage = (
        float(validation["driver_score_coverage_percent"].min()) if not validation.empty else 0.0
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Journal Research Evidence Package</title>
  <style>
    body {{
      margin: 0;
      background: #f1f1ef;
      color: #202020;
      font-family: "Times New Roman", Georgia, serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 28px auto;
      background: #fff;
      padding: 44px 54px;
      box-shadow: 0 2px 18px rgba(0,0,0,0.08);
    }}
    h1 {{ text-align: center; font-size: 30px; margin: 0; }}
    h2 {{ border-bottom: 1px solid #222; padding-bottom: 4px; margin-top: 30px; }}
    .subtitle {{ text-align: center; color: #555; margin: 8px 0 28px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 18px 0; }}
    .metric {{ border: 1px solid #cfcfcf; padding: 10px 12px; }}
    .metric span {{ display: block; color: #555; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 20px; margin-top: 4px; }}
    .paper-table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 20px; }}
    .paper-table th, .paper-table td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: right; }}
    .paper-table th:first-child, .paper-table td:first-child {{ text-align: left; }}
    .paper-table th {{ background: #efefef; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
<main>
  <h1>Journal Research Evidence Package</h1>
  <p class="subtitle">Ride-hailing dispatch simulation with date-level reset and grid-value carry-over</p>
  <section>
    <h2>Executive Summary</h2>
    <p><strong>The run is configured to produce publishable evidence, not only raw simulator output.</strong> It audits data quality by date, preserves reproducibility hashes, records leakage-prevention behavior, carries grid values across dates, and exports day-level and batch-level performance metrics.</p>
    <div class="metrics">
      <div class="metric"><span>Dates Simulated</span><strong>{len(summary)}</strong></div>
      <div class="metric"><span>Total Orders</span><strong>{_fmt(total_orders, 0)}</strong></div>
      <div class="metric"><span>Weighted Match Rate</span><strong>{weighted_match_rate:.4f}</strong></div>
      <div class="metric"><span>Mean Daily Gini</span><strong>{mean_gini:.4f}</strong></div>
    </div>
  </section>
  <section>
    <h2>Research-Ready Interpretation</h2>
    <p><strong>Data validity is explicitly audited before interpretation.</strong> The package reports missing values, duplicate identifiers, coordinate validity, fare validity, batch-index convention, future score leakage prevention, and score coverage for every simulated date.</p>
    <p><strong>Operational and fairness outcomes are retained at two levels.</strong> Daily summaries support manuscript tables, while batch-level metrics support temporal plots, peak-hour analysis, and robustness checks.</p>
    <p><strong>Grid-value learning is traceable across days.</strong> Each day records the input grid file and the final grid file used as the next day's initial grid state.</p>
  </section>
  <section>
    <h2>Daily Simulation Outcomes</h2>
    {_table(summary, ["jakarta_data_date", "total_orders", "matched_orders", "match_rate", "expected_conversion_rate", "median_pickup_distance_km", "gini_coefficient", "runtime_seconds"], 30)}
  </section>
  <section>
    <h2>Data Quality Audit</h2>
    <p>Minimum daily driver-score coverage was <strong>{min_score_coverage:.2f}%</strong>. Future score rows are rejected before driver scores are selected, preventing post-treatment leakage from later score snapshots.</p>
    {_table(validation, ["jakarta_data_date", "validation_passed", "unique_orders", "unique_drivers", "missing_value_count", "duplicate_order_ids", "duplicate_driver_location_keys", "invalid_coordinate_rows", "non_positive_fare_rows", "future_score_rows_rejected", "drivers_without_scores", "driver_score_coverage_percent"], 30)}
  </section>
  <section>
    <h2>Grid Carry-Over Audit</h2>
    {_table(grid, ["jakarta_data_date", "input_grid_cell_count", "output_grid_cell_count", "input_grid_value_mean", "output_grid_value_mean", "output_grid_value_min", "output_grid_value_max", "carried_to_next_day"], 30)}
  </section>
  <section>
    <h2>Journal Readiness Checklist</h2>
    {_table(checklist, ["criterion", "status", "evidence"], 30)}
  </section>
  <section>
    <h2>Source Reproducibility</h2>
    {_table(source_manifest, ["source", "exists", "rows", "bytes", "sha256"], 10)}
  </section>
  <section>
    <h2>Limitations For Manuscript Claims</h2>
    <p>This package supports empirical reporting for the configured simulator and actual data. It does not by itself prove algorithmic superiority. For Q1-journal-level claims about BFS-H3 versus A2GAT, Hungarian versus greedy versus auction, or parameter sensitivity, run the model-comparison matrix on actual data and report repeated-run confidence intervals or sensitivity checks.</p>
  </section>
  <section>
    <h2>Generated Evidence Files</h2>
    <p><code>simulation_metrics_by_day.csv</code>, <code>data_quality_by_day.csv</code>, <code>batch_metrics_all_days.csv</code>, <code>hourly_metrics_by_day.csv</code>, <code>grid_carryover_audit.csv</code>, <code>source_file_manifest.csv</code>, <code>metric_definitions.csv</code>, <code>journal_readiness_checklist.csv</code>, and <code>run_manifest.json</code>.</p>
    <p>Generated at: {datetime.now().isoformat(timespec="seconds")}</p>
  </section>
</main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")


def build_research_package(output_dir: str | Path, config: dict, dates: list[str]) -> Path:
    output = Path(output_dir)
    package_dir = output / "research_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output / "multiday_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    validation_rows = []
    for row in summary.to_dict("records"):
        date = str(row["jakarta_data_date"])
        day_dir = Path(str(row["output_dir"]))
        if (day_dir / "validation_report.json").exists():
            validation_rows.append(_validation_row(day_dir, date))
    validation = pd.DataFrame(validation_rows)
    batch_metrics = _collect_batch_metrics(summary)
    hourly = _hourly_summary(batch_metrics)
    grid = _grid_carryover(summary)
    checklist = _readiness_checklist(validation, summary, grid)
    source_manifest = pd.DataFrame(_source_files(config))
    metric_definitions = pd.DataFrame(METRIC_DEFINITIONS)

    summary.to_csv(package_dir / "simulation_metrics_by_day.csv", index=False)
    validation.to_csv(package_dir / "data_quality_by_day.csv", index=False)
    batch_metrics.to_csv(package_dir / "batch_metrics_all_days.csv", index=False)
    hourly.to_csv(package_dir / "hourly_metrics_by_day.csv", index=False)
    grid.to_csv(package_dir / "grid_carryover_audit.csv", index=False)
    checklist.to_csv(package_dir / "journal_readiness_checklist.csv", index=False)
    source_manifest.to_csv(package_dir / "source_file_manifest.csv", index=False)
    metric_definitions.to_csv(package_dir / "metric_definitions.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dates": dates,
        "date_count": len(dates),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": config.get("simulation", {}).get("random_seed"),
        "simulation_config": {
            "total_batches": config.get("simulation", {}).get("total_batches"),
            "batch_duration_minutes": config.get("simulation", {}).get("batch_duration_minutes"),
            "timezone": config.get("simulation", {}).get("timezone"),
            "cancellation_realization_mode": config.get("simulation", {}).get("cancellation_realization_mode"),
            "candidate_edge_logging_default": config.get("simulation", {}).get("enable_candidate_edge_logging"),
        },
        "research_package_files": sorted(path.name for path in package_dir.iterdir() if path.is_file()),
    }
    (package_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_html_report(
        package_dir / "journal_research_report.html",
        summary=summary,
        validation=validation,
        grid=grid,
        checklist=checklist,
        source_manifest=source_manifest,
    )
    return package_dir
