from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.config import yaml_dump


def write_html_report(
    output_dir: str | Path,
    *,
    scenario_id: str,
    summary: dict,
    validation_report: dict,
) -> Path:
    output_path = Path(output_dir) / "report.html"
    summary_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    warning_items = "\n".join(
        f"<li>{html.escape(str(item))}</li>" for item in validation_report.get("warnings", [])
    )
    error_items = "\n".join(
        f"<li>{html.escape(str(item))}</li>" for item in validation_report.get("blocking_errors", [])
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(scenario_id)} simulation report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    table {{ border-collapse: collapse; min-width: 520px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Ride-Hailing Dispatch Simulation Report</h1>
  <h2>{html.escape(scenario_id)}</h2>
  <h3>Summary Metrics</h3>
  <table>{summary_rows}</table>
  <h3>Validation Warnings</h3>
  <ul>{warning_items or "<li>None</li>"}</ul>
  <h3>Blocking Errors</h3>
  <ul>{error_items or "<li>None</li>"}</ul>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(path: str | Path, summary: dict) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)


def write_resolved_config(path: str | Path, config: dict) -> None:
    sanitized = {key: value for key, value in config.items() if not key.startswith("_")}
    Path(path).write_text(yaml_dump(sanitized), encoding="utf-8")


def export_zip(source_dir: str | Path, zip_path: str | Path) -> Path:
    source = Path(source_dir)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded_dirs = {".venv", ".test_deps", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source.rglob("*"):
            if any(part in excluded_dirs for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"} or path == destination:
                continue
            if path.is_file():
                archive.write(path, path.relative_to(source))
    return destination
