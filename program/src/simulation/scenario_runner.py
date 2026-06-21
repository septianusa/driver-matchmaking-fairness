from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, load_experiment_matrix, resolve_project_path
from src.evaluation.normalization import normalize_against_baseline
from src.evaluation.report import export_zip
from src.simulation.engine import SimulationEngine, SimulationResult


def run_single(config: dict, scenario_id: str | None = None) -> SimulationResult:
    return SimulationEngine(config).run(scenario_id=scenario_id)


def run_comparison(matrix_config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    base_config, scenarios = load_experiment_matrix(matrix_config_path)
    summaries = []
    for scenario_id, config in scenarios.items():
        result = SimulationEngine(config).run(scenario_id=scenario_id)
        summaries.append(result.summary)
    comparison = pd.DataFrame(summaries)
    normalized = normalize_against_baseline(comparison, "baseline")
    base_output = resolve_project_path(
        base_config, base_config.get("output", {}).get("base_directory", "outputs")
    )
    assert base_output is not None
    comparison_dir = base_output / "scenario_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_dir / "scenario_comparison.csv", index=False)
    normalized.to_csv(comparison_dir / "scenario_comparison_normalized.csv", index=False)
    return comparison, normalized, comparison_dir


def package_project(project_root: str | Path) -> Path:
    root = Path(project_root)
    zip_path = root.parent / f"{root.name}.zip"
    return export_zip(root, zip_path)

