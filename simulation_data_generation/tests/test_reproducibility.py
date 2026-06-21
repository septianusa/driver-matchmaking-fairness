from __future__ import annotations

from pathlib import Path

from simulation_data_generation.generator import generate_dataset


def test_small_generation_reproducible(tmp_path: Path) -> None:
    root = Path.cwd()
    first = generate_dataset("config/default.yaml", scale="small", output_dir=tmp_path / "run1", overwrite=True, project_root=root)
    second = generate_dataset("config/default.yaml", scale="small", output_dir=tmp_path / "run2", overwrite=True, project_root=root)
    assert first["digests"]["driver_scores"] == second["digests"]["driver_scores"]
    assert first["digests"]["orders_sample"] == second["digests"]["orders_sample"]

