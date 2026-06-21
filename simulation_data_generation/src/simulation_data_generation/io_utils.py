"""Input/output helpers for Parquet partitions, samples, and manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Create an output directory, optionally replacing an existing generated dataset."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet with parent-directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_partitioned_by_date(df: pd.DataFrame, base_dir: Path, *, date_col: str = "simulation_date") -> None:
    """Write a DataFrame partitioned by simulation_date=YYYY-MM-DD."""
    base_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        write_parquet(df, base_dir / "empty.parquet")
        return
    for date_value, group in df.groupby(date_col, sort=True):
        # The partition directory already stores date_col; keeping it inside the
        # Parquet file can create PyArrow schema conflicts on Windows.
        payload = group.drop(columns=[date_col]).reset_index(drop=True)
        write_parquet(payload, base_dir / f"{date_col}={date_value}" / "part-000.parquet")


def read_partitioned_parquet(path: Path) -> pd.DataFrame:
    """Read a partitioned or single-file Parquet dataset."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "simulation_date" in frame.columns:
        frame["simulation_date"] = frame["simulation_date"].astype(str)
    return frame


def write_sample_csv(df: pd.DataFrame, path: Path, sample_rows: int) -> None:
    """Write a deterministic head sample for Git-safe inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.head(int(sample_rows)).to_csv(path, index=False)


def dataframe_digest(df: pd.DataFrame, rows: int = 2000) -> str:
    """Return a stable digest for reproducibility checks."""
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    csv = df.head(rows).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
