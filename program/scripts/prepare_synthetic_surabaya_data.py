from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROGRAM_DIR = ROOT / "program"
GENERATOR_DIR = ROOT / "simulation_data_generation"


def run_generator(scale: str) -> None:
    command = [
        sys.executable,
        str(GENERATOR_DIR / "scripts" / "generate_data.py"),
        "--config",
        str(GENERATOR_DIR / "config" / "default.yaml"),
        "--scale",
        scale,
        "--overwrite",
    ]
    subprocess.run(command, cwd=GENERATOR_DIR, check=True)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Generated Parquet path not found: {path}")
    return pd.read_parquet(path)


def export_orders(data_dir: Path, raw_dir: Path) -> int:
    orders = read_parquet(data_dir / "orders")
    out = pd.DataFrame(
        {
            "order_id": orders["order_id"].astype(str),
            "customer_id": orders["customer_id"].astype(str),
            "pickup_lat": orders["pickup_latitude"],
            "pickup_lon": orders["pickup_longitude"],
            "dest_lat": orders["dropoff_latitude"],
            "dest_lon": orders["dropoff_longitude"],
            "fare": orders["fare_amount"],
            "jakarta_data_date": orders["simulation_date"].astype(str),
            "batch_step": orders["batch_index"].astype(int),
        }
    )
    out.sort_values(["jakarta_data_date", "batch_step", "order_id"]).to_csv(raw_dir / "orders.csv", index=False)
    return len(out)


def export_driver_locations(data_dir: Path, raw_dir: Path) -> int:
    positions = read_parquet(data_dir / "driver_positions")
    out = pd.DataFrame(
        {
            "driver_id": positions["driver_id"].astype(str),
            "bucket_step": positions["batch_index"].astype(int),
            "avg_latitude": positions["latitude"],
            "avg_longitude": positions["longitude"],
            "jakarta_data_date": positions["simulation_date"].astype(str),
        }
    )
    out.sort_values(["jakarta_data_date", "bucket_step", "driver_id"]).to_csv(
        raw_dir / "driver_locations.csv", index=False
    )
    return len(out)


def export_driver_scores(data_dir: Path, raw_dir: Path) -> int:
    scores = read_parquet(data_dir / "driver_scores.parquet")
    out = pd.DataFrame(
        {
            "driver_id": scores["driver_id"].astype(str),
            "jakarta_period_date": scores["history_end_date"].astype(str),
            "calculated_performance_score": scores["driver_behavior_score"],
            "acceptance_rate": scores["acceptance_rate"],
            "completion_rate": scores["completion_rate"],
            "online_duration_hours": scores["online_duration_hours"],
        }
    )
    out.sort_values(["driver_id", "jakarta_period_date"]).to_csv(raw_dir / "driver_scores.csv", index=False)
    return len(out)


def export_grid_values(raw_dir: Path) -> int:
    grid = pd.DataFrame(columns=["h3_index", "grid_value", "latitude", "longitude"])
    grid.to_csv(raw_dir / "grid_values.csv", index=False)
    return len(grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Surabaya synthetic data and export simulator-ready CSV files."
    )
    parser.add_argument("--scale", choices=["small", "full"], default="full")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only export CSV files from the existing simulation_data_generation/data/generated directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(GENERATOR_DIR / "data" / "generated"),
        help="Generated Parquet directory.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROGRAM_DIR / "data" / "raw"),
        help="Destination folder for simulator CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        run_generator(args.scale)

    counts = {
        "orders": export_orders(data_dir, raw_dir),
        "driver_locations": export_driver_locations(data_dir, raw_dir),
        "driver_scores": export_driver_scores(data_dir, raw_dir),
        "grid_values": export_grid_values(raw_dir),
    }
    print(f"Simulator CSV files written to: {raw_dir}")
    for name, count in counts.items():
        print(f"- {name}: {count:,} rows")
    print("Next: cd program && python main.py validate --config configs/default.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
