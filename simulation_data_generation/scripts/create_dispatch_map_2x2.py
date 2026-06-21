from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection

try:
    import contextily as ctx
except Exception:  # pragma: no cover - optional online basemap dependency
    ctx = None

import h3


WEB_MERCATOR_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class DriverSession:
    driver_id: str
    simulation_date: str
    online_session_id: str


def lonlat_to_web_mercator(longitude: np.ndarray | float, latitude: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    x = WEB_MERCATOR_RADIUS_M * np.deg2rad(lon)
    y = WEB_MERCATOR_RADIUS_M * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return x, y


def extent_from_xy(x_values: np.ndarray, y_values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float, float, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("Cannot create map extent from empty or non-finite coordinates.")
    x = x[finite]
    y = y[finite]
    width = max(float(x.max() - x.min()), 1.0)
    height = max(float(y.max() - y.min()), 1.0)
    return (
        float(x.min() - width * pad_fraction),
        float(y.min() - height * pad_fraction),
        float(x.max() + width * pad_fraction),
        float(y.max() + height * pad_fraction),
    )


def extent_from_lonlat(
    frame: pd.DataFrame,
    longitude_col: str,
    latitude_col: str,
    pad_fraction: float = 0.08,
) -> tuple[float, float, float, float]:
    x, y = lonlat_to_web_mercator(frame[longitude_col].to_numpy(), frame[latitude_col].to_numpy())
    return extent_from_xy(x, y, pad_fraction=pad_fraction)


def read_boundary(config_path: Path) -> tuple[float, float, float, float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    boundary = config["study_boundary"]
    return (
        float(boundary["longitude_min"]),
        float(boundary["latitude_min"]),
        float(boundary["longitude_max"]),
        float(boundary["latitude_max"]),
    )


def h3_grid_lines(grid_reference: pd.DataFrame) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    for cell in grid_reference["h3_index"].dropna().astype(str).unique():
        boundary = list(h3.cell_to_boundary(cell))
        if not boundary:
            continue
        latitudes = np.array([point[0] for point in boundary] + [boundary[0][0]], dtype=float)
        longitudes = np.array([point[1] for point in boundary] + [boundary[0][1]], dtype=float)
        x, y = lonlat_to_web_mercator(longitudes, latitudes)
        lines.append(np.column_stack([x, y]))
    return lines


def choose_driver_sessions(driver_positions: pd.DataFrame) -> list[DriverSession]:
    group_cols = ["driver_id", "simulation_date", "online_session_id"]
    counts = (
        driver_positions.groupby(group_cols, observed=True)
        .size()
        .reset_index(name="ping_count")
        .sort_values("ping_count", ascending=False)
    )
    selected: list[DriverSession] = []
    used_drivers: set[str] = set()
    for row in counts.itertuples(index=False):
        if row.driver_id in used_drivers:
            continue
        selected.append(DriverSession(row.driver_id, str(row.simulation_date), row.online_session_id))
        used_drivers.add(row.driver_id)
        if len(selected) == 2:
            return selected
    raise ValueError("At least two distinct drivers with position rows are required for the 2x2 movement panels.")


def session_frame(driver_positions: pd.DataFrame, session: DriverSession) -> pd.DataFrame:
    mask = (
        (driver_positions["driver_id"] == session.driver_id)
        & (driver_positions["simulation_date"].astype(str) == session.simulation_date)
        & (driver_positions["online_session_id"] == session.online_session_id)
    )
    frame = driver_positions.loc[mask].copy()
    frame["_timestamp_sort"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.sort_values(["_timestamp_sort", "batch_index"]).drop(columns=["_timestamp_sort"])


def add_basemap(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_facecolor("#f4f0e8")
    if ctx is None:
        ax.text(
            0.02,
            0.03,
            "Basemap package unavailable; showing generated Surabaya boundary.",
            transform=ax.transAxes,
            fontsize=7,
            color="#5c5c5c",
            ha="left",
            va="bottom",
        )
        return
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom="auto",
            attribution_size=5,
            reset_extent=False,
        )
    except Exception as exc:  # pragma: no cover - depends on network/tile service
        ax.text(
            0.02,
            0.03,
            f"OSM tile load unavailable; generated boundary shown. {type(exc).__name__}",
            transform=ax.transAxes,
            fontsize=7,
            color="#5c5c5c",
            ha="left",
            va="bottom",
        )
    ax.set_aspect("auto")


def add_grid(ax: plt.Axes, lines: list[np.ndarray]) -> None:
    collection = LineCollection(lines, colors="#303030", linewidths=0.35, alpha=0.32, zorder=3)
    ax.add_collection(collection)


def add_common_format(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#202020", pad=6)
    ax.text(
        0.012,
        0.985,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        color="#575757",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1 / 3)
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
        spine.set_linewidth(0.8)


def plot_order_points(
    ax: plt.Axes,
    orders: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    x, y = lonlat_to_web_mercator(orders[longitude_col].to_numpy(), orders[latitude_col].to_numpy())
    point_count = len(orders)
    point_size = 9 if point_count <= 2000 else 2.2
    point_alpha = 0.55 if point_count <= 2000 else 0.22
    ax.scatter(
        x,
        y,
        s=point_size,
        marker=marker,
        c=color,
        alpha=point_alpha,
        linewidths=0,
        zorder=5,
        label=label,
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.88, fontsize=8)


def plot_driver_session(ax: plt.Axes, positions: pd.DataFrame, color: str, title_label: str) -> None:
    x, y = lonlat_to_web_mercator(positions["longitude"].to_numpy(), positions["latitude"].to_numpy())
    sequence = np.arange(len(positions), dtype=float)
    ax.plot(x, y, color="#444444", linewidth=0.75, alpha=0.55, zorder=4)
    scatter = ax.scatter(
        x,
        y,
        c=sequence,
        cmap="viridis",
        s=12,
        alpha=0.78,
        edgecolors="none",
        zorder=5,
    )
    ax.scatter([x[0]], [y[0]], s=54, marker="s", color=color, edgecolors="white", linewidths=0.8, zorder=6)
    ax.scatter([x[-1]], [y[-1]], s=70, marker="*", color="#111111", edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(
        0.02,
        0.04,
        f"{title_label}\n{len(positions)} one-minute pings\nsquare=start, star=end",
        transform=ax.transAxes,
        fontsize=8,
        color="#202020",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d8d8d8", "alpha": 0.9},
        ha="left",
        va="bottom",
        zorder=8,
    )


def build_figure(config_path: Path, data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    orders = pd.read_parquet(data_dir / "orders")
    driver_positions = pd.read_parquet(data_dir / "driver_positions")
    grid_reference = pd.read_parquet(data_dir / "h3_grid_reference.parquet")

    lines = h3_grid_lines(grid_reference)
    selected_sessions = choose_driver_sessions(driver_positions)
    first = session_frame(driver_positions, selected_sessions[0])
    second = session_frame(driver_positions, selected_sessions[1])

    pickup_extent = extent_from_lonlat(orders, "pickup_longitude", "pickup_latitude", pad_fraction=0.05)
    dropoff_extent = extent_from_lonlat(orders, "dropoff_longitude", "dropoff_latitude", pad_fraction=0.05)
    first_extent = extent_from_lonlat(first, "longitude", "latitude", pad_fraction=0.10)
    second_extent = extent_from_lonlat(second, "longitude", "latitude", pad_fraction=0.10)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig_width = 15.0
    fig_height = 6.0
    panel_width = 0.43
    panel_height = panel_width * fig_width / (3 * fig_height)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=180)
    axes = np.array(
        [
            [
                fig.add_axes([0.04, 0.52, panel_width, panel_height]),
                fig.add_axes([0.53, 0.52, panel_width, panel_height]),
            ],
            [
                fig.add_axes([0.04, 0.085, panel_width, panel_height]),
                fig.add_axes([0.53, 0.085, panel_width, panel_height]),
            ],
        ]
    )

    add_basemap(axes[0, 0], pickup_extent)
    add_grid(axes[0, 0], lines)
    plot_order_points(axes[0, 0], orders, "pickup_latitude", "pickup_longitude", "#155f8a", "o", "Pickup point")
    add_common_format(
        axes[0, 0],
        "Pickup Locations",
        f"All generated orders, n={len(orders):,}; H3 dispatch grid overlay.",
    )

    add_basemap(axes[0, 1], dropoff_extent)
    add_grid(axes[0, 1], lines)
    plot_order_points(axes[0, 1], orders, "dropoff_latitude", "dropoff_longitude", "#b35c00", "^", "Dropoff point")
    add_common_format(
        axes[0, 1],
        "Dropoff Locations",
        f"All generated orders, n={len(orders):,}; H3 dispatch grid overlay.",
    )

    add_basemap(axes[1, 0], first_extent)
    plot_driver_session(axes[1, 0], first, "#155f8a", f"{selected_sessions[0].driver_id}, {selected_sessions[0].simulation_date}")
    add_common_format(axes[1, 0], "Sample Driver Movement A", "One online driver session plotted by one-minute GPS pings.")

    add_basemap(axes[1, 1], second_extent)
    plot_driver_session(axes[1, 1], second, "#b35c00", f"{selected_sessions[1].driver_id}, {selected_sessions[1].simulation_date}")
    add_common_format(axes[1, 1], "Sample Driver Movement B", "Different driver session plotted by one-minute GPS pings.")

    fig.suptitle(
        "Surabaya Dispatch Data Distribution",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#1f1f1f",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "dispatch_spatial_2x2_surabaya.png"
    pdf_path = output_dir / "dispatch_spatial_2x2_surabaya.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 2x2 Surabaya dispatch spatial figure.")
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"), help="Path to generation config.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "generated"), help="Generated Parquet data directory.")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "figures"), help="Figure output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    png_path, pdf_path = build_figure(Path(args.config), Path(args.data_dir), Path(args.output_dir))
    print(f"Wrote PNG: {png_path}")
    print(f"Wrote PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
