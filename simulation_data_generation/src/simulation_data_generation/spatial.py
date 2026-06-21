"""Spatial helpers for H3, distance, timestamps, and boundary checks."""

from __future__ import annotations

import math
from datetime import date
from zoneinfo import ZoneInfo

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from simulation_data_generation.config import StudyBoundary
from simulation_data_generation.constants import MINUTES_PER_DAY


def boundary_polygon(boundary: StudyBoundary) -> Polygon:
    """Return a rectangular study polygon for the configured boundary."""
    return Polygon(
        [
            (boundary.longitude_min, boundary.latitude_min),
            (boundary.longitude_max, boundary.latitude_min),
            (boundary.longitude_max, boundary.latitude_max),
            (boundary.longitude_min, boundary.latitude_max),
        ]
    )


def is_inside_boundary(latitude: float, longitude: float, boundary: StudyBoundary) -> bool:
    """Return True when a point lies inside the configured study boundary."""
    return bool(boundary_polygon(boundary).covers(Point(float(longitude), float(latitude))))


def clip_to_boundary(latitude: float, longitude: float, boundary: StudyBoundary) -> tuple[float, float]:
    """Clip a coordinate to the rectangular study boundary."""
    lat = min(max(float(latitude), boundary.latitude_min), boundary.latitude_max)
    lon = min(max(float(longitude), boundary.longitude_min), boundary.longitude_max)
    return lat, lon


def latlon_to_h3(latitude: float, longitude: float, resolution: int) -> str:
    """Convert latitude and longitude to an H3 cell id."""
    return str(h3.latlng_to_cell(float(latitude), float(longitude), int(resolution)))


def h3_to_latlon(cell: str) -> tuple[float, float]:
    """Return the centroid latitude and longitude for an H3 cell id."""
    lat, lon = h3.cell_to_latlng(str(cell))
    return float(lat), float(lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers."""
    radius = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return float(2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute approximate initial bearing in degrees."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return float((math.degrees(math.atan2(y, x)) + 360) % 360)


def jitter_coordinate(
    latitude: float,
    longitude: float,
    jitter_meters: float,
    boundary: StudyBoundary,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Apply bounded normal jitter to a coordinate and clip it to the study boundary."""
    sigma_deg = float(jitter_meters) / 111_000.0
    lat = float(latitude) + float(rng.normal(0.0, sigma_deg))
    lon_sigma = sigma_deg / max(0.2, math.cos(math.radians(latitude)))
    lon = float(longitude) + float(rng.normal(0.0, lon_sigma))
    return clip_to_boundary(lat, lon, boundary)


def batch_to_timestamp(simulation_date: str | date, batch_index: int, timezone: str) -> pd.Timestamp:
    """Convert one-based batch index to a timezone-aware timestamp."""
    batch = int(batch_index)
    if not 1 <= batch <= MINUTES_PER_DAY:
        raise ValueError(f"batch_index must be in 1..1440, got {batch}")
    return pd.Timestamp(str(simulation_date), tz=ZoneInfo(timezone)) + pd.Timedelta(minutes=batch - 1)


def timestamp_to_batch(timestamp: pd.Timestamp, timezone: str) -> int:
    """Convert a timestamp to the one-based batch index for its local date."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone)
    ts = ts.tz_convert(timezone)
    return int(ts.hour * 60 + ts.minute + 1)


def day_type_for_date(value: str | date) -> str:
    """Return weekday or weekend for a simulation date."""
    ts = pd.Timestamp(str(value))
    return "weekend" if ts.dayofweek >= 5 else "weekday"


def time_period_for_batch(batch_index: int, definitions: dict[str, dict[str, int]]) -> str:
    """Map a one-minute batch index to a configured time period."""
    minute = int(batch_index) - 1
    for name, spec in definitions.items():
        if int(spec["start_minute"]) <= minute <= int(spec["end_minute"]):
            return name
    raise ValueError(f"No time period covers batch_index={batch_index}")


def normalize_probabilities(weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """Return keys and normalized probabilities from a positive weight mapping."""
    keys = list(weights)
    values = np.asarray([max(float(weights[key]), 0.0) for key in keys], dtype=float)
    total = values.sum()
    if total <= 0:
        values = np.ones_like(values) / len(values)
    else:
        values = values / total
    return keys, values

