"""Minute-level driver movement generation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.road_network import RoadNetwork, nearest_node, path_distance_km, shortest_path_nodes
from simulation_data_generation.spatial import (
    batch_to_timestamp,
    bearing_degrees,
    haversine_km,
    latlon_to_h3,
    normalize_probabilities,
    time_period_for_batch,
)


DEMAND_TARGET_BY_PERIOD: dict[str, dict[str, float]] = {
    "morning_peak": {"office": 0.28, "school": 0.18, "university": 0.16, "transport_hub": 0.14, "residential": 0.12, "market": 0.12},
    "daytime_off_peak": {"market": 0.18, "office": 0.18, "hospital": 0.14, "restaurant_or_food_area": 0.14, "shopping_mall": 0.14, "residential": 0.12, "university": 0.10},
    "evening_peak": {"residential": 0.34, "office": 0.16, "shopping_mall": 0.14, "restaurant_or_food_area": 0.14, "transport_hub": 0.12, "market": 0.10},
    "evening_off_peak": {"residential": 0.30, "restaurant_or_food_area": 0.22, "shopping_mall": 0.18, "transport_hub": 0.14, "market": 0.10, "hospital": 0.06},
    "late_evening": {"residential": 0.35, "transport_hub": 0.22, "restaurant_or_food_area": 0.18, "hospital": 0.12, "shopping_mall": 0.08, "market": 0.05},
    "graveyard": {"residential": 0.32, "transport_hub": 0.24, "hospital": 0.18, "restaurant_or_food_area": 0.14, "office": 0.06, "market": 0.06},
}


def _sample_target_poi(pois: pd.DataFrame, period: str, rng: np.random.Generator) -> pd.Series:
    weights = DEMAND_TARGET_BY_PERIOD.get(period, DEMAND_TARGET_BY_PERIOD["daytime_off_peak"])
    categories, probabilities = normalize_probabilities(weights)
    for _ in range(20):
        category = str(rng.choice(categories, p=probabilities))
        candidates = pois[pois["poi_category"] == category]
        if not candidates.empty:
            return candidates.iloc[int(rng.integers(0, len(candidates)))]
    return pois.iloc[int(rng.integers(0, len(pois)))]


def _path_coordinates(network: RoadNetwork, path: Iterable[str]) -> list[tuple[str, float, float]]:
    coords = []
    for node_id in path:
        lat, lon = network.node_positions[str(node_id)]
        coords.append((str(node_id), float(lat), float(lon)))
    return coords


def _interpolate_route(
    network: RoadNetwork,
    source_node: str,
    target_node: str,
    minutes: int,
) -> list[tuple[str, float, float]]:
    path = shortest_path_nodes(network, source_node, target_node)
    coords = _path_coordinates(network, path)
    if minutes <= 1 or len(coords) == 1:
        node, lat, lon = coords[-1]
        return [(node, lat, lon)] * max(1, minutes)
    segment_distances = [0.0]
    for left, right in zip(coords, coords[1:], strict=False):
        segment_distances.append(haversine_km(left[1], left[2], right[1], right[2]))
    cumulative = np.cumsum(segment_distances)
    total = float(cumulative[-1])
    if total <= 0:
        node, lat, lon = coords[-1]
        return [(node, lat, lon)] * minutes
    targets = np.linspace(0, total, minutes)
    interpolated: list[tuple[str, float, float]] = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = min(max(idx, 0), len(coords) - 2)
        start = coords[idx]
        end = coords[idx + 1]
        span = max(cumulative[idx + 1] - cumulative[idx], 1e-9)
        ratio = float((target - cumulative[idx]) / span)
        lat = start[1] + ratio * (end[1] - start[1])
        lon = start[2] + ratio * (end[2] - start[2])
        interpolated.append((end[0], lat, lon))
    return interpolated


def generate_driver_positions_for_day(
    config: GenerationConfig,
    schedules: pd.DataFrame,
    pois: pd.DataFrame,
    network: RoadNetwork,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate minute-level online driver positions for one day."""
    if schedules.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    max_speed = float(config.road_network.maximum_speed_kph)

    for schedule in schedules.itertuples(index=False):
        start_period = time_period_for_batch(int(schedule.start_batch), config.time_period_definitions)
        start_poi = _sample_target_poi(pois, start_period, rng)
        current_node = str(start_poi.road_node_id)
        previous_lat, previous_lon = network.node_positions[current_node]
        batch = int(schedule.start_batch)
        session_end = int(schedule.end_batch)
        minute_in_session = 0

        while batch <= session_end:
            remaining = session_end - batch + 1
            period = time_period_for_batch(batch, config.time_period_definitions)
            wait_probability = 0.35 if period not in {"morning_peak", "evening_peak"} else 0.20
            if rng.random() < wait_probability:
                segment_minutes = min(remaining, int(rng.integers(3, 12)))
                route_points = [(current_node, previous_lat, previous_lon)] * segment_minutes
                status = "available"
            else:
                target_poi = _sample_target_poi(pois, period, rng)
                target_node = str(target_poi.road_node_id)
                path = shortest_path_nodes(network, current_node, target_node)
                distance = path_distance_km(network, path)
                minimum_minutes = max(5, int(math.ceil(distance / max(max_speed / 60.0, 0.1))))
                segment_minutes = min(remaining, int(max(minimum_minutes, rng.integers(10, 31))))
                route_points = _interpolate_route(network, current_node, target_node, segment_minutes)
                status = "moving_to_demand_area"

            for node_id, lat, lon in route_points:
                distance = haversine_km(previous_lat, previous_lon, lat, lon)
                speed = min(max_speed, distance * 60.0)
                heading = bearing_degrees(previous_lat, previous_lon, lat, lon) if distance > 0.002 else 0.0
                rows.append(
                    {
                        "driver_id": schedule.driver_id,
                        "simulation_date": schedule.simulation_date,
                        "timestamp": batch_to_timestamp(schedule.simulation_date, batch, config.timezone).isoformat(),
                        "batch_index": int(batch),
                        "online_session_id": schedule.online_session_id,
                        "online_flag": True,
                        "driver_status": status,
                        "latitude": round(float(lat), 7),
                        "longitude": round(float(lon), 7),
                        "h3_index": latlon_to_h3(lat, lon, config.h3_resolution),
                        "road_node_id": str(node_id),
                        "speed_kph": round(float(speed), 3),
                        "heading_degrees": round(float(heading), 2),
                        "minutes_since_session_start": int(minute_in_session),
                    }
                )
                previous_lat, previous_lon = lat, lon
                current_node = str(node_id) if str(node_id) in network.node_positions else nearest_node(network, lat, lon)
                batch += 1
                minute_in_session += 1
                if batch > session_end:
                    break

    return pd.DataFrame(rows)

