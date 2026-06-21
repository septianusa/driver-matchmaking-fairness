"""Synthetic order request generation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.poi_generator import poi_lookup_by_category
from simulation_data_generation.spatial import (
    batch_to_timestamp,
    haversine_km,
    jitter_coordinate,
    latlon_to_h3,
    normalize_probabilities,
)
from simulation_data_generation.temporal_demand import expected_supply_demand_ratio


def _choose_category(
    config: GenerationConfig,
    day_type: str,
    period: str,
    side: str,
    rng: np.random.Generator,
) -> str:
    matrix = config.od_probability_matrices.get(day_type, {})
    spec = matrix.get(period) or matrix.get("daytime_off_peak") or {}
    weights = spec.get(side, config.poi_category_weights)
    keys, probabilities = normalize_probabilities(weights)
    return str(rng.choice(keys, p=probabilities))


def _sample_poi(category: str, grouped_pois: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.Series:
    pois = grouped_pois.get(category)
    if pois is None or pois.empty:
        pois = next(iter(grouped_pois.values()))
    return pois.iloc[int(rng.integers(0, len(pois)))]


def _sample_destination_poi(
    category: str,
    grouped_pois: dict[str, pd.DataFrame],
    pickup_lat: float,
    pickup_lon: float,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> pd.Series:
    pois = grouped_pois.get(category)
    if pois is None or pois.empty:
        pois = next(iter(grouped_pois.values()))
    if len(pois) == 1:
        return pois.iloc[0]

    distances = np.asarray(
        [
            haversine_km(pickup_lat, pickup_lon, float(row.latitude), float(row.longitude))
            for row in pois.itertuples(index=False)
        ],
        dtype=float,
    )
    distances = np.maximum(distances, 0.08)
    if rng.random() < float(config.long_trip_probability):
        weights = np.power(distances, 1.25)
    else:
        median = max(float(config.distance_preference_median_km), 0.2)
        sigma = max(float(config.distance_preference_sigma), 0.1)
        weights = np.exp(-0.5 * (np.log(distances / median) / sigma) ** 2)
        weights += 0.02
    probabilities = weights / weights.sum()
    return pois.iloc[int(rng.choice(np.arange(len(pois)), p=probabilities))]


def _traffic_label(period: str) -> str:
    if period in {"morning_peak", "evening_peak"}:
        return "heavy"
    if period in {"graveyard", "late_evening"}:
        return "light"
    return "moderate"


def _weather_for_day(config: GenerationConfig, rng: np.random.Generator) -> str:
    keys, probabilities = normalize_probabilities(config.weather_probabilities)
    return str(rng.choice(keys, p=probabilities))


def _fare_components(
    config: GenerationConfig,
    straight_line_distance_km: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    fare = config.fare_parameters
    distance_multiplier = float(fare.get("fare_distance_multiplier", 1.4))
    rate_min = float(fare.get("distance_rate_per_km_min", 1850))
    rate_max = float(fare.get("distance_rate_per_km_max", 2300))
    minimum_fare = float(fare.get("minimum_fare", 10000))
    fare_rate = float(rng.uniform(rate_min, rate_max))
    fare_distance = max(0.0, float(straight_line_distance_km) * distance_multiplier)
    distance_fare = fare_distance * fare_rate
    amount = max(minimum_fare, distance_fare)

    # Kept for schema compatibility with older generated samples.
    base_fare = minimum_fare
    time_fare = 0.0
    surge = 1.0
    weather_mult = 1.0
    return (
        base_fare,
        distance_fare,
        time_fare,
        surge,
        weather_mult,
        amount,
        distance_multiplier,
        fare_distance,
        fare_rate,
    )


def generate_orders_for_day(
    config: GenerationConfig,
    simulation_date: str,
    minute_counts: pd.DataFrame,
    pois: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate orders for one simulation day."""
    grouped_pois = poi_lookup_by_category(pois)
    weather = _weather_for_day(config, rng)
    order_sequence = 0
    customer_sequence = 0
    customer_pool = max(1000, int(minute_counts["order_count"].sum() * 0.75))
    customer_ids = [f"CUST-{idx:07d}" for idx in range(1, customer_pool + 1)]
    rows: list[dict[str, object]] = []
    category_pair_attempts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for row in minute_counts.itertuples(index=False):
        count = int(row.order_count)
        if count <= 0:
            continue
        for _ in range(count):
            order_sequence += 1
            for attempt in range(80):
                origin_category = _choose_category(config, row.day_type, row.time_period, "origin", rng)
                dest_category = _choose_category(config, row.day_type, row.time_period, "destination", rng)
                origin_poi = _sample_poi(origin_category, grouped_pois, rng)
                pickup_lat, pickup_lon = jitter_coordinate(
                    float(origin_poi.latitude),
                    float(origin_poi.longitude),
                    config.coordinate_jitter_meters,
                    config.study_boundary,
                    rng,
                )
                dest_poi = _sample_destination_poi(dest_category, grouped_pois, pickup_lat, pickup_lon, config, rng)
                dropoff_lat, dropoff_lon = jitter_coordinate(
                    float(dest_poi.latitude),
                    float(dest_poi.longitude),
                    config.coordinate_jitter_meters,
                    config.study_boundary,
                    rng,
                )
                straight_line = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                if config.minimum_trip_distance_km <= straight_line <= config.maximum_trip_distance_km:
                    break
                category_pair_attempts[(origin_category, dest_category)] += 1
            else:
                continue

            network_factor = max(
                1.05,
                float(rng.normal(config.network_distance_factor_mean, config.network_distance_factor_std)),
            )
            route_distance = max(straight_line, straight_line * network_factor)
            speed_cfg = config.traffic_speed_factors[row.time_period]
            weather_factor = float(config.weather_speed_factor.get(weather, 1.0))
            base_speed = max(8.0, float(speed_cfg["base_speed_kph"]) * weather_factor)
            duration = max(3.0, route_distance / base_speed * 60.0 * float(rng.normal(1.0, 0.08)))
            supply_demand_ratio = expected_supply_demand_ratio(float(row.relative_intensity), rng)
            pickup_eta = max(1.0, min(18.0, 3.0 + 4.0 / max(supply_demand_ratio, 0.3) + rng.normal(0.0, 1.2)))
            (
                base_fare,
                distance_fare,
                time_fare,
                surge,
                weather_mult,
                amount,
                fare_distance_multiplier,
                fare_distance,
                fare_rate,
            ) = _fare_components(
                config,
                straight_line,
                rng,
            )
            customer_sequence = (customer_sequence % customer_pool) + 1
            timestamp = batch_to_timestamp(simulation_date, int(row.batch_index), config.timezone)
            rows.append(
                {
                    "order_id": f"ORD-{simulation_date.replace('-', '')}-{order_sequence:07d}",
                    "customer_id": customer_ids[customer_sequence - 1],
                    "simulation_date": simulation_date,
                    "request_timestamp": timestamp.isoformat(),
                    "batch_index": int(row.batch_index),
                    "pickup_poi_id": str(origin_poi.poi_id),
                    "pickup_poi_category": origin_category,
                    "pickup_latitude": round(float(pickup_lat), 7),
                    "pickup_longitude": round(float(pickup_lon), 7),
                    "pickup_h3_index": latlon_to_h3(pickup_lat, pickup_lon, config.h3_resolution),
                    "dropoff_poi_id": str(dest_poi.poi_id),
                    "dropoff_poi_category": dest_category,
                    "dropoff_latitude": round(float(dropoff_lat), 7),
                    "dropoff_longitude": round(float(dropoff_lon), 7),
                    "dropoff_h3_index": latlon_to_h3(dropoff_lat, dropoff_lon, config.h3_resolution),
                    "straight_line_distance_km": round(float(straight_line), 4),
                    "route_distance_km": round(float(route_distance), 4),
                    "estimated_trip_duration_min": round(float(duration), 3),
                    "estimated_pickup_eta_min": round(float(pickup_eta), 3),
                    "base_fare": round(float(base_fare), 2),
                    "fare_distance_multiplier": round(float(fare_distance_multiplier), 4),
                    "fare_distance_km": round(float(fare_distance), 4),
                    "fare_rate_per_km": round(float(fare_rate), 2),
                    "distance_fare": round(float(distance_fare), 2),
                    "time_fare": round(float(time_fare), 2),
                    "surge_multiplier": round(float(surge), 4),
                    "weather_multiplier": round(float(weather_mult), 4),
                    "supply_demand_ratio": round(float(supply_demand_ratio), 4),
                    "fare_amount": round(float(amount), 2),
                    "traffic_level": _traffic_label(str(row.time_period)),
                    "weather_condition": weather,
                    "day_type": row.day_type,
                    "time_period": row.time_period,
                }
            )
    return pd.DataFrame(rows)
