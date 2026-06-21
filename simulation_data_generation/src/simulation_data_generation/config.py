"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from simulation_data_generation.constants import DEFAULT_TIMEZONE, MAX_DRIVER_ONLINE_MINUTES, POI_CATEGORIES


class StudyBoundary(BaseModel):
    """Axis-aligned study boundary used by the offline generator."""

    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float
    buffer_km: float = 0.0

    @model_validator(mode="after")
    def validate_bounds(self) -> "StudyBoundary":
        """Ensure minimum coordinates are below maximum coordinates."""
        if self.latitude_min >= self.latitude_max:
            raise ValueError("latitude_min must be lower than latitude_max")
        if self.longitude_min >= self.longitude_max:
            raise ValueError("longitude_min must be lower than longitude_max")
        return self


class OutputConfig(BaseModel):
    """Output locations and storage preferences."""

    data_dir: str = "data/generated"
    sample_dir: str = "data/sample"
    report_dir: str = "reports"
    format: str = "parquet"
    partition_by_date: bool = True
    sample_rows: int = 5000


class RoadNetworkConfig(BaseModel):
    """Road-network acquisition and fallback parameters."""

    use_osm_if_available: bool = False
    cache_path: str = "data/raw_reference/osm_surabaya.graphml"
    offline_grid_rows: int = 18
    offline_grid_cols: int = 22
    maximum_speed_kph: float = 60.0


class GenerationConfig(BaseModel):
    """Validated configuration for dataset generation."""

    random_seed: int = 20260620
    study_area_name: str = "Surabaya, Indonesia"
    timezone: str = DEFAULT_TIMEZONE
    simulation_start_date: str = "2026-06-01"
    number_of_days: int = 7
    number_of_drivers: int = 2875
    daily_order_target: int = 11283
    minimum_daily_orders: int = 9000
    target_daily_order_min: int | None = 9000
    target_daily_order_max: int | None = 13250
    daily_order_profile_factors: list[float] = Field(default_factory=list)
    daily_order_noise_std: float = 0.012
    active_driver_target_min: int | None = 2050
    active_driver_target_max: int | None = 2350
    h3_resolution: int = 8
    cold_start_grid_value: float = 1.0
    dynamic_grid_values: bool = True
    maximum_online_hours_per_day: float = 8.0
    minimum_session_minutes: int = 90
    maximum_session_minutes: int = 300
    study_boundary: StudyBoundary
    output: OutputConfig = Field(default_factory=OutputConfig)
    road_network: RoadNetworkConfig = Field(default_factory=RoadNetworkConfig)
    scales: dict[str, dict[str, Any]] = Field(default_factory=dict)
    time_period_definitions: dict[str, dict[str, int]]
    weekday_demand_profile: dict[str, float]
    weekend_demand_profile: dict[str, float]
    od_probability_matrices: dict[str, dict[str, dict[str, dict[str, float]]]]
    minimum_trip_distance_km: float = 0.8
    maximum_trip_distance_km: float = 18.0
    distance_preference_median_km: float = 2.4
    distance_preference_sigma: float = 0.86
    long_trip_probability: float = 0.14
    coordinate_jitter_meters: float = 180.0
    network_distance_factor_mean: float = 1.34
    network_distance_factor_std: float = 0.07
    traffic_speed_factors: dict[str, dict[str, float]]
    weather_probabilities: dict[str, float]
    weather_speed_factor: dict[str, float]
    weather_multiplier: dict[str, float]
    fare_parameters: dict[str, float]
    surge_parameters: dict[str, float]
    driver_score_targets: dict[str, float]
    driver_participation: dict[str, float]
    poi_category_weights: dict[str, float] = Field(default_factory=dict)
    poi_counts_default: dict[str, int] = Field(default_factory=dict)
    poi_count_scale: float = 1.0

    @field_validator("h3_resolution")
    @classmethod
    def validate_h3_resolution(cls, value: int) -> int:
        """Validate H3 resolution range for urban dispatch experiments."""
        if not 5 <= value <= 11:
            raise ValueError("h3_resolution must be between 5 and 11")
        return value

    @model_validator(mode="after")
    def validate_generation_config(self) -> "GenerationConfig":
        """Validate cross-field constraints."""
        if self.minimum_daily_orders > self.daily_order_target:
            raise ValueError("minimum_daily_orders cannot exceed daily_order_target")
        if self.maximum_online_hours_per_day > 8:
            raise ValueError("maximum_online_hours_per_day must not exceed 8")
        if int(self.maximum_online_hours_per_day * 60) > MAX_DRIVER_ONLINE_MINUTES:
            raise ValueError("maximum online minutes must not exceed 480")
        if self.minimum_session_minutes <= 0:
            raise ValueError("minimum_session_minutes must be positive")
        if self.minimum_session_minutes > self.maximum_session_minutes:
            raise ValueError("minimum_session_minutes cannot exceed maximum_session_minutes")
        missing = set(POI_CATEGORIES) - set(self.poi_category_weights)
        if missing:
            raise ValueError(f"Missing poi_category_weights for: {sorted(missing)}")
        if self.maximum_trip_distance_km <= self.minimum_trip_distance_km:
            raise ValueError("maximum_trip_distance_km must exceed minimum_trip_distance_km")
        fare = self.fare_parameters
        minimum_fare = float(fare.get("minimum_fare", 0.0))
        rate_min = float(fare.get("distance_rate_per_km_min", 0.0))
        rate_max = float(fare.get("distance_rate_per_km_max", 0.0))
        distance_multiplier = float(fare.get("fare_distance_multiplier", 0.0))
        if minimum_fare <= 0:
            raise ValueError("fare_parameters.minimum_fare must be positive")
        if rate_min <= 0 or rate_max < rate_min:
            raise ValueError("fare_parameters distance rate bounds are invalid")
        if distance_multiplier < 1.0:
            raise ValueError("fare_parameters.fare_distance_multiplier must be at least 1.0")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, *, scale: str | None = None) -> GenerationConfig:
    """Load YAML configuration, merge POI weights, apply an optional scale, and validate it."""
    path = Path(config_path).resolve()
    data = _read_yaml(path)
    poi_path = path.parent / "poi_category_weights.yaml"
    if poi_path.exists():
        data = _deep_update(_read_yaml(poi_path), data)
    requested_scale = scale if scale and scale != "default" else None
    if requested_scale:
        scales = data.get("scales", {})
        if requested_scale not in scales:
            raise ValueError(f"Unknown scale '{requested_scale}'. Available scales: {sorted(scales)}")
        data = _deep_update(data, scales[requested_scale])
    try:
        return GenerationConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration {path}: {exc}") from exc
