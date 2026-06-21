from __future__ import annotations

from typing import Any


def compute_base_utility(edge: dict[str, Any], config: dict[str, Any]) -> float:
    fare = float(edge.get("fare", 0.0))
    completion = float(edge.get("predicted_completion_probability", 1.0))
    pickup_distance = float(edge.get("pickup_distance_km", 0.0))
    origin_value = float(edge.get("origin_grid_value_before", 1.0))
    destination_value = float(edge.get("destination_grid_value_before", 1.0))
    gamma = float(config.get("utility", {}).get("gamma_grid_value", 0.9))
    reposition_value = gamma * destination_value - origin_value
    pickup_penalty = pickup_distance * 1000.0
    return fare * completion + fare * 0.05 * reposition_value - pickup_penalty


def driver_score_multiplier(driver_score: float, config: dict[str, Any]) -> float:
    lambda_score = float(config.get("utility", {}).get("lambda_driver_score", 0.0))
    return (1.0 - lambda_score) + lambda_score * float(driver_score)


def compute_final_weight(edge: dict[str, Any], config: dict[str, Any]) -> tuple[float, float, float]:
    base = compute_base_utility(edge, config)
    multiplier = driver_score_multiplier(float(edge.get("driver_score", 0.5)), config)
    return base * multiplier, base, multiplier
