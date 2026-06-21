from __future__ import annotations

import math
from typing import Any


def stable_sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def cancellation_probability(edge: dict[str, Any], config: dict[str, Any]) -> float:
    model = config.get("cancellation_model", {})
    spatial = config.get("spatial", {})
    mode = model.get("mode", "score_distance")
    maximum_distance = float(spatial.get("maximum_pickup_distance_km", 5.0))
    normalized_distance = min(max(float(edge.get("pickup_distance_km", 0.0)) / maximum_distance, 0.0), 1.0)
    beta_0 = float(model.get("beta_0", -2.0))
    if mode == "score_distance":
        score = min(max(float(edge.get("driver_score", 0.5)), 0.0), 1.0)
        z = (
            beta_0
            + float(model.get("beta_1_score_gap", 2.0)) * (1.0 - score)
            + float(model.get("beta_2_pickup_distance", 2.0)) * normalized_distance
        )
    elif mode == "ar_cr_distance":
        if "acceptance_rate" not in edge or "completion_rate" not in edge:
            raise ValueError("ar_cr_distance mode requires acceptance_rate and completion_rate.")
        z = (
            beta_0
            + float(model.get("beta_1_acceptance_gap", 1.5)) * (1.0 - float(edge["acceptance_rate"]))
            + float(model.get("beta_2_completion_gap", 2.0)) * (1.0 - float(edge["completion_rate"]))
            + float(model.get("beta_3_pickup_distance", 2.0)) * normalized_distance
        )
    else:
        raise ValueError(f"Unsupported cancellation model mode: {mode}")
    return min(max(stable_sigmoid(z), 0.0), 1.0)

