"""Driver behavior score generation."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig


def calculate_behavior_score(acceptance_rate: float, completion_rate: float, online_hours: float) -> float:
    """Calculate the driver behavior score from AR, CR, and capped online duration."""
    online_duration_score = min(float(online_hours), 40.0) / 40.0
    return float((acceptance_rate + completion_rate + online_duration_score) / 3.0)


def _sample_segment_metrics(segment: str, rng: np.random.Generator) -> tuple[int, int, int, float, float, float, float]:
    if segment == "high":
        ar = float(rng.beta(22, 1.8))
        cr = float(rng.beta(24, 1.5))
        hours = float(rng.uniform(37, 48))
    elif segment == "medium":
        ar = float(rng.beta(10, 2.8))
        cr = float(rng.beta(11, 2.2))
        hours = float(rng.uniform(27, 42))
    else:
        ar = float(rng.beta(5.5, 4.3))
        cr = float(rng.beta(6.5, 3.6))
        hours = float(rng.uniform(10, 36))

    offered = int(max(20, rng.poisson(170 if segment != "low" else 120)))
    accepted = int(min(offered, round(offered * ar)))
    completed = int(min(accepted, round(accepted * cr)))
    acceptance_rate = accepted / offered if offered > 0 else 0.0
    completion_rate = completed / accepted if accepted > 0 else 0.0
    score = calculate_behavior_score(acceptance_rate, completion_rate, hours)
    return offered, accepted, completed, hours, acceptance_rate, completion_rate, score


def _segment_condition(segment: str, score: float) -> bool:
    if segment == "high":
        return score >= 0.90
    if segment == "medium":
        return 0.70 <= score < 0.90
    return score < 0.70


def generate_driver_scores(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate internally consistent driver behavior score records."""
    start = pd.Timestamp(config.simulation_start_date).date()
    history_end = start - timedelta(days=1)
    history_start = history_end - timedelta(days=6)
    targets = config.driver_score_targets
    n_drivers = int(config.number_of_drivers)
    high_n = int(round(n_drivers * float(targets.get("high", 0.20))))
    medium_n = int(round(n_drivers * float(targets.get("medium", 0.30))))
    low_n = n_drivers - high_n - medium_n
    segment_plan = ["high"] * high_n + ["medium"] * medium_n + ["low"] * low_n
    rng.shuffle(segment_plan)

    rows: list[dict[str, object]] = []
    for index, segment in enumerate(segment_plan, start=1):
        for _ in range(250):
            offered, accepted, completed, hours, ar, cr, score = _sample_segment_metrics(segment, rng)
            if _segment_condition(segment, score):
                break
        else:
            # Deterministic repair if random rejection sampling cannot hit the target band.
            if segment == "high":
                offered, accepted, completed, hours, ar, cr, score = 120, 114, 112, 42.0, 0.95, 112 / 114, 0.0
            elif segment == "medium":
                offered, accepted, completed, hours, ar, cr, score = 120, 94, 88, 32.0, 94 / 120, 88 / 94, 0.0
            else:
                offered, accepted, completed, hours, ar, cr, score = 90, 45, 38, 18.0, 0.50, 38 / 45, 0.0
            score = calculate_behavior_score(ar, cr, hours)
        online_score = min(hours, 40.0) / 40.0
        rows.append(
            {
                "driver_id": f"DRV-{index:06d}",
                "history_start_date": history_start.isoformat(),
                "history_end_date": history_end.isoformat(),
                "total_offered_orders": int(offered),
                "accepted_orders": int(accepted),
                "completed_orders": int(completed),
                "online_duration_hours": round(float(hours), 3),
                "acceptance_rate": round(float(ar), 6),
                "completion_rate": round(float(cr), 6),
                "online_duration_score": round(float(online_score), 6),
                "driver_behavior_score": round(float(score), 6),
                "score_segment": segment,
            }
        )
    return pd.DataFrame(rows)
