"""Driver participation and online-session scheduling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.constants import MAX_DRIVER_ONLINE_MINUTES


def _start_distribution(minute_counts: pd.DataFrame) -> np.ndarray:
    weights = minute_counts["order_count"].to_numpy(dtype=float) + 1.0
    smooth = np.convolve(weights, np.ones(61) / 61, mode="same")
    return smooth / smooth.sum()


def generate_driver_schedules_for_day(
    config: GenerationConfig,
    simulation_date: str,
    driver_scores: pd.DataFrame,
    minute_counts: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate online sessions for active drivers on one day."""
    start_prob = _start_distribution(minute_counts)
    max_minutes = min(MAX_DRIVER_ONLINE_MINUTES, int(config.maximum_online_hours_per_day * 60))
    base_probability = float(config.driver_participation.get("base_probability", 0.70))
    high_bonus = float(config.driver_participation.get("high_score_bonus", 0.08))
    low_penalty = float(config.driver_participation.get("low_score_penalty", 0.06))
    two_session_probability = float(config.driver_participation.get("two_session_probability", 0.28))
    high_minutes_mean = float(config.driver_participation.get("high_online_minutes_mean", 420))
    medium_minutes_mean = float(config.driver_participation.get("medium_online_minutes_mean", 405))
    low_minutes_mean = float(config.driver_participation.get("low_online_minutes_mean", 365))
    online_minutes_std = float(config.driver_participation.get("online_minutes_std", 55))

    demand_factor = float(minute_counts["order_count"].sum() / max(float(config.daily_order_target), 1.0))
    demand_adjustment = np.clip((demand_factor - 1.0) * 0.20, -0.08, 0.08)
    rows: list[dict[str, object]] = []

    for driver in driver_scores.itertuples(index=False):
        score = float(driver.driver_behavior_score)
        probability = base_probability + demand_adjustment
        if score >= 0.90:
            probability += high_bonus
        elif score < 0.70:
            probability -= low_penalty
        probability = float(np.clip(probability, 0.15, 0.95))
        if rng.random() > probability:
            continue

        if score >= 0.90:
            expected_online_minutes = high_minutes_mean
        elif score >= 0.70:
            expected_online_minutes = medium_minutes_mean
        else:
            expected_online_minutes = low_minutes_mean
        total_minutes = int(
            np.clip(
                rng.normal(expected_online_minutes, online_minutes_std),
                config.minimum_session_minutes,
                max_minutes,
            )
        )
        split = rng.random() < two_session_probability and total_minutes >= 180
        session_lengths = [total_minutes]
        if split:
            first = int(np.clip(rng.normal(total_minutes * 0.52, 35), config.minimum_session_minutes, total_minutes - 45))
            second = total_minutes - first
            if second >= 45:
                session_lengths = [first, second]

        previous_end = 0
        for session_index, length in enumerate(session_lengths, start=1):
            for _ in range(100):
                start_batch = int(rng.choice(np.arange(1, 1441), p=start_prob))
                end_batch = min(1440, start_batch + int(length) - 1)
                if end_batch - start_batch + 1 >= 30 and start_batch > previous_end + 30:
                    break
            else:
                start_batch = min(1440, previous_end + 31)
                end_batch = min(1440, start_batch + int(length) - 1)
            if end_batch < start_batch:
                continue
            rows.append(
                {
                    "driver_id": driver.driver_id,
                    "simulation_date": simulation_date,
                    "online_session_id": f"{driver.driver_id}-{simulation_date}-S{session_index}",
                    "session_index": session_index,
                    "start_batch": int(start_batch),
                    "end_batch": int(end_batch),
                    "session_minutes": int(end_batch - start_batch + 1),
                }
            )
            previous_end = end_batch

    return pd.DataFrame(rows)
