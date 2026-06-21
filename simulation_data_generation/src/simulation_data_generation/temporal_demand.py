"""Non-homogeneous minute-level demand generation."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.constants import MINUTES_PER_DAY
from simulation_data_generation.spatial import day_type_for_date, time_period_for_batch


def simulation_dates(config: GenerationConfig) -> list[str]:
    """Return configured simulation dates as ISO strings."""
    start = date.fromisoformat(str(config.simulation_start_date))
    return [(start + timedelta(days=offset)).isoformat() for offset in range(int(config.number_of_days))]


def smooth_minute_profile(config: GenerationConfig, day_type: str) -> np.ndarray:
    """Create a smooth 1440-minute intensity profile for weekday or weekend demand."""
    period_profile = config.weekend_demand_profile if day_type == "weekend" else config.weekday_demand_profile
    weights = np.zeros(MINUTES_PER_DAY, dtype=float)
    for batch in range(1, MINUTES_PER_DAY + 1):
        period = time_period_for_batch(batch, config.time_period_definitions)
        weights[batch - 1] = float(period_profile.get(period, 1.0))

    minutes = np.arange(MINUTES_PER_DAY)
    if day_type == "weekday":
        peaks = (
            0.80 * np.exp(-0.5 * ((minutes - 450) / 65) ** 2),
            0.70 * np.exp(-0.5 * ((minutes - 1050) / 75) ** 2),
        )
    else:
        peaks = (
            0.45 * np.exp(-0.5 * ((minutes - 780) / 120) ** 2),
            0.65 * np.exp(-0.5 * ((minutes - 1110) / 100) ** 2),
        )
    smooth = weights + sum(peaks)
    smooth = np.convolve(smooth, np.ones(17) / 17, mode="same")
    smooth = np.maximum(smooth, 0.01)
    return smooth / smooth.sum()


def daily_order_total(config: GenerationConfig, simulation_date: str, rng: np.random.Generator) -> int:
    """Sample a daily order count with controlled day-to-day variation."""
    factors = list(config.daily_order_profile_factors)
    if factors:
        start = date.fromisoformat(str(config.simulation_start_date))
        current = date.fromisoformat(str(simulation_date))
        offset = max(0, (current - start).days)
        multiplier = float(factors[offset % len(factors)])
    else:
        day_type = day_type_for_date(simulation_date)
        multiplier = 0.97 if day_type == "weekend" else 1.02
    noise = float(rng.normal(1.0, float(config.daily_order_noise_std)))
    total = int(round(float(config.daily_order_target) * multiplier * noise))
    total = max(int(config.minimum_daily_orders), total)
    if config.target_daily_order_min is not None:
        total = max(int(config.target_daily_order_min), total)
    if config.target_daily_order_max is not None:
        total = min(int(config.target_daily_order_max), total)
    return int(total)


def generate_minute_order_counts(
    config: GenerationConfig,
    simulation_date: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate minute-level order counts using a smooth Poisson process."""
    day_type = day_type_for_date(simulation_date)
    target_total = daily_order_total(config, simulation_date, rng)
    probabilities = smooth_minute_profile(config, day_type)
    expected = target_total * probabilities
    counts = rng.poisson(expected)
    if counts.sum() <= 0:
        counts = rng.multinomial(target_total, probabilities)
    difference = int(target_total - counts.sum())
    if difference > 0:
        add_indices = rng.choice(MINUTES_PER_DAY, size=difference, replace=True, p=probabilities)
        np.add.at(counts, add_indices, 1)
    elif difference < 0:
        removable = np.where(counts > 0)[0]
        for idx in rng.choice(removable, size=min(abs(difference), len(removable)), replace=False):
            counts[idx] -= 1
        while counts.sum() > target_total:
            idx = int(rng.choice(np.where(counts > 0)[0]))
            counts[idx] -= 1

    rows = []
    for minute, count in enumerate(counts, start=1):
        period = time_period_for_batch(minute, config.time_period_definitions)
        rows.append(
            {
                "simulation_date": simulation_date,
                "batch_index": minute,
                "day_type": day_type,
                "time_period": period,
                "order_count": int(count),
                "relative_intensity": float(probabilities[minute - 1] * MINUTES_PER_DAY),
            }
        )
    return pd.DataFrame(rows)


def expected_supply_demand_ratio(relative_intensity: float, rng: np.random.Generator) -> float:
    """Return a generated supply-demand ratio used by fare and validation fields."""
    base = 1.15 - 0.23 * math.tanh(float(relative_intensity) - 1.0)
    return float(np.clip(base + rng.normal(0.0, 0.08), 0.45, 2.4))
