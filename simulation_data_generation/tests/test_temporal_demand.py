from __future__ import annotations

import numpy as np

from simulation_data_generation.config import load_config
from simulation_data_generation.spatial import batch_to_timestamp, timestamp_to_batch
from simulation_data_generation.temporal_demand import generate_minute_order_counts, smooth_minute_profile


def test_batch_index_timestamp_roundtrip() -> None:
    timestamp = batch_to_timestamp("2026-06-01", 1, "Asia/Jakarta")
    assert timestamp.isoformat().startswith("2026-06-01T00:00:00")
    assert timestamp_to_batch(timestamp, "Asia/Jakarta") == 1
    assert timestamp_to_batch(batch_to_timestamp("2026-06-01", 1440, "Asia/Jakarta"), "Asia/Jakarta") == 1440


def test_weekday_peaks_exceed_off_peak() -> None:
    config = load_config("config/default.yaml", scale="small")
    profile = smooth_minute_profile(config, "weekday")
    morning = profile[360:540].mean()
    daytime = profile[540:960].mean()
    assert morning > daytime


def test_small_minute_counts_meet_target() -> None:
    config = load_config("config/default.yaml", scale="small")
    counts = generate_minute_order_counts(config, "2026-06-01", np.random.default_rng(1))
    assert counts["order_count"].sum() >= config.minimum_daily_orders
    assert counts["batch_index"].between(1, 1440).all()

