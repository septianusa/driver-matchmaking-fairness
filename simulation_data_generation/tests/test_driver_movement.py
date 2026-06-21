from __future__ import annotations

import numpy as np

from simulation_data_generation.config import load_config
from simulation_data_generation.driver_movement import generate_driver_positions_for_day
from simulation_data_generation.driver_schedule import generate_driver_schedules_for_day
from simulation_data_generation.driver_score_generator import generate_driver_scores
from simulation_data_generation.poi_generator import generate_pois
from simulation_data_generation.road_network import build_road_network
from simulation_data_generation.temporal_demand import generate_minute_order_counts


def test_driver_online_hour_limit_and_continuity() -> None:
    config = load_config("config/default.yaml", scale="small")
    rng = np.random.default_rng(4)
    network = build_road_network(config)
    pois = generate_pois(config, network, rng)
    scores = generate_driver_scores(config, rng).head(20)
    counts = generate_minute_order_counts(config, "2026-06-01", rng)
    schedules = generate_driver_schedules_for_day(config, "2026-06-01", scores, counts, rng)
    positions = generate_driver_positions_for_day(config, schedules, pois, network, rng)
    if not positions.empty:
        assert positions.groupby(["driver_id", "simulation_date"]).size().max() <= 480
        assert positions["speed_kph"].max() <= config.road_network.maximum_speed_kph
        for _, group in positions.sort_values("batch_index").groupby("online_session_id"):
            assert (group["batch_index"].diff().dropna() == 1).all()

