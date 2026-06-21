from __future__ import annotations

import numpy as np

from simulation_data_generation.config import load_config
from simulation_data_generation.driver_score_generator import calculate_behavior_score, generate_driver_scores


def test_score_formula() -> None:
    score = calculate_behavior_score(0.9, 0.8, 20)
    assert abs(score - ((0.9 + 0.8 + 0.5) / 3)) < 1e-9


def test_score_distribution_calibration() -> None:
    config = load_config("config/default.yaml", scale="small")
    scores = generate_driver_scores(config, np.random.default_rng(3))
    props = scores["score_segment"].value_counts(normalize=True).to_dict()
    tolerance = float(config.driver_score_targets.get("tolerance", 0.05)) + 0.02
    for segment in ("high", "medium", "low"):
        assert abs(props.get(segment, 0) - float(config.driver_score_targets[segment])) <= tolerance
    assert (scores["completed_orders"] <= scores["accepted_orders"]).all()
    assert (scores["accepted_orders"] <= scores["total_offered_orders"]).all()
