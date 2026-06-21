from __future__ import annotations

import numpy as np

from simulation_data_generation.config import load_config
from simulation_data_generation.poi_generator import generate_pois
from simulation_data_generation.road_network import build_road_network
from simulation_data_generation.spatial import h3_to_latlon, is_inside_boundary, latlon_to_h3


def test_poi_generation_inside_boundary() -> None:
    config = load_config("config/default.yaml", scale="small")
    network = build_road_network(config)
    pois = generate_pois(config, network, np.random.default_rng(2))
    assert not pois.empty
    assert pois["poi_category"].nunique() >= 8
    assert all(is_inside_boundary(row.latitude, row.longitude, config.study_boundary) for row in pois.itertuples())


def test_h3_mapping_roundtrip() -> None:
    cell = latlon_to_h3(-7.2575, 112.7521, 8)
    lat, lon = h3_to_latlon(cell)
    assert -7.4 < lat < -7.1
    assert 112.5 < lon < 113.0

