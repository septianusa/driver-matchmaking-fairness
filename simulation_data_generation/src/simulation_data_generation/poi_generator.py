"""POI catalogue generation with an offline Surabaya fallback."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.constants import POI_CATEGORIES
from simulation_data_generation.road_network import RoadNetwork, nearest_node
from simulation_data_generation.spatial import clip_to_boundary, jitter_coordinate, latlon_to_h3


SURABAYA_REFERENCE_CENTERS: dict[str, list[tuple[str, float, float]]] = {
    "residential": [
        ("Rungkut Residential Cluster", -7.319, 112.775),
        ("Wiyung Residential Cluster", -7.308, 112.682),
        ("Mulyorejo Residential Cluster", -7.265, 112.795),
        ("Sukomanunggal Residential Cluster", -7.263, 112.704),
    ],
    "office": [
        ("Tunjungan Office Cluster", -7.260, 112.739),
        ("HR Muhammad Business Cluster", -7.286, 112.706),
        ("Darmo Office Cluster", -7.281, 112.735),
    ],
    "school": [
        ("Central School Cluster", -7.279, 112.738),
        ("East School Cluster", -7.296, 112.789),
        ("West School Cluster", -7.274, 112.690),
    ],
    "university": [
        ("ITS University Cluster", -7.281, 112.794),
        ("Airlangga University Cluster", -7.269, 112.758),
        ("Petra University Cluster", -7.339, 112.737),
    ],
    "market": [
        ("Wonokromo Market Cluster", -7.303, 112.738),
        ("Keputran Market Cluster", -7.279, 112.744),
        ("Pabean Market Cluster", -7.236, 112.735),
    ],
    "shopping_mall": [
        ("Tunjungan Mall Cluster", -7.262, 112.739),
        ("Pakuwon Mall Cluster", -7.289, 112.675),
        ("Galaxy Mall Cluster", -7.275, 112.781),
        ("Royal Plaza Cluster", -7.308, 112.735),
    ],
    "hospital": [
        ("Dr Soetomo Hospital Cluster", -7.267, 112.758),
        ("Darmo Hospital Cluster", -7.288, 112.737),
        ("Haji Hospital Cluster", -7.292, 112.781),
    ],
    "transport_hub": [
        ("Gubeng Station Cluster", -7.265, 112.752),
        ("Pasar Turi Station Cluster", -7.246, 112.731),
        ("Purabaya Terminal Cluster", -7.350, 112.724),
        ("Tanjung Perak Access Cluster", -7.215, 112.735),
    ],
    "restaurant_or_food_area": [
        ("G-Walk Food Cluster", -7.291, 112.674),
        ("Dharmahusada Food Cluster", -7.270, 112.770),
        ("Merr Food Cluster", -7.301, 112.782),
        ("Tunjungan Food Cluster", -7.263, 112.741),
    ],
    "recreation": [
        ("Kenjeran Recreation Cluster", -7.245, 112.800),
        ("Bungkul Park Cluster", -7.291, 112.739),
        ("Surabaya Zoo Cluster", -7.295, 112.736),
    ],
}


def _category_counts(config: GenerationConfig) -> dict[str, int]:
    counts = {}
    for category in POI_CATEGORIES:
        base = int(config.poi_counts_default.get(category, 30))
        counts[category] = max(3, int(round(base * float(config.poi_count_scale))))
    return counts


def generate_pois(config: GenerationConfig, network: RoadNetwork, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a POI reference table inside the Surabaya study boundary."""
    rows: list[dict[str, object]] = []
    counts = _category_counts(config)
    seen_by_category: defaultdict[str, int] = defaultdict(int)

    for category, count in counts.items():
        centers = SURABAYA_REFERENCE_CENTERS.get(category, [])
        if not centers:
            centers = [(f"{category.title()} Fallback", -7.265, 112.745)]
        weights = np.ones(len(centers), dtype=float) / len(centers)
        center_indices = rng.choice(len(centers), size=count, replace=True, p=weights)
        for idx in center_indices:
            center_name, lat, lon = centers[int(idx)]
            jittered_lat, jittered_lon = jitter_coordinate(
                lat,
                lon,
                jitter_meters=max(float(config.coordinate_jitter_meters) * 2.5, 250.0),
                boundary=config.study_boundary,
                rng=rng,
            )
            jittered_lat, jittered_lon = clip_to_boundary(jittered_lat, jittered_lon, config.study_boundary)
            node_id = nearest_node(network, jittered_lat, jittered_lon)
            h3_index = latlon_to_h3(jittered_lat, jittered_lon, config.h3_resolution)
            seen_by_category[category] += 1
            sequence = seen_by_category[category]
            rows.append(
                {
                    "poi_id": f"POI-{category.upper().replace('_', '-')}-{sequence:04d}",
                    "poi_name": f"{center_name} {sequence}",
                    "poi_category": category,
                    "latitude": round(jittered_lat, 7),
                    "longitude": round(jittered_lon, 7),
                    "h3_index": h3_index,
                    "road_node_id": node_id,
                    "source_type": network.source_type,
                }
            )
    return pd.DataFrame(rows).sort_values(["poi_category", "poi_id"]).reset_index(drop=True)


def poi_lookup_by_category(pois: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return POIs grouped by category for efficient sampling."""
    return {category: group.reset_index(drop=True) for category, group in pois.groupby("poi_category")}

