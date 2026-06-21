from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.spatial.distance import haversine_km
from src.spatial.h3_utils import breadth_first_cells, latlon_to_h3


@dataclass(frozen=True)
class CandidateRetrievalResult:
    candidates: pd.DataFrame
    h3_cells_visited: int
    max_bfs_depth_reached: int
    candidate_count_before_distance_filter: int
    candidate_count_after_distance_filter: int


def retrieve_candidate_drivers(
    order: pd.Series | dict,
    available_drivers: pd.DataFrame,
    *,
    h3_resolution: int,
    max_h3_hops: int,
    candidate_driver_target: int,
    maximum_pickup_distance_km: float,
) -> CandidateRetrievalResult:
    if available_drivers.empty:
        return CandidateRetrievalResult(pd.DataFrame(), 0, 0, 0, 0)
    order_cell = order.get("origin_h3_index") or latlon_to_h3(
        float(order["origin_latitude"]), float(order["origin_longitude"]), h3_resolution
    )
    drivers = available_drivers.copy()
    if "current_h3_index" not in drivers.columns:
        drivers["current_h3_index"] = [
            latlon_to_h3(lat, lon, h3_resolution)
            for lat, lon in zip(drivers["current_latitude"], drivers["current_longitude"])
        ]
    drivers_by_cell = {
        cell: group.copy()
        for cell, group in drivers.groupby("current_h3_index", sort=False, dropna=False)
    }
    selected_frames: list[pd.DataFrame] = []
    visited_count = 0
    max_depth = 0
    candidate_ids: set[str] = set()
    for cell, depth in breadth_first_cells(str(order_cell), max_h3_hops):
        visited_count += 1
        max_depth = max(max_depth, depth)
        if cell in drivers_by_cell:
            group = drivers_by_cell[cell]
            group = group[~group["driver_id"].astype(str).isin(candidate_ids)]
            if not group.empty:
                selected_frames.append(group)
                candidate_ids.update(group["driver_id"].astype(str))
        if len(candidate_ids) >= candidate_driver_target:
            break
    before = len(candidate_ids)
    if not selected_frames:
        return CandidateRetrievalResult(pd.DataFrame(), visited_count, max_depth, before, 0)
    candidates = pd.concat(selected_frames, ignore_index=True)
    candidates["pickup_distance_km"] = [
        haversine_km(
            row.current_latitude,
            row.current_longitude,
            float(order["origin_latitude"]),
            float(order["origin_longitude"]),
        )
        for row in candidates.itertuples(index=False)
    ]
    candidates = candidates[candidates["pickup_distance_km"] <= maximum_pickup_distance_km].copy()
    return CandidateRetrievalResult(
        candidates=candidates,
        h3_cells_visited=visited_count,
        max_bfs_depth_reached=max_depth,
        candidate_count_before_distance_filter=before,
        candidate_count_after_distance_filter=len(candidates),
    )
