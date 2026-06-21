from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.spatial.bfs_candidates import retrieve_candidate_drivers


@dataclass
class BfsH3CandidateProvider:
    sparse_method: str = "bfs_h3"

    def generate_candidates(
        self,
        orders: pd.DataFrame,
        available_drivers: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        if orders.empty or available_drivers.empty:
            return pd.DataFrame(
                columns=[
                    "order_id",
                    "driver_id",
                    "pickup_distance_km",
                    "retrieval_rank",
                    "retrieval_score",
                    "sparse_method",
                ]
            )
        spatial = config.get("spatial", {})
        rows = []
        for order in orders.to_dict("records"):
            start = time.perf_counter()
            result = retrieve_candidate_drivers(
                order,
                available_drivers,
                h3_resolution=int(spatial.get("h3_resolution", 8)),
                max_h3_hops=int(spatial.get("max_h3_hops", 4)),
                candidate_driver_target=int(spatial.get("candidate_driver_target", 30)),
                maximum_pickup_distance_km=float(spatial.get("maximum_pickup_distance_km", 5.0)),
            )
            runtime = time.perf_counter() - start
            candidates = result.candidates.sort_values(["pickup_distance_km", "driver_id"]).reset_index(drop=True)
            for rank, driver in enumerate(candidates.to_dict("records"), start=1):
                rows.append(
                    {
                        "order_id": str(order["order_id"]),
                        "driver_id": str(driver["driver_id"]),
                        "pickup_distance_km": float(driver["pickup_distance_km"]),
                        "retrieval_rank": rank,
                        "retrieval_score": 1.0 / (1.0 + float(driver["pickup_distance_km"])),
                        "sparse_method": self.sparse_method,
                        "h3_cells_visited": result.h3_cells_visited,
                        "max_bfs_depth_reached": result.max_bfs_depth_reached,
                        "candidate_count_before_distance_filter": result.candidate_count_before_distance_filter,
                        "candidate_count_after_distance_filter": result.candidate_count_after_distance_filter,
                        "candidate_generation_runtime_seconds": runtime,
                        "haversine_filter_runtime_seconds": 0.0,
                    }
                )
        return pd.DataFrame(rows)

