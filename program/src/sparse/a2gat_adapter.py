from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.spatial.distance import haversine_km
from src.spatial.h3_utils import grid_disk, h3_to_latlon, latlon_to_h3


class A2GATUnavailableError(RuntimeError):
    """Raised when an explicitly external A2GAT integration is requested but unavailable."""


@dataclass
class A2GATCandidateProvider:
    """Adaptive anchor-style sparse candidate provider.

    The public A2GAT paper/repository is a graph-learning method, but this simulator
    does not ship trained ride-hailing model weights. This provider implements the
    same sparse-handler role with deterministic adaptive anchors: nearby H3 anchor
    cells are scored, candidate drivers inside the best anchors are ranked with an
    attention-like score, and only the top candidates are passed to matching.
    """

    repository_path: str | None = None
    model_path: str | None = None
    sparse_method: str = "a2gat"

    def availability_status(self) -> tuple[bool, str]:
        if self.repository_path or self.model_path:
            return True, (
                "Built-in A2GAT-style sparse provider is active. External paths are "
                "recorded for traceability but external PyTorch inference is not required."
            )
        return True, "Built-in A2GAT-style adaptive anchor sparse provider is active."

    def generate_candidates(
        self,
        orders: pd.DataFrame,
        available_drivers: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        columns = [
            "order_id",
            "driver_id",
            "driver_score",
            "current_latitude",
            "current_longitude",
            "current_h3_index",
            "position_source",
            "pickup_distance_km",
            "retrieval_rank",
            "retrieval_score",
            "sparse_method",
            "h3_cells_visited",
            "max_bfs_depth_reached",
            "candidate_count_before_distance_filter",
            "candidate_count_after_distance_filter",
            "candidate_generation_runtime_seconds",
            "haversine_filter_runtime_seconds",
        ]
        if orders.empty or available_drivers.empty:
            return pd.DataFrame(columns=columns)

        spatial = config.get("spatial", {})
        sparse_cfg = config.get("sparse", {}).get("a2gat", {})
        h3_resolution = int(spatial.get("h3_resolution", 8))
        maximum_pickup_distance_km = float(spatial.get("maximum_pickup_distance_km", 5.0))
        candidate_target = int(
            sparse_cfg.get("candidate_driver_target", spatial.get("candidate_driver_target", 30))
        )
        anchor_hops = int(sparse_cfg.get("anchor_hops", spatial.get("max_h3_hops", 4)))
        anchor_top_k = int(sparse_cfg.get("anchor_top_k", 8))
        score_weight = float(sparse_cfg.get("driver_score_weight", 0.15))
        distance_weight = float(sparse_cfg.get("distance_weight", 1.0))

        drivers = available_drivers.copy()
        if "current_h3_index" not in drivers.columns:
            drivers["current_h3_index"] = [
                latlon_to_h3(lat, lon, h3_resolution)
                for lat, lon in zip(drivers["current_latitude"], drivers["current_longitude"])
            ]
        if "position_source" not in drivers.columns:
            drivers["position_source"] = "unknown"

        anchors = self._build_anchor_index(drivers)
        rows: list[dict] = []
        for order in orders.to_dict("records"):
            candidate_start = time.perf_counter()
            order_cell = str(
                order.get("origin_h3_index")
                or latlon_to_h3(
                    float(order["origin_latitude"]),
                    float(order["origin_longitude"]),
                    h3_resolution,
                )
            )
            nearby_cells = list(grid_disk(order_cell, anchor_hops))
            anchor_scores = []
            for cell in nearby_cells:
                anchor = anchors.get(str(cell))
                if anchor is None:
                    continue
                anchor_distance = haversine_km(
                    float(order["origin_latitude"]),
                    float(order["origin_longitude"]),
                    anchor["latitude"],
                    anchor["longitude"],
                )
                attention = 1.0 / (1.0 + anchor_distance)
                attention += score_weight * anchor["mean_driver_score"]
                attention += 0.01 * min(anchor["driver_count"], candidate_target)
                anchor_scores.append((attention, str(cell)))
            anchor_scores.sort(key=lambda item: (-item[0], item[1]))
            selected_cells = [cell for _score, cell in anchor_scores[:anchor_top_k]]
            candidate_frames = [anchors[cell]["drivers"] for cell in selected_cells]
            candidate_generation_runtime = time.perf_counter() - candidate_start

            if not candidate_frames:
                continue
            candidate_drivers = pd.concat(candidate_frames, ignore_index=True)
            candidate_drivers = candidate_drivers.drop_duplicates("driver_id", keep="first")
            before_filter = len(candidate_drivers)

            haversine_start = time.perf_counter()
            ranked = []
            for driver in candidate_drivers.to_dict("records"):
                pickup_distance = haversine_km(
                    float(driver["current_latitude"]),
                    float(driver["current_longitude"]),
                    float(order["origin_latitude"]),
                    float(order["origin_longitude"]),
                )
                if pickup_distance > maximum_pickup_distance_km:
                    continue
                driver_score = float(driver.get("driver_score", 0.5))
                retrieval_score = (
                    1.0 / (1.0 + distance_weight * pickup_distance)
                    + score_weight * driver_score
                )
                enriched = dict(driver)
                enriched["pickup_distance_km"] = pickup_distance
                enriched["retrieval_score"] = retrieval_score
                ranked.append(enriched)
            haversine_runtime = time.perf_counter() - haversine_start

            ranked.sort(
                key=lambda item: (
                    -float(item["retrieval_score"]),
                    float(item["pickup_distance_km"]),
                    str(item["driver_id"]),
                )
            )
            selected = ranked[:candidate_target]
            for rank, driver in enumerate(selected, start=1):
                rows.append(
                    {
                        "order_id": str(order["order_id"]),
                        "driver_id": str(driver["driver_id"]),
                        "driver_score": float(driver.get("driver_score", 0.5)),
                        "current_latitude": float(driver["current_latitude"]),
                        "current_longitude": float(driver["current_longitude"]),
                        "current_h3_index": str(driver["current_h3_index"]),
                        "position_source": str(driver.get("position_source", "unknown")),
                        "pickup_distance_km": float(driver["pickup_distance_km"]),
                        "retrieval_rank": rank,
                        "retrieval_score": float(driver["retrieval_score"]),
                        "sparse_method": self.sparse_method,
                        "h3_cells_visited": len(nearby_cells),
                        "max_bfs_depth_reached": anchor_hops,
                        "candidate_count_before_distance_filter": before_filter,
                        "candidate_count_after_distance_filter": len(selected),
                        "candidate_generation_runtime_seconds": candidate_generation_runtime,
                        "haversine_filter_runtime_seconds": haversine_runtime,
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def _build_anchor_index(self, drivers: pd.DataFrame) -> dict[str, dict]:
        anchors: dict[str, dict] = {}
        for cell, group in drivers.groupby("current_h3_index", sort=False, dropna=False):
            cell_id = str(cell)
            try:
                latitude, longitude = h3_to_latlon(cell_id)
            except Exception:
                latitude = float(group["current_latitude"].mean())
                longitude = float(group["current_longitude"].mean())
            anchors[cell_id] = {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "mean_driver_score": float(group.get("driver_score", pd.Series([0.5])).mean()),
                "driver_count": int(len(group)),
                "drivers": group.copy(),
            }
        return anchors
