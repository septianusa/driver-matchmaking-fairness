from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import resolve_project_path
from src.data.file_loader import FileDataLoader
from src.data.maxcompute_loader import MaxComputeDataLoader
from src.evaluation.operational import summarize_results
from src.evaluation.report import (
    write_html_report,
    write_json,
    write_resolved_config,
    write_summary_csv,
)
from src.matching.hungarian import solve_assignment
from src.models.cancellation import cancellation_probability
from src.models.grid_value import GridValueStore
from src.models.utility import compute_final_weight
from src.simulation.events import (
    DRIVER_AVAILABLE,
    DRIVER_OCCUPIED,
    ORDER_ACTIVE,
    ORDER_CANCELLED_REALIZED,
    ORDER_COMPLETED_EXPECTED,
    ORDER_COMPLETED_REALIZED,
    ORDER_EXPIRED,
    ORDER_MATCHED,
)
from src.simulation.state import DriverState
from src.sparse.a2gat_adapter import A2GATCandidateProvider
from src.spatial.distance import eta_minutes, haversine_km
from src.spatial.h3_utils import breadth_first_cells, h3_to_latlon, latlon_to_h3, neighbors
from src.validation import validate_inputs

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    scenario_id: str
    output_dir: Path
    summary: dict
    batch_metrics: pd.DataFrame
    match_log: pd.DataFrame
    driver_metrics: pd.DataFrame
    order_events: pd.DataFrame
    candidate_edges: pd.DataFrame
    grid_values: pd.DataFrame
    validation_report: dict


class SimulationEngine:
    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.default_rng(int(config.get("simulation", {}).get("random_seed", 42)))
        self._last_edge_timing = {
            "candidate_generation_runtime_seconds": 0.0,
            "haversine_filter_runtime_seconds": 0.0,
            "edge_feature_runtime_seconds": 0.0,
            "candidate_pair_count_before_filter": 0,
            "candidate_pair_count_after_filter": 0,
        }

    def _loader(self):
        source = self.config.get("data", {}).get("source", "file")
        if source == "file":
            return FileDataLoader(self.config)
        if source == "maxcompute":
            return MaxComputeDataLoader(self.config)
        raise ValueError(f"Unsupported data source: {source}")

    def _scenario_id(self, scenario_id: str | None) -> str:
        if scenario_id:
            return scenario_id
        simulation = self.config.get("simulation", {})
        strategy = self.config.get("matching", {}).get("strategy", "hungarian")
        lambda_score = self.config.get("utility", {}).get("lambda_driver_score", 0.0)
        return (
            f"{simulation.get('simulation_date')}_{strategy}_"
            f"lambda{str(lambda_score).replace('.', 'p')}_seed{simulation.get('random_seed', 42)}"
        )

    def _output_dir(self, scenario_id: str) -> Path:
        base = resolve_project_path(
            self.config, self.config.get("output", {}).get("base_directory", "outputs")
        )
        assert base is not None
        return base / scenario_id

    def run(
        self,
        scenario_id: str | None = None,
        *,
        write_outputs: bool = True,
        progress_callback: Callable[[dict], None] | None = None,
        progress_interval_batches: int = 25,
    ) -> SimulationResult:
        start_time = time.perf_counter()
        scenario = self._scenario_id(scenario_id)
        output_dir = self._output_dir(scenario)
        output_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_inputs(self.config)
        validation_report = validation.to_dict()
        write_json(output_dir / "validation_report.json", validation_report)
        if not validation.ok:
            raise ValueError("Validation failed: " + "; ".join(validation.blocking_errors))

        loaded = self._loader().load_all()
        orders = loaded.orders.copy()
        locations = loaded.driver_locations.copy()
        scores = loaded.driver_scores.copy()
        grid_store = GridValueStore.from_frame(
            loaded.grid_values,
            cold_start_value=float(self.config.get("grid_values", {}).get("cold_start_value", 1.0)),
            update_mode=self.config.get("grid_values", {}).get("update_mode", "sequential_td"),
        )
        score_map = {
            str(row.driver_id): float(row.driver_score) for row in scores.itertuples(index=False)
        }
        total_batches = int(self.config.get("simulation", {}).get("total_batches", 1440))
        spatial = self.config.get("spatial", {})
        eta_cfg = self.config.get("eta", {})
        driver_cfg = self.config.get("drivers", {})
        utility_cfg = self.config.get("utility", {})
        matching_strategy = self.config.get("matching", {}).get("strategy", "hungarian")
        realization_mode = self.config.get("simulation", {}).get(
            "cancellation_realization_mode", "expected_value"
        )
        edge_logging = bool(
            self.config.get("simulation", {}).get("enable_candidate_edge_logging", True)
        )

        location_by_batch = {
            int(batch): group.copy()
            for batch, group in locations.groupby("batch_id", sort=True, dropna=False)
        }
        orders_by_batch = {
            int(batch): group.copy() for batch, group in orders.groupby("created_batch", sort=True)
        }
        drivers: dict[str, DriverState] = {}
        active_orders: dict[str, dict] = {}
        final_order_status = {str(row.order_id): "new" for row in orders.itertuples(index=False)}
        batch_logs: list[dict] = []
        match_logs: list[dict] = []
        edge_logs: list[dict] = []
        order_events: list[dict] = []
        cumulative_orders_created = 0
        cumulative_matches_created = 0
        cumulative_expired_orders = 0

        for batch in range(1, total_batches + 1):
            batch_start = time.perf_counter()
            self._release_completed_drivers(drivers, batch)
            self._apply_historical_pings(drivers, location_by_batch.get(batch, pd.DataFrame()), batch, score_map)
            created_orders = orders_by_batch.get(batch, pd.DataFrame())
            created_order_count = int(len(created_orders))
            for order in created_orders.to_dict("records"):
                order["order_status"] = ORDER_ACTIVE
                active_orders[str(order["order_id"])] = order
                final_order_status[str(order["order_id"])] = ORDER_ACTIVE
                order_events.append({"batch_id": batch, "order_id": order["order_id"], "event": "created"})

            expired = self._expire_orders(active_orders, final_order_status, order_events, batch)
            available_df = self._available_driver_frame(drivers, batch)
            edges = self._build_candidate_edges(
                active_orders,
                available_df,
                grid_store,
                batch,
            )
            if edge_logging and not edges.empty:
                edge_logs.extend(edges.to_dict("records"))
            solver_start = time.perf_counter()
            assignments = (
                solve_assignment(edges, matching_strategy, self.config)
                if not edges.empty
                else pd.DataFrame()
            )
            solver_runtime = time.perf_counter() - solver_start
            state_update_start = time.perf_counter()
            matches_created = self._apply_assignments(
                assignments,
                active_orders,
                drivers,
                grid_store,
                final_order_status,
                order_events,
                match_logs,
                batch,
                realization_mode,
            )
            state_update_runtime = time.perf_counter() - state_update_start
            cumulative_orders_created += created_order_count
            cumulative_matches_created += int(matches_created)
            cumulative_expired_orders += int(expired)
            self._random_walk_idle_drivers(drivers, batch)
            for state in drivers.values():
                if state.is_initialized:
                    state.cumulative_active_minutes += 1
                    if state.availability_status == DRIVER_AVAILABLE:
                        state.consecutive_idle_batches += 1

            runtime = time.perf_counter() - batch_start
            occupied_count = sum(1 for state in drivers.values() if state.availability_status == DRIVER_OCCUPIED)
            available_count = sum(
                1
                for state in drivers.values()
                if state.availability_status == DRIVER_AVAILABLE
                and state.is_initialized
                and state.available_again_batch <= batch
            )
            batch_record = {
                "batch_id": batch,
                "orders_created": created_order_count,
                "active_orders": int(len(active_orders)),
                "available_drivers": int(available_count),
                "occupied_drivers": int(occupied_count),
                "matches_created": int(matches_created),
                "expired_orders": int(expired),
                "candidate_edges": int(len(edges)),
                "candidate_pair_count_before_filter": int(
                    self._last_edge_timing.get("candidate_pair_count_before_filter", 0)
                ),
                "candidate_pair_count_after_filter": int(
                    self._last_edge_timing.get("candidate_pair_count_after_filter", 0)
                ),
                "candidate_generation_runtime_seconds": float(
                    self._last_edge_timing.get("candidate_generation_runtime_seconds", 0.0)
                ),
                "haversine_filter_runtime_seconds": float(
                    self._last_edge_timing.get("haversine_filter_runtime_seconds", 0.0)
                ),
                "edge_feature_runtime_seconds": float(
                    self._last_edge_timing.get("edge_feature_runtime_seconds", 0.0)
                ),
                "solver_runtime_seconds": solver_runtime,
                "state_update_runtime_seconds": state_update_runtime,
                "runtime_seconds": runtime,
            }
            batch_logs.append(batch_record)
            progress_interval = max(0, int(progress_interval_batches))
            if progress_callback and (
                batch == total_batches
                or (progress_interval > 0 and batch % progress_interval == 0)
            ):
                progress_callback(
                    batch_record
                    | {
                        "current_batch": batch,
                        "total_batches": total_batches,
                        "orders_created_total": cumulative_orders_created,
                        "matches_created_total": cumulative_matches_created,
                        "expired_orders_total": cumulative_expired_orders,
                    }
                )

        for order_id, status in final_order_status.items():
            if status == ORDER_ACTIVE:
                final_order_status[order_id] = ORDER_EXPIRED
        batch_metrics = pd.DataFrame(batch_logs)
        match_log = pd.DataFrame(match_logs)
        driver_metrics = pd.DataFrame([state.to_record() for state in drivers.values()])
        if not driver_metrics.empty:
            driver_metrics = driver_metrics.sort_values("driver_id").reset_index(drop=True)
        order_events_df = pd.DataFrame(order_events)
        status_df = pd.DataFrame(
            [{"order_id": order_id, "final_order_status": status} for order_id, status in final_order_status.items()]
        )
        if not order_events_df.empty:
            order_events_df = order_events_df.merge(status_df, on="order_id", how="left")
        else:
            order_events_df = status_df
        candidate_edges = pd.DataFrame(edge_logs)
        final_grid_values = grid_store.to_frame()
        runtime_seconds = time.perf_counter() - start_time
        summary = summarize_results(orders, match_log, driver_metrics, batch_metrics, runtime_seconds)
        summary["scenario_id"] = scenario
        if write_outputs:
            self._write_outputs(
                output_dir,
                scenario,
                summary,
                validation_report,
                batch_metrics,
                match_log,
                driver_metrics,
                order_events_df,
                candidate_edges,
                grid_store,
            )
        return SimulationResult(
            scenario_id=scenario,
            output_dir=output_dir,
            summary=summary,
            batch_metrics=batch_metrics,
            match_log=match_log,
            driver_metrics=driver_metrics,
            order_events=order_events_df,
            candidate_edges=candidate_edges,
            grid_values=final_grid_values,
            validation_report=validation_report,
        )

    def _release_completed_drivers(self, drivers: dict[str, DriverState], batch: int) -> None:
        for state in drivers.values():
            if state.availability_status == DRIVER_OCCUPIED and state.available_again_batch <= batch:
                if state.simulated_destination_latitude is not None:
                    state.current_latitude = float(state.simulated_destination_latitude)
                    state.current_longitude = float(state.simulated_destination_longitude)
                    state.current_h3_index = str(state.simulated_destination_h3_index)
                state.availability_status = DRIVER_AVAILABLE
                state.position_source = "simulated_dropoff"
                state.consecutive_idle_batches = 0

    def _apply_historical_pings(
        self,
        drivers: dict[str, DriverState],
        pings: pd.DataFrame,
        batch: int,
        score_map: dict[str, float],
    ) -> None:
        if pings.empty:
            return
        mode = self.config.get("drivers", {}).get(
            "positioning_mode", "historical_bootstrap_simulated_state"
        )
        refresh_before_assignment = bool(
            self.config.get("drivers", {}).get(
                "refresh_historical_position_before_first_simulated_assignment", True
            )
        )
        for row in pings.itertuples(index=False):
            driver_id = str(row.driver_id)
            state = drivers.get(driver_id)
            if state is not None and state.availability_status == DRIVER_OCCUPIED:
                continue
            if state is None:
                drivers[driver_id] = DriverState(
                    driver_id=driver_id,
                    driver_score=float(score_map.get(driver_id, 0.5)),
                    current_latitude=float(row.latitude),
                    current_longitude=float(row.longitude),
                    current_h3_index=str(row.h3_index),
                    available_again_batch=batch,
                    position_source="historical_bootstrap",
                )
                continue
            should_refresh = mode == "historical_replay" or (
                mode == "historical_bootstrap_simulated_state"
                and refresh_before_assignment
                and not state.has_simulated_assignment
            )
            if should_refresh:
                state.current_latitude = float(row.latitude)
                state.current_longitude = float(row.longitude)
                state.current_h3_index = str(row.h3_index)
                state.position_source = "historical_refresh"

    def _available_driver_frame(self, drivers: dict[str, DriverState], batch: int) -> pd.DataFrame:
        records = []
        driver_cfg = self.config.get("drivers", {})
        enforce_max = bool(driver_cfg.get("enforce_maximum_active_minutes", False))
        max_minutes = int(driver_cfg.get("maximum_active_minutes", 480))
        for state in drivers.values():
            if not state.is_initialized:
                continue
            if state.availability_status != DRIVER_AVAILABLE or state.available_again_batch > batch:
                continue
            if enforce_max and state.cumulative_active_minutes >= max_minutes:
                continue
            records.append(
                {
                    "driver_id": state.driver_id,
                    "driver_score": state.driver_score,
                    "current_latitude": state.current_latitude,
                    "current_longitude": state.current_longitude,
                    "current_h3_index": state.current_h3_index,
                    "position_source": state.position_source,
                }
            )
        return pd.DataFrame(records)

    def _build_candidate_edges(
        self,
        active_orders: dict[str, dict],
        available_df: pd.DataFrame,
        grid_store: GridValueStore,
        batch: int,
    ) -> pd.DataFrame:
        if not active_orders or available_df.empty:
            return pd.DataFrame()
        spatial = self.config.get("spatial", {})
        eta_cfg = self.config.get("eta", {})
        edge_rows = []
        self._last_edge_timing = {
            "candidate_generation_runtime_seconds": 0.0,
            "haversine_filter_runtime_seconds": 0.0,
            "edge_feature_runtime_seconds": 0.0,
            "candidate_pair_count_before_filter": 0,
            "candidate_pair_count_after_filter": 0,
        }
        sparse_method = self.config.get("sparse", {}).get("method", "bfs_h3")
        a2gat_candidates_by_order: dict[str, pd.DataFrame] = {}
        if sparse_method == "a2gat":
            candidate_frame = A2GATCandidateProvider().generate_candidates(
                pd.DataFrame(active_orders.values()),
                available_df,
                self.config,
            )
            if not candidate_frame.empty:
                a2gat_candidates_by_order = {
                    str(order_id): group.copy()
                    for order_id, group in candidate_frame.groupby("order_id", sort=False)
                }
        elif sparse_method != "bfs_h3":
            raise ValueError(f"Unsupported sparse method: {sparse_method}")
        driver_cell_index = self._driver_cell_index(available_df) if sparse_method == "bfs_h3" else {}
        bfs_cache: dict[str, list[tuple[str, int]]] = {}
        max_h3_hops = int(spatial.get("max_h3_hops", 4))
        candidate_target = int(spatial.get("candidate_driver_target", 30))
        maximum_pickup_distance_km = float(spatial.get("maximum_pickup_distance_km", 5.0))
        for order in active_orders.values():
            if sparse_method == "a2gat":
                candidates, retrieval_stats = self._retrieve_candidates_from_a2gat(
                    order,
                    a2gat_candidates_by_order,
                )
            else:
                candidates, retrieval_stats = self._retrieve_candidates_from_index(
                    order,
                    driver_cell_index,
                    bfs_cache,
                    max_h3_hops=max_h3_hops,
                    candidate_driver_target=candidate_target,
                    maximum_pickup_distance_km=maximum_pickup_distance_km,
                )
            self._last_edge_timing["candidate_generation_runtime_seconds"] += retrieval_stats[
                "candidate_generation_runtime_seconds"
            ]
            self._last_edge_timing["haversine_filter_runtime_seconds"] += retrieval_stats[
                "haversine_filter_runtime_seconds"
            ]
            self._last_edge_timing["candidate_pair_count_before_filter"] += retrieval_stats[
                "candidate_count_before_distance_filter"
            ]
            self._last_edge_timing["candidate_pair_count_after_filter"] += retrieval_stats[
                "candidate_count_after_distance_filter"
            ]
            for driver in candidates:
                feature_start = time.perf_counter()
                pickup_eta = eta_minutes(
                    float(driver["pickup_distance_km"]),
                    float(eta_cfg.get("assumed_pickup_speed_km_per_hour", 20.0)),
                    int(eta_cfg.get("minimum_pickup_eta_minutes", 1)),
                    0.0,
                )
                occupied_duration = pickup_eta + int(order["trip_eta_minutes"])
                origin_value = grid_store.get(str(order["origin_h3_index"]))
                destination_value = grid_store.get(str(order["destination_h3_index"]))
                edge = {
                    "batch_id": batch,
                    "order_id": str(order["order_id"]),
                    "driver_id": str(driver["driver_id"]),
                    "driver_score": float(driver["driver_score"]),
                    "position_source": driver.get("position_source", "unknown"),
                    "driver_latitude_before_match": float(driver["current_latitude"]),
                    "driver_longitude_before_match": float(driver["current_longitude"]),
                    "driver_h3_index_before_match": str(driver["current_h3_index"]),
                    "origin_latitude": float(order["origin_latitude"]),
                    "origin_longitude": float(order["origin_longitude"]),
                    "origin_h3_index": str(order["origin_h3_index"]),
                    "destination_latitude": float(order["destination_latitude"]),
                    "destination_longitude": float(order["destination_longitude"]),
                    "destination_h3_index": str(order["destination_h3_index"]),
                    "pickup_distance_km": float(driver["pickup_distance_km"]),
                    "trip_distance_km": float(order["trip_distance_km"]),
                    "pickup_eta_minutes": pickup_eta,
                    "trip_eta_minutes": int(order["trip_eta_minutes"]),
                    "occupied_duration_minutes": occupied_duration,
                    "driver_available_again_batch": batch + occupied_duration,
                    "fare": float(order["fare"]),
                    "origin_grid_value_before": origin_value,
                    "destination_grid_value_before": destination_value,
                    "h3_cells_visited": retrieval_stats["h3_cells_visited"],
                    "max_bfs_depth_reached": retrieval_stats["max_bfs_depth_reached"],
                    "candidate_count_before_distance_filter": retrieval_stats[
                        "candidate_count_before_distance_filter"
                    ],
                    "candidate_count_after_distance_filter": retrieval_stats[
                        "candidate_count_after_distance_filter"
                    ],
                }
                p_cancel = cancellation_probability(edge, self.config)
                edge["predicted_cancellation_probability"] = p_cancel
                edge["predicted_completion_probability"] = 1.0 - p_cancel
                final_weight, base_utility, multiplier = compute_final_weight(edge, self.config)
                edge["base_utility"] = base_utility
                edge["driver_score_multiplier"] = multiplier
                edge["final_matching_weight"] = final_weight
                edge["economic_utility"] = final_weight
                edge_rows.append(edge)
                self._last_edge_timing["edge_feature_runtime_seconds"] += (
                    time.perf_counter() - feature_start
                )
        return pd.DataFrame(edge_rows)

    def _driver_cell_index(self, available_df: pd.DataFrame) -> dict[str, list[dict]]:
        index: dict[str, list[dict]] = {}
        for record in available_df.to_dict("records"):
            index.setdefault(str(record["current_h3_index"]), []).append(record)
        return index

    def _retrieve_candidates_from_index(
        self,
        order: dict,
        driver_cell_index: dict[str, list[dict]],
        bfs_cache: dict[str, list[tuple[str, int]]],
        *,
        max_h3_hops: int,
        candidate_driver_target: int,
        maximum_pickup_distance_km: float,
    ) -> tuple[list[dict], dict[str, int]]:
        origin_cell = str(order["origin_h3_index"])
        if origin_cell not in bfs_cache:
            bfs_cache[origin_cell] = list(breadth_first_cells(origin_cell, max_h3_hops))
        candidate_records: list[dict] = []
        seen_driver_ids: set[str] = set()
        h3_cells_visited = 0
        max_depth = 0
        candidate_start = time.perf_counter()
        for cell, depth in bfs_cache[origin_cell]:
            h3_cells_visited += 1
            max_depth = max(max_depth, depth)
            for driver in driver_cell_index.get(cell, []):
                driver_id = str(driver["driver_id"])
                if driver_id in seen_driver_ids:
                    continue
                seen_driver_ids.add(driver_id)
                candidate_records.append(driver)
            if len(candidate_records) >= candidate_driver_target:
                break
        candidate_runtime = time.perf_counter() - candidate_start
        before_filter = len(candidate_records)
        feasible: list[dict] = []
        haversine_start = time.perf_counter()
        for driver in candidate_records:
            pickup_distance = haversine_km(
                float(driver["current_latitude"]),
                float(driver["current_longitude"]),
                float(order["origin_latitude"]),
                float(order["origin_longitude"]),
            )
            if pickup_distance <= maximum_pickup_distance_km:
                enriched = dict(driver)
                enriched["pickup_distance_km"] = pickup_distance
                feasible.append(enriched)
        haversine_runtime = time.perf_counter() - haversine_start
        return feasible, {
            "h3_cells_visited": h3_cells_visited,
            "max_bfs_depth_reached": max_depth,
            "candidate_count_before_distance_filter": before_filter,
            "candidate_count_after_distance_filter": len(feasible),
            "candidate_generation_runtime_seconds": candidate_runtime,
            "haversine_filter_runtime_seconds": haversine_runtime,
        }

    def _retrieve_candidates_from_a2gat(
        self,
        order: dict,
        candidates_by_order: dict[str, pd.DataFrame],
    ) -> tuple[list[dict], dict[str, int | float]]:
        candidates = candidates_by_order.get(str(order["order_id"]), pd.DataFrame())
        if candidates.empty:
            return [], {
                "h3_cells_visited": 0,
                "max_bfs_depth_reached": 0,
                "candidate_count_before_distance_filter": 0,
                "candidate_count_after_distance_filter": 0,
                "candidate_generation_runtime_seconds": 0.0,
                "haversine_filter_runtime_seconds": 0.0,
            }
        first = candidates.iloc[0]
        return candidates.to_dict("records"), {
            "h3_cells_visited": int(first.get("h3_cells_visited", 0)),
            "max_bfs_depth_reached": int(first.get("max_bfs_depth_reached", 0)),
            "candidate_count_before_distance_filter": int(
                first.get("candidate_count_before_distance_filter", len(candidates))
            ),
            "candidate_count_after_distance_filter": int(
                first.get("candidate_count_after_distance_filter", len(candidates))
            ),
            "candidate_generation_runtime_seconds": float(
                first.get("candidate_generation_runtime_seconds", 0.0)
            ),
            "haversine_filter_runtime_seconds": float(
                first.get("haversine_filter_runtime_seconds", 0.0)
            ),
        }

    def _apply_assignments(
        self,
        assignments: pd.DataFrame,
        active_orders: dict[str, dict],
        drivers: dict[str, DriverState],
        grid_store: GridValueStore,
        final_order_status: dict[str, str],
        order_events: list[dict],
        match_logs: list[dict],
        batch: int,
        realization_mode: str,
    ) -> int:
        if assignments.empty:
            return 0
        matches_created = 0
        gamma = float(self.config.get("utility", {}).get("gamma_grid_value", 0.9))
        alpha = float(self.config.get("utility", {}).get("alpha_grid_learning_rate", 0.1))
        for edge in assignments.to_dict("records"):
            order_id = str(edge["order_id"])
            driver_id = str(edge["driver_id"])
            if order_id not in active_orders or driver_id not in drivers:
                continue
            driver = drivers[driver_id]
            if driver.availability_status != DRIVER_AVAILABLE or driver.available_again_batch > batch:
                continue
            order = active_orders.pop(order_id)
            p_completion = float(edge["predicted_completion_probability"])
            if realization_mode == "stochastic_realization":
                completed = bool(self.rng.random() <= p_completion)
                income = float(edge["fare"]) if completed else 0.0
                status = ORDER_COMPLETED_REALIZED if completed else ORDER_CANCELLED_REALIZED
            else:
                completed = True
                income = float(edge["fare"]) * p_completion
                status = ORDER_COMPLETED_EXPECTED
            driver.total_expected_income += float(edge["fare"]) * p_completion
            driver.expected_completed_orders += p_completion
            driver.total_realized_income += income
            driver.assigned_orders += 1
            if completed:
                driver.realized_completed_orders += 1
            else:
                driver.cancelled_orders += 1
            driver.availability_status = DRIVER_OCCUPIED
            driver.available_again_batch = int(edge["driver_available_again_batch"])
            driver.has_simulated_assignment = True
            driver.simulated_destination_latitude = float(order["destination_latitude"])
            driver.simulated_destination_longitude = float(order["destination_longitude"])
            driver.simulated_destination_h3_index = str(order["destination_h3_index"])
            driver.consecutive_idle_batches = 0
            final_order_status[order_id] = status if status != ORDER_COMPLETED_EXPECTED else ORDER_MATCHED
            order_events.append({"batch_id": batch, "order_id": order_id, "event": status})
            td_error = grid_store.update_after_match(
                str(order["origin_h3_index"]),
                str(order["destination_h3_index"]),
                float(edge["economic_utility"]) / 10000.0,
                gamma=gamma,
                alpha=alpha,
            )
            edge["realized_trip_status"] = status
            edge["grid_td_error"] = td_error
            match_logs.append(edge)
            matches_created += 1
        return matches_created

    def _expire_orders(
        self,
        active_orders: dict[str, dict],
        final_order_status: dict[str, str],
        order_events: list[dict],
        batch: int,
    ) -> int:
        expired_ids = [
            order_id
            for order_id, order in active_orders.items()
            if batch > int(order["order_expiry_batch"])
        ]
        for order_id in expired_ids:
            active_orders.pop(order_id, None)
            final_order_status[order_id] = ORDER_EXPIRED
            order_events.append({"batch_id": batch, "order_id": order_id, "event": ORDER_EXPIRED})
        return len(expired_ids)

    def _random_walk_idle_drivers(self, drivers: dict[str, DriverState], batch: int) -> None:
        driver_cfg = self.config.get("drivers", {})
        if not bool(driver_cfg.get("enable_idle_random_walk", True)):
            return
        interval = int(driver_cfg.get("idle_random_walk_interval_batches", 5))
        if interval <= 0:
            return
        for state in drivers.values():
            if state.availability_status != DRIVER_AVAILABLE:
                continue
            if not state.has_simulated_assignment:
                continue
            if state.consecutive_idle_batches < interval:
                continue
            cells = sorted(neighbors(state.current_h3_index))
            if not cells:
                continue
            new_cell = str(self.rng.choice(cells))
            lat, lon = h3_to_latlon(new_cell)
            state.current_h3_index = new_cell
            state.current_latitude = float(lat)
            state.current_longitude = float(lon)
            state.position_source = "simulated_random_walk"
            state.consecutive_idle_batches = 0

    def _write_outputs(
        self,
        output_dir: Path,
        scenario_id: str,
        summary: dict,
        validation_report: dict,
        batch_metrics: pd.DataFrame,
        match_log: pd.DataFrame,
        driver_metrics: pd.DataFrame,
        order_events: pd.DataFrame,
        candidate_edges: pd.DataFrame,
        grid_store: GridValueStore,
    ) -> None:
        batch_metrics.to_csv(output_dir / "batch_metrics.csv", index=False)
        match_log.to_csv(output_dir / "match_log.csv", index=False)
        driver_metrics.to_csv(output_dir / "driver_metrics.csv", index=False)
        order_events.to_csv(output_dir / "order_events.csv", index=False)
        candidate_edges.to_csv(output_dir / "candidate_edges.csv", index=False)
        grid_store.to_frame().to_csv(output_dir / "grid_values_final.csv", index=False)
        write_summary_csv(output_dir / "scenario_summary.csv", summary)
        write_resolved_config(output_dir / "resolved_config.yaml", self.config)
        write_html_report(
            output_dir,
            scenario_id=scenario_id,
            summary=summary,
            validation_report=validation_report,
        )
