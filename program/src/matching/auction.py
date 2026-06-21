from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.matching.base import add_solver_metadata, validate_one_to_one


class AuctionDidNotConvergeError(RuntimeError):
    """Raised when the auction solver cannot finish within the configured iteration limit."""


@dataclass
class AuctionSolver:
    epsilon: float = 0.001
    max_iterations: int = 1_000_000
    solver_name: str = "auction"

    def solve(
        self,
        feasible_edges: pd.DataFrame,
        drivers: pd.DataFrame | None = None,
        orders: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        start = time.perf_counter()
        if feasible_edges.empty:
            return add_solver_metadata(feasible_edges.copy(), self.solver_name, time.perf_counter() - start)
        if self.epsilon <= 0:
            raise ValueError("Auction epsilon must be positive.")
        edges = feasible_edges.copy()
        edges["driver_id"] = edges["driver_id"].astype(str)
        edges["order_id"] = edges["order_id"].astype(str)
        edge_records = {
            (row.driver_id, row.order_id): row._asdict()
            for row in edges.itertuples(index=False)
        }
        driver_order_values: dict[str, dict[str, float]] = {}
        for row in edges.itertuples(index=False):
            driver_order_values.setdefault(row.driver_id, {})[row.order_id] = float(
                row.final_matching_weight
            )

        unassigned_drivers = sorted(driver_order_values)
        real_orders = sorted(edges["order_id"].unique())
        placeholder_orders = [f"__placeholder_order_{idx}" for idx in range(len(unassigned_drivers))]
        for driver_id in unassigned_drivers:
            for placeholder_order in placeholder_orders:
                driver_order_values[driver_id][placeholder_order] = 0.0
        prices = {order_id: 0.0 for order_id in real_orders + placeholder_orders}
        owner_by_order: dict[str, str] = {}
        order_by_driver: dict[str, str] = {}
        iterations = 0

        while unassigned_drivers:
            iterations += 1
            if iterations > self.max_iterations:
                raise AuctionDidNotConvergeError(
                    f"Auction did not converge within {self.max_iterations} iterations."
                )
            driver_id = unassigned_drivers.pop(0)
            values = driver_order_values.get(driver_id, {})
            if not values:
                continue
            net_values = sorted(
                ((value - prices[order_id], order_id) for order_id, value in values.items()),
                key=lambda item: (-item[0], item[1]),
            )
            best_value, best_order = net_values[0]
            second_best = net_values[1][0] if len(net_values) > 1 else 0.0
            bid = best_value - second_best + self.epsilon
            prices[best_order] += bid
            previous_driver = owner_by_order.get(best_order)
            if previous_driver is not None and previous_driver != driver_id:
                order_by_driver.pop(previous_driver, None)
                unassigned_drivers.append(previous_driver)
                unassigned_drivers.sort()
            owner_by_order[best_order] = driver_id
            order_by_driver[driver_id] = best_order

        selected = [
            edge_records[(driver_id, order_id)]
            for driver_id, order_id in sorted(order_by_driver.items(), key=lambda item: (item[1], item[0]))
            if (driver_id, order_id) in edge_records
        ]
        result = pd.DataFrame(selected)
        validate_one_to_one(result)
        return add_solver_metadata(result, self.solver_name, time.perf_counter() - start)


def solve_auction(edges: pd.DataFrame, epsilon: float = 0.001, max_iterations: int = 1_000_000) -> pd.DataFrame:
    return AuctionSolver(epsilon=epsilon, max_iterations=max_iterations).solve(edges)
