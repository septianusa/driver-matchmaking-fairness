from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.matching.auction import solve_auction
from src.matching.greedy import solve_greedy

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    linear_sum_assignment = None


def _solve_exhaustive(edges: pd.DataFrame) -> pd.DataFrame:
    orders = sorted(edges["order_id"].astype(str).unique())
    drivers = sorted(edges["driver_id"].astype(str).unique())
    edge_map = {
        (str(row.order_id), str(row.driver_id)): row._asdict()
        for row in edges.itertuples(index=False)
    }
    best_weight = float("-inf")
    best_rows: list[dict] = []
    max_pairs = min(len(orders), len(drivers))
    for size in range(max_pairs + 1):
        for order_subset in itertools.combinations(orders, size):
            for driver_subset in itertools.permutations(drivers, size):
                rows = []
                total = 0.0
                feasible = True
                for order_id, driver_id in zip(order_subset, driver_subset):
                    edge = edge_map.get((order_id, driver_id))
                    if edge is None:
                        feasible = False
                        break
                    rows.append(edge)
                    total += float(edge["final_matching_weight"])
                if feasible and total > best_weight:
                    best_weight = total
                    best_rows = rows
    return pd.DataFrame(best_rows)


def solve_hungarian(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    if linear_sum_assignment is None:
        if edges["order_id"].nunique() <= 8 and edges["driver_id"].nunique() <= 8:
            return _solve_exhaustive(edges)
        return solve_greedy(edges)
    orders = sorted(edges["order_id"].astype(str).unique())
    drivers = sorted(edges["driver_id"].astype(str).unique())
    order_index = {order_id: idx for idx, order_id in enumerate(orders)}
    driver_index = {driver_id: idx for idx, driver_id in enumerate(drivers)}
    max_abs = max(float(edges["final_matching_weight"].abs().max()), 1.0)
    impossible = max_abs * 1_000_000.0
    matrix = np.full((len(orders), len(drivers)), impossible)
    rows_by_pair: dict[tuple[str, str], dict] = {}
    for row in edges.to_dict("records"):
        order_id = str(row["order_id"])
        driver_id = str(row["driver_id"])
        weight = float(row["final_matching_weight"])
        cost = -weight
        matrix[order_index[order_id], driver_index[driver_id]] = cost
        rows_by_pair[(order_id, driver_id)] = row
    row_indices, col_indices = linear_sum_assignment(matrix)
    selected = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        if matrix[row_idx, col_idx] >= impossible / 2:
            continue
        key = (orders[row_idx], drivers[col_idx])
        selected.append(rows_by_pair[key])
    return pd.DataFrame(selected)


def solve_assignment(edges: pd.DataFrame, strategy: str = "hungarian", config: dict | None = None) -> pd.DataFrame:
    if strategy == "hungarian":
        return solve_hungarian(edges)
    if strategy == "greedy":
        return solve_greedy(edges)
    if strategy == "auction":
        auction_cfg = ((config or {}).get("matching") or {}).get("auction", {})
        if not bool(auction_cfg.get("enabled", True)):
            raise RuntimeError("Auction matching is disabled in configuration.")
        return solve_auction(
            edges,
            epsilon=float(auction_cfg.get("epsilon", 0.001)),
            max_iterations=int(auction_cfg.get("max_iterations", 1_000_000)),
        )
    raise ValueError(f"Unsupported matching strategy: {strategy}")
