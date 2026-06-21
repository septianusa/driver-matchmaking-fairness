from __future__ import annotations

from typing import Protocol

import pandas as pd


class MatchingSolver(Protocol):
    solver_name: str

    def solve(
        self,
        feasible_edges: pd.DataFrame,
        drivers: pd.DataFrame | None = None,
        orders: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...


def add_solver_metadata(assignments: pd.DataFrame, solver_name: str, runtime_seconds: float) -> pd.DataFrame:
    result = assignments.copy()
    if result.empty:
        result = pd.DataFrame(columns=["driver_id", "order_id", "final_matching_weight"])
    result["solver_name"] = solver_name
    result["solver_runtime_seconds"] = float(runtime_seconds)
    return result


def validate_one_to_one(assignments: pd.DataFrame) -> None:
    if assignments.empty:
        return
    duplicate_drivers = assignments["driver_id"].astype(str).duplicated().any()
    duplicate_orders = assignments["order_id"].astype(str).duplicated().any()
    if duplicate_drivers or duplicate_orders:
        raise ValueError("Matching output violates one-to-one assignment constraints.")

