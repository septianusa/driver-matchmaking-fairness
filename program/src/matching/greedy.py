from __future__ import annotations

import pandas as pd


def solve_greedy(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    ordered = edges.sort_values(
        ["final_matching_weight", "driver_id", "order_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    used_orders: set[str] = set()
    used_drivers: set[str] = set()
    rows = []
    for row in ordered.to_dict("records"):
        order_id = str(row["order_id"])
        driver_id = str(row["driver_id"])
        if order_id in used_orders or driver_id in used_drivers:
            continue
        used_orders.add(order_id)
        used_drivers.add(driver_id)
        rows.append(row)
    return pd.DataFrame(rows)
