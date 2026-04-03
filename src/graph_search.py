from collections import deque
import pandas as pd
from .utils import haversine_km


def build_h3_neighbors(grid_df: pd.DataFrame, k_neighbors: int = 6) -> dict[str, list[str]]:
    centers = grid_df[["h3Location", "centerLat", "centerLon"]].copy()
    neighbors: dict[str, list[str]] = {}

    for _, row in centers.iterrows():
        current_h3 = row["h3Location"]
        lat1, lon1 = row["centerLat"], row["centerLon"]

        temp = centers[centers["h3Location"] != current_h3].copy()
        temp["dist"] = temp.apply(
            lambda x: haversine_km(lat1, lon1, x["centerLat"], x["centerLon"]),
            axis=1,
        )
        neighbors[current_h3] = temp.sort_values("dist").head(k_neighbors)["h3Location"].tolist()

    return neighbors


def bfs_h3_cells(start_h3: str, h3_neighbors: dict[str, list[str]], max_depth: int = 2):
    visited = {start_h3}
    queue = deque([(start_h3, 0)])
    result = []

    while queue:
        current_h3, depth = queue.popleft()
        result.append((current_h3, depth))

        if depth == max_depth:
            continue

        for nxt in h3_neighbors.get(current_h3, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))

    return result


def get_candidate_drivers_by_bfs(
    order_h3: str,
    available_drivers_df: pd.DataFrame,
    h3_neighbors: dict[str, list[str]],
    max_depth: int = 2,
    candidate_limit: int = 10,
) -> pd.DataFrame:
    visited = {order_h3}
    queue = deque([(order_h3, 0)])
    collected = []
    found = 0

    while queue:
        current_h3, depth = queue.popleft()

        drivers_in_cell = available_drivers_df[available_drivers_df["h3location"] == current_h3]
        if not drivers_in_cell.empty:
            collected.append(drivers_in_cell)
            found += len(drivers_in_cell)
            if candidate_limit is not None and found >= candidate_limit:
                break

        if depth == max_depth:
            continue

        for nxt in h3_neighbors.get(current_h3, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))

    if not collected:
        return available_drivers_df.iloc[0:0].copy()

    return (
        pd.concat(collected, ignore_index=True)
        .drop_duplicates(subset=["driverId"])
        .head(candidate_limit if candidate_limit is not None else 10**9)
        .reset_index(drop=True)
    )