from __future__ import annotations

import math
from typing import Iterable

try:
    import h3  # type: ignore
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    h3 = None


def _fallback_step(resolution: int) -> float:
    return 1.0 / max(1, resolution * 100)


def latlon_to_h3(latitude: float, longitude: float, resolution: int) -> str:
    if h3 is not None:
        return str(h3.latlng_to_cell(float(latitude), float(longitude), int(resolution)))
    step = _fallback_step(resolution)
    lat_key = math.floor(float(latitude) / step)
    lon_key = math.floor(float(longitude) / step)
    return f"fallback:{resolution}:{lat_key}:{lon_key}"


def h3_to_latlon(cell: str) -> tuple[float, float]:
    if h3 is not None and not str(cell).startswith("fallback:"):
        lat, lon = h3.cell_to_latlng(cell)
        return float(lat), float(lon)
    _, resolution, lat_key, lon_key = str(cell).split(":")
    step = _fallback_step(int(resolution))
    return (int(lat_key) + 0.5) * step, (int(lon_key) + 0.5) * step


def grid_disk(cell: str, hops: int) -> set[str]:
    if h3 is not None and not str(cell).startswith("fallback:"):
        return {str(item) for item in h3.grid_disk(cell, int(hops))}
    prefix, resolution, lat_key, lon_key = str(cell).split(":")
    lat_i = int(lat_key)
    lon_i = int(lon_key)
    cells = set()
    for d_lat in range(-hops, hops + 1):
        for d_lon in range(-hops, hops + 1):
            if max(abs(d_lat), abs(d_lon)) <= hops:
                cells.add(f"{prefix}:{resolution}:{lat_i + d_lat}:{lon_i + d_lon}")
    return cells


def neighbors(cell: str) -> set[str]:
    return grid_disk(cell, 1) - {cell}


def breadth_first_cells(start_cell: str, max_hops: int) -> Iterable[tuple[str, int]]:
    visited = {start_cell}
    queue: list[tuple[str, int]] = [(start_cell, 0)]
    while queue:
        cell, depth = queue.pop(0)
        yield cell, depth
        if depth >= max_hops:
            continue
        for neighbor in sorted(neighbors(cell)):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

