"""Road-network acquisition and offline fallback graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from simulation_data_generation.config import GenerationConfig
from simulation_data_generation.spatial import haversine_km


@dataclass(frozen=True)
class RoadNetwork:
    """Container for a road graph and node coordinate lookup."""

    graph: nx.Graph
    node_positions: dict[str, tuple[float, float]]
    source_type: str


def _build_offline_graph(config: GenerationConfig) -> RoadNetwork:
    boundary = config.study_boundary
    rows = int(config.road_network.offline_grid_rows)
    cols = int(config.road_network.offline_grid_cols)
    graph = nx.Graph()
    node_positions: dict[str, tuple[float, float]] = {}

    for r in range(rows):
        lat = boundary.latitude_min + (boundary.latitude_max - boundary.latitude_min) * r / max(rows - 1, 1)
        for c in range(cols):
            lon = boundary.longitude_min + (boundary.longitude_max - boundary.longitude_min) * c / max(cols - 1, 1)
            node_id = f"N{r:02d}_{c:02d}"
            node_positions[node_id] = (float(lat), float(lon))
            graph.add_node(node_id, latitude=float(lat), longitude=float(lon))

    arterial_rows = {rows // 4, rows // 2, (3 * rows) // 4}
    arterial_cols = {cols // 4, cols // 2, (3 * cols) // 4}
    for r in range(rows):
        for c in range(cols):
            current = f"N{r:02d}_{c:02d}"
            neighbors = []
            if r + 1 < rows:
                neighbors.append((r + 1, c))
            if c + 1 < cols:
                neighbors.append((r, c + 1))
            if r + 1 < rows and c + 1 < cols and (r in arterial_rows or c in arterial_cols):
                neighbors.append((r + 1, c + 1))
            for nr, nc in neighbors:
                other = f"N{nr:02d}_{nc:02d}"
                lat1, lon1 = node_positions[current]
                lat2, lon2 = node_positions[other]
                distance = haversine_km(lat1, lon1, lat2, lon2)
                road_type = "arterial" if r in arterial_rows or c in arterial_cols else "local"
                graph.add_edge(current, other, distance_km=distance, weight=distance, road_type=road_type)

    return RoadNetwork(graph=graph, node_positions=node_positions, source_type="offline_fallback")


def build_road_network(config: GenerationConfig) -> RoadNetwork:
    """Build a road network using OSMnx when requested, otherwise use an offline fallback graph."""
    if config.road_network.use_osm_if_available:
        try:
            import osmnx as ox  # type: ignore

            cache_path = Path(config.road_network.cache_path)
            if cache_path.exists():
                graph = ox.load_graphml(cache_path)
            else:
                b = config.study_boundary
                graph = ox.graph_from_bbox(
                    north=b.latitude_max,
                    south=b.latitude_min,
                    east=b.longitude_max,
                    west=b.longitude_min,
                    network_type="drive",
                    simplify=True,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(graph, cache_path)
            undirected = nx.Graph(graph)
            positions = {
                str(node): (float(attrs["y"]), float(attrs["x"]))
                for node, attrs in undirected.nodes(data=True)
                if "x" in attrs and "y" in attrs
            }
            return RoadNetwork(graph=undirected, node_positions=positions, source_type="openstreetmap")
        except Exception:
            # The fallback is intentional and documented. The caller records source_type.
            return _build_offline_graph(config)
    return _build_offline_graph(config)


def nearest_node(network: RoadNetwork, latitude: float, longitude: float) -> str:
    """Return the nearest graph node id by haversine distance."""
    best_node = None
    best_distance = float("inf")
    for node_id, (node_lat, node_lon) in network.node_positions.items():
        distance = haversine_km(latitude, longitude, node_lat, node_lon)
        if distance < best_distance:
            best_node = node_id
            best_distance = distance
    if best_node is None:
        raise ValueError("Road network has no nodes.")
    return str(best_node)


def shortest_path_nodes(network: RoadNetwork, source: str, target: str) -> list[str]:
    """Return a shortest path between two nodes, falling back to direct endpoints if needed."""
    if source == target:
        return [source]
    try:
        return [str(node) for node in nx.shortest_path(network.graph, source=source, target=target, weight="weight")]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [source, target]


def path_distance_km(network: RoadNetwork, path: list[str]) -> float:
    """Return distance along a graph path."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(path, path[1:], strict=False):
        if network.graph.has_edge(left, right):
            total += float(network.graph[left][right].get("distance_km", 0.0))
        else:
            lat1, lon1 = network.node_positions[left]
            lat2, lon2 = network.node_positions[right]
            total += haversine_km(lat1, lon1, lat2, lon2)
    return total
