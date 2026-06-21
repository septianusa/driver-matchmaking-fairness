from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class GridValueStore:
    cold_start_value: float = 1.0
    update_mode: str = "sequential_td"
    values: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame | None, cold_start_value: float = 1.0, update_mode: str = "sequential_td"
    ) -> "GridValueStore":
        store = cls(cold_start_value=cold_start_value, update_mode=update_mode)
        if frame is not None and not frame.empty and {"h3_index", "grid_value"}.issubset(frame.columns):
            for row in frame.itertuples(index=False):
                store.values[str(row.h3_index)] = float(row.grid_value)
        return store

    def get(self, cell: str) -> float:
        cell = str(cell)
        if cell not in self.values:
            self.values[cell] = float(self.cold_start_value)
        return self.values[cell]

    def update_sequential_td(
        self,
        origin_cell: str,
        destination_cell: str,
        reward: float,
        *,
        gamma: float,
        alpha: float,
    ) -> float:
        current = self.get(origin_cell)
        next_value = self.get(destination_cell)
        td_error = float(reward) + float(gamma) * next_value - current
        self.values[str(origin_cell)] = current + float(alpha) * td_error
        return td_error

    def update_batch_mean_td(
        self,
        updates: list[tuple[str, str, float]],
        *,
        gamma: float,
        alpha: float,
    ) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for origin, destination, reward in updates:
            current = self.get(origin)
            td_error = float(reward) + float(gamma) * self.get(destination) - current
            grouped.setdefault(str(origin), []).append(td_error)
        applied = {}
        for origin, errors in sorted(grouped.items()):
            mean_error = sum(errors) / len(errors)
            self.values[origin] = self.get(origin) + float(alpha) * mean_error
            applied[origin] = mean_error
        return applied

    def literal_replication_update(self, origin_cell: str, destination_cell: str) -> None:
        self.values[str(origin_cell)] = self.get(destination_cell)

    def update_after_match(
        self,
        origin_cell: str,
        destination_cell: str,
        reward: float,
        *,
        gamma: float,
        alpha: float,
    ) -> float:
        if self.update_mode == "sequential_td":
            return self.update_sequential_td(origin_cell, destination_cell, reward, gamma=gamma, alpha=alpha)
        if self.update_mode == "literal_replication":
            self.literal_replication_update(origin_cell, destination_cell)
            return 0.0
        if self.update_mode == "none":
            self.get(origin_cell)
            self.get(destination_cell)
            return 0.0
        return self.update_sequential_td(origin_cell, destination_cell, reward, gamma=gamma, alpha=alpha)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"h3_index": cell, "grid_value": value} for cell, value in sorted(self.values.items())]
        )

