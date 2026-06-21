from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DriverState:
    driver_id: str
    driver_score: float
    current_latitude: float
    current_longitude: float
    current_h3_index: str
    is_initialized: bool = True
    availability_status: str = "available"
    available_again_batch: int = 1
    consecutive_idle_batches: int = 0
    cumulative_active_minutes: int = 0
    position_source: str = "historical_bootstrap"
    has_simulated_assignment: bool = False
    simulated_destination_latitude: float | None = None
    simulated_destination_longitude: float | None = None
    simulated_destination_h3_index: str | None = None
    total_expected_income: float = 0.0
    total_realized_income: float = 0.0
    expected_completed_orders: float = 0.0
    realized_completed_orders: int = 0
    cancelled_orders: int = 0
    assigned_orders: int = 0

    def to_record(self) -> dict:
        return asdict(self)

