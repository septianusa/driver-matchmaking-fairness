from __future__ import annotations

import math


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def eta_minutes(
    distance_km: float,
    speed_km_per_hour: float,
    minimum_minutes: int = 1,
    buffer_minutes: float = 0.0,
) -> int:
    if speed_km_per_hour <= 0:
        raise ValueError("Speed must be positive.")
    minutes = math.ceil((max(distance_km, 0.0) / speed_km_per_hour) * 60.0 + buffer_minutes)
    return max(int(minutes), int(minimum_minutes))

