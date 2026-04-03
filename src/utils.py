import math
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_driver_score(acceptance_rate: float, completion_rate: float, online_duration_hour: float) -> float:
    return (
        acceptance_rate / 3.0
        + completion_rate / 3.0
        + (min(online_duration_hour, 40.0) / 40.0) / 3.0
    )


def compute_cancel_probability(
    ar: float,
    cr: float,
    dist_ji_km: float,
    max_distance_km: float,
    beta_0: float,
    beta_1: float,
    beta_2: float,
    beta_3: float,
) -> float:
    dist_norm = clamp01(dist_ji_km / max_distance_km)
    z = beta_0 + beta_1 * (1 - ar) + beta_2 * (1 - cr) + beta_3 * dist_norm
    return logistic(z)


def compute_utility_base(
    fare: float,
    value_dest: float,
    value_origin: float,
    discount_factor: float,
) -> float:
    return max(0.0, fare + (fare + discount_factor * value_dest - value_origin))