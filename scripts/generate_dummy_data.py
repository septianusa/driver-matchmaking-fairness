from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# =========================================================
# CONFIG
# =========================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TOTAL_BATCH = 1440
TOTAL_DEMAND = 100000
TOTAL_DRIVERS = 14000

CENTER_LAT = -7.2575
CENTER_LON = 112.7521

# Surabaya bounding box
LAT_MIN, LAT_MAX = -7.40, -7.16
LON_MIN, LON_MAX = 112.60, 112.90

OUTDIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def batch_to_hour(batch_step: int) -> int:
    return (batch_step - 1) // 60


def sample_point_around(lat, lon, sd_km=1.2):
    dy = np.random.normal(0, sd_km) / 111.0
    dx = np.random.normal(0, sd_km) / (111.0 * math.cos(math.radians(lat)))
    new_lat = np.clip(lat + dy, LAT_MIN, LAT_MAX)
    new_lon = np.clip(lon + dx, LON_MIN, LON_MAX)
    return new_lat, new_lon


def choose_hotspot(hour: int, origin: bool = True):
    morning_origin = [
        ("west_residential", -7.286, 112.674, 0.25),
        ("east_residential", -7.284, 112.804, 0.25),
        ("south_residential", -7.324, 112.730, 0.22),
        ("north_residential", -7.213, 112.744, 0.10),
        ("central", -7.265, 112.742, 0.18),
    ]
    morning_dest = [
        ("cbd", -7.265, 112.742, 0.45),
        ("tunjungan", -7.258, 112.738, 0.25),
        ("east_office", -7.274, 112.780, 0.15),
        ("north_port", -7.205, 112.734, 0.05),
        ("south_office", -7.304, 112.739, 0.10),
    ]
    daytime_mix = [
        ("cbd", -7.265, 112.742, 0.30),
        ("west", -7.286, 112.674, 0.12),
        ("east", -7.284, 112.804, 0.18),
        ("south", -7.324, 112.730, 0.18),
        ("north", -7.213, 112.744, 0.08),
        ("mall_area", -7.276, 112.782, 0.14),
    ]
    evening_origin = morning_dest
    evening_dest = morning_origin
    late_night = [
        ("central", -7.265, 112.742, 0.25),
        ("west_residential", -7.286, 112.674, 0.18),
        ("east_residential", -7.284, 112.804, 0.18),
        ("south_residential", -7.324, 112.730, 0.20),
        ("leisure", -7.272, 112.756, 0.19),
    ]

    if 6 <= hour <= 8:
        pool = morning_origin if origin else morning_dest
    elif 16 <= hour <= 19:
        pool = evening_origin if origin else evening_dest
    elif 9 <= hour <= 15:
        pool = daytime_mix
    else:
        pool = late_night

    probs = np.array([p[3] for p in pool], dtype=float)
    probs = probs / probs.sum()
    idx = np.random.choice(len(pool), p=probs)
    return pool[idx]


# =========================================================
# H3-LIKE GRID
# =========================================================
def create_hex_grid(lat_min, lat_max, lon_min, lon_max, radius_deg=0.012):
    dx = math.sqrt(3) * radius_deg
    dy = 1.5 * radius_deg

    centers = []
    row = 0
    lat = lat_min
    while lat <= lat_max + radius_deg:
        lon_offset = dx / 2 if row % 2 else 0
        lon = lon_min + lon_offset
        while lon <= lon_max + dx:
            centers.append((lat, lon))
            lon += dx
        lat += dy
        row += 1

    rows = []
    for i, (lat, lon) in enumerate(centers, start=1):
        rows.append(
            {
                "locationId": i,
                "h3Location": f"h3_{i:04d}",
                "centerLat": lat,
                "centerLon": lon,
                "valueGrid": 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_h3_assigner(grid_df: pd.DataFrame):
    tree = cKDTree(grid_df[["centerLat", "centerLon"]].to_numpy())

    def assign(lat_arr, lon_arr):
        _, idx = tree.query(np.column_stack([lat_arr, lon_arr]), k=1)
        return (
            grid_df.iloc[idx]["h3Location"].to_numpy(),
            grid_df.iloc[idx]["locationId"].to_numpy(),
        )

    return assign


# =========================================================
# GENERATORS
# =========================================================
def create_order_dataset(grid_df: pd.DataFrame) -> pd.DataFrame:
    hourly_weights = np.array([
        0.18, 0.14, 0.11, 0.10, 0.14, 0.30,
        0.62, 0.96, 0.84, 0.58, 0.52, 0.56,
        0.64, 0.70, 0.76, 0.86, 1.00, 0.96,
        0.82, 0.68, 0.56, 0.44, 0.32, 0.24
    ], dtype=float)

    minute_weights = np.repeat(hourly_weights, 60)
    minute_weights = minute_weights / minute_weights.sum()

    order_batches = np.random.choice(
        np.arange(1, TOTAL_BATCH + 1),
        size=TOTAL_DEMAND,
        p=minute_weights,
    )
    order_batches.sort()

    rows = []
    for order_id, batch_step in enumerate(order_batches, start=1):
        hour = batch_to_hour(int(batch_step))

        _, o_lat_center, o_lon_center, _ = choose_hotspot(hour, origin=True)
        _, d_lat_center, d_lon_center, _ = choose_hotspot(hour, origin=False)

        o_lat, o_lon = sample_point_around(o_lat_center, o_lon_center, sd_km=np.random.uniform(0.5, 2.0))
        d_lat, d_lon = sample_point_around(d_lat_center, d_lon_center, sd_km=np.random.uniform(0.5, 2.2))

        trip_km = float(haversine_km(o_lat, o_lon, d_lat, d_lon))
        fare = round(7000 + trip_km * 2200 + np.random.uniform(0, 4000), 2)
        eta_batch = int(np.clip(round(trip_km * 3.2 + np.random.uniform(4, 10)), 3, 40))

        rows.append(
            {
                "orderId": order_id,
                "customerId": np.random.randint(1, 7001),
                "latitudeOrigin": o_lat,
                "longitudeOrigin": o_lon,
                "latitudeDestination": d_lat,
                "longitudeDestination": d_lon,
                "fare": fare,
                "estimatedTimeArrivalInBatch": eta_batch,
                "batchStep": int(batch_step),
            }
        )

    order_df = pd.DataFrame(rows)
    assign_h3 = build_h3_assigner(grid_df)

    order_df["h3Origin"], _ = assign_h3(
        order_df["latitudeOrigin"].to_numpy(),
        order_df["longitudeOrigin"].to_numpy(),
    )
    order_df["h3Destination"], _ = assign_h3(
        order_df["latitudeDestination"].to_numpy(),
        order_df["longitudeDestination"].to_numpy(),
    )

    return order_df[
        [
            "orderId",
            "customerId",
            "latitudeOrigin",
            "longitudeOrigin",
            "h3Origin",
            "latitudeDestination",
            "longitudeDestination",
            "h3Destination",
            "fare",
            "estimatedTimeArrivalInBatch",
            "batchStep",
        ]
    ].sort_values(["batchStep", "orderId"]).reset_index(drop=True)


def create_ping_and_driver_perf_dataset(grid_df: pd.DataFrame):
    assign_h3 = build_h3_assigner(grid_df)

    start_hour_weights = np.array([
        0.10, 0.08, 0.06, 0.05, 0.06, 0.16,
        0.40, 0.55, 0.45, 0.28, 0.26, 0.30,
        0.36, 0.38, 0.42, 0.54, 0.64, 0.60,
        0.40, 0.24, 0.16, 0.12, 0.10, 0.08
    ], dtype=float)
    start_minute_weights = np.repeat(start_hour_weights, 60)
    start_minute_weights = start_minute_weights / start_minute_weights.sum()

    ping_chunks = []
    perf_rows = []

    for driver_id in range(1, TOTAL_DRIVERS + 1):
        start_batch = int(np.random.choice(np.arange(1, TOTAL_BATCH + 1), p=start_minute_weights))
        shift_length = int(np.clip(np.random.normal(300, 75), 120, 480))
        end_batch = min(TOTAL_BATCH, start_batch + shift_length - 1)

        start_hour = batch_to_hour(start_batch)
        _, lat_center, lon_center, _ = choose_hotspot(start_hour, origin=True)
        lat0, lon0 = sample_point_around(lat_center, lon_center, sd_km=np.random.uniform(0.4, 1.5))

        n = end_batch - start_batch + 1
        batches = np.arange(start_batch, end_batch + 1)

        lat_steps = np.random.normal(0, 0.12 / 111.0, size=n)
        lon_steps = np.random.normal(0, 0.12 / (111.0 * math.cos(math.radians(CENTER_LAT))), size=n)

        lat_path = np.clip(lat0 + np.cumsum(lat_steps), LAT_MIN, LAT_MAX)
        lon_path = np.clip(lon0 + np.cumsum(lon_steps), LON_MIN, LON_MAX)

        h3_vals, _ = assign_h3(lat_path, lon_path)

        ping_chunks.append(
            pd.DataFrame(
                {
                    "driverId": driver_id,
                    "batchStep": batches,
                    "avgLatitude": lat_path,
                    "avgLongitude": lon_path,
                    "h3location": h3_vals,
                }
            )
        )

        acceptance_rate = round(np.random.uniform(0.70, 0.98), 4)
        completion_rate = round(np.random.uniform(0.75, 0.99), 4)
        online_duration_hour = round(n / 60.0, 2)
        driver_score = round(
            acceptance_rate / 3.0
            + completion_rate / 3.0
            + (min(online_duration_hour, 40.0) / 40.0) / 3.0,
            6,
        )

        perf_rows.append(
            {
                "driverId": driver_id,
                "acceptanceRate": acceptance_rate,
                "completionRate": completion_rate,
                "onlineDurationHour": online_duration_hour,
                "driverScore": driver_score,
            }
        )

    ping_df = pd.concat(ping_chunks, ignore_index=True).sort_values(["batchStep", "driverId"]).reset_index(drop=True)
    perf_df = pd.DataFrame(perf_rows).sort_values("driverId").reset_index(drop=True)

    return ping_df, perf_df


# =========================================================
# MAIN
# =========================================================
def main():
    print("Generating h3GridValue_dataset ...")
    grid_df = create_hex_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, radius_deg=0.012)

    print("Generating order_dataset ...")
    order_df = create_order_dataset(grid_df)

    print("Generating ping_dataset and driverPerformance_dataset ...")
    ping_df, perf_df = create_ping_and_driver_perf_dataset(grid_df)

    order_file = OUTDIR / "order_dataset_surabaya_1day.csv"
    ping_file = OUTDIR / "ping_dataset_surabaya_1day.csv"
    perf_file = OUTDIR / "driverPerformance_dataset_surabaya_1day.csv"
    grid_file = OUTDIR / "h3GridValue_dataset_surabaya_1day.csv"

    order_df.to_csv(order_file, index=False)
    ping_df.to_csv(ping_file, index=False)
    perf_df.to_csv(perf_file, index=False)
    grid_df.to_csv(grid_file, index=False)

    print("\nDone.")
    print(order_file)
    print(ping_file)
    print(perf_file)
    print(grid_file)

    print("\nSummary:")
    print(f"order rows   : {len(order_df):,}")
    print(f"ping rows    : {len(ping_df):,}")
    print(f"driver rows  : {len(perf_df):,}")
    print(f"grid rows    : {len(grid_df):,}")


if __name__ == "__main__":
    main()