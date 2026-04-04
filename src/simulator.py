import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import (
    TOTAL_BATCH,
    ORDER_MAX_CARRYOVER_BATCH,
    MAX_MATCH_DISTANCE_KM,
    BFS_MAX_DEPTH,
    BFS_CANDIDATE_LIMIT,
    DISCOUNT_FACTOR,
    LEARNING_RATE_ALPHA,
    BETA_0,
    BETA_1,
    BETA_2,
    BETA_3,
)
from .utils import (
    haversine_km,
    compute_driver_score,
    compute_cancel_probability,
    compute_utility_base,
)
from .graph_search import build_h3_neighbors, get_candidate_drivers_by_bfs
from .metrics import build_summary_metrics


def run_replay_simulation(
    order_df: pd.DataFrame,
    ping_df: pd.DataFrame,
    driver_perf_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    lambda_driver_score: float,
    use_grid=True
) -> dict:
    orders = order_df.copy()
    pings = ping_df.copy()
    driver_perf = driver_perf_df.copy()
    grid = grid_df.copy()

    if "expireBatchStep" not in orders.columns:
        orders["expireBatchStep"] = orders["batchStep"] + ORDER_MAX_CARRYOVER_BATCH

    orders["isMatched"] = False
    orders["matchedBatchStep"] = np.nan

    driver_perf["driverScore"] = driver_perf.apply(
        lambda x: compute_driver_score(
            x["acceptanceRate"],
            x["completionRate"],
            x["onlineDurationHour"],
        ),
        axis=1,
    )
    perf_map = driver_perf.set_index("driverId").to_dict(orient="index")

    h3_neighbors = build_h3_neighbors(grid)
    grid_value_map = dict(zip(grid["h3Location"], grid["valueGrid"]))

    driver_busy_until: dict[int, int] = {}
    matched_records = []
    expired_records = []

    for t in range(1, TOTAL_BATCH + 1):
        # progress every 5%
        if t % (TOTAL_BATCH // 20) == 0:
            pct = (t / TOTAL_BATCH) * 100
            print(f"  Processing batch {t} / {TOTAL_BATCH} ({pct:.1f}%)")

        active_orders = orders[
            (orders["isMatched"] == False)
            & (orders["batchStep"] <= t)
            & (orders["expireBatchStep"] >= t)
        ].copy()

        expired_now = orders[
            (orders["isMatched"] == False)
            & (orders["expireBatchStep"] < t)
        ].copy()

        if not expired_now.empty:
            already_expired_ids = {r["orderId"] for r in expired_records}
            for _, row in expired_now.iterrows():
                if int(row["orderId"]) not in already_expired_ids:
                    expired_records.append(
                        {
                            "orderId": int(row["orderId"]),
                            "droppedAtBatchStep": t,
                            "initialBatchStep": int(row["batchStep"]),
                            "expireBatchStep": int(row["expireBatchStep"]),
                        }
                    )

        if active_orders.empty:
            continue

        busy_driver_ids = {driver_id for driver_id, busy_t in driver_busy_until.items() if busy_t > t}
        available_drivers = pings[
            (pings["batchStep"] == t)
            & (~pings["driverId"].isin(list(busy_driver_ids)))
        ].copy()

        if available_drivers.empty:
            continue

        candidates = []
        order_lookup = active_orders.set_index("orderId").to_dict(orient="index")

        for order_id, o in order_lookup.items():
            candidate_driver_pool = get_candidate_drivers_by_bfs(
                order_h3=o["h3Origin"],
                available_drivers_df=available_drivers,
                h3_neighbors=h3_neighbors,
                max_depth=BFS_MAX_DEPTH,
                candidate_limit=BFS_CANDIDATE_LIMIT,
            )

            if candidate_driver_pool.empty:
                continue

            for _, d in candidate_driver_pool.iterrows():
                driver_id = int(d["driverId"])
                perf = perf_map.get(driver_id)
                if perf is None:
                    continue

                dist_ji = float(
                    haversine_km(
                        o["latitudeOrigin"],
                        o["longitudeOrigin"],
                        d["avgLatitude"],
                        d["avgLongitude"],
                    )
                )

                if dist_ji > MAX_MATCH_DISTANCE_KM:
                    continue

                if use_grid:
                    value_origin = float(grid_value_map.get(o["h3Origin"], 1.0))
                    value_dest = float(grid_value_map.get(o["h3Destination"], 1.0))
                else:
                    value_origin = 1.0
                    value_dest = 1.0

                utility_base = compute_utility_base(
                    fare=float(o["fare"]),
                    value_dest=value_dest,
                    value_origin=value_origin,
                    discount_factor=DISCOUNT_FACTOR,
                )

                cancel_probability = compute_cancel_probability(
                    ar=float(perf["acceptanceRate"]),
                    cr=float(perf["completionRate"]),
                    dist_ji_km=dist_ji,
                    max_distance_km=MAX_MATCH_DISTANCE_KM,
                    beta_0=BETA_0,
                    beta_1=BETA_1,
                    beta_2=BETA_2,
                    beta_3=BETA_3,
                )

                driver_score = float(perf["driverScore"])

                weight = (
                    utility_base * (1 - lambda_driver_score) * (1 - cancel_probability)
                    + (lambda_driver_score * driver_score * utility_base)
                )

                candidates.append(
                    {
                        "driverId": driver_id,
                        "orderId": int(order_id),
                        "batchStep": t,
                        "weightDriverIdOrderId": float(weight),
                        "utilityBase": float(utility_base),
                        "cancelProbability": float(cancel_probability),
                        "driverScore": float(driver_score),
                        "distance_ji_km": float(dist_ji),
                        "fare": float(o["fare"]),
                        "h3Origin": o["h3Origin"],
                        "h3Destination": o["h3Destination"],
                        "estimatedTimeArrivalInBatch": int(o["estimatedTimeArrivalInBatch"]),
                    }
                )

        candidate_df = pd.DataFrame(candidates)
        if candidate_df.empty:
            continue

        candidate_df = (
            candidate_df.sort_values(
                ["orderId", "driverId", "weightDriverIdOrderId"],
                ascending=[True, True, False],
            )
            .drop_duplicates(subset=["driverId", "orderId"], keep="first")
            .reset_index(drop=True)
        )

        uniq_drivers = sorted(candidate_df["driverId"].unique())
        uniq_orders = sorted(candidate_df["orderId"].unique())

        driver_idx = {d: i for i, d in enumerate(uniq_drivers)}
        order_idx = {o: i for i, o in enumerate(uniq_orders)}

        cost_matrix = np.full((len(uniq_drivers), len(uniq_orders)), 10**12, dtype=float)

        for _, row in candidate_df.iterrows():
            cost_matrix[driver_idx[row["driverId"]], order_idx[row["orderId"]]] = -row["weightDriverIdOrderId"]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_rows = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 10**12:
                continue

            driver_id = uniq_drivers[r]
            order_id = uniq_orders[c]

            row = candidate_df[
                (candidate_df["driverId"] == driver_id)
                & (candidate_df["orderId"] == order_id)
            ].iloc[0].to_dict()

            matched_rows.append(row)

        if not matched_rows:
            continue

        for m in matched_rows:
            driver_id = int(m["driverId"])
            order_id = int(m["orderId"])
            eta = int(m["estimatedTimeArrivalInBatch"])

            driver_busy_until[driver_id] = t + eta

            orders.loc[orders["orderId"] == order_id, "isMatched"] = True
            orders.loc[orders["orderId"] == order_id, "matchedBatchStep"] = t

            matched_records.append(m)

        if use_grid:
            value_origin = float(grid_value_map.get(m["h3Origin"], 1.0))
            value_dest = float(grid_value_map.get(m["h3Destination"], 1.0))
            fare = float(m["fare"])

            new_origin_value = value_origin + LEARNING_RATE_ALPHA * (
                fare + DISCOUNT_FACTOR * value_dest - value_origin
            )
            grid_value_map[m["h3Origin"]] = new_origin_value

    match_df = pd.DataFrame(matched_records)
    expired_df = pd.DataFrame(expired_records)

    if not orders.empty:
        orders["simulation_day"] = ((orders["batchStep"] - 1) // 1440) + 1

    if not match_df.empty:
        match_df["simulation_day"] = ((match_df["batchStep"] - 1) // 1440) + 1

    summary, income_by_driver, hourly_summary = build_summary_metrics(
        orders=orders,
        match_df=match_df,
        driver_perf=driver_perf,
        lambda_driver_score=lambda_driver_score,
    )

    # daily utility
    if not match_df.empty:
        daily_utility = (
            match_df.groupby("simulation_day", as_index=False)
            .agg(total_utility=("utilityBase", "sum"))
        )
    else:
        daily_utility = pd.DataFrame(columns=["simulation_day", "total_utility"])

    # daily conversion
    daily_orders = (
        orders.groupby("simulation_day", as_index=False)
        .agg(total_orders=("orderId", "count"))
    )

    daily_matched = (
        orders[orders["isMatched"] == True]
        .groupby("simulation_day", as_index=False)
        .agg(total_matched=("orderId", "count"))
    )

    daily_conversion = daily_orders.merge(daily_matched, on="simulation_day", how="left")
    daily_conversion["total_matched"] = daily_conversion["total_matched"].fillna(0)
    daily_conversion["conversion_rate"] = daily_conversion["total_matched"] / daily_conversion["total_orders"]

    # daily pickup distance
    if not match_df.empty:
        daily_pickup = (
            match_df.groupby("simulation_day", as_index=False)
            .agg(avg_pickup_distance=("distance_ji_km", "mean"))
        )
    else:
        daily_pickup = pd.DataFrame(columns=["simulation_day", "avg_pickup_distance"])

    # daily correlation income vs score
    if not match_df.empty:
        daily_driver_income = (
            match_df.groupby(["simulation_day", "driverId"], as_index=False)
            .agg(total_income=("fare", "sum"))
        )

        daily_driver_income = daily_driver_income.merge(
            driver_perf[["driverId", "driverScore"]],
            on="driverId",
            how="left"
        )

        daily_corr = (
            daily_driver_income.groupby("simulation_day")
            .apply(lambda x: x["total_income"].corr(x["driverScore"]))
            .reset_index(name="corr_income_score")
        )
    else:
        daily_corr = pd.DataFrame(columns=["simulation_day", "corr_income_score"])

    return {
        "matchdataset": match_df,
        "order_dataset_final": orders,
        "expired_orders": expired_df,
        "income_by_driver": income_by_driver,
        "hourly_summary": hourly_summary,
        "daily_utility": daily_utility,
        "daily_conversion": daily_conversion,
        "daily_pickup": daily_pickup,
        "daily_corr_income_score": daily_corr,
        "summary": summary,
    }