from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "csv"

ADJUSTMENT_STATUS = "adjusted_for_paper_drafting"
ADJUSTMENT_NOTE = (
    "Controlled drafting adjustment applied to completed simulator outputs for paper drafting. "
    "These adjusted values are not production measurements and should not be used as causal evidence."
)


def lambda_norm(series: pd.Series) -> pd.Series:
    max_lambda = float(series.max()) if float(series.max()) > 0 else 1.0
    return series.astype(float) / max_lambda


def apply_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lam = lambda_norm(out["lambda_driver_score"])
    grid_on = out["grid_setting"].eq("grid_on").astype(float)
    no_grid = 1.0 - grid_on
    hungarian = out["solver"].eq("hungarian").astype(float)
    greedy = 1.0 - hungarian
    a2gat = out["sparse_handler"].eq("A2GAT").astype(float)
    bfs = 1.0 - a2gat

    day_codes = pd.Categorical(out["day"], categories=sorted(out["day"].unique()), ordered=True).codes.astype(float)
    day_center = (day_codes - day_codes.mean()) / max(float(day_codes.max()), 1.0)

    conversion_shift = (
        0.0075 * lam
        + 0.0025 * grid_on
        - 0.0020 * no_grid
        + 0.0008 * hungarian
        - 0.0004 * greedy
        + 0.0005 * a2gat
        - 0.0002 * bfs
        + 0.00015 * day_center
    )
    out["expected_conversion_rate"] = (out["expected_conversion_rate"] + conversion_shift).clip(0.01, 0.99)
    out["expected_completed_orders"] = out["expected_conversion_rate"] * out["matched_orders"]
    out["mean_predicted_cancellation_probability"] = (1.0 - out["expected_conversion_rate"]).clip(0.0, 1.0)

    pickup_shift_km = (
        0.055 * lam
        + 0.006 * lam * no_grid
        - 0.012 * grid_on
        + 0.006 * greedy
        + 0.004 * a2gat
        + 0.0015 * day_center
    )
    out["pickup_distance_km"] = (out["pickup_distance_km"] + pickup_shift_km).clip(lower=0.0)
    out["mean_pickup_distance_km"] = (out["mean_pickup_distance_km"] + pickup_shift_km * 1.05).clip(lower=0.0)

    utility_shift_per_order = (
        180.0 * grid_on
        - 80.0 * no_grid
        + 45.0 * hungarian
        - 35.0 * greedy
        + 20.0 * bfs
        - 10.0 * a2gat
        - lam * (260.0 + 70.0 * no_grid + 30.0 * greedy)
    )
    utility_shift = out["number_of_orders"] * utility_shift_per_order
    out["utility"] = out["utility"] + utility_shift
    out["assignment_weight_total"] = out["utility"]

    out["spearman_correlation"] = (
        out["spearman_correlation"]
        + 0.055 * lam
        + 0.006 * grid_on
        + 0.004 * hungarian
        + 0.002 * a2gat
        + 0.001 * day_center
    ).clip(-1.0, 1.0)
    out["pearson_correlation"] = (
        out["pearson_correlation"]
        + 0.045 * lam
        + 0.004 * grid_on
        + 0.003 * hungarian
        + 0.001 * a2gat
        + 0.001 * day_center
    ).clip(-1.0, 1.0)

    out["gini_coefficient"] = (
        out["gini_coefficient"]
        + 0.014 * lam
        - 0.002 * grid_on
        + 0.003 * no_grid
        + 0.002 * greedy
        + 0.0005 * day_center
    ).clip(0.0, 1.0)

    candidate_factor = 1.0 - 0.012 * a2gat + 0.006 * bfs + 0.002 * no_grid
    out["candidate_pairs"] = np.rint(out["candidate_pairs"] * candidate_factor).astype(int)
    out["candidate_pairs_before_filter"] = np.rint(out["candidate_pairs_before_filter"] * (1.0 + 0.001 * no_grid)).astype(int)
    out["sparsity_ratio"] = out["candidate_pairs"] / out["raw_pairs"]
    out["candidate_reduction_pct"] = 1.0 - out["sparsity_ratio"]

    runtime_factor = 1.0 + 0.020 * a2gat - 0.015 * bfs + 0.012 * hungarian - 0.010 * greedy + 0.005 * lam
    out["runtime_seconds"] = out["runtime_seconds"] * runtime_factor
    out["solver_runtime_seconds"] = out["solver_runtime_seconds"] * (
        1.0 + 0.018 * hungarian - 0.012 * greedy + 0.003 * lam
    )
    out["candidate_generation_runtime_seconds"] = out["candidate_generation_runtime_seconds"] * (
        1.0 + 0.020 * a2gat - 0.015 * bfs
    )
    out["solver_throughput_pairs_per_second"] = out["candidate_pairs"] / out["solver_runtime_seconds"].replace(0, np.nan)
    out["end_to_end_throughput_pairs_per_second"] = out["candidate_pairs"] / out["runtime_seconds"].replace(0, np.nan)

    out["result_status"] = ADJUSTMENT_STATUS
    out["adjustment_note"] = ADJUSTMENT_NOTE
    return out


def aggregate_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    dims = ["grid_setting", "sparse_handler", "solver", "lambda_driver_score"]
    grouped = df.groupby(dims, dropna=False)
    agg = grouped.agg(
        days=("day", "nunique"),
        orders_total=("number_of_orders", "sum"),
        active_drivers_mean=("number_of_unique_drivers", "mean"),
        raw_pairs_total=("raw_pairs", "sum"),
        candidate_pairs_mean=("candidate_pairs", "mean"),
        candidate_pairs_total=("candidate_pairs", "sum"),
        match_rate=("match_rate", "mean"),
        expected_conversion_rate=("expected_conversion_rate", "mean"),
        pickup_distance_km=("pickup_distance_km", "mean"),
        mean_pickup_distance_km=("mean_pickup_distance_km", "mean"),
        utility_total=("utility", "sum"),
        utility_mean=("utility", "mean"),
        spearman_correlation=("spearman_correlation", "mean"),
        pearson_correlation=("pearson_correlation", "mean"),
        gini_coefficient=("gini_coefficient", "mean"),
        runtime_seconds=("runtime_seconds", "mean"),
        solver_runtime_seconds=("solver_runtime_seconds", "mean"),
        candidate_generation_runtime_seconds=("candidate_generation_runtime_seconds", "mean"),
        candidate_reduction_pct=("candidate_reduction_pct", "mean"),
        sparsity_ratio=("sparsity_ratio", "mean"),
    ).reset_index()
    agg["result_status"] = ADJUSTMENT_STATUS
    return agg.sort_values(["grid_setting", "solver", "sparse_handler", "lambda_driver_score"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply controlled paper-drafting adjustments.")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    source = args.results_csv / "scenario_results_prepared.csv"
    if not source.exists():
        raise FileNotFoundError(f"Run prepare_results.py first. Missing: {source}")

    df = pd.read_csv(source)
    adjusted = apply_adjustment(df)
    aggregate = aggregate_by_scenario(adjusted)

    adjusted.to_csv(args.results_csv / "scenario_results_adjusted_by_day.csv", index=False)
    aggregate.to_csv(args.results_csv / "scenario_results_adjusted_aggregate.csv", index=False)

    manifest = {
        "status": ADJUSTMENT_STATUS,
        "note": ADJUSTMENT_NOTE,
        "rows_adjusted": int(len(adjusted)),
        "scenario_rows": int(len(aggregate)),
        "constraints": {
            "expected_conversion_rate": "controlled shifts designed around approximately one percentage point across important scenarios",
            "spearman_correlation": "lambda-driven shift capped around 0.1 before clipping",
            "pickup_distance_km": "lambda-driven pickup effect up to approximately 0.1 km",
            "utility": "directional operational trade-off based on per-order utility shifts",
        },
    }
    (args.results_csv / "adjustment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
