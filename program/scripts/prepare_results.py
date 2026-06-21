from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_REPO = PROJECT_ROOT.parent / "ride_hailing_dispatch_simulator"
RESULTS_CSV = PROJECT_ROOT / "results" / "csv"


SOURCE_RUNS = [
    {
        "grid_setting": "grid_on",
        "solver": "hungarian",
        "relative_path": "outputs/model_comparison_days_20260607_100820/comparison/comparison_summary_by_day.csv",
    },
    {
        "grid_setting": "grid_off",
        "solver": "hungarian",
        "relative_path": "outputs/model_comparison_days_no_grid_20260607_221133/comparison/comparison_summary_by_day.csv",
    },
    {
        "grid_setting": "grid_on",
        "solver": "greedy",
        "relative_path": "outputs/model_comparison_days_greedy_grid_20260613_233414/comparison/comparison_summary_by_day.csv",
    },
    {
        "grid_setting": "grid_off",
        "solver": "greedy",
        "relative_path": "outputs/model_comparison_days_greedy_no_grid_20260614_084846/comparison/comparison_summary_by_day.csv",
    },
]


RENAME_MAP = {
    "simulation_date": "day",
    "number_of_orders": "number_of_orders",
    "number_of_unique_drivers": "number_of_unique_drivers",
    "number_of_batches": "number_of_batches",
    "raw_cartesian_pair_count_total_mean": "raw_pairs",
    "candidate_pair_count_before_filter_total_mean": "candidate_pairs_before_filter",
    "candidate_pair_count_after_filter_total_mean": "candidate_pairs",
    "sparsity_ratio_mean": "sparsity_ratio",
    "candidate_reduction_pct_mean": "candidate_reduction_pct",
    "candidate_generation_runtime_seconds_mean": "candidate_generation_runtime_seconds",
    "solver_runtime_seconds_mean": "solver_runtime_seconds",
    "end_to_end_runtime_seconds_mean": "runtime_seconds",
    "peak_memory_mb_mean": "peak_memory_mb",
    "solver_throughput_pairs_per_second_mean": "solver_throughput_pairs_per_second",
    "end_to_end_throughput_pairs_per_second_mean": "end_to_end_throughput_pairs_per_second",
    "matched_order_count_mean": "matched_orders",
    "expected_completed_order_count_mean": "expected_completed_orders",
    "match_rate_mean": "match_rate",
    "expected_conversion_rate_mean": "expected_conversion_rate",
    "expired_unfulfilled_rate_mean": "expired_unfulfilled_rate",
    "assignment_weight_total_mean": "assignment_weight_total",
    "expected_economic_utility_total_mean": "utility",
    "median_pickup_distance_km_mean": "pickup_distance_km",
    "mean_pickup_distance_km_mean": "mean_pickup_distance_km",
    "mean_predicted_cancellation_probability_mean": "mean_predicted_cancellation_probability",
    "pearson_driver_score_income_mean": "pearson_correlation",
    "spearman_driver_score_income_mean": "spearman_correlation",
    "income_gini_mean": "gini_coefficient",
}


OUTPUT_COLUMNS = [
    "day",
    "grid_setting",
    "sparse_handler",
    "solver",
    "lambda_driver_score",
    "result_status",
    "successful_run_count",
    "skipped_run_count",
    "failed_run_count",
    "number_of_orders",
    "number_of_unique_drivers",
    "number_of_batches",
    "raw_pairs",
    "candidate_pairs_before_filter",
    "candidate_pairs",
    "sparsity_ratio",
    "candidate_reduction_pct",
    "matched_orders",
    "match_rate",
    "expected_completed_orders",
    "expected_conversion_rate",
    "expired_unfulfilled_rate",
    "pickup_distance_km",
    "mean_pickup_distance_km",
    "utility",
    "assignment_weight_total",
    "spearman_correlation",
    "pearson_correlation",
    "gini_coefficient",
    "runtime_seconds",
    "solver_runtime_seconds",
    "candidate_generation_runtime_seconds",
    "peak_memory_mb",
    "solver_throughput_pairs_per_second",
    "end_to_end_throughput_pairs_per_second",
    "source_run_folder",
    "source_file",
]


def normalize_sparse(value: str) -> str:
    value = str(value).lower()
    if value in {"bfs", "bfs_h3", "bfsh3"}:
        return "BFS"
    if value in {"a2gat", "adaptive_anchor"}:
        return "A2GAT"
    return value.upper()


def read_source_runs(source_repo: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for spec in SOURCE_RUNS:
        path = source_repo / spec["relative_path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing source comparison summary: {path}")
        raw = pd.read_csv(path)
        solver = spec["solver"]
        filtered = raw[raw["matching_algorithm"].str.lower() == solver].copy()
        filtered["grid_setting"] = spec["grid_setting"]
        filtered["source_run_folder"] = Path(spec["relative_path"]).parts[1]
        filtered["source_file"] = str(path)
        frames.append(filtered)
        manifest.append(
            {
                "grid_setting": spec["grid_setting"],
                "solver": solver,
                "source_file": str(path),
                "raw_rows": int(len(raw)),
                "used_rows": int(len(filtered)),
            }
        )
    return pd.concat(frames, ignore_index=True), manifest


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=RENAME_MAP).copy()
    out["sparse_handler"] = out["sparse_method"].map(normalize_sparse)
    out["solver"] = out["matching_algorithm"].str.lower()
    out["lambda_driver_score"] = out["lambda_driver_score"].astype(float).round(3)
    out["result_status"] = "unadjusted_simulation_output"
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].sort_values(
        ["day", "grid_setting", "solver", "sparse_handler", "lambda_driver_score"]
    )
    return out.reset_index(drop=True)


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    days = sorted(df["day"].dropna().unique())
    grid_values = ["grid_on", "grid_off"]
    sparse_values = ["BFS", "A2GAT"]
    solvers = ["hungarian", "greedy"]
    lambdas = [0.0, 0.1, 0.2, 0.3]
    expected = pd.MultiIndex.from_product(
        [days, grid_values, sparse_values, solvers, lambdas],
        names=["day", "grid_setting", "sparse_handler", "solver", "lambda_driver_score"],
    ).to_frame(index=False)
    observed = (
        df.groupby(["day", "grid_setting", "sparse_handler", "solver", "lambda_driver_score"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    coverage = expected.merge(
        observed,
        how="left",
        on=["day", "grid_setting", "sparse_handler", "solver", "lambda_driver_score"],
    )
    coverage["row_count"] = coverage["row_count"].fillna(0).astype(int)
    coverage["covered"] = coverage["row_count"].eq(1)
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize current dispatch experiment summaries.")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    args.results_csv.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_source_runs(args.source_repo)
    prepared = normalize_columns(raw)
    coverage = build_coverage(prepared)

    raw.to_csv(args.results_csv / "scenario_results_raw_merged.csv", index=False)
    prepared.to_csv(args.results_csv / "scenario_results_prepared.csv", index=False)
    coverage.to_csv(args.results_csv / "scenario_coverage_check.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.results_csv / "source_manifest.csv", index=False)

    summary = {
        "source_repo": str(args.source_repo),
        "prepared_rows": int(len(prepared)),
        "covered_combinations": int(coverage["covered"].sum()),
        "expected_combinations": int(len(coverage)),
        "days": sorted(prepared["day"].unique().tolist()),
        "grid_settings": sorted(prepared["grid_setting"].unique().tolist()),
        "sparse_handlers": sorted(prepared["sparse_handler"].unique().tolist()),
        "solvers": sorted(prepared["solver"].unique().tolist()),
        "lambda_values": sorted(prepared["lambda_driver_score"].unique().tolist()),
    }
    (args.results_csv / "prepare_results_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["covered_combinations"] != summary["expected_combinations"]:
        raise SystemExit("Scenario coverage is incomplete. Inspect scenario_coverage_check.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

