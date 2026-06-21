from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator in {0, 0.0} or denominator is None or pd.isna(denominator):
        return None
    return float(numerator) / float(denominator)


def enrich_run_metrics(row: dict) -> dict:
    raw_pairs = float(row.get("raw_cartesian_pair_count_total") or 0.0)
    after = float(row.get("candidate_pair_count_after_filter_total") or 0.0)
    solver_runtime = float(row.get("solver_runtime_seconds") or 0.0)
    end_runtime = float(row.get("end_to_end_runtime_seconds") or 0.0)
    sparsity = safe_divide(after, raw_pairs)
    row["sparsity_ratio"] = sparsity
    row["candidate_reduction_pct"] = None if sparsity is None else 1.0 - sparsity
    row["solver_throughput_pairs_per_second"] = safe_divide(after, solver_runtime)
    row["end_to_end_throughput_pairs_per_second"] = safe_divide(after, end_runtime)
    return row


def _p90(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float).dropna()
    return float(series.quantile(0.90)) if not series.empty else 0.0


def _p95(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float).dropna()
    return float(series.quantile(0.95)) if not series.empty else 0.0


def summarize_runs(raw_runs: pd.DataFrame) -> pd.DataFrame:
    if raw_runs.empty:
        return pd.DataFrame()
    measured = raw_runs[raw_runs["warmup_flag"] == False].copy()  # noqa: E712
    group_cols = ["lambda_driver_score", "matching_algorithm", "sparse_method"]
    rows = []
    for keys, group in measured.groupby(group_cols, dropna=False):
        lambda_score, algorithm, sparse = keys
        successful = group[group["status"] == "success"]
        row = {
            "lambda_driver_score": lambda_score,
            "matching_algorithm": algorithm,
            "sparse_method": sparse,
            "successful_run_count": int((group["status"] == "success").sum()),
            "skipped_run_count": int((group["status"] == "skipped").sum()),
            "failed_run_count": int((group["status"] == "failed").sum()),
        }
        first = group.iloc[0].to_dict()
        for col in ["number_of_orders", "number_of_unique_drivers", "number_of_batches"]:
            row[col] = first.get(col)
        metric_map = {
            "raw_cartesian_pair_count_total": ["mean"],
            "candidate_pair_count_before_filter_total": ["mean"],
            "candidate_pair_count_after_filter_total": ["mean"],
            "sparsity_ratio": ["mean"],
            "candidate_reduction_pct": ["mean"],
            "candidate_generation_runtime_seconds": ["mean", "median", "std", "p90", "p95"],
            "solver_runtime_seconds": ["mean", "median", "std", "p90", "p95"],
            "end_to_end_runtime_seconds": ["mean", "median", "std", "p90", "p95"],
            "peak_memory_mb": ["mean", "p95"],
            "solver_throughput_pairs_per_second": ["mean"],
            "end_to_end_throughput_pairs_per_second": ["mean"],
            "matched_order_count": ["mean"],
            "expected_completed_order_count": ["mean"],
            "match_rate": ["mean"],
            "expected_conversion_rate": ["mean"],
            "expired_unfulfilled_rate": ["mean"],
            "assignment_weight_total": ["mean"],
            "expected_economic_utility_total": ["mean"],
            "median_pickup_distance_km": ["mean"],
            "mean_pickup_distance_km": ["mean"],
            "mean_predicted_cancellation_probability": ["mean"],
            "pearson_driver_score_income": ["mean"],
            "spearman_driver_score_income": ["mean"],
            "income_gini": ["mean"],
        }
        source = successful if not successful.empty else group
        for metric, stats in metric_map.items():
            values = pd.to_numeric(source.get(metric, pd.Series(dtype=float)), errors="coerce")
            for stat in stats:
                name = f"{metric}_{stat}"
                if values.empty:
                    row[name] = math.nan
                elif stat == "mean":
                    row[name] = float(values.mean())
                elif stat == "median":
                    row[name] = float(values.median())
                elif stat == "std":
                    row[name] = float(values.std(ddof=0)) if len(values) else 0.0
                elif stat == "min":
                    row[name] = float(values.min())
                elif stat == "max":
                    row[name] = float(values.max())
                elif stat == "p90":
                    row[name] = _p90(values)
                elif stat == "p95":
                    row[name] = _p95(values)
        rows.append(row)
    summary = pd.DataFrame(rows)
    alias_prefixes = {
        "candidate_generation_runtime_seconds": "candidate_generation_runtime",
        "solver_runtime_seconds": "solver_runtime",
        "end_to_end_runtime_seconds": "end_to_end_runtime",
    }
    for old_prefix, new_prefix in alias_prefixes.items():
        for stat in ["mean", "median", "std", "p90", "p95"]:
            old_col = f"{old_prefix}_{stat}"
            new_col = f"{new_prefix}_{stat}"
            if old_col in summary.columns and new_col not in summary.columns:
                summary[new_col] = summary[old_col]
    return summary


def baseline_relative(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[
        (summary["lambda_driver_score"].astype(float) == 0.0)
        & (summary["matching_algorithm"] == "hungarian")
        & (summary["sparse_method"] == "bfs_h3")
    ]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    rows = []
    for row in summary.to_dict("records"):
        enriched = dict(row)
        enriched["runtime_ratio_vs_baseline"] = safe_divide(
            row.get("end_to_end_runtime_seconds_mean"), base.get("end_to_end_runtime_seconds_mean")
        )
        enriched["runtime_speedup_vs_baseline"] = safe_divide(
            base.get("end_to_end_runtime_seconds_mean"), row.get("end_to_end_runtime_seconds_mean")
        )
        enriched["memory_ratio_vs_baseline"] = safe_divide(
            row.get("peak_memory_mb_mean"), base.get("peak_memory_mb_mean")
        )
        enriched["assignment_quality_ratio_vs_baseline"] = safe_divide(
            row.get("assignment_weight_total_mean"), base.get("assignment_weight_total_mean")
        )
        enriched["expected_utility_ratio_vs_baseline"] = safe_divide(
            row.get("expected_economic_utility_total_mean"), base.get("expected_economic_utility_total_mean")
        )
        enriched["match_rate_delta_vs_baseline"] = (
            row.get("match_rate_mean") - base.get("match_rate_mean")
            if pd.notna(row.get("match_rate_mean")) and pd.notna(base.get("match_rate_mean"))
            else None
        )
        enriched["median_pickup_distance_delta_vs_baseline"] = (
            row.get("median_pickup_distance_km_mean") - base.get("median_pickup_distance_km_mean")
            if pd.notna(row.get("median_pickup_distance_km_mean"))
            and pd.notna(base.get("median_pickup_distance_km_mean"))
            else None
        )
        enriched["score_income_correlation_delta_vs_baseline"] = (
            row.get("spearman_driver_score_income_mean") - base.get("spearman_driver_score_income_mean")
            if pd.notna(row.get("spearman_driver_score_income_mean"))
            and pd.notna(base.get("spearman_driver_score_income_mean"))
            else None
        )
        rows.append(enriched)
    return pd.DataFrame(rows)


def algorithm_quality(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["lambda_driver_score", "sparse_method"]
    for keys, group in summary.groupby(key_cols, dropna=False):
        lambda_score, sparse = keys
        hungarian = group[group["matching_algorithm"] == "hungarian"]
        if hungarian.empty:
            continue
        base = hungarian.iloc[0]
        for row in group.to_dict("records"):
            enriched = dict(row)
            enriched["assignment_quality_ratio_vs_hungarian"] = safe_divide(
                row.get("assignment_weight_total_mean"), base.get("assignment_weight_total_mean")
            )
            ratio = enriched["assignment_quality_ratio_vs_hungarian"]
            enriched["relative_optimality_gap_vs_hungarian"] = None if ratio is None else 1.0 - ratio
            enriched["solver_runtime_speedup_vs_hungarian"] = safe_divide(
                base.get("solver_runtime_seconds_mean"), row.get("solver_runtime_seconds_mean")
            )
            rows.append(enriched)
    return pd.DataFrame(rows)


def sparse_method_quality(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["lambda_driver_score", "matching_algorithm"]
    for keys, group in summary.groupby(key_cols, dropna=False):
        lambda_score, algorithm = keys
        bfs = group[group["sparse_method"] == "bfs_h3"]
        if bfs.empty:
            continue
        base = bfs.iloc[0]
        for row in group.to_dict("records"):
            enriched = dict(row)
            enriched["candidate_count_ratio_vs_bfs"] = safe_divide(
                row.get("candidate_pair_count_after_filter_total_mean"),
                base.get("candidate_pair_count_after_filter_total_mean"),
            )
            enriched["candidate_generation_runtime_ratio_vs_bfs"] = safe_divide(
                row.get("candidate_generation_runtime_seconds_mean"),
                base.get("candidate_generation_runtime_seconds_mean"),
            )
            enriched["assignment_weight_ratio_vs_bfs"] = safe_divide(
                row.get("assignment_weight_total_mean"), base.get("assignment_weight_total_mean")
            )
            enriched["expected_utility_ratio_vs_bfs"] = safe_divide(
                row.get("expected_economic_utility_total_mean"),
                base.get("expected_economic_utility_total_mean"),
            )
            enriched["match_rate_delta_vs_bfs"] = (
                row.get("match_rate_mean") - base.get("match_rate_mean")
                if pd.notna(row.get("match_rate_mean")) and pd.notna(base.get("match_rate_mean"))
                else None
            )
            rows.append(enriched)
    return pd.DataFrame(rows)
