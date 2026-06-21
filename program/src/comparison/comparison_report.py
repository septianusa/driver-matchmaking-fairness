from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _plotly_available():
    try:
        import plotly.express as px  # type: ignore
        import plotly.graph_objects as go  # type: ignore  # noqa: F401

        return px
    except Exception:
        return None


def write_comparison_report(
    output_dir: str | Path,
    *,
    dataset_profile: dict,
    summary: pd.DataFrame,
    raw_runs: pd.DataFrame,
    batch_metrics: pd.DataFrame,
) -> Path:
    output = Path(output_dir)
    report_path = output / "comparison_report.html"
    px = _plotly_available()
    charts: list[str] = []
    if px is not None and not summary.empty:
        frame = summary.copy()
        frame["variant"] = (
            "lambda="
            + frame["lambda_driver_score"].astype(str)
            + " | "
            + frame["matching_algorithm"].astype(str)
            + " | "
            + frame["sparse_method"].astype(str)
        )
        chart_specs = [
            ("end_to_end_runtime_seconds_mean", "End-to-End Runtime by Model Variant", "bar", "variant"),
            ("solver_runtime_seconds_mean", "Solver Runtime by Matching Algorithm", "box", "matching_algorithm"),
            ("candidate_generation_runtime_seconds_mean", "Candidate-Generation Runtime by Sparse Method", "box", "sparse_method"),
            ("peak_memory_mb_mean", "Peak Memory by Model Variant", "bar", "variant"),
            ("solver_throughput_pairs_per_second_mean", "Throughput by Matching Algorithm", "box", "matching_algorithm"),
            ("end_to_end_throughput_pairs_per_second_mean", "Throughput by Sparse Method", "box", "sparse_method"),
            ("end_to_end_runtime_seconds_mean", "Runtime by Driver-Score Weight", "line", "lambda_driver_score"),
            ("match_rate_mean", "Match Rate by Model Variant", "bar", "variant"),
            ("expected_economic_utility_total_mean", "Expected Economic Utility by Model Variant", "bar", "variant"),
            ("median_pickup_distance_km_mean", "Median Pickup Distance by Model Variant", "bar", "variant"),
            ("spearman_driver_score_income_mean", "Score-Income Spearman Correlation by Model Variant", "bar", "variant"),
            ("income_gini_mean", "Income Gini Coefficient by Model Variant", "bar", "variant"),
            ("candidate_reduction_pct_mean", "Candidate Reduction Percentage by Sparse Method", "box", "sparse_method"),
        ]
        for metric, title, kind, x_col in chart_specs:
            if metric not in frame:
                continue
            if kind == "bar":
                fig = px.bar(frame, x=x_col, y=metric, color="matching_algorithm", title=title)
            elif kind == "box":
                fig = px.box(frame, x=x_col, y=metric, color="sparse_method", title=title)
            else:
                grouped = frame.groupby(x_col, as_index=False)[metric].mean()
                fig = px.line(grouped, x=x_col, y=metric, title=title, markers=True)
            charts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        if {"end_to_end_runtime_seconds_mean", "assignment_weight_total_mean"}.issubset(frame.columns):
            fig = px.scatter(
                frame,
                x="end_to_end_runtime_seconds_mean",
                y="assignment_weight_total_mean",
                color="matching_algorithm",
                symbol="sparse_method",
                hover_name="variant",
                title="Assignment Quality Ratio versus Runtime",
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        if {"solver_runtime_speedup_vs_hungarian", "relative_optimality_gap_vs_hungarian"}.issubset(frame.columns):
            fig = px.scatter(
                frame,
                x="solver_runtime_speedup_vs_hungarian",
                y="relative_optimality_gap_vs_hungarian",
                color="matching_algorithm",
                title="Relative Optimality Gap versus Solver Speedup",
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    if px is not None and not raw_runs.empty:
        components = [
            "candidate_generation_runtime_seconds",
            "haversine_filter_runtime_seconds",
            "edge_feature_runtime_seconds",
            "solver_runtime_seconds",
            "state_update_runtime_seconds",
        ]
        available = [col for col in components if col in raw_runs.columns]
        if available:
            melted = raw_runs[raw_runs["status"] == "success"].melt(
                id_vars=["matching_algorithm", "sparse_method"],
                value_vars=available,
                var_name="component",
                value_name="runtime_seconds",
            )
            fig = px.bar(
                melted,
                x="component",
                y="runtime_seconds",
                color="matching_algorithm",
                barmode="group",
                title="Runtime Breakdown by Component",
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    if px is not None and not batch_metrics.empty:
        if "runtime_seconds" in batch_metrics.columns:
            fig = px.line(
                batch_metrics,
                x="batch_id",
                y="runtime_seconds",
                color="variant_id",
                title="Batch-Level Runtime Trend",
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        if {"raw_cartesian_pair_count", "candidate_pair_count_after_filter"}.issubset(batch_metrics.columns):
            sample = batch_metrics.head(5000).copy()
            fig = px.line(
                sample,
                x="batch_id",
                y=["raw_cartesian_pair_count", "candidate_pair_count_after_filter"],
                title="Raw Cartesian Pairs versus Feasible Pairs by Batch",
            )
            charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    summary_html = summary.head(100).to_html(index=False) if not summary.empty else "<p>No summary rows.</p>"
    profile_html = "<pre>" + html.escape(json.dumps(dataset_profile, indent=2)) + "</pre>"
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.4; }}
    table {{ border-collapse: collapse; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 7px; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Model Comparison Report</h1>
  <h2>Dataset Profile</h2>
  {profile_html}
  <h2>Summary</h2>
  {summary_html}
  <h2>Charts</h2>
  {''.join(charts) if charts else '<p>Plotly is unavailable or no successful rows were produced.</p>'}
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path

