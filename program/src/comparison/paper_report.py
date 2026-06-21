from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _latest_comparison_dir(project_root: Path, data_mode: str) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for profile_path in (project_root / "outputs").glob("*/comparison/dataset_profile.json"):
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        comparison_dir = profile_path.parent
        summary_path = comparison_dir / "comparison_summary.csv"
        if profile.get("data_mode") == data_mode and summary_path.exists():
            candidates.append((summary_path.stat().st_mtime, comparison_dir))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}"
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return "<p>No table data available.</p>"
    return df[existing].head(max_rows).to_html(index=False, classes="paper-table", border=0)


def _metric_cards(profile: dict, summary: pd.DataFrame, raw_runs: pd.DataFrame) -> str:
    successful = int((raw_runs.get("status", pd.Series(dtype=str)) == "success").sum()) if not raw_runs.empty else 0
    skipped = int((raw_runs.get("status", pd.Series(dtype=str)) == "skipped").sum()) if not raw_runs.empty else 0
    failed = int((raw_runs.get("status", pd.Series(dtype=str)) == "failed").sum()) if not raw_runs.empty else 0
    best_runtime = summary["end_to_end_runtime_mean"].min() if "end_to_end_runtime_mean" in summary else None
    best_match = summary["match_rate_mean"].max() if "match_rate_mean" in summary else None
    best_gini = summary["income_gini_mean"].min() if "income_gini_mean" in summary else None
    cards = [
        ("Orders", _fmt(profile.get("number_of_orders"), 0)),
        ("Drivers", _fmt(profile.get("number_of_unique_drivers"), 0)),
        ("Raw Pairs", _fmt(profile.get("raw_cartesian_pair_count_total"), 0)),
        ("Successful Runs", _fmt(successful, 0)),
        ("Skipped Runs", _fmt(skipped, 0)),
        ("Failed Runs", _fmt(failed, 0)),
        ("Best Runtime", f"{_fmt(best_runtime)} s"),
        ("Best Match Rate", _fmt(best_match)),
        ("Lowest Gini", _fmt(best_gini)),
    ]
    return "\n".join(
        f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>" for label, value in cards
    )


def _simple_bar_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str) -> str:
    if df.empty or label_col not in df or value_col not in df:
        return ""
    chart = df[[label_col, value_col]].dropna().head(12).copy()
    if chart.empty:
        return ""
    max_value = float(chart[value_col].max()) or 1.0
    rows = []
    for row in chart.to_dict("records"):
        width = max(2.0, float(row[value_col]) / max_value * 100.0)
        rows.append(
            f"<div class='bar-row'><span class='bar-label'>{row[label_col]}</span>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
            f"<span class='bar-value'>{_fmt(row[value_col])}</span></div>"
        )
    return f"<figure><figcaption>{title}</figcaption>{''.join(rows)}</figure>"


def _paper_html_for_run(comparison_dir: Path, data_mode: str) -> str:
    profile = json.loads((comparison_dir / "dataset_profile.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(comparison_dir / "comparison_summary.csv")
    raw_runs = pd.read_csv(comparison_dir / "comparison_raw_runs.csv") if (comparison_dir / "comparison_raw_runs.csv").exists() else pd.DataFrame()
    baseline = pd.read_csv(comparison_dir / "comparison_baseline_relative.csv") if (comparison_dir / "comparison_baseline_relative.csv").exists() else pd.DataFrame()
    algorithm = pd.read_csv(comparison_dir / "comparison_algorithm_quality.csv") if (comparison_dir / "comparison_algorithm_quality.csv").exists() else pd.DataFrame()

    summary = summary.copy()
    summary["variant"] = (
        "lambda="
        + summary["lambda_driver_score"].astype(str)
        + ", "
        + summary["matching_algorithm"].astype(str)
        + ", "
        + summary["sparse_method"].astype(str)
    )
    successful_summary = summary[summary["successful_run_count"] > 0].copy()
    fastest = successful_summary.sort_values("end_to_end_runtime_mean").head(1)
    highest_utility = successful_summary.sort_values("expected_economic_utility_total_mean", ascending=False).head(1)
    fairest = successful_summary.sort_values("income_gini_mean").head(1)

    abstract = (
        f"This report summarizes a model-comparison experiment using {data_mode} data for "
        f"{profile.get('simulation_date')}. The experiment evaluates driver-score weighting, "
        "matching algorithms, and sparse candidate retrieval while holding the dataset snapshot "
        "and simulation configuration fixed across variants."
    )
    if not fastest.empty:
        row = fastest.iloc[0]
        abstract += (
            f" The fastest successful variant was {row['variant']} with mean end-to-end runtime "
            f"{_fmt(row.get('end_to_end_runtime_mean'))} seconds."
        )
    if not highest_utility.empty:
        row = highest_utility.iloc[0]
        abstract += (
            f" The highest expected utility variant was {row['variant']} with mean utility "
            f"{_fmt(row.get('expected_economic_utility_total_mean'))}."
        )

    runtime_chart = _simple_bar_chart(
        successful_summary.sort_values("end_to_end_runtime_mean"),
        "variant",
        "end_to_end_runtime_mean",
        "Figure 1. Mean end-to-end runtime by variant",
    )
    match_chart = _simple_bar_chart(
        successful_summary.sort_values("match_rate_mean", ascending=False),
        "variant",
        "match_rate_mean",
        "Figure 2. Mean match rate by variant",
    )
    gini_chart = _simple_bar_chart(
        successful_summary.sort_values("income_gini_mean"),
        "variant",
        "income_gini_mean",
        "Figure 3. Income Gini coefficient by variant",
    )

    return _wrap_paper(
        title=f"Model Comparison Performance Report ({data_mode.title()} Data)",
        body=f"""
<section>
  <h2>Abstract</h2>
  <p>{abstract}</p>
</section>
<section>
  <h2>Dataset and Experimental Design</h2>
  <div class="metrics">{_metric_cards(profile, summary, raw_runs)}</div>
  <p>The experiment uses one dataset snapshot and reuses it for all variants. Raw Cartesian pairs are computed as active orders multiplied by available drivers per batch, summed over the selected horizon.</p>
</section>
<section>
  <h2>Methods</h2>
  <p>The controlled factors are driver-score weight, matching algorithm, and sparse candidate method. All variants reuse the same simulation date, batch range, H3 configuration, ETA assumptions, random seed, order carry-over, grid-value initialization, cancellation model, and positioning mode.</p>
  <p>A2GAT variants are retained in the design matrix but are skipped unless a trained A2GAT integration is configured. This avoids fabricated candidate-generation results.</p>
</section>
<section>
  <h2>Primary Results</h2>
  {runtime_chart}
  {match_chart}
  {gini_chart}
  <h3>Top Runtime Variants</h3>
  {_table(successful_summary.sort_values("end_to_end_runtime_mean"), ["variant", "successful_run_count", "end_to_end_runtime_mean", "solver_runtime_mean", "candidate_generation_runtime_mean", "match_rate_mean", "income_gini_mean"])}
  <h3>Top Utility Variants</h3>
  {_table(successful_summary.sort_values("expected_economic_utility_total_mean", ascending=False), ["variant", "expected_economic_utility_total_mean", "assignment_weight_total_mean", "match_rate_mean", "median_pickup_distance_km_mean", "spearman_driver_score_income_mean", "income_gini_mean"])}
  <h3>Fairest Variants by Income Gini</h3>
  {_table(successful_summary.sort_values("income_gini_mean"), ["variant", "income_gini_mean", "spearman_driver_score_income_mean", "match_rate_mean", "expected_economic_utility_total_mean", "end_to_end_runtime_mean"])}
</section>
<section>
  <h2>Baseline-Relative Performance</h2>
  {_table(baseline, ["lambda_driver_score", "matching_algorithm", "sparse_method", "runtime_ratio_vs_baseline", "runtime_speedup_vs_baseline", "memory_ratio_vs_baseline", "assignment_quality_ratio_vs_baseline", "expected_utility_ratio_vs_baseline", "match_rate_delta_vs_baseline"], 30)}
</section>
<section>
  <h2>Matching Algorithm Quality</h2>
  {_table(algorithm, ["lambda_driver_score", "matching_algorithm", "sparse_method", "assignment_quality_ratio_vs_hungarian", "relative_optimality_gap_vs_hungarian", "solver_runtime_speedup_vs_hungarian"], 30)}
</section>
<section>
  <h2>Discussion</h2>
  <p>The comparison separates computational performance from dispatch quality. Runtime and throughput indicate operational feasibility, while match rate, pickup distance, expected utility, score-income correlation, and income Gini indicate dispatch quality and fairness effects.</p>
  <p>Warm-up runs are stored in raw outputs but excluded from summary statistics. Skipped variants are documented rather than replaced by fallback methods.</p>
</section>
<section>
  <h2>Reproducibility</h2>
  <p>Source comparison directory: <code>{comparison_dir}</code></p>
  <p>Generated at: {datetime.now().isoformat(timespec="seconds")}</p>
</section>
""",
    )


def _missing_run_html(data_mode: str) -> str:
    dry_run_command = (
        "python main.py compare-models --config configs/model_comparison.yaml "
        f"--data-mode {data_mode} --dry-run"
    )
    run_command = (
        "python main.py compare-models --config configs/model_comparison.yaml "
        f"--data-mode {data_mode}"
    )
    report_command = "python main.py paper-reports"
    return _wrap_paper(
        title=f"Model Comparison Performance Report ({data_mode.title()} Data)",
        body=f"""
<section>
  <h2>Abstract</h2>
  <p>No completed {data_mode} model-comparison output was found in <code>outputs/*/comparison</code>. This page is a placeholder so the paper-report folder has a stable structure.</p>
</section>
<section>
  <h2>Required Run</h2>
  <p>Generate the {data_mode} comparison first, then rebuild this report folder.</p>
  <pre>{dry_run_command}
{run_command}
{report_command}</pre>
</section>
<section>
  <h2>Interpretation Note</h2>
  <p>No performance conclusions are reported here because no completed {data_mode} comparison summary is available. The report intentionally does not fabricate values.</p>
</section>
""",
    )


def _wrap_paper(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      background: #f1f1ef;
      color: #1d1d1f;
      font-family: "Times New Roman", Georgia, serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1040px, calc(100% - 32px));
      margin: 28px auto;
      background: #fff;
      padding: 44px 54px;
      box-shadow: 0 2px 18px rgba(0,0,0,0.08);
    }}
    h1 {{ font-size: 30px; text-align: center; margin: 0 0 8px; }}
    .subtitle {{ text-align: center; color: #555; margin-bottom: 32px; }}
    h2 {{ border-bottom: 1px solid #111; padding-bottom: 4px; margin-top: 30px; }}
    h3 {{ margin-top: 22px; }}
    code, pre {{ font-family: Consolas, monospace; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 16px 0; }}
    .metric {{ border: 1px solid #cfcfcf; padding: 10px 12px; }}
    .metric span {{ display: block; color: #555; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 20px; margin-top: 4px; }}
    .paper-table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 20px; }}
    .paper-table th, .paper-table td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: right; }}
    .paper-table th:first-child, .paper-table td:first-child {{ text-align: left; }}
    .paper-table th {{ background: #efefef; }}
    figure {{ margin: 20px 0; border: 1px solid #d7d7d7; padding: 12px; }}
    figcaption {{ font-weight: bold; margin-bottom: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 320px 1fr 80px; gap: 10px; align-items: center; margin: 6px 0; }}
    .bar-label {{ font-size: 12px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
    .bar-track {{ background: #e8e8e8; height: 14px; }}
    .bar-fill {{ background: #355f8c; height: 14px; }}
    .bar-value {{ font-size: 12px; text-align: right; }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="subtitle">Driver Behavior-Based Dispatching to Improve Income Fairness in Ride-Hailing Platforms</p>
  {body}
</main>
</body>
</html>
"""


def build_paper_report_folder(project_root: str | Path = ".") -> Path:
    root = Path(project_root).resolve()
    output = root / "outputs" / "paper_style_model_reports"
    output.mkdir(parents=True, exist_ok=True)
    pages = {}
    for data_mode in ["actual"]:
        comparison_dir = _latest_comparison_dir(root, data_mode)
        filename = f"{data_mode}_model_comparison_paper.html"
        if comparison_dir is None:
            html = _missing_run_html(data_mode)
        else:
            html = _paper_html_for_run(comparison_dir, data_mode)
        (output / filename).write_text(html, encoding="utf-8")
        pages[data_mode] = filename
    index = _wrap_paper(
        title="Model Comparison Paper-Style Reports",
        body=f"""
<section>
  <h2>Reports</h2>
  <ul>
    <li><a href="{pages['actual']}">Actual data performance report</a></li>
  </ul>
  <p>This folder summarizes the latest completed actual-data model-comparison run.</p>
</section>
""",
    )
    (output / "index.html").write_text(index, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build_paper_report_folder())
