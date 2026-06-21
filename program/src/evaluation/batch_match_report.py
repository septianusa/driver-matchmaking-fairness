from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


COLORS = [
    "#2f5f8f",
    "#b04a3a",
    "#3b7f58",
    "#8b5a9f",
    "#b7791f",
    "#18747d",
    "#6f6f6f",
    "#c2416b",
    "#5c6f2f",
    "#7a4b2a",
    "#3f51b5",
    "#00897b",
    "#ad1457",
    "#827717",
    "#5d4037",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_comparison_dir(project_root: Path, data_mode: str) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for profile_path in (project_root / "outputs").glob("*/comparison/dataset_profile.json"):
        try:
            profile = _read_json(profile_path)
        except Exception:
            continue
        comparison_dir = profile_path.parent
        batch_path = comparison_dir / "comparison_batch_metrics.csv"
        if profile.get("data_mode") == data_mode and batch_path.exists():
            candidates.append((batch_path.stat().st_mtime, comparison_dir))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _prepare_match_by_batch(comparison_dir: Path) -> pd.DataFrame:
    batch = pd.read_csv(comparison_dir / "comparison_batch_metrics.csv")
    if "warmup_flag" in batch.columns:
        measured = batch[batch["warmup_flag"] == False].copy()  # noqa: E712
        if not measured.empty:
            batch = measured
    if "variant_id" not in batch.columns:
        batch["variant_id"] = "scenario"
    grouped = (
        batch.groupby(["variant_id", "batch_id"], as_index=False)
        .agg(matches_created=("matches_created", "mean"))
        .sort_values(["variant_id", "batch_id"])
    )
    grouped["cumulative_matches"] = grouped.groupby("variant_id")["matches_created"].cumsum()
    return grouped


def _prepare_all_days_orders(project_root: Path) -> pd.DataFrame:
    orders_path = project_root / "data" / "raw" / "orders.csv"
    if not orders_path.exists():
        return pd.DataFrame(columns=["jakarta_data_date", "batch_id", "orders_created", "cumulative_orders"])
    orders = pd.read_csv(orders_path)
    date_col = "jakarta_data_date"
    batch_col = "batch_step"
    if date_col not in orders.columns or batch_col not in orders.columns:
        return pd.DataFrame(columns=["jakarta_data_date", "batch_id", "orders_created", "cumulative_orders"])
    orders[date_col] = pd.to_datetime(orders[date_col]).dt.date.astype(str)
    orders[batch_col] = pd.to_numeric(orders[batch_col], errors="coerce").fillna(0).astype(int)
    grouped = (
        orders.groupby([date_col, batch_col], as_index=False)
        .size()
        .rename(columns={batch_col: "batch_id", "size": "orders_created"})
        .sort_values([date_col, "batch_id"])
    )
    dates = sorted(grouped[date_col].unique())
    full_index = pd.MultiIndex.from_product([dates, range(1, 1441)], names=[date_col, "batch_id"])
    full = (
        grouped.set_index([date_col, "batch_id"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    full["cumulative_orders"] = full.groupby(date_col)["orders_created"].cumsum()
    return full


def _short_label(label: str) -> str:
    label = label.replace("lambda=", "l=")
    label = label.replace("__solver=", ", ")
    label = label.replace("__sparse=", ", ")
    label = label.replace("hungarian", "hung")
    return label


def _svg_line_chart(
    frame: pd.DataFrame,
    *,
    group_col: str,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 1120,
    height: int = 780,
) -> str:
    if frame.empty:
        return "<p>No chart data available.</p>"
    data = frame[[group_col, x_col, y_col]].dropna().copy()
    if data.empty:
        return "<p>No chart data available.</p>"
    margin_left = 82
    margin_right = 34
    margin_top = 46
    margin_bottom = 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_max = max(float(data[x_col].max()), 1.0)
    y_min = min(float(data[y_col].min()), 1.0)
    y_max = max(float(data[y_col].max()), y_min + 1.0)

    def sx(value: float) -> float:
        return margin_left + (float(value) / x_max) * plot_width

    def sy(value: float) -> float:
        return margin_top + ((float(value) - y_min) / (y_max - y_min)) * plot_height

    groups = list(dict.fromkeys(data[group_col].astype(str).tolist()))
    polylines = []
    legends = []
    for idx, group in enumerate(groups):
        group_df = data[data[group_col].astype(str) == group].sort_values(y_col)
        points = " ".join(f"{sx(row[x_col]):.2f},{sy(row[y_col]):.2f}" for row in group_df.to_dict("records"))
        color = COLORS[idx % len(COLORS)]
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-opacity="0.82" />'
        )
        legends.append(
            f"<tr><td><span style='background:{color}'></span></td>"
            f"<td>{html.escape(_short_label(group))}</td></tr>"
        )

    x_ticks = []
    for tick in range(0, 6):
        value = x_max * tick / 5
        x = sx(value)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" '
            f'stroke="#e5e5e5" />'
            f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle">{value:,.0f}</text>'
        )
    y_ticks = []
    for value in [1, 240, 480, 720, 960, 1200, 1440]:
        if value < y_min or value > y_max:
            continue
        y = sy(value)
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            f'stroke="#eeeeee" />'
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{value}</text>'
        )

    svg = f"""
<figure class="chart-block">
  <figcaption>{html.escape(title)}</figcaption>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#fff" />
    {''.join(x_ticks)}
    {''.join(y_ticks)}
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222" />
    <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#222" />
    {''.join(polylines)}
    <text x="{margin_left + plot_width / 2:.2f}" y="{height - 16}" text-anchor="middle">{html.escape(x_label)}</text>
    <text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22,{margin_top + plot_height / 2:.2f})" text-anchor="middle">{html.escape(y_label)}</text>
  </svg>
</figure>
<table class="legend-table"><tbody>{''.join(legends)}</tbody></table>
"""
    return svg


def _summary_table(match_by_batch: pd.DataFrame) -> str:
    if match_by_batch.empty:
        return "<p>No scenario summary available.</p>"
    summary = (
        match_by_batch.groupby("variant_id", as_index=False)
        .agg(total_matches=("matches_created", "sum"), max_batch_total=("cumulative_matches", "max"))
        .sort_values("total_matches", ascending=False)
    )
    summary["scenario"] = summary["variant_id"].map(_short_label)
    return summary[["scenario", "total_matches"]].to_html(index=False, classes="paper-table", border=0)


def _write_html(
    output_dir: Path,
    *,
    data_mode: str,
    comparison_dir: Path | None,
    profile: dict[str, Any] | None,
    match_by_batch: pd.DataFrame,
    orders_by_batch: pd.DataFrame,
) -> None:
    match_chart = _svg_line_chart(
        match_by_batch,
        group_col="variant_id",
        x_col="cumulative_matches",
        y_col="batch_id",
        title="Scenario Comparison: Cumulative Total Matches by Batch",
        x_label="Total matches",
        y_label="Batch",
    )
    demand_chart = _svg_line_chart(
        orders_by_batch,
        group_col="jakarta_data_date",
        x_col="cumulative_orders",
        y_col="batch_id",
        title="All Days Input Demand: Cumulative Orders by Batch",
        x_label="Total orders",
        y_label="Batch",
    )
    simulation_date = profile.get("simulation_date") if profile else "n/a"
    note = (
        "The match chart uses the latest completed comparison batch metrics. "
        "The all-days chart uses raw order demand because completed match outputs were not found for every date."
    )
    if orders_by_batch.empty:
        note = "The match chart uses the latest completed comparison batch metrics."

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Batch Match Scenario Comparison</title>
  <style>
    body {{
      margin: 0;
      background: #f1f1ef;
      color: #1f1f1f;
      font-family: "Times New Roman", Georgia, serif;
      line-height: 1.5;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 28px auto;
      background: #fff;
      padding: 42px 52px;
      box-shadow: 0 2px 18px rgba(0,0,0,0.08);
    }}
    h1 {{ text-align: center; font-size: 30px; margin: 0; }}
    .subtitle {{ text-align: center; color: #555; margin: 8px 0 28px; }}
    h2 {{ border-bottom: 1px solid #222; padding-bottom: 4px; margin-top: 32px; }}
    code {{ font-family: Consolas, monospace; }}
    .chart-block {{ border: 1px solid #d7d7d7; padding: 12px; margin: 18px 0; overflow-x: auto; }}
    figcaption {{ font-weight: bold; margin-bottom: 8px; }}
    svg text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #222; }}
    .legend-table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0 22px; }}
    .legend-table td {{ border: 1px solid #e1e1e1; padding: 4px 7px; }}
    .legend-table span {{ display: inline-block; width: 24px; height: 4px; vertical-align: middle; }}
    .paper-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .paper-table th, .paper-table td {{ border: 1px solid #d8d8d8; padding: 6px 8px; }}
    .paper-table th {{ background: #efefef; }}
  </style>
</head>
<body>
<main>
  <h1>Batch Match Scenario Comparison</h1>
  <p class="subtitle">x-axis = total matches, y-axis = batch</p>
  <section>
    <h2>Source</h2>
    <p>Data mode: <code>{html.escape(data_mode)}</code></p>
    <p>Simulation date for match outputs: <code>{html.escape(str(simulation_date))}</code></p>
    <p>Comparison directory: <code>{html.escape(str(comparison_dir or 'not found'))}</code></p>
    <p>{html.escape(note)}</p>
  </section>
  <section>
    <h2>Scenario Match Graph</h2>
    {match_chart}
  </section>
  <section>
    <h2>Scenario Totals</h2>
    {_summary_table(match_by_batch)}
  </section>
  <section>
    <h2>All Days Context</h2>
    {demand_chart}
  </section>
  <section>
    <h2>Generated Files</h2>
    <p><code>scenario_total_matches_by_batch.csv</code></p>
    <p><code>all_days_orders_by_batch.csv</code></p>
    <p>Generated at: {datetime.now().isoformat(timespec="seconds")}</p>
  </section>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_batch_match_report(project_root: str | Path = ".", data_mode: str = "actual") -> Path:
    root = Path(project_root).resolve()
    output_dir = root / "outputs" / "batch_match_comparison_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = _latest_comparison_dir(root, data_mode)
    profile: dict[str, Any] | None = None
    if comparison_dir is not None:
        profile = _read_json(comparison_dir / "dataset_profile.json")
        match_by_batch = _prepare_match_by_batch(comparison_dir)
    else:
        match_by_batch = pd.DataFrame(
            columns=["variant_id", "batch_id", "matches_created", "cumulative_matches"]
        )
    orders_by_batch = _prepare_all_days_orders(root)
    match_by_batch.to_csv(output_dir / "scenario_total_matches_by_batch.csv", index=False)
    orders_by_batch.to_csv(output_dir / "all_days_orders_by_batch.csv", index=False)
    _write_html(
        output_dir,
        data_mode=data_mode,
        comparison_dir=comparison_dir,
        profile=profile,
        match_by_batch=match_by_batch,
        orders_by_batch=orders_by_batch,
    )
    return output_dir
