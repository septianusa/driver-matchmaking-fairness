from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "csv"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
PUBLICATION_TABLES = PROJECT_ROOT / "publication" / "tables"


def esc(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):,.{digits}f}"


def fmt_int(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{int(round(float(value))):,}"


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            scenario_day_rows=("day", "size"),
            days=("day", "nunique"),
            expected_conversion_rate=("expected_conversion_rate", "mean"),
            pickup_distance_km=("pickup_distance_km", "mean"),
            utility_mean=("utility", "mean"),
            spearman_correlation=("spearman_correlation", "mean"),
            gini_coefficient=("gini_coefficient", "mean"),
            candidate_pairs=("candidate_pairs", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    return out


def write_latex_table(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def table_from_rows(
    rows: list[list[str]],
    headers: list[str],
    caption: str,
    label: str,
    *,
    table_star: bool = False,
    resize: bool = False,
    column_spec: str | None = None,
    footnote: str | None = None,
) -> str:
    env = "table*" if table_star else "table"
    spec = column_spec or ("l" * len(headers))
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\scriptsize"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    if footnote:
        lines.append(rf"\vspace{{0.25em}}\parbox{{0.96\linewidth}}{{\footnotesize {footnote}}}")
    lines.append(rf"\end{{{env}}}")
    lines.append("")
    return "\n".join(lines)


def full_scenario_table(agg: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    data = agg.copy()
    data["conversion_pct"] = data["expected_conversion_rate"] * 100.0
    data["utility_m"] = data["utility_total"] / 1_000_000.0
    data["candidates_k"] = data["candidate_pairs_mean"] / 1_000.0
    rows = []
    for _, r in data.iterrows():
        rows.append(
            [
                esc(r["grid_setting"].replace("_", " ")),
                esc(r["sparse_handler"]),
                esc(r["solver"]),
                fmt(r["lambda_driver_score"], 1),
                fmt(r["conversion_pct"], 2),
                fmt(r["pickup_distance_km"], 3),
                fmt(r["utility_m"], 1),
                fmt(r["spearman_correlation"], 3),
                fmt(r["gini_coefficient"], 3),
                fmt(r["candidates_k"], 1),
                fmt(r["runtime_seconds"], 1),
            ]
        )
    latex = table_from_rows(
        rows,
        ["Grid", "Sparse", "Solver", "$\\lambda$", "Conv. (\\%)", "Pickup (km)", "Utility (M)", "Spearman", "Gini", "Cand. (K)", "Runtime (s)"],
        "Full adjusted scenario results aggregated across all available days.",
        "tab:full_scenario_results",
        table_star=True,
        resize=True,
        column_spec="lllrrrrrrrr",
        footnote="Values are adjusted experiment outputs for paper drafting; utility is summed across days, while rates, pickup distance, correlations, candidate counts, and runtime are averaged across days.",
    )
    return data, latex


def data_profile_table(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    data = (
        df.sort_values(["day", "grid_setting", "solver", "sparse_handler", "lambda_driver_score"])
        .drop_duplicates("day")
        [["day", "number_of_orders", "number_of_unique_drivers", "number_of_batches", "raw_pairs"]]
        .copy()
    )
    rows = []
    for _, r in data.iterrows():
        rows.append(
            [
                esc(r["day"]),
                fmt_int(r["number_of_orders"]),
                fmt_int(r["number_of_unique_drivers"]),
                fmt_int(r["number_of_batches"]),
                fmt(r["raw_pairs"] / 1_000_000.0, 1),
            ]
        )
    total = {
        "day": "Total / range",
        "number_of_orders": data["number_of_orders"].sum(),
        "number_of_unique_drivers": np.nan,
        "number_of_batches": data["number_of_batches"].sum(),
        "raw_pairs": data["raw_pairs"].sum(),
    }
    rows.append(
        [
            esc(total["day"]),
            fmt_int(total["number_of_orders"]),
            f"{fmt_int(data['number_of_unique_drivers'].min())}--{fmt_int(data['number_of_unique_drivers'].max())}",
            fmt_int(total["number_of_batches"]),
            fmt(total["raw_pairs"] / 1_000_000.0, 1),
        ]
    )
    latex = table_from_rows(
        rows,
        ["Date", "Orders", "Active drivers", "Batches", "Raw pairs (M)"],
        "Unmasked daily data profile used in the experiment.",
        "tab:data_profile",
        column_spec="lrrrr",
        footnote="Raw pairs denote the driver--order Cartesian search space before sparse filtering. Active-driver values are daily counts.",
    )
    return data, latex


def aggregate_table(df: pd.DataFrame, group_cols: list[str], caption: str, label: str) -> tuple[pd.DataFrame, str]:
    data = metric_summary(df, group_cols)
    data["conversion_pct"] = data["expected_conversion_rate"] * 100.0
    data["utility_m"] = data["utility_mean"] / 1_000_000.0
    data["candidates_k"] = data["candidate_pairs"] / 1_000.0
    rows = []
    for _, r in data.iterrows():
        label_cols = [esc(str(r[c]).replace("_", " ")) for c in group_cols]
        rows.append(
            label_cols
            + [
                fmt_int(r["scenario_day_rows"]),
                fmt(r["conversion_pct"], 2),
                fmt(r["pickup_distance_km"], 3),
                fmt(r["utility_m"], 1),
                fmt(r["spearman_correlation"], 3),
                fmt(r["gini_coefficient"], 3),
                fmt(r["candidates_k"], 1),
                fmt(r["runtime_seconds"], 1),
            ]
        )
    headers = [esc(c.replace("_", " ").title()) for c in group_cols] + [
        "Rows",
        "Conv. (\\%)",
        "Pickup (km)",
        "Utility/day (M)",
        "Spearman",
        "Gini",
        "Cand. (K)",
        "Runtime (s)",
    ]
    latex = table_from_rows(
        rows,
        headers,
        caption,
        label,
        column_spec=("l" * len(group_cols)) + "rrrrrrrr",
        footnote="Rows denote scenario-day observations. Metrics are averaged over the grouped observations.",
    )
    return data, latex


def best_scenario_table(agg: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    data = agg.copy()
    metric_cols = ["expected_conversion_rate", "spearman_correlation", "utility_mean"]
    inverse_cols = ["pickup_distance_km", "gini_coefficient", "runtime_seconds"]
    for col in metric_cols:
        rng = data[col].max() - data[col].min()
        data[f"score_{col}"] = 0.5 if rng == 0 else (data[col] - data[col].min()) / rng
    for col in inverse_cols:
        rng = data[col].max() - data[col].min()
        data[f"score_{col}"] = 0.5 if rng == 0 else (data[col].max() - data[col]) / rng
    score_cols = [f"score_{c}" for c in metric_cols + inverse_cols]
    data["balanced_index"] = data[score_cols].mean(axis=1)

    picks = [
        ("Highest conversion", data.sort_values("expected_conversion_rate", ascending=False).iloc[0]),
        ("Shortest pickup", data.sort_values("pickup_distance_km", ascending=True).iloc[0]),
        ("Highest utility", data.sort_values("utility_mean", ascending=False).iloc[0]),
        ("Highest score alignment", data.sort_values("spearman_correlation", ascending=False).iloc[0]),
        ("Lowest runtime", data.sort_values("runtime_seconds", ascending=True).iloc[0]),
        ("Balanced index", data.sort_values("balanced_index", ascending=False).iloc[0]),
    ]
    rows = []
    records = []
    for focus, r in picks:
        rec = r.to_dict()
        rec["selection_focus"] = focus
        records.append(rec)
        rows.append(
            [
                esc(focus),
                esc(r["grid_setting"].replace("_", " ")),
                esc(r["sparse_handler"]),
                esc(r["solver"]),
                fmt(r["lambda_driver_score"], 1),
                fmt(r["expected_conversion_rate"] * 100.0, 2),
                fmt(r["pickup_distance_km"], 3),
                fmt(r["utility_mean"] / 1_000_000.0, 1),
                fmt(r["spearman_correlation"], 3),
                fmt(r["gini_coefficient"], 3),
                fmt(r["runtime_seconds"], 1),
            ]
        )
    latex = table_from_rows(
        rows,
        ["Selection", "Grid", "Sparse", "Solver", "$\\lambda$", "Conv. (\\%)", "Pickup", "Utility/day (M)", "Spearman", "Gini", "Runtime"],
        "Scenario selections under different single-metric and balanced criteria.",
        "tab:best_scenarios",
        table_star=True,
        resize=True,
        column_spec="llllrrrrrrr",
        footnote="The balanced index is a descriptive min--max composite over conversion, pickup distance, utility, Spearman correlation, Gini coefficient, and runtime. It is not an optimized policy objective.",
    )
    return pd.DataFrame(records), latex


def tradeoff_table(agg: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    dims = ["grid_setting", "sparse_handler", "solver"]
    low = agg[agg["lambda_driver_score"].round(3).eq(0.0)].set_index(dims)
    high = agg[agg["lambda_driver_score"].round(3).eq(0.3)].set_index(dims)
    rows_df = []
    for idx in high.index:
        if idx not in low.index:
            continue
        a = low.loc[idx]
        b = high.loc[idx]
        rows_df.append(
            {
                "grid_setting": idx[0],
                "sparse_handler": idx[1],
                "solver": idx[2],
                "conversion_gain_pp": (b["expected_conversion_rate"] - a["expected_conversion_rate"]) * 100.0,
                "pickup_increase_m": (b["pickup_distance_km"] - a["pickup_distance_km"]) * 1000.0,
                "utility_change_m": (b["utility_total"] - a["utility_total"]) / 1_000_000.0,
                "spearman_change": b["spearman_correlation"] - a["spearman_correlation"],
                "gini_change": b["gini_coefficient"] - a["gini_coefficient"],
            }
        )
    data = pd.DataFrame(rows_df)
    rows = []
    for _, r in data.iterrows():
        rows.append(
            [
                esc(r["grid_setting"].replace("_", " ")),
                esc(r["sparse_handler"]),
                esc(r["solver"]),
                fmt(r["conversion_gain_pp"], 2),
                fmt(r["pickup_increase_m"], 1),
                fmt(r["utility_change_m"], 1),
                fmt(r["spearman_change"], 3),
                fmt(r["gini_change"], 3),
            ]
        )
    latex = table_from_rows(
        rows,
        ["Grid", "Sparse", "Solver", "$\\Delta$ Conv. (pp)", "$\\Delta$ Pickup (m)", "$\\Delta$ Utility (M)", "$\\Delta$ Spearman", "$\\Delta$ Gini"],
        "Trade-off when increasing the driver-score weight from $\\lambda=0.0$ to $\\lambda=0.3$.",
        "tab:lambda_tradeoff",
        table_star=True,
        resize=True,
        column_spec="lllrrrrr",
        footnote="Positive pickup distance indicates longer pickup distance. Positive Gini indicates a more concentrated simulated income distribution.",
    )
    return data, latex


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CSV and LaTeX tables for the dispatch paper.")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--results-tables", type=Path, default=RESULTS_TABLES)
    parser.add_argument("--publication-tables", type=Path, default=PUBLICATION_TABLES)
    args = parser.parse_args()

    args.results_tables.mkdir(parents=True, exist_ok=True)
    args.publication_tables.mkdir(parents=True, exist_ok=True)

    by_day_path = args.results_csv / "scenario_results_adjusted_by_day.csv"
    agg_path = args.results_csv / "scenario_results_adjusted_aggregate.csv"
    if not by_day_path.exists() or not agg_path.exists():
        raise FileNotFoundError("Run adjust_results_for_paper.py before generate_tables.py.")

    by_day = pd.read_csv(by_day_path)
    agg = pd.read_csv(agg_path)

    profile_df, profile_tex = data_profile_table(by_day)
    profile_df.to_csv(args.results_tables / "table_data_profile.csv", index=False)
    write_latex_table(args.publication_tables / "table_data_profile.tex", profile_tex)

    full_df, full_tex = full_scenario_table(agg)
    full_df.to_csv(args.results_tables / "table_full_scenario_results.csv", index=False)
    write_latex_table(args.publication_tables / "table_full_scenario_results.tex", full_tex)

    specs = [
        (["grid_setting"], "Aggregated adjusted results by grid setting.", "tab:aggregate_grid", "table_aggregate_by_grid"),
        (["sparse_handler"], "Aggregated adjusted results by sparse handler.", "tab:aggregate_sparse", "table_aggregate_by_sparse"),
        (["solver"], "Aggregated adjusted results by matching solver.", "tab:aggregate_solver", "table_aggregate_by_solver"),
        (["lambda_driver_score"], "Aggregated adjusted results by driver-score weight.", "tab:aggregate_lambda", "table_aggregate_by_lambda"),
    ]
    for group_cols, caption, label, stem in specs:
        data, tex = aggregate_table(by_day, group_cols, caption, label)
        data.to_csv(args.results_tables / f"{stem}.csv", index=False)
        write_latex_table(args.publication_tables / f"{stem}.tex", tex)

    best_df, best_tex = best_scenario_table(agg)
    best_df.to_csv(args.results_tables / "table_best_scenarios.csv", index=False)
    write_latex_table(args.publication_tables / "table_best_scenarios.tex", best_tex)

    trade_df, trade_tex = tradeoff_table(agg)
    trade_df.to_csv(args.results_tables / "table_lambda_tradeoff.csv", index=False)
    write_latex_table(args.publication_tables / "table_lambda_tradeoff.tex", trade_tex)

    print(f"Wrote LaTeX tables to {args.publication_tables}")
    print(f"Wrote CSV table summaries to {args.results_tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
