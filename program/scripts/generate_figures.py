from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "csv"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
PUBLICATION_FIGURES = PROJECT_ROOT / "publication" / "figures"


TOKENS = {
    "surface": colors.HexColor("#FCFCFD"),
    "panel": colors.white,
    "ink": colors.HexColor("#1F2430"),
    "muted": colors.HexColor("#6F768A"),
    "grid": colors.HexColor("#E6E8F0"),
    "axis": colors.HexColor("#D7DBE7"),
    "blue": colors.HexColor("#2E4780"),
    "orange": colors.HexColor("#804126"),
    "olive": colors.HexColor("#386411"),
    "neutral": colors.HexColor("#7A828F"),
    "light_blue": colors.HexColor("#CEDFFE"),
    "light_orange": colors.HexColor("#FFBDA1"),
}


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def make_canvas(path: Path, width: float = 6.7 * inch, height: float = 4.1 * inch) -> canvas.Canvas:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setFillColor(TOKENS["surface"])
    c.rect(0, 0, width, height, fill=1, stroke=0)
    return c


def header(c: canvas.Canvas, title: str, subtitle: str, width: float, height: float) -> None:
    c.setFillColor(TOKENS["ink"])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(34, height - 28, title)
    c.setFillColor(TOKENS["muted"])
    c.setFont("Helvetica", 8.5)
    c.drawString(34, height - 42, subtitle)


def y_ticks(y_min: float, y_max: float, count: int = 5) -> list[float]:
    if y_max == y_min:
        y_max = y_min + 1
    return [y_min + i * (y_max - y_min) / (count - 1) for i in range(count)]


def draw_marker(c: canvas.Canvas, marker: str, x: float, y: float, color: colors.Color) -> None:
    c.setFillColor(color)
    c.setStrokeColor(color)
    if marker == "square":
        c.rect(x - 3.5, y - 3.5, 7, 7, fill=1, stroke=0)
    elif marker == "triangle":
        c.line(x, y + 5, x - 5, y - 4)
        c.line(x - 5, y - 4, x + 5, y - 4)
        c.line(x + 5, y - 4, x, y + 5)
    else:
        c.circle(x, y, 3.8, fill=1, stroke=0)


def draw_line_chart(
    path: Path,
    data: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    subtitle: str,
    y_label: str,
    y_transform: Callable[[float], float] = lambda x: x,
    y_digits: int = 2,
) -> None:
    width, height = 6.7 * inch, 4.05 * inch
    c = make_canvas(path, width, height)
    header(c, title, subtitle, width, height)

    grouped = (
        data.groupby(["lambda_driver_score", "grid_setting"], dropna=False)[y_col]
        .mean()
        .reset_index()
        .sort_values(["grid_setting", "lambda_driver_score"])
    )
    grouped["plot_value"] = grouped[y_col].map(y_transform)
    x_values = sorted(grouped["lambda_driver_score"].unique())
    series_names = ["grid_on", "grid_off"]
    colors_map = {"grid_on": TOKENS["blue"], "grid_off": TOKENS["orange"]}
    markers = {"grid_on": "circle", "grid_off": "square"}
    dashes = {"grid_on": None, "grid_off": [4, 3]}

    y_min = float(grouped["plot_value"].min())
    y_max = float(grouped["plot_value"].max())
    pad = (y_max - y_min) * 0.16 if y_max > y_min else 1.0
    y_min -= pad
    y_max += pad

    x0, y0 = 64, 56
    plot_w, plot_h = width - 98, height - 120

    c.setStrokeColor(TOKENS["axis"])
    c.setLineWidth(1)
    c.line(x0, y0, x0, y0 + plot_h)
    c.line(x0, y0, x0 + plot_w, y0)

    c.setFont("Helvetica", 7.5)
    for tick in y_ticks(y_min, y_max):
        ty = y0 + (tick - y_min) / (y_max - y_min) * plot_h
        c.setStrokeColor(TOKENS["grid"])
        c.setDash(1, 2)
        c.line(x0, ty, x0 + plot_w, ty)
        c.setDash()
        c.setFillColor(TOKENS["muted"])
        c.drawRightString(x0 - 7, ty - 2.5, fmt_num(tick, y_digits))

    x_positions = {
        x: x0 + i * plot_w / (len(x_values) - 1 if len(x_values) > 1 else 1)
        for i, x in enumerate(x_values)
    }
    c.setFillColor(TOKENS["muted"])
    for x in x_values:
        c.drawCentredString(x_positions[x], y0 - 16, f"{x:.1f}")
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(x0 + plot_w / 2, 22, "Driver-score weight (lambda)")
    c.saveState()
    c.translate(14, y0 + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_label)
    c.restoreState()

    for s in series_names:
        sdf = grouped[grouped["grid_setting"].eq(s)].sort_values("lambda_driver_score")
        points = []
        for _, r in sdf.iterrows():
            x = x_positions[float(r["lambda_driver_score"])]
            y = y0 + (float(r["plot_value"]) - y_min) / (y_max - y_min) * plot_h
            points.append((x, y))
        c.setStrokeColor(colors_map[s])
        c.setFillColor(colors_map[s])
        c.setLineWidth(1.7)
        if dashes[s]:
            c.setDash(*dashes[s])
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            c.line(x1, y1, x2, y2)
        c.setDash()
        for x, y in points:
            draw_marker(c, markers[s], x, y, colors_map[s])

    legend_x, legend_y = width - 150, height - 30
    c.setFont("Helvetica", 8)
    for i, s in enumerate(series_names):
        y = legend_y - i * 14
        c.setStrokeColor(colors_map[s])
        c.setLineWidth(1.5)
        if dashes[s]:
            c.setDash(*dashes[s])
        c.line(legend_x, y, legend_x + 18, y)
        c.setDash()
        draw_marker(c, markers[s], legend_x + 9, y, colors_map[s])
        c.setFillColor(TOKENS["ink"])
        c.drawString(legend_x + 25, y - 3, s.replace("_", " "))

    c.showPage()
    c.save()


def draw_runtime_bar(path: Path, data: pd.DataFrame) -> None:
    width, height = 6.7 * inch, 4.05 * inch
    c = make_canvas(path, width, height)
    header(c, "Runtime comparison by solver", "Actual-data simulation outputs averaged across days, grid settings, lambdas, and sparse handlers", width, height)

    grouped = (
        data.groupby(["solver", "sparse_handler"], dropna=False)["runtime_seconds"]
        .mean()
        .reset_index()
    )
    solvers = ["hungarian", "greedy"]
    sparse = ["BFS", "A2GAT"]
    values = {
        (r["solver"], r["sparse_handler"]): float(r["runtime_seconds"])
        for _, r in grouped.iterrows()
    }
    y_max = max(values.values()) * 1.15
    x0, y0 = 64, 56
    plot_w, plot_h = width - 110, height - 120
    c.setStrokeColor(TOKENS["axis"])
    c.line(x0, y0, x0, y0 + plot_h)
    c.line(x0, y0, x0 + plot_w, y0)
    c.setFont("Helvetica", 7.5)
    for tick in y_ticks(0, y_max):
        ty = y0 + tick / y_max * plot_h
        c.setStrokeColor(TOKENS["grid"])
        c.setDash(1, 2)
        c.line(x0, ty, x0 + plot_w, ty)
        c.setDash()
        c.setFillColor(TOKENS["muted"])
        c.drawRightString(x0 - 7, ty - 2.5, fmt_num(tick, 0))

    group_w = plot_w / len(solvers)
    bar_w = 38
    color_map = {"BFS": TOKENS["blue"], "A2GAT": TOKENS["orange"]}
    for i, solver in enumerate(solvers):
        center = x0 + group_w * (i + 0.5)
        for j, sp in enumerate(sparse):
            value = values.get((solver, sp), 0.0)
            x = center + (j - 0.5) * (bar_w + 10)
            h = value / y_max * plot_h
            c.setFillColor(color_map[sp])
            c.rect(x - bar_w / 2, y0, bar_w, h, fill=1, stroke=0)
            c.setFillColor(TOKENS["ink"])
            c.setFont("Helvetica", 7)
            c.drawCentredString(x, y0 + h + 5, fmt_num(value, 0))
        c.setFillColor(TOKENS["muted"])
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(center, y0 - 18, solver.title())

    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(x0 + plot_w / 2, 22, "Solver")
    c.saveState()
    c.translate(14, y0 + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Runtime (seconds)")
    c.restoreState()

    c.setFont("Helvetica", 8)
    lx, ly = width - 132, height - 30
    for i, sp in enumerate(sparse):
        c.setFillColor(color_map[sp])
        c.rect(lx, ly - i * 14 - 6, 10, 10, fill=1, stroke=0)
        c.setFillColor(TOKENS["ink"])
        c.drawString(lx + 16, ly - i * 14 - 4, sp)
    c.showPage()
    c.save()


def factor_delta(data: pd.DataFrame, metric: str, high_filter: pd.Series, low_filter: pd.Series, transform: Callable[[float], float]) -> float:
    return transform(float(data.loc[high_filter, metric].mean()) - float(data.loc[low_filter, metric].mean()))


def draw_delta_panel(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    labels: list[str],
    values: list[float],
    digits: int,
) -> None:
    c.setFillColor(colors.white)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=0)
    c.setFillColor(TOKENS["ink"])
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 10, y + h - 18, title)
    max_abs = max(max(abs(v) for v in values), 1e-9) * 1.2
    center_x = x + w * 0.57
    bar_area = w * 0.32
    row_gap = (h - 46) / len(labels)
    c.setStrokeColor(TOKENS["axis"])
    c.line(center_x, y + 12, center_x, y + h - 28)
    c.setFont("Helvetica", 7.2)
    for i, (label, value) in enumerate(zip(labels, values)):
        yy = y + h - 42 - i * row_gap
        c.setFillColor(TOKENS["muted"])
        c.drawRightString(x + w * 0.38, yy - 2, label)
        length = abs(value) / max_abs * bar_area
        c.setFillColor(TOKENS["blue"] if value >= 0 else TOKENS["orange"])
        if value >= 0:
            c.rect(center_x, yy - 5, length, 10, fill=1, stroke=0)
            label_x = center_x + length + 4
        else:
            c.rect(center_x - length, yy - 5, length, 10, fill=1, stroke=0)
            label_x = center_x - length - 28
        c.setFillColor(TOKENS["ink"])
        c.drawString(label_x, yy - 2, fmt_num(value, digits))


def draw_scenario_comparison(path: Path, data: pd.DataFrame) -> None:
    width, height = 7.2 * inch, 4.8 * inch
    c = make_canvas(path, width, height)
    header(
        c,
        "Scenario-factor comparison",
        "Mean directional deltas across the scenario-day matrix; positive values mean the first level is higher.",
        width,
        height,
    )
    labels = ["Grid on - off", "A2GAT - BFS", "Greedy - Hung.", "Lambda .3 - .0"]
    filters = [
        (data["grid_setting"].eq("grid_on"), data["grid_setting"].eq("grid_off")),
        (data["sparse_handler"].eq("A2GAT"), data["sparse_handler"].eq("BFS")),
        (data["solver"].eq("greedy"), data["solver"].eq("hungarian")),
        (data["lambda_driver_score"].round(3).eq(0.3), data["lambda_driver_score"].round(3).eq(0.0)),
    ]
    metrics = [
        ("Expected conversion (pp)", "expected_conversion_rate", lambda d: d * 100.0, 2),
        ("Pickup distance (m)", "pickup_distance_km", lambda d: d * 1000.0, 1),
        ("Spearman correlation", "spearman_correlation", lambda d: d, 3),
        ("Runtime (s)", "runtime_seconds", lambda d: d, 1),
    ]
    panel_w = (width - 88) / 2
    panel_h = (height - 104) / 2
    positions = [(34, height - 58 - panel_h), (54 + panel_w, height - 58 - panel_h), (34, 34), (54 + panel_w, 34)]
    for (title, metric, transform, digits), (x, y) in zip(metrics, positions):
        values = [factor_delta(data, metric, hi, lo, transform) for hi, lo in filters]
        draw_delta_panel(c, x, y, panel_w, panel_h, title, labels, values, digits)
    c.showPage()
    c.save()


def copy_figure(drawer: Callable[[Path], None], filename: str, output_dirs: list[Path]) -> None:
    for directory in output_dirs:
        drawer(directory / filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate publication-ready PDF figures.")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--results-figures", type=Path, default=RESULTS_FIGURES)
    parser.add_argument("--publication-figures", type=Path, default=PUBLICATION_FIGURES)
    args = parser.parse_args()

    by_day_path = args.results_csv / "scenario_results_by_day.csv"
    if not by_day_path.exists():
        raise FileNotFoundError("Run prepare_results.py before generate_figures.py.")
    data = pd.read_csv(by_day_path)

    for d in [args.results_figures, args.publication_figures]:
        d.mkdir(parents=True, exist_ok=True)

    outputs = [args.results_figures, args.publication_figures]

    figure_specs: list[dict[str, str]] = []
    line_specs = [
        (
            "fig_conversion_by_lambda.pdf",
            "expected_conversion_rate",
            "Expected conversion rate by lambda",
            "Mean simulator-expected conversion; grid on/off shown with distinct marker shapes",
            "Expected conversion (%)",
            lambda x: x * 100.0,
            2,
        ),
        (
            "fig_pickup_distance_by_lambda.pdf",
            "pickup_distance_km",
            "Pickup distance by lambda",
            "Mean median pickup distance across solvers, sparse handlers, and days",
            "Pickup distance (km)",
            lambda x: x,
            3,
        ),
        (
            "fig_utility_by_lambda.pdf",
            "utility",
            "Utility by lambda",
            "Mean daily utility across solvers, sparse handlers, and days",
            "Utility/day (M)",
            lambda x: x / 1_000_000.0,
            1,
        ),
        (
            "fig_spearman_by_lambda.pdf",
            "spearman_correlation",
            "Score-income rank alignment by lambda",
            "Mean Spearman correlation across solvers, sparse handlers, and days",
            "Spearman correlation",
            lambda x: x,
            3,
        ),
        (
            "fig_gini_by_lambda.pdf",
            "gini_coefficient",
            "Income Gini coefficient by lambda",
            "Mean Gini coefficient across solvers, sparse handlers, and days",
            "Gini coefficient",
            lambda x: x,
            3,
        ),
    ]
    for filename, y_col, title, subtitle, ylabel, transform, digits in line_specs:
        copy_figure(
            lambda path, y_col=y_col, title=title, subtitle=subtitle, ylabel=ylabel, transform=transform, digits=digits: draw_line_chart(
                path,
                data,
                y_col=y_col,
                title=title,
                subtitle=subtitle,
                y_label=ylabel,
                y_transform=transform,
                y_digits=digits,
            ),
            filename,
            outputs,
        )
        figure_specs.append({"file": filename, "chart_type": "line", "metric": y_col})

    copy_figure(lambda path: draw_runtime_bar(path, data), "fig_runtime_by_solver.pdf", outputs)
    figure_specs.append({"file": "fig_runtime_by_solver.pdf", "chart_type": "grouped_bar", "metric": "runtime_seconds"})

    copy_figure(lambda path: draw_scenario_comparison(path, data), "fig_scenario_comparison.pdf", outputs)
    figure_specs.append({"file": "fig_scenario_comparison.pdf", "chart_type": "delta_panels", "metric": "multiple"})

    pd.DataFrame(figure_specs).to_csv(args.results_csv / "figure_manifest.csv", index=False)
    print(f"Wrote PDF figures to {args.results_figures} and {args.publication_figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
