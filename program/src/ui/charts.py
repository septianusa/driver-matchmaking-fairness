from __future__ import annotations

import pandas as pd


def _px():
    import plotly.express as px  # type: ignore

    return px


def score_income_scatter(driver_metrics: pd.DataFrame):
    px = _px()
    return px.scatter(
        driver_metrics,
        x="driver_score",
        y="total_expected_income",
        size="assigned_orders",
        hover_name="driver_id",
        labels={
            "driver_score": "Driver score",
            "total_expected_income": "Expected income",
            "assigned_orders": "Assigned orders",
        },
    )


def score_completed_scatter(driver_metrics: pd.DataFrame):
    px = _px()
    return px.scatter(
        driver_metrics,
        x="driver_score",
        y="expected_completed_orders",
        hover_name="driver_id",
        labels={"driver_score": "Driver score", "expected_completed_orders": "Expected completed orders"},
    )


def income_histogram(driver_metrics: pd.DataFrame):
    px = _px()
    return px.histogram(driver_metrics, x="total_expected_income", nbins=20)


def decile_bar(driver_metrics: pd.DataFrame, metric: str, title: str):
    px = _px()
    frame = driver_metrics.copy()
    if frame.empty:
        return px.bar(pd.DataFrame({"score_decile": [], metric: []}), x="score_decile", y=metric)
    frame["score_decile"] = pd.qcut(
        frame["driver_score"].rank(method="first"), q=min(10, len(frame)), labels=False
    ) + 1
    grouped = frame.groupby("score_decile", as_index=False)[metric].mean()
    return px.bar(grouped, x="score_decile", y=metric, title=title)


def hourly_rate(batch_metrics: pd.DataFrame, numerator: str, denominator: str, title: str):
    px = _px()
    frame = batch_metrics.copy()
    if frame.empty:
        return px.line(pd.DataFrame({"hour": [], "rate": []}), x="hour", y="rate", title=title)
    frame["hour"] = ((frame["batch_id"] - 1) // 60).astype(int)
    grouped = frame.groupby("hour", as_index=False)[[numerator, denominator]].sum()
    grouped["rate"] = grouped[numerator] / grouped[denominator].replace(0, pd.NA)
    grouped["rate"] = grouped["rate"].fillna(0.0)
    return px.line(grouped, x="hour", y="rate", title=title)


def distribution(frame: pd.DataFrame, column: str, title: str):
    px = _px()
    if frame.empty or column not in frame:
        return px.histogram(pd.DataFrame({column: []}), x=column, title=title)
    return px.histogram(frame, x=column, nbins=30, title=title)


def runtime_by_batch(batch_metrics: pd.DataFrame):
    px = _px()
    return px.line(batch_metrics, x="batch_id", y="runtime_seconds", title="Runtime by batch")


def grid_value_distribution(grid_values: pd.DataFrame):
    px = _px()
    return px.histogram(grid_values, x="grid_value", nbins=30, title="Grid-value distribution")


def h3_map(match_log: pd.DataFrame):
    px = _px()
    if match_log.empty:
        return px.scatter_mapbox(pd.DataFrame({"lat": [], "lon": []}), lat="lat", lon="lon")
    frame = match_log.rename(
        columns={"destination_latitude": "lat", "destination_longitude": "lon"}
    ).copy()
    return px.scatter_mapbox(
        frame,
        lat="lat",
        lon="lon",
        color="destination_grid_value_before",
        hover_name="destination_h3_index",
        zoom=11,
        height=420,
        mapbox_style="open-street-map",
    )

