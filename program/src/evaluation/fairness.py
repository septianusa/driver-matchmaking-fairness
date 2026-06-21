from __future__ import annotations

import math

import numpy as np
import pandas as pd


def gini(values) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return 0.0
    if np.amin(array) < 0:
        array = array - np.amin(array)
    if np.allclose(array, 0):
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def safe_correlation(x, y, method: str = "pearson") -> float:
    series_x = pd.Series(x, dtype=float)
    series_y = pd.Series(y, dtype=float)
    if len(series_x) < 2 or series_x.nunique(dropna=True) <= 1 or series_y.nunique(dropna=True) <= 1:
        return 0.0
    if method == "spearman":
        series_x = series_x.rank(method="average")
        series_y = series_y.rank(method="average")
    x_values = series_x.to_numpy(dtype=float)
    y_values = series_y.to_numpy(dtype=float)
    x_centered = x_values - np.nanmean(x_values)
    y_centered = y_values - np.nanmean(y_values)
    denominator = math.sqrt(float(np.nansum(x_centered**2) * np.nansum(y_centered**2)))
    if denominator == 0:
        return 0.0
    value = float(np.nansum(x_centered * y_centered) / denominator)
    if math.isnan(value):
        return 0.0
    return value
