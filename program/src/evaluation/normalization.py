from __future__ import annotations

import pandas as pd


LOWER_IS_BETTER = {"gini_coefficient", "median_pickup_distance_km", "runtime_seconds"}


def normalize_against_baseline(results: pd.DataFrame, baseline_scenario: str = "baseline") -> pd.DataFrame:
    if results.empty or "scenario_id" not in results:
        return results.copy()
    baseline_rows = results[results["scenario_id"] == baseline_scenario]
    if baseline_rows.empty:
        baseline_rows = results.head(1)
    baseline = baseline_rows.iloc[0].to_dict()
    rows = []
    for row in results.to_dict("records"):
        normalized = {"scenario_id": row["scenario_id"]}
        for key, value in row.items():
            if key == "scenario_id" or not isinstance(value, (int, float)):
                continue
            base = baseline.get(key)
            if not isinstance(base, (int, float)) or base == 0:
                normalized[f"{key}_normalized"] = 1.0 if value == base else 0.0
            elif key in LOWER_IS_BETTER:
                normalized[f"{key}_normalized"] = float(base) / float(value) if value else 1.0
            else:
                normalized[f"{key}_normalized"] = float(value) / float(base)
        rows.append(normalized)
    return pd.DataFrame(rows)

