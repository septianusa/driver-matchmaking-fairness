from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_project_path, yaml_dump
from src.comparison.comparison_runner import run_model_comparison
from src.evaluation.report import export_zip
from src.simulation.engine import SimulationEngine
from src.simulation.scenario_runner import run_comparison
from src.ui import charts
from src.validation import validate_inputs


def _st():
    import streamlit as st  # type: ignore

    return st


def load_default_config() -> dict:
    return load_config("configs/default.yaml")


def page_home() -> None:
    st = _st()
    st.title("Driver Behavior-Based Dispatching")
    st.subheader("Improve income fairness in ride-hailing platforms")
    st.write(
        "This local application runs discrete-event dispatch simulations that connect "
        "driver behavior, allocation priority, completed orders, and driver income."
    )
    st.markdown(
        """
        **Research objective:** test whether driver-controllable behavior can be included in
        the matchmaking objective while preserving operational guardrails.

        **Simulation overview:** one-minute batches, H3 candidate retrieval, strict haversine
        feasibility, one-to-one matching, counterfactual driver state, order expiry, grid-value
        learning, and fairness evaluation.

        **Required datasets:** `orders.csv`, `driver_locations.csv`, and `driver_scores.csv`.
        Optional `grid_values.csv` can initialize spatial values.
        """
    )


def page_data_upload() -> None:
    st = _st()
    st.header("Data Upload")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    uploads = {
        "orders.csv": st.file_uploader("orders.csv", type=["csv", "parquet"], key="orders_upload"),
        "driver_locations.csv": st.file_uploader(
            "driver_locations.csv", type=["csv", "parquet"], key="locations_upload"
        ),
        "driver_scores.csv": st.file_uploader(
            "driver_scores.csv", type=["csv", "parquet"], key="scores_upload"
        ),
        "grid_values.csv": st.file_uploader(
            "grid_values.csv", type=["csv", "parquet"], key="grid_upload"
        ),
    }
    for filename, upload in uploads.items():
        if upload is not None:
            target = raw_dir / filename
            target.write_bytes(upload.getbuffer())
            st.success(f"Saved {target}")
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        st.subheader(path.name)
        st.caption(f"{len(frame):,} rows, {len(frame.columns):,} columns")
        st.dataframe(frame.head(20), use_container_width=True)
        st.write("Missing values")
        st.json({column: int(count) for column, count in frame.isna().sum().items() if count})


def page_validation(config: dict) -> None:
    st = _st()
    st.header("Validation")
    report = validate_inputs(config)
    left, right = st.columns(2)
    left.metric("Orders", report.row_counts.get("orders", 0))
    right.metric("Unique drivers", report.unique_driver_count)
    if report.blocking_errors:
        st.error("Blocking errors")
        for item in report.blocking_errors:
            st.write(f"- {item}")
    else:
        st.success("Validation passed")
    if report.warnings:
        st.warning("Warnings")
        for item in report.warnings:
            st.write(f"- {item}")
    st.subheader("Report")
    st.json(report.to_dict())
    st.download_button(
        "Download validation_report.json",
        data=json.dumps(report.to_dict(), indent=2),
        file_name="validation_report.json",
        mime="application/json",
    )


def page_configuration(config: dict) -> dict:
    st = _st()
    st.header("Configuration")
    updated = json.loads(json.dumps({k: v for k, v in config.items() if not k.startswith("_")}))
    simulation = updated.setdefault("simulation", {})
    spatial = updated.setdefault("spatial", {})
    utility = updated.setdefault("utility", {})
    cancellation = updated.setdefault("cancellation_model", {})
    drivers = updated.setdefault("drivers", {})
    matching = updated.setdefault("matching", {})
    col1, col2, col3 = st.columns(3)
    simulation["simulation_date"] = str(col1.text_input("Simulation date", simulation["simulation_date"]))
    simulation["total_batches"] = int(col2.number_input("Batch horizon", 1, 1440, int(simulation["total_batches"])))
    simulation["random_seed"] = int(col3.number_input("Random seed", 0, 1_000_000, int(simulation["random_seed"])))
    spatial["h3_resolution"] = int(st.slider("H3 resolution", 5, 11, int(spatial["h3_resolution"])))
    spatial["maximum_pickup_distance_km"] = float(
        st.slider("Maximum pickup distance", 0.5, 15.0, float(spatial["maximum_pickup_distance_km"]))
    )
    spatial["max_h3_hops"] = int(st.slider("Maximum H3 hops", 0, 8, int(spatial["max_h3_hops"])))
    spatial["candidate_driver_target"] = int(
        st.number_input("Candidate target", 1, 500, int(spatial["candidate_driver_target"]))
    )
    matching["strategy"] = st.selectbox(
        "Matching strategy", ["hungarian", "greedy"], index=0 if matching["strategy"] == "hungarian" else 1
    )
    drivers["positioning_mode"] = st.selectbox(
        "Positioning mode",
        [
            "historical_bootstrap_simulated_state",
            "historical_replay",
            "initial_snapshot_simulated_state",
        ],
    )
    drivers["enable_idle_random_walk"] = st.checkbox(
        "Enable idle random walk", bool(drivers["enable_idle_random_walk"])
    )
    utility["lambda_driver_score"] = float(
        st.slider("Lambda driver score", 0.0, 2.0, float(utility["lambda_driver_score"]))
    )
    utility["gamma_grid_value"] = float(
        st.slider("Gamma grid value", 0.0, 2.0, float(utility["gamma_grid_value"]))
    )
    utility["alpha_grid_learning_rate"] = float(
        st.slider("Alpha learning rate", 0.0, 1.0, float(utility["alpha_grid_learning_rate"]))
    )
    cancellation["mode"] = st.selectbox("Cancellation model", ["score_distance", "ar_cr_distance"])
    cancellation["beta_0"] = float(st.number_input("Beta 0", value=float(cancellation["beta_0"])))
    cancellation["beta_1_score_gap"] = float(
        st.number_input("Beta score gap", value=float(cancellation.get("beta_1_score_gap", 2.0)))
    )
    cancellation["beta_2_pickup_distance"] = float(
        st.number_input(
            "Beta pickup distance", value=float(cancellation.get("beta_2_pickup_distance", 2.0))
        )
    )
    st.download_button(
        "Download resolved YAML",
        data=yaml_dump(updated),
        file_name="resolved_config.yaml",
        mime="text/yaml",
    )
    st.session_state["config"] = updated | {"_project_root": str(Path(".").resolve())}
    return st.session_state["config"]


def page_run_simulation(config: dict) -> None:
    st = _st()
    st.header("Run Simulation")
    mode = st.radio("Run mode", ["Single scenario", "Predefined matrix"], horizontal=True)
    if mode == "Single scenario":
        scenario_id = st.text_input("Scenario ID", "streamlit_single")
        if st.button("Run scenario"):
            progress = st.progress(0)
            status = st.empty()

            def update(record):
                progress.progress(record["current_batch"] / record["total_batches"])
                status.write(
                    f"Batch {record['current_batch']} | active orders {record['active_orders']} | "
                    f"matches {record['matches_created']}"
                )

            result = SimulationEngine(config).run(scenario_id=scenario_id, progress_callback=update)
            st.session_state["last_result_dir"] = str(result.output_dir)
            st.success(f"Completed {result.scenario_id}")
            st.json(result.summary)
    else:
        if st.button("Run experiment matrix"):
            comparison, normalized, output_dir = run_comparison("configs/experiment_matrix.yaml")
            st.session_state["last_result_dir"] = str(output_dir)
            st.success(f"Comparison written to {output_dir}")
            st.dataframe(comparison, use_container_width=True)
            st.dataframe(normalized, use_container_width=True)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def page_results() -> None:
    st = _st()
    st.header("Results Dashboard")
    result_dir = Path(st.session_state.get("last_result_dir", "outputs"))
    if result_dir.is_dir() and not (result_dir / "scenario_summary.csv").exists():
        scenario_dirs = [path for path in result_dir.iterdir() if path.is_dir()]
        if scenario_dirs:
            result_dir = max(scenario_dirs, key=lambda path: path.stat().st_mtime)
    st.caption(str(result_dir))
    summary = _read_optional_csv(result_dir / "scenario_summary.csv")
    batch = _read_optional_csv(result_dir / "batch_metrics.csv")
    matches = _read_optional_csv(result_dir / "match_log.csv")
    drivers = _read_optional_csv(result_dir / "driver_metrics.csv")
    grid = _read_optional_csv(result_dir / "grid_values_final.csv")
    if summary.empty:
        st.info("Run a simulation first.")
        return
    metrics = summary.iloc[0].to_dict()
    cols = st.columns(4)
    for idx, key in enumerate(
        [
            "total_orders",
            "match_rate",
            "expected_conversion_rate",
            "gini_coefficient",
            "median_pickup_distance_km",
            "total_expected_economic_utility",
            "pearson_score_income_correlation",
            "runtime_seconds",
        ]
    ):
        value = metrics.get(key, 0)
        cols[idx % 4].metric(key.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, float) else value)
    st.plotly_chart(charts.score_income_scatter(drivers), use_container_width=True)
    st.plotly_chart(charts.score_completed_scatter(drivers), use_container_width=True)
    st.plotly_chart(charts.income_histogram(drivers), use_container_width=True)
    st.plotly_chart(charts.decile_bar(drivers, "total_expected_income", "Average income by score decile"), use_container_width=True)
    st.plotly_chart(charts.decile_bar(drivers, "expected_completed_orders", "Completed orders by score decile"), use_container_width=True)
    st.plotly_chart(charts.hourly_rate(batch, "matches_created", "orders_created", "Match rate by hour"), use_container_width=True)
    st.plotly_chart(charts.distribution(matches, "pickup_distance_km", "Pickup-distance distribution"), use_container_width=True)
    st.plotly_chart(charts.distribution(matches, "predicted_cancellation_probability", "Predicted cancellation distribution"), use_container_width=True)
    st.plotly_chart(charts.distribution(matches, "candidate_count_after_distance_filter", "Candidate-count distribution"), use_container_width=True)
    st.plotly_chart(charts.distribution(matches, "max_bfs_depth_reached", "BFS-hop distribution"), use_container_width=True)
    st.plotly_chart(charts.runtime_by_batch(batch), use_container_width=True)
    st.plotly_chart(charts.grid_value_distribution(grid), use_container_width=True)
    st.plotly_chart(charts.h3_map(matches), use_container_width=True)


def page_export() -> None:
    st = _st()
    st.header("Export")
    result_dir = Path(st.session_state.get("last_result_dir", "outputs"))
    if not result_dir.exists():
        st.info("No output directory found yet.")
        return
    for path in sorted(result_dir.glob("*")):
        if path.is_file():
            st.download_button(path.name, path.read_bytes(), file_name=path.name)
    if st.button("Build ZIP archive"):
        zip_path = export_zip(result_dir, result_dir.with_suffix(".zip"))
        st.success(f"Created {zip_path}")
        st.download_button("Download ZIP", zip_path.read_bytes(), file_name=zip_path.name)


def page_model_comparison() -> None:
    st = _st()
    st.header("Model Comparison")
    config = load_config("configs/model_comparison.yaml")
    comparison = config.setdefault("comparison", {})
    data_mode = "actual"
    st.caption("Data mode: actual")
    selected_lambdas = st.multiselect(
        "Driver-score weights",
        [0.0, 0.1, 0.2, 0.3, 0.4],
        default=comparison.get("lambda_driver_score_values", [0.0, 0.1, 0.2, 0.3, 0.4]),
    )
    selected_algorithms = st.multiselect(
        "Matching algorithms",
        ["hungarian", "greedy", "auction"],
        default=comparison.get("matching_algorithms", ["hungarian", "greedy", "auction"]),
    )
    selected_sparse = st.multiselect(
        "Sparse methods",
        ["bfs_h3", "a2gat"],
        default=comparison.get("sparse_methods", ["bfs_h3", "a2gat"]),
    )
    col1, col2, col3 = st.columns(3)
    repeats = int(col1.number_input("Repeats", min_value=1, max_value=20, value=int(comparison.get("repeats", 3))))
    warmups = int(col2.number_input("Warm-up runs", min_value=0, max_value=10, value=int(comparison.get("warmup_runs", 1))))
    simulation_date = col3.text_input("Simulation date", "2025-04-23")
    col4, col5 = st.columns(2)
    batch_start = int(col4.number_input("Batch start", min_value=1, max_value=1440, value=1))
    batch_end = int(col5.number_input("Batch end", min_value=batch_start, max_value=1440, value=1440))
    measure_memory = st.checkbox("Measure memory", value=bool(comparison.get("measure_memory", True)))
    measure_cpu = st.checkbox("Measure CPU", value=bool(comparison.get("measure_cpu", True)))

    comparison.update(
        {
            "data_mode": data_mode,
            "lambda_driver_score_values": selected_lambdas,
            "matching_algorithms": selected_algorithms,
            "sparse_methods": selected_sparse,
            "repeats": repeats,
            "warmup_runs": warmups,
            "measure_memory": measure_memory,
            "measure_cpu": measure_cpu,
            "batch_start": batch_start,
            "batch_end": batch_end,
        }
    )
    config.setdefault("data", {}).setdefault("actual", {})["simulation_date"] = simulation_date
    run_count = len(selected_lambdas) * len(selected_algorithms) * len(selected_sparse)
    st.metric("Variant count", run_count)
    st.caption(f"Measured runs: {run_count * repeats}; warm-up runs: {run_count * warmups}")
    tmp_config = Path("outputs") / "streamlit_model_comparison_config.yaml"
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(yaml_dump({k: v for k, v in config.items() if not k.startswith("_")}), encoding="utf-8")

    if st.button("Dry-run validation"):
        plan = run_model_comparison(tmp_config, data_mode_override=data_mode, dry_run_only=True)
        st.session_state["model_comparison_plan"] = plan
        st.json(plan)

    if st.button("Run full comparison"):
        progress = st.progress(0)
        status = st.empty()

        def update(record: dict) -> None:
            if record.get("event") in {"setup_start", "dataset_profile_ready", "comparison_complete"}:
                status.write(record)
                return
            total = max(int(record.get("total_runs", 1)), 1)
            progress.progress(min(float(record.get("run_counter", 0)) / total, 1.0))
            status.write(
                f"{record.get('variant_id')} | repeat {record.get('repeat_index')} | "
                f"warmup={record.get('warmup_flag')}"
            )

        result = run_model_comparison(tmp_config, data_mode_override=data_mode, progress_callback=update)
        st.session_state["model_comparison_output_dir"] = str(result.output_dir)
        st.success(f"Comparison completed: {result.run_id}")
        st.subheader("Dataset Profile")
        st.json(result.dataset_profile)
        st.subheader("Summary")
        st.dataframe(result.summary, use_container_width=True)

    output_dir = Path(st.session_state.get("model_comparison_output_dir", ""))
    if output_dir.exists():
        st.subheader("Downloads")
        for path in sorted(output_dir.glob("comparison_*.csv")):
            st.download_button(path.name, path.read_bytes(), file_name=path.name)
        report = output_dir / "comparison_report.html"
        if report.exists():
            st.download_button("comparison_report.html", report.read_bytes(), file_name=report.name)
        summary = _read_optional_csv(output_dir / "comparison_summary.csv")
        if not summary.empty:
            st.subheader("Latest Summary")
            st.dataframe(summary, use_container_width=True)
