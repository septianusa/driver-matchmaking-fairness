from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import load_config, resolve_project_path
from src.comparison.comparison_runner import run_model_comparison, run_model_comparison_days
from src.comparison.paper_report import build_paper_report_folder
from src.evaluation.batch_match_report import build_batch_match_report
from src.evaluation.report import export_zip
from src.simulation.engine import SimulationEngine
from src.simulation.multiday_runner import run_multiday_actual
from src.simulation.scenario_runner import run_comparison
from src.validation import validate_inputs


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = validate_inputs(config)
    output_dir = resolve_project_path(config, config.get("output", {}).get("base_directory", "outputs"))
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.json"
    report.write(report_path)
    print(f"Validation report written to {report_path}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    if report.blocking_errors:
        print("Blocking errors:")
        for error in report.blocking_errors:
            print(f"  - {error}")
        return 1
    print("Validation passed.")
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = SimulationEngine(config).run(scenario_id=args.scenario_id)
    print(f"Simulation completed: {result.scenario_id}")
    print(f"Outputs written to {result.output_dir}")
    print(f"Match rate: {result.summary['match_rate']:.3f}")
    print(f"Gini coefficient: {result.summary['gini_coefficient']:.3f}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    comparison, normalized, output_dir = run_comparison(args.config)
    print(f"Scenario comparison written to {output_dir}")
    print(comparison[["scenario_id", "match_rate", "gini_coefficient"]].to_string(index=False))
    print("Normalized metrics:")
    print(normalized.head().to_string(index=False))
    return 0


def command_simulate_days(args: argparse.Namespace) -> int:
    def progress(record: dict) -> None:
        event = record.get("event")
        elapsed = float(record.get("elapsed_total_seconds") or 0.0)
        if event == "multiday_start":
            print(
                "[simulate-days] "
                f"dates={record.get('date_count')} "
                f"range={record.get('start_date')}..{record.get('end_date')} "
                f"output={record.get('output_dir')} "
                f"batch_interval={record.get('progress_interval_batches')} "
                f"candidate_edges={record.get('candidate_edge_logging')}",
                flush=True,
            )
            return
        if event == "day_start":
            day_index = int(record.get("day_index") or 0)
            day_count = int(record.get("day_count") or 0)
            completed_days = max(day_index - 1, 0)
            eta = None
            if completed_days > 0 and day_count:
                eta = (elapsed / completed_days) * (day_count - completed_days)
            percent = ((completed_days / day_count) * 100.0) if day_count else 0.0
            print(
                f"[day {day_index}/{day_count} | {percent:5.1f}%] START "
                f"date={record.get('date')} scenario={record.get('scenario_id')} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
                f"grid_input={record.get('input_grid_values_path')}",
                flush=True,
            )
            return
        if event == "batch_progress":
            day_index = int(record.get("day_index") or 0)
            day_count = int(record.get("day_count") or 0)
            batch = int(record.get("current_batch") or 0)
            total_batches = int(record.get("total_batches") or 0)
            batch_pct = (batch / total_batches * 100.0) if total_batches else 0.0
            print(
                f"[day {day_index}/{day_count} {record.get('date')} | "
                f"batch {batch}/{total_batches} {batch_pct:5.1f}%] "
                f"day_elapsed={_format_duration(record.get('day_elapsed_seconds'))} "
                f"day_eta={_format_duration(record.get('day_eta_seconds'))} "
                f"orders={record.get('orders_created_total')} "
                f"matched={record.get('matches_created_total')} "
                f"active={record.get('active_orders')} "
                f"available_drivers={record.get('available_drivers')} "
                f"candidates={record.get('candidate_edges')}",
                flush=True,
            )
            return
        if event == "day_success":
            day_index = int(record.get("day_index") or 0)
            day_count = int(record.get("day_count") or 0)
            eta = None
            if day_index > 0 and day_count:
                eta = (elapsed / day_index) * (day_count - day_index)
            print(
                f"[day {day_index}/{day_count}] DONE "
                f"date={record.get('date')} "
                f"runtime={_format_duration(record.get('day_runtime_seconds'))} "
                f"orders={record.get('total_orders')} "
                f"matched={record.get('matched_orders')} "
                f"match_rate={float(record.get('match_rate') or 0):.3f} "
                f"gini={float(record.get('gini_coefficient') or 0):.3f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                flush=True,
            )
            return
        if event == "day_failed":
            print(
                f"[day {record.get('day_index')}/{record.get('day_count')}] FAIL "
                f"date={record.get('date')} "
                f"runtime={_format_duration(record.get('day_runtime_seconds'))} "
                f"error={record.get('error_message')}",
                flush=True,
            )
            return
        if event == "research_package_start":
            print(
                "[research-package] START "
                f"dates={record.get('date_count')} elapsed={_format_duration(elapsed)}",
                flush=True,
            )
            return
        if event == "multiday_complete":
            print(
                "[complete] "
                f"dates={record.get('date_count')} "
                f"elapsed={_format_duration(elapsed)} "
                f"research_package_runtime={_format_duration(record.get('research_package_runtime_seconds'))} "
                f"output={record.get('output_dir')} "
                f"research_package={record.get('research_package_dir')}",
                flush=True,
            )

    result = run_multiday_actual(
        args.config,
        output_directory=args.output_directory,
        start_date=args.start_date,
        end_date=args.end_date,
        max_days=args.max_days,
        scenario_prefix=args.scenario_prefix,
        enable_candidate_edge_logging=args.enable_candidate_edge_logging,
        progress_callback=progress,
        progress_interval_batches=args.progress_interval_batches,
    )
    print(f"Multi-day actual simulation completed for {len(result.dates)} day(s).")
    print(f"Dates: {', '.join(result.dates)}")
    print(f"Outputs written to {result.output_dir}")
    print(f"Summary: {result.output_dir / 'multiday_summary.csv'}")
    print(f"Research package: {result.research_package_dir}")
    print(f"Research report: {result.research_package_dir / 'journal_research_report.html'}")
    print(result.summary[["jakarta_data_date", "total_orders", "matched_orders", "match_rate", "gini_coefficient"]].to_string(index=False))
    return 0


def command_compare_models(args: argparse.Namespace) -> int:
    if args.dry_run:
        result = run_model_comparison(
            args.config,
            data_mode_override=args.data_mode,
            dry_run_only=True,
        )
        print("Model comparison dry run")
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0

    plan = run_model_comparison(
        args.config,
        data_mode_override=args.data_mode,
        dry_run_only=True,
    )
    print("Model comparison starting")
    print(f"Data mode: {plan['data_mode']}")
    print(f"Variants: {plan['variant_count']}")
    print(f"Measured runs: {plan['measured_runs']}")
    print(f"Warm-up runs: {plan['warmup_runs']}")
    print(f"Algorithms: {', '.join(plan['matching_algorithms'])}")
    print(f"Sparse methods: {', '.join(plan['sparse_methods'])}")
    a2gat_status = plan.get("sparse_integration_status", {}).get("a2gat", {})
    print(f"A2GAT status: {a2gat_status.get('message') or plan['unavailable_optional_integrations'].get('a2gat')}")
    print("")

    start_time = time.perf_counter()

    def _fmt(seconds: float | None) -> str:
        if seconds is None:
            return "unknown"
        seconds = max(0, int(seconds))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {sec}s"
        if minutes:
            return f"{minutes}m {sec}s"
        return f"{sec}s"

    def progress(record: dict) -> None:
        event = record.get("event")
        elapsed = time.perf_counter() - start_time
        if event == "setup_start":
            print(
                f"[setup] run_id={record.get('run_id')} data_mode={record.get('data_mode')} "
                f"variants={record.get('variant_count')} output={record.get('output_dir')}",
                flush=True,
            )
            return
        if event == "dataset_profile_ready":
            print(
                "[dataset] "
                f"orders={record.get('number_of_orders')} "
                f"drivers={record.get('number_of_unique_drivers')} "
                f"raw_pairs={record.get('raw_cartesian_pair_count_total')}",
                flush=True,
            )
            return
        total = int(record.get("total_runs") or 0)
        counter = int(record.get("run_counter") or 0)
        percent = (counter / total * 100.0) if total else 0.0
        completed_before = max(counter - 1, 0)
        eta = None
        if event in {"run_start"} and completed_before > 0 and total:
            eta = (elapsed / completed_before) * (total - completed_before)
        run_kind = "warmup" if record.get("warmup_flag") else "measured"
        if event == "run_start":
            print(
                f"[{counter}/{total} | {percent:5.1f}%] START {run_kind} "
                f"repeat={record.get('repeat_index')} "
                f"lambda={record.get('lambda_driver_score')} "
                f"solver={record.get('matching_algorithm')} "
                f"sparse={record.get('sparse_method')} "
                f"elapsed={_fmt(elapsed)} eta={_fmt(eta)}",
                flush=True,
            )
        elif event == "run_success":
            print(
                f"[{counter}/{total}] DONE {record.get('variant_id')} "
                f"runtime={_fmt(record.get('runtime_seconds'))} "
                f"match_rate={float(record.get('match_rate') or 0):.3f} "
                f"candidate_pairs={record.get('candidate_pair_count_after_filter_total')}",
                flush=True,
            )
        elif event == "run_skipped":
            print(
                f"[{counter}/{total}] SKIP {record.get('variant_id')} "
                f"reason={record.get('skip_reason')}",
                flush=True,
            )
        elif event == "run_failed":
            print(
                f"[{counter}/{total}] FAIL {record.get('variant_id')} "
                f"error={record.get('error_message')}",
                flush=True,
            )
        elif event == "comparison_complete":
            print(
                "[complete] "
                f"success={record.get('successful_runs')} "
                f"skipped={record.get('skipped_runs')} "
                f"failed={record.get('failed_runs')} "
                f"elapsed={_fmt(elapsed)} "
                f"output={record.get('output_dir')}",
                flush=True,
            )

    result = run_model_comparison(
        args.config,
        data_mode_override=args.data_mode,
        progress_callback=progress,
    )
    print(f"Model comparison completed: {result.run_id}")
    print(f"Outputs written to {result.output_dir}")
    print(result.summary[["lambda_driver_score", "matching_algorithm", "sparse_method", "successful_run_count", "skipped_run_count", "failed_run_count"]].to_string(index=False))
    return 0


def command_compare_models_days(args: argparse.Namespace) -> int:
    if args.dry_run:
        plan = run_model_comparison_days(
            args.config,
            data_mode_override=args.data_mode,
            start_date=args.start_date,
            end_date=args.end_date,
            max_days=args.max_days,
            dry_run_only=True,
        )
        print("All-days model comparison dry run")
        print(f"Data mode: {plan['data_mode']}")
        print(f"Dates ({plan['date_count']}): {', '.join(plan['dates'])}")
        print(f"Variants: {plan['variant_count']}")
        print(f"Runs per variant-day: {plan['runs_per_variant_day']}")
        print(f"Measured runs: {plan['measured_runs']}")
        print(f"Warm-up runs: {plan['warmup_runs']}")
        print(f"Total runs: {plan['total_runs']}")
        print(f"Lambdas: {', '.join(str(value) for value in plan['lambda_driver_score_values'])}")
        print(f"Algorithms: {', '.join(plan['matching_algorithms'])}")
        print(f"Sparse methods: {', '.join(plan['sparse_methods'])}")
        return 0

    plan = run_model_comparison_days(
        args.config,
        data_mode_override=args.data_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        max_days=args.max_days,
        dry_run_only=True,
    )
    print("All-days model comparison starting")
    print(f"Data mode: {plan['data_mode']}")
    print(f"Dates ({plan['date_count']}): {', '.join(plan['dates'])}")
    print(f"Variants: {plan['variant_count']}")
    print(f"Runs per variant-day: {plan['runs_per_variant_day']}")
    print(f"Total runs: {plan['total_runs']}")
    print(f"Grid carry-over: {not args.no_grid_carryover}")
    print(f"Candidate edge logging: {args.enable_candidate_edge_logging}")
    print("")

    start_time = time.perf_counter()

    def progress(record: dict) -> None:
        event = record.get("event")
        elapsed = time.perf_counter() - start_time
        if event == "multiday_comparison_setup_start":
            print(
                "[setup] "
                f"run_id={record.get('run_id')} "
                f"data_mode={record.get('data_mode')} "
                f"variants={record.get('variant_count')} "
                f"repeats={record.get('repeats')} "
                f"warmups={record.get('warmup_runs')} "
                f"output={record.get('output_dir')} "
                f"checkpoint={record.get('checkpoint_state_path')}",
                flush=True,
            )
            return
        if event == "multiday_comparison_plan_ready":
            print(
                "[plan] "
                f"dates={record.get('date_count')} "
                f"variants={record.get('variant_count')} "
                f"total_runs={record.get('total_runs')}",
                flush=True,
            )
            return
        if event == "day_profile_ready":
            print(
                "[dataset] "
                f"date={record.get('date')} "
                f"orders={record.get('number_of_orders')} "
                f"drivers={record.get('number_of_unique_drivers')} "
                f"raw_pairs={record.get('raw_cartesian_pair_count_total')}",
                flush=True,
            )
            return
        total = int(record.get("total_runs") or 0)
        counter = int(record.get("run_counter") or 0)
        percent = (counter / total * 100.0) if total else 0.0
        run_kind = "warmup" if record.get("warmup_flag") else "measured"
        if event == "multiday_run_start":
            print(
                f"[{counter}/{total} | {percent:5.1f}%] START {run_kind} "
                f"date={record.get('simulation_date')} "
                f"repeat={record.get('repeat_index')} "
                f"lambda={record.get('lambda_driver_score')} "
                f"solver={record.get('matching_algorithm')} "
                f"sparse={record.get('sparse_method')} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(record.get('eta_seconds'))}",
                flush=True,
            )
        elif event == "multiday_run_batch_progress":
            batch = int(record.get("current_batch") or 0)
            total_batches = int(record.get("total_batches") or 0)
            batch_pct = (batch / total_batches * 100.0) if total_batches else 0.0
            print(
                f"[{counter}/{total}] "
                f"date={record.get('simulation_date')} "
                f"batch={batch}/{total_batches} {batch_pct:5.1f}% "
                f"matched={record.get('matches_created_total')} "
                f"active={record.get('active_orders')} "
                f"candidates={record.get('candidate_edges')}",
                flush=True,
            )
        elif event == "multiday_run_success":
            print(
                f"[{counter}/{total}] DONE "
                f"date={record.get('simulation_date')} "
                f"{record.get('variant_id')} "
                f"runtime={_format_duration(record.get('runtime_seconds'))} "
                f"match_rate={float(record.get('match_rate') or 0):.3f} "
                f"candidate_pairs={record.get('candidate_pair_count_after_filter_total')}",
                flush=True,
            )
        elif event == "multiday_run_skipped":
            print(
                f"[{counter}/{total}] SKIP "
                f"date={record.get('simulation_date')} "
                f"{record.get('variant_id')} "
                f"reason={record.get('skip_reason')}",
                flush=True,
            )
        elif event == "multiday_run_failed":
            print(
                f"[{counter}/{total}] FAIL "
                f"date={record.get('simulation_date')} "
                f"{record.get('variant_id')} "
                f"error={record.get('error_message')}",
                flush=True,
            )
        elif event == "multiday_comparison_complete":
            print(
                "[complete] "
                f"success={record.get('successful_runs')} "
                f"skipped={record.get('skipped_runs')} "
                f"failed={record.get('failed_runs')} "
                f"elapsed={_format_duration(record.get('elapsed_seconds'))} "
                f"output={record.get('output_dir')}",
                flush=True,
            )

    result = run_model_comparison_days(
        args.config,
        data_mode_override=args.data_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        max_days=args.max_days,
        progress_callback=progress,
        grid_carryover=not args.no_grid_carryover,
        enable_candidate_edge_logging=args.enable_candidate_edge_logging,
        progress_interval_batches=args.progress_interval_batches,
    )
    print(f"All-days model comparison completed: {result.run_id}")
    print(f"Outputs written to {result.output_dir}")
    print(f"Report: {result.output_dir / 'comparison_multiday_report.html'}")
    print(
        result.summary_across_days[
            [
                "lambda_driver_score",
                "matching_algorithm",
                "sparse_method",
                "successful_run_count",
                "skipped_run_count",
                "failed_run_count",
                "match_rate_mean",
                "spearman_driver_score_income_mean",
                "income_gini_mean",
            ]
        ].to_string(index=False)
    )
    return 0


def command_package(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    zip_path = export_zip(root, root.parent / f"{root.name}.zip")
    print(f"Project archive written to {zip_path}")
    return 0


def command_paper_reports(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    output_dir = build_paper_report_folder(root)
    print(f"Paper-style report folder: {output_dir}")
    print(f"Index: {output_dir / 'index.html'}")
    print(f"Actual report: {output_dir / 'actual_model_comparison_paper.html'}")
    return 0


def command_batch_match_report(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    output_dir = build_batch_match_report(root, data_mode=args.data_mode)
    print(f"Batch match graph folder: {output_dir}")
    print(f"HTML report: {output_dir / 'index.html'}")
    print(f"Scenario data: {output_dir / 'scenario_total_matches_by_batch.csv'}")
    print(f"All-days demand data: {output_dir / 'all_days_orders_by_batch.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ride-hailing dispatch simulator for driver-income fairness experiments."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate configured input datasets.")
    validate.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    validate.set_defaults(func=command_validate)

    simulate = subparsers.add_parser("simulate", help="Run one simulation scenario.")
    simulate.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    simulate.add_argument("--scenario-id", default=None, help="Optional output scenario id.")
    simulate.set_defaults(func=command_simulate)

    compare = subparsers.add_parser("compare", help="Run scenario comparison matrix.")
    compare.add_argument(
        "--config", default="configs/experiment_matrix.yaml", help="Path to matrix YAML config."
    )
    compare.set_defaults(func=command_compare)

    simulate_days = subparsers.add_parser(
        "simulate-days",
        help="Run actual data one date at a time, resetting daily state and carrying grid values forward.",
    )
    simulate_days.add_argument(
        "--config", default="configs/default_thesis_actual.yaml", help="Path to actual-data YAML config."
    )
    simulate_days.add_argument(
        "--output-directory",
        default="outputs/thesis_actual_multiday",
        help="Output folder for date-by-date runs.",
    )
    simulate_days.add_argument("--start-date", default=None, help="Optional first jakarta_data_date, YYYY-MM-DD.")
    simulate_days.add_argument("--end-date", default=None, help="Optional last jakarta_data_date, YYYY-MM-DD.")
    simulate_days.add_argument("--max-days", type=int, default=None, help="Limit number of dates for smoke tests.")
    simulate_days.add_argument("--scenario-prefix", default="thesis_actual", help="Prefix for scenario output folders.")
    simulate_days.add_argument(
        "--progress-interval-batches",
        type=int,
        default=60,
        help="Print one in-day progress line every N batches; use 0 for only final batch per day.",
    )
    simulate_days.add_argument(
        "--enable-candidate-edge-logging",
        action="store_true",
        help="Write candidate_edges.csv for each day; this can create very large files.",
    )
    simulate_days.set_defaults(func=command_simulate_days)

    compare_models = subparsers.add_parser("compare-models", help="Run model-variant comparison matrix.")
    compare_models.add_argument(
        "--config", default="configs/model_comparison.yaml", help="Path to model-comparison YAML config."
    )
    compare_models.add_argument("--data-mode", choices=["actual"], default=None)
    compare_models.add_argument("--dry-run", action="store_true", help="Validate and print the comparison plan only.")
    compare_models.set_defaults(func=command_compare_models)

    compare_models_days = subparsers.add_parser(
        "compare-models-days",
        help="Run every model-comparison scenario for every actual jakarta_data_date.",
    )
    compare_models_days.add_argument(
        "--config", default="configs/model_comparison_thesis_actual.yaml", help="Path to model-comparison YAML config."
    )
    compare_models_days.add_argument("--data-mode", choices=["actual"], default="actual")
    compare_models_days.add_argument("--start-date", default=None, help="Optional first jakarta_data_date, YYYY-MM-DD.")
    compare_models_days.add_argument("--end-date", default=None, help="Optional last jakarta_data_date, YYYY-MM-DD.")
    compare_models_days.add_argument("--max-days", type=int, default=None, help="Limit number of dates for smoke tests.")
    compare_models_days.add_argument("--dry-run", action="store_true", help="Print the all-days plan without running.")
    compare_models_days.add_argument(
        "--no-grid-carryover",
        action="store_true",
        help="Disable per-scenario grid value carry-over between dates.",
    )
    compare_models_days.add_argument(
        "--enable-candidate-edge-logging",
        action="store_true",
        help="Keep candidate edge logs in memory for each run; this is expensive and normally unnecessary.",
    )
    compare_models_days.add_argument(
        "--progress-interval-batches",
        type=int,
        default=0,
        help="Print in-run batch progress every N batches; 0 prints only run start/done lines.",
    )
    compare_models_days.set_defaults(func=command_compare_models_days)

    paper_reports = subparsers.add_parser(
        "paper-reports", help="Build paper-style HTML reports for the latest actual model comparison."
    )
    paper_reports.set_defaults(func=command_paper_reports)

    batch_match_report = subparsers.add_parser(
        "batch-match-report",
        help="Build an HTML graph comparing scenario total matches by batch.",
    )
    batch_match_report.add_argument("--data-mode", choices=["actual"], default="actual")
    batch_match_report.set_defaults(func=command_batch_match_report)

    package = subparsers.add_parser("package", help="Create a ZIP archive of the full project.")
    package.set_defaults(func=command_package)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except Exception as exc:
        logging.exception("Command failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
