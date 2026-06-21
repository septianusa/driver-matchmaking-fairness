# Journal Research Run Guide

This project can produce a research evidence package from actual daily data.
The command is:

```powershell
python main.py simulate-days --config configs/default_thesis_actual.yaml --output-directory outputs/thesis_actual_multiday
```

The runner treats each `jakarta_data_date` as a separate simulation day. Driver
state, active orders, and daily metrics reset for each date. The learned grid
value file is carried forward:

```text
day N grid_values_final.csv -> day N+1 input grid_values
```

By default, candidate edge logging is disabled because the file can be very
large. Enable it only when the paper needs edge-level audit evidence:

```powershell
python main.py simulate-days --config configs/default_thesis_actual.yaml --enable-candidate-edge-logging
```

## Research Package Outputs

After the run completes, the folder `research_package` is created inside the
selected output directory.

Core files:

- `journal_research_report.html`: reader-facing research evidence report.
- `simulation_metrics_by_day.csv`: daily match, conversion, utility, pickup,
  fairness, and runtime metrics.
- `data_quality_by_day.csv`: validation results, missing values, duplicate
  checks, coordinate checks, fare checks, score coverage, and leakage-prevention
  counts.
- `batch_metrics_all_days.csv`: all date-level batch metrics combined for
  temporal and peak-hour analysis.
- `hourly_metrics_by_day.csv`: hourly aggregation for commuting and recreation
  pattern analysis.
- `grid_carryover_audit.csv`: input and output grid value paths and summary
  statistics per day.
- `source_file_manifest.csv`: source paths, file sizes, row counts, and SHA-256
  hashes for reproducibility.
- `metric_definitions.csv`: definitions and interpretation guidance for the
  reported metrics.
- `journal_readiness_checklist.csv`: audit checklist for publication readiness.
- `run_manifest.json`: runtime, date, seed, platform, and configuration metadata.

## What This Supports

This evidence package supports journal tables and methods sections for:

- input data quality and leakage prevention;
- daily dispatch performance;
- expected completion behavior;
- pickup-distance efficiency;
- driver-income fairness;
- score-income relationship;
- batch/hour temporal behavior;
- grid-value learning carry-over;
- reproducibility and source provenance.

For claims that one algorithm is superior to another, also run the model
comparison matrix on actual data and report repeated-run sensitivity or
confidence intervals:

```powershell
python main.py compare-models --config configs/model_comparison_thesis_actual.yaml --data-mode actual
python main.py paper-reports
```

To run every configured model-comparison scenario on every actual
`jakarta_data_date`, use the all-days comparison command:

```powershell
python main.py compare-models-days --config configs/model_comparison_thesis_actual.yaml --data-mode actual
```

Check the full run size first:

```powershell
python main.py compare-models-days --config configs/model_comparison_thesis_actual.yaml --data-mode actual --dry-run
```

The all-days comparison carries grid values forward within each
scenario/repeat chain by default. This aligns the comparison with the multiday
simulation assumption:

```text
day N grid_values_final.csv -> same scenario/repeat day N+1 input grid_values
```

Use `--no-grid-carryover` only for an independent-per-day ablation.

The thesis actual comparison config is intentionally scoped as the core thesis
matrix:

```text
7 actual days
x lambda_driver_score 0.0, 0.1, 0.2, 0.3
x Hungarian solver
x BFS-H3 and A2GAT sparse methods
x repeats 1, warmup 0
= 56 runs
```

Checkpoint files are written after every completed run, so an interrupted run
still leaves partial evidence:

- `checkpoint_state.json`
- `comparison_raw_runs_all_days_checkpoint.csv`
- `comparison_batch_metrics_all_days_checkpoint.csv`
- `comparison_grid_carryover_audit_checkpoint.csv`
- `comparison_summary_by_day_checkpoint.csv`
- `comparison_summary_across_days_checkpoint.csv`
