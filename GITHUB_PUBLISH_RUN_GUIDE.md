# GitHub Publish Run Guide

This repository is safe to publish when it contains the simulator code, the
synthetic Surabaya data generator, this run guide, and selected PNG result
figures. Do not publish private or company raw data.

## What Is Included

- `program/`: dispatch simulator, matching solvers, sparse-data handlers,
  comparison runner, and paper-result scripts.
- `simulation_data_generation/`: public synthetic Surabaya dispatch-data
  generator.
- `results/figures/`: selected PNG figures only.
- `GITHUB_PUBLISH_RUN_GUIDE.md`: reproducible run instructions.

## Scenario Matrix

The public full experiment is:

```text
7 simulation days
x 2 grid settings: on, off
x 2 solvers: hungarian, greedy
x 2 sparse handlers: bfs_h3, a2gat
x 4 driver-score weights: 0.0, 0.1, 0.2, 0.3
= 224 measured scenario-day runs
```

The grid-on config runs 112 scenario-day runs. The grid-off config runs another
112 scenario-day runs.

## Install Dependencies

Use Python 3.10 or newer. From the repository root:

```bash
cd simulation_data_generation
python -m pip install -r requirements.txt

cd ../program
python -m pip install -r requirements.txt
```

## Generate Synthetic Surabaya Data

Small smoke-test data:

```bash
cd program
python scripts/prepare_synthetic_surabaya_data.py --scale small
```

Full seven-day data:

```bash
cd program
python scripts/prepare_synthetic_surabaya_data.py --scale full
```

This command first runs the generator under `simulation_data_generation/`, then
exports simulator-ready CSV files into:

```text
program/data/raw/orders.csv
program/data/raw/driver_locations.csv
program/data/raw/driver_scores.csv
program/data/raw/grid_values.csv
```

These generated CSV files can be large and should not be committed to Git.

## Validate Inputs

```bash
cd program
python main.py validate --config configs/default_surabaya_synthetic.yaml
```

## Check the Full Experiment Plan

Grid on:

```bash
python main.py compare-models-days --config configs/model_comparison_surabaya_grid_on.yaml --dry-run
```

Grid off:

```bash
python main.py compare-models-days --config configs/model_comparison_surabaya_grid_off.yaml --dry-run --no-grid-carryover
```

Each dry run should report 7 dates, 16 variants, and 112 measured runs.

## Run the Full 224 Scenario-Day Experiment

Grid on:

```bash
python main.py compare-models-days --config configs/model_comparison_surabaya_grid_on.yaml --progress-interval-batches 0
```

Grid off:

```bash
python main.py compare-models-days --config configs/model_comparison_surabaya_grid_off.yaml --no-grid-carryover --progress-interval-batches 0
```

The commands print progress for each scenario-day run and write outputs under
`program/outputs/`.

## Publish Notes

- Commit source code, configs, docs, and selected PNG figures.
- Do not commit `program/data/raw/*.csv`, `program/outputs/`, generated Parquet
  data, or local ZIP files.
- A2GAT in this repository is an adaptive-anchor sparse candidate handler. It is
  not a pretrained external neural model.
- The generated dataset is for reproducible simulation research, not production
  forecasting.
