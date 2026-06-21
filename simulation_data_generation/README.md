# Surabaya Dispatch Simulation Data Generator

This project generates ride-hailing dispatch simulation data for assignment
experiments in Surabaya, Indonesia. It creates interconnected orders, driver
positions, driver behavior scores, POI references, and H3 grid values over
one-minute dispatch batches.

The dataset is designed as a public research fixture: large enough to exercise
dispatch algorithms, small enough to run on a workstation, and structured
closely enough to the simulator schema to support reproducible experiments.

## Dataset Overview

The default configuration uses:

- seven consecutive simulation dates starting from `2026-06-01`;
- one-based one-minute batches, `1..1440`, for each day;
- approximately 9,200 to 13,100 order requests per day;
- approximately 78,000 to 80,000 order requests across the full week;
- approximately 2,875 registered drivers;
- approximately 2,050 to 2,350 active drivers per day in the full scale;
- at most 480 online one-minute position records per driver per day;
- fareable distance equal to `1.4 * straight_line_distance_km`;
- fare rate sampled from Rp1,850 to Rp2,300 per km, with Rp10,000 minimum fare;
- H3 resolution `8` for urban dispatch cells;
- timezone-aware timestamps in `Asia/Jakarta`.

The small scale is intended for tests and Git demonstrations. It generates two
days, 120 drivers, and a few hundred orders per day.

## Project Structure

```text
simulation_data_generation/
|-- config/
|-- docs/
|-- src/simulation_data_generation/
|-- scripts/
|-- tests/
|-- data/
|   |-- raw_reference/
|   |-- sample/
|   `-- generated/
`-- reports/
```

Generated Parquet data are written to `data/generated/`. Small CSV previews are
written to `data/sample/`. Validation figures and the Markdown validation report
are written to `reports/`.

## Installation

```bash
cd C:\Users\62812\Documents\Codex\experiment_dispatch_matching\simulation_data_generation
python -m pip install -r requirements.txt
```

For editable local development:

```bash
python -m pip install -e ".[test]"
```

The generator does not require paid APIs. OSMnx/geopandas are optional. When
OpenStreetMap downloads are not available, the generator uses a documented
offline road graph approximation inside the Surabaya boundary.

## Generate Data

Small test dataset:

```bash
python scripts/generate_sample.py
```

Equivalent explicit command:

```bash
python -m simulation_data_generation.cli generate --config config/default.yaml --scale small --overwrite
```

Default full-scale dataset:

```bash
python -m simulation_data_generation.cli generate --config config/default.yaml --scale full --overwrite
```

The full run can produce millions of driver-position rows. Keep full generated
data out of Git.

## Validate Data

```bash
python -m simulation_data_generation.cli validate --config config/default.yaml --scale small --data-dir data/generated --report-dir reports
```

The generator also runs validation automatically unless `--no-validate` is
provided.

## Create Spatial Figure

```bash
python scripts/create_dispatch_map_2x2.py
```

This creates `reports/figures/dispatch_spatial_2x2_surabaya.png` and `.pdf`
with pickup points, dropoff points, H3 grid overlays, and two sample driver
sessions on a Surabaya basemap.

## Run Tests

```bash
python -m pytest
```

The test suite uses a local pytest temp directory so it does not depend on the
system temp folder.

## Output Files

Minimum generated datasets:

```text
data/generated/poi_reference.parquet
data/generated/driver_scores.parquet
data/generated/orders/simulation_date=YYYY-MM-DD/part-000.parquet
data/generated/driver_positions/simulation_date=YYYY-MM-DD/part-000.parquet
data/generated/h3_grid_reference.parquet
data/generated/h3_grid_values/simulation_date=YYYY-MM-DD/part-000.parquet
data/generated/generation_manifest.json
```

Small CSV samples:

```text
data/sample/poi_reference_sample.csv
data/sample/driver_scores_sample.csv
data/sample/orders_sample.csv
data/sample/driver_positions_sample.csv
data/sample/h3_grid_values_sample.csv
```

## Reproducibility

The default seed is `20260620`. Generation uses deterministic seed offsets by
module and day. Re-running the same configuration and scale should produce the
same score and sample digests in `generation_manifest.json`.

## Computational Expectations

The small scale usually completes in under two minutes on a laptop. The default
full scale may create roughly 78,000 to 80,000 orders, several million
driver-position rows from roughly 2,050 to 2,350 active drivers per day, and a
large dynamic grid-value table if minute-level grid values are enabled. Parquet
partitioning by date is used to keep files manageable.

## Git Guidance

Commit source code, config, docs, tests, and small CSV samples only when
appropriate. Do not commit large Parquet outputs under `data/generated/`; this
folder is ignored except for `.gitkeep`.

## Known Limitations

- Spatial patterns are POI-driven and intended for simulation, not city demand
  forecasting.
- The offline road graph is an approximation for route-shaped movement.
- Fare, traffic, weather, and driver behavior are simulated variables.
- Data generated by this tool are suitable for experimental validation, not
  operational forecasting or deployment claims.
