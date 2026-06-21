# Validation

Validation is implemented in `src/simulation_data_generation/validation.py`.
It reads the generated Parquet datasets and writes:

```text
reports/validation_summary.md
reports/figures/*.png
```

## General Checks

- Expected simulation dates.
- Batch indices in `1..1440`.
- Required columns.
- Unique order IDs.
- Timezone-aware timestamp format.
- H3 consistency between orders, drivers, and the grid reference.

## Order Checks

- Default seven-day runs exceed the configured minimum daily order count.
- Pickup and dropoff coordinates are inside the configured study boundary.
- Pickup and dropoff coordinates are not identical.
- Route distance is not below straight-line distance.
- Fare, ETA, and duration are positive.

## Driver Checks

- Position records reference known driver IDs.
- No driver has more than 480 online one-minute rows per day.
- Position batches are consecutive inside an online session.
- Speeds do not exceed the configured maximum.

## Driver Score Checks

- Completed orders do not exceed accepted orders.
- Accepted orders do not exceed offered orders.
- Acceptance and completion rates match their count columns.
- Score-segment proportions are compared with configured targets.

## Grid Checks

- All order and driver H3 cells exist in the grid reference.
- Grid values are finite.
- Grid-value rows are counted and summarized.

## Figures

The validation step creates diagnostic figures for orders by minute/day, pickup
locations, trip distance, fare, active drivers, online hours, score
distribution, score segment proportions, and supply-demand ratio by time.

