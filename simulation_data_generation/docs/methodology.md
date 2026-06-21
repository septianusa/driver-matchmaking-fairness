# Methodology

## Simulation Design

The generator models a seven-day dispatch market using one-minute batches over
a 24-hour horizon. Each date has 1,440 global batches. Batch `1` corresponds to
`00:00:00` to `00:00:59` in `Asia/Jakarta`, and batch `1440` corresponds to
`23:59:00` to `23:59:59`.

The default start date is `2026-06-01`. The date is only a reproducible calendar
anchor for weekday and weekend patterns.

## Spatial Boundary and H3

The study boundary is a rectangular Surabaya envelope configured in
`config/default.yaml`. H3 resolution `8` is the default because it is small
enough for urban dispatch analysis while keeping grid tables manageable. The
exact cell area varies by latitude; in this use it is treated as a dispatch-cell
index rather than a claim about a fixed physical unit.

## POIs and Road Network

The generator can attempt OpenStreetMap acquisition when configured, but the
default path uses an offline fallback. The fallback creates POI-like clusters
around public Surabaya reference areas and snaps each POI to a connected
road-graph approximation. The road graph contains local-road grid links and
arterial-style diagonal connectors.

This fallback makes the package usable without network access. Its purpose is
spatial plausibility, not road-network fidelity.

## Temporal Demand

Minute-level demand uses a smooth non-homogeneous Poisson process. Period-level
weights define graveyard, morning peak, daytime off-peak, evening peak, evening
off-peak, and late evening behavior. Weekday demand emphasizes worker and
student commuting peaks. Weekend demand shifts toward shopping, leisure, and
food-area travel.

Daily totals are sampled with controlled variation around the configured target
profile. For the default seven-day run, the full-scale setting targets roughly
9,200 to 13,100 orders per day and roughly 78,000 to 80,000 orders across the
week.

## Origin-Destination Logic

Orders sample origin and destination categories from configurable OD matrices by
day type and time period. Weekday morning demand favors residential origins and
office, school, university, or transport destinations. Weekday evening demand
shifts back toward residential destinations. Weekend demand increases flows
toward malls, markets, recreation, and food areas.

Pickup and dropoff points are jittered around POIs and clipped to the study
boundary. Invalid trips outside configured distance bounds are resampled.

## Distance, ETA, and Fare

Straight-line distance is computed with the haversine formula. Route distance is
estimated by multiplying straight-line distance by a configurable network
factor. The default route-distance factor is `1.4`, representing the assumption
that road travel is longer than the direct pickup--dropoff line. Duration
depends on route distance, time-period speed assumptions, and weather speed
factors.

Fare uses a transparent distance-based rule. The fareable distance is defined
as:

```text
fare_distance_km = 1.4 * straight_line_distance_km
```

The per-kilometer fare is sampled between Rp1,850 and Rp2,300, following the
configured public guideline range used in this study. The final fare is then:

```text
fare = max(10000, fare_distance_km * fare_rate_per_km)
```

The minimum fare is Rp10,000. The generated order table keeps older component
columns such as `base_fare`, `time_fare`, and `surge_multiplier` for schema
compatibility, but the effective fare calculation in the default configuration
is the distance-based rule above.

## Driver Scores

The driver-score dataset is generated from historical-style offered, accepted,
completed, and online-hour values. The final score is calculated as:

```text
s_j = (AR_j + CR_j + min(H_j, 40) / 40) / 3
```

where `AR_j = accepted / offered` and `CR_j = completed / accepted`, with zero
fallbacks for zero denominators. Segment targets use bounded distributions and
rejection sampling so that the full-scale profile contains approximately 42%
high-score, 36% medium-score, and 22% low-score drivers.

## Driver Participation and Movement

Driver participation varies by day, expected demand, and behavior segment.
The full-scale setting uses 2,875 registered drivers and targets approximately
2,050 to 2,350 active drivers per day. Online sessions are one or two blocks per
day, and total online duration never exceeds eight hours.
Movement is generated along the road graph, with occasional waiting near demand
areas and repositioning toward POIs that are plausible for the current time
period.

## Grid Values

H3 grid reference cells are built from all generated POI, order, and driver
cells. Dynamic grid values are minute-level when enabled. They use temporal
persistence:

```text
V_t = 0.88 V_{t-1} + 0.12 target_t
```

The target increases with generated demand intensity and decreases mildly with
available supply. These values are state features for dispatch experiments.

## Dataset Interconnection

Driver IDs in positions are drawn from the driver-score table. Order and driver
coordinates map to H3 cells in the grid reference. POI IDs in orders map to the
POI reference. Timestamps, dates, and one-based batch indices follow a single
Asia/Jakarta convention across all datasets.
