# Data Dictionary

The tables below describe the generated output columns. All timestamps are
timezone-aware ISO strings in `Asia/Jakarta`.

## POI Reference

| dataset | column | data type | unit | nullable | description | generation rule |
| --- | --- | --- | --- | --- | --- | --- |
| poi_reference | poi_id | string | none | no | Stable POI identifier | Category and sequence |
| poi_reference | poi_name | string | none | no | POI-like name | Public reference cluster plus sequence |
| poi_reference | poi_category | string | none | no | POI class | Configured category |
| poi_reference | latitude | float | degrees | no | POI latitude | Jittered cluster coordinate |
| poi_reference | longitude | float | degrees | no | POI longitude | Jittered cluster coordinate |
| poi_reference | h3_index | string | H3 cell | no | POI H3 cell | H3 mapping |
| poi_reference | road_node_id | string | none | no | Nearest road node | Nearest graph node |
| poi_reference | source_type | string | none | no | OSM or fallback source | Network source |

## Driver Scores

| dataset | column | data type | unit | nullable | description | generation rule |
| --- | --- | --- | --- | --- | --- | --- |
| driver_scores | driver_id | string | none | no | Stable driver identifier | Sequential generated ID |
| driver_scores | history_start_date | date | date | no | History window start | Seven days before simulation |
| driver_scores | history_end_date | date | date | no | History window end | Day before simulation |
| driver_scores | total_offered_orders | integer | orders | no | Offered count | Segment-specific distribution |
| driver_scores | accepted_orders | integer | orders | no | Accepted count | Offered times AR |
| driver_scores | completed_orders | integer | orders | no | Completed count | Accepted times CR |
| driver_scores | online_duration_hours | float | hours | no | Historical online hours | Segment-specific distribution |
| driver_scores | acceptance_rate | float | 0..1 | no | Accepted / offered | Calculated |
| driver_scores | completion_rate | float | 0..1 | no | Completed / accepted | Calculated |
| driver_scores | online_duration_score | float | 0..1 | no | min(hours, 40) / 40 | Calculated |
| driver_scores | driver_behavior_score | float | 0..1 | no | Mean of AR, CR, online score | Calculated |
| driver_scores | score_segment | string | none | no | high, medium, low | Target score band |

## Orders

| dataset | column | data type | unit | nullable | description | generation rule |
| --- | --- | --- | --- | --- | --- | --- |
| orders | order_id | string | none | no | Stable order ID | Date and sequence |
| orders | customer_id | string | none | no | Customer ID | Reused generated pool |
| orders | simulation_date | date | date | no | Simulation date | Partition value |
| orders | request_timestamp | timestamp | local time | no | Request time | Batch to timestamp |
| orders | batch_index | integer | minute batch | no | One-based batch | 1..1440 |
| orders | pickup_poi_id | string | none | no | Pickup POI | OD category sample |
| orders | pickup_poi_category | string | none | no | Pickup category | OD category sample |
| orders | pickup_latitude | float | degrees | no | Pickup latitude | POI plus jitter |
| orders | pickup_longitude | float | degrees | no | Pickup longitude | POI plus jitter |
| orders | pickup_h3_index | string | H3 cell | no | Pickup H3 | H3 mapping |
| orders | dropoff_poi_id | string | none | no | Dropoff POI | OD category sample |
| orders | dropoff_poi_category | string | none | no | Dropoff category | OD category sample |
| orders | dropoff_latitude | float | degrees | no | Dropoff latitude | POI plus jitter |
| orders | dropoff_longitude | float | degrees | no | Dropoff longitude | POI plus jitter |
| orders | dropoff_h3_index | string | H3 cell | no | Dropoff H3 | H3 mapping |
| orders | straight_line_distance_km | float | km | no | Haversine distance | Calculated |
| orders | route_distance_km | float | km | no | Estimated route distance | Distance times network factor |
| orders | estimated_trip_duration_min | float | minutes | no | Trip duration | Distance and speed factors |
| orders | estimated_pickup_eta_min | float | minutes | no | Pickup ETA | Synthetic supply-demand logic |
| orders | base_fare | float | Rp | no | Minimum fare threshold retained for compatibility | Config fare rule |
| orders | fare_distance_multiplier | float | multiplier | no | Route-distance assumption for fare | Default 1.4 |
| orders | fare_distance_km | float | km | no | Fareable distance | 1.4 times straight-line distance |
| orders | fare_rate_per_km | float | Rp/km | no | Per-kilometer fare rate | Uniform sample from Rp1,850--Rp2,300 |
| orders | distance_fare | float | Rp | no | Distance fare before minimum-fare floor | fare_distance_km times fare_rate_per_km |
| orders | time_fare | float | Rp | no | Time fare retained for compatibility | Zero in default distance-based rule |
| orders | surge_multiplier | float | multiplier | no | Surge column retained for compatibility | One in default distance-based rule |
| orders | weather_multiplier | float | multiplier | no | Weather fare column retained for compatibility | One in default distance-based rule |
| orders | supply_demand_ratio | float | ratio | no | Supply/demand ratio | Demand intensity rule |
| orders | fare_amount | float | Rp | no | Final fare | max(10000, distance_fare) |
| orders | traffic_level | string | none | no | light, moderate, heavy | Time period |
| orders | weather_condition | string | none | no | Weather state | Daily sample |
| orders | day_type | string | none | no | weekday/weekend | Calendar date |
| orders | time_period | string | none | no | Demand period | Batch mapping |

## Driver Positions

| dataset | column | data type | unit | nullable | description | generation rule |
| --- | --- | --- | --- | --- | --- | --- |
| driver_positions | driver_id | string | none | no | Driver ID | Driver score table |
| driver_positions | simulation_date | date | date | no | Simulation date | Partition value |
| driver_positions | timestamp | timestamp | local time | no | Position timestamp | Batch to timestamp |
| driver_positions | batch_index | integer | minute batch | no | One-based batch | 1..1440 |
| driver_positions | online_session_id | string | none | no | Session ID | Driver-date-session |
| driver_positions | online_flag | boolean | none | no | Online indicator | Always true for stored rows |
| driver_positions | driver_status | string | none | no | Movement status | Waiting or repositioning |
| driver_positions | latitude | float | degrees | no | Driver latitude | Road path interpolation |
| driver_positions | longitude | float | degrees | no | Driver longitude | Road path interpolation |
| driver_positions | h3_index | string | H3 cell | no | Driver H3 | H3 mapping |
| driver_positions | road_node_id | string | none | no | Road node | Interpolated path node |
| driver_positions | speed_kph | float | km/h | no | Minute movement speed | Distance from previous point |
| driver_positions | heading_degrees | float | degrees | no | Movement heading | Bearing from previous point |
| driver_positions | minutes_since_session_start | integer | minutes | no | Session elapsed time | Counter |

## H3 Grid Reference and Values

| dataset | column | data type | unit | nullable | description | generation rule |
| --- | --- | --- | --- | --- | --- | --- |
| h3_grid_reference | location_id | string | none | no | Cell location ID | Sequential generated ID |
| h3_grid_reference | h3_index | string | H3 cell | no | H3 cell | Observed POI/order/driver cells |
| h3_grid_reference | h3_resolution | integer | H3 resolution | no | H3 resolution | Config value |
| h3_grid_reference | centroid_latitude | float | degrees | no | Cell centroid latitude | H3 centroid |
| h3_grid_reference | centroid_longitude | float | degrees | no | Cell centroid longitude | H3 centroid |
| h3_grid_values | simulation_date | date | date | no | Simulation date | Partition value |
| h3_grid_values | batch_index | integer | minute batch | no | Batch | 1..1440 |
| h3_grid_values | batch_timestamp | timestamp | local time | no | Batch timestamp | Batch to timestamp |
| h3_grid_values | location_id | string | none | no | Location ID | Grid reference |
| h3_grid_values | h3_index | string | H3 cell | no | H3 cell | Grid reference |
| h3_grid_values | h3_resolution | integer | H3 resolution | no | H3 resolution | Config value |
| h3_grid_values | centroid_latitude | float | degrees | no | Cell centroid latitude | H3 centroid |
| h3_grid_values | centroid_longitude | float | degrees | no | Cell centroid longitude | H3 centroid |
| h3_grid_values | grid_value | float | generated value | no | Persistent grid state | Demand/supply update |
| h3_grid_values | demand_intensity | float | normalized | no | Demand intensity | Active orders / max demand |
| h3_grid_values | available_driver_count | integer | drivers | no | Online drivers in cell | Aggregated positions |
| h3_grid_values | active_order_count | integer | orders | no | Orders in cell | Aggregated pickup cells |
