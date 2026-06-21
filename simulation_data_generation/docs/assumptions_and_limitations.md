# Assumptions and Limitations

- The dataset is generated for dispatch simulation and benchmarking.
- Spatial demand is POI-driven and should not be used for city-level demand
  forecasting.
- The offline road graph is a connected approximation for route-shaped driver
  movement.
- Fare, traffic, weather, and driver behavior are simulated variables.
- Fare uses a distance-based rule with fareable distance set to 1.4 times the
  straight-line distance, a Rp1,850--Rp2,300 per-km rate band, and a Rp10,000
  minimum fare.
- Driver behavior scores are generated from internally consistent activity
  counts, not from observed driver outcomes.
- H3 grid values are state features with temporal persistence.
- The generator is intended for experimental validation and simulator
  development, not operational forecasting or deployment claims.
