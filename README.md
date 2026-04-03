# Driver Behaviour-Based Matchmaking to Improve Income Fairness in Ride-Hailing Platforms

This simulation implements the Learning-to-Dispatch framework developed by Dingyuan Shi, based on the work presented in “Combinatorial Optimization Meets Reinforcement Learning: Effective Taxi Order Dispatching at Large-Scale” (TKDE 2022). The original source code is available at: https://github.com/dingyuan-shi/Learning-To-Dispatch
## Abstract

This repository presents a simulation framework for evaluating driver behaviour-based matchmaking strategies in ride-hailing systems. The study focuses on improving income fairness among drivers while maintaining operational efficiency, such as high match rates and low pickup distances. The framework models spatial-temporal dynamics using synthetic demand and supply data, constructs candidate matches via graph-based search, and solves assignment using weighted bipartite matching.

---

## 1. Introduction

Ride-hailing platforms rely on efficient matchmaking between drivers and customers. While current systems are optimized for operational metrics (e.g., conversion rate, pickup time), they often overlook fairness in income distribution among drivers.

This project investigates:
- Integration of driver behavior into matchmaking
- Trade-offs between efficiency and fairness
- Impact on income distribution

---

## 2. Methodology Overview

### Simulation Framework
Minute-level simulation (1440 batches per day) with:
- Dynamic orders
- Driver movement
- Carry-over orders
- Batch-wise matching

### Candidate Retrieval
- H3-like spatial grid
- BFS-based neighbor search
- Distance constraints

### Matching
- Bipartite graph (driver-order)
- Hungarian algorithm optimization

### Weight Function
Combination of:
- Utility base
- Cancellation probability
- Driver behavior score

### Reinforcement Learning
Grid value updates to guide repositioning.

---

## 3. Evaluation Metrics

### Efficiency
- Match rate
- Pickup distance
- Utility

### Fairness
- Gini index
- Income distribution (P10, P50, P90)
- Score-income correlation

---

## 4. Repository Structure

thesis/
├── src/
├── scripts/
├── data/
├── docs/
├── results/
├── requirements.txt
└── README.md

---

## 5. Setup (macOS)

cd thesis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## 6. Data

Raw datasets are NOT included.

Generate synthetic data:
python3 scripts/generate_dummy_data.py

Or place your own data in:
data/raw/

---

## 7. Run Simulation

python3 -m src.main

Outputs saved in:
data/output/

---

## 8. Experiment Design

Scenarios:
- Base (10k demand, 3k drivers)
- High demand
- Low supply

Lambda variations:
- 0.0 (efficiency)
- 0.3 (balanced)
- 0.6 (fairness)

---

## 9. Key Insight

Behavior-aware matchmaking improves fairness while maintaining efficiency when properly tuned.

---

## 10. Reproducibility

python3 scripts/generate_dummy_data.py
python3 -m src.main

---

## 11. Limitations

- Synthetic data
- Approximate models
- Simulated driver movement

---

## 12. Future Work

- Real-world data integration
- RL improvements
- Multi-objective optimization

---

## 13. License

MIT License

---

## 14. Author

Septia Nusa
