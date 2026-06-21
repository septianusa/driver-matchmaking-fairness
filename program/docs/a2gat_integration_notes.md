# A2GAT Integration Notes

Reference repository: <https://github.com/nldmz/A2GAT>

The repository describes **A2GAT: Adaptive Anchor-based Graph Attention Networks**
as an implementation for large-scale sparse bipartite graph embedding. Its public
README lists these top-level files and folders:

- `dataset/`
- `models/`
- `train.py`
- `train_lp.py`
- `model.py`
- `utils.py`
- `calculate.py`
- `dataloader.py`
- `evaluation.py`

The README lists requirements:

- Python `>=3.8`
- PyTorch `>=1.8.0`
- NumPy `>=1.19.2`
- scikit-learn `>=0.24.2`
- SciPy `>=1.6.2`

The expected repository data format is graph-oriented rather than ride-hailing
dispatch-oriented:

- `train.csr.pickle`
- `test.csr.pickle`
- `lp.train.npz`
- `lp.test.npz`

## Can It Generate Driver-Order Candidate Edges?

Yes, through the built-in simulator adapter in `src/sparse/a2gat_adapter.py`.
The adapter implements the sparse data-handler role expected from A2GAT without
requiring external PyTorch weights:

1. Build adaptive anchors from current driver H3 cells.
2. Score nearby anchors around each order origin.
3. Collect drivers from the highest-scoring anchors.
4. Apply the authoritative pickup-distance feasibility filter.
5. Rank candidate drivers with an attention-like score based on distance and
   driver behavior score.
6. Return only the top sparse candidates to the matching solver.

A2GAT can plausibly be used as a candidate-pruning or link-prediction model if
ride-hailing dispatch is converted into a bipartite graph:

- One node type for orders.
- One node type for drivers.
- Edges representing feasible or historically accepted/matched driver-order
  relationships.
- Edge labels or scores representing match relevance, completion likelihood, or
  candidate priority.

The public repository still does not provide a ready-made ride-hailing trained
model or feature contract. For that reason, this project labels its method as a
built-in A2GAT-style adaptive anchor provider rather than claiming pretrained
external A2GAT inference.

## Required Graph Structure

A future adapter would need to construct a sparse bipartite graph with:

- Driver nodes.
- Order nodes.
- Sparse feasible candidate edges.
- Optional edge features such as pickup distance, ETA, driver score, origin H3,
  destination H3, grid value, and time-of-day.
- Optional node features such as driver score, current H3 cell, order fare,
  origin/destination cells, trip distance, and batch id.

## Model-Weight Requirements

The built-in adapter does not require model weights. If a future trained
ride-hailing A2GAT model becomes available, the same provider boundary can be
extended to replace the deterministic anchor scoring with PyTorch inference.

## CPU/GPU Requirements

The repository supports GPU selection in its example commands. A CPU-only
execution path may be possible through PyTorch, but large bipartite graph
training or inference is expected to be materially faster on GPU.

The current thesis simulator remains runnable without PyTorch, GPU drivers, or
the A2GAT repository.

## Required Features For This Simulator

A usable A2GAT adapter should output:

```text
order_id
driver_id
pickup_distance_km
retrieval_rank
retrieval_score
sparse_method
```

It must also preserve the simulator's final haversine feasibility filter. A2GAT
can only prune or rank candidates; it must not override the authoritative pickup
distance limit.

## Current Adapter Behavior

The current adapter is active by default when a comparison variant uses:

```yaml
sparse:
  method: a2gat
```

The model-comparison config exposes these parameters:

```yaml
sparse:
  a2gat:
    anchor_hops: 4
    anchor_top_k: 8
    candidate_driver_target: 30
    distance_weight: 1.0
    driver_score_weight: 0.15
```

This keeps the experiment reproducible while making A2GAT a real sparse handler
inside the simulator.
