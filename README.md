# Supply Chain Fraud & Disruption Detection using Graph Attention Networks

An explainable, graph-based anomaly detection system for supply chains. Instead of treating
transactions as independent rows in a table (the standard tabular ML approach), this project
models the **entire supply chain as a graph** — suppliers, manufacturers, warehouses, and
retailers as nodes, and shipments/transactions as edges — and trains a **custom Graph Attention
Network (GAT)**, built from scratch in PyTorch, to flag fraudulent or anomalous transaction lanes.

## Why this project is different

Most undergraduate "anomaly detection" projects run Isolation Forest or a simple autoencoder
on a flat CSV. That approach throws away the single most important signal in supply chain data:
**who is transacting with whom**. A transaction that looks perfectly normal in isolation (a
plausible quantity, a plausible cost) can still be part of a fraud ring if you look at the
surrounding network structure — recurring loops between the same small set of entities,
transactions that skip tiers, or a shipment that shouldn't geographically exist.

This project treats the graph structure as first-class information and uses **attention weights
as an explainability mechanism** — so for every flagged transaction, you can show *why* the model
was suspicious, not just a bare "anomaly score."

## Project structure

```
scgnn/
├── data/
│   └── generate_graph.py        # Synthetic multi-tier supply chain + injected anomalies
├── models/
│   ├── gat.py                   # Custom Graph Attention Network (pure PyTorch)
│   └── train.py                 # Training pipeline, evaluation, artifact export
├── dashboard/
│   ├── app.py                   # Interactive Streamlit dashboard
│   └── export_static_visuals.py # Static PNG exports for README/portfolio
├── outputs/                     # Generated after running train.py
└── requirements.txt
```

## The data

Since real, labeled supply-chain fraud data is not publicly available (for obvious reasons),
this project **generates a synthetic but realistic supply chain network**:

- **250 entities** across 4 tiers: 60 suppliers → 40 manufacturers → 30 warehouses → 120 retailers
- **511 transaction lanes** (edges), built with region-aware, tier-respecting logic (a
  manufacturer preferentially sources from suppliers in its own region, etc.)
- Each lane has realistic features: average order quantity, average cost, lead time, delay
  variance, order frequency, and transport mode

**Four distinct anomaly patterns are then injected** (giving us ground truth for evaluation):

| Type | Pattern | Real-world analogue |
|---|---|---|
| `FRAUD_RING` | A small cluster of entities over-trading with suspiciously uniform order sizes and frequency | Collusion / kickback schemes |
| `GHOST_SHIPMENT` | Near-zero quantity paired with abnormally high cost | Invoice fraud |
| `ROUTE_HIJACK` | Edges that skip tiers or connect geographically illogical entities with implausibly fast lead times | Diverted/rerouted shipments |
| `DELAY_CASCADE` | A hub node whose entire outgoing lane set suddenly shows spiked delay variance | Disruption event propagating downstream |

Overall anomaly rate: **~8%** of all lanes — a realistic, imbalanced detection setting (not an
artificially balanced 50/50 toy problem).

## The model

`models/gat.py` implements a two-layer, multi-head **Graph Attention Network from scratch**
(no `torch_geometric` dependency — this was a deliberate choice so the mechanics are fully
visible and explainable in an interview, and so the project has zero fragile compiled-binary
dependencies):

1. **Layer 1**: 4-head GAT layer, node features → 32-dim hidden representation per head, concatenated
2. **Layer 2**: single-head GAT layer, produces final 32-dim node embeddings
3. **Edge scoring head**: an MLP that takes `[embedding(source) || embedding(destination) ||
   raw edge features]` and outputs an anomaly probability for that specific transaction

This is the key design decision: **the model classifies edges, not nodes** — matching the real
problem (a transaction is suspicious, not necessarily a company). Node embeddings capture
*relational* context (who this entity usually deals with, what tier/region it sits in);
raw edge features capture the transaction's own numeric signature. Combining both lets the model
catch anomalies that look fine in isolation but are wrong in context, and vice versa.

Class imbalance (~8% positive rate) is handled with a weighted BCE loss (`pos_weight` scaled to
the inverse class ratio) rather than naive oversampling.

### Explainability

GAT's attention mechanism naturally produces a weight for every neighbor relationship during
message passing. We aggregate these into an **"attention received"** score per node — a proxy
for how much the model's learned structure singles an entity out — and surface it directly in
the dashboard (larger node = more attention received). This turns the model from a black box
into something a supply chain analyst could actually act on.

## Results

On a held-out test set (20% of edges, stratified):

| Metric | Score |
|---|---|
| Precision | 0.75 |
| Recall | 0.75 |
| F1 | 0.75 |
| ROC-AUC | 0.987 |
| PR-AUC | 0.891 |

Notably, **the model was never given the anomaly *type* during training** — only a binary
`is_anomaly` label. The fact that it recovers strong detection rates across all four
structurally distinct anomaly types (fraud rings, ghost shipments, route hijacks, delay
cascades) suggests it is learning real underlying graph and feature patterns, not memorizing
a narrow signature.

See `outputs/fig_anomaly_breakdown.png` for the per-type detection breakdown and
`outputs/fig_network_anomalies.png` for the full network visualization.

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Generate the synthetic supply chain graph + injected anomalies
python data/generate_graph.py

# 2. Train the GAT and export all evaluation artifacts
python models/train.py

# 3. (Optional) export static PNGs for a README/portfolio
python dashboard/export_static_visuals.py

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py
```

## Possible extensions (good "future work" talking points for interviews)

- **Temporal GNN**: model the graph as a sequence of snapshots over time (e.g. TGN or a
  GNN+LSTM hybrid) to catch anomalies as they *emerge*, rather than scoring a static graph
- **Real-world validation**: re-run the same pipeline on a real dataset (e.g. the DataCo
  Supply Chain dataset) restructured into a graph, to validate beyond synthetic data
- **Active learning loop**: since real fraud labels are expensive to obtain, wrap the model
  in an active-learning setup where the highest-uncertainty flagged edges are prioritized for
  human review
- **Multi-hop counterfactuals**: extend the explainability layer to answer "what minimal change
  to this transaction would make the model consider it normal?"

## Tech stack

PyTorch (custom GAT implementation) · NetworkX (graph construction) · scikit-learn (metrics,
preprocessing) · Streamlit + Plotly (interactive dashboard) · Matplotlib (static exports)
