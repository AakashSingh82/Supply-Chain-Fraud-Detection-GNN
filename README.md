# 🔗 Supply Chain Fraud & Disruption Detection using Graph Attention Networks

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An explainable, graph-based anomaly detection system for supply chains. Instead of treating
transactions as independent rows in a spreadsheet, this project models the **entire supply
chain as a graph** — suppliers, manufacturers, warehouses, and retailers as nodes, shipments as
edges — and trains a **Graph Attention Network (GAT), built from scratch in PyTorch**, to flag
fraudulent or anomalous transaction lanes.

**[🚀 Live Demo] https://supply-chain-fraud-detection-gnn-nxmecamzghas7yrua9kj7f.streamlit.app/)](https://supply-chain-fraud-detection-gnn-24p2kqt3bmodytbdkjnf5z.streamlit.app/**

---

## 🧠 Why this project is different

Most anomaly detection projects run Isolation Forest or an autoencoder on a flat CSV — treating
every transaction as an independent row. That throws away the single most important signal in
supply chain data: **who is transacting with whom**.

A transaction can look perfectly normal in isolation (a plausible quantity, a plausible cost)
and still be part of a fraud ring — the giveaway is the *pattern of relationships* around it:
a tight loop of the same few companies trading repeatedly, a shipment that skips logical tiers,
or a disruption spreading outward from one hub.

This project treats the graph structure as first-class information, and uses the GAT's
**attention weights as an explainability mechanism** — so every flagged transaction comes with
a "why," not just a bare anomaly score. On top of that, it includes a full **reliability layer**
(calibration, out-of-distribution detection, an independent cross-check model) so the system is
honest about when it's confident versus when it's guessing — see the section below.

---

## 📂 Project structure

```
scgnn/
├── data/
│   ├── generate_graph.py          # Synthetic multi-tier supply chain + injected anomalies
│   ├── nodes.csv                  # Generated: 250 entities across 4 tiers
│   ├── edges.csv                  # Generated: 511 transaction lanes
│   └── graph.gpickle              # Generated: NetworkX graph object
├── models/
│   ├── gat.py                     # Custom Graph Attention Network (pure PyTorch)
│   ├── train.py                   # Training pipeline, calibration, evaluation, artifact export
│   └── inference.py                # Production inference: validation, OOD detection, reliability scoring
├── dashboard/
│   ├── app.py                     # Interactive Streamlit dashboard
│   ├── export_static_visuals.py   # Static PNG exports for README/portfolio
│   ├── sample_nodes_template.csv  # Starter template for the "try your own data" feature
│   └── sample_edges_template.csv  # Starter template for the "try your own data" feature
├── outputs/                        # Trained model + evaluation artifacts (model, calibrator, isolation forest, logs)
├── assets/                         # README images
├── requirements.txt
└── README.md
```

---

## 🏭 The data

Real, labeled supply-chain fraud data isn't publicly available (for obvious reasons), so this
project **generates a synthetic but realistic supply chain network**:

- **250 entities** across 4 tiers: 60 suppliers → 40 manufacturers → 30 warehouses → 120 retailers
- **511 transaction lanes**, built with region-aware, tier-respecting logic (a manufacturer
  preferentially sources from suppliers in its own region, etc.)
- Each lane carries realistic features: average order quantity, average cost, lead time, delay
  variance, order frequency, transport mode

**Four distinct anomaly patterns are injected** (giving ground truth for evaluation):

| Type | Pattern | Real-world analogue |
|---|---|---|
| `FRAUD_RING` | A small cluster of entities over-trading with suspiciously uniform order sizes/frequency | Collusion / kickback schemes |
| `GHOST_SHIPMENT` | Near-zero quantity paired with abnormally high cost | Invoice fraud |
| `ROUTE_HIJACK` | Edges that skip tiers or connect geographically illogical entities with implausibly fast lead times | Diverted/rerouted shipments |
| `DELAY_CASCADE` | A hub node whose entire outgoing lane set suddenly shows spiked delay variance | Disruption propagating downstream |

Overall anomaly rate: **~8%** of all lanes — a realistic, imbalanced detection setting.

---

## 🕸️ The model

`models/gat.py` implements a two-layer, multi-head **Graph Attention Network from scratch**
(no `torch_geometric` dependency — a deliberate choice so every operation is transparent and
explainable, and the project has zero fragile compiled-binary dependencies):

1. **Layer 1** — 4-head GAT layer: node features → 32-dim hidden representation per head, concatenated
2. **Layer 2** — single-head GAT layer → final 32-dim node embeddings
3. **Edge scoring head** — an MLP over `[embedding(source) || embedding(destination) || raw edge features]` → anomaly probability for that specific transaction

**Key design decision:** the model classifies **edges (transactions), not nodes (companies)** —
matching the real problem. Node embeddings capture *relational* context; raw edge features
capture the transaction's own numeric signature. Combining both catches anomalies that look
fine alone but wrong in context, and vice versa.

Class imbalance (~8% positive) is handled with a weighted BCE loss rather than naive oversampling.

### Explainability

GAT's attention mechanism produces a weight for every neighbor relationship during message
passing. These are aggregated into an **"attention received"** score per node — a proxy for how
much the model's learned structure singles an entity out — and surfaced directly in the
dashboard (larger node = more attention received).

---

## 📊 Results

| Metric | Score |
|---|---|
| Precision | 0.75 |
| Recall | 0.75 |
| F1 | 0.75 |
| ROC-AUC | **0.987** |
| PR-AUC | 0.891 |
| Brier score (calibrated) | 0.024 |

The model was **never given the anomaly type during training** — only a binary `is_anomaly`
label. Recovering strong detection across all four structurally distinct anomaly types suggests
it learned real underlying graph/feature patterns rather than memorizing one signature.

---

## 🧪 Try it on your own data — and its honest limits

The dashboard includes a **"Try It On Your Own Data"** section where you can upload any
supply chain (matching the documented schema) and get every transaction scored by the
already-trained model, no retraining required.

**Important honesty note:** this model was trained entirely on synthetic data with 4
deliberately injected anomaly patterns. It has never seen real fraud. So rather than silently
pretending to be accurate on any dataset thrown at it, the inference pipeline includes several
layers that make the system honest about its own limits:

- **Input validation** — malformed CSVs (missing columns, bad tiers, negative values, dangling
  references) get clear, specific error messages instead of a stack trace.
- **Calibrated probabilities** — the GAT's raw sigmoid output is passed through a Platt-scaling
  calibrator (fit on a held-out validation set) so a "0.9" score means something closer to an
  actual 90% likelihood, not just an arbitrary large number.
- **Out-of-distribution (OOD) detection** — every uploaded transaction's numeric features are
  checked against the 1st-99th percentile range seen during training. Transactions outside that
  range are flagged — this is what happens when someone's cost/quantity scales are simply
  different from the synthetic training data.
- **An independent Isolation Forest cross-check** — a completely separate unsupervised model,
  trained directly on raw features with no graph structure and no labels, gives a second
  opinion. When it agrees with the GAT, confidence goes up; when it disagrees, that's flagged.
- **A combined reliability score (🟢 High / 🟡 Medium / 🔴 Low)** shown per transaction, so a
  user knows how much to trust each specific result rather than treating every output as
  equally certain.
- **Logging** — every inference run logs input size, timing, how many were flagged, and how
  many were out-of-distribution, to `outputs/inference.log`.

This is the honest engineering answer to "would this work on real company data?" — the *pipeline*
is production-grade (validated inputs, calibrated outputs, a documented reliability signal), but
the *learned patterns* are only proven on synthetic data, and the system says so rather than
guessing silently.

---

## ⚙️ Running it locally

```bash
git clone https://github.com/AakashSingh82/Supply-Chain-Fraud-Detection-GNN.git
cd Supply-Chain-Fraud-Detection-GNN

pip install -r requirements.txt

# 1. Generate the synthetic supply chain graph + injected anomalies
python data/generate_graph.py

# 2. Train the GAT, fit the calibrator + isolation forest, and export evaluation artifacts
python models/train.py

# 3. (Optional) export static PNGs for a README/portfolio
python dashboard/export_static_visuals.py

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py
```

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this entire repo to GitHub (make sure `data/`, `outputs/` generated files are committed —
   the dashboard reads them directly, and Streamlit Cloud won't run the training scripts for you)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**, select your repo/branch
4. Set **Main file path** to: `dashboard/app.py`
5. Deploy — Streamlit Cloud automatically redeploys on every future push to `main`

---

## 🔭 Possible extensions

- **Temporal GNN** — model the graph as time-ordered snapshots to catch anomalies as they emerge
- **Real-world validation** — re-run the pipeline on a real dataset (e.g. DataCo Supply Chain)
- **Active learning loop** — prioritize the most uncertain flagged edges for human review
- **Multi-hop counterfactuals** — "what minimal change would make this transaction look normal?"
- **Domain adaptation** — cheaply re-fit the scaler/calibrator on a new company's data distribution
  without full retraining, to reduce out-of-distribution rates on real-world uploads

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core implementation |
| **Deep Learning** | PyTorch | Custom Graph Attention Network (built from scratch, no `torch_geometric`) |
| **Graph construction** | NetworkX | Building and representing the supply chain as a graph |
| **Classical ML** | scikit-learn | Train/test splitting, evaluation metrics, `StandardScaler`, Platt-scaling calibration (`LogisticRegression`), `IsolationForest` cross-check |
| **Data handling** | pandas, NumPy | Feature engineering, data manipulation |
| **Dashboard/UI** | Streamlit | Interactive web dashboard, file upload, live scoring |
| **Visualization** | Plotly | Interactive network graph and charts inside the dashboard |
| **Visualization** | Matplotlib | Static PNG exports for README/portfolio |
| **Deployment** | Streamlit Community Cloud | Free hosting, auto-redeploys on every GitHub push |
| **Version control** | Git + GitHub | Source control and collaboration |

---

## 📄 License

MIT — free to use, modify, and build on.
