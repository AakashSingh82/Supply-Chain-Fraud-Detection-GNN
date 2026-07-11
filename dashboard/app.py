"""
Supply Chain Fraud & Disruption Detection — Interactive Dashboard
------------------------------------------------------------------
Run with:  streamlit run app.py

Shows:
  - Network graph with anomalous lanes highlighted
  - Model performance metrics (precision/recall/F1/ROC-AUC/PR-AUC)
  - Training curves
  - Ranked list of top flagged transactions with model confidence
  - Node-level "suspicion score" from attention weights (explainability)
  - Drill-down: click any flagged edge to see WHY it was flagged
"""

import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

import sys
sys.path.append(PROJECT_ROOT)
from models.inference import score_new_supply_chain, ValidationError

st.set_page_config(page_title="Supply Chain GNN Anomaly Detection", layout="wide")


@st.cache_data
def load_all():
    nodes_df = pd.read_csv(f"{DATA_DIR}/nodes.csv")
    edges_pred_df = pd.read_csv(f"{OUT_DIR}/edges_with_predictions.csv")
    nodes_attn_df = pd.read_csv(f"{OUT_DIR}/nodes_with_attention.csv")
    with open(f"{OUT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{OUT_DIR}/history.json") as f:
        history = json.load(f)
    return nodes_df, edges_pred_df, nodes_attn_df, metrics, history


nodes_df, edges_df, nodes_attn_df, metrics, history = load_all()

# ------------------------------------------------------------------ header
st.title("🔗 Supply Chain Fraud & Disruption Detection")
st.caption(
    "Graph Attention Network trained on a simulated multi-tier supply chain "
    "(suppliers → manufacturers → warehouses → retailers) to flag fraudulent "
    "or anomalous transactions using both node relationships and transaction features."
)

# ------------------------------------------------------------------ top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precision", f"{metrics['test_precision']:.2f}")
c2.metric("Recall", f"{metrics['test_recall']:.2f}")
c3.metric("F1 Score", f"{metrics['test_f1']:.2f}")
c4.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
c5.metric("PR-AUC", f"{metrics['test_pr_auc']:.3f}")

st.markdown(
    f"**Network size:** {metrics['n_nodes']} nodes · {metrics['n_edges']} transaction lanes · "
    f"**{metrics['n_anomalies']} true anomalies** ({metrics['anomaly_rate']:.1%} of all lanes)"
)

st.divider()

# ============================================================
# Let visitors upload and score their OWN supply chain data
# ============================================================
st.header("🧪 Try It On Your Own Data")
st.markdown(
    "Upload your own supply chain network and this **already-trained** model will "
    "score every transaction for anomaly risk — no retraining needed. This works on "
    "*any* supply chain with the schema below, not just the synthetic demo data below."
)

with st.expander("📄 Required file format (click to expand)"):
    st.markdown("""
**`nodes.csv`** — one row per entity:
| column | description |
|---|---|
| `node_id` | unique integer ID |
| `tier` | one of: `Supplier`, `Manufacturer`, `Warehouse`, `Retailer` |
| `name` | display name |
| `region` | any string (e.g. `North`, `Delhi`, `EU-West`) |

**`edges.csv`** — one row per recurring transaction/shipment lane:
| column | description |
|---|---|
| `src` | source node_id |
| `dst` | destination node_id |
| `avg_qty` | average order quantity |
| `avg_cost` | average order cost |
| `lead_time_days` | average delivery lead time |
| `delay_variance` | std-dev of delivery delays |
| `order_freq_per_qtr` | orders per quarter on this lane |
| `transport_mode` | e.g. `Road`, `Rail`, `Air`, `Sea` |
""")
    st.caption("Download starter templates below and edit them with your own data.")
    col_a, col_b = st.columns(2)
    with open(f"{SCRIPT_DIR}/sample_nodes_template.csv", "rb") as f:
        col_a.download_button("Download nodes.csv template", f, file_name="nodes_template.csv")
    with open(f"{SCRIPT_DIR}/sample_edges_template.csv", "rb") as f:
        col_b.download_button("Download edges.csv template", f, file_name="edges_template.csv")

up_col1, up_col2 = st.columns(2)
uploaded_nodes = up_col1.file_uploader("Upload nodes.csv", type="csv", key="nodes_upload")
uploaded_edges = up_col2.file_uploader("Upload edges.csv", type="csv", key="edges_upload")

if uploaded_nodes is not None and uploaded_edges is not None:
    try:
        user_nodes_df = pd.read_csv(uploaded_nodes)
        user_edges_df = pd.read_csv(uploaded_edges)

        with st.spinner("Validating and scoring your supply chain with the trained GAT model..."):
            scored_nodes, scored_edges = score_new_supply_chain(user_nodes_df, user_edges_df)

        n_flagged = int(scored_edges["predicted_anomaly"].sum())
        n_ood = int(scored_edges["is_out_of_distribution"].sum())
        reliability_counts = scored_edges["reliability"].value_counts().to_dict()

        st.success(f"Done! Flagged **{n_flagged} / {len(scored_edges)}** transactions as anomalous.")

        if n_ood > 0:
            st.warning(
                f"⚠️ **{n_ood} transactions ({n_ood/len(scored_edges):.1%}) fall outside the data range "
                f"this model was trained on.** Predictions for these are marked **Low reliability** below — "
                f"treat them as a rough signal, not a confident verdict. This happens when your data's scale "
                f"or patterns (e.g. cost/quantity ranges) differ substantially from the training data."
            )

        rel_col1, rel_col2, rel_col3 = st.columns(3)
        rel_col1.metric("🟢 High reliability", reliability_counts.get("High", 0))
        rel_col2.metric("🟡 Medium reliability", reliability_counts.get("Medium", 0))
        rel_col3.metric("🔴 Low reliability", reliability_counts.get("Low", 0))
        st.caption(
            "**High** = model and an independent Isolation Forest agree, and data is in-distribution. "
            "**Medium** = the two models disagree (ambiguous case). "
            "**Low** = data falls outside the model's training range — treat with caution."
        )

        node_name_lookup_user = dict(zip(scored_nodes.node_id, scored_nodes.name))
        result_display = scored_edges.sort_values("predicted_prob_calibrated", ascending=False).copy()
        result_display["source"] = result_display["src"].map(node_name_lookup_user)
        result_display["destination"] = result_display["dst"].map(node_name_lookup_user)
        result_display["confidence"] = (result_display["predicted_prob_calibrated"] * 100).round(1).astype(str) + "%"
        result_display["reliability_icon"] = result_display["reliability"].map(
            {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}
        )

        st.dataframe(
            result_display[["source", "destination", "confidence", "reliability_icon", "ood_features",
                             "avg_qty", "avg_cost", "lead_time_days", "delay_variance",
                             "order_freq_per_qtr"]].rename(columns={
                                 "reliability_icon": "reliability", "ood_features": "flagged_features"
                             }).reset_index(drop=True),
            width="stretch", height=350
        )

        csv_bytes = scored_edges.to_csv(index=False).encode("utf-8")
        st.download_button("Download full results as CSV", csv_bytes, file_name="scored_transactions.csv")

    except ValidationError as e:
        st.error(f"❌ Your files didn't pass validation:\n\n{str(e).replace(' | ', chr(10) + '- ')}")
        st.caption("Fix the issues above and re-upload. Check the format guide above for the expected schema.")
    except Exception as e:
        st.error(f"Something unexpected went wrong: {e}")
        st.caption("Double check your CSVs match the required format above, including exact column names and types.")
else:
    st.info("Upload both files above to get anomaly scores on your own data.")

st.divider()
st.header("📊 Demo: Synthetic Supply Chain Network")
st.caption("Everything below uses the synthetic demo data this model was trained on.")


# ------------------------------------------------------------------ sidebar controls
st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect(
    "Show tiers", options=nodes_df["tier"].unique().tolist(),
    default=nodes_df["tier"].unique().tolist()
)
confidence_threshold = st.sidebar.slider("Flag threshold (predicted probability)", 0.0, 1.0, 0.5, 0.05)
show_only_flagged = st.sidebar.checkbox("Show only flagged lanes on graph", value=True)

# ------------------------------------------------------------------ build layout graph
G = nx.DiGraph()
for _, row in nodes_df.iterrows():
    G.add_node(row.node_id, tier=row.tier, name=row.name, region=row.region)
for _, row in edges_df.iterrows():
    G.add_edge(row.src, row.dst)

tier_order = {"Supplier": 0, "Manufacturer": 1, "Warehouse": 2, "Retailer": 3}
pos = {}
tier_counts = {t: 0 for t in tier_order}
for n, d in G.nodes(data=True):
    t = d["tier"]
    x = tier_order[t]
    y = tier_counts[t]
    tier_counts[t] += 1
    pos[n] = (x, y)

# normalize y within each tier so nodes are vertically centered
max_count = max(tier_counts.values())
for n, d in G.nodes(data=True):
    t = d["tier"]
    x, y = pos[n]
    total_in_tier = list(nodes_df[nodes_df.tier == t].node_id).index(n)
    pos[n] = (x, (total_in_tier - tier_counts[t] / 2))

# ------------------------------------------------------------------ plot graph
flagged_edges = edges_df[edges_df["predicted_prob"] >= confidence_threshold]
normal_edges = edges_df[edges_df["predicted_prob"] < confidence_threshold]

edge_traces = []

def edge_trace(df, color, width, name, dash=None):
    xs, ys = [], []
    for _, row in df.iterrows():
        x0, y0 = pos[row.src]
        x1, y1 = pos[row.dst]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
        hoverinfo="none", name=name, showlegend=True,
    )

if not show_only_flagged:
    edge_traces.append(edge_trace(normal_edges, "rgba(150,150,150,0.25)", 1, "Normal lane"))
edge_traces.append(edge_trace(flagged_edges, "rgba(220,30,30,0.85)", 2.5, "Flagged anomaly"))

node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
tier_color_map = {"Supplier": "#4C72B0", "Manufacturer": "#55A868", "Warehouse": "#C44E52", "Retailer": "#8172B2"}

attn_lookup = dict(zip(nodes_attn_df.node_id, nodes_attn_df.attention_received))

for n, d in G.nodes(data=True):
    if d["tier"] not in tier_filter:
        continue
    x, y = pos[n]
    node_x.append(x)
    node_y.append(y)
    node_color.append(tier_color_map[d["tier"]])
    suspicion = attn_lookup.get(n, 0)
    node_text.append(f"{d['name']} ({d['tier']}, {d['region']})<br>Attention received: {suspicion:.2f}")
    node_size.append(6 + min(suspicion * 2, 14))

node_trace = go.Scatter(
    x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
    marker=dict(color=node_color, size=node_size, line=dict(width=0.5, color="white")),
    name="Entities", showlegend=False,
)

fig = go.Figure(data=edge_traces + [node_trace])
fig.update_layout(
    height=650,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
               tickmode="array", tickvals=[0, 1, 2, 3],
               ticktext=["Suppliers", "Manufacturers", "Warehouses", "Retailers"]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

st.subheader("Network View")
st.plotly_chart(fig, use_container_width=True)
st.caption("Node size = attention received (a proxy for how much the model's learned "
           "relationships single this entity out). Red lanes = flagged transactions.")

st.divider()

# ------------------------------------------------------------------ ranked table + drilldown
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Top Flagged Transactions")
    node_name_lookup = dict(zip(nodes_df.node_id, nodes_df.name))
    display_df = flagged_edges.sort_values("predicted_prob", ascending=False).copy()
    display_df["source"] = display_df["src"].map(node_name_lookup)
    display_df["destination"] = display_df["dst"].map(node_name_lookup)
    display_df["confidence"] = (display_df["predicted_prob"] * 100).round(1).astype(str) + "%"
    display_df["correct?"] = np.where(
        display_df["is_anomaly"] == display_df["predicted_anomaly"], "✅", "❌"
    )
    cols = ["source", "destination", "anomaly_type", "confidence", "avg_qty", "avg_cost",
            "lead_time_days", "delay_variance", "order_freq_per_qtr", "correct?"]
    st.dataframe(display_df[cols].reset_index(drop=True), use_container_width=True, height=400)

with right:
    st.subheader("Anomaly Type Breakdown")
    type_counts = edges_df[edges_df["is_anomaly"] == 1]["anomaly_type"].value_counts()
    caught_counts = edges_df[
        (edges_df["is_anomaly"] == 1) & (edges_df["predicted_anomaly"] == 1)
    ]["anomaly_type"].value_counts()

    breakdown_fig = go.Figure()
    breakdown_fig.add_trace(go.Bar(x=type_counts.index, y=type_counts.values, name="True anomalies",
                                    marker_color="rgba(150,150,150,0.5)"))
    breakdown_fig.add_trace(go.Bar(x=caught_counts.index, y=caught_counts.values, name="Caught by model",
                                    marker_color="rgba(220,30,30,0.85)"))
    breakdown_fig.update_layout(barmode="overlay", height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(breakdown_fig, use_container_width=True)

    st.markdown("**Why this matters:** the model was never told the anomaly *type* during "
                "training — only a binary is_anomaly label. Recovering type-level structure "
                "this well suggests it's learning real underlying patterns (collusion rings, "
                "invoice fraud, route anomalies, disruption cascades), not just memorizing.")

st.divider()

# ------------------------------------------------------------------ training curve
st.subheader("Training Curves")
hist_df = pd.DataFrame(history)
curve_fig = go.Figure()
curve_fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.train_loss, name="Train loss"))
curve_fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.val_loss, name="Val loss"))
curve_fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.val_f1, name="Val F1", yaxis="y2"))
curve_fig.update_layout(
    height=350,
    yaxis=dict(title="Loss"),
    yaxis2=dict(title="F1", overlaying="y", side="right", range=[0, 1.05]),
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(curve_fig, use_container_width=True)

st.caption(
    "Built with a custom Graph Attention Network (PyTorch) — no external graph-learning "
    "framework dependency. Edge scoring combines learned node embeddings with raw "
    "transaction features; attention weights double as an explainability signal."
)
