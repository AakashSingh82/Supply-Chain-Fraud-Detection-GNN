"""
Exports standalone static visuals (PNG) for use in the README / resume /
LinkedIn post — so the project has evidence even without someone running
the live Streamlit app.
"""
import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

nodes_df = pd.read_csv(f"{DATA_DIR}/nodes.csv")
edges_df = pd.read_csv(f"{OUT_DIR}/edges_with_predictions.csv")
with open(f"{OUT_DIR}/metrics.json") as f:
    metrics = json.load(f)
with open(f"{OUT_DIR}/history.json") as f:
    history = json.load(f)

tier_order = {"Supplier": 0, "Manufacturer": 1, "Warehouse": 2, "Retailer": 3}
tier_color_map = {"Supplier": "#4C72B0", "Manufacturer": "#55A868", "Warehouse": "#C44E52", "Retailer": "#8172B2"}

pos = {}
tier_groups = {t: nodes_df[nodes_df.tier == t].node_id.tolist() for t in tier_order}
for t, ids in tier_groups.items():
    for i, n in enumerate(ids):
        pos[n] = (tier_order[t], i - len(ids) / 2)

# ---------------------------- Fig 1: network with anomalies ----------------------------
fig, ax = plt.subplots(figsize=(13, 9))
for _, row in edges_df.iterrows():
    x0, y0 = pos[row.src]
    x1, y1 = pos[row.dst]
    if row.predicted_prob >= 0.5:
        ax.plot([x0, x1], [y0, y1], color="#D62728", linewidth=1.8, alpha=0.85, zorder=3)
    else:
        ax.plot([x0, x1], [y0, y1], color="#CCCCCC", linewidth=0.5, alpha=0.4, zorder=1)

for t, ids in tier_groups.items():
    xs = [pos[n][0] for n in ids]
    ys = [pos[n][1] for n in ids]
    ax.scatter(xs, ys, s=40, color=tier_color_map[t], edgecolor="white", linewidth=0.5, zorder=4, label=t)

ax.set_xticks(list(tier_order.values()))
ax.set_xticklabels(list(tier_order.keys()), fontsize=12)
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Supply Chain Network — Flagged Anomalous Transactions in Red", fontsize=14, pad=15)
handles = [mpatches.Patch(color=c, label=t) for t, c in tier_color_map.items()]
handles.append(mpatches.Patch(color="#D62728", label="Flagged anomalous lane"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_network_anomalies.png", dpi=180)
plt.close()

# ---------------------------- Fig 2: training curves ----------------------------
hist_df = pd.DataFrame(history)
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(hist_df.epoch, hist_df.train_loss, label="Train loss", color="#4C72B0")
ax1.plot(hist_df.epoch, hist_df.val_loss, label="Val loss", color="#C44E52")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss")
ax2 = ax1.twinx()
ax2.plot(hist_df.epoch, hist_df.val_f1, label="Val F1", color="#55A868", linestyle="--")
ax2.set_ylabel("Val F1")
ax2.set_ylim(0, 1.05)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("Training Curves")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_training_curves.png", dpi=180)
plt.close()

# ---------------------------- Fig 3: anomaly type breakdown ----------------------------
type_counts = edges_df[edges_df["is_anomaly"] == 1]["anomaly_type"].value_counts()
caught_counts = edges_df[
    (edges_df["is_anomaly"] == 1) & (edges_df["predicted_anomaly"] == 1)
]["anomaly_type"].value_counts().reindex(type_counts.index, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(type_counts))
width = 0.35
ax.bar(x - width / 2, type_counts.values, width, label="True anomalies", color="#AAAAAA")
ax.bar(x + width / 2, caught_counts.values, width, label="Caught by model", color="#D62728")
ax.set_xticks(x)
ax.set_xticklabels(type_counts.index, rotation=15)
ax.set_ylabel("Count")
ax.set_title("Detection Rate by Anomaly Type")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_anomaly_breakdown.png", dpi=180)
plt.close()

print("Saved: fig_network_anomalies.png, fig_training_curves.png, fig_anomaly_breakdown.png")
