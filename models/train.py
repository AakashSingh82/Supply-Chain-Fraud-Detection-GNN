"""
Training pipeline for the Supply Chain GAT anomaly detector.

Pipeline:
  1. Load nodes.csv / edges.csv
  2. Build node features (one-hot tier, one-hot region, degree stats)
  3. Build edge features (normalized qty, cost, lead time, delay var, freq, mode)
  4. Build dense adjacency matrix (+ self loops) for message passing
  5. Train/val/test split (stratified on is_anomaly, since it's imbalanced)
  6. Train GAT with weighted BCE loss (handles ~8% positive rate)
  7. Evaluate: Precision, Recall, F1, ROC-AUC, PR-AUC
  8. Save model + all artifacts needed by the dashboard (embeddings,
     attention weights, predictions) to outputs/
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
from models.gat import SupplyChainGAT

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    nodes_df = pd.read_csv(f"{DATA_DIR}/nodes.csv")
    edges_df = pd.read_csv(f"{DATA_DIR}/edges.csv")
    return nodes_df, edges_df


def build_node_features(nodes_df):
    tier_dummies = pd.get_dummies(nodes_df["tier"], prefix="tier")
    region_dummies = pd.get_dummies(nodes_df["region"], prefix="region")
    feat_df = pd.concat([tier_dummies, region_dummies], axis=1)
    feat_matrix = feat_df.values.astype(np.float32)
    return feat_matrix, feat_df.columns.tolist()


def build_edge_features(edges_df):
    mode_dummies = pd.get_dummies(edges_df["transport_mode"], prefix="mode")
    numeric_cols = ["avg_qty", "avg_cost", "lead_time_days", "delay_variance", "order_freq_per_qtr"]
    numeric = edges_df[numeric_cols].copy()

    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(numeric), columns=[c + "_z" for c in numeric_cols]
    )

    edge_feat_df = pd.concat([numeric_scaled, mode_dummies.reset_index(drop=True)], axis=1)
    return edge_feat_df.values.astype(np.float32), edge_feat_df.columns.tolist(), scaler


def build_adjacency(n_nodes, edges_df):
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in edges_df.iterrows():
        adj[int(row.src), int(row.dst)] = 1.0
        adj[int(row.dst), int(row.src)] = 1.0  # symmetric for message passing
    np.fill_diagonal(adj, 1.0)  # self loops
    return adj


def main():
    nodes_df, edges_df = load_data()
    n_nodes = len(nodes_df)

    node_feats_np, node_feat_names = build_node_features(nodes_df)
    edge_feats_np, edge_feat_names, scaler = build_edge_features(edges_df)
    adj_np = build_adjacency(n_nodes, edges_df)

    node_feats = torch.tensor(node_feats_np)
    edge_feats = torch.tensor(edge_feats_np)
    adj = torch.tensor(adj_np)
    edge_index = torch.tensor(edges_df[["src", "dst"]].values, dtype=torch.long)
    labels = torch.tensor(edges_df["is_anomaly"].values, dtype=torch.float32)

    # Stratified split over EDGES (this is what we're classifying)
    idx = np.arange(len(edges_df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.4, stratify=labels.numpy(), random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels.numpy()[temp_idx], random_state=42
    )

    model = SupplyChainGAT(
        node_in_dim=node_feats.shape[1],
        edge_in_dim=edge_feats.shape[1],
        hidden_dim=32,
        n_heads=4,
        dropout=0.3,
    )

    # Handle class imbalance (~8% positive) with pos_weight in BCE
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    n_epochs = 200
    best_val_f1 = -1
    best_state = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(node_feats, adj, edge_index, edge_feats)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval, _, _ = model(node_feats, adj, edge_index, edge_feats)
            val_loss = criterion(logits_eval[val_idx], labels[val_idx]).item()
            val_probs = torch.sigmoid(logits_eval[val_idx]).numpy()
            val_preds = (val_probs > 0.5).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                labels[val_idx].numpy(), val_preds, average="binary", zero_division=0
            )
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                         "val_precision": p, "val_recall": r, "val_f1": f1})

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                  f"| val_P {p:.3f} val_R {r:.3f} val_F1 {f1:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_final, node_embeddings, (attn1, attn2) = model(node_feats, adj, edge_index, edge_feats)
        test_probs = torch.sigmoid(logits_final[test_idx]).numpy()
        test_labels = labels[test_idx].numpy()
        test_preds = (test_probs > 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary", zero_division=0)
    roc_auc = roc_auc_score(test_labels, test_probs)
    pr_auc = average_precision_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds).tolist()
    uncalibrated_brier = brier_score_loss(test_labels, test_probs)

    print("\n=== TEST SET RESULTS (raw / uncalibrated) ===")
    print(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}  PR-AUC: {pr_auc:.3f}")
    print(f"Confusion matrix: {cm}")
    print(f"Brier score (uncalibrated): {uncalibrated_brier:.4f}  (lower is better, 0=perfect)")

    # ------------------------------------------------------------------
    # CALIBRATION: the GAT's raw sigmoid outputs are NOT guaranteed to be
    # well-calibrated probabilities (a 0.9 output doesn't necessarily mean
    # "90% likely to be fraud"). We fit a simple Platt-scaling calibrator
    # (logistic regression on the raw logit) using the VALIDATION set
    # (never the test set, to keep evaluation honest), then re-evaluate
    # calibration quality on the held-out test set.
    # ------------------------------------------------------------------
    with torch.no_grad():
        val_logits_np = logits_final[val_idx].detach().numpy().reshape(-1, 1)
    val_labels_np = labels[val_idx].numpy()

    calibrator = LogisticRegression()
    calibrator.fit(val_logits_np, val_labels_np)

    test_logits_np = logits_final[test_idx].detach().numpy().reshape(-1, 1)
    calibrated_test_probs = calibrator.predict_proba(test_logits_np)[:, 1]
    calibrated_brier = brier_score_loss(test_labels, calibrated_test_probs)

    print(f"Brier score (calibrated):   {calibrated_brier:.4f}")
    print("(Calibration fitted on validation set only, evaluated on held-out test set)")

    # ------------------------------------------------------------------
    # UNSUPERVISED CROSS-CHECK: an Isolation Forest trained directly on
    # the raw (unscaled) edge features, completely independent of the
    # GAT and its labels. This gives a second, structurally different
    # opinion on each transaction. When the GAT and the Isolation Forest
    # agree, we can be more confident; when they disagree, that's a
    # signal the case is ambiguous or potentially out-of-distribution
    # for the GAT specifically.
    # ------------------------------------------------------------------
    numeric_cols = ["avg_qty", "avg_cost", "lead_time_days", "delay_variance", "order_freq_per_qtr"]
    iso_forest = IsolationForest(
        n_estimators=200, contamination=float(labels.mean().item()), random_state=42
    )
    iso_forest.fit(edges_df[numeric_cols].values)
    # decision_function: higher = more normal, lower/negative = more anomalous.
    # We flip and min-max normalize to a 0-1 "isolation anomaly score" for easy combination.
    iso_raw_scores = -iso_forest.decision_function(edges_df[numeric_cols].values)
    iso_score_min, iso_score_max = iso_raw_scores.min(), iso_raw_scores.max()

    print(f"\nIsolation Forest fitted independently on raw edge features "
          f"(contamination={labels.mean().item():.3f})")

    # ---------------- Save all artifacts for the dashboard ----------------
    all_probs = torch.sigmoid(logits_final).detach().numpy()
    all_logits_np = logits_final.detach().numpy().reshape(-1, 1)
    all_calibrated_probs = calibrator.predict_proba(all_logits_np)[:, 1]
    all_iso_scores = (iso_raw_scores - iso_score_min) / max(iso_score_max - iso_score_min, 1e-9)

    edges_df_out = edges_df.copy()
    edges_df_out["predicted_prob"] = all_probs
    edges_df_out["predicted_prob_calibrated"] = all_calibrated_probs
    edges_df_out["predicted_anomaly"] = (all_probs > 0.5).astype(int)
    edges_df_out["isolation_forest_score"] = all_iso_scores
    edges_df_out["split"] = "train"
    edges_df_out.loc[val_idx, "split"] = "val"
    edges_df_out.loc[test_idx, "split"] = "test"
    edges_df_out.to_csv(f"{OUT_DIR}/edges_with_predictions.csv", index=False)

    # average attention received per node (heads averaged) -> "suspicion" signal
    avg_attn = attn2.mean(dim=0).squeeze(0).detach().numpy()  # (N, N)
    node_attn_in = avg_attn.sum(axis=0)  # how much attention a node receives in total

    nodes_df_out = nodes_df.copy()
    nodes_df_out["attention_received"] = node_attn_in
    nodes_df_out.to_csv(f"{OUT_DIR}/nodes_with_attention.csv", index=False)

    np.save(f"{OUT_DIR}/node_embeddings.npy", node_embeddings.detach().numpy())

    metrics = {
        "test_precision": p, "test_recall": r, "test_f1": f1,
        "test_roc_auc": roc_auc, "test_pr_auc": pr_auc,
        "confusion_matrix": cm,
        "test_brier_uncalibrated": uncalibrated_brier,
        "test_brier_calibrated": calibrated_brier,
        "n_nodes": n_nodes, "n_edges": len(edges_df),
        "n_anomalies": int(labels.sum().item()),
        "anomaly_rate": float(labels.mean().item()),
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{OUT_DIR}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), f"{OUT_DIR}/gat_model.pt")

    # OOD reference stats: per-feature training distribution (post-scaling,
    # so these are in z-score units) used to flag uploaded data that falls
    # far outside what the model has ever seen.
    edge_feats_scaled_df = pd.DataFrame(edge_feats_np, columns=edge_feat_names)
    numeric_z_cols = [c for c in edge_feat_names if c.endswith("_z")]
    ood_reference_stats = {
        col: {
            "p01": float(np.percentile(edge_feats_scaled_df[col], 1)),
            "p99": float(np.percentile(edge_feats_scaled_df[col], 99)),
        }
        for col in numeric_z_cols
    }

    with open(f"{OUT_DIR}/preprocessing.pkl", "wb") as f:
        pickle.dump({
            "node_feat_names": node_feat_names,
            "edge_feat_names": edge_feat_names,
            "scaler": scaler,
            "node_in_dim": node_feats.shape[1],
            "edge_in_dim": edge_feats.shape[1],
            "numeric_cols": numeric_cols,
            "ood_reference_stats": ood_reference_stats,
        }, f)

    with open(f"{OUT_DIR}/calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

    with open(f"{OUT_DIR}/isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": iso_forest, "score_min": float(iso_score_min), "score_max": float(iso_score_max)}, f)

    print(f"\nAll artifacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
