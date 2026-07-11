"""
Production-hardened inference pipeline.

This is what actually runs when someone uploads their own supply chain data
through the dashboard. On top of basic GAT scoring, it adds the pieces a
real deployment needs that a pure "run the model" script doesn't:

  1. INPUT VALIDATION — clear, specific error messages instead of cryptic
     stack traces when uploaded data is malformed.
  2. LOGGING — every inference run is logged (input size, timing, how many
     flagged, how many out-of-distribution) for auditability.
  3. CALIBRATED PROBABILITIES — the GAT's raw sigmoid output is passed
     through a Platt-scaling calibrator (fit on a held-out validation set
     during training) so "0.9" means something closer to "90% likely."
  4. OUT-OF-DISTRIBUTION (OOD) DETECTION — flags transactions whose
     numeric features fall far outside the range the model was ever
     trained on. This is the honest answer to "will this work on our
     data" — instead of silently guessing, it tells you when it's
     guessing.
  5. INDEPENDENT UNSUPERVISED CROSS-CHECK — an Isolation Forest trained
     directly on raw features (no GAT, no learned graph structure) gives
     a second opinion. Agreement between the two models raises
     confidence; disagreement lowers it.
  6. RELIABILITY SCORE — combines 4 and 5 into a single High/Medium/Low
     label per transaction, so a user knows how much to trust each
     result rather than treating every output as equally certain.

Honest scope note: none of this makes the model's underlying pattern
recognition more accurate on real, unseen fraud types — no model can
claim that without real labeled data to learn from. What it does is make
the SYSTEM honest about its own limits, which is the correct engineering
response to a genuine distribution-shift problem.
"""

import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
from models.gat import SupplyChainGAT

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ------------------------------------------------------------------ logging
logger = logging.getLogger("scgnn_inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_path = os.path.join(OUT_DIR, "inference.log")
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


REQUIRED_NODE_COLS = {"node_id", "tier", "name", "region"}
REQUIRED_EDGE_COLS = {"src", "dst", "avg_qty", "avg_cost", "lead_time_days",
                       "delay_variance", "order_freq_per_qtr", "transport_mode"}
VALID_TIERS = {"Supplier", "Manufacturer", "Warehouse", "Retailer"}


class ValidationError(Exception):
    """Raised when uploaded data fails schema/sanity checks. The message
    is written to be shown directly to an end user, not just a developer."""
    pass


# ============================================================
# 1. INPUT VALIDATION
# ============================================================
def validate_inputs(nodes_df, edges_df):
    errors = []

    missing_node_cols = REQUIRED_NODE_COLS - set(nodes_df.columns)
    if missing_node_cols:
        errors.append(f"nodes.csv is missing required columns: {sorted(missing_node_cols)}")

    missing_edge_cols = REQUIRED_EDGE_COLS - set(edges_df.columns)
    if missing_edge_cols:
        errors.append(f"edges.csv is missing required columns: {sorted(missing_edge_cols)}")

    if errors:
        raise ValidationError(" | ".join(errors))

    if nodes_df.empty:
        errors.append("nodes.csv has no rows.")
    if edges_df.empty:
        errors.append("edges.csv has no rows.")

    if nodes_df["node_id"].duplicated().any():
        dupes = nodes_df[nodes_df["node_id"].duplicated()]["node_id"].tolist()[:5]
        errors.append(f"nodes.csv has duplicate node_id values (e.g. {dupes}). Each node_id must be unique.")

    bad_tiers = set(nodes_df["tier"].unique()) - VALID_TIERS
    if bad_tiers:
        errors.append(
            f"nodes.csv has unrecognized tier value(s): {sorted(bad_tiers)}. "
            f"Must be one of: {sorted(VALID_TIERS)}."
        )

    valid_ids = set(nodes_df["node_id"])
    bad_src = set(edges_df["src"]) - valid_ids
    bad_dst = set(edges_df["dst"]) - valid_ids
    if bad_src:
        errors.append(f"edges.csv references src node_id(s) not present in nodes.csv: {sorted(bad_src)[:5]}")
    if bad_dst:
        errors.append(f"edges.csv references dst node_id(s) not present in nodes.csv: {sorted(bad_dst)[:5]}")

    numeric_cols = ["avg_qty", "avg_cost", "lead_time_days", "delay_variance", "order_freq_per_qtr"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(edges_df[col]):
            errors.append(f"edges.csv column '{col}' must be numeric — found non-numeric values.")
            continue
        if edges_df[col].isna().any():
            errors.append(f"edges.csv column '{col}' has missing (NaN) values.")
        if (edges_df[col] < 0).any():
            errors.append(f"edges.csv column '{col}' has negative values, which isn't physically meaningful here.")

    if errors:
        raise ValidationError(" | ".join(errors))


# ============================================================
# Feature building (reused from training's exact logic)
# ============================================================
def align_columns(df_dummies, expected_columns):
    return df_dummies.reindex(columns=expected_columns, fill_value=0)


def build_node_features_for_inference(nodes_df, node_feat_names):
    tier_dummies = pd.get_dummies(nodes_df["tier"], prefix="tier")
    region_dummies = pd.get_dummies(nodes_df["region"], prefix="region")
    feat_df = pd.concat([tier_dummies, region_dummies], axis=1)
    feat_df = align_columns(feat_df, node_feat_names)
    return feat_df.values.astype(np.float32)


def build_edge_features_for_inference(edges_df, edge_feat_names, scaler, numeric_cols):
    numeric = edges_df[numeric_cols].copy()
    numeric_scaled = pd.DataFrame(
        scaler.transform(numeric), columns=[c + "_z" for c in numeric_cols]
    )
    mode_dummies = pd.get_dummies(edges_df["transport_mode"], prefix="mode")
    edge_feat_df = pd.concat([numeric_scaled, mode_dummies.reset_index(drop=True)], axis=1)
    edge_feat_df = align_columns(edge_feat_df, edge_feat_names)
    return edge_feat_df.values.astype(np.float32), numeric_scaled


def build_adjacency(n_nodes, edges_df, node_id_to_idx):
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in edges_df.iterrows():
        u, v = node_id_to_idx[row.src], node_id_to_idx[row.dst]
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


# ============================================================
# 4. OUT-OF-DISTRIBUTION DETECTION
# ============================================================
def compute_ood_flags(numeric_scaled_df, ood_reference_stats):
    """For each transaction, flags it as out-of-distribution if ANY of its
    scaled numeric features falls outside the 1st-99th percentile range
    observed in the training data. Returns a boolean array + a list of
    which feature(s) triggered it, for transparency."""
    n = len(numeric_scaled_df)
    is_ood = np.zeros(n, dtype=bool)
    triggering_features = [[] for _ in range(n)]

    for col, bounds in ood_reference_stats.items():
        if col not in numeric_scaled_df.columns:
            continue
        values = numeric_scaled_df[col].values
        out_of_range = (values < bounds["p01"]) | (values > bounds["p99"])
        is_ood |= out_of_range
        for i in np.where(out_of_range)[0]:
            triggering_features[i].append(col.replace("_z", ""))

    return is_ood, triggering_features


# ============================================================
# 5. RELIABILITY SCORE (combines OOD + GAT/IsolationForest agreement)
# ============================================================
def compute_reliability(is_ood, gat_flag, iso_flag):
    """
    High   : in-distribution AND both models agree
    Medium : in-distribution but models disagree (ambiguous case)
    Low    : out-of-distribution (model is extrapolating beyond training data)
    """
    reliability = []
    for ood, g, i in zip(is_ood, gat_flag, iso_flag):
        if ood:
            reliability.append("Low")
        elif g == i:
            reliability.append("High")
        else:
            reliability.append("Medium")
    return reliability


def load_pretrained_artifacts():
    with open(f"{OUT_DIR}/preprocessing.pkl", "rb") as f:
        prep = pickle.load(f)
    with open(f"{OUT_DIR}/calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(f"{OUT_DIR}/isolation_forest.pkl", "rb") as f:
        iso_bundle = pickle.load(f)

    model = SupplyChainGAT(
        node_in_dim=prep["node_in_dim"],
        edge_in_dim=prep["edge_in_dim"],
        hidden_dim=32,
        n_heads=4,
        dropout=0.3,
    )
    model.load_state_dict(torch.load(f"{OUT_DIR}/gat_model.pt", map_location="cpu"))
    model.eval()
    return model, prep, calibrator, iso_bundle


def score_new_supply_chain(nodes_df, edges_df):
    """
    Main entry point. Validates inputs, then scores every edge with:
      - predicted_prob            (raw GAT sigmoid output)
      - predicted_prob_calibrated (Platt-scaled, more honest probability)
      - predicted_anomaly         (thresholded calibrated prediction)
      - isolation_forest_score    (independent unsupervised anomaly score)
      - is_out_of_distribution    (bool)
      - ood_features              (which raw features triggered OOD, if any)
      - reliability               (High / Medium / Low)
    """
    t0 = time.time()
    logger.info(f"Inference request received: {len(nodes_df)} nodes, {len(edges_df)} edges")

    validate_inputs(nodes_df, edges_df)

    model, prep, calibrator, iso_bundle = load_pretrained_artifacts()

    node_ids = nodes_df["node_id"].tolist()
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)

    node_feats_np = build_node_features_for_inference(nodes_df, prep["node_feat_names"])
    edge_feats_np, numeric_scaled_df = build_edge_features_for_inference(
        edges_df, prep["edge_feat_names"], prep["scaler"], prep["numeric_cols"]
    )
    adj_np = build_adjacency(n_nodes, edges_df, node_id_to_idx)

    node_feats = torch.tensor(node_feats_np)
    edge_feats = torch.tensor(edge_feats_np)
    adj = torch.tensor(adj_np)

    edge_index_np = np.array([
        [node_id_to_idx[s], node_id_to_idx[d]] for s, d in zip(edges_df.src, edges_df.dst)
    ])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    with torch.no_grad():
        logits, node_embeddings, (attn1, attn2) = model(node_feats, adj, edge_index, edge_feats)
        raw_probs = torch.sigmoid(logits).numpy()

    calibrated_probs = calibrator.predict_proba(logits.numpy().reshape(-1, 1))[:, 1]

    iso_model = iso_bundle["model"]
    iso_raw = -iso_model.decision_function(edges_df[prep["numeric_cols"]].values)
    iso_scores = (iso_raw - iso_bundle["score_min"]) / max(iso_bundle["score_max"] - iso_bundle["score_min"], 1e-9)
    iso_scores = np.clip(iso_scores, 0, 1)
    iso_flags = (iso_scores > 0.5).astype(int)

    is_ood, ood_features = compute_ood_flags(numeric_scaled_df, prep["ood_reference_stats"])
    gat_flags = (calibrated_probs > 0.5).astype(int)
    reliability = compute_reliability(is_ood, gat_flags, iso_flags)

    edges_out = edges_df.copy()
    edges_out["predicted_prob"] = raw_probs
    edges_out["predicted_prob_calibrated"] = calibrated_probs
    edges_out["predicted_anomaly"] = gat_flags
    edges_out["isolation_forest_score"] = iso_scores
    edges_out["is_out_of_distribution"] = is_ood
    edges_out["ood_features"] = [", ".join(f) if f else "" for f in ood_features]
    edges_out["reliability"] = reliability

    avg_attn = attn2.mean(dim=0).squeeze(0).numpy()
    node_attn_in = avg_attn.sum(axis=0)
    nodes_out = nodes_df.copy()
    nodes_out["attention_received"] = [node_attn_in[node_id_to_idx[nid]] for nid in node_ids]

    elapsed = time.time() - t0
    n_flagged = int(gat_flags.sum())
    n_ood = int(is_ood.sum())
    logger.info(
        f"Inference complete in {elapsed:.2f}s | flagged={n_flagged}/{len(edges_df)} "
        f"| out_of_distribution={n_ood}/{len(edges_df)} "
        f"| reliability_counts={pd.Series(reliability).value_counts().to_dict()}"
    )

    return nodes_out, edges_out
