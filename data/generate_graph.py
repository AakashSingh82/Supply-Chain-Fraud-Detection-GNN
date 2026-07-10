"""
Synthetic Supply Chain Graph Generator
----------------------------------------
Builds a realistic, multi-tier supply chain network as a directed graph:

    Raw Material Suppliers -> Manufacturers -> Warehouses/DCs -> Retailers

Each edge represents a recurring shipment/transaction lane between two
entities, with engineered features (avg order qty, cost, lead time,
delay variance, transport mode, order frequency).

We then inject four distinct classes of anomalies into a subset of edges
so we have ground-truth labels for evaluation:

    1. FRAUD_RING      - a small cluster of nodes trading with unusually
                         high frequency / round-trip volume (collusion)
    2. GHOST_SHIPMENT   - edges with abnormally high cost but near-zero
                         quantity (invoice fraud pattern)
    3. ROUTE_HIJACK     - an edge that suddenly appears between two nodes
                         with no geographic/tier logic (diverted shipment)
    4. DELAY_CASCADE    - a set of edges with sharply increased lead-time
                         variance (disruption / bottleneck signature)

The output is saved as:
    - nodes.csv   (node_id, tier, name, region)
    - edges.csv   (src, dst, features..., is_anomaly, anomaly_type)
    - graph.gpickle (networkx DiGraph, for visualization)
"""

import os
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------- CONFIG ---------------------------------
N_SUPPLIERS = 60
N_MANUFACTURERS = 40
N_WAREHOUSES = 30
N_RETAILERS = 120

REGIONS = ["North", "South", "East", "West", "Central"]
TRANSPORT_MODES = ["Road", "Rail", "Air", "Sea"]

TOTAL_NODES = N_SUPPLIERS + N_MANUFACTURERS + N_WAREHOUSES + N_RETAILERS


def build_topology():
    G = nx.DiGraph()
    node_id = 0
    tiers = {}

    def add_nodes(n, tier_name):
        nonlocal node_id
        ids = []
        for i in range(n):
            region = random.choice(REGIONS)
            G.add_node(
                node_id,
                tier=tier_name,
                name=f"{tier_name[:4].upper()}_{i:03d}",
                region=region,
            )
            ids.append(node_id)
            node_id += 1
        tiers[tier_name] = ids
        return ids

    suppliers = add_nodes(N_SUPPLIERS, "Supplier")
    manufacturers = add_nodes(N_MANUFACTURERS, "Manufacturer")
    warehouses = add_nodes(N_WAREHOUSES, "Warehouse")
    retailers = add_nodes(N_RETAILERS, "Retailer")

    # --- Tier-respecting edges (the "legitimate" backbone) ---
    # Supplier -> Manufacturer (each manufacturer sources from 2-5 suppliers,
    # preferentially from the same region)
    for m in manufacturers:
        m_region = G.nodes[m]["region"]
        same_region = [s for s in suppliers if G.nodes[s]["region"] == m_region]
        pool = same_region if len(same_region) >= 2 else suppliers
        k = random.randint(2, 5)
        chosen = random.sample(pool, min(k, len(pool)))
        for s in chosen:
            add_transaction_edge(G, s, m)

    # Manufacturer -> Warehouse
    for w in warehouses:
        w_region = G.nodes[w]["region"]
        same_region = [m for m in manufacturers if G.nodes[m]["region"] == w_region]
        pool = same_region if len(same_region) >= 2 else manufacturers
        k = random.randint(2, 4)
        chosen = random.sample(pool, min(k, len(pool)))
        for m in chosen:
            add_transaction_edge(G, m, w)

    # Warehouse -> Retailer
    for r in retailers:
        r_region = G.nodes[r]["region"]
        same_region = [w for w in warehouses if G.nodes[w]["region"] == r_region]
        pool = same_region if len(same_region) >= 2 else warehouses
        k = random.randint(1, 3)
        chosen = random.sample(pool, min(k, len(pool)))
        for w in chosen:
            add_transaction_edge(G, w, r)

    return G, tiers


def add_transaction_edge(G, u, v, anomaly_type="NONE"):
    """Adds an edge with realistic baseline transaction features."""
    base_qty = np.random.gamma(shape=5.0, scale=80)          # units per order
    base_cost = base_qty * np.random.uniform(8, 25)          # currency units
    lead_time = np.random.normal(loc=5, scale=1.5)            # days
    lead_time = max(0.5, lead_time)
    delay_variance = np.random.exponential(scale=0.8)         # days std-dev
    freq = np.random.poisson(lam=12) + 1                      # orders/quarter
    mode = random.choice(TRANSPORT_MODES)

    G.add_edge(
        u, v,
        avg_qty=round(base_qty, 2),
        avg_cost=round(base_cost, 2),
        lead_time_days=round(lead_time, 2),
        delay_variance=round(delay_variance, 3),
        order_freq_per_qtr=int(freq),
        transport_mode=mode,
        is_anomaly=0,
        anomaly_type=anomaly_type,
    )


def inject_anomalies(G, tiers):
    """Injects 4 distinct anomaly patterns into the graph, returns list of
    (u, v, anomaly_type) for logging."""
    injected = []
    suppliers, manufacturers, warehouses, retailers = (
        tiers["Supplier"], tiers["Manufacturer"], tiers["Warehouse"], tiers["Retailer"]
    )

    # 1. FRAUD_RING: pick 3 manufacturers + 3 warehouses, make them
    #    over-trade with each other at implausibly high frequency
    #    and near-identical order sizes (collusion signature).
    ring_m = random.sample(manufacturers, 3)
    ring_w = random.sample(warehouses, 3)
    for u in ring_m:
        for v in ring_w:
            if G.has_edge(u, v):
                G[u][v]["order_freq_per_qtr"] = int(np.random.uniform(80, 120))
                G[u][v]["avg_qty"] = round(np.random.uniform(490, 510), 2)  # suspiciously uniform
                G[u][v]["is_anomaly"] = 1
                G[u][v]["anomaly_type"] = "FRAUD_RING"
            else:
                add_transaction_edge(G, u, v, anomaly_type="FRAUD_RING")
                G[u][v]["order_freq_per_qtr"] = int(np.random.uniform(80, 120))
                G[u][v]["avg_qty"] = round(np.random.uniform(490, 510), 2)
                G[u][v]["is_anomaly"] = 1
            injected.append((u, v, "FRAUD_RING"))

    # 2. GHOST_SHIPMENT: high cost, near-zero quantity (invoice fraud) on
    #    random existing supplier->manufacturer edges
    existing_sm_edges = [(u, v) for u, v in G.edges() if G.nodes[u]["tier"] == "Supplier"]
    ghost_edges = random.sample(existing_sm_edges, min(12, len(existing_sm_edges)))
    for u, v in ghost_edges:
        G[u][v]["avg_qty"] = round(np.random.uniform(0.5, 3), 2)
        G[u][v]["avg_cost"] = round(np.random.uniform(15000, 40000), 2)
        G[u][v]["is_anomaly"] = 1
        G[u][v]["anomaly_type"] = "GHOST_SHIPMENT"
        injected.append((u, v, "GHOST_SHIPMENT"))

    # 3. ROUTE_HIJACK: create edges that skip tiers or cross regions with
    #    no logical reason (e.g. a Retailer directly to a Supplier,
    #    or Warehouse to Warehouse across distant regions)
    hijack_pairs = []
    for _ in range(10):
        pattern = random.choice(["retailer_to_supplier", "cross_warehouse", "manufacturer_skip"])
        if pattern == "retailer_to_supplier":
            u, v = random.choice(retailers), random.choice(suppliers)
        elif pattern == "cross_warehouse":
            u, v = random.sample(warehouses, 2)
        else:
            u, v = random.choice(suppliers), random.choice(warehouses)  # skips manufacturer tier
        hijack_pairs.append((u, v))

    for u, v in hijack_pairs:
        add_transaction_edge(G, u, v, anomaly_type="ROUTE_HIJACK")
        G[u][v]["is_anomaly"] = 1
        G[u][v]["lead_time_days"] = round(np.random.uniform(0.1, 1.0), 2)  # implausibly fast
        injected.append((u, v, "ROUTE_HIJACK"))

    # 4. DELAY_CASCADE: pick a warehouse hub and spike delay variance
    #    across all its outgoing edges (simulates a disruption event
    #    propagating downstream)
    hub = random.choice(warehouses)
    for v in list(G.successors(hub)):
        G[hub][v]["delay_variance"] = round(np.random.uniform(6, 12), 2)
        G[hub][v]["lead_time_days"] = round(G[hub][v]["lead_time_days"] * np.random.uniform(2.5, 4), 2)
        G[hub][v]["is_anomaly"] = 1
        G[hub][v]["anomaly_type"] = "DELAY_CASCADE"
        injected.append((hub, v, "DELAY_CASCADE"))

    return injected


def graph_to_dataframes(G):
    node_rows = []
    for n, d in G.nodes(data=True):
        node_rows.append({"node_id": n, "tier": d["tier"], "name": d["name"], "region": d["region"]})
    nodes_df = pd.DataFrame(node_rows)

    edge_rows = []
    for u, v, d in G.edges(data=True):
        row = {"src": u, "dst": v}
        row.update(d)
        edge_rows.append(row)
    edges_df = pd.DataFrame(edge_rows)

    return nodes_df, edges_df


def main():
    G, tiers = build_topology()
    injected = inject_anomalies(G, tiers)

    nodes_df, edges_df = graph_to_dataframes(G)

    nodes_df.to_csv(os.path.join(SCRIPT_DIR, "nodes.csv"), index=False)
    edges_df.to_csv(os.path.join(SCRIPT_DIR, "edges.csv"), index=False)
    with open(os.path.join(SCRIPT_DIR, "graph.gpickle"), "wb") as f:
        pickle.dump(G, f)

    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"Injected anomalies: {len(injected)}")
    print(edges_df["anomaly_type"].value_counts())
    print(f"Anomaly rate: {edges_df['is_anomaly'].mean():.3%}")


if __name__ == "__main__":
    main()
