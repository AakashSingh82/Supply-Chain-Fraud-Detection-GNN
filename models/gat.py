"""
Custom Graph Attention Network (GAT) — built directly on PyTorch tensors
(no external graph library dependency), so it's transparent and easy to
explain in an interview.

Two things this model does that make it more than a toy:

1. EDGE-LEVEL ANOMALY SCORING
   We don't just classify nodes — we score every EDGE (transaction/lane)
   by combining the learned representations of its two endpoint nodes
   with the edge's own raw features. This matches the real problem: an
   anomaly in supply chain data is a suspicious *transaction*, not
   necessarily a "bad" company.

2. ATTENTION AS EXPLAINABILITY
   GAT learns an attention weight for every edge during message passing.
   We expose these weights so the dashboard can show *why* the model
   flagged a lane as suspicious (i.e. it attended unusually to certain
   neighbors) — this is what elevates the project from "black box
   classifier" to "explainable AI system".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """A single-head (extendable to multi-head) graph attention layer,
    operating on a dense adjacency mask — fine for graphs up to a few
    thousand nodes, which covers our use case and keeps the code
    dependency-free and readable."""

    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2, alpha=0.2, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h:   (N, in_dim) node features
        adj: (N, N) binary adjacency (1 where edge i->j exists, plus self loops)
        returns: (N, out_dim * n_heads) if concat else averaged (N, out_dim)
                 also returns attention tensor (n_heads, N, N) for explainability
        """
        N = h.size(0)
        # (n_heads, N, out_dim)
        Wh = torch.einsum("ni,hio->hno", h, self.W)

        # attention scores e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j)
        src_scores = torch.einsum("hno,ho->hn", Wh, self.a_src)  # (heads, N)
        dst_scores = torch.einsum("hno,ho->hn", Wh, self.a_dst)  # (heads, N)
        e = src_scores.unsqueeze(2) + dst_scores.unsqueeze(1)     # (heads, N, N)
        e = self.leakyrelu(e)

        mask = (adj == 0).unsqueeze(0)
        e = e.masked_fill(mask, float("-1e9"))
        attn = F.softmax(e, dim=-1)          # (heads, N, N)
        attn = self.dropout(attn)

        h_prime = torch.einsum("hnm,hmo->hno", attn, Wh)  # (heads, N, out_dim)

        if self.concat:
            out = h_prime.permute(1, 0, 2).reshape(N, -1)  # (N, heads*out_dim)
            out = F.elu(out)
        else:
            out = h_prime.mean(dim=0)  # (N, out_dim)

        return out, attn


class SupplyChainGAT(nn.Module):
    """Two-layer GAT that produces node embeddings, plus an edge-scoring
    MLP head that combines endpoint embeddings + raw edge features to
    output an anomaly probability per edge."""

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=32, n_heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_in_dim, hidden_dim, n_heads=n_heads,
                                         dropout=dropout, concat=True)
        self.gat2 = GraphAttentionLayer(hidden_dim * n_heads, hidden_dim, n_heads=1,
                                         dropout=dropout, concat=False)

        # Edge classifier: [h_u || h_v || edge_features] -> anomaly logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, node_feats, adj, edge_index, edge_feats):
        """
        node_feats: (N, node_in_dim)
        adj:        (N, N) binary adjacency w/ self loops
        edge_index: (E, 2) long tensor of (src, dst) for edges we want to SCORE
                     (this can be a superset including negative-sampled non-edges)
        edge_feats: (E, edge_in_dim) raw features for each scored edge
        """
        h, attn1 = self.gat1(node_feats, adj)
        h, attn2 = self.gat2(h, adj)  # (N, hidden_dim), final node embeddings

        src, dst = edge_index[:, 0], edge_index[:, 1]
        h_src = h[src]
        h_dst = h[dst]
        edge_input = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        logits = self.edge_mlp(edge_input).squeeze(-1)

        return logits, h, (attn1, attn2)
