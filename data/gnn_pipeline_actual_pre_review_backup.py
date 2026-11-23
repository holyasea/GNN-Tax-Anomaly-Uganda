"""
EFRIS VAT Anomaly Detection — GNN Pipeline (Actual Data)
=========================================================
RGCN and GAT on real EFRIS transaction graph (~30K nodes, ~144K edges).
Self-supervised link prediction for both models.

Requirements:
    pip install torch torch-geometric pandas numpy scikit-learn
"""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.utils import negative_sampling

SEED = 7142
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE GRAPH DATA
# ══════════════════════════════════════════════════════════════════

def load_data(data_dir="data"):
    """Load node features and edge list from actual EFRIS data."""
    nodes = pd.read_csv(os.path.join(data_dir, "node_features_actual.csv"))
    edges = pd.read_csv(os.path.join(data_dir, "edge_list_actual.csv"))

    # All unique node IDs across edges and feature table
    all_node_ids = sorted(set(
        nodes["node_id"].tolist() +
        edges["src"].tolist() +
        edges["dst"].tolist()
    ))
    id2idx = {nid: idx for idx, nid in enumerate(all_node_ids)}
    n_nodes = len(all_node_ids)

    # Node feature matrix — only 614 nodes have features (taxpayers)
    # Counterparty nodes get zero features
    feature_cols = [
        "total_sales", "total_purchases", "output_input_vat_ratio",
        "n_unique_partners", "avg_invoice_value", "var_sale_value",
        "var_purchase_value", "purchase_to_sales_ratio",
        "n_months_active_sales", "n_months_active_purch"
    ]

    # Build full feature matrix (zeros for counterparty-only nodes)
    X_full = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
    if_scores_full = np.zeros(n_nodes, dtype=np.float32)
    is_taxpayer = np.zeros(n_nodes, dtype=bool)

    for _, row in nodes.iterrows():
        idx = id2idx[row["node_id"]]
        X_full[idx] = [row[c] for c in feature_cols]
        if_scores_full[idx] = row["if_anomaly_score"]
        is_taxpayer[idx] = True

    # Log-transform financial features to handle extreme skew
    # sign(x) * log1p(|x|) preserves sign for negative values
    log_cols = [0, 1, 4, 5, 6]  # sales, purchases, avg_invoice, var_sale, var_purchase
    for c in log_cols:
        X_full[:, c] = np.sign(X_full[:, c]) * np.log1p(np.abs(X_full[:, c]))

    # Clip extreme values
    for c in range(X_full.shape[1]):
        p1, p99 = np.percentile(X_full[:, c], [1, 99])
        X_full[:, c] = np.clip(X_full[:, c], p1, p99)

    # Standardize
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)
    # Final clip to prevent any remaining extremes
    X_scaled = np.clip(X_scaled, -5, 5)

    # Edge index and types
    # sells_to = 0, buys_from = 1
    edge_type_map = {"sells_to": 0, "buys_from": 1}

    src_indices = edges["src"].map(id2idx).values
    dst_indices = edges["dst"].map(id2idx).values
    edge_types = edges["edge_type"].map(edge_type_map).values

    # Log-transform and normalize edge amounts
    edge_amounts = edges["total_amount"].fillna(0).values
    edge_amounts_log = np.sign(edge_amounts) * np.log1p(np.abs(edge_amounts))
    edge_amounts_norm = (edge_amounts_log - np.median(edge_amounts_log)) / (
        np.percentile(edge_amounts_log, 75) - np.percentile(edge_amounts_log, 25) + 1e-8
    )
    edge_amounts_norm = np.clip(edge_amounts_norm, -5, 5)

    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src_indices, dst_indices]), dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_attr=torch.tensor(edge_amounts_norm, dtype=torch.float32).unsqueeze(1),
    )

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"  Taxpayer nodes (with features): {is_taxpayer.sum()}")
    print(f"  Counterparty-only nodes: {(~is_taxpayer).sum()}")
    print(f"  Node features: {len(feature_cols)}")
    print(f"  Relation types: 2 (sells_to, buys_from)")

    return data, all_node_ids, id2idx, is_taxpayer, if_scores_full, scaler


# ══════════════════════════════════════════════════════════════════
# 2. RGCN MODEL
# ══════════════════════════════════════════════════════════════════

class RGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, num_bases=2):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_type)
        return h


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin1 = nn.Linear(2 * emb_dim, emb_dim)
        self.lin2 = nn.Linear(emb_dim, 1)

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).squeeze()


def train_rgcn(data, hidden_dim=64, emb_dim=32, epochs=150, lr=0.005,
               mask_ratio=0.15):
    """Train RGCN with self-supervised link prediction."""
    data = data.to(DEVICE)
    num_edges = data.edge_index.size(1)

    perm = torch.randperm(num_edges)
    n_mask = int(num_edges * mask_ratio)
    train_idx = perm[n_mask:]
    mask_idx = perm[:n_mask]

    train_edge_index = data.edge_index[:, train_idx]
    train_edge_type = data.edge_type[train_idx]
    masked_edge_index = data.edge_index[:, mask_idx]

    encoder = RGCNEncoder(
        data.x.size(1), hidden_dim, emb_dim,
        num_relations=2, num_bases=2
    ).to(DEVICE)
    predictor = LinkPredictor(emb_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )

    t_start = time.time()
    losses = []

    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()

        z = encoder(data.x, train_edge_index, train_edge_type)

        pos_src, pos_dst = masked_edge_index
        pos_score = predictor(z[pos_src], z[pos_dst])

        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=n_mask
        )
        neg_src, neg_dst = neg_edge_index
        neg_score = predictor(z[neg_src], z[neg_dst])

        loss = (
            F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) +
            F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
        )
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 25 == 0:
            print(f"  RGCN Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    train_time = time.time() - t_start

    # Compute per-node reconstruction error as anomaly score
    encoder.eval()
    predictor.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index, data.edge_type)

        node_errors = torch.zeros(data.num_nodes, device=DEVICE)
        node_counts = torch.zeros(data.num_nodes, device=DEVICE)

        src, dst = data.edge_index
        scores = torch.sigmoid(predictor(z[src], z[dst]))
        errors = (1 - scores) ** 2

        node_errors.scatter_add_(0, src, errors)
        node_errors.scatter_add_(0, dst, errors)
        node_counts.scatter_add_(0, src, torch.ones_like(errors))
        node_counts.scatter_add_(0, dst, torch.ones_like(errors))

        anomaly_scores = (node_errors / (node_counts + 1e-8)).cpu().numpy()
        embeddings = z.cpu().numpy()

    print(f"  RGCN training time: {train_time:.1f}s")
    print(f"  Memory: {torch.cuda.max_memory_allocated()/1e6:.0f}MB" if DEVICE.type == 'cuda' else "  CPU mode")

    return anomaly_scores, embeddings, losses, train_time


# ══════════════════════════════════════════════════════════════════
# 3. GAT MODEL
# ══════════════════════════════════════════════════════════════════

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h


def train_gat(data, hidden_dim=32, emb_dim=32, heads=4, epochs=150, lr=0.005):
    """Train GAT via link prediction, then cluster embeddings."""
    data = data.to(DEVICE)

    encoder = GATEncoder(
        data.x.size(1), hidden_dim, emb_dim, heads=heads
    ).to(DEVICE)
    predictor = LinkPredictor(emb_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )

    t_start = time.time()
    losses = []

    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()

        z = encoder(data.x, data.edge_index)

        # Sample subset of positive edges per epoch
        n_pos = min(data.edge_index.size(1) // 4, 20000)
        perm = torch.randperm(data.edge_index.size(1))[:n_pos]
        pos_edge = data.edge_index[:, perm]
        neg_edge = negative_sampling(
            data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=n_pos
        )

        pos_score = predictor(z[pos_edge[0]], z[pos_edge[1]])
        neg_score = predictor(z[neg_edge[0]], z[neg_edge[1]])

        loss = (
            F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) +
            F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        )

        if torch.isnan(loss):
            print(f"  GAT Epoch {epoch+1}: NaN loss detected, skipping update")
            losses.append(float("nan"))
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
        )
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 25 == 0:
            print(f"  GAT Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    train_time = time.time() - t_start

    # Extract embeddings
    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index)
        embeddings = z.cpu().numpy()

    # Extract attention weights
    with torch.no_grad():
        h1 = encoder.conv1(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        h1 = F.elu(h1)
        _, (attn_edge_index, attn_weights) = encoder.conv2(
            h1, data.edge_index.to(DEVICE), return_attention_weights=True
        )
        attn_weights_np = attn_weights.cpu().numpy()

    # Handle NaN embeddings (replace with 0)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # K-means clustering
    best_sil = -1
    best_k = 5
    for k in [3, 5, 7, 10]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=100)
        labels = km.fit_predict(embeddings)
        # Sample for silhouette on large data
        sample_idx = np.random.choice(len(embeddings), min(5000, len(embeddings)), replace=False)
        sil = silhouette_score(embeddings[sample_idx], labels[sample_idx])
        print(f"  k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
            best_km = km

    print(f"  Best k={best_k} (silhouette={best_sil:.4f})")

    # Anomaly score = distance to nearest cluster center (normalized)
    distances = best_km.transform(embeddings)
    min_distances = distances.min(axis=1)
    gat_scores = (min_distances - min_distances.min()) / (
        min_distances.max() - min_distances.min() + 1e-8
    )

    print(f"  GAT training time: {train_time:.1f}s")
    print(f"  Cluster sizes: {np.bincount(best_labels)}")

    return gat_scores, embeddings, best_labels, attn_weights_np, losses, train_time


# ══════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EFRIS VAT Anomaly Detection — GNN Pipeline (Actual Data)")
    print("=" * 60)

    data, all_node_ids, id2idx, is_taxpayer, if_scores, scaler = load_data("data")

    # ── RGCN ─────────────────────────────────────────────────────
    print("\n── Training RGCN ──")
    rgcn_scores, rgcn_emb, rgcn_losses, rgcn_time = train_rgcn(
        data, hidden_dim=64, emb_dim=32, epochs=150
    )

    # ── GAT ──────────────────────────────────────────────────────
    print("\n── Training GAT ──")
    gat_scores, gat_emb, gat_clusters, attn_weights, gat_losses, gat_time = train_gat(
        data, hidden_dim=32, emb_dim=32, heads=4, epochs=150
    )

    # ── Save results (taxpayer nodes only) ───────────────────────
    taxpayer_mask = is_taxpayer
    taxpayer_ids = [all_node_ids[i] for i in range(len(all_node_ids)) if taxpayer_mask[i]]

    results_df = pd.DataFrame({
        "node_id": taxpayer_ids,
        "if_score": if_scores[taxpayer_mask],
        "rgcn_score": rgcn_scores[taxpayer_mask],
        "gat_score": gat_scores[taxpayer_mask],
        "gat_cluster": gat_clusters[taxpayer_mask],
    })
    results_df.to_csv("data/all_model_scores_actual.csv", index=False)

    # Save all node results (including counterparties)
    all_results_df = pd.DataFrame({
        "node_id": all_node_ids,
        "is_taxpayer": is_taxpayer,
        "if_score": if_scores,
        "rgcn_score": rgcn_scores,
        "gat_score": gat_scores,
        "gat_cluster": gat_clusters,
    })
    all_results_df.to_csv("data/all_node_scores_actual.csv", index=False)

    # Save embeddings
    np.save("data/rgcn_embeddings_actual.npy", rgcn_emb)
    np.save("data/gat_embeddings_actual.npy", gat_emb)

    # Training losses
    max_len = max(len(rgcn_losses), len(gat_losses))
    loss_df = pd.DataFrame({
        "epoch": list(range(1, max_len + 1)),
        "rgcn_loss": rgcn_losses + [None] * (max_len - len(rgcn_losses)),
        "gat_loss": gat_losses + [None] * (max_len - len(gat_losses)),
    })
    loss_df.to_csv("data/training_losses_actual.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY (Taxpayer nodes only)")
    print("=" * 60)

    for name, scores in [("Isolation Forest", if_scores[taxpayer_mask]),
                          ("RGCN", rgcn_scores[taxpayer_mask]),
                          ("GAT", gat_scores[taxpayer_mask])]:
        q90 = np.percentile(scores, 90)
        q95 = np.percentile(scores, 95)
        flagged_90 = (scores >= q90).sum()
        flagged_95 = (scores >= q95).sum()
        print(f"\n{name}:")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
        print(f"  Flagged (top 10%): {flagged_90} | (top 5%): {flagged_95}")

    # Overlap analysis
    tp_if = if_scores[taxpayer_mask]
    tp_rgcn = rgcn_scores[taxpayer_mask]
    tp_gat = gat_scores[taxpayer_mask]

    top10_if = set(np.where(tp_if >= np.percentile(tp_if, 90))[0])
    top10_rgcn = set(np.where(tp_rgcn >= np.percentile(tp_rgcn, 90))[0])
    top10_gat = set(np.where(tp_gat >= np.percentile(tp_gat, 90))[0])

    print(f"\nOverlap (top 10% flagged):")
    print(f"  IF ∩ RGCN: {len(top10_if & top10_rgcn)} / {len(top10_if | top10_rgcn)}")
    print(f"  IF ∩ GAT:  {len(top10_if & top10_gat)} / {len(top10_if | top10_gat)}")
    print(f"  RGCN ∩ GAT: {len(top10_rgcn & top10_gat)} / {len(top10_rgcn | top10_gat)}")
    print(f"  All three:  {len(top10_if & top10_rgcn & top10_gat)}")

    # Timing comparison
    print(f"\nTraining time:")
    print(f"  RGCN: {rgcn_time:.1f}s")
    print(f"  GAT:  {gat_time:.1f}s")

    # Save comparison JSON
    comparison = {
        "rgcn": {"train_time": rgcn_time, "epochs": 150},
        "gat": {"train_time": gat_time, "epochs": 150},
        "graph": {"nodes": int(data.num_nodes), "edges": int(data.num_edges),
                  "taxpayer_nodes": int(is_taxpayer.sum())},
    }
    with open("data/model_comparison_actual.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nAll results saved to data/")
    return results_df


if __name__ == "__main__":
    results = main()
