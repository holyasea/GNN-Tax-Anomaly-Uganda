"""
EFRIS VAT Anomaly Detection — GNN Pipeline (Actual Data), rev. 2
=================================================================
RGCN and GAT on the real EFRIS transaction graph (~165K nodes).
Self-supervised link prediction for both models, identical protocols.

Changes in rev. 2 (post supervisor-review):
  * Node-feature clipping/scaling computed on taxpayer rows only
    (the previous full-matrix percentile clip zeroed every feature).
  * Structural features (log in/out-degree, PageRank) added for ALL nodes,
    so counterparty nodes are no longer all-zero feature vectors.
  * Edge masking removes reciprocal edges from the training graph
    (inverse-relation leakage); negatives are filtered against both
    directions and never coincide with held-out edges.
  * GAT trained under the same 85/15 edge mask as RGCN; both models
    scored by link-reconstruction error on HELD-OUT edges (fallback to
    training edges when a node has no held-out incident edge; recorded).
  * Early stopping on held-out validation loss (patience 20).
  * 3 seeds per model; scores averaged across seeds; pairwise Spearman
    rank stability reported (data/seed_stability.csv).
  * Score-vs-degree correlations exported (data/score_degree_correlation.csv).
  * Edge amounts (edge_attr) are NOT consumed by RGCNConv/GATConv and are
    therefore no longer attached to the Data object (dead code removed).

Requirements:
    pip install torch torch-geometric pandas numpy scikit-learn scipy
"""

import os
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.utils import negative_sampling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [7142, 2718, 3141]          # seed 7142 keeps continuity with prior runs
MASK_RATIO = 0.15                   # edges held out for link prediction
EPOCHS = 150
PATIENCE = 20


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE GRAPH DATA
# ══════════════════════════════════════════════════════════════════

def pagerank(src, dst, n_nodes, damping=0.85, iters=50):
    """Power-iteration PageRank on the directed aggregate graph."""
    out_deg = np.bincount(src, minlength=n_nodes).astype(np.float64)
    r = np.full(n_nodes, 1.0 / n_nodes)
    for _ in range(iters):
        flow = np.zeros(n_nodes)
        # every edge source has out_deg > 0 by construction; nodes with no
        # outgoing edges contribute their mass as dangling mass
        np.add.at(flow, dst, r[src] / out_deg[src])
        dangling = r[out_deg == 0].sum()
        r = damping * flow + (1 - damping + damping * dangling) / n_nodes
    return r


def load_data(data_dir="data"):
    """Load node features and edge list; build the model input matrix."""
    nodes = pd.read_csv(os.path.join(data_dir, "node_features_actual.csv"))
    edges = pd.read_csv(os.path.join(data_dir, "edge_list_actual.csv"))

    all_node_ids = sorted(set(
        nodes["node_id"].tolist() +
        edges["src"].tolist() +
        edges["dst"].tolist()
    ))
    id2idx = {nid: idx for idx, nid in enumerate(all_node_ids)}
    n_nodes = len(all_node_ids)

    fin_cols = [
        "total_sales", "total_purchases", "output_input_vat_ratio",
        "n_unique_partners", "avg_invoice_value", "var_sale_value",
        "var_purchase_value", "purchase_to_sales_ratio",
        "n_months_active_sales", "n_months_active_purch"
    ]
    is_taxpayer = np.zeros(n_nodes, dtype=bool)
    X_fin = np.zeros((n_nodes, len(fin_cols)), dtype=np.float64)
    if_scores_full = np.zeros(n_nodes, dtype=np.float32)

    for row in nodes.to_dict("records"):
        idx = id2idx[row["node_id"]]
        X_fin[idx] = [row[c] for c in fin_cols]
        if_scores_full[idx] = row["if_anomaly_score"]
        is_taxpayer[idx] = True

    # --- Financial features: transform TAXPAYER ROWS ONLY ---
    # sign(x)*log1p(|x|) preserves sign for negative (credit-note) sums.
    log_cols = [0, 1, 4, 5, 6]  # sales, purchases, avg_invoice, var_sale, var_purchase
    for c in log_cols:
        X_fin[is_taxpayer, c] = np.sign(X_fin[is_taxpayer, c]) * np.log1p(np.abs(X_fin[is_taxpayer, c]))
    # Clip extremes within the taxpayer distribution only (the previous
    # full-matrix clip removed the very tails an anomaly detector targets).
    for c in range(X_fin.shape[1]):
        vals = X_fin[is_taxpayer, c]
        p1, p99 = np.percentile(vals, [1, 99])
        X_fin[is_taxpayer, c] = np.clip(vals, p1, p99)
    scaler = RobustScaler().fit(X_fin[is_taxpayer])
    X_fin = scaler.transform(X_fin)
    X_fin[~is_taxpayer] = 0.0          # counterparties carry no financial data

    # --- Structural features for ALL nodes ---
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    out_deg = np.bincount(src, minlength=n_nodes).astype(np.float64)
    in_deg = np.bincount(dst, minlength=n_nodes).astype(np.float64)
    pr = pagerank(src, dst, n_nodes)

    X_struct = np.column_stack([
        np.log1p(out_deg), np.log1p(in_deg), np.log1p(pr * 1e6)
    ]).astype(np.float64)
    struct_scaler = RobustScaler().fit(X_struct)   # all rows have real values
    X_struct = struct_scaler.transform(X_struct)

    X_all = np.hstack([X_fin, X_struct]).astype(np.float32)
    X_all = np.clip(X_all, -5, 5)

    feature_names = fin_cols + ["log_out_degree", "log_in_degree", "log_pagerank"]

    edge_type_map = {"sells_to": 0, "buys_from": 1}
    edge_types = edges["edge_type"].map(edge_type_map).to_numpy()

    data = Data(
        x=torch.tensor(X_all, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
    )

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"  Taxpayer nodes (financial features): {is_taxpayer.sum()}")
    print(f"  Counterparty-only nodes: {(~is_taxpayer).sum()}")
    print(f"  Node features: {len(feature_names)} (10 financial + 3 structural)")

    return data, all_node_ids, id2idx, is_taxpayer, if_scores_full, feature_names


# ══════════════════════════════════════════════════════════════════
# 2. EDGE MASKING WITH INVERSE-RELATION LEAKAGE CONTROL
# ══════════════════════════════════════════════════════════════════

def edge_key(src_arr, dst_arr, n_nodes):
    return src_arr.astype(np.int64) * n_nodes + dst_arr.astype(np.int64)


def build_split(data, seed):
    """Hold out MASK_RATIO of edges; remove reciprocal edges from training."""
    set_seed(seed)
    n_nodes = data.num_nodes
    perm = torch.randperm(data.num_edges)
    n_hold = int(data.num_edges * MASK_RATIO)
    hold_idx = perm[:n_hold]

    hold_keys = edge_key(data.edge_index[0, hold_idx].numpy(),
                         data.edge_index[1, hold_idx].numpy(), n_nodes)
    hold_keys = np.union1d(hold_keys, hold_keys - n_nodes)  # add reverse direction

    # Training set: everything else, minus edges whose reciprocal is held out
    rest = perm[n_hold:]
    rest_keys = edge_key(data.edge_index[0, rest].numpy(),
                         data.edge_index[1, rest].numpy(), n_nodes)
    keep = ~(np.isin(rest_keys, hold_keys) |
             np.isin(rest_keys - n_nodes, hold_keys) |
             np.isin(rest_keys + n_nodes, hold_keys))
    train_idx = rest[torch.from_numpy(keep)]
    print(f"  split: {len(train_idx)} train / {len(hold_idx)} held-out edges "
          f"({len(rest) - len(train_idx)} dropped as reciprocal-of-held-out)")
    return train_idx, hold_idx


def sample_negatives(data_full, n_samples, seed):
    """Negatives from the FULL graph (held-out edges never become negatives),
    filtered so no negative is the reverse of a real edge."""
    set_seed(seed)
    neg = negative_sampling(
        data_full.edge_index, num_nodes=data_full.num_nodes,
        num_neg_samples=int(n_samples * 1.2)
    )
    n_nodes = data_full.num_nodes
    keys = edge_key(neg[0].numpy(), neg[1].numpy(), n_nodes)
    real_keys = edge_key(data_full.edge_index[0].numpy(),
                         data_full.edge_index[1].numpy(), n_nodes)
    real_rev = np.sort(real_keys - n_nodes)
    real_fwd = np.sort(real_keys + n_nodes)
    bad = np.isin(keys - n_nodes, real_rev) | np.isin(keys + n_nodes, real_fwd)
    neg = neg[:, torch.from_numpy(~bad)][:, :n_samples]
    return neg


# ══════════════════════════════════════════════════════════════════
# 3. MODELS
# ══════════════════════════════════════════════════════════════════

class RGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations=2, num_bases=2):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = self.dropout(h)
        return self.conv2(h, edge_index, edge_type)


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
        h = F.elu(self.conv1(x, edge_index))
        h = self.dropout(h)
        return self.conv2(h, edge_index)


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin1 = nn.Linear(2 * emb_dim, emb_dim)
        self.lin2 = nn.Linear(emb_dim, 1)

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).squeeze()


def bce_loss(predictor, z, pos_slice, edge_index, neg_edge_index):
    pos_score = predictor(z[edge_index[0, pos_slice]], z[edge_index[1, pos_slice]])
    neg_score = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
    return (F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score)) +
            F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)))


def train_link_model(model_name, encoder, predictor, data,
                     train_idx, hold_idx, seed):
    """Shared training loop — identical protocol for RGCN and GAT."""
    data = data.to(DEVICE)
    train_edge_index = data.edge_index[:, train_idx]
    train_edge_type = data.edge_type[train_idx]
    neg_train = sample_negatives(data, min(len(train_idx) // 4, 20000), seed).to(DEVICE)
    neg_val = sample_negatives(data, len(hold_idx), seed + 1).to(DEVICE)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=1e-5)

    def encode():
        if model_name == "RGCN":
            return encoder(data.x, train_edge_index, train_edge_type)
        return encoder(data.x, train_edge_index)

    losses, val_losses = [], []
    best_val, best_state, bad_epochs = np.inf, None, 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        encoder.train(); predictor.train()
        optimizer.zero_grad()
        z = encode()
        pos_idx = torch.randperm(train_edge_index.size(1),
                                 device=train_edge_index.device)[:20000]
        loss = bce_loss(predictor, z, pos_idx, train_edge_index, neg_train)
        if torch.isnan(loss):
            losses.append(np.nan)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        encoder.eval(); predictor.eval()
        with torch.no_grad():
            z_eval = encode()
            val_loss = bce_loss(predictor, z_eval,
                                torch.arange(hold_idx.size(0), device=data.edge_index.device),
                                data.edge_index, neg_val).item()
        val_losses.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in
                          list(encoder.state_dict().items()) +
                          list(predictor.state_dict().items())}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  {model_name} early stop at epoch {epoch+1} "
                      f"(best val loss {best_val:.4f})")
                break
        if (epoch + 1) % 25 == 0:
            print(f"  {model_name} Epoch {epoch+1}/{EPOCHS} | "
                  f"loss {loss.item():.4f} | val {val_loss:.4f}")

    if best_state is not None:
        enc_sd = encoder.state_dict()
        pred_sd = predictor.state_dict()
        encoder.load_state_dict({k: best_state[k] for k in enc_sd})
        predictor.load_state_dict({k: best_state[k] for k in pred_sd})

    return losses, val_losses, time.time() - t0


def node_link_scores(z, predictor, edge_index, idx, num_nodes):
    """Mean squared reconstruction error over each node's incident edges."""
    with torch.no_grad():
        s = torch.sigmoid(predictor(z[edge_index[0, idx]], z[edge_index[1, idx]]))
        err = (1 - s) ** 2
        node_err = torch.zeros(num_nodes, device=z.device)
        counts = torch.zeros(num_nodes, device=z.device)
        src, dst = edge_index[0, idx], edge_index[1, idx]
        node_err.scatter_add_(0, src, err)
        node_err.scatter_add_(0, dst, err)
        counts.scatter_add_(0, src, torch.ones_like(err))
        counts.scatter_add_(0, dst, torch.ones_like(err))
    return (node_err / counts.clamp(min=1)).cpu().numpy(), counts.cpu().numpy()


def run_model(model_name, data, seed):
    """Train one model under one seed; scores from held-out edges."""
    data = data.to(DEVICE)
    set_seed(seed)
    train_idx, hold_idx = build_split(data, seed)

    if model_name == "RGCN":
        encoder = RGCNEncoder(data.x.size(1), 64, 32).to(DEVICE)
    else:
        encoder = GATEncoder(data.x.size(1), 32, 32, heads=4).to(DEVICE)
    predictor = LinkPredictor(32).to(DEVICE)

    losses, val_losses, train_time = train_link_model(
        model_name, encoder, predictor, data, train_idx, hold_idx, seed)

    # Embeddings computed from the TRAINING graph only (held-out edges unseen)
    encoder.eval(); predictor.eval()
    with torch.no_grad():
        if model_name == "RGCN":
            z = encoder(data.x, data.edge_index[:, train_idx],
                        data.edge_type[train_idx])
        else:
            z = encoder(data.x, data.edge_index[:, train_idx])

    # Primary score: reconstruction error on HELD-OUT incident edges.
    hold_scores, hold_counts = node_link_scores(
        z, predictor, data.edge_index, hold_idx, data.num_nodes)
    # Fallback: nodes with no held-out incident edge use training-edge error.
    train_scores, _ = node_link_scores(
        z, predictor, data.edge_index, train_idx, data.num_nodes)
    scores = np.where(hold_counts > 0, hold_scores, train_scores)
    n_fallback = int((hold_counts == 0).sum())

    embeddings = np.nan_to_num(z.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {model_name} seed {seed}: train {train_time:.1f}s | "
          f"fallback nodes: {n_fallback} ({100*n_fallback/data.num_nodes:.1f}%)")
    return scores, embeddings, losses, val_losses, train_time


# ══════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EFRIS VAT Anomaly Detection — GNN Pipeline (rev. 2)")
    print("=" * 60)
    print(f"Device: {DEVICE} | seeds: {SEEDS} | mask: {MASK_RATIO}")

    data, all_node_ids, id2idx, is_taxpayer, if_scores, feat_names = load_data("data")
    n_nodes = data.num_nodes

    out_deg = np.bincount(data.edge_index[0].numpy(), minlength=n_nodes)
    in_deg = np.bincount(data.edge_index[1].numpy(), minlength=n_nodes)

    rgcn_seed_scores, gat_seed_scores = [], []
    rgcn_losses_s1 = gat_losses_s1 = None
    embeddings, clusters = {}, None

    for seed in SEEDS:
        print(f"\n── Seed {seed} ──")
        rg = run_model("RGCN", data, seed)
        ga = run_model("GAT", data, seed)
        rgcn_seed_scores.append(rg[0])
        gat_seed_scores.append(ga[0])
        if seed == SEEDS[0]:
            rgcn_losses_s1, gat_losses_s1 = rg[2], ga[2]
            embeddings = {"rgcn": rg[1], "gat": ga[1]}
            # Descriptive clustering on seed-1 GAT embeddings
            best_sil, best_k, best_labels = -1, 5, None
            for k in [3, 5, 7, 10]:
                km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=100)
                labels = km.fit_predict(embeddings["gat"])
                sidx = np.random.RandomState(seed).choice(
                    len(labels), min(5000, len(labels)), replace=False)
                sil = silhouette_score(embeddings["gat"][sidx], labels[sidx])
                print(f"  k={k}: silhouette={sil:.4f}")
                if sil > best_sil:
                    best_sil, best_k, best_labels = sil, k, labels
            print(f"  GAT clustering: best k={best_k} (silhouette={best_sil:.4f})")
            clusters = best_labels

    # Seed-averaged scores
    rgcn_scores = np.mean(rgcn_seed_scores, axis=0)
    gat_scores = np.mean(gat_seed_scores, axis=0)

    # --- Seed stability (supervisor requirement) ---
    stab_rows = []
    for name, per_seed in [("rgcn", rgcn_seed_scores), ("gat", gat_seed_scores)]:
        for i in range(len(per_seed)):
            for j in range(i + 1, len(per_seed)):
                stab_rows.append({
                    "model": name, "seed_a": SEEDS[i], "seed_b": SEEDS[j],
                    "spearman_all_nodes": spearmanr(per_seed[i], per_seed[j]).correlation,
                    "spearman_taxpayers": spearmanr(
                        per_seed[i][is_taxpayer], per_seed[j][is_taxpayer]).correlation,
                })
    pd.DataFrame(stab_rows).to_csv("data/seed_stability.csv", index=False)
    print("\nSeed stability (Spearman across seeds):")
    print(pd.DataFrame(stab_rows).to_string(index=False))

    # --- Score vs degree diagnostics (supervisor request) ---
    deg_rows = []
    for name, sc in [("if", if_scores), ("rgcn", rgcn_scores), ("gat", gat_scores)]:
        for scope, mask in [("all_nodes", np.ones(n_nodes, bool)),
                            ("taxpayers", is_taxpayer)]:
            deg_rows.append({
                "model": name, "scope": scope,
                "spearman_log_total_degree": spearmanr(
                    np.log1p(out_deg + in_deg)[mask], sc[mask]).correlation,
                "spearman_log_in_degree": spearmanr(
                    np.log1p(in_deg)[mask], sc[mask]).correlation,
                "spearman_log_out_degree": spearmanr(
                    np.log1p(out_deg)[mask], sc[mask]).correlation,
            })
    pd.DataFrame(deg_rows).to_csv("data/score_degree_correlation.csv", index=False)
    print("\nScore-degree correlations:")
    print(pd.DataFrame(deg_rows).to_string(index=False))

    # --- Save results (same filenames; downstream R scripts unchanged) ---
    tm = is_taxpayer
    taxpayer_ids = [all_node_ids[i] for i in range(n_nodes) if tm[i]]
    pd.DataFrame({
        "node_id": taxpayer_ids,
        "if_score": if_scores[tm],
        "rgcn_score": rgcn_scores[tm],
        "gat_score": gat_scores[tm],
        "gat_cluster": clusters[tm],
    }).to_csv("data/all_model_scores_actual.csv", index=False)

    pd.DataFrame({
        "node_id": all_node_ids, "is_taxpayer": is_taxpayer,
        "if_score": if_scores, "rgcn_score": rgcn_scores, "gat_score": gat_scores,
        "gat_cluster": clusters,
    }).to_csv("data/all_node_scores_actual.csv", index=False)

    np.save("data/rgcn_embeddings_actual.npy", embeddings["rgcn"])
    np.save("data/gat_embeddings_actual.npy", embeddings["gat"])

    max_len = max(len(rgcn_losses_s1), len(gat_losses_s1))
    pd.DataFrame({
        "epoch": list(range(1, max_len + 1)),
        "rgcn_loss": rgcn_losses_s1 + [None] * (max_len - len(rgcn_losses_s1)),
        "gat_loss": gat_losses_s1 + [None] * (max_len - len(gat_losses_s1)),
    }).to_csv("data/training_losses_actual.csv", index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (taxpayer nodes only, seed-averaged scores)")
    print("=" * 60)
    for name, sc in [("Isolation Forest", if_scores[tm]),
                     ("RGCN", rgcn_scores[tm]), ("GAT", gat_scores[tm])]:
        q90, q95 = np.percentile(sc, [90, 95])
        print(f"\n{name}: range [{sc.min():.4f}, {sc.max():.4f}], "
              f"mean {sc.mean():.4f}, sd {sc.std():.4f}")
        print(f"  top 10%: {(sc >= q90).sum()} | top 5%: {(sc >= q95).sum()}")

    top10 = {n: set(np.where(s[tm] >= np.percentile(s[tm], 90))[0])
             for n, s in [("if", if_scores), ("rgcn", rgcn_scores), ("gat", gat_scores)]}
    print("\nTop-10% overlap:")
    print(f"  IF & RGCN:  {len(top10['if'] & top10['rgcn'])}")
    print(f"  IF & GAT:   {len(top10['if'] & top10['gat'])}")
    print(f"  RGCN & GAT: {len(top10['rgcn'] & top10['gat'])}")
    print(f"  All three:  {len(top10['if'] & top10['rgcn'] & top10['gat'])}")

    with open("data/model_comparison_actual.json", "w") as f:
        json.dump({
            "seeds": SEEDS, "mask_ratio": MASK_RATIO,
            "epochs_max": EPOCHS, "patience": PATIENCE,
            "graph": {"nodes": int(n_nodes), "edges": int(data.num_edges),
                      "taxpayer_nodes": int(is_taxpayer.sum())},
            "features": feat_names,
        }, f, indent=2)

    print("\nAll results saved to data/")
    return None


if __name__ == "__main__":
    main()
