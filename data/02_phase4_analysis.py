"""
PHASE 4: Enhanced Analysis for Objective 4 (Comparative Analysis)
=================================================================

This script strengthens Objective 4 by:
  1. GAT attention analysis: Embedding isolation & anomaly types
  2. RGCN relation-specific error: Decompose by relation type

Output files:
  - gat_analysis.csv (embedding isolation for 20 top flagged nodes)
  - rgcn_analysis.csv (reconstruction error by relation type)

Objective 4 strengthening: Explains WHY each model flagged nodes
  - IF: Feature magnitude extremity
  - RGCN: Relational edge pattern anomalies
  - GAT: Behavioral embedding isolation
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

print("\n=== PHASE 4: Enhanced Analysis (Objective 4) ===\n")

# Load data
node_feats = pd.read_csv("data/node_features_actual.csv")
edge_list = pd.read_csv("data/edge_list_actual.csv")
scores = pd.read_csv("data/all_model_scores_actual.csv")

# Load embeddings
gat_embeddings = np.load("data/gat_embeddings_actual.npy")
rgcn_embeddings = np.load("data/rgcn_embeddings_actual.npy")

# Get top flagged nodes
consensus_case_cards = pd.read_csv("data/case_cards_consensus_flagged.csv")
top_flagged_ids = consensus_case_cards['node_id'].head(20).tolist()

print(f"Analyzing {len(top_flagged_ids)} top flagged nodes\n")

# Build node mapping
all_nodes = sorted(set(
    node_feats['node_id'].tolist() +
    edge_list['src'].tolist() +
    edge_list['dst'].tolist()
))
node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}

# ──────────────────────────────────────────────────────────────────────────
# GAT ANALYSIS: Embedding Isolation
# ──────────────────────────────────────────────────────────────────────────

print("Computing GAT embedding isolation...")

gat_analysis = []

for node_id in top_flagged_ids:
    if node_id not in node_to_idx:
        continue
    
    node_idx = node_to_idx[node_id]
    node_embedding = gat_embeddings[node_idx]
    
    # Get neighbors (all connected nodes)
    neighbors = set()
    neighbors.update(edge_list[edge_list['src'] == node_id]['dst'].tolist())
    neighbors.update(edge_list[edge_list['dst'] == node_id]['src'].tolist())
    
    if not neighbors:
        continue
    
    neighbor_indices = [node_to_idx[n] for n in neighbors if n in node_to_idx]
    if not neighbor_indices:
        continue
    
    neighbor_embeddings = gat_embeddings[neighbor_indices]
    
    # Compute distances
    distances = [euclidean(node_embedding, ne) for ne in neighbor_embeddings]
    
    # Get case info
    case_row = consensus_case_cards[consensus_case_cards['node_id'] == node_id]
    if len(case_row) > 0:
        case_row = case_row.iloc[0]
        sales = case_row['total_sales'] / 1e9
        purchases = case_row['total_purchases'] / 1e9
    else:
        sales = 0
        purchases = 0
    
    gat_analysis.append({
        'node_id': node_id,
        'sales_B_UGX': sales,
        'purchases_B_UGX': purchases,
        'n_neighbors': len(neighbors),
        'embedding_isolation_score': np.mean(distances),
        'is_purchase_only': 1 if sales == 0 and purchases > 0 else 0,
        'anomaly_type': 'Input_VAT_Fraud' if (sales == 0 and purchases > 0) 
                       else 'Behavioral_Isolation'
    })

gat_df = pd.DataFrame(gat_analysis)
gat_df.to_csv('data/gat_analysis.csv', index=False)
print(f"✓ gat_analysis.csv ({len(gat_df)} nodes)\n")

# ──────────────────────────────────────────────────────────────────────────
# RGCN ANALYSIS: Relation-Specific Error
# ──────────────────────────────────────────────────────────────────────────

print("Computing RGCN relation-specific errors...")

rgcn_analysis = []

for node_id in top_flagged_ids:
    if node_id not in node_to_idx:
        continue
    
    node_idx = node_to_idx[node_id]
    node_embedding = rgcn_embeddings[node_idx]
    
    # Get edges by type
    sells_to = edge_list[(edge_list['src'] == node_id) & 
                         (edge_list['edge_type'] == 'sells_to')]
    buys_from = edge_list[(edge_list['dst'] == node_id) & 
                          (edge_list['edge_type'] == 'buys_from')]
    
    # Compute reconstruction errors
    sells_errors = []
    for _, row in sells_to.iterrows():
        neighbor_id = row['dst']
        if neighbor_id in node_to_idx:
            neighbor_idx = node_to_idx[neighbor_id]
            neighbor_embedding = rgcn_embeddings[neighbor_idx]
            error = euclidean(node_embedding, neighbor_embedding)
            sells_errors.append(error)
    
    buys_errors = []
    for _, row in buys_from.iterrows():
        neighbor_id = row['src']
        if neighbor_id in node_to_idx:
            neighbor_idx = node_to_idx[neighbor_id]
            neighbor_embedding = rgcn_embeddings[neighbor_idx]
            error = euclidean(node_embedding, neighbor_embedding)
            buys_errors.append(error)
    
    sells_mean = np.mean(sells_errors) if sells_errors else np.nan
    buys_mean = np.mean(buys_errors) if buys_errors else np.nan
    
    # Determine primary anomaly
    if np.isnan(sells_mean) and not np.isnan(buys_mean):
        primary = 'BUYS_FROM'
    elif np.isnan(buys_mean) and not np.isnan(sells_mean):
        primary = 'SELLS_TO'
    elif not np.isnan(sells_mean) and not np.isnan(buys_mean):
        primary = 'SELLS_TO' if sells_mean > buys_mean else 'BUYS_FROM'
    else:
        primary = 'NONE'
    
    rgcn_analysis.append({
        'node_id': node_id,
        'n_sells_to_edges': len(sells_to),
        'n_buys_from_edges': len(buys_from),
        'sells_to_error': sells_mean,
        'buys_from_error': buys_mean,
        'primary_anomaly': primary
    })

rgcn_df = pd.DataFrame(rgcn_analysis)
rgcn_df.to_csv('data/rgcn_analysis.csv', index=False)
print(f"✓ rgcn_analysis.csv ({len(rgcn_df)} nodes)\n")

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────

print("OBJECTIVE 4 ENHANCEMENT SUMMARY:\n")
print(f"GAT Analysis:")
print(f"  Embedding isolation range: {gat_df['embedding_isolation_score'].min():.2f} - {gat_df['embedding_isolation_score'].max():.2f}")
print(f"  Input VAT Fraud cases: {sum(gat_df['is_purchase_only'])}")
print(f"  Purchase-only (zero sales): {sum(gat_df['is_purchase_only'])} of {len(gat_df)}")

print(f"\nRGCN Analysis:")
print(f"  Selling anomalies (SELLS_TO): {sum(rgcn_df['primary_anomaly'] == 'SELLS_TO')}")
print(f"  Buying anomalies (BUYS_FROM): {sum(rgcn_df['primary_anomaly'] == 'BUYS_FROM')}")

print(f"\nModel Complementarity:")
print(f"  Each model detects different anomaly mechanisms")
print(f"  IF: Feature magnitude extremity")
print(f"  RGCN: Relational edge pattern anomalies")
print(f"  GAT: Behavioral embedding isolation")
print(f"  → Consensus (2+ models) = high-confidence fraud signals\n")

