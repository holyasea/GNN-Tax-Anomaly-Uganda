"""
EFRIS VAT Anomaly Detection — Synthetic-Injection Benchmark (rev. 2)
=====================================================================
BOND-style evaluation: inject known anomaly patterns into the real EFRIS
transaction graph, run every detector on the SAME augmented graph, and
report recall@k / precision@k / average precision against the known
ground truth. Neutral verdict design: no detector is privileged; the
numbers decide.

Injection patterns (ground truth = injected NEW nodes):
  ring            k new nodes in a closed cycle of high-value invoices
  shell_pair      pairs of new nodes round-tripping large invoices
  missing_trader  new node with large purchases, NO sales (VAT sink)

Detectors (all ranked on the same universe: 614 taxpayers + injected):
  D1_rule_q50/q90/1B  sales == 0 & purchases > tau (tau sensitivity)
  D2_if_fin           Isolation Forest on 10 financial features
  D3_if_fin_graph     D2 + structural features (log degrees, PageRank,
                      reciprocity, counterparty concentration)
  D4_scc_cycle        strongly-connected-component size (cycle baseline)
  D5_if_rgcn          D3 + 32 RGCN embedding dimensions
  D6_if_gat           D3 + 32 GAT embedding dimensions
  D7_*_link_err       GNN link-reconstruction error (caveat documented:
                      injected edges are in the training graph)

Seeds: GNN_SEEDS x INJ_SEEDS; tabular detectors use a fixed seed so that
only the GNN component varies across GNN seeds. Tabular detectors are
evaluated once per injection instance (they do not depend on GNN seed).

Requirements: torch, torch-geometric, sklearn, scipy, pandas, numpy.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_pipeline_actual import load_data, run_model   # rev. 2 pipeline

INJ_SEEDS = [101, 202]
GNN_SEEDS = [7142, 2718, 3141]
TAB_SEED = 7142                      # IF randomness fixed across GNN seeds
PATTERNS = ["ring", "shell_pair", "missing_trader"]
K_RING = 10
N_SHELL_PAIRS = 5
N_MISSING = 10


def set_seed(seed):
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════
# 1. INJECTION (unchanged from rev. 1)
# ══════════════════════════════════════════════════════════════════

def agg_edge_row(src, dst, total, n_inv, edge_type):
    return {
        "src": src, "dst": dst, "total_amount": total,
        "total_vat": 0.18 * total, "n_invoices": n_inv,
        "avg_amount": total / n_inv, "sd_amount": total / (10 * n_inv),
        "n_months_active": 6, "edge_type": edge_type,
    }


def node_feature_row(nid, sales_amt, sales_inv, purch_amt, purch_inv,
                     partners_out, partners_in, months_s, months_p, isic=4669):
    var_s = np.var([sales_amt / sales_inv] * 2) if sales_inv > 1 else 0.0
    var_p = np.var([purch_amt / purch_inv] * 2) if purch_inv > 1 else 0.0
    n_inv_total = max(sales_inv + purch_inv, 1)
    return {
        "node_id": nid,
        "total_sales": sales_amt, "total_purchases": purch_amt,
        "output_input_vat_ratio": (0.18 * sales_amt) / (0.18 * purch_amt + 1),
        "n_unique_partners": len(set(partners_out) | set(partners_in)),
        "avg_invoice_value": (sales_amt + purch_amt) / n_inv_total,
        "var_sale_value": var_s, "var_purchase_value": var_p,
        "purchase_to_sales_ratio": purch_amt / (sales_amt + 1),
        "n_months_active_sales": months_s, "n_months_active_purch": months_p,
        "if_anomaly_score": 0.0,
    }


def inject_pattern(edges, feats_raw, taxpayer_ids, start_id, pattern, rng):
    new_edges, new_feats, injected = [], [], []
    nid = start_id

    def add_node():
        nonlocal nid
        new_feats.append(node_feature_row(nid, 0, 0, 0, 0, [], [], 0, 0))
        injected.append(nid)
        nid += 1
        return nid - 1

    def feat(nid_):
        return next(r for r in new_feats if r["node_id"] == nid_)

    if pattern == "ring":
        ids = [add_node() for _ in range(K_RING)]
        for i in range(K_RING):
            a = float(rng.lognormal(mean=20.0, sigma=0.5))
            new_edges.append(agg_edge_row(ids[i], ids[(i + 1) % K_RING], a, 3, "sells_to"))
        for v in ids:
            inflow = sum(e["total_amount"] for e in new_edges if e["dst"] == v)
            outflow = sum(e["total_amount"] for e in new_edges if e["src"] == v)
            f = feat(v)
            f.update(total_sales=inflow, total_purchases=outflow,
                     n_unique_partners=2, n_months_active_sales=6,
                     n_months_active_purch=6,
                     avg_invoice_value=(inflow + outflow) / 6,
                     output_input_vat_ratio=inflow / (outflow + 1),
                     purchase_to_sales_ratio=outflow / (inflow + 1),
                     var_sale_value=np.var([inflow / 3] * 2),
                     var_purchase_value=np.var([outflow / 3] * 2))

    elif pattern == "shell_pair":
        for _ in range(N_SHELL_PAIRS):
            a = float(rng.lognormal(mean=20.0, sigma=0.4))
            u, v = add_node(), add_node()
            new_edges.append(agg_edge_row(u, v, a, 4, "sells_to"))
            new_edges.append(agg_edge_row(v, u, a, 4, "sells_to"))
            for node in (u, v):
                f = feat(node)
                f.update(total_sales=a, total_purchases=a, n_unique_partners=1,
                         n_months_active_sales=6, n_months_active_purch=6,
                         avg_invoice_value=a / 4,
                         output_input_vat_ratio=a / (a + 1),
                         purchase_to_sales_ratio=a / (a + 1),
                         var_sale_value=np.var([a / 4] * 2),
                         var_purchase_value=np.var([a / 4] * 2))

    elif pattern == "missing_trader":
        for _ in range(N_MISSING):
            suppliers = rng.choice(taxpayer_ids, size=3, replace=False)
            purch = float(rng.lognormal(mean=19.5, sigma=0.4))
            v = add_node()
            for s in suppliers:
                new_edges.append(agg_edge_row(s, v, purch / 3, 1, "sells_to"))
            f = feat(v)
            f.update(total_sales=0.0, total_purchases=purch,
                     n_unique_partners=3, n_months_active_purch=6,
                     avg_invoice_value=purch / 3,
                     output_input_vat_ratio=0.0,
                     purchase_to_sales_ratio=purch,
                     var_sale_value=0.0,
                     var_purchase_value=np.var([purch / 3] * 2))
    else:
        raise ValueError(pattern)

    return new_edges, new_feats, injected


# ══════════════════════════════════════════════════════════════════
# 2. STRUCTURAL EXTRAS + SCC
# ══════════════════════════════════════════════════════════════════

def graph_extras(u_src, u_dst, total, n_nodes, universe_idx):
    A = csr_matrix((np.ones(len(u_src)), (u_src, u_dst)), shape=(n_nodes, n_nodes))
    recip_node = np.asarray(A.multiply(A.T).sum(axis=1)).ravel()
    out_w = csr_matrix((np.abs(total), (u_src, u_dst)), shape=(n_nodes, n_nodes))
    out_tot = np.asarray(out_w.sum(axis=1)).ravel()
    out_max = np.asarray(out_w.max(axis=1).todense()).ravel()
    in_w = csr_matrix((np.abs(total), (u_dst, u_src)), shape=(n_nodes, n_nodes))
    in_tot = np.asarray(in_w.sum(axis=1)).ravel()
    in_max = np.asarray(in_w.max(axis=1).todense()).ravel()

    rows = []
    for nid in sorted(universe_idx.keys()):
        i = universe_idx[nid]
        rows.append({"node_id": nid,
                     "reciprocity": recip_node[i] / max(out_tot[i], 1e-9),
                     "max_out_share": out_max[i] / max(out_tot[i], 1e-9),
                     "max_in_share": in_max[i] / max(in_tot[i], 1e-9)})
    return pd.DataFrame(rows).set_index("node_id")


def scc_sizes(src_idx, dst_idx, n_nodes):
    A = csr_matrix((np.ones(len(src_idx)), (src_idx, dst_idx)),
                   shape=(n_nodes, n_nodes))
    ncomp, labels = connected_components(A, directed=True, connection="strong")
    return np.bincount(labels, minlength=ncomp)[labels].astype(np.float64)


# ══════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate(name, scores, universe_ids, injected_set, k, gnn_seed, inj_seed, pattern):
    df = pd.DataFrame({"node_id": universe_ids, "score": np.asarray(scores, dtype=float)})
    df["y"] = df["node_id"].isin(injected_set).astype(int)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    ap = average_precision_score(df["y"], df["score"]) if df["y"].sum() else np.nan
    ranks = df.index[df["y"] == 1] + 1
    return {
        "pattern": pattern, "inj_seed": inj_seed, "gnn_seed": gnn_seed,
        "detector": name, "k": k, "n_injected": int(df["y"].sum()),
        "recall_at_k": float(df.head(k)["y"].mean()),
        "precision_at_k": float(df.head(k)["y"].mean()),
        "average_precision": float(ap),
        "best_rank_injected": int(ranks.min()) if len(ranks) else -1,
        "worst_rank_injected": int(ranks.max()) if len(ranks) else -1,
    }


def run_if(X, seed):
    iso = IsolationForest(n_estimators=500, random_state=seed, n_jobs=-1)
    iso.fit(X)
    return -iso.decision_function(X)


# ══════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Injection benchmark rev.2 "
          f"({len(PATTERNS)} patterns x {len(INJ_SEEDS)} inj seeds "
          f"x {len(GNN_SEEDS)} GNN seeds)")
    print("=" * 60)

    os.makedirs("data/injection", exist_ok=True)
    results, gt_rows = [], []

    for pattern in PATTERNS:
        for inj_seed in INJ_SEEDS:
            tag = f"{pattern}_s{inj_seed}"
            print(f"\n── Scenario {tag} ──")
            rng = np.random.default_rng(inj_seed)

            feats_raw = pd.read_csv("data/node_features_actual.csv")
            edges = pd.read_csv("data/edge_list_actual.csv")
            taxpayer_ids = feats_raw["node_id"].tolist()
            start_id = int(max(edges["src"].max(), edges["dst"].max(),
                               feats_raw["node_id"].max()) + 1)

            new_edges, new_feats, injected = inject_pattern(
                edges, feats_raw, taxpayer_ids, start_id, pattern, rng)
            edges_i = pd.concat([edges, pd.DataFrame(new_edges)], ignore_index=True)
            feats_i = pd.concat([feats_raw, pd.DataFrame(new_feats)], ignore_index=True)

            scene_dir = f"data/injection/{tag}"
            os.makedirs(scene_dir, exist_ok=True)
            edges_i.to_csv(f"{scene_dir}/edge_list_actual.csv", index=False)
            feats_i.to_csv(f"{scene_dir}/node_features_actual.csv", index=False)

            data, all_ids, id2idx, is_tax, _, feat_names = load_data(scene_dir)
            n_nodes = data.num_nodes
            universe_ids = taxpayer_ids + injected
            universe_idx = {nid: id2idx[nid] for nid in universe_ids}
            injected_set = set(injected)
            k = len(injected)
            for nid in injected:
                gt_rows.append({"pattern": pattern, "inj_seed": inj_seed,
                                "node_id": nid, "role": pattern})

            uidx = np.array([id2idx[nid] for nid in universe_ids])
            raw = feats_i.set_index("node_id").loc[universe_ids]

            # ---- D1 rule with tau sensitivity ----
            purch = raw["total_purchases"].to_numpy()
            sales = raw["total_sales"].to_numpy()
            pos_purch = purch[purch > 0]
            taus = {
                "D1_rule_q50": float(np.median(pos_purch)),
                "D1_rule_q90": float(np.percentile(pos_purch, 90)),
                "D1_rule_1B": 1e9,
            }

            # ---- Tabular detectors (GNN-seed independent) ----
            X_data = data.x.numpy()
            X_fin = X_data[uidx][:, :10]
            e_src = edges_i["src"].map(id2idx).to_numpy()
            e_dst = edges_i["dst"].map(id2idx).to_numpy()
            extras = graph_extras(e_src, e_dst,
                                  edges_i["total_amount"].to_numpy(),
                                  n_nodes, universe_idx)
            X_graph = np.hstack([
                X_data[uidx],
                RobustScaler().fit_transform(
                    extras[["reciprocity", "max_out_share", "max_in_share"]].to_numpy())])

            scc_sc = np.log1p(scc_sizes(
                data.edge_index[0].numpy(), data.edge_index[1].numpy(), n_nodes))[uidx]

            tabular = {f"D1_{nm}": np.where((sales == 0) & (purch > tv), purch, -1.0)
                       for nm, tv in taus.items()}
            tabular["D2_if_fin"] = run_if(X_fin, TAB_SEED)
            tabular["D3_if_fin_graph"] = run_if(X_graph, TAB_SEED)
            tabular["D4_scc_cycle"] = scc_sc

            for name, sc in tabular.items():
                results.append(evaluate(name, sc, universe_ids, injected_set,
                                        k, gnn_seed=0, inj_seed=inj_seed, pattern=pattern))

            # ---- GNN detectors across seeds ----
            for gnn_seed in GNN_SEEDS:
                rg = run_model("RGCN", data, gnn_seed)
                ga = run_model("GAT", data, gnn_seed)
                E_rgcn = RobustScaler().fit_transform(rg[1][uidx])
                E_gat = RobustScaler().fit_transform(ga[1][uidx])
                results.append(evaluate("D5_if_rgcn", run_if(np.hstack([X_graph, E_rgcn]), TAB_SEED),
                                        universe_ids, injected_set, k, gnn_seed, inj_seed, pattern))
                results.append(evaluate("D6_if_gat", run_if(np.hstack([X_graph, E_gat]), TAB_SEED),
                                        universe_ids, injected_set, k, gnn_seed, inj_seed, pattern))
                results.append(evaluate("D7_rgcn_link_err", rg[0][uidx],
                                        universe_ids, injected_set, k, gnn_seed, inj_seed, pattern))
                results.append(evaluate("D7_gat_link_err", ga[0][uidx],
                                        universe_ids, injected_set, k, gnn_seed, inj_seed, pattern))

            # running summary for this scenario
            df = pd.DataFrame([r for r in results if r["pattern"] == pattern
                               and r["inj_seed"] == inj_seed])
            print(df.groupby("detector")["recall_at_k"].mean().round(3).to_string())

    out = pd.DataFrame(results)
    out.to_csv("data/injection_benchmark_results.csv", index=False)
    pd.DataFrame(gt_rows).to_csv("data/injection/ground_truth.csv", index=False)

    print("\n" + "=" * 60)
    print("RECALL@k: mean (sd) by detector and pattern")
    print("=" * 60)
    g = out.groupby(["detector", "pattern"])["recall_at_k"]
    summary = g.agg(["mean", "std", "count"]).unstack("pattern").round(3)
    print(summary.to_string())
    summary.to_csv("data/injection_benchmark_summary.csv")

    print("\nSaved: injection_benchmark_results.csv, "
          "injection_benchmark_summary.csv, injection/ground_truth.csv")


if __name__ == "__main__":
    main()
