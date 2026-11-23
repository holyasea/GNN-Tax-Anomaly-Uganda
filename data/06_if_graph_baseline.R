# ============================================================
#  06_if_graph_baseline.R
#  Real-data analog of detector D3 from the injection benchmark:
#  Isolation Forest trained on financial features PLUS hand-built
#  graph features, to test whether the GNN embeddings add
#  information beyond simple graph statistics on the actual data.
#
#  Inputs : data/node_features_actual.csv   (614 taxpayers)
#           data/edge_list_actual.csv       (directed, both directions)
#           data/all_model_scores_actual.csv (IF / RGCN / GAT scores)
#  Outputs: data/if_graph_baseline_flags.csv
#           data/if_graph_baseline_summary.txt
#
#  Seed: 7142. IF settings: isotree, ntrees = 200, sample_size = 256
#  (within the range script 01 selected), top 10% flagged.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(igraph)
  library(isotree)
})

set.seed(7142)

nodes  <- fread("data/node_features_actual.csv")
edges  <- fread("data/edge_list_actual.csv")
scores <- fread("data/all_model_scores_actual.csv")

N   <- nrow(scores)
TOP <- ceiling(0.10 * N)
rk  <- function(x) rank(-x, ties.method = "first") <= TOP

cat(sprintf("Taxpayers: %d | directed edges: %s\n", N, format(nrow(edges), big.mark = ",")))

# ------------------------------------------------------------
# Directed graph over all nodes (taxpayers + counterparties)
# ------------------------------------------------------------
g <- graph_from_data_frame(edges[, .(src, dst, total_amount)],
                           directed = TRUE)

# Per-node graph features
out_w  <- strength(g, mode = "out")
in_w   <- strength(g, mode = "in")
pr     <- page_rank(g, damping = 0.85)$vector

# Reciprocity (value share): fraction of out-weight going to nodes
# that also send weight back. Computed from the edge list directly.
el <- as.data.table(edges)
out_tot <- el[, .(out_w_sum = sum(total_amount)), by = src]
el2 <- merge(el, out_tot, by = "src")
# mark edges whose reverse (dst -> src) also exists
rev_exist <- el[, .(has_rev = TRUE), by = .(src = dst, dst = src)]
el2 <- merge(el2, rev_exist, by = c("src", "dst"), all.x = TRUE)
el2[, has_rev := fifelse(is.na(has_rev), FALSE, has_rev)]
recip <- el2[, .(recip_w = sum(total_amount * has_rev)), by = src]
el2[, has_rev := NULL]
recip <- merge(recip, out_tot, by = "src")
recip[, reciprocated_share := recip_w / pmax(out_w_sum, 1)]

# Largest-counterparty concentration (share of out-weight)
max_cp <- el2[, .(max_share = max(total_amount) / pmax(out_w_sum[1], 1)), by = src]

out_deg <- degree(g, mode = "out")
in_deg  <- degree(g, mode = "in")
gf <- data.table(
  node_id = as.integer(V(g)$name),
  log_out_degree = sign(out_deg) * log1p(out_deg),
  log_in_degree  = sign(in_deg)  * log1p(in_deg),
  log_out_weight = sign(out_w) * log1p(abs(out_w)),
  log_in_weight  = sign(in_w)  * log1p(abs(in_w)),
  page_rank      = pr
)
gf <- merge(gf, recip[, .(node_id = src, reciprocated_share)], by = "node_id", all.x = TRUE)
gf <- merge(gf, max_cp[, .(node_id = src, max_counterparty_share = max_share)], by = "node_id", all.x = TRUE)
gf[is.na(reciprocated_share), reciprocated_share := 0]
gf[is.na(max_counterparty_share), max_counterparty_share := 0]

# ------------------------------------------------------------
# Feature sets
# ------------------------------------------------------------
fin_feats   <- c("total_sales", "total_purchases", "output_input_vat_ratio",
                 "purchase_to_sales_ratio", "n_unique_partners")
graph_feats <- c("log_out_degree", "log_in_degree", "log_out_weight",
                 "log_in_weight", "page_rank", "reciprocated_share",
                 "max_counterparty_share")

# Sign-preserving log transform for heavy-tailed financial columns
X_fin <- copy(nodes[, fin_feats, with = FALSE])
for (cl in c("total_sales", "total_purchases", "n_unique_partners")) {
  X_fin[[cl]] <- sign(X_fin[[cl]]) * log1p(abs(X_fin[[cl]]))
}
X_fin_graph <- cbind(X_fin, nodes[, .(node_id)],
                     gf[match(nodes$node_id, node_id), graph_feats, with = FALSE])
X_fin_graph[, node_id := NULL]

cat("Fitting IF-financial (baseline reproduction)...\n")
m_fin <- isolation.forest(X_fin, ntrees = 200, sample_size = 256, seed = 7142)
s_fin <- predict(m_fin, X_fin)

cat("Fitting IF-financial+graph (D3 analog)...\n")
m_fg <- isolation.forest(X_fin_graph, ntrees = 200, sample_size = 256, seed = 7142)
s_fg <- predict(m_fg, X_fin_graph)

res <- data.table(
  node_id = nodes$node_id,
  if_fin_score = s_fin,
  if_fin_graph_score = s_fg,
  flag_if_fin = rk(s_fin),
  flag_if_fin_graph = rk(s_fg)
)

# Model flags from the GNN pipeline
res[, flag_if_orig   := rk(scores$if_score)]
res[, flag_rgcn      := rk(scores$rgcn_score)]
res[, flag_gat       := rk(scores$gat_score)]
res[, consensus_gnn  := flag_if_orig + flag_rgcn + flag_gat >= 2]
res[, in_top10_gnn_any := flag_rgcn | flag_gat]

ov <- function(a, b) sum(a & b)
cat("\n===== IF + hand-built graph features vs GNN flags =====\n")
cat(sprintf("Consensus cases (n = %d) recovered by IF-fin:        %d\n",
            sum(res$consensus_gnn), ov(res$flag_if_fin, res$consensus_gnn)))
cat(sprintf("Consensus cases (n = %d) recovered by IF-fin+graph: %d\n",
            sum(res$consensus_gnn), ov(res$flag_if_fin_graph, res$consensus_gnn)))
cat(sprintf("GNN-flagged (RGCN|GAT, n = %d) recovered by IF-fin:        %d\n",
            sum(res$in_top10_gnn_any), ov(res$flag_if_fin, res$in_top10_gnn_any)))
cat(sprintf("GNN-flagged (RGCN|GAT, n = %d) recovered by IF-fin+graph:  %d\n",
            sum(res$in_top10_gnn_any), ov(res$flag_if_fin_graph, res$in_top10_gnn_any)))

# Permutation test: is IF-fin+graph's overlap with the GNN-flagged set
# higher than a random top-k set of the same size would achieve?
B <- 10000
set.seed(7142)
perm_any <- vapply(seq_len(B), function(i)
  sum(rk(sample(s_fg)) & res$in_top10_gnn_any), numeric(1))
p_any <- (1 + sum(perm_any >= ov(res$flag_if_fin_graph, res$in_top10_gnn_any))) / (B + 1)
perm_cons <- vapply(seq_len(B), function(i)
  sum(rk(sample(s_fg)) & res$consensus_gnn), numeric(1))
p_cons <- (1 + sum(perm_cons >= ov(res$flag_if_fin_graph, res$consensus_gnn))) / (B + 1)

cat(sprintf("\nPermutation test (10,000 draws), IF-fin+graph vs GNN-flagged: p = %.4f\n", p_any))
cat(sprintf("Permutation test (10,000 draws), IF-fin+graph vs consensus:   p = %.4f\n", p_cons))

fwrite(res, "data/if_graph_baseline_flags.csv")

txt <- c(
  "IF + hand-built graph features (real-data analog of injection detector D3)",
  "============================================================",
  sprintf("Taxpayers: %d. Flagged top 10%%: n = %d per detector.", N, TOP),
  "IF settings: isotree, ntrees = 200, sample_size = 256, seed 7142.",
  "Graph features: log in/out-degree, log in/out-weight, PageRank,",
  "reciprocated value share, largest-counterparty share.",
  "",
  sprintf("Consensus cases (GNN pipeline, n = %d): recovered by IF-fin = %d; by IF-fin+graph = %d.",
          sum(res$consensus_gnn), ov(res$flag_if_fin, res$consensus_gnn),
          ov(res$flag_if_fin_graph, res$consensus_gnn)),
  sprintf("GNN-flagged (RGCN or GAT, n = %d): recovered by IF-fin = %d; by IF-fin+graph = %d.",
          sum(res$in_top10_gnn_any), ov(res$flag_if_fin, res$in_top10_gnn_any),
          ov(res$flag_if_fin_graph, res$in_top10_gnn_any)),
  sprintf("Permutation p (IF-fin+graph overlap vs GNN-flagged): %.4f.", p_any),
  sprintf("Permutation p (IF-fin+graph overlap vs consensus):   %.4f.", p_cons),
  "",
  "Reading: if IF-fin+graph recovers most GNN-flagged cases, the GNN embeddings add",
  "little beyond simple graph statistics on the real data; a large gap supports the",
  "claim that learned embeddings capture structure the hand-built features miss."
)
writeLines(txt, "data/if_graph_baseline_summary.txt")
cat("\nOutputs written: data/if_graph_baseline_flags.csv, data/if_graph_baseline_summary.txt\n")
