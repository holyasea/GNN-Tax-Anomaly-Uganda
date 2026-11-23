# ══════════════════════════════════════════════════════════════════════════════
# 07_regenerate_plots.R — regenerate all 13 report figures
# ══════════════════════════════════════════════════════════════════════════════
# Regenerates every figure in data/Plots/ from the rev-2/rev-3 outputs
# (35-consensus era), keeping the original filenames so report references
# stay valid.
#
# Inputs:
#   all_model_scores_actual.csv   if/rgcn/gat scores, 614 taxpayers
#   node_features_actual.csv      10 financial features + if_anomaly_score
#   edge_list_actual.csv          src, dst, total_amount, ..., edge_type
#   hypothesis_tests/H1_network_topology.csv  modularity + power-law fit
#   training_losses_actual.csv    rgcn_loss, gat_loss per epoch (seed 7142)
#
# Conventions mirrored from the pipeline:
#   - IF feature matrix: 10 financial features, NA->0, Inf->0 (script 01, s.5.1)
#   - IF grid search: ntrees {100,300,500} x sample_size {64,128,256} x ndim {1,2},
#     best = highest score variance (script 01, s.5.2-5.3), seed 7142
#   - IF convergence sweep: tree counts c(10,25,50,100,200,300,500), Spearman
#     vs the final best-params model (script 01, s.5.7)
#   - H1 graph: simplify(graph_from_data_frame(src, dst, directed = FALSE))
#   - Flags: top 10% per model (rank-based, ties by first)
#
# NOTE (report captions): "Feature Importance" plot is a descriptive
# flagged-vs-normal z-profile, NOT model-based importance. The power-law line
# is descriptive; the bootstrap KS test rejects a pure power law (H1).
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(igraph); library(isotree)
  library(patchwork); library(ggraph)
})

data_dir <- "C:/Users/Holyweek/Desktop/Project/Project1/data"
plot_dir <- file.path(data_dir, "Plots")

scores <- fread(file.path(data_dir, "all_model_scores_actual.csv"))
feats  <- fread(file.path(data_dir, "node_features_actual.csv"))
# fread mis-parses the two variance columns as character; coerce to numeric
for (cl in names(feats)) {
  if (is.character(feats[[cl]])) {
    feats[[cl]] <- suppressWarnings(as.numeric(feats[[cl]]))
  }
}
edges  <- fread(file.path(data_dir, "edge_list_actual.csv"))
h1     <- fread(file.path(data_dir, "hypothesis_tests/H1_network_topology.csv"))
losses <- fread(file.path(data_dir, "training_losses_actual.csv"))

d <- merge(feats, scores[, .(node_id, if_score, rgcn_score, gat_score)], by = "node_id")
stopifnot(nrow(d) == nrow(scores))
cat(sprintf("Taxpayers: %d | if_score vs if_anomaly_score: Spearman = %.4f, max |diff| = %.6f\n",
            nrow(d), cor(d$if_score, d$if_anomaly_score, method = "spearman"),
            max(abs(d$if_score - d$if_anomaly_score))))

# ─── Flags and groups (top 10% per model) ────────────────────────────────────
TOP <- ceiling(0.10 * nrow(d))
rk  <- function(x) rank(-x, ties.method = "first") <= TOP
d[, flag_if := rk(if_score)]
d[, flag_rg := rk(rgcn_score)]
d[, flag_ga := rk(gat_score)]
d[, n_flags := flag_if + flag_rg + flag_ga]
d[, group := fcase(
  n_flags == 0, "Non-flagged",
  n_flags >= 2, "Consensus",
  flag_if,      "IF only",
  flag_rg,      "R-GCN only",
  default =     "GAT only"
)]
d[, group := factor(group, c("Non-flagged", "IF only", "R-GCN only",
                             "GAT only", "Consensus"))]

cat(sprintf("Flag protocol: top-%d per model\n", TOP))
print(d[, .N, by = group][order(group)])
cat(sprintf("Consensus (2+ models): %d | all three: %d | 2-without-3: %d\n",
            sum(d$n_flags >= 2), sum(d$n_flags == 3),
            sum(d$n_flags == 2)))

pal   <- c("Non-flagged" = "grey70", "IF only" = "#1f77b4",
           "R-GCN only" = "#d62728", "GAT only" = "#2ca02c",
           "Consensus" = "#7570b3")
theme_set(theme_minimal(base_size = 13))
savep <- function(p, name, w = 9, h = 6) {
  # cairo-type png: independent of RStudio's plot device, which produced blank
  # files and "Graphics API version mismatch" errors in this session (ragg too).
  path <- file.path(plot_dir, name)
  png(path, width = w, height = h, units = "in", res = 300, type = "cairo")
  print(p)
  dev.off()
  cat("saved:", name, sprintf(" (%.0f KB)\n", file.size(path) / 1024))
}

# ─── Graphs ──────────────────────────────────────────────────────────────────
gu <- simplify(graph_from_data_frame(edges[, .(src, dst)], directed = FALSE))
gd <- graph_from_data_frame(edges[, .(src, dst, edge_type)], directed = TRUE)
cat(sprintf("Graph: %d nodes, %d undirected edges, %d directed edges\n",
            vcount(gu), ecount(gu), ecount(gd)))

# ─── Communities (H1 protocol: unweighted Louvain on simplified graph) ──────
set.seed(7142)   # Louvain is stochastic; same seeding as script 04's H1 run
com  <- cluster_louvain(gu)
mod  <- modularity(com)
# sizes() returns a table object here; table(membership()) is safer
csz <- data.table(size = as.integer(table(membership(com))))
setorder(csz, -size)
csz[, comm_rank := seq_len(.N)]
h1v  <- function(k) h1[metric == k, value]
cat(sprintf("Louvain: %d communities, modularity %.4f (H1 CSV: %.4f)\n",
            nrow(csz), mod, h1v("modularity_observed")))

p9 <- ggplot(csz, aes(x = comm_rank, y = size)) +
  geom_col(fill = "#1f77b4", alpha = 0.85) +
  scale_y_log10() +
  labs(title = "Community Size Distribution",
       subtitle = sprintf("%d Louvain communities; modularity Q = %.4f vs %.4f under\nconfiguration-model nulls (z = %.1f, p = 0.001)",
                          nrow(csz), mod, h1v("modularity_null_mean"), h1v("modularity_z")),
       x = "Community (ranked by size)", y = "Nodes (log scale)") +
  theme(panel.grid.minor = element_blank())
savep(p9, "Community Size Analysis.png")

# ─── Degree distribution with power-law fit line (descriptive) ──────────────
dv      <- degree(gu)
deg_tab <- data.table(degree = as.integer(names(table(dv))),
                      count  = as.integer(table(dv)))
alpha   <- h1v("powerlaw_alpha")
xmin    <- h1v("powerlaw_xmin")
Ntot    <- h1v("graph_nodes")
kk      <- seq(xmin, max(deg_tab$degree))
Z       <- sum(kk^-alpha)          # normalising sum, xmin = 1
fit_df  <- data.table(degree = kk, expected = Ntot * kk^-alpha / Z)

p10 <- ggplot(deg_tab, aes(x = degree, y = count)) +
  geom_point(size = 1.3, alpha = 0.6, color = "grey35") +
  geom_line(data = fit_df, aes(x = degree, y = expected),
            color = "#d62728", linewidth = 0.8) +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Degree Distribution",
       subtitle = sprintf("Heavy-tailed: fitted power law alpha = %.2f (xmin = %d, red line).\nBootstrap KS test rejects a pure power law (p < 0.001) - tail is broader than power-law in the mid-range",
                          alpha, xmin),
       x = "Degree (trading partners)", y = "Number of nodes") +
  theme(panel.grid.minor = element_blank())
savep(p10, "Count Distribution.png")

# ─── IF: replicate script 01's feature matrix + grid search ─────────────────
feature_cols <- c("total_sales", "total_purchases", "output_input_vat_ratio",
                  "n_unique_partners", "avg_invoice_value", "var_sale_value",
                  "var_purchase_value", "purchase_to_sales_ratio",
                  "n_months_active_sales", "n_months_active_purch")
tf <- as.data.frame(d)
for (col in feature_cols) tf[[col]][is.na(tf[[col]])] <- 0
tf$output_input_vat_ratio[is.infinite(tf$output_input_vat_ratio)] <- 0
tf$purchase_to_sales_ratio[is.infinite(tf$purchase_to_sales_ratio)] <- 0
X <- tf[, feature_cols]

param_grid <- expand.grid(ntrees = c(100, 300, 500),
                          sample_size = c(64, 128, 256),
                          ndim = c(1, 2))
gdf <- do.call(rbind, lapply(seq_len(nrow(param_grid)), function(i) {
  p <- param_grid[i, ]
  m <- isolation.forest(X, ntrees = p$ntrees,
                        sample_size = min(p$sample_size, nrow(X)),
                        ndim = p$ndim, seed = 7142)
  data.frame(ntrees = p$ntrees, sample_size = p$sample_size, ndim = p$ndim,
             sd_score = sd(predict(m, X)))
}))
best <- gdf[which.max(gdf$sd_score), ]
cat(sprintf("Best IF params (by score variance): ntrees=%d, sample_size=%d, ndim=%d\n",
            best$ntrees, best$sample_size, best$ndim))

iforest_final <- isolation.forest(X, ntrees = best$ntrees,
                                  sample_size = min(best$sample_size, nrow(X)),
                                  ndim = best$ndim, seed = 7142)
final_scores_if <- predict(iforest_final, X)
cat(sprintf("Reproduced IF scores vs saved if_score: Spearman = %.4f\n",
            cor(final_scores_if, d$if_score, method = "spearman")))

# ─── IF convergence sweep (script 01 s.5.7 protocol) ─────────────────────────
tree_counts <- c(10, 25, 50, 100, 200, 300, 500)
stab <- do.call(rbind, lapply(tree_counts, function(nt) {
  m <- isolation.forest(X, ntrees = nt,
                        sample_size = min(best$sample_size, nrow(X)),
                        ndim = best$ndim, seed = 7142)
  s <- predict(m, X)
  data.frame(ntrees = nt, mean_score = mean(s), sd_score = sd(s),
             rank_cor = cor(s, final_scores_if, method = "spearman"))
}))
fwrite(stab, file.path(data_dir, "if_convergence_actual.csv"))
cat("IF convergence:\n"); print(stab)

p6a <- ggplot(stab, aes(x = ntrees, y = rank_cor)) +
  geom_line(linewidth = 0.8, color = "#1f77b4") +
  geom_point(size = 2.5, color = "#1f77b4") +
  coord_cartesian(ylim = c(0.85, 1.0)) +
  labs(title = "Score Stability vs Number of Trees",
       subtitle = sprintf("Spearman rank correlation with the final %d-tree model", best$ntrees),
       x = "Number of trees", y = "Spearman rank correlation") +
  theme(panel.grid.minor = element_blank())
p6b <- ggplot(stab, aes(x = ntrees, y = sd_score)) +
  geom_line(linewidth = 0.8, color = "coral") +
  geom_point(size = 2.5, color = "coral") +
  labs(title = "Score Spread vs Number of Trees",
       subtitle = "Score variance (discrimination) plateaus by ~100 trees",
       x = "Number of trees", y = "Score standard deviation") +
  theme(panel.grid.minor = element_blank())
p6 <- p6a + p6b + plot_annotation(
  title = "Isolation Forest Convergence Analysis",
  subtitle = sprintf("ntrees = %d, sample_size = %d, ndim = %d, seed = 7142",
                     best$ntrees, best$sample_size, best$ndim))
savep(p6, "Isolation Forest Convergence Analysis.png", w = 10, h = 5)

# ─── IF score distribution ───────────────────────────────────────────────────
q90 <- quantile(d$if_score, 0.9)
p5 <- ggplot(d, aes(x = if_score)) +
  geom_histogram(bins = 50, fill = "#1f77b4", alpha = 0.75) +
  geom_vline(xintercept = q90, linetype = "dashed", color = "#d62728") +
  annotate("text", x = q90, y = Inf, vjust = 2, hjust = -0.05,
           label = sprintf("Top 10%% threshold = %.3f", q90),
           color = "#d62728", size = 3.5) +
  labs(title = "Isolation Forest Anomaly Score Distribution",
       subtitle = sprintf("Right-skewed | ntrees = %d, sample_size = %d, ndim = %d | %d taxpayers flagged (top 10%%)",
                          best$ntrees, best$sample_size, best$ndim,
                          sum(d$if_score >= q90)),
       x = "Anomaly score", y = "Taxpayers") +
  theme(panel.grid.minor = element_blank())
savep(p5, "Isolation Forest Anomaly Score Distribution.png")

# ─── IF z-profile (flagged vs normal; descriptive, not importance) ──────────
flagged <- d[flag_if == TRUE]
normal  <- d[flag_if == FALSE]
zprof <- sapply(feature_cols, function(f) {
  x_f <- sign(flagged[[f]]) * log1p(abs(flagged[[f]]))
  x_n <- sign(normal[[f]]) * log1p(abs(normal[[f]]))
  mean(abs(x_f - mean(x_n)) / (sd(x_n) + 1e-6))
})
zp <- data.table(feature = names(zprof), avg_abs_z = as.numeric(zprof))
zp[, feature := factor(feature, levels = feature[order(avg_abs_z)])]
p7 <- ggplot(zp, aes(x = avg_abs_z, y = feature)) +
  geom_col(fill = "#1f77b4", alpha = 0.85) +
  labs(title = "Flagged vs Non-flagged Taxpayers: Feature Deviation Profile",
       subtitle = "Average |z-score| of flagged taxpayers relative to non-flagged taxpayers,\ncomputed on sign-preserving log-transformed features (tames heavy tails).\nDescriptive profile - NOT model-based feature importance",
       x = "Average |z-score| deviation", y = NULL) +
  theme(panel.grid.minor = element_blank())
savep(p7, "Isolation Forest Feature Importance for Flagged Taxpayers.png")

# ─── IF feature space: PCA of the 10 financial features ─────────────────────
Xp <- copy(X)
for (cl in c("total_sales", "total_purchases", "n_unique_partners",
             "avg_invoice_value", "var_sale_value", "var_purchase_value")) {
  Xp[[cl]] <- sign(Xp[[cl]]) * log1p(abs(Xp[[cl]]))
}
pcs   <- prcomp(Xp, center = TRUE, scale. = TRUE)
ve    <- summary(pcs)$importance[2, 1:2] * 100
pca_df <- data.table(PC1 = pcs$x[, 1], PC2 = pcs$x[, 2],
                     if_score = d$if_score, flagged = d$flag_if)
p8 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = if_score)) +
  geom_point(size = 1.9, alpha = 0.75) +
  geom_point(data = pca_df[flagged == TRUE],
             aes(x = PC1, y = PC2), shape = 21, fill = NA,
             color = "#d62728", size = 2.8, stroke = 0.8) +
  scale_color_viridis_c(option = "magma", name = "IF score") +
  labs(title = "Isolation Forest Feature Space (PCA)",
       subtitle = sprintf("10 financial features, sign-preserving log transform, scaled. Circles = flagged (top 10%%)"),
       x = sprintf("PC1 (%.1f%% variance)", ve[1]),
       y = sprintf("PC2 (%.1f%% variance)", ve[2])) +
  theme(panel.grid.minor = element_blank())
savep(p8, "Isolation Forest Feature Space.png")

# ─── GNN training loss curves (seed 7142) ────────────────────────────────────
last_ep  <- function(col) max(which(!is.na(losses[[col]])))
last_val <- function(col) losses[[col]][last_ep(col)]
p1 <- ggplot(losses, aes(x = epoch, y = rgcn_loss)) +
  geom_line(color = "#d62728", linewidth = 0.8) +
  labs(title = "R-GCN",
       subtitle = sprintf("Early stop at epoch %d (patience 20); final loss %.4f",
                          last_ep("rgcn_loss"), last_val("rgcn_loss")),
       x = "Epoch", y = "Mean link-reconstruction loss") +
  theme(panel.grid.minor = element_blank())
p2 <- ggplot(losses, aes(x = epoch, y = gat_loss)) +
  geom_line(color = "#2ca02c", linewidth = 0.8) +
  labs(title = "GAT",
       subtitle = sprintf("Early stop at epoch %d (patience 20); final loss %.4f",
                          last_ep("gat_loss"), last_val("gat_loss")),
       x = "Epoch", y = "Mean link-reconstruction loss") +
  theme(panel.grid.minor = element_blank())
p_gnn <- p1 + p2 + plot_annotation(
  title = "GNN Training Loss Curves (seed 7142)",
  subtitle = "Different loss scales per model - panels are not comparable in absolute value")
savep(p_gnn, "Model Convergence GAT and RGCN.png", w = 10, h = 5)

# ─── Score distributions (all three models) ─────────────────────────────────
sc_long <- melt(d[, .(if_score, rgcn_score, gat_score)],
                measure.vars = c("if_score", "rgcn_score", "gat_score"),
                variable.name = "model", value.name = "score")
sc_long[, model := factor(model, c("if_score", "rgcn_score", "gat_score"),
                          labels = c("Isolation Forest", "R-GCN", "GAT"))]
p_sdist <- ggplot(sc_long, aes(x = score)) +
  geom_density(fill = "#1f77b4", alpha = 0.4, color = "#1f77b4") +
  geom_vline(aes(xintercept = quantile(score, 0.9)),
             linetype = "dashed", color = "#d62728") +
  facet_wrap(~model, scales = "free") +
  labs(title = "Anomaly Score Distributions by Model",
       subtitle = "Right-skewed for all three models; dashed line = top-10% flag threshold",
       x = "Anomaly score", y = "Density") +
  theme(panel.grid.minor = element_blank())
savep(p_sdist, "Anomaly Score Distribution.png")

# ─── Pairwise score scatter with Spearman ────────────────────────────────────
pw <- rbindlist(list(
  d[, .(x = if_score,   y = rgcn_score, pair = "IF vs R-GCN")],
  d[, .(x = if_score,   y = gat_score,  pair = "IF vs GAT")],
  d[, .(x = rgcn_score, y = gat_score,  pair = "R-GCN vs GAT")]
))
sp <- pw[, .(sp = sprintf("Spearman rho = %.2f", cor(x, y, method = "spearman"))), by = pair]
sp[, pair := factor(pair, c("IF vs R-GCN", "IF vs GAT", "R-GCN vs GAT"))]
pw[, pair := factor(pair, levels(sp$pair))]
p3 <- ggplot(pw, aes(x = x, y = y)) +
  geom_point(size = 1.5, alpha = 0.5, color = "#1f77b4") +
  geom_text(data = sp, aes(label = sp), x = Inf, y = Inf,
            hjust = 1.05, vjust = 1.4, size = 3.4, color = "grey25") +
  facet_wrap(~pair, scales = "free") +
  labs(title = "Pairwise Anomaly Scores",
       subtitle = "Near-zero full-range rank agreement; see top-k overlap analysis (H4a) for the decision test",
       x = NULL, y = NULL) +
  theme(panel.grid.minor = element_blank())
savep(p3, "Pairwise Anomaly Score.png")

# ─── Flag matrix: which model flagged which taxpayer ────────────────────────
fl <- d[n_flags >= 1]
fm <- rbindlist(list(
  fl[, .(node_id, group, model = "IF",    flagged = flag_if)],
  fl[, .(node_id, group, model = "R-GCN", flagged = flag_rg)],
  fl[, .(node_id, group, model = "GAT",   flagged = flag_ga)]
))
fm[, model := factor(model, c("IF", "R-GCN", "GAT"))]
fm[, group := factor(group, c("Consensus", "IF only", "R-GCN only", "GAT only"))]
setorder(fm, group, node_id)
fl_rows <- unique(fm[, .(node_id, group)])
fl_rows[, row_id := seq_len(.N), by = group]
fm <- merge(fm, fl_rows, by = c("node_id", "group"))
p4 <- ggplot(fm, aes(x = model, y = row_id, fill = flagged)) +
  geom_tile(color = "white", linewidth = 0.3) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  scale_fill_manual(values = c("TRUE" = "#d62728", "FALSE" = "grey90"),
                    labels = c("TRUE" = "Flagged", "FALSE" = "Not flagged"),
                    name = NULL) +
  labs(title = "Which Model Flagged Which Taxpayer",
       subtitle = sprintf("Each row = one of the %d taxpayers flagged by at least one model\n(top 10%% per model), grouped by flag pattern",
                          nrow(fl_rows)),
       x = NULL, y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        panel.grid = element_blank())
savep(p4, "Anomaly Rank Comparision.png", w = 7, h = 8)

# ─── Sales vs purchases scatter by flag group ────────────────────────────────
d[, log_sales := log10(pmax(total_sales, 0) + 1)]
d[, log_purch := log10(pmax(total_purchases, 0) + 1)]
p11 <- ggplot(d, aes(x = log_purch, y = log_sales, color = group)) +
  geom_point(size = 2, alpha = 0.75) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = pal) +
  labs(title = "Total Sales vs Total Purchases by Flag Group",
       subtitle = "Dashed line = equal sales and purchases; axes log10(UGX + 1)",
       x = "Total purchases, log10(UGX + 1)", y = "Total sales, log10(UGX + 1)") +
  theme(panel.grid.minor = element_blank())
savep(p11, "Sales Vs Purchases.png")

# ─── Feature profiles by flag group (Figure 11 replacement) ─────────────────
pv <- data.table(
  group = d$group,
  `Total purchases` = log10(pmax(d$total_purchases, 0) + 1),
  `Total sales`     = log10(pmax(d$total_sales, 0) + 1),
  `Unique partners` = log10(pmax(d$n_unique_partners, 0) + 1),
  `P/S ratio`       = log10(pmax(d$purchase_to_sales_ratio, 0) + 1)
)
prof_long <- melt(pv, id.vars = "group", variable.name = "feature",
                  value.name = "value")
prof <- prof_long[, .(mean = mean(value), se = sd(value) / sqrt(.N)),
                  by = .(group, feature)]
p12 <- ggplot(prof, aes(x = group, y = mean, fill = group)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se),
                width = 0.2, linewidth = 0.4) +
  facet_wrap(~feature, scales = "free_y") +
  scale_fill_manual(values = pal) +
  labs(title = "Feature Profiles by Flag Group",
       subtitle = "Group means with SE bars; all features on log10(x + 1) scale",
       x = NULL, y = "Group mean") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1),
        panel.grid.minor = element_blank())
savep(p12, "Feature Profiles by Model.png", w = 10, h = 6.5)

# ─── Ego network of node 339 ─────────────────────────────────────────────────
e339 <- edges[(src == 339 | dst == 339) & src != dst]
n339 <- uniqueN(c(e339$src, e339$dst)) - 1   # exclude node 339 itself
cat(sprintf("Node 339 ego network: %d partners, %d directed edges (self-loops excluded)\n",
            n339, nrow(e339)))
ng <- graph_from_data_frame(e339, directed = TRUE)
V(ng)$is_focus <- V(ng)$name == "339"
f339 <- d[node_id == 339]
flg339 <- paste(c("IF", "R-GCN", "GAT")[c(f339$flag_if, f339$flag_rg, f339$flag_ga)],
                collapse = ", ")
p13 <- ggraph(ng, layout = "fr") +
  geom_edge_link(aes(color = edge_type),
                 arrow = arrow(length = unit(3, "mm"), type = "closed"),
                 end_cap = circle(4, "mm")) +
  geom_node_point(aes(size = is_focus, shape = is_focus, color = is_focus)) +
  scale_size_manual(values = c("TRUE" = 7, "FALSE" = 3), guide = "none") +
  scale_shape_manual(values = c("TRUE" = 17, "FALSE" = 16), guide = "none") +
  scale_color_manual(values = c("TRUE" = "#d62728", "FALSE" = "grey30"), guide = "none") +
  geom_node_text(aes(label = name), repel = TRUE, size = 3.3) +
  scale_edge_color_manual(values = c(sells_to = "#1f77b4", buys_from = "#ff7f0e"),
                          name = "Edge type") +
  labs(title = "Ego Network of Node 339",
       subtitle = sprintf("%d trading partners | flagged by: %s.\nSparse local structure explains the high R-GCN reconstruction error",
                          n339, flg339),
       x = NULL, y = NULL) +
  theme_graph(base_size = 12)
savep(p13, "ego_01_node_339.png", w = 8, h = 6)

cat("\nAll 13 plots regenerated in", plot_dir, "\n")
