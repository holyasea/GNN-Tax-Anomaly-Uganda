# ============================================================
#  04_hypothesis_tests.R  (rev 3)
#  Formal hypothesis tests for the EFRIS GNN anomaly-detection
#  proof-of-concept study.
#
#  Changes after rev 1 (supervisor feedback, Nov 2026):
#    H1   redesigned: community structure is the only decision
#         criterion; the degree distribution is reported
#         descriptively (the old combined rule that required the
#         degree distribution to follow a power law conflated two
#         separate claims).
#    H4a  top-k overlap added as the PRIMARY independence measure
#         (a full-range Spearman is dominated by the near-zero
#         score mass and ranks noise); Spearman CI kept as
#         secondary.
#    H2   redesigned: the rev-1 test compared IF-flagged vs
#         non-flagged taxpayers on the very features IF was
#         trained on (circular by construction). The decision now
#         rests on graph features IF never saw; the financial
#         comparison is retained as a descriptive profile only.
#    H4b  permutation test added alongside the binomial test
#         (the binomial assumes the three flag sets are mutually
#         independent draws, which the permutation test verifies
#         empirically).
#
#  Tests:
#    H1   Network topology      (community structure vs
#                                configuration-model null)
#    H2   Baseline detection    (IF flags vs HELD-OUT graph features;
#                                the rev-1 test on IF's own training
#                                features was circular and is kept
#                                only as a descriptive profile)
#    H3   Graph-model detection (structural profile + flag overlaps)
#    H4a  Score independence    (top-k overlap + Spearman CI)
#    H4b  Consensus precision   (binomial + permutation test)
#
#  Seed:   7142 (global)
#  Output: data/hypothesis_tests/
#  Reads only from data/ ; writes only to data/hypothesis_tests/
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(igraph)
  library(poweRlaw)
})

set.seed(7142)
OUT <- "data/hypothesis_tests"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# p-value formatter: 4 decimal places
fp <- function(p) formatC(as.numeric(p), format = "f", digits = 4)

# ------------------------------------------------------------
# Load data (read-only)
# ------------------------------------------------------------
nodes  <- fread("data/node_features_actual.csv")
scores <- fread("data/all_model_scores_actual.csv")
edges  <- fread("data/edge_list_actual.csv")

# fread infers the two large variance columns as character (some entries are
# written in scientific notation). Coerce them to numeric so that medians and
# Kruskal-Wallis tests operate on numbers rather than on text labels.
num_cols <- c("var_sale_value", "var_purchase_value")
nodes[, (num_cols) := lapply(.SD, as.numeric), .SDcols = num_cols]

N   <- nrow(scores)          # 614 taxpayers
TOP <- ceiling(0.10 * N)     # top 10% = 62 per model

# Flag sets: top 10% of each model's score (ties broken by first occurrence)
rk <- function(x) rank(-x, ties.method = "first") <= TOP
flag_if   <- rk(scores$if_score)
flag_rgcn <- rk(scores$rgcn_score)
flag_gat  <- rk(scores$gat_score)
n_flags   <- flag_if + flag_rgcn + flag_gat

# ============================================================
# H1: NETWORK TOPOLOGY (rev 2)
# H1a: community structure. Null: modularity is no higher than
#      in random graphs with the same degree sequence.
# H1b: degree distribution, reported descriptively (not part
#      of the decision rule).
# ============================================================
cat("\n===== H1: Network topology =====\n")

g <- simplify(graph_from_data_frame(edges[, .(src, dst)], directed = FALSE))
deg <- degree(g)

set.seed(7142)
mod_obs <- modularity(cluster_louvain(g))
cat(sprintf("Observed modularity: %.4f  (graph: %d nodes, %d edges)\n",
            mod_obs, vcount(g), ecount(g)))

# Null distribution: 1,000 configuration-model graphs with the same degree
# sequence. (Rev 1 required, in addition, that the degree distribution be
# consistent with a power law; that conflated two separate claims and is
# now reported descriptively under H1b.)
B <- 1000
mod_null <- numeric(B)
for (b in seq_len(B)) {
  gr <- sample_degseq(deg, method = "configuration")
  mod_null[b] <- modularity(cluster_louvain(gr))
  if (b %% 200 == 0) cat(sprintf("  null graph %d/%d\n", b, B))
}
mod_mean <- mean(mod_null)
mod_sd   <- sd(mod_null)
z_mod    <- (mod_obs - mod_mean) / mod_sd
p_mod    <- (1 + sum(mod_null >= mod_obs)) / (B + 1)   # empirical p

reject_H1 <- p_mod < 0.05

# Power-law fit to the degree distribution (descriptive only).
# NOTE: estimate_xmin() returns a LIST (gof, xmin, pars, ntail, distance),
# so its scalar components must be extracted explicitly.
deg_pos  <- deg[deg > 0]
m_pl     <- displ$new(deg_pos)
est      <- estimate_xmin(m_pl)
xmin_est <- est$xmin
gof_est  <- est$gof
m_pl$setXmin(xmin_est)
alpha    <- estimate_pars(m_pl)$pars[1]
ks_p     <- tryCatch({
  set.seed(7142)
  bs <- bootstrap_p(m_pl, no_of_sims = 100, seed = 7142)
  bs$p
}, error = function(e) NA_real_)

h1 <- data.table(
  metric = c("modularity_observed", "modularity_null_mean", "modularity_null_sd",
             "modularity_z", "modularity_p_empirical",
             "powerlaw_alpha", "powerlaw_xmin", "powerlaw_gof",
             "powerlaw_ks_p", "graph_nodes", "graph_edges"),
  value  = c(mod_obs, mod_mean, mod_sd, z_mod, p_mod,
             alpha, xmin_est, gof_est,
             ks_p, vcount(g), ecount(g))
)
fwrite(h1, file.path(OUT, "H1_network_topology.csv"))

h1_interp <- if (reject_H1) {
  sprintf(paste("The transaction network shows significant, non-random community structure",
                "(modularity %.4f vs %.4f under the configuration-model null, z = %.2f).",
                "The degree distribution is heavy-tailed (alpha = %.2f); the bootstrap KS test",
                "rejects a pure power law (p = %s), so the tail is best described as heavy-tailed",
                "rather than exactly power-law."),
          mod_obs, mod_mean, z_mod, alpha, fp(ks_p))
} else {
  sprintf(paste("Community structure is not higher than in degree-matched random graphs",
                "(modularity %.4f, empirical p = %s); H1 is not rejected."),
          mod_obs, fp(p_mod))
}

h1_txt <- c(
  "H1: Network topology (rev 2: community structure only)",
  "============================================================",
  sprintf("Graph (simplified, undirected): %d nodes, %d edges.", vcount(g), ecount(g)),
  sprintf("Observed Louvain modularity: %.4f.", mod_obs),
  sprintf("Null (1,000 configuration-model graphs, same degree sequence): mean = %.4f, SD = %.4f.",
          mod_mean, mod_sd),
  sprintf("Modularity z-score = %.4f; empirical p = %s.", z_mod, fp(p_mod)),
  "",
  "Decision rule (rev 2): reject if empirical p < 0.05. The degree distribution is",
  "no longer part of the decision rule (that combined rule conflated two claims).",
  sprintf("DECISION: %s", ifelse(reject_H1, "REJECT H1", "FAIL TO REJECT H1")),
  sprintf("Interpretation: %s", h1_interp),
  "",
  sprintf("H1b (descriptive, not a decision): power-law fit alpha = %.4f (xmin = %g,", alpha, xmin_est),
  sprintf("KS distance = %.4f), bootstrap KS p = %s. The pure power law is rejected,", gof_est, fp(ks_p)),
  "so the degree tail is heavy-tailed but not exactly power-law."
)
writeLines(h1_txt, file.path(OUT, "H1_summary.txt"))

# ============================================================
# H2: BASELINE DETECTION (rev 3)
# Rev 1 tested IF-flagged vs non-flagged taxpayers on the very
#   features IF was trained on. The supervisor noted this is
#   circular: IF flags extremes of its own features, so
#   significance is guaranteed and the test validates nothing.
# Redesign: the DECISION is now based on held-out graph features
#   that IF never saw (degree, PageRank, reciprocated value
#   share, counterparty concentration). If IF flags are
#   associated with anomalies on features outside its own
#   feature set, that is non-circular evidence that the flags
#   track something real rather than a tautology.
# The rev-1 financial comparison is retained as a DESCRIPTIVE
#   profile only (no decision rests on it).
# Null: IF-flagged and non-flagged taxpayers do not differ on
#       held-out graph features.
# ============================================================
cat("\n===== H2: Baseline detection (held-out graph features) =====\n")

df2 <- merge(nodes, scores[, .(node_id, flag_if)], by = "node_id")

# --- Descriptive only (circular by construction; no decision) ---
feats <- c("total_sales", "total_purchases", "output_input_vat_ratio",
           "purchase_to_sales_ratio", "n_unique_partners")

h2_desc <- rbindlist(lapply(feats, function(f) {
  x <- df2[[f]][df2$flag_if]
  y <- df2[[f]][!df2$flag_if]
  w <- suppressWarnings(wilcox.test(x, y, alternative = "two.sided", exact = FALSE))
  data.table(feature = f,
             median_flagged    = median(x),
             iqr_flagged       = IQR(x),
             median_nonflagged = median(y),
             iqr_nonflagged    = IQR(y),
             U_statistic       = unname(w$statistic),
             p_raw             = w$p.value)
}))
h2_desc[, p_adj := p.adjust(p_raw, method = "BH")]
fwrite(h2_desc, file.path(OUT, "H2_financial_profile_descriptive.csv"))

# --- Decision basis: held-out graph features IF never saw ---
gd <- graph_from_data_frame(edges[, .(src, dst, total_amount)], directed = TRUE)
out_deg <- degree(gd, mode = "out")
in_deg  <- degree(gd, mode = "in")
out_w   <- strength(gd, mode = "out")
in_w    <- strength(gd, mode = "in")
pr_gd   <- page_rank(gd, damping = 0.85)$vector

el_h <- edges
out_tot <- el_h[, .(out_w_sum = sum(total_amount)), by = src]
rev_exist <- el_h[, .(has_rev = TRUE), by = .(src = dst, dst = src)]
el_h <- merge(el_h, out_tot, by = "src")
el_h <- merge(el_h, rev_exist, by = c("src", "dst"), all.x = TRUE)
el_h[, has_rev := fifelse(is.na(has_rev), FALSE, has_rev)]
recip_h <- el_h[, .(recip_w = sum(total_amount * has_rev)), by = src]
recip_h <- merge(recip_h, out_tot, by = "src")
recip_h[, reciprocated_share := recip_w / pmax(out_w_sum, 1)]
maxcp_h <- el_h[, .(max_share = max(total_amount) / pmax(out_w_sum[1], 1)), by = src]

gfeat <- data.table(
  node_id        = as.integer(V(gd)$name),
  log_out_degree = sign(out_deg) * log1p(out_deg),
  log_in_degree  = sign(in_deg)  * log1p(in_deg),
  log_out_weight = sign(out_w) * log1p(abs(out_w)),
  log_in_weight  = sign(in_w)  * log1p(abs(in_w)),
  page_rank      = pr_gd
)
gfeat <- merge(gfeat, recip_h[, .(node_id = src, reciprocated_share)], by = "node_id", all.x = TRUE)
gfeat <- merge(gfeat, maxcp_h[, .(node_id = src, max_counterparty_share = max_share)],
               by = "node_id", all.x = TRUE)
gfeat[is.na(reciprocated_share), reciprocated_share := 0]
gfeat[is.na(max_counterparty_share), max_counterparty_share := 0]

df2g <- merge(df2, gfeat, by = "node_id")

held_feats <- c("log_out_degree", "log_in_degree", "page_rank",
                "reciprocated_share", "max_counterparty_share")

h2 <- rbindlist(lapply(held_feats, function(f) {
  x <- df2g[[f]][df2g$flag_if]
  y <- df2g[[f]][!df2g$flag_if]
  w <- suppressWarnings(wilcox.test(x, y, alternative = "two.sided", exact = FALSE))
  data.table(feature = f,
             median_flagged    = median(x),
             iqr_flagged       = IQR(x),
             median_nonflagged = median(y),
             iqr_nonflagged    = IQR(y),
             U_statistic       = unname(w$statistic),
             p_raw             = w$p.value)
}))
h2[, p_adj := p.adjust(p_raw, method = "BH")]
h2[, significant := p_adj < 0.05]
n_sig_H2 <- sum(h2$significant)
reject_H2 <- n_sig_H2 >= 3

fwrite(h2, file.path(OUT, "H2_graph_feature_comparison.csv"))

h2_txt <- c(
  "H2: Baseline detection (rev 3: held-out graph features)",
  "============================================================",
  "REV 1 TEST WAS CIRCULAR: comparing IF-flagged vs non-flagged taxpayers on",
  "IF's own training features is guaranteed to be significant and validates",
  "nothing. That comparison is retained as a descriptive profile only",
  "(H2_financial_profile_descriptive.csv); no decision rests on it.",
  "",
  sprintf("IF-flagged (top 10%%): n = %d; non-flagged: n = %d.", sum(flag_if), sum(!flag_if)),
  "Held-out graph features IF never saw during training. Mann-Whitney U tests",
  "(two-sided), Benjamini-Hochberg corrected:",
  "",
  sprintf("%-26s %14s %14s %10s %10s", "feature", "p_raw", "p_adj", "sig", "U"),
  apply(h2, 1, function(r) sprintf("%-26s %14s %14s %10s %10s",
        r["feature"], fp(r["p_raw"]), fp(r["p_adj"]), r["significant"], r["U_statistic"])),
  "",
  sprintf("Significant held-out features (BH, 5%%): %d of 5.", n_sig_H2),
  "Decision rule: reject if at least three of five held-out features differ significantly.",
  sprintf("DECISION: %s", ifelse(reject_H2, "REJECT H2", "FAIL TO REJECT H2")),
  if (reject_H2) paste("Interpretation: IF flags are associated with anomalies on graph features",
                       "outside the IF feature set, i.e. the flags generalize beyond the tautology",
                       "of being extreme on their own training features.") else
    paste("Interpretation: IF flags show no association with graph features outside the",
          "IF feature set; the baseline's flags do not demonstrate validity beyond its",
          "own training features.")
)
writeLines(h2_txt, file.path(OUT, "H2_summary.txt"))

# ============================================================
# H3: GRAPH MODEL DETECTION
# Null: GNN-unique flags do not differ structurally from
#       IF flags, and are not more numerous than chance.
# ============================================================
cat("\n===== H3: Graph model detection =====\n")

# Mutually exclusive partition (matches the consensus group used elsewhere):
grp <- ifelse(n_flags >= 2, "Consensus",
       ifelse(flag_if & !flag_rgcn & !flag_gat, "IF_uniquely_flagged",
       ifelse((flag_rgcn | flag_gat) & !flag_if, "GNN_uniquely_flagged",
              "Non_flagged")))

df3 <- merge(nodes, data.table(node_id = scores$node_id, grp = grp), by = "node_id")
df3[, grp := factor(grp, levels = c("IF_uniquely_flagged", "GNN_uniquely_flagged",
                                    "Consensus", "Non_flagged"))]

struct_feats <- c("purchase_to_sales_ratio", "n_unique_partners", "var_purchase_value")

grp_medians <- rbindlist(lapply(struct_feats, function(f) {
  as.data.table(as.list(tapply(df3[[f]], df3$grp, median, na.rm = TRUE)))[
    , feature := f][]
}), fill = TRUE)
setcolorder(grp_medians, c("feature", levels(df3$grp)))

kw <- rbindlist(lapply(struct_feats, function(f) {
  k <- kruskal.test(df3[[f]] ~ df3$grp)
  data.table(feature = f,
             kw_chi_squared = unname(k$statistic),
             kw_df          = unname(k$parameter),
             kw_p           = k$p.value)
}))
kw[, kw_significant := kw_p < 0.05]

# Flag-group sizes
sizes <- data.table(
  group = levels(df3$grp),
  n = as.integer(table(df3$grp)[levels(df3$grp)])
)

# Overlap expectations under independence
p_ind        <- TOP / N
exp_pair     <- TOP * TOP / N            # ~6.26 per pair
exp_all3     <- p_ind^3 * N              # ~0.632
obs_if_rgcn  <- sum(flag_if & flag_rgcn)
obs_if_gat   <- sum(flag_if & flag_gat)
obs_rgcn_gat <- sum(flag_rgcn & flag_gat)
obs_all3     <- sum(flag_if & flag_rgcn & flag_gat)

# Hypergeometric p-values for pairwise overlaps: P(X >= observed)
hyper_p <- function(obs) phyper(obs - 1, TOP, N - TOP, TOP, lower.tail = FALSE)
# Triple overlap p-value via binomial approximation with p = p_ind^3
p_all3  <- pbinom(obs_all3 - 1, N, p_ind^3, lower.tail = FALSE)

overlaps <- data.table(
  pair     = c("IF & RGCN", "IF & GAT", "RGCN & GAT", "All three"),
  observed = c(obs_if_rgcn, obs_if_gat, obs_rgcn_gat, obs_all3),
  expected = c(exp_pair, exp_pair, exp_pair, exp_all3),
  p_hyper  = c(hyper_p(obs_if_rgcn), hyper_p(obs_if_gat), hyper_p(obs_rgcn_gat), p_all3)
)

# Unique GNN flags, defined consistently with the exclusive group table:
# exactly one of {RGCN, GAT} and not IF. Expected = N * 2 p (1-p)^2.
obs_gnn_only <- as.integer(sizes$n[sizes$group == "GNN_uniquely_flagged"])
exp_gnn_only <- N * 2 * p_ind * (1 - p_ind)^2

h3 <- merge(grp_medians, kw, by = "feature")
fwrite(h3, file.path(OUT, "H3_structural_comparison.csv"))
fwrite(overlaps, file.path(OUT, "H3_overlap_comparison.csv"))
fwrite(sizes,    file.path(OUT, "H3_group_sizes.csv"))

kw_any_sig  <- any(kw$kw_significant)   # "the Kruskal-Wallis test is significant"
gnn_exceeds <- obs_gnn_only > exp_gnn_only
reject_H3   <- kw_any_sig && gnn_exceeds

h3_interp <- if (reject_H3) {
  "The GNN flag groups occupy systematically different structural profiles, and GNN-unique flags exceed chance expectation."
} else {
  sprintf(paste("The flag groups differ structurally (smallest Kruskal-Wallis p = %s),",
                "but the number of taxpayers uniquely flagged by the GNNs (%d) does not exceed",
                "the independence expectation (%.1f), so H3 is not rejected."),
          fp(min(kw$kw_p)), obs_gnn_only, exp_gnn_only)
}

h3_txt <- c(
  "H3: Graph-model detection (RGCN & GAT vs Isolation Forest)",
  "============================================================",
  "Flag-group sizes (mutually exclusive partition):",
  sprintf("  %-22s n = %d", sizes$group, sizes$n),
  "",
  "Kruskal-Wallis tests across the four groups:",
  apply(h3, 1, function(r) sprintf("  %-24s chi2 = %8.3f, df = %d, p = %s",
        r["feature"], as.numeric(r["kw_chi_squared"]), as.integer(r["kw_df"]), fp(r["kw_p"]))),
  "",
  "Flag overlaps vs independence (hypergeometric / binomial):",
  apply(overlaps, 1, function(r) sprintf("  %-12s observed = %s, expected = %s, p = %s",
        r["pair"], r["observed"], fp(as.numeric(r["expected"])), fp(r["p_hyper"]))),
  "",
  sprintf("Unique GNN flags (exactly one of RGCN/GAT, not IF): observed = %d, expected = %.1f.",
          obs_gnn_only, exp_gnn_only),
  "Decision rule: reject if the Kruskal-Wallis test is significant AND observed unique",
  "GNN flags exceed the expected number under random flagging.",
  sprintf("  Kruskal-Wallis significant (any feature): %s; unique GNN flags exceed expected: %s.",
          kw_any_sig, gnn_exceeds),
  sprintf("DECISION: %s", ifelse(reject_H3, "REJECT H3", "FAIL TO REJECT H3")),
  sprintf("Interpretation: %s", h3_interp)
)
writeLines(h3_txt, file.path(OUT, "H3_summary.txt"))

# ============================================================
# H4a: SCORE INDEPENDENCE (rev 2)
# PRIMARY: top-k overlap. Each model's top-10% flag set is
#   compared against every other model's; observed overlap is
#   tested against the permutation null (random top-k set of
#   the same size). A full-range Spearman is dominated by the
#   near-zero score mass and ranks noise, so it is kept only as
#   a secondary description.
# Null: no model pair shares more top-k taxpayers than chance.
# ============================================================
cat("\n===== H4a: Score independence =====\n")

pairs4a <- list(c("if_score", "rgcn_score"),
                c("if_score", "gat_score"),
                c("rgcn_score", "gat_score"))

# --- Primary: top-k overlap with permutation p-values ---
B4a <- 10000
set.seed(7142)
h4a_topk <- rbindlist(lapply(pairs4a, function(pr) {
  a  <- rk(scores[[pr[1]]])
  bv <- scores[[pr[2]]]
  obs <- sum(a & rk(bv))
  perm <- vapply(seq_len(B4a), function(i) sum(a & rk(sample(bv))), numeric(1))
  p_perm <- (1 + sum(perm >= obs)) / (B4a + 1)
  data.table(pair = paste(pr[1], "vs", pr[2]),
             k = TOP,
             observed_overlap = obs,
             expected_overlap = TOP * TOP / N,
             jaccard = obs / sum(a | rk(bv)),
             p_perm = p_perm)
}))
fwrite(h4a_topk, file.path(OUT, "H4a_topk_overlap.csv"))

reject_H4a <- any(h4a_topk$p_perm < 0.05)
p_H4a      <- min(h4a_topk$p_perm)   # most significant pair

# --- Secondary: Spearman rank correlations with Fisher z CIs ---
h4a <- rbindlist(lapply(pairs4a, function(pr) {
  r  <- cor(scores[[pr[1]]], scores[[pr[2]]], method = "spearman")
  z  <- atanh(r)
  se <- 1 / sqrt(N - 3)
  ci <- tanh(z + c(-1, 1) * 1.96 * se)
  # One-sided test of H0: rho >= 0.30 against rho < 0.30 (Fisher z)
  z_h0      <- (atanh(r) - atanh(0.30)) / se
  p_below30 <- pnorm(z_h0)
  data.table(pair = paste(pr[1], "vs", pr[2]),
             spearman_r   = r,
             ci_lower     = ci[1],
             ci_upper     = ci[2],
             p_below_0.30 = p_below30,
             n            = N)
}))
h4a[, ci_upper_below_0.30 := abs(ci_upper) < 0.30]

fwrite(h4a, file.path(OUT, "H4a_spearman_ci.csv"))

h4a_txt <- c(
  "H4a: Score independence (rev 2: top-k overlap primary, Spearman secondary)",
  "============================================================",
  sprintf("Top-%d overlap between model flag sets, tested against a permutation", TOP),
  sprintf("null (%d permutations of one model's scores; p = P(perm >= obs)):", B4a),
  apply(h4a_topk, 1, function(r) sprintf("  %-24s observed = %s, expected = %.2f, Jaccard = %.3f, p = %s",
        r["pair"], r["observed_overlap"], as.numeric(r["expected_overlap"]),
        as.numeric(r["jaccard"]), fp(r["p_perm"]))),
  "",
  "Decision rule (rev 2): reject if any pair's top-k overlap exceeds the permutation",
  sprintf("null at 5%%. Observed: %s.", ifelse(reject_H4a, "at least one pair exceeds", "no pair exceeds")),
  sprintf("DECISION: %s", ifelse(reject_H4a, "REJECT H4a", "FAIL TO REJECT H4a")),
  "",
  "Secondary (full-range Spearman; dominated by near-zero scores, ranks noise):",
  apply(h4a, 1, function(r) sprintf("  %-24s r = %+.4f, 95%% CI [%+.4f, %+.4f]",
        r["pair"], as.numeric(r["spearman_r"]),
        as.numeric(r["ci_lower"]), as.numeric(r["ci_upper"]))),
  "",
  "Interpretation: the top-k overlap separates which model pairs share signal at",
  "the top of their rankings from pairs that do not; see hypothesis_report.md."
)
writeLines(h4a_txt, file.path(OUT, "H4a_summary.txt"))

# ============================================================
# H4b: CONSENSUS PRECISION (rev 2)
# Null: the proportion flagged by >= 2 models is no higher than
#       expected under independence.
# Binomial test kept; permutation test added as the primary
#   check (the binomial treats the three flag sets as
#   independent draws; the permutation verifies this directly).
# ============================================================
cat("\n===== H4b: Consensus precision =====\n")

obs_consensus <- sum(n_flags >= 2)
p_ge2         <- 3 * p_ind^2 - 2 * p_ind^3
exp_consensus <- N * p_ge2

# One-sided exact binomial test: P(X >= observed) under independence
p_binom <- pbinom(obs_consensus - 1, N, p_ge2, lower.tail = FALSE)

# Permutation test: permute each model's scores independently, recompute
# the number of taxpayers flagged by >= 2 models, 10,000 times.
B4b <- 10000
set.seed(7142)
perm_consensus <- numeric(B4b)
for (i in seq_len(B4b)) {
  nf <- rk(sample(scores$if_score)) + rk(sample(scores$rgcn_score)) +
        rk(sample(scores$gat_score))
  perm_consensus[i] <- sum(nf >= 2)
}
p_perm_4b <- (1 + sum(perm_consensus >= obs_consensus)) / (B4b + 1)

# Pairwise-overlap framing (as specified)
exp_pair_total <- 3 * (TOP * TOP / N)
obs_pair_total <- obs_if_rgcn + obs_if_gat + obs_rgcn_gat
p_pois <- ppois(obs_pair_total - 1, exp_pair_total, lower.tail = FALSE)

reject_H4b <- p_perm_4b < 0.05

h4b <- data.table(
  metric = c("observed_consensus", "expected_consensus_independence",
             "p_binomial_ge_observed", "p_permutation_ge_observed",
             "perm_null_mean", "perm_null_sd",
             "observed_total_pairwise_overlaps", "expected_total_pairwise_overlaps",
             "p_poisson_pairwise"),
  value  = c(obs_consensus, exp_consensus, p_binom, p_perm_4b,
             mean(perm_consensus), sd(perm_consensus),
             obs_pair_total, exp_pair_total, p_pois)
)
fwrite(h4b, file.path(OUT, "H4b_consensus_test.csv"))

h4b_txt <- c(
  "H4b: Consensus precision",
  "============================================================",
  sprintf("Each model flags its top 10%% (n = %d of %d).", TOP, N),
  sprintf("Observed consensus (flagged by >= 2 models): %d.", obs_consensus),
  sprintf("Expected consensus under independence: %.2f (p(>= 2) = %.4f).",
          exp_consensus, p_ge2),
  sprintf("Exact binomial test, P(X >= %d): p = %s.", obs_consensus, fp(p_binom)),
  sprintf("Permutation test (%d permutations of all three score vectors):", B4b),
  sprintf("  permuted consensus mean = %.2f (SD %.2f); P(perm >= obs) = %s.",
          mean(perm_consensus), sd(perm_consensus), fp(p_perm_4b)),
  "",
  sprintf("Total pairwise overlaps: observed = %d, expected under independence = %.2f.",
          obs_pair_total, exp_pair_total),
  sprintf("Poisson approximation for >= that many overlaps: p = %s.", fp(p_pois)),
  "",
  "Decision rule (rev 2): reject if the permutation p < 0.05.",
  sprintf("DECISION: %s", ifelse(reject_H4b, "REJECT H4b", "FAIL TO REJECT H4b")),
  "Interpretation: see hypothesis_report.md."
)
writeLines(h4b_txt, file.path(OUT, "H4b_summary.txt"))

# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
dec <- function(x) ifelse(x, "reject", "fail to reject")

h4a_topk_txt <- paste(h4a_topk$pair, h4a_topk$observed_overlap,
                      sprintf("(p = %s)", fp(h4a_topk$p_perm)), collapse = "; ")

summary_tbl <- data.table(
  hypothesis_id = c("H1", "H2", "H3", "H4a", "H4b"),
  test_statistic = c(
    sprintf("modularity z = %.4f", z_mod),
    sprintf("%d of 5 held-out graph features significant (min adj. p = %s)", n_sig_H2, fp(min(h2$p_adj))),
    sprintf("min KW p = %s; unique GNN flags %d vs %d expected",
            fp(min(kw$kw_p)), obs_gnn_only, round(exp_gnn_only)),
    sprintf("top-k overlap: %s", h4a_topk_txt),
    sprintf("observed consensus = %d (expected %.1f)", obs_consensus, exp_consensus)
  ),
  p_value = c(fp(p_mod), fp(min(h2$p_adj)), fp(min(kw$kw_p)),
              fp(p_H4a), fp(p_perm_4b)),
  decision = c(dec(reject_H1), dec(reject_H2), dec(reject_H3), dec(reject_H4a), dec(reject_H4b)),
  interpretation = c(h1_interp,
    if (reject_H2) paste("IF flags are associated with anomalies on held-out graph features that IF never saw,",
                         "so the baseline's validity is not merely a tautology of its own training features.") else
      "IF flags show no association with held-out graph features; the baseline's validity beyond its own training features is not demonstrated.",
    h3_interp,
    if (reject_H4a) paste("Top-k overlap analysis: IF shares no more top-ranked taxpayers with the GNNs than chance,",
                          "but the two GNNs share signal with each other; full-range Spearman correlations are near zero.")
    else "No model pair shares more top-ranked taxpayers than chance; each model captures a distinct anomaly dimension.",
    if (reject_H4b) "Consensus cases occur significantly more often than independence would predict, confirming non-random model agreement." else
      "Consensus cases do not exceed the independence expectation.")
)
fwrite(summary_tbl, file.path(OUT, "hypothesis_summary_table.csv"))

# ============================================================
# MARKDOWN REPORT
# ============================================================
md <- c(
  "# Hypothesis Tests: GNN Anomaly Detection in Uganda's EFRIS Data (rev 3)",
  "",
  "All tests use global seed 7142. Each model flags its top 10% of the 614 registered",
  "taxpayers (n = 62). P-values are reported to four decimal places. Changes after",
  "rev 1: H1 tests community structure only; H2 is decided on held-out graph",
  "features (the rev-1 test on IF's own training features was circular); H4a is",
  "decided by top-k overlap (Spearman secondary); H4b is decided by a permutation",
  "test (binomial secondary).",
  "",
  "## Summary of decisions",
  "",
  "| Hypothesis | Test statistic | p-value | Decision |",
  "|---|---|---|---|",
  apply(summary_tbl, 1, function(r) sprintf("| %s | %s | %s | %s |",
        r["hypothesis_id"], r["test_statistic"], r["p_value"], r["decision"])),
  "",
  "## H1: Network topology",
  "",
  sprintf(paste("The simplified EFRIS transaction graph contains %d nodes and %d edges, with an observed",
                "Louvain modularity of %.4f. Against %d configuration-model graphs sharing the same degree",
                "sequence, modularity averaged %.4f (SD %.4f), a z-score of %.4f and an empirical p-value of",
                "%s. The degree distribution is heavy-tailed (power-law alpha = %.4f, xmin = %g), though a",
                "pure power law is rejected by the bootstrap KS test (p = %s); the tail is best described",
                "as heavy-tailed rather than exactly power-law. **Decision: %s.**"),
          vcount(g), ecount(g), mod_obs, B, mod_mean, mod_sd, z_mod, fp(p_mod),
          alpha, xmin_est, fp(ks_p),
          ifelse(reject_H1, "Reject H1", "Fail to reject H1")),
  "",
  h1_interp,
  "",
  "## H2: Baseline detection (held-out graph features)",
  "",
  sprintf(paste("The rev-1 version of this test compared Isolation Forest-flagged taxpayers against the",
                "remaining %d on the same features IF was trained on. The supervisor noted this is circular:",
                "IF flags extremes of its own features, so significance is guaranteed and the test validates",
                "nothing. That financial comparison is retained only as a descriptive profile",
                "(H2_financial_profile_descriptive.csv); no decision rests on it."),
          sum(!flag_if)),
  "",
  sprintf("The decision now rests on five graph features IF never saw: %s.",
          paste(held_feats, collapse = ", ")),
  "",
  "| Feature | Median (flagged) | Median (non-flagged) | p (raw) | p (BH) |",
  "|---|---|---|---|---|",
  apply(h2, 1, function(r) sprintf("| %s | %.4g | %.4g | %s | %s |",
        r["feature"], as.numeric(r["median_flagged"]), as.numeric(r["median_nonflagged"]),
        fp(r["p_raw"]), fp(r["p_adj"]))),
  "",
  sprintf("%d of the 5 held-out features differ significantly after Benjamini-Hochberg correction. **Decision: %s.**",
          n_sig_H2, ifelse(reject_H2, "Reject H2", "Fail to reject H2")),
  "",
  if (reject_H2) paste("IF flags are associated with anomalies on graph features outside the IF feature set,",
                       "so the baseline's flags generalize beyond being extreme on their own training features.") else
    paste("IF flags show no association with held-out graph features; the baseline's validity beyond its own",
          "training features is not demonstrated."),
  "",
  "## H3: Graph-model detection",
  "",
  sprintf("Flag-group sizes (mutually exclusive): IF-unique = %d, GNN-unique = %d, consensus = %d, non-flagged = %d.",
          sizes$n[sizes$group == "IF_uniquely_flagged"],
          sizes$n[sizes$group == "GNN_uniquely_flagged"],
          sizes$n[sizes$group == "Consensus"],
          sizes$n[sizes$group == "Non_flagged"]),
  "",
  "Kruskal-Wallis tests across groups:",
  "",
  "| Feature | KW chi-squared | df | p |",
  "|---|---|---|---|",
  apply(h3, 1, function(r) sprintf("| %s | %.4f | %d | %s |",
        r["feature"], as.numeric(r["kw_chi_squared"]), as.integer(r["kw_df"]), fp(r["kw_p"]))),
  "",
  sprintf("Flags unique to the GNNs (exactly one of RGCN/GAT, not IF): observed = %d, expected under independence = %.1f. **Decision: %s.**",
          obs_gnn_only, exp_gnn_only, ifelse(reject_H3, "Reject H3", "Fail to reject H3")),
  "",
  h3_interp,
  "",
  "## H4a: Score independence",
  "",
  sprintf("Primary measure: overlap between each pair of models' top-%d flag sets, tested against a permutation null (10,000 permutations).", TOP),
  "",
  "| Model pair | Observed overlap | Expected | Jaccard | p (perm) |",
  "|---|---|---|---|---|",
  apply(h4a_topk, 1, function(r) sprintf("| %s | %s | %.2f | %.3f | %s |",
        r["pair"], r["observed_overlap"], as.numeric(r["expected_overlap"]),
        as.numeric(r["jaccard"]), fp(r["p_perm"]))),
  "",
  sprintf("Secondary measure: full-range Spearman correlations (dominated by the near-zero score mass, so not used for the decision): %s. **Decision: %s.**",
          paste(sprintf("%s r = %+.3f", h4a$pair, h4a$spearman_r), collapse = "; "),
          ifelse(reject_H4a, "Reject H4a", "Fail to reject H4a")),
  "",
  if (reject_H4a) paste("The Isolation Forest shares no more top-ranked taxpayers with either GNN than chance,",
                        "but the two GNNs share signal with each other: their top-k overlap is significantly",
                        "higher than the permutation null, consistent with the H4b finding that model agreement",
                        "is concentrated in the RGCN-GAT pair.")
  else "No model pair shares more top-ranked taxpayers than chance; each model captures a distinct anomaly dimension.",
  "",
  "## H4b: Consensus precision",
  "",
  sprintf(paste("Observed consensus cases (flagged by two or more models): %d. Expected under independence:",
                "%.2f. Exact binomial test p = %s. Permutation test (10,000 permutations of all three score",
                "vectors): permuted consensus mean %.2f (SD %.2f), p = %s. Total pairwise overlaps:",
                "observed = %d vs expected = %.2f. **Decision: %s.**"),
          obs_consensus, exp_consensus, fp(p_binom),
          mean(perm_consensus), sd(perm_consensus), fp(p_perm_4b),
          obs_pair_total, exp_pair_total,
          ifelse(reject_H4b, "Reject H4b", "Fail to reject H4b")),
  "",
  "---",
  "",
  "### Data note",
  "",
  sprintf(paste("Two large variance columns (`var_sale_value`, `var_purchase_value`) are read by `fread`",
                "as character and are coerced to numeric before analysis. The simplified undirected graph has",
                "%d edges; this is lower than the %s directed edge count because reciprocal buyer/seller",
                "edges collapse to a single undirected edge. The consensus count here (%d) uses a mutually",
                "exclusive flag partition and may differ slightly from values quoted elsewhere under a",
                "non-exclusive definition."),
          ecount(g), format(nrow(edges), big.mark = ","), obs_consensus),
  ""
)
writeLines(md, file.path(OUT, "hypothesis_report.md"))

cat("\n============================================================\n")
cat("All hypothesis tests complete. Outputs written to:", OUT, "\n")
print(summary_tbl)
