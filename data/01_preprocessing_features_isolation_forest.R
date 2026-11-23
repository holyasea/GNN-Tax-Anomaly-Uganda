# ══════════════════════════════════════════════════════════════════════════════
# EFRIS VAT Anomaly Detection — Full R Pipeline
# ══════════════════════════════════════════════════════════════════════════════
# This script:
#   1. Loads raw EFRIS sales and purchase invoice data
#   2. Performs data quality checks and cleaning
#   3. Engineers taxpayer-level features
#   4. Builds the transaction graph and exports edge list
#   5. Runs Isolation Forest with hyper-parameter tuning
#   6. Exports results for the GNN pipeline (Python)
#   7. Loads GNN results and runs comparative analysis
#   8. Analyzes what each model uniquely detects
#   9. Produces all visualizations
# ══════════════════════════════════════════════════════════════════════════════

set.seed(7142)

library(tidyverse)
library(data.table)
library(isotree)
library(igraph)
library(pROC)
library(patchwork)
library(ggraph)
library(tidygraph)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ──────────────────────────────────────────────────────────────────────────────
set.seed(7142)
sales <- fread("data/Sales_Merged.csv")
purchases <- fread("data/purchases_edges_int_2018_2022.csv")
node_map <- fread("data/node_id_mapping.csv")

cat("=== Raw Data Loaded ===\n")
cat(sprintf("Sales invoices:    %s rows\n", format(nrow(sales), big.mark = ",")))
cat(sprintf("Purchase invoices: %s rows\n", format(nrow(purchases), big.mark = ",")))
cat(sprintf("Node mappings:     %s nodes\n", format(nrow(node_map), big.mark = ",")))
cat(sprintf("Sales date range:    %d–%d\n", min(sales$return_year), max(sales$return_year)))
cat(sprintf("Purchases date range: %d–%d\n", min(purchases$return_year), max(purchases$return_year)))

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA QUALITY CHECKS
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Data Quality Checks ===\n")

# 2.1 Missing values
cat("\n--- Missing Values ---\n")
cat("Sales:\n")
print(colSums(is.na(sales)))
cat("\nPurchases:\n")
print(colSums(is.na(purchases)))

# 2.2 Amount distributions and anomalies
cat("\n--- Sales Amount Distribution ---\n")
print(summary(sales$amount))
cat("\n--- Purchase Amount Distribution ---\n")
print(summary(purchases$amount))

# 2.3 Negative/zero amounts (credit notes, reversals)
cat("\n--- Negative or Zero Amounts ---\n")
cat(sprintf("Sales <= 0:     %s (%.2f%%)\n",
            format(sum(sales$amount <= 0), big.mark = ","),
            100 * mean(sales$amount <= 0)))
cat(sprintf("Purchases <= 0: %s (%.2f%%)\n",
            format(sum(purchases$amount <= 0), big.mark = ","),
            100 * mean(purchases$amount <= 0)))

# 2.4 Duplicate check
cat("\n--- Duplicate Rows ---\n")
sales_dupes <- sum(duplicated(sales))
purch_dupes <- sum(duplicated(purchases))
cat(sprintf("Sales duplicates:    %d\n", sales_dupes))
cat(sprintf("Purchase duplicates: %d\n", purch_dupes))

# 2.5 Network topology overview
cat("\n--- Network Topology ---\n")
cat(sprintf("Unique sellers in sales:      %d\n", uniqueN(sales$src)))
cat(sprintf("Unique buyers in sales:       %d\n", uniqueN(sales$dst)))
cat(sprintf("Unique buyers in purchases:   %d\n", uniqueN(purchases$src)))
cat(sprintf("Unique suppliers in purchases: %d\n", uniqueN(purchases$dst)))

# 2.6 VAT rate check
cat("\n--- VAT Rate Consistency ---\n")
sales[, vat_rate := vat_paid / amount]
cat("Sales VAT rates (where amount > 0):\n")
print(summary(sales[amount > 0, vat_rate]))

purchases[, vat_rate := vat_paid / amount]
cat("\nPurchase VAT rates (where amount > 0):\n")
print(summary(purchases[amount > 0, vat_rate]))

# Clean up temporary column
sales[, vat_rate := NULL]
purchases[, vat_rate := NULL]

# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING — TAXPAYER-LEVEL FEATURES
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Feature Engineering ===\n")

# 3.1 Sales-side features (src = seller)
sales_features <- sales[, .(
  total_sales          = sum(amount),
  total_output_vat     = sum(vat_paid),
  n_sales_invoices     = .N,
  n_unique_buyers      = uniqueN(dst),
  avg_sale_value       = mean(amount),
  var_sale_value       = var(amount),
  median_sale_value    = median(amount),
  n_activity_codes_sales = uniqueN(activity_isic4_id),
  n_months_active_sales  = uniqueN(paste(return_year, return_month)),
  n_years_active_sales   = uniqueN(return_year)
), by = .(node_id = src)]

# 3.2 Purchase-side features (src = buyer)
purchase_features <- purchases[, .(
  total_purchases      = sum(amount),
  total_input_vat      = sum(vat_paid),
  n_purchase_invoices  = .N,
  n_unique_sellers     = uniqueN(dst),
  avg_purchase_value   = mean(amount),
  var_purchase_value   = var(amount),
  median_purchase_value = median(amount),
  n_activity_codes_purch = uniqueN(activity_isic4_id),
  n_months_active_purch  = uniqueN(paste(return_year, return_month)),
  n_years_active_purch   = uniqueN(return_year)
), by = .(node_id = src)]

# 3.3 Merge sales and purchase features
taxpayer_features <- merge(sales_features, purchase_features,
                           by = "node_id", all = TRUE)

# 3.4 Derived ratio features
taxpayer_features[, `:=`(
  # Output-to-input VAT ratio (add 1 to denominator to avoid division by zero)
  output_input_vat_ratio = total_output_vat / (total_input_vat + 1),

  # Purchase-to-sales ratio
  purchase_to_sales_ratio = total_purchases / (total_sales + 1),

  # Total unique trading partners
  n_unique_partners = fifelse(is.na(n_unique_buyers), 0L, n_unique_buyers) +
                      fifelse(is.na(n_unique_sellers), 0L, n_unique_sellers),

  # Overall average invoice value
  avg_invoice_value = (fifelse(is.na(total_sales), 0, total_sales) +
                       fifelse(is.na(total_purchases), 0, total_purchases)) /
                      (fifelse(is.na(n_sales_invoices), 0L, n_sales_invoices) +
                       fifelse(is.na(n_purchase_invoices), 0L, n_purchase_invoices)),

  # Combined variance of invoice values
  var_invoice_value = fifelse(
    is.na(var_sale_value) & is.na(var_purchase_value), 0,
    fifelse(is.na(var_sale_value), var_purchase_value,
    fifelse(is.na(var_purchase_value), var_sale_value,
            (var_sale_value + var_purchase_value) / 2))
  )
)]

tf <- taxpayer_features  # alias used throughout downstream code

cat(sprintf("Feature matrix: %d taxpayers × %d columns\n",
            nrow(taxpayer_features), ncol(taxpayer_features)))
cat(sprintf("  With both sales and purchases: %d\n",
            sum(!is.na(taxpayer_features$total_sales) &
                !is.na(taxpayer_features$total_purchases))))
cat(sprintf("  Sales only:    %d\n",
            sum(!is.na(taxpayer_features$total_sales) &
                is.na(taxpayer_features$total_purchases))))
cat(sprintf("  Purchases only: %d\n",
            sum(is.na(taxpayer_features$total_sales) &
                !is.na(taxpayer_features$total_purchases))))

# 3.5 Feature summary
cat("\n--- Feature Summary ---\n")
feature_summary <- taxpayer_features[, .(
  total_sales          = median(total_sales, na.rm = TRUE),
  total_purchases      = median(total_purchases, na.rm = TRUE),
  output_input_vat_ratio = median(output_input_vat_ratio, na.rm = TRUE),
  purchase_to_sales_ratio = median(purchase_to_sales_ratio, na.rm = TRUE),
  n_unique_partners    = median(n_unique_partners),
  avg_invoice_value    = median(avg_invoice_value)
)]
print(feature_summary)

# ──────────────────────────────────────────────────────────────────────────────
# 4. TRANSACTION GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Graph Construction ===\n")

# 4.1 Aggregate sales edges
sales_agg <- sales[, .(
  total_amount    = sum(amount),
  total_vat       = sum(vat_paid),
  n_invoices      = .N,
  avg_amount      = mean(amount),
  sd_amount       = sd(amount),
  n_months_active = uniqueN(paste(return_year, return_month))
), by = .(src, dst)]
sales_agg[, edge_type := "sells_to"]

# 4.2 Aggregate purchase edges
purch_agg <- purchases[, .(
  total_amount    = sum(amount),
  total_vat       = sum(vat_paid),
  n_invoices      = .N,
  avg_amount      = mean(amount),
  sd_amount       = sd(amount),
  n_months_active = uniqueN(paste(return_year, return_month))
), by = .(src, dst)]
purch_agg[, edge_type := "buys_from"]

# 4.3 Combined edge list
all_edges <- rbind(sales_agg, purch_agg)

cat(sprintf("Aggregated edges: %s\n", format(nrow(all_edges), big.mark = ",")))
cat(sprintf("  Sales edges:    %s\n", format(nrow(sales_agg), big.mark = ",")))
cat(sprintf("  Purchase edges: %s\n", format(nrow(purch_agg), big.mark = ",")))
cat(sprintf("  Unique nodes in graph: %d\n",
            uniqueN(c(all_edges$src, all_edges$dst))))

# 4.4 Graph properties (using igraph)
g <- graph_from_data_frame(
  all_edges[, .(src, dst)],
  directed = TRUE
)

cat(sprintf("\nGraph: %d nodes, %d edges\n", vcount(g), ecount(g)))
cat(sprintf("Density: %.6f\n", edge_density(g)))
cat(sprintf("Connected components (weak): %d\n",
            components(g, mode = "weak")$no))

deg <- degree(g)
in_deg <- degree(g, mode = "in")
out_deg <- degree(g, mode = "out")
cat(sprintf("Degree distribution: min=%d, median=%d, mean=%.1f, max=%d\n",
            min(deg), median(deg), mean(deg), max(deg)))

# Clustering coefficient
cc <- transitivity(g, type = "average")
cat(sprintf("Mean clustering coefficient: %.4f\n", cc))

# 4.5 Objective 1 Plots — Network properties

# Plot 1: Degree distribution (log-log scale)
deg_df <- data.frame(degree = deg)
p_degree <- ggplot(deg_df, aes(x = degree)) +
  geom_histogram(bins = 60, fill = "steelblue", alpha = 0.7) +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "Degree Distribution (Log-Log Scale)",
       subtitle = sprintf("Nodes: %d | Mean: %.1f | Median: %d | Max: %d",
                          length(deg), mean(deg), median(deg), max(deg)),
       x = "Degree (log scale)", y = "Count (log scale)") +
  theme_minimal()
print(p_degree)

# Plot 2: In-degree vs out-degree
deg_io <- data.frame(
  in_degree = in_deg,
  out_degree = out_deg,
  node_id = as.integer(V(g)$name)
) |>
  mutate(is_taxpayer = node_id %in% taxpayer_features$node_id)

p_deg_io <- ggplot(deg_io, aes(x = in_degree + 1, y = out_degree + 1,
                                color = is_taxpayer)) +
  geom_point(alpha = 0.3, size = 1) +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "In-Degree vs Out-Degree",
       x = "In-Degree (log)", y = "Out-Degree (log)",
       color = "Taxpayer node") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p_deg_io)

# Plot 3: Temporal transaction patterns (monthly volume over time)
sales_monthly <- sales[, .(
  n_invoices = .N,
  total_amount = sum(amount)
), by = .(return_year, return_month)]
sales_monthly[, date := as.Date(sprintf("%d-%02d-01", return_year, return_month))]
sales_monthly[, type := "Sales"]

purch_monthly <- purchases[, .(
  n_invoices = .N,
  total_amount = sum(amount)
), by = .(return_year, return_month)]
purch_monthly[, date := as.Date(sprintf("%d-%02d-01", return_year, return_month))]
purch_monthly[, type := "Purchases"]

monthly_all <- rbind(sales_monthly, purch_monthly)

p_vol <- ggplot(monthly_all, aes(x = date, y = n_invoices, color = type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  labs(title = "Monthly Invoice Volume Over Time",
       x = NULL, y = "Number of Invoices", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")

p_amt <- ggplot(monthly_all, aes(x = date, y = total_amount / 1e9, color = type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  labs(title = "Monthly Transaction Value Over Time",
       x = NULL, y = "Total Amount (Billions)", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")

p_temporal <- p_vol / p_amt +
  plot_annotation(title = "Temporal Transaction Patterns (2018–2022)")
print(p_temporal)

# 4.6 Descriptive statistics of raw invoice data

cat("\n=== Descriptive Statistics — Sales Invoices ===\n")
cat(sprintf("Total invoices: %s\n", format(nrow(sales), big.mark = ",")))
cat(sprintf("Date range: %d–%d\n", min(sales$return_year), max(sales$return_year)))
cat(sprintf("Unique sellers: %d\n", uniqueN(sales$src)))
cat(sprintf("Unique buyers: %d\n", uniqueN(sales$dst)))
cat(sprintf("Unique seller-buyer pairs: %s\n",
            format(uniqueN(sales[, .(src, dst)]), big.mark = ",")))
cat(sprintf("Total sales value: %s\n", format(sum(sales$amount), big.mark = ",")))
cat(sprintf("Total VAT collected: %s\n", format(sum(sales$vat_paid), big.mark = ",")))
cat("\nAmount distribution:\n")
print(summary(sales$amount))
cat(sprintf("\nNegative amounts (credit notes): %s (%.2f%%)\n",
            format(sum(sales$amount < 0), big.mark = ","),
            100 * mean(sales$amount < 0)))
cat(sprintf("Zero amounts: %d\n", sum(sales$amount == 0)))
cat(sprintf("Activity codes (ISIC4): %d unique\n", uniqueN(sales$activity_isic4_id)))

cat("\n=== Descriptive Statistics — Purchase Invoices ===\n")
cat(sprintf("Total invoices: %s\n", format(nrow(purchases), big.mark = ",")))
cat(sprintf("Unique buyers: %d\n", uniqueN(purchases$src)))
cat(sprintf("Unique suppliers: %d\n", uniqueN(purchases$dst)))
cat(sprintf("Unique buyer-supplier pairs: %s\n",
            format(uniqueN(purchases[, .(src, dst)]), big.mark = ",")))
cat(sprintf("Total purchase value: %s\n", format(sum(purchases$amount), big.mark = ",")))
cat(sprintf("Total input VAT: %s\n", format(sum(purchases$vat_paid), big.mark = ",")))
cat("\nAmount distribution:\n")
print(summary(purchases$amount))
cat(sprintf("\nNegative amounts (credit notes): %s (%.2f%%)\n",
            format(sum(purchases$amount < 0), big.mark = ","),
            100 * mean(purchases$amount < 0)))
cat(sprintf("Zero amounts: %d\n", sum(purchases$amount == 0)))
cat(sprintf("Activity codes (ISIC4): %d unique\n", uniqueN(purchases$activity_isic4_id)))

# Invoices per taxpayer
invoices_per_seller <- sales[, .N, by = src]
invoices_per_buyer <- purchases[, .N, by = src]
cat("\n=== Invoices per Taxpayer ===\n")
cat("Sales (per seller):\n")
print(summary(invoices_per_seller$N))
cat("\nPurchases (per buyer):\n")
print(summary(invoices_per_buyer$N))

# Yearly breakdown
yearly_sales <- sales[, .(n_invoices = .N, total_amount = sum(amount),
                           n_sellers = uniqueN(src), n_buyers = uniqueN(dst)),
                       by = return_year]
yearly_purch <- purchases[, .(n_invoices = .N, total_amount = sum(amount),
                               n_buyers = uniqueN(src), n_suppliers = uniqueN(dst)),
                           by = return_year]
cat("\nSales by year:\n")
print(yearly_sales[order(return_year)])
cat("\nPurchases by year:\n")
print(yearly_purch[order(return_year)])

# 4.7 Community detection (Louvain algorithm)

cat("\n=== Community Detection (Louvain) ===\n")

g_undir <- as_undirected(g, mode = "collapse")
if ("weight" %in% edge_attr_names(g_undir)) {
  g_undir <- delete_edge_attr(g_undir, "weight")
}

communities <- cluster_louvain(g_undir)

cat(sprintf("Number of communities: %d\n", length(communities)))
cat(sprintf("Modularity: %.4f\n", modularity(communities)))

comm_sizes <- sizes(communities)
cat(sprintf("Community sizes: min=%d, median=%d, max=%d\n", 
            as.integer(min(comm_sizes)), 
            as.integer(median(comm_sizes)), 
            as.integer(max(comm_sizes))))

cat("\nTop 10 communities by size:\n")
top_comms <- sort(comm_sizes, decreasing = TRUE)[1:10]
for (i in seq_along(top_comms)) {
  cat(sprintf("  Community %s: %s nodes\n", names(top_comms)[i],
              format(top_comms[i], big.mark = ",")))
}

# Map communities to nodes
node_communities <- data.frame(
  node_id = as.integer(V(g_undir)$name),
  community = as.integer(membership(communities))
)
node_communities$is_taxpayer <- node_communities$node_id %in% taxpayer_features$node_id

taxpayer_comms <- node_communities[node_communities$is_taxpayer, ]
cat(sprintf("\nTaxpayers spread across %d communities\n",
            length(unique(taxpayer_comms$community))))

# Plot: Community sizes and flagged taxpayers per community
comm_summary <- node_communities |>
  group_by(community) |>
  summarise(total_nodes = n(), n_taxpayers = sum(is_taxpayer), .groups = "drop") |>
  arrange(desc(total_nodes))

p_comm1 <- ggplot(
  comm_summary |> mutate(community = factor(community, levels = community)),
  aes(x = community, y = total_nodes)
) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_text(aes(label = format(total_nodes, big.mark = ",")),
            vjust = -0.3, size = 2.8) +
  labs(title = "Community Sizes (Louvain Algorithm)",
       subtitle = sprintf("%d communities | Modularity: %.4f",
                          length(communities), modularity(communities)),
       x = "Community", y = "Number of Nodes") +
  theme_minimal()
print(p_comm1)

# 4.8 Feature importance analysis — moved to section 5.5 (after IF scores assigned)

# ──────────────────────────────────────────────────────────────────────────────
# 5. ISOLATION FOREST
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Isolation Forest ===\n")

# 5.1 Prepare feature matrix
feature_cols <- c(
  "total_sales", "total_purchases", "output_input_vat_ratio",
  "n_unique_partners", "avg_invoice_value", "var_sale_value",
  "var_purchase_value", "purchase_to_sales_ratio",
  "n_months_active_sales", "n_months_active_purch"
)

tf <- as.data.frame(taxpayer_features)

# Replace NAs with 0 for taxpayers missing one side (sales or purchases)
for (col in feature_cols) {
  tf[[col]][is.na(tf[[col]])] <- 0
}

# Handle infinite values in ratios
tf$output_input_vat_ratio[is.infinite(tf$output_input_vat_ratio)] <- 0
tf$purchase_to_sales_ratio[is.infinite(tf$purchase_to_sales_ratio)] <- 0

X <- tf[, feature_cols]

cat(sprintf("Feature matrix for IF: %d × %d\n", nrow(X), ncol(X)))
cat(sprintf("Any NAs: %s | Any Inf: %s\n",
            any(is.na(X)), any(is.infinite(as.matrix(X)))))

# 5.2 Hyperparameter grid search
param_grid <- expand.grid(
  ntrees      = c(100, 300, 500),
  sample_size = c(64, 128, 256),
  ndim        = c(1, 2)
)

cat(sprintf("\nGrid search: %d configurations\n", nrow(param_grid)))

results <- list()
for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]

  model <- isolation.forest(
    X,
    ntrees      = params$ntrees,
    sample_size = min(params$sample_size, nrow(X)),
    ndim        = params$ndim,
    seed        = 7142
  )

  scores <- predict(model, X)

  results[[i]] <- list(
    params    = params,
    scores    = scores,
    mean_score = mean(scores),
    sd_score   = sd(scores),
    q90        = quantile(scores, 0.90),
    q95        = quantile(scores, 0.95)
  )
}

# 5.3 Select best model (highest score variance = best discrimination)
best_idx <- which.max(sapply(results, function(r) r$sd_score))
best <- results[[best_idx]]

cat(sprintf("\nBest model:\n"))
cat(sprintf("  ntrees=%d, sample_size=%d, ndim=%d\n",
            best$params$ntrees, best$params$sample_size, best$params$ndim))
cat(sprintf("  Score mean=%.4f, sd=%.4f\n", best$mean_score, best$sd_score))
cat(sprintf("  90th percentile: %.4f\n", best$q90))
cat(sprintf("  95th percentile: %.4f\n", best$q95))

# 5.4 Assign anomaly scores and flag top 10%
tf$if_anomaly_score <- best$scores
tf$if_flagged <- tf$if_anomaly_score >= best$q90

cat(sprintf("\nFlagged as anomalous (top 10%%): %d taxpayers\n", sum(tf$if_flagged)))

# 5.5 Feature importance analysis for flagged taxpayers
cat("\n=== Feature Importance Analysis ===\n")

flagged_nodes   <- tf[tf$if_flagged, ]
normal_taxpayers <- tf[!tf$if_flagged, ]

feature_cols <- c(
  "total_sales", "total_purchases", "output_input_vat_ratio",
  "n_unique_partners", "avg_invoice_value", "var_sale_value",
  "var_purchase_value", "purchase_to_sales_ratio",
  "n_months_active_sales", "n_months_active_purch"
)

feature_importance <- data.frame(
  node_id  = flagged_nodes$node_id,
  if_score = flagged_nodes$if_anomaly_score
)

for (feat in feature_cols) {
  feat_mean <- mean(normal_taxpayers[[feat]], na.rm = TRUE)
  feat_sd   <- sd(normal_taxpayers[[feat]],   na.rm = TRUE)
  z_scores  <- (flagged_nodes[[feat]] - feat_mean) / (feat_sd + 1e-6)
  feature_importance[[feat]] <- abs(z_scores)
}

feature_avg_importance <- colMeans(feature_importance[, feature_cols], na.rm = TRUE)
feature_avg_importance <- sort(feature_avg_importance, decreasing = TRUE)

cat("Feature importance (average |z-score| deviation):\n")
print(round(feature_avg_importance, 2))

p_feat_imp <- data.frame(
  feature    = names(feature_avg_importance),
  importance = as.numeric(feature_avg_importance)
) |>
  mutate(feature = factor(feature, levels = feature)) |>
  ggplot(aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_text(aes(label = round(importance, 2)), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(title = "Isolation Forest - Feature Importance for Flagged Taxpayers",
       subtitle = "Average absolute z-score deviation from normal taxpayers",
       x = "Feature", y = "Average |Z-Score| Deviation") +
  theme_minimal()
print(p_feat_imp)

write.csv(feature_importance,
          "data/feature_contributions_flagged.csv", row.names = FALSE)
write.csv(
  data.frame(feature       = names(feature_avg_importance),
             avg_abs_zscore = as.numeric(feature_avg_importance),
             rank           = seq_along(feature_avg_importance)),
  "data/feature_importance_summary.csv", row.names = FALSE
)
cat("  ✓ feature_contributions_flagged.csv\n")
cat("  ✓ feature_importance_summary.csv\n")

# 5.6 Refit final model with best parameters (for saving)
iforest_final <- isolation.forest(
  X,
  ntrees      = best$params$ntrees,
  sample_size = min(best$params$sample_size, nrow(X)),
  ndim        = best$params$ndim,
  seed        = 7142
)

# 5.6 Display top flagged taxpayers
cat("\n=== Top 20 Flagged Taxpayers ===\n")
top_flagged <- tf[order(-tf$if_anomaly_score),
                  c("node_id", "total_sales", "total_purchases",
                    "output_input_vat_ratio", "purchase_to_sales_ratio",
                    "n_unique_partners", "if_anomaly_score")]
print(head(top_flagged, 20))

# 5.7 Isolation Forest convergence analysis
# Show how scores stabilize with increasing number of trees
tree_counts <- c(10, 25, 50, 100, 200, 300, 500)
stability <- lapply(tree_counts, function(nt) {
  m <- isolation.forest(X, ntrees = nt, sample_size = best$params$sample_size,
                        ndim = best$params$ndim, seed = 7142)
  s <- predict(m, X)
  data.frame(ntrees = nt, mean_score = mean(s), sd_score = sd(s),
             q90 = quantile(s, 0.9), q95 = quantile(s, 0.95))
})
stability_df <- do.call(rbind, stability)

final_scores_if <- predict(iforest_final, X)
stability_df$rank_cor_vs_final <- sapply(tree_counts, function(nt) {
  m <- isolation.forest(X, ntrees = nt, sample_size = best$params$sample_size,
                        ndim = best$params$ndim, seed = 7142)
  s <- predict(m, X)
  cor(s, final_scores_if, method = "spearman")
})

p_if_conv1 <- ggplot(stability_df, aes(x = ntrees, y = rank_cor_vs_final)) +
  geom_line(linewidth = 0.8, color = "steelblue") +
  geom_point(size = 2.5, color = "steelblue") +
  geom_hline(yintercept = 0.99, linetype = "dashed", color = "red", alpha = 0.6) +
  annotate("text", x = 50, y = 0.985, label = "0.99 threshold",
           color = "red", size = 3) +
  scale_y_continuous(limits = c(0.9, 1.0)) +
  labs(title = "Isolation Forest - Score Convergence",
       subtitle = sprintf("Rank correlation with %d-tree model stabilizes by ~200 trees",
                          best$params$ntrees),
       x = "Number of Trees", y = "Spearman Rank Correlation vs Final Model") +
  theme_minimal()

p_if_conv2 <- ggplot(stability_df, aes(x = ntrees, y = sd_score)) +
  geom_line(linewidth = 0.8, color = "coral") +
  geom_point(size = 2.5, color = "coral") +
  labs(title = "Score Discrimination (Variance)",
       subtitle = "Score spread stabilizes after ~100 trees",
       x = "Number of Trees", y = "Score Std. Deviation") +
  theme_minimal()

p_if_convergence <- p_if_conv1 + p_if_conv2 +
  plot_annotation(
    title = "Isolation Forest Convergence Analysis",
    subtitle = sprintf("Scores stabilize well before %d trees (sample_size=%d, ndim=%d)",
                       best$params$ntrees, best$params$sample_size, best$params$ndim)
  )
print(p_if_convergence)

p_if_scores <- ggplot(tf, aes(x = if_anomaly_score)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = best$q90, linetype = "dashed", color = "red") +
  annotate("text", x = best$q90 + 0.01, y = Inf, vjust = 2, hjust = 0,
           label = sprintf("Top 10%% threshold = %.3f", best$q90),
           color = "red", size = 3.5) +
  labs(title = "Isolation Forest Anomaly Score Distribution",
       subtitle = sprintf("Best model: ntrees=%d, sample_size=%d, ndim=%d | %d flagged",
                          best$params$ntrees, best$params$sample_size,
                          best$params$ndim, sum(tf$if_flagged)),
       x = "Anomaly Score", y = "Count") +
  theme_minimal()
print(p_if_scores)

# 5.9 Isolation Forest — Anomaly scores by feature (top features)
# Show which features drive the highest-scoring taxpayers
if_top <- tf[order(-tf$if_anomaly_score), ]
if_top$rank <- 1:nrow(if_top)
if_top$group <- ifelse(if_top$if_flagged, "Flagged (top 10%)", "Normal")

p_if_feat1 <- ggplot(if_top, aes(x = log1p(abs(total_sales)),
                                  y = log1p(abs(total_purchases)),
                                  color = group)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(values = c("Flagged (top 10%)" = "red", "Normal" = "grey60")) +
  labs(title = "Sales vs Purchases",
       x = "log(Total Sales)", y = "log(Total Purchases)", color = NULL) +
  theme_minimal() + theme(legend.position = "bottom")

p_if_feat2 <- ggplot(if_top, aes(
    x = log1p(abs(purchase_to_sales_ratio)),
    y = log1p(abs(output_input_vat_ratio)),
    color = group)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(values = c("Flagged (top 10%)" = "red", "Normal" = "grey60")) +
  labs(title = "P/S Ratio vs VAT Ratio",
       x = "log(Purchase/Sales Ratio)", y = "log(Output/Input VAT Ratio)",
       color = NULL) +
  theme_minimal() + theme(legend.position = "bottom")

p_if_features <- p_if_feat1 + p_if_feat2 +
  plot_annotation(
    title = "Isolation Forest - Flagged vs Normal Taxpayers in Feature Space"
  )
print(p_if_features)

# ──────────────────────────────────────────────────────────────────────────────
# 6. EXPORT FOR GNN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Exporting Data ===\n")

# 6.1 Node features (for Python GNN pipeline)
write.csv(
  tf[, c("node_id", feature_cols, "if_anomaly_score")],
  "data/node_features_actual.csv",
  row.names = FALSE
)

# 6.2 Aggregated edge list
write.csv(all_edges, "data/edge_list_actual.csv", row.names = FALSE)

# 6.3 Save R workspace
save(
  taxpayer_features, tf, X, all_edges, g,
  iforest_final, best, results, feature_cols,
  sales_features, purchase_features,
  file = "data/r_workspace_actual.RData"
)

cat("Exported:\n")
cat("  data/node_features_actual.csv  (taxpayer features + IF scores)\n")
cat("  data/edge_list_actual.csv      (aggregated transaction edges)\n")
cat("  data/r_workspace_actual.RData  (R objects for later analysis)\n")

cat("\n=== Preprocessing and Isolation Forest Complete ===\n")
cat("Next step: Run gnn_pipeline_actual.py for RGCN and GAT models.\n")

# ══════════════════════════════════════════════════════════════════════════════
# AFTER RUNNING gnn_pipeline_actual.py, CONTINUE WITH THE SECTIONS BELOW
# ══════════════════════════════════════════════════════════════════════════════
system("python C:/Users/Holyweek/Desktop/Project/Project1/data/gnn_pipeline_actual.py")

# ──────────────────────────────────────────────────────────────────────────────
# 7. LOAD GNN RESULTS AND COMPARATIVE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Loading GNN Results ===\n")

scores <- read.csv("data/all_model_scores_actual.csv")
cat(sprintf("Loaded scores for %d taxpayers\n", nrow(scores)))

# 7.1 Flag top 10% per model
scores$if_flagged   <- scores$if_score   >= quantile(scores$if_score, 0.9)
scores$rgcn_flagged <- scores$rgcn_score >= quantile(scores$rgcn_score, 0.9)
scores$gat_flagged  <- scores$gat_score  >= quantile(scores$gat_score, 0.9)

# Consensus: flagged by 2+ models
scores$n_flags  <- scores$if_flagged + scores$rgcn_flagged + scores$gat_flagged
scores$consensus <- scores$n_flags >= 2

cat(sprintf("\nFlagged as anomalous (top 10%% per model):\n"))
cat(sprintf("  Isolation Forest: %d\n", sum(scores$if_flagged)))
cat(sprintf("  RGCN:             %d\n", sum(scores$rgcn_flagged)))
cat(sprintf("  GAT:              %d\n", sum(scores$gat_flagged)))
cat(sprintf("  Consensus (2+):   %d\n", sum(scores$consensus)))
cat(sprintf("  All 3 models:     %d\n", sum(scores$n_flags == 3)))

# 7.2 Pairwise rank correlations
cat("\n=== Pairwise Rank Correlations (Spearman) ===\n")
cat(sprintf("IF vs RGCN:  %.4f\n",
            cor(scores$if_score, scores$rgcn_score, method = "spearman")))
cat(sprintf("IF vs GAT:   %.4f\n",
            cor(scores$if_score, scores$gat_score, method = "spearman")))
cat(sprintf("RGCN vs GAT: %.4f\n",
            cor(scores$rgcn_score, scores$gat_score, method = "spearman")))

# 7.3 Consensus-flagged taxpayers
consensus_taxpayers <- scores |>
  dplyr::filter(consensus) |>
  merge(tf[, c("node_id", "total_sales", "total_purchases",
               "purchase_to_sales_ratio", "output_input_vat_ratio",
               "n_unique_partners")], by = "node_id") |>
  arrange(desc(n_flags), desc(if_score))

cat(sprintf("\n=== %d Consensus-Flagged Taxpayers ===\n", nrow(consensus_taxpayers)))
print(consensus_taxpayers[, c("node_id", "n_flags", "if_score", "rgcn_score",
                               "gat_score", "total_sales", "total_purchases",
                               "purchase_to_sales_ratio", "n_unique_partners")])


# ──────────────────────────────────────────────────────────────────────────────
# 8. MODEL-SPECIFIC ANOMALY PROFILING
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Model-Specific Anomaly Analysis ===\n")

# 8.1 Classify each taxpayer by which model(s) flagged them
scores_full <- scores |>
  left_join(
    tf |> dplyr::select(node_id, total_sales, total_purchases,
                        output_input_vat_ratio, purchase_to_sales_ratio,
                        n_unique_partners, avg_invoice_value,
                        var_sale_value, var_purchase_value,
                        n_months_active_sales, n_months_active_purch),
    by = "node_id"
  ) |>
  mutate(
    flag_group = case_when(
      if_flagged & !rgcn_flagged & !gat_flagged  ~ "IF only",
      !if_flagged & rgcn_flagged & !gat_flagged  ~ "RGCN only",
      !if_flagged & !rgcn_flagged & gat_flagged  ~ "GAT only",
      n_flags >= 2                                ~ "Consensus (2+)",
      TRUE                                        ~ "Not flagged"
    )
  )

cat("\n--- Flagging Group Counts ---\n")
print(scores_full |> count(flag_group) |> arrange(desc(n)))

# 8.2 Median feature profiles by group
cat("\n--- Median Feature Profiles by Group ---\n")
feature_profiles <- scores_full |>
  group_by(flag_group) |>
  summarise(
    n = n(),
    med_total_sales       = median(total_sales, na.rm = TRUE),
    med_total_purchases   = median(total_purchases, na.rm = TRUE),
    med_ps_ratio          = median(purchase_to_sales_ratio, na.rm = TRUE),
    med_vat_ratio         = median(output_input_vat_ratio, na.rm = TRUE),
    med_partners          = median(n_unique_partners, na.rm = TRUE),
    med_avg_invoice       = median(avg_invoice_value, na.rm = TRUE),
    pct_zero_sales        = round(100 * mean(total_sales == 0, na.rm = TRUE)),
    med_months_active     = median(
      pmax(n_months_active_sales, n_months_active_purch, na.rm = TRUE),
      na.rm = TRUE
    ),
    .groups = "drop"
  )
print(feature_profiles)


# ──────────────────────────────────────────────────────────────────────────────
# 9. VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

cat("\n=== Generating Visualizations ===\n")

# 9.1 Anomaly score distributions by model
scores_long <- scores |>
  tidyr::pivot_longer(
    cols = c(if_score, rgcn_score, gat_score),
    names_to = "model", values_to = "score"
  ) |>
  mutate(model = recode(model,
    "if_score"   = "Isolation Forest",
    "rgcn_score" = "RGCN",
    "gat_score"  = "GAT"
  ))

p_hist <- ggplot(scores_long, aes(x = score)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  facet_wrap(~model, scales = "free") +
  labs(title = "Anomaly Score Distributions - Actual EFRIS Data",
       x = "Anomaly Score", y = "Count") +
  theme_minimal()
print(p_hist)

# 9.2 Pairwise scatter plots with consensus highlights
p_s1 <- ggplot(scores, aes(if_score, rgcn_score)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_point(data = scores |> dplyr::filter(consensus),
             color = "red", size = 2) +
  labs(x = "Isolation Forest", y = "RGCN", title = "IF vs RGCN") +
  theme_minimal()

p_s2 <- ggplot(scores, aes(if_score, gat_score)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_point(data = scores |> dplyr::filter(consensus),
             color = "red", size = 2) +
  labs(x = "Isolation Forest", y = "GAT", title = "IF vs GAT") +
  theme_minimal()

p_s3 <- ggplot(scores, aes(rgcn_score, gat_score)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_point(data = scores |> dplyr::filter(consensus),
             color = "red", size = 2) +
  labs(x = "RGCN", y = "GAT", title = "RGCN vs GAT") +
  theme_minimal()

p_scatter <- p_s1 + p_s2 + p_s3 +
  plot_annotation(
    title = "Pairwise Anomaly Score Comparison",
    subtitle = "Red = flagged by 2+ models (consensus)"
  )
print(p_scatter)

# 9.3 Training loss curves
losses <- read.csv("data/training_losses_actual.csv")

losses_long <- losses |>
  tidyr::pivot_longer(
    cols = c(rgcn_loss, gat_loss),
    names_to = "model", values_to = "loss"
  ) |>
  mutate(model = recode(model, "rgcn_loss" = "RGCN", "gat_loss" = "GAT")) |>
  dplyr::filter(!is.na(loss))

p_loss <- ggplot(losses_long, aes(epoch, loss, color = model)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = c(0.0807, 0.2505), linetype = "dashed",
             color = c("coral", "turquoise4"), alpha = 0.6) +
  annotate("text", x = 5, y = 0.0807 + 0.015, label = "RGCN final: 0.0807",
           size = 3, hjust = 0, color = "coral") +
  annotate("text", x = 5, y = 0.2505 + 0.015, label = "GAT final: 0.2505",
           size = 3, hjust = 0, color = "turquoise4") +
  labs(title = "Model Convergence - Training Loss Curves",
       subtitle = "Both models converge well before 150 epochs, validating the epoch choice",
       x = "Epoch", y = "Loss (BCE)", color = "Model") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p_loss)

# 9.4 Feature profiles by flagging group (boxplots)
plot_data <- scores_full |>
  dplyr::filter(flag_group != "Not flagged") |>
  bind_rows(
    scores_full |>
      dplyr::filter(flag_group == "Not flagged") |>
      mutate(flag_group = "Baseline")
  ) |>
  mutate(
    flag_group = factor(flag_group,
      levels = c("IF only", "RGCN only", "GAT only", "Consensus (2+)", "Baseline")
    ),
    log_purchases = log1p(abs(total_purchases)),
    log_ps_ratio  = sign(purchase_to_sales_ratio) *
                    log1p(abs(purchase_to_sales_ratio)),
    log_vat_ratio = sign(output_input_vat_ratio) *
                    log1p(abs(output_input_vat_ratio))
  )

p_b1 <- ggplot(plot_data, aes(y = flag_group, x = log_purchases)) +
  geom_boxplot(aes(fill = flag_group), show.legend = FALSE, outlier.size = 0.8) +
  labs(x = "log(Total Purchases)", y = NULL, title = "Purchase Volume") +
  theme_minimal(base_size = 10)

p_b2 <- ggplot(plot_data, aes(y = flag_group, x = log_ps_ratio)) +
  geom_boxplot(aes(fill = flag_group), show.legend = FALSE, outlier.size = 0.8) +
  labs(x = "log(Purchase/Sales Ratio)", y = NULL,
       title = "Purchase-to-Sales Ratio") +
  theme_minimal(base_size = 10)

p_b3 <- ggplot(plot_data, aes(y = flag_group, x = n_unique_partners)) +
  geom_boxplot(aes(fill = flag_group), show.legend = FALSE, outlier.size = 0.8) +
  scale_x_log10() +
  labs(x = "Unique Partners (log scale)", y = NULL,
       title = "Trading Partners") +
  theme_minimal(base_size = 10)

p_b4 <- ggplot(plot_data, aes(y = flag_group, x = log_vat_ratio)) +
  geom_boxplot(aes(fill = flag_group), show.legend = FALSE, outlier.size = 0.8) +
  labs(x = "log(Output/Input VAT Ratio)", y = NULL, title = "VAT Ratio") +
  theme_minimal(base_size = 10)

p_boxplots <- (p_b1 + p_b2) / (p_b3 + p_b4) +
  plot_annotation(
    title = "Feature Profiles by Model Flagging Group",
    subtitle = "How do taxpayers uniquely flagged by each model differ?"
  )
print(p_boxplots)

# 9.5 Ego network visualizations for top consensus-flagged taxpayers

# Top 30 taxpayers by average rank across all 3 models
top_ego_ids <- scores |>
  mutate(avg_rank = (rank(-if_score) + rank(-rgcn_score) + rank(-gat_score)) / 3) |>
  arrange(avg_rank) |>
  head(30) |>
  left_join(
    tf |> dplyr::select(node_id, total_sales, total_purchases,
                        purchase_to_sales_ratio, n_unique_partners),
    by = "node_id"
  )

# Save each ego network as PNG (ggraph conflicts with RStudio device in loops)
dir.create("data/ego_networks", showWarnings = FALSE)

for (i in seq_len(nrow(top_ego_ids))) {
  row <- top_ego_ids[i, ]

  tryCatch({
    label <- sprintf(
      "%d flag(s) | P/S=%.2g, %d partners",
      row$n_flags, row$purchase_to_sales_ratio, row$n_unique_partners
    )

    focal_v <- which(V(g)$name == as.character(row$node_id))
    if (length(focal_v) == 0) {
      cat(sprintf("  Ego network %d/%d: Node %d — skipped (not in graph)\n",
                  i, nrow(top_ego_ids), row$node_id))
      next
    }

    ego_nodes <- unlist(ego(g, order = 1, nodes = focal_v, mode = "all"))
    if (length(ego_nodes) > 80) {
      others    <- setdiff(ego_nodes, focal_v)
      n_sample  <- min(79, length(others))
      ego_nodes <- c(focal_v, sample(others, n_sample))
    }

    subg <- induced_subgraph(g, ego_nodes)
    if (igraph::ecount(subg) == 0) {
      cat(sprintf("  Ego network %d/%d: Node %d — skipped (no edges)\n",
                  i, nrow(top_ego_ids), row$node_id))
      next
    }
    if ("weight" %in% edge_attr_names(subg)) subg <- delete_edge_attr(subg, "weight")

    tg <- as_tbl_graph(subg) |>
      mutate(
        node_id  = as.integer(name),
        is_focal = node_id == row$node_id,
        role = case_when(
          is_focal                        ~ "Focal (consensus)",
          node_id %in% consensus_taxpayers$node_id ~ "Other consensus",
          node_id %in% scores$node_id    ~ "Taxpayer",
          TRUE                            ~ "Counterparty"
        )
      )

    p_ego_i <- ggraph(tg, layout = "fr") +
      geom_edge_link(
        arrow      = arrow(length = unit(1.5, "mm"), type = "closed"),
        end_cap    = circle(2, "mm"),
        edge_colour = "grey60", edge_alpha = 0.3
      ) +
      geom_node_point(aes(size = is_focal, color = role, shape = role)) +
      scale_color_manual(values = c(
        "Focal (consensus)" = "red",  "Other consensus" = "orange",
        "Taxpayer"          = "steelblue", "Counterparty" = "grey70"
      )) +
      scale_shape_manual(values = c(
        "Focal (consensus)" = 18, "Other consensus" = 17,
        "Taxpayer" = 16, "Counterparty" = 1
      )) +
      scale_size_manual(values = c("TRUE" = 6, "FALSE" = 2), guide = "none") +
      labs(
        title    = sprintf("Node %d", row$node_id),
        subtitle = sprintf("%s\n%d nodes, %d edges", label,
                           igraph::vcount(subg), igraph::ecount(subg)),
        color = NULL, shape = NULL
      ) +
      theme_void() +
      theme(
        legend.position  = "bottom",
        plot.title       = element_text(face = "bold", size = 12),
        plot.subtitle    = element_text(size = 9, color = "grey40")
      )

    fname <- sprintf("data/ego_networks/ego_%02d_node_%d.png", i, row$node_id)
    png(fname, width = 1050, height = 900, res = 150)
    print(p_ego_i)
    dev.off()
    cat(sprintf("  Ego network %d/%d: Node %d → %s\n",
                i, nrow(top_ego_ids), row$node_id, fname))

  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    cat(sprintf("  Ego network %d/%d: Node %d — skipped (%s)\n",
                i, nrow(top_ego_ids), row$node_id, conditionMessage(e)))
  })
}

cat("Ego networks complete.\n")

# 9.6 Heatmap — Score rank comparison across models
rank_data <- scores |>
  mutate(
    if_rank   = rank(-if_score),
    rgcn_rank = rank(-rgcn_score),
    gat_rank  = rank(-gat_score)
  ) |>
  dplyr::filter(if_flagged | rgcn_flagged | gat_flagged) |>
  tidyr::pivot_longer(
    cols = c(if_rank, rgcn_rank, gat_rank),
    names_to = "model", values_to = "rank"
  ) |>
  mutate(
    model = recode(model,
      "if_rank"   = "Isolation Forest",
      "rgcn_rank" = "RGCN",
      "gat_rank"  = "GAT"
    ),
    node_label = as.factor(node_id)
  )

# Show top 30 most-flagged (lowest average rank across models) 
avg_ranks <- scores |>
  mutate(
    avg_rank = (rank(-if_score) + rank(-rgcn_score) + rank(-gat_score)) / 3
  ) |>
  arrange(avg_rank) |>
  head(30) |>
  pull(node_id)

rank_top30 <- rank_data |>
  dplyr::filter(node_id %in% avg_ranks) |>
  mutate(node_label = factor(node_id, levels = rev(avg_ranks)))

p_heatmap <- ggplot(rank_top30, aes(x = model, y = node_label, fill = rank)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "red", high = "grey90",
                      name = "Rank\n(lower = more\nanomalous)") +
  labs(title = "Anomaly Rank Comparisons - Top 30 Taxpayers",
       subtitle = "Ordered by average rank across all 3 models",
       x = NULL, y = "Node ID") +
  theme_minimal(base_size = 10) +
  theme(axis.text.y = element_text(size = 7))
(p_heatmap)

cat("\n=== All Analyses Complete ===\n")
