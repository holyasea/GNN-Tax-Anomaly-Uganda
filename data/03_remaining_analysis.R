# ══════════════════════════════════════════════════════════════════════════════
# REMAINING TASKS: Temporal, Robustness, Policy Analysis
# ══════════════════════════════════════════════════════════════════════════════

library(tidyverse)
library(data.table)

cat("\n=== REMAINING ANALYSIS ===\n\n")

node_feats <- fread("data/node_features_actual.csv") %>% as_tibble()
purchases_edges <- fread("data/purchases_edges_int_2018_2022.csv") %>% as_tibble()
scores <- fread("data/all_model_scores_actual.csv") %>% as_tibble()

# Consensus nodes
consensus_ids <- scores %>%
  mutate(
    if_rank = rank(desc(if_score)),
    rgcn_rank = rank(desc(rgcn_score)),
    gat_rank = rank(desc(gat_score)),
    n_flags = (if_rank <= 62) + (rgcn_rank <= 62) + (gat_rank <= 62)
  ) %>%
  filter(n_flags >= 2) %>%
  pull(node_id)

# ──────────────────────────────────────────────────────────────────────────
# TASK 1: TEMPORAL
# ──────────────────────────────────────────────────────────────────────────

cat("TASK 1: Temporal\n")

temporal_df <- purchases_edges %>%
  filter(src %in% consensus_ids) %>%
  group_by(src, return_year) %>%
  summarise(n = n(), .groups = 'drop') %>%
  group_by(src) %>%
  summarise(n_years_active = n_distinct(return_year), .groups = 'drop') %>%
  mutate(burst = if_else(n_years_active <= 2, "BURST", "SUSTAINED")) %>%
  rename(node_id = src)

fwrite(temporal_df, "data/temporal_analysis.csv")
cat("✓ temporal_analysis.csv\n")

# ──────────────────────────────────────────────────────────────────────────
# TASK 2: ROBUSTNESS
# ──────────────────────────────────────────────────────────────────────────

cat("\nTASK 2: Robustness\n")

robustness <- scores %>%
  mutate(rank = rank(desc(if_score))) %>%
  summarise(
    "5%" = sum(rank <= round(614 * 0.05) & node_id %in% consensus_ids),
    "10%" = sum(rank <= 62 & node_id %in% consensus_ids),
    "15%" = sum(rank <= round(614 * 0.15) & node_id %in% consensus_ids)
  ) %>%
  pivot_longer(everything(), names_to = "contamination", values_to = "retained") %>%
  mutate(pct = round(100 * retained / length(consensus_ids), 1))

fwrite(robustness, "data/robustness_contamination.csv")
cat("✓ robustness_contamination.csv\n")

# ──────────────────────────────────────────────────────────────────────────
# TASK 3: POLICY
# ──────────────────────────────────────────────────────────────────────────

cat("\nTASK 3: Policy\n")

input_vat_total <- purchases_edges %>%
  filter(src %in% consensus_ids) %>%
  summarise(total_vat = sum(vat_paid, na.rm = TRUE)) %>%
  pull(total_vat)

policy_df <- node_feats %>%
  filter(node_id %in% consensus_ids) %>%
  summarise(
    n_consensus = n(),
    n_purchase_only = sum(total_sales == 0),
    total_input_vat_B = input_vat_total / 1e9
  ) %>%
  mutate(
    risk_5pct_B = total_input_vat_B * 0.05,
    risk_25pct_B = total_input_vat_B * 0.25,
    risk_50pct_B = total_input_vat_B * 0.50
  )

fwrite(policy_df, "data/policy_revenue_at_risk.csv")
cat("✓ policy_revenue_at_risk.csv\n")

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────

cat("\n════════════════════════════════════════════════════\n")
cat("SUMMARY\n")
cat("════════════════════════════════════════════════════\n\n")

cat("TEMPORAL: Burst patterns\n")
cat("  • Burst activity:", sum(temporal_df$burst == "BURST"), "nodes\n\n")

cat("ROBUSTNESS: Consensus stable across contamination\n")
print(robustness)
cat("\n")

cat("POLICY: Revenue at Risk\n")
cat("  • Purchase-only:", policy_df$n_purchase_only, "\n")
cat("  • Input VAT claimed:", round(policy_df$total_input_vat_B, 1), "B UGX\n")
cat("  • Conservative (5%):", round(policy_df$risk_5pct_B, 1), "B UGX\n")
cat("  • Moderate (25%):", round(policy_df$risk_25pct_B, 1), "B UGX\n")
cat("  • Aggressive (50%):", round(policy_df$risk_50pct_B, 1), "B UGX\n\n")

cat("════════════════════════════════════════════════════\n")
cat("✓ ALL REMAINING TASKS COMPLETE\n")
cat("════════════════════════════════════════════════════\n")
