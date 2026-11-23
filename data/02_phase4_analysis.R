# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Enhanced Analysis for Objective 4 (Comparative Analysis)
# ══════════════════════════════════════════════════════════════════════════════
# 
# This script strengthens Objective 4 by:
#   1. Creating case cards for 25 consensus nodes + 10 matched controls
#   2. Preparing blinded auditor review template
#   3. Exporting matched control metadata
#
# Output files:
#   - case_cards_consensus_flagged.csv (25 nodes with 2+ model flags)
#   - case_cards_control_matched.csv (10 matched non-flagged nodes)
#   - auditor_review_blinded.csv (shuffled, no labels, for auditor)
#
# Objective 4 strengthening: Adds independent auditor validation to model consensus
# ══════════════════════════════════════════════════════════════════════════════

library(tidyverse)
library(data.table)

cat("\n=== PHASE 4: Enhanced Analysis (Objective 4) ===\n")

# Load data
scores_all <- fread("data/all_model_scores_actual.csv") %>%
  as_tibble() %>%
  mutate(
    if_rank = rank(desc(if_score)),
    rgcn_rank = rank(desc(rgcn_score)),
    gat_rank = rank(desc(gat_score)),
    n_flags = (if_rank <= 62) + (rgcn_rank <= 62) + (gat_rank <= 62),
    avg_rank = (if_rank + rgcn_rank + gat_rank) / 3
  )

node_feats <- fread("data/node_features_actual.csv") %>% as_tibble()

# Consensus nodes (2+ models flagged)
consensus_ids <- scores_all %>% filter(n_flags >= 2) %>% pull(node_id)

# Merge and prepare
full_data <- node_feats %>%
  left_join(scores_all, by = "node_id") %>%
  arrange(avg_rank)

# Consensus cases
consensus_df <- full_data %>%
  filter(node_id %in% consensus_ids) %>%
  select(node_id, total_sales, total_purchases, n_unique_partners,
         output_input_vat_ratio, n_months_active_sales, n_months_active_purch,
         if_score, if_rank, rgcn_score, rgcn_rank, gat_score, gat_rank, n_flags, avg_rank)

cat("Consensus nodes: ", nrow(consensus_df), "\n")

# Control selection (matched on activity)
set.seed(9413)
non_consensus <- full_data %>% filter(!(node_id %in% consensus_ids))
log_cs <- log1p(consensus_df$total_sales)
log_cp <- log1p(consensus_df$total_purchases)

controls <- non_consensus %>%
  mutate(
    d_s = (log1p(total_sales) - median(log_cs)) / (sd(log_cs) + 0.01),
    d_p = (log1p(total_purchases) - median(log_cp)) / (sd(log_cp) + 0.01),
    dist = sqrt(d_s^2 + d_p^2)
  ) %>%
  filter(dist < 1.5) %>%
  arrange(dist) %>%
  slice(1:10) %>%
  select(node_id, total_sales, total_purchases, n_unique_partners,
         output_input_vat_ratio, n_months_active_sales, n_months_active_purch,
         if_score, if_rank, rgcn_score, rgcn_rank, gat_score, gat_rank)

cat("Control nodes (matched): ", nrow(controls), "\n")

# Export
fwrite(consensus_df, "data/case_cards_consensus_flagged.csv")
fwrite(controls, "data/case_cards_control_matched.csv")
cat("✓ case_cards_consensus_flagged.csv\n")
cat("✓ case_cards_control_matched.csv\n")

# Blinded auditor template
all_cases <- bind_rows(
  consensus_df %>% mutate(case_type = "CONSENSUS"),
  controls %>% mutate(case_type = "CONTROL")
) %>%
  mutate(case_num = row_number()) %>%
  select(case_num, node_id, total_sales, total_purchases, n_unique_partners) %>%
  slice(sample(1:n()))  # Shuffle for blinding

fwrite(all_cases, "data/auditor_review_blinded.csv")
cat("✓ auditor_review_blinded.csv\n")

cat("\nOBJECTIVE 4 ENHANCEMENT:\n")
cat("  Consensus (2+ models): ", nrow(consensus_df), "\n")
cat("  Control (matched): ", nrow(controls), "\n")
cat("  Auditor will rate all", nrow(all_cases), "cases on fraud likelihood (1-10)\n")
cat("  This enables validation: AUC-ROC of auditor vs model consensus\n")
cat("\n")
