# =============================================================================
# COMPLETE SCRIPT: Extract large taxpayers, build edge list for IF, GCN & GAT
# =============================================================================

library(data.table)
library(haven)

# ---- 0. Ensure we have the original data sets ----
if (!exists("Schedule_1") | !exists("Schedule_2")) {
  message("Reading original .dta files ...")
  Schedule_1 <- read_dta("C:/Users/Lab User/Desktop/WorkBench Holyweek/Data/Schedule 1.dta")
  Schedule_2 <- read_dta("C:/Users/Lab User/Desktop/WorkBench Holyweek/Data/Schedule 2.dta")
}

# Convert to data.table (no copy)
setDT(Schedule_1)
setDT(Schedule_2)

# ---- 1. Extract Large Taxpayer TINs from Schedule 2 (2018–2022) ----
large_s2_subset <- Schedule_2[
  taxstation == "LARGE TAXPAYERS OFFICE" & return_year %in% 2018:2022
]

large_tins <- unique(large_s2_subset$tin_no)
cat("\nNumber of unique Large Taxpayer TINs (from Schedule 2):", length(large_tins), "\n")

# ---- 2. Filter Schedule 1 for those TINs and same years ----
large_s1_subset <- Schedule_1[
  tin_no %in% large_tins & return_year %in% 2018:2022
]
cat("Rows in filtered Schedule 1:", nrow(large_s1_subset), "\n")
cat("Rows in filtered Schedule 2:", nrow(large_s2_subset), "\n")

# Optional: free original heavy frames if needed
# rm(Schedule_1, Schedule_2); gc()

# ---- 3. Create unified edge tables (with counter party TINs) ----
# Schedule 1: Sales (taxpayer sells to purchaser)
edges_s1 <- large_s1_subset[, .(
  taxpayer_tin = tin_no,                          # seller
  counterparty_tin = sch1_purchaser_tin,           # buyer
  transaction_type = "Sales",
  direction = "Out",                               # from taxpayer to counter party
  amount = sch1_amount,
  vat_paid = sch1_vat_paid,
  invoice_dt = sch1_invoice_dt,
  invoice_no = sch1_invoice_no,
  goods_desc = sch1_goods_description,
  activity_isic4_id = sch1_activity_isic4_id,
  return_year,
  return_month,
  return_dt = returndate,
  return_from_dt = returnfrom_date,
  return_to_dt = returntodate,
  rtn_id, rtn_no,
  created_dt = sch1_created_dt,
  updated_dt = sch1_updated_dt,
  prev_rtn_id = sch1_prev_rtn_id,
  is_amended = sch1_is_amended,
  location_id
)]

# Schedule 2: Purchases (taxpayer buys from supplier)
edges_s2 <- large_s2_subset[, .(
  taxpayer_tin = tin_no,                          # buyer
  counterparty_tin = sch2_supplier_tin,            # seller
  transaction_type = "Purchases",
  direction = "In",                               # from counterparty to taxpayer
  amount = sch2_amount,
  vat_paid = sch2_vat_paid,
  invoice_dt = sch2_invoice_dt,
  invoice_no = sch2_invoice_no,
  goods_desc = sch2_goods_description,
  activity_isic4_id = sch2_activity_isic4_id,
  return_year,
  return_month,
  return_dt = rtn_dt,
  return_from_dt = rtn_from_dt,
  return_to_dt = rtn_to_dt,
  rtn_id, rtn_no,
  created_dt = sch2_created_dt,
  updated_dt = sch2_updated_dt,
  prev_rtn_id = sch2_prev_rtn_id,
  is_amended = sch2_is_amended,
  taxstation
)]

# ---- 4. Combine into one edge list ----
all_edges <- rbindlist(list(edges_s1, edges_s2), use.names = TRUE, fill = TRUE)
cat("\nTotal edges (transactions):", nrow(all_edges), "\n")

# ---- 5. Node counts (unique taxpayers) ----
# Nodes = all unique TINs appearing in either column
all_tins <- unique(c(all_edges$taxpayer_tin, all_edges$counterparty_tin))
cat("Total unique taxpayers (nodes):", length(all_tins), "\n")

# Count of large taxpayer TINs actually present in edges
large_in_edges <- unique(all_edges$taxpayer_tin)  # because we filtered on these
cat("Large taxpayers (taxpayer_tin) in edges:", length(large_in_edges), "\n")

# Count of counter parties (may include other large taxpayers, other taxpayers, or missing)
counterparties <- unique(all_edges$counterparty_tin)
cat("Unique counterparty TINs:", length(counterparties), "\n")

# Remove rows with missing counterpart TIN (can't form an edge)
missing_counter <- sum(is.na(all_edges$counterparty_tin) | all_edges$counterparty_tin == "")
cat("Rows with missing counterparty TIN (removed):", missing_counter, "\n")
all_edges_clean <- all_edges[!is.na(counterparty_tin) & counterparty_tin != ""]

cat("Clean edges (with valid counterparty):", nrow(all_edges_clean), "\n")

# ---- 6. Save to CSV ----
fwrite(all_edges_clean, "large_taxpayers_edges_2018_2022.csv")
cat("\nCSV file written: large_taxpayers_edges_2018_2022.csv\n")

# ---- Optional: save separate tables if needed ----
fwrite(edges_s1, "edges_sales_only.csv")
fwrite(edges_s2, "edges_purchases_only.csv")

# ---- 7. Quick summary ----
cat("\n-------- SUMMARY --------\n")
cat("Large taxpayer TINs extracted:", length(large_tins), "\n")
cat("Total edges (clean):", nrow(all_edges_clean), "\n")
cat("Total nodes (unique TINs):", length(all_tins), "\n")
cat("Memory-efficient script completed.\n")


# Run the cleaning with the CORRECT date conversion
keep_cols <- c("taxpayer_tin", "counterparty_tin", "amount", "vat_paid",
               "invoice_dt", "activity_isic4_id", "return_year", "return_month")

# ---- Sales edges ----
edges_s1_clean <- edges_s1[, ..keep_cols]
edges_s1_clean <- edges_s1_clean[!is.na(amount) & !is.na(vat_paid)]
edges_s1_clean[, invoice_dt := as.IDate(invoice_dt, format = "%d/%m/%Y")]  # FIXED

# ---- Purchases edges ----
edges_s2_clean <- edges_s2[, ..keep_cols]
edges_s2_clean <- edges_s2_clean[!is.na(amount) & !is.na(vat_paid) & !is.na(activity_isic4_id)]
edges_s2_clean[, invoice_dt := as.IDate(invoice_dt, format = "%d/%m/%Y")]  # FIXED

# ---- Verify ----
cat("Sales dates:\n")
print(head(edges_s1_clean$invoice_dt, 10))
cat("\nPurchases dates:\n")
print(head(edges_s2_clean$invoice_dt, 10))

# ---- Now convert to integer encoding for compact export ----
# Create unified node mapping
all_nodes <- unique(c(edges_s1_clean$taxpayer_tin, 
                      edges_s1_clean$counterparty_tin,
                      edges_s2_clean$taxpayer_tin, 
                      edges_s2_clean$counterparty_tin))
node_to_id <- setNames(seq_along(all_nodes) - 1, all_nodes)  # 0-indexed for Python

# Save mapping
fwrite(data.table(tin = names(node_to_id), node_id = node_to_id), "node_id_mapping.csv")

# Convert edges to integer IDs
edges_s1_int <- edges_s1_clean[, .(
  src = node_to_id[taxpayer_tin],
  dst = node_to_id[counterparty_tin],
  amount, vat_paid,
  invoice_dt = as.integer(invoice_dt),  # days since 1970-01-01
  activity_isic4_id,
  return_year, return_month
)]

edges_s2_int <- edges_s2_clean[, .(
  src = node_to_id[taxpayer_tin],
  dst = node_to_id[counterparty_tin],
  amount, vat_paid,
  invoice_dt = as.integer(invoice_dt),
  activity_isic4_id,
  return_year, return_month
)]

# ---- Export ----
fwrite(edges_s1_int, "sales_edges_int_2018_2022.csv")
fwrite(edges_s2_int, "purchases_edges_int_2018_2022.csv")

# ---- Final summary ----
cat("\n--- FINAL SUMMARY ---\n")
cat("Sales edges:", nrow(edges_s1_int), "\n")
cat("Purchases edges:", nrow(edges_s2_int), "\n")
cat("Total nodes:", length(all_nodes), "\n")

cat("\nFile sizes:\n")
cat("  Sales:", file.info("sales_edges_int_2018_2022.csv")$size / 1e6, "MB\n")
cat("  Purchases:", file.info("purchases_edges_int_2018_2022.csv")$size / 1e6, "MB\n")
cat("  Node mapping:", file.info("node_id_mapping.csv")$size / 1e6, "MB\n")
