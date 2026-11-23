# ══════════════════════════════════════════════════════════════════════════════
# 08_ego_networks_review.R
# Regenerates the ego-network plots for all 45 blinded-review cases in the
# elaborate style of the Plots showcase figure (directed edges, arrowheads,
# edge colour = relation type, edge width = transaction value, focal node
# highlighted, node labels).
#
# Case list: data/auditor_review_blinded.csv (Nov 5 21:31 - authoritative;
#   35 consensus + 10 controls, shuffled for review). Filenames
#   ego_<case>_<node>.png follow THIS file's case numbering. Earlier ego
#   files used a stale ordering (Nov 5 18:03 run and Sep 16 run) and are
#   moved to ego_networks/_stale/.
#
# Blinding: subtitles deliberately exclude model flags so the reviewer cannot
#   tell flagged from control cases. Set show_flags <- TRUE to include them
#   (as in the Plots showcase figure).
#
# Note on partner counts: n_unique_partners in the case cards counts partners
#   in the taxpayer's OWN records (buyers + suppliers). The plotted network
#   also shows relationships recorded by the OTHER side (e.g. other taxpayers
#   reporting sales TO this node), so the plotted node count can exceed the
#   recorded partner count.
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(igraph); library(ggraph)
})

data_dir  <- "C:/Users/Holyweek/Desktop/Project/Project1/data"
ego_dir   <- file.path(data_dir, "ego_networks")
stale_dir <- file.path(ego_dir, "_stale")

blinded <- fread(file.path(data_dir, "auditor_review_blinded.csv"))
edges   <- fread(file.path(data_dir, "edge_list_actual.csv"))

MAX_SHOWN  <- 30   # cap displayed neighbours for hub taxpayers
show_flags <- FALSE

fmt_ugx <- function(x) {
  a <- abs(x)
  if (a >= 1e12) sprintf("%.1fT", x / 1e12) else
  if (a >= 1e9)  sprintf("%.1fB", x / 1e9)  else
  if (a >= 1e6)  sprintf("%.1fM", x / 1e6)  else
                 sprintf("%.0fK", x / 1e3)
}

if (show_flags) {
  scores <- fread(file.path(data_dir, "all_model_scores_actual.csv"))
  TOP <- ceiling(0.10 * nrow(scores))
  rk <- function(x) rank(-x, ties.method = "first") <= TOP
  scores[, `:=`(flag_if = rk(if_score), flag_rg = rk(rgcn_score),
                flag_ga = rk(gat_score))]
}

for (i in seq_len(nrow(blinded))) {
  cn <- blinded$case_num[i]; nd <- blinded$node_id[i]
  e_nd <- edges[(src == nd | dst == nd) & src != dst]
  if (nrow(e_nd) == 0) {
    cat(sprintf("case %d node %d: no edges - skipped\n", cn, nd)); next
  }
  e_nd[, partner := fifelse(src == nd, dst, src)]
  n_linked <- uniqueN(e_nd$partner)
  top   <- e_nd[, .(val = sum(total_amount)), by = partner][order(-val)]
  shown <- head(top$partner, MAX_SHOWN)
  e_show <- e_nd[partner %in% shown,
                 .(value = sum(total_amount)), by = .(src, dst, edge_type)]

  ng <- graph_from_data_frame(e_show, directed = TRUE)
  V(ng)$is_focus <- V(ng)$name == as.character(nd)

  if (show_flags) {
    fl <- scores[node_id == nd]
    flagtxt <- sprintf(" | flags: %s",
      paste(c("IF", "R-GCN", "GAT")[c(fl$flag_if, fl$flag_rg, fl$flag_ga)],
            collapse = ", "))
  } else flagtxt <- ""

  p <- ggraph(ng, layout = "fr") +
    geom_edge_link(aes(width = log10(abs(value) + 1), color = edge_type),
                   arrow = arrow(length = unit(3, "mm"), type = "closed"),
                   end_cap = circle(4.5, "mm")) +
    geom_node_point(aes(size = is_focus, shape = is_focus, color = is_focus)) +
    scale_size_manual(values = c("TRUE" = 8, "FALSE" = 3), guide = "none") +
    scale_shape_manual(values = c("TRUE" = 17, "FALSE" = 16), guide = "none") +
    scale_color_manual(values = c("TRUE" = "#d62728", "FALSE" = "grey30"),
                       guide = "none") +
    geom_node_text(aes(label = name), repel = TRUE, size = 3.2) +
    scale_edge_color_manual(values = c(sells_to = "#1f77b4", buys_from = "#ff7f0e"),
                            name = "Edge type") +
    scale_edge_width(range = c(0.4, 2.4), guide = "none") +
    labs(title = sprintf("Ego Network - Case %d (Node %d)", cn, nd),
         subtitle = sprintf("%d linked nodes (%d shown, ranked by value) | sales %s, purchases %s UGX%s",
                            n_linked, length(shown),
                            fmt_ugx(blinded$total_sales[i]),
                            fmt_ugx(blinded$total_purchases[i]), flagtxt),
         x = NULL, y = NULL) +
    theme_graph(base_family = "sans")

  f <- file.path(ego_dir, sprintf("ego_%02d_node_%d.png", cn, nd))
  png(f, width = 8, height = 6, units = "in", res = 300, type = "cairo")
  print(p); dev.off()
  cat(sprintf("case %2d node %6d: %d linked (%2d shown) -> %s\n",
              cn, nd, n_linked, length(shown), basename(f)))
}

# Move stale ego files (older case numbering) out of the way
new_files <- sprintf("ego_%02d_node_%d.png", blinded$case_num, blinded$node_id)
old <- setdiff(list.files(ego_dir, pattern = "\\.png$"), new_files)
if (length(old)) {
  dir.create(stale_dir, showWarnings = FALSE)
  file.rename(file.path(ego_dir, old), file.path(stale_dir, old))
  cat(sprintf("moved %d stale files to ego_networks/_stale/\n", length(old)))
}
cat("done\n")
