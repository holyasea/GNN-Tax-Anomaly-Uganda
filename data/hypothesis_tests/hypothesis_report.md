# Hypothesis Tests: GNN Anomaly Detection in Uganda's EFRIS Data (rev 3)

All tests use global seed 7142. Each model flags its top 10% of the 614 registered
taxpayers (n = 62). P-values are reported to four decimal places. Changes after
rev 1: H1 tests community structure only; H2 is decided on held-out graph
features (the rev-1 test on IF's own training features was circular); H4a is
decided by top-k overlap (Spearman secondary); H4b is decided by a permutation
test (binomial secondary).

## Summary of decisions

| Hypothesis | Test statistic | p-value | Decision |
|---|---|---|---|
| H1 | modularity z = 438.2976 | 0.0010 | reject |
| H2 | 3 of 5 held-out graph features significant (min adj. p = 0.0004) | 0.0004 | reject |
| H3 | min KW p = 0.0000; unique GNN flags 63 vs 100 expected | 0.0000 | fail to reject |
| H4a | top-k overlap: if_score vs rgcn_score 13 (p = 0.0065); if_score vs gat_score 8 (p = 0.2756); rgcn_score vs gat_score 26 (p = 0.0001) | 0.0001 | reject |
| H4b | observed consensus = 35 (expected 17.5) | 0.0001 | reject |

## H1: Network topology

The simplified EFRIS transaction graph contains 165094 nodes and 445730 edges, with an observed Louvain modularity of 0.4874. Against 1000 configuration-model graphs sharing the same degree sequence, modularity averaged 0.3676 (SD 0.0003), a z-score of 438.2976 and an empirical p-value of 0.0010. The degree distribution is heavy-tailed (power-law alpha = 2.2065, xmin = 1), though a pure power law is rejected by the bootstrap KS test (p = 0.0000); the tail is best described as heavy-tailed rather than exactly power-law. **Decision: Reject H1.**

The transaction network shows significant, non-random community structure (modularity 0.4874 vs 0.3676 under the configuration-model null, z = 438.30). The degree distribution is heavy-tailed (alpha = 2.21); the bootstrap KS test rejects a pure power law (p = 0.0000), so the tail is best described as heavy-tailed rather than exactly power-law.

## H2: Baseline detection (held-out graph features)

The rev-1 version of this test compared Isolation Forest-flagged taxpayers against the remaining 552 on the same features IF was trained on. The supervisor noted this is circular: IF flags extremes of its own features, so significance is guaranteed and the test validates nothing. That financial comparison is retained only as a descriptive profile (H2_financial_profile_descriptive.csv); no decision rests on it.

The decision now rests on five graph features IF never saw: log_out_degree, log_in_degree, page_rank, reciprocated_share, max_counterparty_share.

| Feature | Median (flagged) | Median (non-flagged) | p (raw) | p (BH) |
|---|---|---|---|---|
| log_out_degree | 4.513 | 5.455 | 0.0017 | 0.0029 |
| log_in_degree | 3.892 | 4.025 | 0.9092 | 0.9092 |
| page_rank | 6.747e-06 | 6.761e-06 | 0.5125 | 0.6406 |
| reciprocated_share | 0.5893 | 0.3556 | 0.0002 | 0.0004 |
| max_counterparty_share | 0.3749 | 0.2531 | 0.0001 | 0.0004 |

3 of the 5 held-out features differ significantly after Benjamini-Hochberg correction. **Decision: Reject H2.**

IF flags are associated with anomalies on graph features outside the IF feature set, so the baseline's flags generalize beyond being extreme on their own training features.

## H3: Graph-model detection

Flag-group sizes (mutually exclusive): IF-unique = 47, GNN-unique = 63, consensus = 35, non-flagged = 469.

Kruskal-Wallis tests across groups:

| Feature | KW chi-squared | df | p |
|---|---|---|---|
| n_unique_partners | 77.5542 | 3 | 0.0000 |
| purchase_to_sales_ratio | 4.7962 | 3 | 0.1873 |
| var_purchase_value | 13.0593 | 3 | 0.0045 |

Flags unique to the GNNs (exactly one of RGCN/GAT, not IF): observed = 63, expected under independence = 100.2. **Decision: Fail to reject H3.**

The flag groups differ structurally (smallest Kruskal-Wallis p = 0.0000), but the number of taxpayers uniquely flagged by the GNNs (63) does not exceed the independence expectation (100.2), so H3 is not rejected.

## H4a: Score independence

Primary measure: overlap between each pair of models' top-62 flag sets, tested against a permutation null (10,000 permutations).

| Model pair | Observed overlap | Expected | Jaccard | p (perm) |
|---|---|---|---|---|
| if_score vs rgcn_score | 13 | 6.26 | 0.117 | 0.0065 |
| if_score vs gat_score |  8 | 6.26 | 0.069 | 0.2756 |
| rgcn_score vs gat_score | 26 | 6.26 | 0.265 | 0.0001 |

Secondary measure: full-range Spearman correlations (dominated by the near-zero score mass, so not used for the decision): if_score vs rgcn_score r = -0.105; if_score vs gat_score r = +0.038; rgcn_score vs gat_score r = +0.233. **Decision: Reject H4a.**

The Isolation Forest shares no more top-ranked taxpayers with either GNN than chance, but the two GNNs share signal with each other: their top-k overlap is significantly higher than the permutation null, consistent with the H4b finding that model agreement is concentrated in the RGCN-GAT pair.

## H4b: Consensus precision

Observed consensus cases (flagged by two or more models): 35. Expected under independence: 17.52. Exact binomial test p = 0.0001. Permutation test (10,000 permutations of all three score vectors): permuted consensus mean 17.52 (SD 3.38), p = 0.0001. Total pairwise overlaps: observed = 47 vs expected = 18.78. **Decision: Reject H4b.**

---

### Data note

Two large variance columns (`var_sale_value`, `var_purchase_value`) are read by `fread` as character and are coerced to numeric before analysis. The simplified undirected graph has 445730 edges; this is lower than the 471,167 directed edge count because reciprocal buyer/seller edges collapse to a single undirected edge. The consensus count here (35) uses a mutually exclusive flag partition and may differ slightly from values quoted elsewhere under a non-exclusive definition.

