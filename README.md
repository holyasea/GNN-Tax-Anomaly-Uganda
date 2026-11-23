# Graph Neural Networks for Anomaly Detection in Unlabeled Data
## A Proof of Concept Study Using Uganda's Electronic Fiscal Receipting and Invoicing System (EFRIS) Data

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.8+-orange.svg)](https://pyg.org/)

This repository contains the empirical methodology, data processing workflows, model training code, and working paper for the study:  
**"Graph Neural Networks for Anomaly Detection in Unlabeled Data: A Proof of Concept Study Using Uganda's Electronic Fiscal Receipting and Invoicing System Data"**.

This project represents an applied statistical research collaboration between **Makerere University** (School of Statistics and Planning) and the **Uganda Revenue Authority** (Strategy and Risk Management Department).

---

## Authors and Affiliations

- **Holyweek Asea** (Corresponding Author)  
  *Enterprise Risk Management Division, Strategy and Risk Management Department, Uganda Revenue Authority*  
  *Department of Statistical Methods, School of Statistics and Planning, Makerere University*  
  Emails: `hasea@ura.go.ug` | `asea.holyweek@students.mak.ac.ug` | `holyasea@gmail.com`
- **Dr. Francis F. Bbosa, PhD**  
  *Department of Statistical Methods, School of Statistics and Planning, Makerere University*  
  Email: `francis.bbosa@mak.ac.ug`
- **Dr. Douglas Candia Andabati, PhD**  
  *Department of Planning and Applied Statistics, School of Statistics and Planning, Makerere University*  
  Email: `douglascandia@gmail.com` | `douglas.candia@mak.ac.ug`

---

## Executive Summary

Detecting tax compliance anomalies without historical audit labels is an essential operational challenge for revenue authorities in developing economies. Standard unsupervised tabular models evaluate corporate taxpayers as isolated records and fail to capture networked evasion schemes such as carousel loops and shell company supply chains. 

This study evaluates the empirical feasibility of unsupervised Graph Neural Networks (GNNs) for surfacing relational anomalies in unlabeled administrative data from Uganda's Electronic Fiscal Receipting and Invoicing System (EFRIS).

### Core Empirical Findings
- **Data Scope:** 5 years of administrative transaction records (2018 to 2022).
  - **17,511,132** sales invoice records (216.5 trillion UGX in sales turnover, 35.4 trillion UGX output VAT).
  - **5,049,092** purchase invoice records (97.8 trillion UGX in purchase value, 17.0 trillion UGX invoiced input VAT).
- **Network Topology:** Directed multi-relational trade network spanning **165,094** unique commercial entities and **471,167** directed trade edges (445,730 undirected) centered on **614** focal large corporate taxpayers.
- **Model Consensus:** Surfaced **35 consensus anomalies** flagged by two or more independent models ($p < 0.0001$, permutation test), with **6 taxpayers flagged simultaneously across all three detectors**.
- **Revenue Exposure:** The severe purchase-to-sales asymmetry consensus cohort accounts for **3,019.8 billion UGX** in invoiced input VAT, offering high-priority candidate leads for risk-based tax compliance audits.
- **Controlled Synthetic Injection Benchmark (36 runs):** Combining Isolation Forest with Graph Attention Network (GAT) embeddings achieves **0.98 recall on circular trading rings** and **0.97 on reciprocal shell pairs** (compared to only 0.20 and 0.05 for purely tabular models).
- **Blinded Auditor Review Set:** **45 cases** comprising 35 consensus cases + 10 size-matched non-flagged control cases shuffled without labels (`data/auditor_review_blinded.csv`) for URA compliance officers to evaluate discrimination.

---

## Technical Architecture

```text
[ Administrative EFRIS Data (2018-2022) ]
       │
       ├── Sales Invoices (17.5M records, 216.5T UGX)
       └── Purchase Invoices (5.0M records, 97.8T UGX)
       │
       ▼
[ Multi-Relational Graph Construction ]
  • Vertices: 165,094 entities
  • Edges: 471,167 directed transaction links (sells_to, buys_from)
  • Focal Population: 614 large taxpayers
       │
       ├─────────────────────────────────┬────────────────────────────────┐
       ▼                                 ▼                                ▼
[ Detector 1: Tabular ]           [ Detector 2: R-GCN ]            [ Detector 3: GAT ]
  • Extended Isolation Forest       • 2-layer Relational GCN         • 2-layer Multi-Head GAT
  • 10 financial features           • Link reconstruction error      • Self-attention weights
  • 500 isolation trees             • Reciprocal masking             • K-means clustering
       │                                 │                                │
       └─────────────────────────────────┼────────────────────────────────┘
                                         ▼
                       [ Consensus Ensemble (>= 2 Flags) ]
                         • 35 Consensus Risk Cases
                         • 6 Universal Flag Overlaps
                                         │
                                         ▼
                    [ Blinded Auditor Review Set (45 Cases) ]
                      • 35 Flagged Consensus Cases
                      • 10 Matched Control Cases
```

---

## Repository Structure

```text
.
├── data/                                 # Analysis code, models and summary tables
│   ├── 01_preprocessing_features_isolation_forest.R   # Feature extraction & Extended IF
│   ├── 02_phase4_analysis.R / .py                     # Post-model score aggregation
│   ├── 03_remaining_analysis.R                        # Longitudinal & demographic checks
│   ├── 04_hypothesis_tests.R                          # Formal hypothesis tests (H1-H4b)
│   ├── 05_injection_benchmark.py                      # 36-run synthetic injection benchmark
│   ├── 06_if_graph_baseline.R                         # Hand-built graph features baseline
│   ├── 07_regenerate_plots.R                          # Publication-grade figures generator
│   ├── 08_ego_networks_review.R                       # Ego-network extraction & case cards
│   ├── gnn_pipeline_actual.py                         # PyTorch Geometric GNN models (R-GCN & GAT)
│   ├── StatComp4_TaxAnomaly.cpp                       # Fast C++ statistical computing routines
│   ├── process_log.txt                                # Master experimental execution log
│   ├── auditor_review_blinded.csv                     # 45-case blinded URA auditor review set
│   ├── case_cards_consensus_flagged.csv               # Detailed metrics for 35 consensus cases
│   ├── case_cards_control_matched.csv                 # Detailed metrics for 10 control cases
│   ├── injection_benchmark_summary.csv                # Benchmark recall@k across all 8 detectors
│   ├── model_comparison_actual.json                   # Hyperparameters, graph metadata & seeds
│   ├── Plots/                                         # Generated anomaly plots & score distributions
│   └── ego_networks/                                  # Visualized ego networks for candidate leads
│
├── paper/                                # Working paper & executive documentation
│   ├── main.tex                                       # Two-column working paper LaTeX source
│   ├── main.pdf                                       # Compiled 9-page research paper
│   ├── references.bib                                 # Complete BibTeX citations
│   ├── figures/                                       # High-resolution figures referenced in paper
│   ├── URA_Research_Summary_EFRIS_GNN.docx            # Executive brief for URA leadership & staff
│   └── generate_ura_summary.py                        # Automated generator for the Word document
│
├── .gitignore                            # Excludes massive raw data & credentials
└── README.md                             # Project overview and reproduction guide
```

---

## Replication Pipeline

To replicate the experimental results and models, execute the pipeline scripts in the following order:

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/holyasea/GNN-Tax-Anomaly-Uganda.git
cd GNN-Tax-Anomaly-Uganda

# Python Environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scipy pandas numpy scikit-learn matplotlib
```

### 2. Tabular Feature Extraction and Isolation Forest
```bash
Rscript data/01_preprocessing_features_isolation_forest.R
```
Extracts 10 taxpayer-level features from EFRIS invoice tables and fits the Extended Isolation Forest (500 trees, 18 grid configurations).

### 3. Multi-Relational Graph Neural Network Training
```bash
python data/gnn_pipeline_actual.py
```
Constructs the directed multi-relational trade network, fits 2-layer R-GCN and 2-layer multi-head GAT across three random seeds (7142, 2718, 3141) with negative-sampling link reconstruction, and generates seed-averaged anomaly scores.

### 4. Synthetic Anomaly Injection Benchmark
```bash
python data/05_injection_benchmark.py
```
Executes the 36-run BOND-style injection experiment across three structural fraud archetypes (missing traders, cyclic trading rings, and reciprocal shell pairs) and computes recall@k across all 8 detector configurations.

### 5. Hypothesis Testing and Statistical Validation
```bash
Rscript data/04_hypothesis_tests.R
```
Evaluates formal hypothesis decisions (H1: Network Modularity, H2: Baseline Outlier Generalization, H3: GNN Specificity, H4a: Score Orthogonality, H4b: Consensus Significance via 10,000 permutations).

### 6. Ego Network Case Analysis
```bash
Rscript data/08_ego_networks_review.R
```
Constructs local ego networks for consensus taxpayers and generates the **45-case Blinded Auditor Review Set** (`auditor_review_blinded.csv`).

---

## Summary of Detection Models

| Detector | Architecture | Feature / Topological Input | Primary Detection Capability |
| :--- | :--- | :--- | :--- |
| **D1** | Deterministic Filter | `Sales == 0` and `Purchases > tau` | Purchase-only filers with zero sales |
| **D2** | Extended Isolation Forest | 10 Financial & Accounting Features | Gross financial and turnover outliers |
| **D3** | Augmented Isolation Forest| Financial + Hand-Built Graph Centralities | Local connectivity and degree outliers |
| **D5** | IF + R-GCN Embeddings | Relational Convolutions (sells / buys) | Asymmetric multi-relational structures |
| **D6** | IF + GAT Embeddings | Multi-Head Self-Attention Embeddings | **Circular trading loops & shell pairs (98% / 97% recall)** |
| **Consensus**| Multi-Model Ensemble | Jointly flagged by >= 2 detectors | **35 statistically validated candidate leads** |

---

## Confidentiality and Data Governance Notice

The raw transaction data used in this study represents proprietary, confidential administrative tax declarations governed by the **Uganda Revenue Authority Act**, the **Tax Procedures Code Act**, and the **Value Added Tax Act Cap 344**. 

In compliance with statutory taxpayer privacy requirements:
- Raw administrative invoice tables and individual taxpayer identification numbers (TINs) are withheld from public distribution.
- All taxpayer identifiers are anonymized to persistent numeric node identifiers (e.g., `Node 600`, `Node 339`).
- Synthetic injection scripts and anonymized node feature vectors are provided to allow independent validation of all mathematical algorithms and GNN architectures.

---

## Working Paper Citation

If you use this code, methodology, or findings in your research or tax administration work, please cite:

```bibtex
@article{asea2026gnn,
  author    = {Asea, Holyweek and Bbosa, Francis F. and Andabati, Douglas Candia},
  title     = {Graph Neural Networks for Anomaly Detection in Unlabeled Data: 
               A Proof of Concept Study Using Uganda's Electronic Fiscal Receipting 
               and Invoicing System Data},
  journal   = {Working Paper Series, Makerere University and Uganda Revenue Authority},
  year      = {2026},
  url       = {https://github.com/holyasea/GNN-Tax-Anomaly-Uganda}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
