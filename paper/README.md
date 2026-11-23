# Paper: GNN-Based VAT Anomaly Detection in Uganda's EFRIS

## Files

| File | Description |
|------|-------------|
| `main.tex` | Full LaTeX source |
| `references.bib` | BibTeX bibliography |
| `figures/` | All figures (PNG, 300 dpi) |

## Figure Inventory

| File | Used in Paper | Description |
|------|--------------|-------------|
| `fig_temporal.png` | Figure 1 | Monthly invoice volume over time |
| `fig_community.png` | Figure 3 | Louvain community sizes |
| `fig_if_convergence.png` | Figure 4 | Isolation Forest convergence |
| `fig_if_scores.png` | Figure 5 | IF anomaly score distribution |
| `fig_feat_importance.png` | Figure 6 | Feature importance for flagged taxpayers |
| `fig_if_feature_space.png` | Figure 7 | Flagged vs. normal in feature space |
| `fig_gnn_loss.png` | Figure 8 | GNN training loss curves |
| `fig_score_hist.png` | Figure 9 | Score distributions by model |
| `fig_pairwise.png` | Figure 10 | Pairwise anomaly score scatterplots |
| `fig_boxplots.png` | Figure 11 | Feature profiles by model group |
| `fig_heatmap.png` | Figure 12 | Anomaly rank heatmap (top 30) |
| `fig_ego_339.png` | Figure 13 | Ego network: Node 339 (all-3 flagged) |
| `fig_ego_270.png` | Figure 14 | Ego network: Node 270 (all-3 flagged) |

## Compilation Instructions

### Prerequisites

Install a full LaTeX distribution:
- **Windows**: [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)
- **Mac**: [MacTeX](https://tug.org/mactex/)
- **Linux**: `sudo apt install texlive-full`

Required LaTeX packages (included in most full distributions):
`geometry`, `graphicx`, `amsmath`, `booktabs`, `caption`, `subcaption`,
`hyperref`, `microtype`, `natbib`, `titlesec`, `fancyhdr`, `xcolor`,
`enumitem`, `multirow`, `tabularx`, `lmodern`, `balance`

### Compile (3-pass for references + citations)

From the `paper/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk` (recommended — handles passes automatically):

```bash
latexmk -pdf -shell-escape main.tex
```

### Output

`main.pdf` — ready-to-submit arXiv preprint (~14 pages, two-column)

### Customisation

- **Author name / affiliation**: Replace `[Author Name]` and
  `[Institution / Affiliation]` in the title block.
- **Email**: Replace `[email@institution.edu]`.
- **Abstract**: Modify to reflect any updates to the analysis.
- **References**: Add additional references to `references.bib`.

## Notes

- All figures are 300 dpi PNG files copied from `data/Plots/` and
  `data/ego_networks/` by the R session.
- Seed 7142 is used globally (R + Python) for full reproducibility.
- The auditor validation section references `auditor_review_blinded.csv`
  (37 cases, blinded); update Section 7 when validation is complete.
