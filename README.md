# Completed Draft Snapshot

Snapshot date: 2026-06-13.

Paper: *Causal Inference for Compliance-Gated Administrative Data: A Tested
Data-Generating Process Approach with Application to Municipal Energy
Benchmarking* (Xuan Chinh Mai).

The paper compares causally-blind vs causally-informed modelling pipelines
across a 2x3 factorial experiment (2 pipelines x 3 model types = 6 models:
M1-M6) on the Chicago Energy Benchmarking dataset.

## What is present

This bundle contains the **compiled draft** (`tex/`), the **flat-staged
figures** embedded in that draft (`tex/figures/`), the **analysis scripts** that
produced them (`scripts/`), human-readable result/description summaries
(`results/docs/`), and an experiments checkpoint carrying the six-model
cross-validation numbers and LaTeX comparison table (`results/experiments/`).

```
tex/                  self-contained, arXiv-uploadable bundle
  main.tex            entry point (\input sections/*.tex)
  main.pdf            compiled draft (rebuilt by arXiv from source; not uploaded)
  sections/           00_abstract.tex, 01_introduction.tex,
                      02_section.tex … 08_section.tex, references.tex
  figures/            24 figures (PNG) referenced by the draft (flat, by basename)
scripts/              generators, grouped by source folder
  data/                         eda_dataset.py, shadow_matrix_values.py,
                                candidate_dag.py
  experiments/                  experiment.py, models.py, pipeline.py, reporting.py
  inference/                    m3_inference.py, m6_inference.py
  counterfactual/               counterfactual.py
  sensitivity/                  sensitivity.py
  diagnostic/                   m2_shap.py
  mediator/                     mediator_to_outcome.py, upstream_to_mediator.py
results/                script outputs are routed here, grouped by source
  experiments/        results.pkl, table_comparison.tex           (populated)
  data/                                                           (empty)
  diagnostic/                                                     (empty)
  inference/
    m3/                                                           (empty)
  counterfactual/                                                 (empty)
  sensitivity/                                                    (empty)
  paper_figures/                                                  (empty)
  docs/               human-readable result/description summaries (populated)
```

## What is NOT present

- **Most result artifacts** (EDA plots, SHAP figures, counterfactual/sensitivity
  figures, MCMC traces, paper PDFs in `results/paper_figures/`) have not been
  captured in this snapshot. Their directories exist as scaffolding but are
  empty. Run the relevant generators to populate them.
- **NetCDF posterior traces** (`m3_trace.nc`, `m6_trace.nc`) are excluded by
  design (~8.3 GB combined). Regenerate by running
  `scripts/experiments/experiment.py` (full MCMC), which writes both to
  `results/inference/` as a side effect of fitting M3 and M6.
- **Three scripts referenced by earlier snapshots** —
  `generate_figure_2b3d.py`, `render_draft.py`, and `xgb_standard_pipeline.py` —
  are no longer part of this tree. The six-model figures and comparison table
  are produced by `experiments/experiment.py` (and regenerated standalone from
  `results.pkl` by `experiments/reporting.py`); the LaTeX/PDF render that
  `render_draft.py` used to produce is already frozen under `tex/`.
## Environment

The exact dependency set is pinned in `pyproject.toml` and `uv.lock`, with the
interpreter recorded in `.python-version` (Python 3.14). To reproduce the
environment with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Core packages: `numpy`, `scipy`, `pandas`, `polars`, `matplotlib`, `requests`,
`xgboost`, `optuna`, `scikit-learn`, `shap`, `pymc`, `arviz`.

## License and citation

This work is released under the **Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0)** license — see [`LICENSE`](LICENSE). You may
share and adapt the paper and code with attribution, but **not for commercial
purposes**. The underlying Chicago Energy Benchmarking dataset is published
separately by the City of Chicago under its own terms.

Citation metadata is in [`CITATION.cff`](CITATION.cff). The paper and this
repository are archived on Zenodo; cite the Zenodo DOI for the specific version
you used (the DOI will be added to `CITATION.cff` once the deposit is minted).

## Regenerating results

See **[`REPLICATION.md`](REPLICATION.md)** for the full step-by-step run order,
the figure-to-script map, and the minimal end-to-end command sequence. In brief:

1. `scripts/experiments/experiment.py` is the main entry point — it fits all six
   models, writes `results/experiments/results.pkl`, the comparison table, the
   summary figures, and the two MCMC traces under `results/inference/`.
2. The inference, counterfactual, sensitivity, and M3 diagnostic scripts consume
   those traces, so run the experiment first.
3. The EDA, DAG, mediator, and M2-SHAP scripts are standalone (they re-fetch the
   data) and can run in any order.

Scripts under `scripts/` write outputs into `results/<source>/` using paths
anchored to `__file__`, so they can be run from any working directory.
Shared-module imports (`from pipeline import ...`) resolve from
`scripts/experiments/`.

Running a full analysis also requires a live network connection for the
Chicago Socrata data fetch.

Stdout-only scripts (`data/shadow_matrix_values.py`, `mediator/*.py`) print to
the console and do not write files.
