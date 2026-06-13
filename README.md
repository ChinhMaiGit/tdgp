# Completed Draft Snapshot

Snapshot date: 2026-06-13.

Paper: *Causal Inference for Compliance-Gated Administrative Data: A Tested
Data-Generating Process Approach with Application to Municipal Energy
Benchmarking* (Chinh Mai).

The paper compares causally-blind vs causally-informed modelling pipelines
across a 2x3 factorial experiment (2 pipelines x 3 model types = 6 models:
M1-M6) on the Chicago Energy Benchmarking dataset.

## What is present

This bundle contains the **compiled draft** (`tex/`), the **flat-staged
figures** embedded in that draft (`figures/`), the **analysis scripts** that
produced them (`scripts/`), human-readable result/description summaries
(`results/docs/`), and an experiments checkpoint carrying the six-model
cross-validation numbers and LaTeX comparison table (`results/experiments/`).

```
tex/
  main.tex            entry point (\input sections/*.tex)
  main.pdf            compiled draft
  sections/           00_abstract.tex, 01_introduction.tex,
                      02_section.tex … 08_section.tex, references.tex
figures/              24 figures (PNG) referenced by the draft (flat, by basename)
scripts/              generators, grouped by source folder
  generate_figure_2b3d.py       2x3 six-model publication figures + LaTeX table
  render_draft.py                Markdown-to-LaTeX build tool
  xgb_standard_pipeline.py      standard ML benchmark + SHAP mediator attribution
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
  design (~8.3 GB combined). Regenerate by running `scripts/experiments/models.py`
  (full MCMC), which writes both to `results/inference/`.
- **The `drafts/` source tree.** `render_draft.py` reads Markdown sources from
  `drafts/final/` and writes to `drafts/render/` — neither directory exists
  here. The LaTeX and PDF under `tex/` represent the frozen render output.
- **Dependency specification.** No `pyproject.toml` or `uv.lock` is included.
  Required packages: `numpy`, `pandas`, `polars`, `matplotlib`, `requests`,
  `xgboost`, `optuna`, `scikit-learn`, `pymc`, `arviz`.

## Regenerating results

Scripts under `scripts/` write outputs into `results/<source>/` using paths
anchored to `__file__`, so they can be run from any working directory.
Shared-module imports (`from pipeline import ...`) resolve from
`scripts/experiments/`.

Running a full analysis also requires a live network connection for the
Chicago Socrata data fetch.

Stdout-only scripts (`data/shadow_matrix_values.py`, `mediator/*.py`,
`xgb_standard_pipeline.py`) print to the console and do not write files.
