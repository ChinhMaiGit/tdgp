# Replication Guide

How to run the scripts in `scripts/` to regenerate every result, figure, and
table reported in the paper *Causal Inference for Compliance-Gated
Administrative Data* (Chinh Mai).

The paper's core artifact is a **2×3 factorial experiment** — two pipelines
(causally-blind vs causal) × three model families (frequentist / ML / Bayesian)
= six models **M1–M6** — fit to the Chicago Energy Benchmarking dataset. All
other scripts produce the supporting EDA, diagnostic, inference, counterfactual,
and sensitivity material.

---

## 1. Prerequisites

### Environment

- **Python 3.14** (the committed bytecode is `cpython-314`).
- **No internet connection required.** The exact input data is frozen in
  `data/data_full.parquet` and loaded automatically. The City of Chicago
  Socrata API (`https://data.cityofchicago.org/resource/xq83-jr8c.json`) is
  used only as a fallback when that file is absent — and note that the live
  API has since dropped two columns the paper uses, so a fresh fetch no longer
  reproduces the paper (see `data/SNAPSHOT.txt`).

### Packages

No `pyproject.toml`/lockfile ships with this snapshot. Install:

```bash
pip install numpy scipy pandas polars matplotlib requests \
            scikit-learn xgboost optuna shap pymc arviz
```

(`shap` is only needed for the M2 SHAP diagnostic; `pymc`/`arviz` only for the
Bayesian models M3/M6 and everything downstream of their traces.)

### How scripts resolve paths

Every script anchors its inputs and outputs to its own location (`__file__`), so
**you can run them from any working directory** — the examples below use the
repo root (`D:\projects\TDGP`). Shared-module imports (`from pipeline import …`,
`from eda_dataset import …`) resolve automatically because each script inserts
its sibling source folder onto `sys.path`.

Outputs are routed to `results/<source-folder>/`, mirroring the script layout.
Those directories already exist as empty scaffolding and will be populated as
you run each script.

---

## 2. Configuration (fixed; matches the paper)

These constants live in `scripts/experiments/pipeline.py` and reproduce the
published numbers — do not change them to replicate:

| Constant | Value | Meaning |
|----------|-------|---------|
| `SEED` | `20` | Seeds Python, NumPy, XGBoost, PyMC chains, Optuna |
| `N_TRIALS` | `60` | Optuna trials per fold for the tuned XGB models |
| `APP_TOKEN` | *(committed)* | Socrata app token for the data fetch |

---

## 3. Run order

All scripts read the frozen `data/data_full.parquet` snapshot, so none requires
network access. There are three tiers: **Tier A** scripts are standalone,
**Tier B** is the main experiment and must run before **Tier C**, which consumes
artifacts produced by Tier B (`results.pkl` and/or the MCMC traces).

### Tier A — standalone descriptive / structural outputs

Run in any order; each loads the data snapshot independently.

```bash
# EDA: 6 figures -> results/data/  + text report to stdout
python scripts/data/eda_dataset.py > results/docs/data/eda_report.txt

# Candidate DAG figure -> results/data/fig_08_candidate_dag.png
python scripts/data/candidate_dag.py

# Stdout-only descriptive numbers (print to console; write no files)
python scripts/data/shadow_matrix_values.py
python scripts/mediator/upstream_to_mediator.py
python scripts/mediator/mediator_to_outcome.py
```

### Tier B — the main 2×3 experiment (run this before Tier C)

```bash
python scripts/experiments/experiment.py
```

This single entry point:

1. Fetches the data.
2. Fits all six models (M1–M6), runs cross-validation, and the deployment test
   on the 5,510 non-compliant building-years.
3. Writes **`results/experiments/results.pkl`** (metrics, out-of-fold
   predictions, preprocessing context, deployment predictions).
4. Writes the two **MCMC traces** `results/inference/m3_trace.nc` and
   `results/inference/m6_trace.nc` — **these are the inputs Tier C needs.**
5. Prints the console comparison table, writes the LaTeX
   `results/experiments/table_comparison.tex`, and saves four summary figures
   (`fig_comparison`, `fig_quintile`, `fig_deployment`, `fig_pred_vs_actual`) to
   `results/experiments/`.

> **Runtime & disk.** The two Bayesian models run full MCMC (4 chains) and the
> tuned XGB models run 60 Optuna trials per fold — budget tens of minutes to a
> few hours depending on hardware. The traces are large (the paper notes
> ~8.3 GB combined) and are written to `results/inference/`. They are
> intentionally excluded from `results.pkl` and from this repo snapshot.
> Ensure adequate free disk before running.

If you only need to **regenerate the experiment figures/tables** from an
existing `results.pkl` (without refitting), run the reporting module standalone:

```bash
python scripts/experiments/reporting.py
```

### Tier C — inference, diagnostics, counterfactuals, sensitivity

These load the Tier B traces (`m3_trace.nc` / `m6_trace.nc`) and/or the data
snapshot; run them only after Tier B has produced the traces. Order among them
does not matter.

```bash
# Posterior diagnostics & posterior-predictive checks
python scripts/inference/m6_inference.py     # -> results/inference/
python scripts/inference/m3_inference.py     # -> results/inference/m3/

# do-operator counterfactuals (analytic, from the M6 posterior)
python scripts/counterfactual/counterfactual.py   # -> results/counterfactual/

# Prior/spec sensitivity sweep around the M6 specification
python scripts/sensitivity/sensitivity.py    # -> results/sensitivity/

# M2 SHAP attribution (refits M2 on full CB data; needs `shap`)
python scripts/diagnostic/m2_shap.py         # -> results/diagnostic/
```

> `m2_shap.py` does **not** depend on Tier B — it reloads the data snapshot and
> refits M2 on the full causally-blind set for SHAP stability. It is grouped here
> only because it is a diagnostic; it can equally run in Tier A.

---

## 4. Figure → script map

Output filename (in `results/<folder>/`) and the script that produces it. These
basenames match the figures embedded in the draft under `tex/figures/`.

| Figure(s) | Produced by | Output dir |
|-----------|-------------|------------|
| `fig_01_records_per_year`, `fig_02_ghg_distribution`, `fig_03_missingness_by_variable`, `fig_04_shadow_matrix`, `fig_05_floor_area_by_compliance`, `fig_06_chicago_energy_rating_by_year`, `fig_07_property_type_frequency` | `data/eda_dataset.py` | `results/data/` |
| `fig_08_candidate_dag` | `data/candidate_dag.py` | `results/data/` |
| `fig_comparison`, `fig_quintile`, `fig_deployment`, `fig_pred_vs_actual` + `table_comparison.tex` | `experiments/experiment.py` (or `experiments/reporting.py` from `results.pkl`) | `results/experiments/` |
| `fig_coef_forest`, `fig_type_intercepts`, `fig_ppc`, `fig_trace`, `fig_rhat_ess`, `fig_pareto_k`, `fig_loo_pit`, `fig_energy`, `fig_m6_deployment` | `inference/m6_inference.py` | `results/inference/` |
| `fig_m3_combined`, `fig_m3_trace`, `fig_m3_rhat_ess`, `fig_m3_mediator_slopes`, `fig_m3_upstream_vs_m6` | `inference/m3_inference.py` | `results/inference/m3/` |
| `fig_cf1_compliance`, `fig_cf2_reduction`, `fig_trend` | `counterfactual/counterfactual.py` | `results/counterfactual/` |
| `fig_sensitivity_forest` | `sensitivity/sensitivity.py` | `results/sensitivity/` |
| `fig_shap_beeswarm`, `fig_shap_bar`, `fig_shap_group`, `fig_shap_combined` | `diagnostic/m2_shap.py` | `results/diagnostic/` |

The paper's `tex/figures/` also contains hand-authored figures
(`figure2_1`, `figure4_7`, `figure4_8`) that are not script-generated.

---

## 5. Minimal end-to-end replication

```bash
# from repo root
# 1. main experiment: six models + traces + comparison table/figures
python scripts/experiments/experiment.py

# 2. everything downstream of the traces
python scripts/inference/m6_inference.py
python scripts/inference/m3_inference.py
python scripts/counterfactual/counterfactual.py
python scripts/sensitivity/sensitivity.py
python scripts/diagnostic/m2_shap.py

# 3. descriptive / EDA material (independent; run anytime)
python scripts/data/eda_dataset.py > results/docs/data/eda_report.txt
python scripts/data/candidate_dag.py
python scripts/data/shadow_matrix_values.py
python scripts/mediator/upstream_to_mediator.py
python scripts/mediator/mediator_to_outcome.py
```

After step 2 completes, `results/` holds every figure referenced by the draft;
the headline six-model comparison table is `results/experiments/table_comparison.tex`.

---

## 6. Notes & gotchas

- **Reproducibility of the numbers** depends on the fixed `SEED = 20` and on the
  frozen `data/data_full.parquet` snapshot, which is the exact vintage used in
  the paper (~21,714 CB / ~21,406 causal rows; verified against
  `results/experiments/results.pkl`). Do **not** delete the snapshot to force a
  live fetch: the Socrata feed has since dropped two columns the paper uses, so a
  fresh fetch silently changes the inputs (the scripts back-fill the missing
  columns as null) and the metrics will differ. See `data/SNAPSHOT.txt`.
- **Three scripts referenced in the old README are not present** in this working
  tree (`generate_figure_2b3d.py`, `render_draft.py`, `xgb_standard_pipeline.py`
  were deleted). The experiment figures/table they would have produced are
  covered by `experiments/experiment.py` + `experiments/reporting.py`; the
  Markdown→LaTeX build (`render_draft.py`) is not needed because the frozen
  LaTeX/PDF already live in `tex/`.
- **Stdout-only scripts** (`data/shadow_matrix_values.py`, both `mediator/*.py`)
  print to the console and write no files — redirect them if you want a record.
- If a Tier C script errors with a missing-trace message, Tier B has not been
  run (or did not finish writing `results/inference/*.nc`).
