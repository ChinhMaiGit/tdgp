# Causal Inference for Compliance-Gated Administrative Data

**A Tested Data-Generating Process Approach with Application to Municipal Energy Benchmarking**

Xuân Chính Mai · [ORCID 0009-0000-7102-9546](https://orcid.org/0009-0000-7102-9546)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20686630.svg)](https://doi.org/10.5281/zenodo.20686630)

This repository is the reproducibility bundle for the paper above. It contains
the LaTeX source and compiled PDF, the analysis code that produces every figure
and table, and the result summaries that back the reported numbers.

The paper develops the **Tested Data-Generating Process (TDGP)** pipeline — a
three-stage method that locates a compliance gate in a dataset's missingness
structure, formulates a candidate causal graph from that structure alone, and
tests it against the institutional documents that define the reporting
obligation. It is demonstrated on the **Chicago Energy Benchmarking** dataset
through a **2×3 factorial experiment** crossing two pipelines (causally-blind vs.
causal) with three model families (OLS, XGBoost, hierarchical Bayes), giving six
models, M1–M6.

---

## Repository structure

```
.
├── README.md                 this file
├── REPLICATION.md            full step-by-step run order + figure-to-script map
├── LICENSE                   CC BY-NC 4.0
├── CITATION.cff              citation metadata
├── pyproject.toml            project + dependency declaration
├── uv.lock                   fully pinned dependency lock
├── .python-version           interpreter version (3.14)
│
├── data/                     frozen input data
│   ├── data_full.parquet     exact dataset vintage used in the paper
│   └── SNAPSHOT.txt          provenance, checksum, source-drift note
│
├── tex/                      paper source and compiled output
│   ├── main.tex              entry point (\input sections/*.tex)
│   ├── main.pdf              compiled draft (the published artifact)
│   ├── sections/             00_abstract.tex … 08_section.tex, references.tex
│   └── figures/              24 figures (PNG) embedded in the draft
│
├── scripts/                  analysis code, grouped by stage
│   ├── data/                 eda_dataset.py, candidate_dag.py, shadow_matrix_values.py
│   ├── experiments/          pipeline.py, models.py, experiment.py, reporting.py
│   ├── inference/            m3_inference.py, m6_inference.py
│   ├── diagnostic/           m2_shap.py
│   ├── counterfactual/       counterfactual.py
│   ├── sensitivity/          sensitivity.py
│   └── mediator/             upstream_to_mediator.py, mediator_to_outcome.py
│
└── results/                  analysis outputs (routed here by stage)
    ├── docs/                 human-readable result/description summaries  [shipped]
    └── experiments/          results.pkl, table_comparison.tex            [shipped]
```

The remaining `results/` subdirectories (`data/`, `inference/`, `diagnostic/`,
`counterfactual/`, `sensitivity/`) are **created on demand** by the scripts that
write into them, so they are not stored in the repository when empty.

---

## Requirements

- **Python 3.14** (recorded in `.python-version`).
- **No internet connection required.** The exact input data is frozen in
  `data/data_full.parquet` and loaded automatically; the City of Chicago
  Socrata API is used only as a fallback when that file is absent.

The exact dependency set is pinned in `pyproject.toml` and `uv.lock`. Reproduce
the environment with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Core packages: `numpy`, `scipy`, `pandas`, `polars`, `matplotlib`, `requests`,
`xgboost`, `optuna`, `scikit-learn`, `shap`, `pymc`, `arviz`.

---

## Reproducing the results

See **[`REPLICATION.md`](REPLICATION.md)** for the complete run order, the
figure-to-script map, and the minimal end-to-end command sequence. The essential
ordering is:

1. **Run the experiment first.** `scripts/experiments/experiment.py` is the main
   entry point — it fits all six models and writes `results/experiments/results.pkl`,
   the comparison table, the summary figures, and the two MCMC traces under
   `results/inference/`.
2. **Then the trace consumers.** The inference, counterfactual, sensitivity, and
   M3-diagnostic scripts read those traces, so they must run after step 1.
3. **Standalone anytime.** The EDA, DAG, mediator, and M2-SHAP scripts re-fetch
   the data themselves and can run in any order.

Scripts anchor all paths to their own location (`__file__`), so they can be run
from any working directory; shared-module imports (`from pipeline import ...`)
resolve from `scripts/experiments/`.

---

## Outputs

Each script routes its outputs into `results/<stage>/` and creates the target
directory automatically. The three stdout-only scripts
(`data/shadow_matrix_values.py`, `mediator/*.py`) print to the console and write
no files.

**Excluded by design:** the NetCDF posterior traces `m3_trace.nc` and
`m6_trace.nc` (~8.3 GB combined) are not stored in the repository. They are
regenerated into `results/inference/` as a side effect of running
`scripts/experiments/experiment.py`.

---

## Data

The analysis uses the [Chicago Energy Benchmarking](https://data.cityofchicago.org/resource/xq83-jr8c.json)
dataset (28,329 building-year records, 2014–2023), published by the City of
Chicago Data Portal (dataset `xq83-jr8c`).

The **exact vintage used in the paper is frozen** in `data/data_full.parquet`
and loaded automatically by the scripts, so results reproduce deterministically.
Its provenance and SHA-256 checksum are recorded in
[`data/SNAPSHOT.txt`](data/SNAPSHOT.txt).

> **Source drift.** The live API no longer returns two columns the paper uses
> (`all_other_fuel_use_kbtu`, `weather_normalized_source_eui_kbtu_sq_ft`), so a
> fresh fetch does *not* reproduce the paper. The frozen snapshot is therefore
> the authoritative input; the API path remains only as a fallback.

The dataset is published under the City of Chicago's open-data terms and is
redistributed here unmodified for reproducibility; it is **not** covered by this
repository's CC BY-NC 4.0 license.

---

## License

Released under the **Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0)** license — see [`LICENSE`](LICENSE). You may share
and adapt the paper and code with attribution, but **not for commercial
purposes**. The underlying dataset is covered by the City of Chicago's terms.

---

## Citation

This repository is archived on Zenodo with DOI
[10.5281/zenodo.20686630](https://doi.org/10.5281/zenodo.20686630).
Structured citation metadata is in [`CITATION.cff`](CITATION.cff); please cite
the DOI for the version you used.
