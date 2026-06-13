"""
generate_figure_2b3d.py
=======================
Publication figure + LaTeX-table generator for the 2x3 six-model factorial
(M1-M6) of the arXiv paper "Causal Inference for Compliance-Gated
Administrative Data".

This is the port of the legacy ``generate_figures.py`` (four-model design:
Baseline OLS / IPW OLS / IPW XGBoost / IPW HBM) onto the completed draft's
2x3 design. The two designs are not interchangeable: the legacy script varied
only IPW on/off over one upstream feature set, whereas the 2x3 design crosses
the causal *pipeline* (causally-blind vs causal) with the functional *form*
(OLS / XGBoost / Hierarchical Bayes):

                  Frequentist      ML               Bayesian
    Causally-blind  M1 CB OLS       M2 CB XGB        M3 CB Bayes
                    ALL features, KFold, no IPW
    Causal          M4 Causal OLS   M5 Causal XGB    M6 Causal Bayes
                    upstream only, GroupKFold, IPW back-door adjustment

Rather than re-derive that full specification (the causally-blind feature set,
structural zeros, the two CV protocols, IPW) a second time, this script reuses
the authoritative experiments pipeline as its model engine, so every number
matches the paper exactly:
    scripts/experiments/pipeline.py  -- fetch, preprocess, feature sets, IPW
    scripts/experiments/models.py    -- run_all_models -> M1..M6 + deployment
    scripts/experiments/reporting.py -- six-model comparison plotters, LaTeX table

Figures written to results/paper_figures/ (publication PDFs):
  fig_model_comparison_2x3.pdf  -- R2 / MdAPE / ELPD with IPW-effect deltas, M1..M6
  fig_mape_quintile_2x3.pdf     -- MdAPE by floor-area quintile, all six models
  fig_deployment_2x3.pdf        -- deployment density on non-compliant records
  fig_pred_vs_actual_2x3.pdf    -- predicted vs actual heatmaps (2x3 grid)
  fig_mcmc_diagnostics_m6.pdf   -- M6 R-hat + ESS per scalar parameter
  fig_posterior_forest_m6.pdf   -- M6 global coefficient posteriors (89% CI)
  fig_alpha_forest_m6.pdf       -- M6 property-type intercepts (55 trained types)
Plus the six-model LaTeX comparison table, printed to stdout.

The design-independent figures from the legacy script (annual coverage,
shadow-matrix heatmap, propensity distribution, IPW weights/balance,
sensitivity forest, GHG projection) do not change with the 2x3 design and are
produced by the dedicated folder generators -- data/, sensitivity/,
counterfactual/, inference/ -- so they are not duplicated here.

Runs the full pipeline (Optuna tuning for M2/M5, MCMC for M3/M6); expect a
long runtime (roughly 40-60 min). Requires the live Socrata data fetch.

Run (from drafts/complete/):
    uv run python scripts/generate_figure_2b3d.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

# ---------------------------------------------------------------------------
# Make the experiments package importable: it is the shared 2x3 model engine.
# This file lives at drafts/complete/scripts/; the package is in
# drafts/complete/scripts/experiments/.
# ---------------------------------------------------------------------------
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS / "experiments"))

from pipeline import APP_TOKEN, SEED, fetch_data            # noqa: E402
from models import run_all_models                           # noqa: E402
from reporting import (                                     # noqa: E402
    validate_ctx,
    print_comparison,
    print_latex_table,
    plot_main_comparison,
    plot_mape_by_quintile,
    plot_deployment_test,
    plot_pred_vs_actual,
)

# Outputs go to the bundle's results/paper_figures/ directory.
_RESULTS = _SCRIPTS.parent / "results" / "paper_figures"
_RESULTS.mkdir(parents=True, exist_ok=True)

# Publication style. Applied after importing reporting (whose module-level
# rcParams are sans-serif / 150 dpi) so these settings win for every figure.
PUB_STYLE = {
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
}

BAYES_COLOUR = "#8172B2"   # matches M6 Causal Bayes in MODEL_META


def _savefig(name: str) -> None:
    path = str(_RESULTS / name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# arviz helpers (DataTree API, matching the project's arviz version)
# ---------------------------------------------------------------------------

def get_rhat(trace, param):
    return float(az.rhat(trace, var_names=[param])[param].values)


def get_ess(trace, param):
    return float(az.ess(trace, var_names=[param])[param].values)


def get_posterior_values(trace, param):
    return trace["posterior"][param].values.flatten()


# ---------------------------------------------------------------------------
# Bayesian figures from the live M6 trace
# ---------------------------------------------------------------------------

def fig_mcmc_diagnostics_m6(trace, scalar_params, label):
    """R-hat and ESS per scalar parameter for the M6 Causal Bayes model."""
    rhat_vals = [get_rhat(trace, p) for p in scalar_params]
    ess_vals  = [get_ess(trace, p) for p in scalar_params]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colours_r = ["#DD5143" if v > 1.01 else "#4C72B0" for v in rhat_vals]
    axes[0].bar(scalar_params, rhat_vals, color=colours_r, alpha=0.85)
    axes[0].axhline(1.01, color="red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("R-hat")
    axes[0].set_title("R-hat per scalar parameter")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    colours_e = ["#DD5143" if v < 400 else "#43997A" for v in ess_vals]
    axes[1].bar(scalar_params, ess_vals, color=colours_e, alpha=0.85)
    axes[1].axhline(400, color="red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("ESS")
    axes[1].set_title("Effective sample size")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(f"{label} --- MCMC diagnostics", fontsize=12)
    fig.tight_layout()
    _savefig("fig_mcmc_diagnostics_m6.pdf")


def fig_posterior_forest_m6(trace, scalar_params, label):
    """Posterior means and 89% credible intervals for the M6 global scalars."""
    means, lo, hi = [], [], []
    for p in scalar_params:
        v = get_posterior_values(trace, p)
        means.append(np.mean(v))
        lo.append(np.percentile(v, 5.5))
        hi.append(np.percentile(v, 94.5))

    err_lo = [m - l for m, l in zip(means, lo)]
    err_hi = [h - m for m, h in zip(means, hi)]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = range(len(scalar_params))
    ax.errorbar(means, y_pos, xerr=[err_lo, err_hi], fmt="o",
                color="#4C72B0", capsize=4, markersize=5, linewidth=1.5)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scalar_params)
    ax.set_xlabel("Posterior value")
    ax.set_title(f"{label} --- posterior means and 89\\% credible intervals")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    _savefig("fig_posterior_forest_m6.pdf")


def fig_alpha_forest_m6(trace, all_types, label):
    """Ranked property-type intercepts alpha_j for the M6 model.

    Only the types present in the compliant training set are estimated
    (55 of the 63 the dataset records); the figure ranks those.
    """
    alpha_draws = trace["posterior"]["alpha"].values.reshape(-1, len(all_types))
    alpha_means = alpha_draws.mean(axis=0)
    alpha_lo = np.percentile(alpha_draws, 5.5, axis=0)
    alpha_hi = np.percentile(alpha_draws, 94.5, axis=0)
    alpha_bar = float(get_posterior_values(trace, "alpha_bar").mean())

    order = np.argsort(alpha_means)
    sorted_types = [all_types[i] for i in order]
    sorted_means = alpha_means[order]
    sorted_lo = alpha_lo[order]
    sorted_hi = alpha_hi[order]

    fig, ax = plt.subplots(figsize=(7, 10))
    y_pos = range(len(sorted_types))
    ax.errorbar(sorted_means, y_pos,
                xerr=[sorted_means - sorted_lo, sorted_hi - sorted_means],
                fmt="o", color=BAYES_COLOUR, capsize=2, markersize=3, linewidth=1)
    ax.axvline(alpha_bar, color="grey", linestyle="--", linewidth=1,
               label=r"global mean $\bar{\alpha}$")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_types, fontsize=6)
    ax.set_xlabel(r"$\alpha_j$ (log-GHG intercept at mean characteristics)")
    ax.set_title(f"{label} --- property-type intercepts "
                 f"({len(all_types)} trained types)")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    _savefig("fig_alpha_forest_m6.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Set global seeds before run_all_models (mirrors experiment.py): PyMC
    # jitter+adapt_diag and any NumPy draws read the global state.
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"  Random seeds set: Python={SEED}, NumPy={SEED}")

    # -- Data + six-model engine ----------------------------------------------
    print("\n[1/4] Fetching Chicago Energy Benchmarking data ...")
    df_raw = fetch_data(APP_TOKEN)

    print("\n[2/4] Running the 2x3 factorial (M1-M6) + deployment test ...")
    print("      (Optuna tuning for M2/M5, MCMC for M3/M6 -- this is slow)")
    results, preds, ctx, traces, deploy_preds = run_all_models(df_raw)
    validate_ctx(ctx)

    # Apply publication style now that reporting has been imported.
    plt.rcParams.update(PUB_STYLE)

    # -- Model-design-dependent figures (reuse the six-model plotters) --------
    print("\n[3/4] Rendering 2x3 comparison figures ...")
    plot_main_comparison(results,            str(_RESULTS / "fig_model_comparison_2x3.pdf"))
    plot_mape_by_quintile(preds, ctx,        str(_RESULTS / "fig_mape_quintile_2x3.pdf"))
    plot_deployment_test(deploy_preds,       str(_RESULTS / "fig_deployment_2x3.pdf"))
    plot_pred_vs_actual(preds, ctx, results, str(_RESULTS / "fig_pred_vs_actual_2x3.pdf"))

    # -- Bayesian figures from the live M6 trace ------------------------------
    print("\n[4/4] Rendering M6 Bayesian figures ...")
    m6_key = "M6 Causal Bayes"
    if m6_key in traces:
        trace_m6 = traces[m6_key]
        scalar_params = ["alpha_bar", "sigma_alpha",
                         "beta_A", "beta_B", "beta_T", "beta_Y", "sigma"]
        all_types = [str(t) for t in trace_m6["posterior"].coords["type"].values]
        fig_mcmc_diagnostics_m6(trace_m6, scalar_params, m6_key)
        fig_posterior_forest_m6(trace_m6, scalar_params, m6_key)
        fig_alpha_forest_m6(trace_m6, all_types, m6_key)
    else:
        print(f"  WARNING: '{m6_key}' trace not returned; skipping Bayesian figures.")

    # -- Tables ---------------------------------------------------------------
    print_comparison(results)
    print_latex_table(results)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total / 60:.1f} min")
    print(f"  Figures saved to: {_RESULTS}")
    print("  Design-independent figures (EDA, propensity, IPW balance, "
          "sensitivity, projection) come from the data/, sensitivity/, "
          "counterfactual/ and inference/ generators.")
    print("  Done!")


if __name__ == "__main__":
    main()
