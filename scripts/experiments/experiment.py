"""
grand_test.py
=============
Main entry point for the 2x3 TDGP factorial experiment.

Experiment structure
--------------------
Two pipelines × three modelling approaches = six models.
This implementation uses the **realistic comparison design**:
each pipeline receives exactly the data its analyst would actually use.

                  Frequentist      ML               Bayesian
               ────────────────────────────────────────────────────
Causally-      M1  CB OLS       M2  CB XGB       M3  CB Bayes
blind          no IPW           no IPW           no IPW
               ALL features     ALL features     ALL features
               all obs rows     all obs rows     all obs rows
               KFold            KFold tuned      PSIS-LOO

Causal         M4  Causal OLS   M5  Causal XGB   M6  Causal Bayes
               IPW              IPW              IPW
               upstream only    upstream only    upstream only
               compliant rows   compliant rows   compliant rows
               GroupKFold       GroupKFold tuned PSIS-LOO

Row counts (intentional)
------------------------
Causally-blind : ~21,714 rows — all rows with an observed GHG target.
                 A naïve analyst has no reason to filter on compliance.
Causal         : ~21,406 rows — compliant records with an observed
                 GHG target. The causal analyst knows about the
                 compliance gate and corrects for it via IPW.

The row-count difference is itself part of the argument: the naïve
analyst uses *more* data and still produces the wrong model.

Four intentional differences between pipelines
----------------------------------------------
  1. Feature set    : ALL available columns (CB) vs upstream only (causal)
                      CB includes derived metrics (ghg_intensity, EUIs,
                      energy_star_score, etc.) that the causal analyst
                      excludes as circular per the tested DAG.
  2. Row subset     : all observed-target rows (CB) vs compliant rows only
  3. CV protocol    : KFold (CB, no building grouping) vs
                      GroupKFold on building ID (causal)
  4. IPW correction : none (CB) vs back-door adjustment (causal)

Plus one additional Bayesian distinction:
  5. Model/priors   : flat generic (M3) vs hierarchical calibrated (M6)

Random seed policy
------------------
Full reproducibility is achieved by setting seeds at four levels:

  1. Global Python / NumPy state:
       random.seed(SEED)
       np.random.seed(SEED)
     Set once at the top of __main__ before any library call.
     Controls NumPy operations inside PyMC's jitter+adapt_diag
     initialisation.

  2. PyMC chain seeds (CHAIN_SEEDS in models.py):
     One seed per chain [SEED, SEED+1, SEED+2, SEED+3] passed to
     both pm.sample() and pm.sample_posterior_predictive().

  3. XGBoost random_state=SEED on every XGBRegressor instance.
     Full-data refits and outer fold models use n_jobs=1 for
     deterministic floating-point accumulation.
     Inner Optuna models use n_jobs=-1 for speed (metrics only).

  4. Optuna TPE sampler seeded per fold: seed=SEED+k.

Deployment test (all six models)
---------------------------------
All six models applied to 5,510 non-compliant building-year records.
Log-scale predictions clipped to [-20, 15] before exponentiation.
Median is the primary deployment statistic.

CB deployment imputation:
  Non-compliant buildings receive column-median imputation for
  all features (mediators + derived metrics + water use), exactly
  as the naïve analyst would have done during training.
  This produces miscalibrated but non-zero predictions (M2 no longer
  collapses to near-zero; it predicts ~median_ghg_intensity × area).

Causal deployment:
  Non-compliant buildings receive structural zeros for mediators
  (tested DAG). Causal models remain stable.

Output files
------------
results.pkl            : pickled results, preds, ctx, deploy_preds
table_comparison.tex   : LaTeX comparison table (3-panel)
fig_comparison.png     : 4-panel main comparison figure
fig_quintile.png       : MdAPE by floor-area quintile (log scale)
fig_deployment.png     : deployment test density plot (all 6 models)
fig_pred_vs_actual.png : predicted vs actual heatmaps (2x3 grid)
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import os
import random
import time
import pickle
from pathlib import Path
import numpy as np

# Bundle results directory: drafts/complete/results/experiments/
_OUT = Path(__file__).resolve().parents[2] / "results" / "experiments"
_OUT.mkdir(parents=True, exist_ok=True)

from pipeline import fetch_data, APP_TOKEN, SEED
from models   import run_all_models
from reporting import (
    validate_ctx,
    print_comparison,
    print_latex_table,
    plot_main_comparison,
    plot_mape_by_quintile,
    plot_deployment_test,
    plot_pred_vs_actual,
)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    # ── Set global random seeds before any library call ───────────────────────
    # This must happen before importing or calling anything that draws
    # random numbers, including PyMC model construction.
    #
    # random.seed    : seeds Python's built-in random module
    # np.random.seed : seeds NumPy's global random state, which PyMC uses
    #                  internally for jitter+adapt_diag initialisation
    #
    # Additional model-level seeds are set inside models.py:
    #   - CHAIN_SEEDS = [SEED+0, SEED+1, SEED+2, SEED+3] for PyMC chains
    #   - random_state=SEED for every XGBRegressor instance
    #   - n_jobs=1 for deterministic XGBoost floating-point accumulation
    #   - Optuna TPESampler(seed=SEED+k) per fold
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"  Random seeds set: Python random={SEED}, NumPy={SEED}")
    print(f"  Model-level seeds: XGBoost random_state={SEED}, "
          f"PyMC chains=[{SEED},{SEED+1},{SEED+2},{SEED+3}]")

    t_start = time.time()

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FETCHING DATA")
    print("=" * 70)
    df_raw = fetch_data(APP_TOKEN)

    # ── Step 2: Run all six models and the deployment test ────────────────────
    # run_all_models() handles all preprocessing, model fitting,
    # cross-validation, and the deployment test for all six models.
    #
    # Returns:
    #   results      : dict  model_name -> metrics_dict
    #   preds        : dict  model_name -> oof_pred_log
    #   ctx          : dict  preprocessing context (realistic design)
    #   traces       : dict  Bayesian model name -> PyMC InferenceData
    #   deploy_preds : dict  model_name -> predicted GHG for non-compliant
    #
    # Traces are large (~1-2 GB each). The deployment test is run inside
    # run_all_models() using the live traces before they go out of scope,
    # so deploy_preds is fully populated when run_all_models() returns.
    results, preds, ctx, traces, deploy_preds = run_all_models(df_raw)

    # ── Step 3: Validate context keys ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATING CONTEXT")
    print("=" * 70)
    validate_ctx(ctx)

    # ── Step 4: Save results to disk ──────────────────────────────────────────
    # Traces are excluded from the pickle because they are large (~4-6 GB
    # combined) and already consumed by the deployment test inside
    # run_all_models(). All figures and tables can be regenerated from
    # results.pkl using reporting.py standalone mode.
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_dict = dict(
        results      = results,
        preds        = preds,
        ctx          = ctx,
        deploy_preds = deploy_preds,
        # traces     = traces,   # uncomment if disk space permits (~4-6 GB)
    )
    with open(_OUT / "results.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    print(f"  Saved -> {_OUT / 'results.pkl'}")

    # ── Step 5: Printed comparison tables ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLES")
    print("=" * 70)
    print_comparison(results)
    print_latex_table(results)

    # ── Step 6: Generate all figures ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_main_comparison(
        results   = results,
        save_path = str(_OUT / "fig_comparison.png"),
    )

    plot_mape_by_quintile(
        preds     = preds,
        ctx       = ctx,
        save_path = str(_OUT / "fig_quintile.png"),
    )

    plot_deployment_test(
        deploy_preds = deploy_preds,
        save_path    = str(_OUT / "fig_deployment.png"),
    )

    # results passed for in-panel R2/MdAPE annotations in each subplot
    plot_pred_vs_actual(
        preds     = preds,
        ctx       = ctx,
        results   = results,
        save_path = str(_OUT / "fig_pred_vs_actual.png"),
    )

    # ── Step 7: Runtime and output summary ────────────────────────────────────
    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Total runtime : {minutes}m {seconds}s")
    print(f"  Random seed   : {SEED} (Python, NumPy, XGBoost, PyMC, Optuna)")

    # ── Output file inventory ─────────────────────────────────────────────────
    print(f"\n  Output files")
    print(f"  {'─' * 50}")
    output_files = [
        str(_OUT / "results.pkl"),
        str(_OUT / "table_comparison.tex"),
        str(_OUT / "fig_comparison.png"),
        str(_OUT / "fig_quintile.png"),
        str(_OUT / "fig_deployment.png"),
        str(_OUT / "fig_pred_vs_actual.png"),
    ]
    for fname in output_files:
        exists = os.path.exists(fname)
        status = "OK" if exists else "MISSING"
        size   = (f"{os.path.getsize(fname) / 1024:.0f} KB"
                  if exists else "—")
        print(f"  {fname:<35} [{status}]  {size:>10}")

    # ── Model performance summary ──────────────────────────────────────────────
    print(f"\n  Model performance summary (out-of-sample)")
    print(f"  {'─' * 58}")
    print(f"  {'Model':<25} {'R2(log)':>9} {'MdAPE':>8} {'ELPD':>10}")
    print(f"  {'─' * 58}")
    model_order = [
        "M1 CB OLS", "M2 CB XGB", "M3 CB Bayes",
        "M4 Causal OLS", "M5 Causal XGB", "M6 Causal Bayes",
    ]
    for i, mname in enumerate(model_order):
        if mname in results:
            m = results[mname]
            if i == 3:
                print(f"  {'─' * 58}")
            print(f"  {mname:<25} "
                  f"{m['r2_mean']:>+9.4f} "
                  f"{m['mape_mean']:>7.2f}% "
                  f"{m['elpd']:>+10.0f}")

    # ── IPW effect summary ────────────────────────────────────────────────────
    print(f"\n  IPW effect (causal minus causally-blind, same approach)")
    print(f"  {'─' * 58}")
    print(f"  {'Approach':<20} {'Delta R2':>10} {'Delta MdAPE':>13}")
    print(f"  {'─' * 58}")
    ipw_pairs = [
        ("Frequentist", "M1 CB OLS",   "M4 Causal OLS"),
        ("ML",          "M2 CB XGB",   "M5 Causal XGB"),
        ("Bayesian",    "M3 CB Bayes", "M6 Causal Bayes"),
    ]
    for approach, cb_m, c_m in ipw_pairs:
        if cb_m in results and c_m in results:
            dr2   = results[c_m]["r2_mean"]   - results[cb_m]["r2_mean"]
            dmape = results[c_m]["mape_mean"] - results[cb_m]["mape_mean"]
            print(f"  {approach:<20} {dr2:>+10.4f} {dmape:>+12.2f}%")

    # ── Deployment test summary ───────────────────────────────────────────────
    # Median is the primary statistic.
    # M1 CB OLS mean can be very large due to OLS extrapolation outside
    # the training support for a small number of buildings — documented
    # and expected. The median is unaffected.
    print(f"\n  Deployment test summary (non-compliant buildings)")
    print(f"  Log-scale predictions clipped to [-20, 15] before exponentiation")
    print(f"  Median = primary statistic  |  "
          f"Mean may be inflated by OLS extrapolation outliers")
    print(f"  {'─' * 68}")
    print(f"  {'Model':<25} {'Pipeline':<15} "
          f"{'Median (t CO2e)':>16} {'Mean (t CO2e)':>14}")
    print(f"  {'─' * 68}")

    for i, mname in enumerate(model_order):
        pipeline = "causally-blind" if i < 3 else "causal"
        if i == 3:
            print(f"  {'─' * 68}")

        if mname in deploy_preds and deploy_preds[mname] is not None:
            p        = deploy_preds[mname]
            median_v = np.median(p)
            mean_v   = np.mean(p)

            # Flag large-but-finite mean values from OLS extrapolation
            if not np.isfinite(mean_v):
                mean_str = "       inf (!)"
            elif mean_v > 1_000_000:
                mean_str = f"{mean_v:>11,.0f} (!)"
            elif mean_v > 100_000:
                mean_str = f"{mean_v:>11,.0f}  *"
            else:
                mean_str = f"{mean_v:>14,.0f}"

            print(f"  {mname:<25} {pipeline:<15} "
                  f"{median_v:>16,.0f} "
                  f"{mean_str}")
        else:
            print(f"  {mname:<25} {pipeline:<15} {'FAILED':>16}")

    print(f"""
  Mean annotation key
  {'─' * 40}
  (!)  : mean > 1M metric tons — OLS extrapolation artefact.
  *    : mean > 100K metric tons — moderate extrapolation effect.
  Median is always the correct summary for skewed distributions.

  Interpretation
  {'─' * 58}
  CB miscalibration (M2 CB XGB):
    With ghg_intensity and other derived metrics imputed to
    training medians, M2 predicts roughly (median_ghg_intensity ×
    floor_area) for each non-compliant building. This is still
    causally invalid and produces systematically wrong totals
    because the imputed intensity does not reflect the actual
    (unknown) energy behaviour of non-compliant records.
    No longer near-zero, but still useless for deployment.

  CB partial shift (M1 CB OLS, M3 CB Bayes):
    M1/M3 receive the same median-imputed derived features.
    OLS extrapolation can still inflate the mean for extreme
    upstream combinations. M3 shows apparent stability because
    standardised mediator values are moderate, but the model
    is not reasoning causally — it is simply using the best
    available (but circular) features.

  Causal stability (M4, M5, M6, median ~850–1,100 tons):
    Consistent with the observed compliant building median of
    ~1,018 metric tons. The causal pipeline is deployable on
    the full city building stock because it never relied on
    post-treatment or circular features.

  Row-count note
  {'─' * 58}
  The causally-blind pipeline trained on ~21,714 rows while the
  causal pipeline used ~21,406 rows. The naïve analyst used more
  data and still produced the wrong model — a key part of the
  demonstration.

  Reproducibility
  {'─' * 58}
  All results seeded with SEED={SEED}.
  Exact numerical reproducibility holds on the same machine
  and software versions. Minor floating-point differences may
  appear across different OS, CPU architecture, or library
  versions due to non-associative floating-point arithmetic.
  Package versions are pinned in uv.lock for full environment
  reproducibility.
  """)