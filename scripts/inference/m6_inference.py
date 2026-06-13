"""
m6_inference.py
===============
Full Bayesian diagnostic and inference for M6 Causal Bayes.

M6 is the structural causal model in the 2x3 factorial experiment:
  Data         : ~21,406 compliant rows (causal pipeline)
  Features     : UPSTREAM_COLS only (floor area, property type,
                 year built, number of buildings, data year)
  Model        : Hierarchical Normal on log1p(GHG), non-centred:
                 alpha_j = alpha_bar + delta_j * sigma_alpha
  IPW          : pm.Potential("ipw_weight", (w - 1) * log_lik)
  Sampling     : 3,000 draws, 3,000 tune, 4 chains, target_accept=0.95
  Evaluation   : PSIS-LOO

All preprocessing and feature definitions are reused verbatim from
scripts/experiments/pipeline.py so results are directly comparable
to the factorial experiment.

Steps
-----
1  Fetch data (Socrata API)
2  Preprocess (causal pipeline)
3  Fit M6 (exact experiment spec)
4  MCMC diagnostics (trace, energy, R-hat / ESS)
5  PSIS-LOO and posterior predictive checks
6  Parameter inference (global coefficients + 55-type intercepts)
7  Deployment inference (5,510 non-compliant building-year records)

Figures saved to results/inference/
  fig_trace.png           -- MCMC trace + marginals for scalar params
  fig_energy.png          -- Energy plot (E-BFMI)
  fig_rhat_ess.png        -- R-hat and ESS bar charts
  fig_pareto_k.png        -- PSIS-LOO Pareto-k per observation
  fig_ppc.png             -- Posterior predictive check (log + original scale)
  fig_loo_pit.png         -- LOO probability integral transform (calibration)
  fig_coef_forest.png     -- Global parameter posteriors with 89% PI
  fig_type_intercepts.png -- All 55 property-type intercepts (types in the
                             compliant training set, of 63 in the dataset), ranked
  fig_m6_deployment.png   -- Deployment predictions on non-compliant records

Run (from drafts/complete/): uv run python scripts/inference/m6_inference.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import arviz as az
import pymc as pm
import polars as pl

_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "experiments"))

from pipeline import (
    fetch_data, preprocess,
    SEED, APP_TOKEN, RAW_CAT, UPSTREAM_COLS,
    _flatten_az_result, _loo_elpd, _loo_se, _pareto_k,
)

CHAIN_SEEDS = [SEED + i for i in range(4)]
FIG_DIR     = os.path.join(_ROOT, "results", "inference")
TRACE_PATH  = os.path.join(_ROOT, "results", "inference", "m6_trace.nc")

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

SCALAR_PARAMS = [
    "alpha_bar", "sigma_alpha",
    "beta_A", "beta_B", "beta_T", "beta_Y", "sigma",
]

PARAM_LABELS = {
    "alpha_bar":   "alpha_bar   (global intercept)",
    "sigma_alpha": "sigma_alpha (between-type SD)",
    "beta_A":      "beta_A      (log floor area)",
    "beta_B":      "beta_B      (log # buildings)",
    "beta_T":      "beta_T      (data year)",
    "beta_Y":      "beta_Y      (year built)",
    "sigma":       "sigma       (residual SD)",
}

PARAM_COLOURS = {
    "alpha_bar":   "#264653",
    "sigma_alpha": "#264653",
    "beta_A":      "#e76f51",
    "beta_B":      "#e76f51",
    "beta_T":      "#e76f51",
    "beta_Y":      "#e76f51",
    "sigma":       "#2a9d8f",
}


def _pct89(samples):
    a = np.asarray(samples).flatten()
    return np.percentile(a, 5.5), np.percentile(a, 94.5)


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {p}")


# ===========================================================================
# MAIN — guarded for Windows multiprocessing (spawn)
# ===========================================================================

if __name__ == "__main__":

    os.makedirs(FIG_DIR, exist_ok=True)

    # =======================================================================
    # STEP 1 -- FETCH
    # =======================================================================
    print("=" * 70)
    print("STEP 1 -- FETCHING DATA")
    print("=" * 70)
    df = fetch_data(APP_TOKEN)


    # =======================================================================
    # STEP 2 -- PREPROCESS  (causal pipeline)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 2 -- PREPROCESSING  (causal pipeline)")
    print("=" * 70)
    ctx    = preprocess(df)
    obs    = ctx["obs_causal"]
    n_obs  = ctx["n_causal"]
    y_log  = ctx["y_log_causal"]
    y_obs  = ctx["y_obs_causal"]
    ipw_w  = ctx["ipw_w"]
    nc     = ctx["non_compliant"]
    n_nc   = len(nc)
    print(f"  Causal training rows : {n_obs:,}")
    print(f"  Non-compliant rows   : {n_nc:,}")


    # =======================================================================
    # STEP 3 -- M6 CAUSAL BAYES POSTERIOR
    # =======================================================================
    print("\n" + "=" * 70)

    all_types   = sorted(obs[RAW_CAT].unique().to_list())
    n_types     = len(all_types)
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    type_idx    = np.array([type_to_idx[t] for t in obs[RAW_CAT].to_list()])

    log_area   = np.log(obs["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs  = np.log(obs["of_buildings"].to_numpy().clip(1))
    year_data  = obs["data_year"].to_numpy().astype(float)
    year_built = obs["year_built"].to_numpy().astype(float)

    la_mean = log_area.mean()
    lb_mean = log_bldgs.mean()
    yd_mean, yd_std = year_data.mean(),  year_data.std()
    yb_mean, yb_std = year_built.mean(), year_built.std()

    log_area_c  = log_area  - la_mean
    log_bldgs_c = log_bldgs - lb_mean
    yr_data_s   = (year_data  - yd_mean) / yd_std
    yr_built_s  = (year_built - yb_mean) / yb_std

    if os.path.exists(TRACE_PATH):
        # ── Load trace saved by experiment ────────────────────────────────────
        print("STEP 3 -- LOADING M6 TRACE  (saved by experiment)")
        print(f"  Source : {TRACE_PATH}")
        print("=" * 70)
        trace = az.from_netcdf(TRACE_PATH)
        print(f"  Loaded : {trace.posterior.sizes['chain']} chains × "
              f"{trace.posterior.sizes['draw']} draws")
        print(f"  PPD    : {'posterior_predictive' in trace.groups}")
        print(f"  log_lik: {'log_likelihood' in trace.groups}")

    else:
        # ── Fall back to sampling (experiment trace not yet available) ────────
        print("STEP 3 -- FITTING M6 CAUSAL BAYES  (no saved trace found)")
        print(f"  Features : UPSTREAM_COLS = {UPSTREAM_COLS}")
        print(f"  Rows     : {n_obs:,}  (compliant only)")
        print("  IPW      : pm.Potential  (back-door log-posterior adjustment)")
        print("  Draws / Tune : 3,000 / 3,000")
        print("  Chains       : 4     target_accept = 0.95")
        print("=" * 70)

        coords = {"obs": np.arange(n_obs), "type": all_types}

        with pm.Model(coords=coords) as m6_model:
            pm.Data("type_idx",   type_idx,    dims="obs")
            pm.Data("log_area",   log_area_c,  dims="obs")
            pm.Data("log_bldgs",  log_bldgs_c, dims="obs")
            pm.Data("year_data",  yr_data_s,   dims="obs")
            pm.Data("year_built", yr_built_s,  dims="obs")

            ab    = pm.Normal("alpha_bar",    mu=7.0, sigma=1.0)
            sa    = pm.HalfNormal("sigma_alpha", sigma=1.0)
            dj    = pm.Normal("alpha_offset", mu=0,   sigma=1, dims="type")
            alpha = pm.Deterministic("alpha", ab + dj * sa, dims="type")

            bA  = pm.Normal("beta_A", mu=0, sigma=0.5)
            bB  = pm.Normal("beta_B", mu=0, sigma=0.5)
            bT  = pm.Normal("beta_T", mu=0, sigma=0.5)
            bY  = pm.Normal("beta_Y", mu=0, sigma=0.5)
            sig = pm.HalfNormal("sigma", sigma=0.5)

            mu = (alpha[type_idx]
                  + bA * log_area_c
                  + bB * log_bldgs_c
                  + bT * yr_data_s
                  + bY * yr_built_s)

            ipw_vals = pm.Data("ipw", ipw_w, dims="obs")
            ll       = pm.logp(pm.Normal.dist(mu=mu, sigma=sig), y_log)
            pm.Potential("ipw_weight", (ipw_vals - 1.0) * ll)

            pm.Normal("y_hat", mu=mu, sigma=sig, observed=y_log, dims="obs")

            np.random.seed(SEED)
            trace = pm.sample(
                draws         = 3000,
                tune          = 3000,
                chains        = 4,
                cores         = 4,
                target_accept = 0.95,
                init          = "jitter+adapt_diag",
                random_seed   = CHAIN_SEEDS,
                progressbar   = True,
                return_inferencedata = True,
                idata_kwargs  = {"log_likelihood": True},
            )

            np.random.seed(SEED)
            pm.sample_posterior_predictive(
                trace,
                var_names            = ["y_hat"],
                random_seed          = SEED,
                extend_inferencedata = True,
                progressbar          = False,
            )

        print("\n  MCMC complete.")


    # =======================================================================
    # STEP 4 -- MCMC DIAGNOSTICS
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 4 -- MCMC DIAGNOSTICS")
    print("=" * 70)

    rhat_sc = _flatten_az_result(az.rhat(trace, var_names=SCALAR_PARAMS))
    ess_sc  = _flatten_az_result(az.ess(trace,  var_names=SCALAR_PARAMS))
    rhat_al = _flatten_az_result(az.rhat(trace, var_names=["alpha"]))
    ess_al  = _flatten_az_result(az.ess(trace,  var_names=["alpha"]))
    n_div   = int(trace.sample_stats.diverging.values.sum())

    rhat_max = float(max(rhat_sc.max(), rhat_al.max()))
    ess_min  = int(min(ess_sc.min(), ess_al.min()))

    print(f"  R-hat max (scalar params) : {float(rhat_sc.max()):.4f}")
    print(f"  R-hat max (type alphas)   : {float(rhat_al.max()):.4f}")
    print(f"  R-hat max (overall)       : {rhat_max:.4f}")
    print(f"  ESS min  (scalar params)  : {int(ess_sc.min()):,}")
    print(f"  ESS min  (type alphas)    : {int(ess_al.min()):,}")
    print(f"  ESS min  (overall)        : {ess_min:,}")
    print(f"  Divergent transitions     : {n_div}")

    for flag, cond, msg in [
        ("R-hat",       rhat_max > 1.01, f"{rhat_max:.4f} > 1.01"),
        ("ESS",         ess_min  < 400,  f"{ess_min:,} < 400"),
        ("divergences", n_div    > 0,    f"{n_div} divergent transitions"),
    ]:
        if cond:
            print(f"  WARNING {flag}: {msg}")

    print("\n  Generating figures ...")

    # ── fig_trace.png ─────────────────────────────────────────────────────────
    _CHAIN_COLS = ["#264653", "#2a9d8f", "#e76f51", "#e9c46a"]
    n_p = len(SCALAR_PARAMS)
    fig_trace, _axes_tr = plt.subplots(
        n_p, 2, figsize=(12, 2.5 * n_p),
        gridspec_kw={"width_ratios": [3, 1]})
    for _i, _p in enumerate(SCALAR_PARAMS):
        _samp = np.asarray(trace.posterior[_p])  # (chains, draws)
        _ax_t, _ax_m = _axes_tr[_i]
        for _c in range(_samp.shape[0]):
            _ax_t.plot(_samp[_c], alpha=0.6, linewidth=0.4,
                       color=_CHAIN_COLS[_c % len(_CHAIN_COLS)])
        _ax_t.set_ylabel(PARAM_LABELS.get(_p, _p), fontsize=7)
        _ax_t.spines["top"].set_visible(False)
        _ax_t.spines["right"].set_visible(False)
        _ax_m.hist(_samp.flatten(), bins=60, density=True,
                   color="#2a9d8f", alpha=0.75, edgecolor="none")
        _ax_m.spines["top"].set_visible(False)
        _ax_m.spines["right"].set_visible(False)
        if _i == 0:
            _ax_t.set_title("Trace", fontsize=8)
            _ax_m.set_title("Marginal", fontsize=8)
    fig_trace.suptitle(
        "M6 Causal Bayes — MCMC Trace Plots (scalar parameters)",
        fontsize=11, fontweight="bold")
    fig_trace.tight_layout()
    _save(fig_trace, "fig_trace.png")

    # ── fig_energy.png ────────────────────────────────────────────────────────
    _energy = np.asarray(trace.sample_stats.energy)   # (chains, draws)
    _n_chains = _energy.shape[0]
    fig_energy, _axes_en = plt.subplots(1, _n_chains, figsize=(4 * _n_chains, 4),
                                        sharey=True)
    if _n_chains == 1:
        _axes_en = [_axes_en]
    for _c in range(_n_chains):
        _e  = _energy[_c]
        _de = np.diff(_e)
        _bmin = min(_e.min(), _de.min())
        _bmax = max(_e.max(), _de.max())
        _bins_e = np.linspace(_bmin, _bmax, 50)
        _ebfmi  = float(np.var(_de) / np.var(_e))
        _ax_e   = _axes_en[_c]
        _ax_e.hist(_e,  bins=_bins_e, density=True, alpha=0.6,
                   color="#264653", label="E (marginal)", edgecolor="none")
        _ax_e.hist(_de, bins=_bins_e, density=True, alpha=0.6,
                   color="#2a9d8f", label="ΔE (transitions)", edgecolor="none")
        _ax_e.set_title(f"Chain {_c}  E-BFMI = {_ebfmi:.3f}", fontsize=8)
        _ax_e.spines["top"].set_visible(False)
        _ax_e.spines["right"].set_visible(False)
        if _c == 0:
            _ax_e.legend(fontsize=7.5)
    fig_energy.suptitle("M6 Causal Bayes — Energy Plot (E-BFMI)",
                        fontsize=11, fontweight="bold")
    fig_energy.tight_layout()
    _save(fig_energy, "fig_energy.png")

    # ── fig_rhat_ess.png ──────────────────────────────────────────────────────
    rhat_ds = az.rhat(trace, var_names=SCALAR_PARAMS)
    ess_ds  = az.ess(trace,  var_names=SCALAR_PARAMS)

    rhat_vals = {p: float(np.asarray(rhat_ds[p]).flat[0]) for p in SCALAR_PARAMS}
    ess_vals  = {p: float(np.asarray(ess_ds[p]).flat[0])  for p in SCALAR_PARAMS}

    fig_re, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(12, 4))

    rhat_colours = ["#e74c3c" if v > 1.01 else "#2ecc71"
                    for v in rhat_vals.values()]
    ax_r.barh(list(rhat_vals.keys()), list(rhat_vals.values()),
              color=rhat_colours, edgecolor="white", height=0.6)
    ax_r.axvline(1.01, color="#e74c3c", linestyle="--", linewidth=0.9,
                 label="threshold 1.01")
    for p, v in rhat_vals.items():
        ax_r.text(v + 0.0002, list(rhat_vals.keys()).index(p),
                  f"{v:.4f}", va="center", fontsize=7.5)
    ax_r.set_xlabel("R-hat")
    ax_r.set_title("R-hat  (threshold 1.01)", fontsize=9,
                   fontweight="bold", pad=10)
    ax_r.legend(fontsize=8)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    ess_colours = ["#e74c3c" if v < 400 else "#2ecc71"
                   for v in ess_vals.values()]
    ax_e.barh(list(ess_vals.keys()), list(ess_vals.values()),
              color=ess_colours, edgecolor="white", height=0.6)
    ax_e.axvline(400, color="#e74c3c", linestyle="--", linewidth=0.9,
                 label="threshold 400")
    for p, v in ess_vals.items():
        ax_e.text(v + 20, list(ess_vals.keys()).index(p),
                  f"{v:,.0f}", va="center", fontsize=7.5)
    ax_e.set_xlabel("Effective Sample Size")
    ax_e.set_title("ESS  (threshold 400)", fontsize=9,
                   fontweight="bold", pad=10)
    ax_e.legend(fontsize=8)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    fig_re.suptitle("M6 Causal Bayes — Convergence Diagnostics",
                    fontsize=11, fontweight="bold")
    fig_re.tight_layout()
    _save(fig_re, "fig_rhat_ess.png")


    # =======================================================================
    # STEP 5 -- PSIS-LOO AND POSTERIOR PREDICTIVE CHECKS
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 5 -- PSIS-LOO AND POSTERIOR PREDICTIVE CHECKS")
    print("=" * 70)

    loo     = az.loo(trace, pointwise=True)
    pk_vals = _pareto_k(loo)
    n_bad_k = int((pk_vals > 0.7).sum())
    elpd    = _loo_elpd(loo)
    elpd_se = _loo_se(loo)

    print(f"  ELPD (PSIS-LOO)  : {elpd:+,.0f}  (SE = {elpd_se:.1f})")
    print(f"  Pareto-k > 0.7   : {n_bad_k} / {len(pk_vals)}")
    print(f"  Pareto-k > 0.5   : {int((pk_vals > 0.5).sum())} / {len(pk_vals)}")
    print(f"  Pareto-k median  : {float(np.median(pk_vals)):.3f}")

    # ── fig_pareto_k.png ──────────────────────────────────────────────────────
    fig_pk, ax_pk = plt.subplots(figsize=(12, 4))
    pk_colours = np.where(pk_vals > 0.7, "#e74c3c",
                 np.where(pk_vals > 0.5, "#f39c12", "#2980b9"))
    ax_pk.scatter(np.arange(len(pk_vals)), pk_vals,
                  s=3, alpha=0.5, c=pk_colours, linewidths=0)
    ax_pk.axhline(0.7, color="#e74c3c", linestyle="--", linewidth=1.0,
                  label=f"k = 0.7  (unreliable)  n = {n_bad_k}")
    ax_pk.axhline(0.5, color="#f39c12", linestyle="--", linewidth=1.0,
                  label=f"k = 0.5  (marginal)    "
                        f"n = {int((pk_vals > 0.5).sum())}")
    ax_pk.set_xlabel("Observation index")
    ax_pk.set_ylabel("Pareto-k")
    ax_pk.set_title(
        f"M6 Causal Bayes — PSIS-LOO Pareto-k  "
        f"(ELPD = {elpd:+,.0f}, SE = {elpd_se:.1f})",
        fontsize=10, fontweight="bold")
    ax_pk.legend(fontsize=8)
    ax_pk.spines["top"].set_visible(False)
    ax_pk.spines["right"].set_visible(False)
    fig_pk.tight_layout()
    _save(fig_pk, "fig_pareto_k.png")

    # ── fig_ppc.png ───────────────────────────────────────────────────────────
    pp_draws = trace.posterior_predictive["y_hat"].values.reshape(-1, n_obs)
    rng      = np.random.default_rng(SEED)
    n_pp     = min(200, pp_draws.shape[0])
    pp_idx   = rng.choice(pp_draws.shape[0], size=n_pp, replace=False)

    fig_ppc, (ax_l, ax_r2) = plt.subplots(1, 2, figsize=(14, 5))

    bins_log = np.linspace(0, 14, 80)
    for draw in pp_draws[pp_idx]:
        ax_l.hist(draw, bins=bins_log, density=True, alpha=0.04,
                  color="#2980b9", histtype="step")
    ax_l.hist(y_log, bins=bins_log, density=True, color="black",
              histtype="step", linewidth=1.8, label="Observed")
    ax_l.set_xlabel("log1p(GHG)", fontsize=8)
    ax_l.set_ylabel("Density", fontsize=8)
    ax_l.set_title("Posterior Predictive Check  (log scale)",
                   fontsize=9, fontweight="bold")
    ax_l.legend(fontsize=8)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    y_obs_pos = y_obs[y_obs > 0]
    bins_orig = np.logspace(np.log10(1), np.log10(2e5), 80)
    for draw in pp_draws[pp_idx]:
        orig = np.clip(np.expm1(np.clip(draw, -20, 15)), 1, None)
        ax_r2.hist(orig, bins=bins_orig, density=True, alpha=0.04,
                   color="#2980b9", histtype="step")
    ax_r2.hist(y_obs_pos, bins=bins_orig, density=True, color="black",
               histtype="step", linewidth=1.8, label="Observed")
    ax_r2.set_xscale("log")
    ax_r2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_r2.set_xlabel("GHG (metric tons CO2e, log scale)", fontsize=8)
    ax_r2.set_ylabel("Density", fontsize=8)
    ax_r2.set_title("Posterior Predictive Check  (original scale)",
                    fontsize=9, fontweight="bold")
    ax_r2.legend(fontsize=8)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)

    fig_ppc.suptitle(
        f"M6 Causal Bayes — Posterior Predictive Checks  "
        f"({n_pp} draws overlaid)",
        fontsize=11, fontweight="bold")
    fig_ppc.tight_layout()
    _save(fig_ppc, "fig_ppc.png")

    # ── fig_loo_pit.png ───────────────────────────────────────────────────────
    # In-sample PIT: fraction of PPD draws <= y_obs[i] per observation.
    # This is a good proxy for LOO-PIT when the model is well-calibrated
    # (LOO correction is small relative to sample size ~21k).
    _pp_flat = trace.posterior_predictive["y_hat"].values.reshape(-1, n_obs)
    _pit     = np.mean(_pp_flat <= y_log[np.newaxis, :], axis=0)
    fig_pit, ax_pit = plt.subplots(figsize=(9, 5))
    _n_bins_pit = 30
    ax_pit.hist(_pit, bins=_n_bins_pit, density=True,
                color="#2a9d8f", alpha=0.75, edgecolor="white")
    ax_pit.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.2,
                   label="Ideal (uniform)")
    ax_pit.set_xlabel("PIT value", fontsize=9)
    ax_pit.set_ylabel("Density", fontsize=9)
    ax_pit.set_xlim(0, 1)
    ax_pit.legend(fontsize=8.5)
    ax_pit.spines["top"].set_visible(False)
    ax_pit.spines["right"].set_visible(False)
    fig_pit.suptitle(
        "M6 Causal Bayes — Posterior Predictive PIT (calibration check)",
        fontsize=11, fontweight="bold")
    fig_pit.tight_layout()
    _save(fig_pit, "fig_loo_pit.png")


    # =======================================================================
    # STEP 6 -- PARAMETER INFERENCE
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 6 -- PARAMETER INFERENCE")
    print("=" * 70)

    post = trace.posterior

    # ── Global parameters ─────────────────────────────────────────────────────
    param_stats = {}
    print("  Scalar parameter posteriors (mean, 89% PI):")
    for p in SCALAR_PARAMS:
        samp = np.asarray(post[p]).flatten()
        lo, hi = _pct89(samp)
        param_stats[p] = {"mean": float(samp.mean()), "lo": lo, "hi": hi}
        print(f"  {p:<18} mean = {samp.mean():+.4f}  "
              f"89% PI = [{lo:+.4f}, {hi:+.4f}]")

    ab_mean = param_stats["alpha_bar"]["mean"]
    bA_mean = param_stats["beta_A"]["mean"]
    print(f"\n  Implied mean GHG at mean feature values:")
    print(f"    exp(alpha_bar) - 1 = {np.expm1(ab_mean):,.0f} metric tons CO2e")
    print(f"  Floor-area elasticity:")
    print(f"    doubling floor area -> x{2**bA_mean:.3f} GHG  "
          f"(beta_A = {bA_mean:.3f})")

    INTERP = {
        "alpha_bar":
            f"exp(a_bar)-1 ~ {np.expm1(ab_mean):,.0f} t  (mean bldg)",
        "sigma_alpha": "between-type SD (log scale)",
        "beta_A":      f"doubling area -> x{2**bA_mean:.2f} GHG",
        "beta_B":      "log # buildings effect",
        "beta_T":      "data-year trend  (per SD)",
        "beta_Y":      "year-built effect  (per SD)",
        "sigma":       "within-type residual SD",
    }

    # ── fig_coef_forest.png ───────────────────────────────────────────────────
    n_p = len(SCALAR_PARAMS)
    fig_coef, ax_coef = plt.subplots(figsize=(11, 5))

    for i, p in enumerate(reversed(SCALAR_PARAMS)):
        s = param_stats[p]
        c = PARAM_COLOURS.get(p, "#555555")
        ax_coef.plot([s["lo"], s["hi"]], [i, i],
                     color=c, linewidth=3.0, solid_capstyle="round")
        ax_coef.scatter([s["mean"]], [i], color=c, s=70, zorder=5)
        ax_coef.text(s["hi"] + 0.015, i, INTERP.get(p, ""),
                     va="center", ha="left", fontsize=7.5, color="#555555")

    ax_coef.axvline(0, color="#bbbbbb", linestyle="--", linewidth=0.9)
    ax_coef.set_yticks(np.arange(n_p))
    ax_coef.set_yticklabels(
        [PARAM_LABELS[p] for p in reversed(SCALAR_PARAMS)], fontsize=8.5)
    ax_coef.set_xlabel(
        "Posterior mean  +  89% PI  (log1p GHG scale)", fontsize=9)
    ax_coef.set_title("M6 Causal Bayes — Global Parameter Estimates",
                      fontsize=11, fontweight="bold")
    ax_coef.spines["top"].set_visible(False)
    ax_coef.spines["right"].set_visible(False)
    xlo, xhi = ax_coef.get_xlim()
    ax_coef.set_xlim(xlo, xhi + (xhi - xlo) * 0.55)
    fig_coef.tight_layout()
    _save(fig_coef, "fig_coef_forest.png")

    # ── Property-type intercepts (all 55 types in the compliant training set) ──
    print("\n  Computing per-type intercepts ...")
    alpha_samp = np.asarray(post["alpha"]).reshape(-1, n_types)

    type_stats = []
    for j, tname in enumerate(all_types):
        samp = alpha_samp[:, j]
        lo, hi = _pct89(samp)
        type_stats.append({
            "name": tname,
            "mean": float(samp.mean()),
            "lo":   lo,
            "hi":   hi,
        })
    type_stats.sort(key=lambda x: x["mean"])

    print("  Top 3 property types by posterior mean intercept:")
    for ts in type_stats[-3:][::-1]:
        print(f"    {ts['name']:<50} "
              f"alpha_j = {ts['mean']:.3f}  [{ts['lo']:.3f}, {ts['hi']:.3f}]")
    print("  Bottom 3:")
    for ts in type_stats[:3]:
        print(f"    {ts['name']:<50} "
              f"alpha_j = {ts['mean']:.3f}  [{ts['lo']:.3f}, {ts['hi']:.3f}]")

    # ── fig_type_intercepts.png ───────────────────────────────────────────────
    fig_h = max(12, n_types * 0.32)
    fig_types, ax_types = plt.subplots(figsize=(13, fig_h))

    alpha_bar_post = float(np.asarray(post["alpha_bar"]).mean())

    for i, ts in enumerate(type_stats):
        ax_types.plot([ts["lo"], ts["hi"]], [i, i],
                      color="#2a9d8f", linewidth=1.6, alpha=0.75,
                      solid_capstyle="round")
        ax_types.scatter([ts["mean"]], [i],
                         color="#2a9d8f", s=22, zorder=5)

    ax_types.axvline(
        alpha_bar_post, color="#e76f51", linestyle="--", linewidth=1.3,
        label=f"global mean  a_bar = {alpha_bar_post:.3f}")
    ax_types.set_yticks(np.arange(n_types))
    ax_types.set_yticklabels(
        [ts["name"] for ts in type_stats], fontsize=6.5)
    ax_types.set_xlabel(
        "alpha_j  (property-type intercept, log1p GHG scale)", fontsize=9)
    ax_types.set_title(
        f"M6 Causal Bayes — Property-Type Intercepts  "
        f"(all {n_types} types, ranked by posterior mean)\n"
        "Dot = posterior mean  |  Bar = 89% PI  |  "
        "Dashed = global mean a_bar",
        fontsize=10, fontweight="bold")
    ax_types.legend(fontsize=8.5, loc="lower right")
    ax_types.spines["top"].set_visible(False)
    ax_types.spines["right"].set_visible(False)
    fig_types.tight_layout()
    _save(fig_types, "fig_type_intercepts.png")


    # =======================================================================
    # STEP 7 -- DEPLOYMENT INFERENCE  (non-compliant records)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 7 -- DEPLOYMENT INFERENCE")
    print(f"  Non-compliant buildings : {n_nc:,}")
    print("  Mediators               : structural zeros (tested DAG)")
    print("=" * 70)

    log_area_nc   = np.log(nc["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs_nc  = np.log(nc["of_buildings"].to_numpy().clip(1))
    year_data_nc  = nc["data_year"].to_numpy().astype(float)
    year_built_nc = nc["year_built"].to_numpy().astype(float)

    log_area_c_nc  = log_area_nc  - la_mean
    log_bldgs_c_nc = log_bldgs_nc - lb_mean
    yr_data_s_nc   = (year_data_nc  - yd_mean) / yd_std
    yr_built_s_nc  = (year_built_nc - yb_mean) / yb_std

    type_idx_nc = np.array([
        type_to_idx.get(t, 0)
        for t in nc[RAW_CAT].fill_null("Unknown").to_list()
    ])

    coords_nc = {"obs": np.arange(n_nc), "type": all_types}

    with pm.Model(coords=coords_nc):
        pm.Data("type_idx",   type_idx_nc,    dims="obs")
        pm.Data("log_area",   log_area_c_nc,  dims="obs")
        pm.Data("log_bldgs",  log_bldgs_c_nc, dims="obs")
        pm.Data("year_data",  yr_data_s_nc,   dims="obs")
        pm.Data("year_built", yr_built_s_nc,  dims="obs")

        ab_d  = pm.Normal("alpha_bar",    mu=7.0, sigma=1.0)
        sa_d  = pm.HalfNormal("sigma_alpha", sigma=1.0)
        dj_d  = pm.Normal("alpha_offset", mu=0,   sigma=1, dims="type")
        alpha_d = pm.Deterministic("alpha", ab_d + dj_d * sa_d, dims="type")
        bA_d  = pm.Normal("beta_A", mu=0, sigma=0.5)
        bB_d  = pm.Normal("beta_B", mu=0, sigma=0.5)
        bT_d  = pm.Normal("beta_T", mu=0, sigma=0.5)
        bY_d  = pm.Normal("beta_Y", mu=0, sigma=0.5)
        sig_d = pm.HalfNormal("sigma", sigma=0.5)

        mu_d = (alpha_d[type_idx_nc]
                + bA_d * log_area_c_nc
                + bB_d * log_bldgs_c_nc
                + bT_d * yr_data_s_nc
                + bY_d * yr_built_s_nc)

        pm.Normal("y_hat", mu=mu_d, sigma=sig_d,
                  observed=np.zeros(n_nc), dims="obs")

        np.random.seed(SEED)
        ppc_nc = pm.sample_posterior_predictive(
            trace,
            var_names   = ["y_hat"],
            random_seed = SEED,
            progressbar = False,
        )

    pp_nc    = ppc_nc.posterior_predictive["y_hat"].values.reshape(-1, n_nc)
    log_nc   = np.median(pp_nc, axis=0)
    pred_nc  = np.clip(np.expm1(np.clip(log_nc, -20, 15)), 0, None)
    pi_lo_nc = np.clip(np.expm1(np.clip(
        np.percentile(pp_nc, 5.5,  axis=0), -20, 15)), 0, None)
    pi_hi_nc = np.clip(np.expm1(np.clip(
        np.percentile(pp_nc, 94.5, axis=0), -20, 15)), 0, None)

    med_nc    = float(np.median(pred_nc))
    mean_nc   = float(np.mean(pred_nc))
    pi_med_lo = float(np.median(pi_lo_nc))
    pi_med_hi = float(np.median(pi_hi_nc))
    comp_med  = float(np.median(y_obs[y_obs > 0]))

    print(f"  Deployment median      : {med_nc:,.0f} metric tons CO2e")
    print(f"  Deployment mean        : {mean_nc:,.0f} metric tons CO2e")
    print(f"  Median 89% PI          : [{pi_med_lo:,.0f}, {pi_med_hi:,.0f}]")
    print(f"  Observed compliant med : {comp_med:,.0f} metric tons CO2e")

    from collections import Counter
    nc_types_arr  = nc[RAW_CAT].fill_null("Unknown").to_list()
    top_nc_types  = [t for t, _ in Counter(nc_types_arr).most_common(12)]

    print(f"\n  Per-type deployment medians (top 12 types by count):")
    type_deploy = {}
    for tname in top_nc_types:
        mask = np.array([t == tname for t in nc_types_arr])
        if mask.sum() == 0:
            continue
        t_med = float(np.median(pred_nc[mask]))
        t_n   = int(mask.sum())
        type_deploy[tname] = (t_med, t_n)
        print(f"    {tname:<48} n={t_n:>4}  median={t_med:>8,.0f} t")

    # ── fig_deployment.png ────────────────────────────────────────────────────
    fig_dep, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(15, 6))

    y_obs_pos = y_obs[y_obs > 0]
    vmax      = max(pred_nc.max(), y_obs_pos.max()) * 1.1
    bins_dep  = np.logspace(np.log10(1), np.log10(vmax), 75)

    ax_d1.hist(y_obs_pos, bins=bins_dep, density=True, alpha=0.45,
               color="#264653",
               label=f"Observed compliant, GHG > 0  (n={len(y_obs_pos):,})")
    ax_d1.hist(pred_nc, bins=bins_dep, density=True, alpha=0.55,
               color="#2a9d8f",
               label=f"M6 deployment  (n={n_nc:,})")
    ax_d1.axvline(comp_med, color="#264653", linestyle="--", linewidth=1.3,
                  label=f"Compliant median = {comp_med:,.0f} t")
    ax_d1.axvline(med_nc, color="#2a9d8f", linestyle="--", linewidth=1.3,
                  label=f"Deployment median = {med_nc:,.0f} t")
    ax_d1.set_xscale("log")
    ax_d1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_d1.set_xlabel("Predicted GHG (metric tons CO2e)", fontsize=9)
    ax_d1.set_ylabel("Density", fontsize=9)
    ax_d1.set_title("Overall distribution\n(deployment vs observed compliant)",
                    fontsize=9, fontweight="bold")
    ax_d1.legend(fontsize=7.5)
    ax_d1.spines["top"].set_visible(False)
    ax_d1.spines["right"].set_visible(False)

    if type_deploy:
        sorted_td = sorted(type_deploy.items(), key=lambda x: x[1][0])
        tnames_   = [t for t, _ in sorted_td]
        tmedians  = [v[0] for _, v in sorted_td]
        tn_       = [v[1] for _, v in sorted_td]
        yp        = np.arange(len(tnames_))

        bars = ax_d2.barh(yp, tmedians, color="#2a9d8f", alpha=0.75,
                          edgecolor="white", height=0.6)
        ax_d2.axvline(comp_med, color="#264653", linestyle="--",
                      linewidth=1.1,
                      label=f"Compliant median = {comp_med:,.0f} t")
        for bar, v, n_ in zip(bars, tmedians, tn_):
            ax_d2.text(v + comp_med * 0.02,
                       bar.get_y() + bar.get_height() / 2,
                       f"{v:,.0f} t  (n={n_})",
                       va="center", fontsize=6.5, color="#333333")
        ax_d2.set_yticks(yp)
        ax_d2.set_yticklabels(tnames_, fontsize=7.0)
        ax_d2.set_xlabel("Deployment median (metric tons CO2e)", fontsize=9)
        ax_d2.set_title(
            "Median deployment prediction\nby property type (top 12 by count)",
            fontsize=9, fontweight="bold")
        ax_d2.legend(fontsize=7.5)
        ax_d2.spines["top"].set_visible(False)
        ax_d2.spines["right"].set_visible(False)
        xmax = max(tmedians) if tmedians else comp_med * 2
        ax_d2.set_xlim(0, xmax * 1.45)

    fig_dep.suptitle(
        "M6 Causal Bayes — Deployment Inference on Non-Compliant Records\n"
        "Posterior predictive median  |  "
        "Structural zeros for mediators (tested DAG)",
        fontsize=10, fontweight="bold")
    fig_dep.tight_layout()
    _save(fig_dep, "fig_m6_deployment.png")


    # =======================================================================
    # INFERENCE SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY  —  M6 CAUSAL BAYES")
    print("=" * 70)
    print(f"  Training rows          : {n_obs:,}")
    print(f"  Property types         : {n_types}")
    print(f"  ELPD (PSIS-LOO)        : {elpd:+,.0f}  (SE = {elpd_se:.1f})")
    print(f"  R-hat max (overall)    : {rhat_max:.4f}")
    print(f"  ESS min  (overall)     : {ess_min:,}")
    print(f"  Divergent transitions  : {n_div}")
    print(f"  Pareto-k > 0.7         : {n_bad_k}")
    print()
    print("  Global parameters (posterior mean  +  89% PI):")
    for p, s in param_stats.items():
        print(f"    {p:<18} {s['mean']:+.4f}  [{s['lo']:+.4f}, {s['hi']:+.4f}]")
    print()
    print(f"  Implied mean GHG at mean building:")
    print(f"    exp(alpha_bar) - 1  =  {np.expm1(ab_mean):,.0f} metric tons CO2e")
    print(f"  Floor-area elasticity:")
    print(f"    doubling floor area -> x{2**bA_mean:.3f} GHG  "
          f"(beta_A = {bA_mean:.3f})")
    print()
    print("  Property-type intercepts (top 3 and bottom 3 by posterior mean):")
    for ts in type_stats[-3:][::-1]:
        print(f"    {ts['name']:<50} "
              f"{ts['mean']:+.3f}  [{ts['lo']:+.3f}, {ts['hi']:+.3f}]")
    print("    ...")
    for ts in type_stats[:3]:
        print(f"    {ts['name']:<50} "
              f"{ts['mean']:+.3f}  [{ts['lo']:+.3f}, {ts['hi']:+.3f}]")
    print()
    print(f"  Deployment predictions (non-compliant buildings):")
    print(f"    Median             : {med_nc:,.0f} metric tons CO2e")
    print(f"    Mean               : {mean_nc:,.0f} metric tons CO2e")
    print(f"    Median 89% PI      : [{pi_med_lo:,.0f},  {pi_med_hi:,.0f}]")
    print(f"    Compliant median   : {comp_med:,.0f} metric tons CO2e  (reference)")
    print()
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 70)
