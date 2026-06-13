"""
counterfactual.py
=================
Counterfactual analysis for M6 Causal Bayes.

All outputs are computed analytically from the saved posterior samples --
no MCMC is run. The trace is loaded from results/inference/m6_trace.nc.

All feature standardisation uses the training-set means and standard
deviations, identical to the M6 specification in the experiment.

Outputs
-------
Output 4 -- Decarbonisation pace and policy gap
  Translates beta_T from the standardised scale to a per-calendar-year
  rate, projects the GHG trajectory forward to 2040, and compares it
  against a stylised net-zero-by-2040 benchmark (a 99% reduction; this is
  stricter than Chicago's official target of a 62% reduction by 2040).

Output 5a -- Counterfactual 1: do(data_year = 2030)
  Holds all building characteristics fixed. Sets data_year = 2030 for
  every building. Computes each building's posterior predictive GHG in
  2030 and classifies it as on-track, uncertain, or high-risk relative
  to a compliance threshold (25th percentile of observed GHG).

Output 5b -- Counterfactual 2: do(year_built = 2010)
  Replaces each building's year_built with 2010. Computes the posterior
  predictive GHG difference (factual - counterfactual) for each building.
  Illustrates a methodological finding: beta_Y captures cross-sectional
  use-intensity differences across vintage cohorts, not the causal effect
  of system-level retrofits.

Figures saved to results/counterfactual/
  fig_trend.png         GHG trajectory vs stylised 2040 net-zero benchmark
  fig_cf1_compliance.png  Distribution of P(compliant in 2030)
  fig_cf2_reduction.png   Distribution of GHG delta under do(year_built=2010)

Run (from drafts/complete/): uv run python scripts/counterfactual/counterfactual.py
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
import polars as pl

# _ROOT resolves to the bundle root (drafts/complete).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "experiments"))

from pipeline import (
    fetch_data, preprocess,
    SEED, APP_TOKEN, RAW_CAT, TARGET,
)

TRACE_PATH = os.path.join(_ROOT, "results", "inference", "m6_trace.nc")
FIG_DIR    = os.path.join(_ROOT, "results", "counterfactual")

# Policy parameters
BASE_YEAR   = 2023
TARGET_YEAR = 2040
CF1_YEAR    = 2030   # do(data_year = 2030)
CF2_YEAR    = 2010   # do(year_built = 2010)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {p}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    os.makedirs(FIG_DIR, exist_ok=True)

    # =======================================================================
    # STEP 1 -- FETCH + PREPROCESS
    # =======================================================================
    print("=" * 70)
    print("STEP 1 -- FETCHING DATA")
    print("=" * 70)
    df = fetch_data(APP_TOKEN)

    print("\n" + "=" * 70)
    print("STEP 2 -- PREPROCESSING")
    print("=" * 70)
    ctx   = preprocess(df)
    obs   = ctx["obs_causal"]
    n_obs = ctx["n_causal"]
    y_obs = ctx["y_obs_causal"]
    y_log = ctx["y_log_causal"]
    print(f"  Causal training rows : {n_obs:,}")

    # =======================================================================
    # STEP 2 -- LOAD M6 TRACE
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 3 -- LOADING M6 TRACE")
    print(f"  Source : {TRACE_PATH}")
    print("=" * 70)

    if not os.path.exists(TRACE_PATH):
        raise FileNotFoundError(
            f"M6 trace not found at {TRACE_PATH}.\n"
            "Generate the trace first: "
            "uv run python scripts/experiments/models.py"
        )

    trace = az.from_netcdf(TRACE_PATH)
    post  = trace.posterior
    print(f"  Loaded : {post.sizes['chain']} chains x {post.sizes['draw']} draws")

    # =======================================================================
    # STEP 3 -- RECONSTRUCT FEATURE ARRAYS  (identical to M6)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 4 -- RECONSTRUCTING FEATURE ARRAYS")
    print("=" * 70)

    all_types   = sorted(obs[RAW_CAT].unique().to_list())
    n_types     = len(all_types)
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    type_idx    = np.array([type_to_idx[t] for t in obs[RAW_CAT].to_list()])

    log_area   = np.log(obs["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs  = np.log(obs["of_buildings"].to_numpy().clip(1))
    year_data  = obs["data_year"].to_numpy().astype(float)
    year_built = obs["year_built"].to_numpy().astype(float)

    # Training-set statistics -- must match M6 exactly
    la_mean = log_area.mean()
    lb_mean = log_bldgs.mean()
    yd_mean = year_data.mean();   yd_std = year_data.std()
    yb_mean = year_built.mean();  yb_std = year_built.std()

    log_area_c  = log_area  - la_mean
    log_bldgs_c = log_bldgs - lb_mean
    yr_data_s   = (year_data  - yd_mean) / yd_std
    yr_built_s  = (year_built - yb_mean) / yb_std

    print(f"  year_data  : mean={yd_mean:.2f}  std={yd_std:.3f}")
    print(f"  year_built : mean={yb_mean:.2f}  std={yb_std:.3f}")

    # Extract posterior arrays -- shape (n_draws, n_types) or (n_draws,)
    n_draws  = post.sizes["chain"] * post.sizes["draw"]
    beta_T   = np.asarray(post["beta_T"]).flatten()          # (n_draws,)
    beta_A   = np.asarray(post["beta_A"]).flatten()
    beta_B   = np.asarray(post["beta_B"]).flatten()
    beta_Y   = np.asarray(post["beta_Y"]).flatten()
    sigma    = np.asarray(post["sigma"]).flatten()
    alpha    = np.asarray(post["alpha"]).reshape(n_draws, n_types)  # (n_draws, n_types)

    print(f"  Posterior draws : {n_draws:,}")

    # =======================================================================
    # OUTPUT 4 -- DECARBONISATION PACE AND POLICY GAP
    # =======================================================================
    print("\n" + "=" * 70)
    print("OUTPUT 4 -- DECARBONISATION PACE AND POLICY GAP")
    print("=" * 70)

    # Convert beta_T from standardised scale to per-calendar-year rate
    annual_rate_draws = beta_T / yd_std          # per year, log scale
    annual_rate_mean  = float(annual_rate_draws.mean())
    annual_rate_lo    = float(np.percentile(annual_rate_draws, 5.5))
    annual_rate_hi    = float(np.percentile(annual_rate_draws, 94.5))

    pct_per_year_mean = (np.exp(annual_rate_mean) - 1) * 100
    pct_per_year_lo   = (np.exp(annual_rate_lo)   - 1) * 100
    pct_per_year_hi   = (np.exp(annual_rate_hi)   - 1) * 100

    # Years to reach 80% reduction at current pace (from BASE_YEAR)
    # 0.2 = exp(annual_rate_mean * years)  ->  years = log(0.2) / annual_rate_mean
    years_to_80pct  = np.log(0.20) / annual_rate_mean
    reach_80_year   = BASE_YEAR + years_to_80pct

    # Required annual rate to reach net-zero (99% reduction) by TARGET_YEAR
    years_to_target = TARGET_YEAR - BASE_YEAR
    required_rate   = np.log(0.01) / years_to_target   # log scale per year
    required_pct    = (np.exp(required_rate) - 1) * 100
    rate_gap        = required_rate - annual_rate_mean  # additional needed

    print(f"  beta_T (standardised)     : {float(beta_T.mean()):.4f}  "
          f"[{float(np.percentile(beta_T, 5.5)):.4f}, "
          f"{float(np.percentile(beta_T, 94.5)):.4f}]")
    print(f"  year_data std             : {yd_std:.3f} years")
    print(f"  Annual rate (log scale)   : {annual_rate_mean:.4f}/yr  "
          f"[{annual_rate_lo:.4f}, {annual_rate_hi:.4f}]")
    print(f"  Annual rate (% GHG change): {pct_per_year_mean:.2f}%/yr  "
          f"[{pct_per_year_lo:.2f}%, {pct_per_year_hi:.2f}%]")
    print(f"  At current pace, 80% reduction by: ~{reach_80_year:.0f}")
    print(f"  Chicago net-zero target   : {TARGET_YEAR}")
    print(f"  Required annual rate      : {required_pct:.2f}%/yr  "
          f"(gap = {(required_rate - annual_rate_mean):.4f} log/yr)")
    if reach_80_year > TARGET_YEAR:
        print(f"  *** Current pace is INSUFFICIENT for net-zero by {TARGET_YEAR} ***")
    else:
        print(f"  Current pace is sufficient for net-zero by {TARGET_YEAR}")

    # -- fig_trend.png ---------------------------------------------------------
    # Project median building GHG forward from BASE_YEAR using beta_T posterior
    # "Median building" = mean feature values, mean type intercept = alpha_bar
    alpha_bar_mean = float(np.asarray(post["alpha_bar"]).mean())
    proj_years  = np.arange(BASE_YEAR, TARGET_YEAR + 1)
    t_proj_s    = (proj_years - yd_mean) / yd_std    # standardised

    # Posterior mean trajectory
    ghg_proj_mean = np.expm1(
        alpha_bar_mean + annual_rate_mean * (proj_years - BASE_YEAR)
        + float(beta_T.mean()) / yd_std * (BASE_YEAR - yd_mean)
    )
    # Simpler: project from BASE_YEAR GHG
    ghg_base = np.expm1(
        alpha_bar_mean
        + float(beta_T.mean()) * ((BASE_YEAR - yd_mean) / yd_std)
    )
    ghg_proj  = ghg_base * np.exp(annual_rate_mean * (proj_years - BASE_YEAR))
    ghg_lo    = ghg_base * np.exp(annual_rate_lo   * (proj_years - BASE_YEAR))
    ghg_hi    = ghg_base * np.exp(annual_rate_hi   * (proj_years - BASE_YEAR))
    netzero_line = ghg_base * np.linspace(1, 0, len(proj_years))

    fig_tr, ax_tr = plt.subplots(figsize=(11, 5))
    ax_tr.fill_between(proj_years, ghg_lo, ghg_hi,
                       color="#2a9d8f", alpha=0.2, label="89% posterior CI")
    ax_tr.plot(proj_years, ghg_proj, color="#2a9d8f", linewidth=2.2,
               label=f"Current trajectory  ({pct_per_year_mean:.1f}%/yr)")
    ax_tr.plot(proj_years, netzero_line, color="#e63946", linewidth=1.5,
               linestyle="--", label=f"Required path to net-zero by {TARGET_YEAR}")
    ax_tr.axvline(TARGET_YEAR, color="#e63946", linewidth=0.8,
                  linestyle=":", alpha=0.7)
    ax_tr.axvline(CF1_YEAR, color="#264653", linewidth=0.8,
                  linestyle=":", alpha=0.6)
    ax_tr.text(CF1_YEAR + 0.2, ghg_proj.max() * 0.95,
               str(CF1_YEAR), fontsize=8, color="#264653")
    ax_tr.text(TARGET_YEAR + 0.2, ghg_proj.max() * 0.95,
               f"{TARGET_YEAR}\n(target)", fontsize=8, color="#e63946")
    ax_tr.set_xlabel("Year", fontsize=9)
    ax_tr.set_ylabel("Predicted GHG (metric tons CO2e)", fontsize=9)
    ax_tr.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_tr.legend(fontsize=8.5, loc="upper right")
    ax_tr.set_title(
        "M6 Causal Bayes -- Projected GHG Trajectory vs Chicago Net-Zero Target\n"
        "Median building at population mean  |  "
        f"Current pace: {pct_per_year_mean:.1f}%/yr  |  "
        f"Required: {required_pct:.1f}%/yr",
        fontsize=10, fontweight="bold")
    fig_tr.tight_layout()
    _save(fig_tr, "fig_trend.png")

    # =======================================================================
    # OUTPUT 5a -- CF1: do(data_year = 2030)
    # =======================================================================
    print("\n" + "=" * 70)
    print(f"OUTPUT 5a -- COUNTERFACTUAL 1: do(data_year = {CF1_YEAR})")
    print("=" * 70)

    # Standardise the counterfactual year on training scale
    t_cf1_s = (CF1_YEAR - yd_mean) / yd_std

    # Posterior predictive mu for each draw x building
    # alpha[:, type_idx] -> (n_draws, n_obs)
    mu_cf1 = (
        alpha[:, type_idx]                         # (n_draws, n_obs)
        + beta_A[:, None] * log_area_c[None, :]
        + beta_B[:, None] * log_bldgs_c[None, :]
        + beta_T[:, None] * t_cf1_s
        + beta_Y[:, None] * yr_built_s[None, :]
    )

    # Add residual noise for full posterior predictive
    rng      = np.random.default_rng(SEED)
    eps_cf1  = rng.normal(0, sigma[:, None], size=mu_cf1.shape)
    ypred_cf1 = np.clip(np.expm1(mu_cf1 + eps_cf1), 0, None)

    # Compliance threshold: 25th percentile of observed GHG (same as workbook)
    tau_2030 = float(np.percentile(y_obs[y_obs > 0], 25))

    # P(compliant) per building = fraction of draws where prediction <= threshold
    prob_compliant = (ypred_cf1 <= tau_2030).mean(axis=0)   # (n_obs,)

    n_on_track  = int((prob_compliant > 0.8).sum())
    n_uncertain = int(((prob_compliant >= 0.2) & (prob_compliant <= 0.8)).sum())
    n_high_risk = int((prob_compliant < 0.2).sum())
    med_prob    = float(np.median(prob_compliant))

    print(f"  Compliance threshold (25th pct)  : {tau_2030:,.0f} metric tons CO2e")
    print(f"  Counterfactual year              : {CF1_YEAR}")
    print(f"  -----------------------------------------------------")
    print(f"  On-track    P(compliant) > 0.8   : "
          f"{n_on_track:,}  ({n_on_track/n_obs*100:.1f}%)")
    print(f"  Uncertain   P(compliant) 0.2-0.8 : "
          f"{n_uncertain:,}  ({n_uncertain/n_obs*100:.1f}%)")
    print(f"  High-risk   P(compliant) < 0.2   : "
          f"{n_high_risk:,}  ({n_high_risk/n_obs*100:.1f}%)")
    print(f"  Median P(compliant) across all   : {med_prob:.3f}")

    # -- fig_cf1_compliance.png ------------------------------------------------
    fig_cf1, ax_cf1 = plt.subplots(figsize=(10, 5))

    bins_p = np.linspace(0, 1, 41)
    colours_cf1 = np.where(
        prob_compliant > 0.8,  "#2a9d8f",
        np.where(prob_compliant < 0.2, "#e63946", "#e9c46a")
    )

    n_bins = 40
    counts, edges = np.histogram(prob_compliant, bins=n_bins, range=(0, 1))
    bin_centers   = (edges[:-1] + edges[1:]) / 2

    bar_colors = np.where(
        bin_centers > 0.8,  "#2a9d8f",
        np.where(bin_centers < 0.2, "#e63946", "#e9c46a")
    )
    ax_cf1.bar(bin_centers, counts, width=(edges[1] - edges[0]) * 0.9,
               color=bar_colors, edgecolor="white", linewidth=0.4)

    ax_cf1.axvline(0.8, color="#2a9d8f", linestyle="--", linewidth=1.2,
                   label=f"P=0.8  on-track  ({n_on_track:,}, {n_on_track/n_obs*100:.1f}%)")
    ax_cf1.axvline(0.2, color="#e63946", linestyle="--", linewidth=1.2,
                   label=f"P=0.2  high-risk ({n_high_risk:,}, {n_high_risk/n_obs*100:.1f}%)")
    ax_cf1.axvline(med_prob, color="#264653", linestyle="-", linewidth=1.5,
                   label=f"Median P = {med_prob:.2f}")

    ax_cf1.set_xlabel(f"P(GHG <= {tau_2030:,.0f} t in {CF1_YEAR})", fontsize=9)
    ax_cf1.set_ylabel("Number of buildings", fontsize=9)
    ax_cf1.legend(fontsize=8.5)
    ax_cf1.set_title(
        f"M6 Causal Bayes -- Compliance Risk under do(data_year = {CF1_YEAR})\n"
        f"Threshold = {tau_2030:,.0f} t CO2e (25th pct of observed GHG)  |  "
        f"n = {n_obs:,} compliant buildings",
        fontsize=10, fontweight="bold")
    fig_cf1.tight_layout()
    _save(fig_cf1, "fig_cf1_compliance.png")

    # =======================================================================
    # OUTPUT 5b -- CF2: do(year_built = 2010)
    # =======================================================================
    print("\n" + "=" * 70)
    print(f"OUTPUT 5b -- COUNTERFACTUAL 2: do(year_built = {CF2_YEAR})")
    print("=" * 70)

    yb_cf2_s = (CF2_YEAR - yb_mean) / yb_std   # counterfactual year_built, standardised

    # Factual mu (no noise -- posterior mean comparison)
    mu_factual = (
        alpha[:, type_idx]
        + beta_A[:, None] * log_area_c[None, :]
        + beta_B[:, None] * log_bldgs_c[None, :]
        + beta_T[:, None] * yr_data_s[None, :]
        + beta_Y[:, None] * yr_built_s[None, :]
    )

    # Counterfactual mu -- replace year_built with CF2_YEAR for all buildings
    mu_cf2 = (
        alpha[:, type_idx]
        + beta_A[:, None] * log_area_c[None, :]
        + beta_B[:, None] * log_bldgs_c[None, :]
        + beta_T[:, None] * yr_data_s[None, :]
        + beta_Y[:, None] * yb_cf2_s          # scalar broadcast
    )

    ypred_factual = np.clip(np.expm1(mu_factual), 0, None)
    ypred_cf2     = np.clip(np.expm1(mu_cf2),     0, None)

    # GHG reduction = factual - counterfactual
    # Positive = factual > counterfactual -> intervention reduces GHG
    # Negative = factual < counterfactual -> newer vintage predicts MORE GHG
    reduction     = ypred_factual - ypred_cf2        # (n_draws, n_obs)
    red_mean      = reduction.mean(axis=0)           # (n_obs,)
    red_lo        = np.percentile(reduction, 5.5,  axis=0)
    red_hi        = np.percentile(reduction, 94.5, axis=0)

    med_red   = float(np.median(red_mean))
    mean_red  = float(np.mean(red_mean))
    med_lo    = float(np.median(red_lo))
    med_hi    = float(np.median(red_hi))

    # Buildings with credible effect (89% CI excludes zero)
    credible_pos = int(((red_lo > 0)).sum())    # factual > cf2 -> older emits more
    credible_neg = int(((red_hi < 0)).sum())    # factual < cf2 -> older emits less
    no_effect    = n_obs - credible_pos - credible_neg

    print(f"  Counterfactual year_built        : {CF2_YEAR}")
    print(f"  Actual mean year_built           : {yb_mean:.1f}")
    print(f"  -----------------------------------------------------")
    print(f"  GHG 'reduction' median           : {med_red:>+10,.0f} metric tons CO2e")
    print(f"  GHG 'reduction' mean             : {mean_red:>+10,.0f} metric tons CO2e")
    print(f"  Median 89% CI                    : [{med_lo:>+,.0f},  {med_hi:>+,.0f}]")
    print(f"  -----------------------------------------------------")
    print(f"  Credible reduction (CI excludes zero, pos): "
          f"{credible_pos:,}  ({credible_pos/n_obs*100:.1f}%)")
    print(f"  Credible increase  (CI excludes zero, neg): "
          f"{credible_neg:,}  ({credible_neg/n_obs*100:.1f}%)")
    print(f"  No credible effect (CI includes zero)     : "
          f"{no_effect:,}  ({no_effect/n_obs*100:.1f}%)")
    if med_red < 0:
        print(f"\n  *** Counterintuitive result: 'newer vintage' predicts HIGHER GHG. ***")
        print(f"  This is a methodological finding -- beta_Y captures cross-sectional")
        print(f"  use-intensity differences, not retrofit impact. See analysis.txt.")

    # -- fig_cf2_reduction.png -------------------------------------------------
    fig_cf2, (ax_c2l, ax_c2r) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: distribution of per-building mean GHG reduction
    vmax_r = max(abs(red_mean.min()), abs(red_mean.max()))
    bins_r = np.linspace(-vmax_r * 1.05, vmax_r * 1.05, 60)
    neg_mask = red_mean < 0
    pos_mask = red_mean >= 0
    if pos_mask.any():
        ax_c2l.hist(red_mean[pos_mask], bins=bins_r,
                    color="#2a9d8f", alpha=0.75, edgecolor="none",
                    label=f"Factual > CF  ({pos_mask.sum():,})")
    if neg_mask.any():
        ax_c2l.hist(red_mean[neg_mask], bins=bins_r,
                    color="#e63946", alpha=0.75, edgecolor="none",
                    label=f"Factual < CF  ({neg_mask.sum():,})")
    ax_c2l.axvline(0,       color="#333333", linewidth=1.0, linestyle="-")
    ax_c2l.axvline(med_red, color="#264653", linewidth=1.5, linestyle="--",
                   label=f"Median = {med_red:+,.0f} t")
    ax_c2l.set_xlabel("Factual GHG - CF GHG (metric tons CO2e)", fontsize=9)
    ax_c2l.set_ylabel("Number of buildings", fontsize=9)
    ax_c2l.set_title("Distribution of GHG delta\n(positive = CF reduces emissions)",
                     fontsize=9, fontweight="bold")
    ax_c2l.legend(fontsize=8)
    ax_c2l.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))
    ax_c2l.spines["top"].set_visible(False)
    ax_c2l.spines["right"].set_visible(False)

    # Right: scatter of factual vs counterfactual median GHG by building
    med_fct_bld = ypred_factual.mean(axis=0)
    med_cf2_bld = ypred_cf2.mean(axis=0)
    vmax_s      = max(med_fct_bld.max(), med_cf2_bld.max())
    ax_c2r.scatter(med_fct_bld, med_cf2_bld,
                   s=2, alpha=0.25, color="#264653", linewidths=0)
    ax_c2r.plot([0, vmax_s], [0, vmax_s], color="#aaaaaa",
                linewidth=1.0, linestyle="--", label="No change")
    ax_c2r.set_xscale("log"); ax_c2r.set_yscale("log")
    ax_c2r.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_c2r.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_c2r.set_xlabel("Factual GHG (metric tons CO2e)", fontsize=9)
    ax_c2r.set_ylabel(f"CF GHG  do(year_built={CF2_YEAR})  (metric tons CO2e)",
                      fontsize=9)
    ax_c2r.set_title("Factual vs counterfactual GHG\n(log-log scale, each dot = one building)",
                     fontsize=9, fontweight="bold")
    ax_c2r.legend(fontsize=8)
    ax_c2r.spines["top"].set_visible(False)
    ax_c2r.spines["right"].set_visible(False)

    fig_cf2.suptitle(
        f"M6 Causal Bayes -- Counterfactual 2: do(year_built = {CF2_YEAR})\n"
        f"Posterior mean GHG delta: median = {med_red:+,.0f} t  |  "
        f"89% CI: [{med_lo:+,.0f}, {med_hi:+,.0f}] t",
        fontsize=10, fontweight="bold")
    fig_cf2.tight_layout()
    _save(fig_cf2, "fig_cf2_reduction.png")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("COUNTERFACTUAL SUMMARY  --  M6 CAUSAL BAYES")
    print("=" * 70)
    print(f"\n  Output 4 -- Decarbonisation pace")
    print(f"    beta_T (std scale)  : {float(beta_T.mean()):.4f}  "
          f"[{float(np.percentile(beta_T, 5.5)):.4f}, "
          f"{float(np.percentile(beta_T, 94.5)):.4f}]")
    print(f"    Annual rate         : {pct_per_year_mean:.2f}%/yr  "
          f"[{pct_per_year_lo:.2f}%, {pct_per_year_hi:.2f}%]")
    print(f"    80% reduction by    : ~{reach_80_year:.0f}")
    print(f"    Required for 2040   : {required_pct:.2f}%/yr")
    print(f"    Policy gap          : {(required_rate - annual_rate_mean):.4f} log/yr")
    print(f"\n  Output 5a -- CF1: do(data_year = {CF1_YEAR})")
    print(f"    Threshold (25th pct): {tau_2030:,.0f} metric tons CO2e")
    print(f"    On-track  (P > 0.8) : {n_on_track:,}  ({n_on_track/n_obs*100:.1f}%)")
    print(f"    Uncertain (0.2-0.8) : {n_uncertain:,}  ({n_uncertain/n_obs*100:.1f}%)")
    print(f"    High-risk (P < 0.2) : {n_high_risk:,}  ({n_high_risk/n_obs*100:.1f}%)")
    print(f"    Median P(compliant) : {med_prob:.3f}")
    print(f"\n  Output 5b -- CF2: do(year_built = {CF2_YEAR})")
    print(f"    Median GHG delta    : {med_red:+,.0f} metric tons CO2e")
    print(f"    Mean GHG delta      : {mean_red:+,.0f} metric tons CO2e")
    print(f"    Median 89% CI       : [{med_lo:+,.0f},  {med_hi:+,.0f}]")
    print(f"    Credible reduction  : {credible_pos:,}  ({credible_pos/n_obs*100:.1f}%)")
    print(f"    Credible increase   : {credible_neg:,}  ({credible_neg/n_obs*100:.1f}%)")
    print(f"\n  Figures saved to: {FIG_DIR}")
    print("=" * 70)
