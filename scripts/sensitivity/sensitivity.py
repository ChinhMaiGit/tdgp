"""
sensitivity.py
==============
Sensitivity analysis for M6 Causal Bayes.

Tests robustness of the four key parameters (beta_A, beta_T, sigma_alpha,
sigma) across two independent dimensions of analyst choice:

  Sensitivity 1 — Prior specification  (IPW fixed at 99th-pct cap)
    Tight : slope sigma=0.25, scale sigma=0.50
    Base  : slope sigma=0.50, scale sigma=1.00  <- M6 spec (loaded from trace)
    Wide  : slope sigma=1.00, scale sigma=2.00

  Sensitivity 2 — IPW trim cap  (priors fixed at base spec)
    Aggressive : 95th-percentile cap
    Base       : 99th-percentile cap  <- M6 spec (loaded from trace)
    Lenient    : 99.9th-percentile cap

All new fits use the exact M6 sampling spec:
  draws=3000, tune=3000, chains=4, cores=4,
  target_accept=0.95, init="jitter+adapt_diag",
  random_seed=[20, 21, 22, 23]

The base specification is loaded directly from inference/m6_trace.nc
(the trace saved by the experiment) — no re-sampling needed.

Run (from drafts/complete/): uv run python scripts/sensitivity/sensitivity.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import polars as pl

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "experiments"))

from pipeline import (
    fetch_data, preprocess,
    SEED, APP_TOKEN, RAW_CAT, TARGET, UPSTREAM_COLS,
    _flatten_az_result,
)

CHAIN_SEEDS = [SEED + i for i in range(4)]
FIG_DIR     = os.path.join(_ROOT, "results", "sensitivity")
TRACE_PATH  = os.path.join(_ROOT, "results", "inference", "m6_trace.nc")

SENS_PARAMS = ["beta_A", "beta_T", "sigma_alpha", "sigma"]

PARAM_LABELS = {
    "beta_A":      "beta_A  (log floor area)",
    "beta_T":      "beta_T  (data year trend)",
    "sigma_alpha": "sigma_alpha  (between-type SD)",
    "sigma":       "sigma  (residual SD)",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ipw(raw_w: np.ndarray, pct: float):
    """Trim raw IPW weights at the given percentile and normalise."""
    cap = float(np.percentile(raw_w, pct))
    w   = np.clip(raw_w, None, cap)
    w   = w / w.mean()
    return w, cap


def posterior_stats(trace, param: str) -> dict:
    v = np.asarray(trace.posterior[param]).flatten()
    lo, hi = float(np.percentile(v, 5.5)), float(np.percentile(v, 94.5))
    return {
        "mean": float(v.mean()),
        "lo":   lo,
        "hi":   hi,
        "hw":   (hi - lo) / 2,
    }


def diagnostics(trace, label: str) -> dict:
    sp   = ["alpha_bar", "sigma_alpha", "beta_A", "beta_B",
            "beta_T", "beta_Y", "sigma"]
    rh   = float(_flatten_az_result(az.rhat(trace, var_names=sp)).max())
    ess  = int(_flatten_az_result(az.ess(trace,  var_names=sp)).min())
    ndiv = int(trace.sample_stats.diverging.values.sum())
    return {"label": label, "rhat": rh, "ess": ess, "ndiv": ndiv}


def fit_sensitivity(
    prior_slope_sigma: float,
    prior_scale_sigma: float,
    ipw_w: np.ndarray,
    all_types, type_idx,
    log_area_c, log_bldgs_c, yr_data_s, yr_built_s,
    y_log: np.ndarray,
    n_obs: int,
    label: str,
):
    """
    Refit M6 with alternative prior widths or IPW weights.
    Model structure is identical to M6; only prior_slope_sigma,
    prior_scale_sigma, and ipw_w are varied.
    alpha_bar prior is always Normal(7.0, 1.0) — not varied.
    """
    print(f"\n  Fitting: {label}")
    print(f"    prior_slope_sigma = {prior_slope_sigma}  "
          f"prior_scale_sigma = {prior_scale_sigma}  "
          f"ipw max = {ipw_w.max():.3f}")

    coords = {"obs": np.arange(n_obs), "type": all_types}

    with pm.Model(coords=coords):
        pm.Data("type_idx",   type_idx,    dims="obs")
        pm.Data("log_area",   log_area_c,  dims="obs")
        pm.Data("log_bldgs",  log_bldgs_c, dims="obs")
        pm.Data("year_data",  yr_data_s,   dims="obs")
        pm.Data("year_built", yr_built_s,  dims="obs")

        ab  = pm.Normal("alpha_bar",    mu=7.0, sigma=1.0)
        sa  = pm.HalfNormal("sigma_alpha", sigma=prior_scale_sigma)
        dj  = pm.Normal("alpha_offset", mu=0,   sigma=1, dims="type")
        pm.Deterministic("alpha", ab + dj * sa, dims="type")

        bA  = pm.Normal("beta_A", mu=0, sigma=prior_slope_sigma)
        bB  = pm.Normal("beta_B", mu=0, sigma=prior_slope_sigma)
        bT  = pm.Normal("beta_T", mu=0, sigma=prior_slope_sigma)
        bY  = pm.Normal("beta_Y", mu=0, sigma=prior_slope_sigma)
        sig = pm.HalfNormal("sigma", sigma=prior_scale_sigma)

        mu = (ab + dj * sa)[type_idx] + bA*log_area_c + bB*log_bldgs_c \
             + bT*yr_data_s + bY*yr_built_s

        ipw_data = pm.Data("ipw", ipw_w, dims="obs")
        ll       = pm.logp(pm.Normal.dist(mu=mu, sigma=sig), y_log)
        pm.Potential("ipw_weight", (ipw_data - 1.0) * ll)
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
        )

    return trace


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
    ctx    = preprocess(df)
    obs    = ctx["obs_causal"]
    n_obs  = ctx["n_causal"]
    y_log  = ctx["y_log_causal"]
    y_obs  = ctx["y_obs_causal"]
    print(f"  Causal training rows : {n_obs:,}")

    # =======================================================================
    # STEP 2 -- BUILD IPW WEIGHT ARRAYS AT THREE CAPS
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 3 -- IPW WEIGHT ARRAYS  (95th / 99th / 99.9th pct caps)")
    print("=" * 70)

    # Reconstruct raw weights from propensity scores in df_full
    p_causal = (
        ctx["df_full"]
        .filter((pl.col("compliant") == 1) & pl.col(TARGET).is_not_null())
        ["p_compliant"]
        .to_numpy()
    )
    raw_w = 1.0 / p_causal

    ipw_95,  cap_95  = make_ipw(raw_w, 95.0)
    ipw_99,  cap_99  = make_ipw(raw_w, 99.0)    # base — matches M6
    ipw_999, cap_999 = make_ipw(raw_w, 99.9)

    print(f"  {'Cap':<12} {'Raw cap val':>12} {'Max (norm)':>12} {'Mean (norm)':>12}")
    print("  " + "─" * 52)
    for lbl, cap_v, w in [
        ("95th pct",  cap_95,  ipw_95),
        ("99th pct",  cap_99,  ipw_99),
        ("99.9th pct",cap_999, ipw_999),
    ]:
        print(f"  {lbl:<12} {cap_v:>12.3f} {w.max():>12.3f} {w.mean():>12.3f}")

    # =======================================================================
    # STEP 3 -- FEATURE STANDARDISATION  (identical to M6)
    # =======================================================================
    all_types   = sorted(obs[RAW_CAT].unique().to_list())
    n_types     = len(all_types)
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    type_idx    = np.array([type_to_idx[t] for t in obs[RAW_CAT].to_list()])

    log_area   = np.log(obs["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs  = np.log(obs["of_buildings"].to_numpy().clip(1))
    year_data  = obs["data_year"].to_numpy().astype(float)
    year_built = obs["year_built"].to_numpy().astype(float)

    log_area_c  = log_area  - log_area.mean()
    log_bldgs_c = log_bldgs - log_bldgs.mean()
    yr_data_s   = (year_data  - year_data.mean())  / year_data.std()
    yr_built_s  = (year_built - year_built.mean()) / year_built.std()

    # =======================================================================
    # STEP 4 -- LOAD BASE TRACE  (M6 from experiment)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 4 -- LOADING BASE TRACE  (M6 from experiment)")
    print(f"  Source : {TRACE_PATH}")
    print("=" * 70)

    if not os.path.exists(TRACE_PATH):
        raise FileNotFoundError(
            f"M6 trace not found at {TRACE_PATH}.\n"
            "Generate the trace first: uv run python scripts/experiments/models.py"
        )

    trace_base = az.from_netcdf(TRACE_PATH)
    print(f"  Loaded : {trace_base.posterior.sizes['chain']} chains x "
          f"{trace_base.posterior.sizes['draw']} draws")

    # =======================================================================
    # STEP 5 -- FIT 4 SENSITIVITY MODELS
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 5 -- FITTING SENSITIVITY MODELS  (4 x 3000/3000 draws)")
    print("  Tight prior : slope sigma=0.25, scale sigma=0.50, IPW 99th")
    print("  Wide prior  : slope sigma=1.00, scale sigma=2.00, IPW 99th")
    print("  IPW 95th    : base prior (sigma=0.50), 95th-pct cap")
    print("  IPW 99.9th  : base prior (sigma=0.50), 99.9th-pct cap")
    print("=" * 70)

    _shared = dict(
        all_types   = all_types,
        type_idx    = type_idx,
        log_area_c  = log_area_c,
        log_bldgs_c = log_bldgs_c,
        yr_data_s   = yr_data_s,
        yr_built_s  = yr_built_s,
        y_log       = y_log,
        n_obs       = n_obs,
    )

    trace_tight = fit_sensitivity(
        prior_slope_sigma = 0.25,
        prior_scale_sigma = 0.50,
        ipw_w = ipw_99,
        label = "Tight prior  (slope=0.25, scale=0.50, IPW 99th)",
        **_shared,
    )

    trace_wide = fit_sensitivity(
        prior_slope_sigma = 1.00,
        prior_scale_sigma = 2.00,
        ipw_w = ipw_99,
        label = "Wide prior   (slope=1.00, scale=2.00, IPW 99th)",
        **_shared,
    )

    trace_ipw95 = fit_sensitivity(
        prior_slope_sigma = 0.50,
        prior_scale_sigma = 1.00,
        ipw_w = ipw_95,
        label = "IPW 95th pct (base prior, aggressive cap)",
        **_shared,
    )

    trace_ipw999 = fit_sensitivity(
        prior_slope_sigma = 0.50,
        prior_scale_sigma = 1.00,
        ipw_w = ipw_999,
        label = "IPW 99.9th pct (base prior, lenient cap)",
        **_shared,
    )

    print("\n  All sensitivity models fitted.")

    # =======================================================================
    # STEP 6 -- DIAGNOSTICS TABLE
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 6 -- MCMC DIAGNOSTICS")
    print("=" * 70)

    all_traces = [
        (trace_base,   "Base (M6, slope=0.50, IPW 99th)"),
        (trace_tight,  "Tight prior  (slope=0.25, IPW 99th)"),
        (trace_wide,   "Wide prior   (slope=1.00, IPW 99th)"),
        (trace_ipw95,  "IPW 95th pct (base prior)"),
        (trace_ipw999, "IPW 99.9th pct (base prior)"),
    ]

    diags = [diagnostics(t, lbl) for t, lbl in all_traces]

    print(f"\n  {'Specification':<45} {'Div':>5} {'R-hat':>8} {'ESS':>6}")
    print("  " + "─" * 68)
    for d in diags:
        flag = "  ⚠" if d["rhat"] > 1.01 or d["ess"] < 400 or d["ndiv"] > 0 else ""
        print(f"  {d['label']:<45} {d['ndiv']:>5} "
              f"{d['rhat']:>8.4f} {d['ess']:>6,}{flag}")

    # =======================================================================
    # STEP 7 -- PARAMETER COMPARISON TABLES
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 7 -- PARAMETER COMPARISON")
    print("=" * 70)

    # Collect all stats
    all_stats = {}
    for trace, lbl in all_traces:
        all_stats[lbl] = {p: posterior_stats(trace, p) for p in SENS_PARAMS}

    base_lbl = "Base (M6, slope=0.50, IPW 99th)"

    # Prior sensitivity table
    prior_specs = [
        "Base (M6, slope=0.50, IPW 99th)",
        "Tight prior  (slope=0.25, IPW 99th)",
        "Wide prior   (slope=1.00, IPW 99th)",
    ]
    print(f"\n  Prior specification sensitivity (IPW fixed at 99th pct cap)")
    print(f"  {'Parameter':<16} " +
          "  ".join(f"{'mean [89% PI]':>30}" for _ in prior_specs))
    print(f"  {'':16} " +
          "  ".join(f"{lbl[:30]:>30}" for lbl in prior_specs))
    print("  " + "─" * (16 + 34 * len(prior_specs)))
    for p in SENS_PARAMS:
        row = f"  {p:<16}"
        for lbl in prior_specs:
            s = all_stats[lbl][p]
            row += f"  {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"
        print(row)

    # IPW sensitivity table
    ipw_specs = [
        "IPW 95th pct (base prior)",
        "Base (M6, slope=0.50, IPW 99th)",
        "IPW 99.9th pct (base prior)",
    ]
    print(f"\n  IPW trim cap sensitivity (priors fixed at base spec)")
    print(f"  {'Parameter':<16} " +
          "  ".join(f"{'mean [89% PI]':>30}" for _ in ipw_specs))
    print(f"  {'':16} " +
          "  ".join(f"{lbl[:30]:>30}" for lbl in ipw_specs))
    print("  " + "─" * (16 + 34 * len(ipw_specs)))
    for p in SENS_PARAMS:
        row = f"  {p:<16}"
        for lbl in ipw_specs:
            s = all_stats[lbl][p]
            row += f"  {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"
        print(row)

    # Robustness assessment
    print(f"\n  Robustness assessment  (shift = max |Δmean| vs base)")
    print(f"  {'Parameter':<16} {'Prior shift':>12} {'% of CI hw':>12} "
          f"{'IPW shift':>12} {'% of CI hw':>12}  Flag")
    print("  " + "─" * 76)
    for p in SENS_PARAMS:
        base_s  = all_stats[base_lbl][p]
        prior_shifts = [
            abs(all_stats[lbl][p]["mean"] - base_s["mean"])
            for lbl in prior_specs if lbl != base_lbl
        ]
        ipw_shifts = [
            abs(all_stats[lbl][p]["mean"] - base_s["mean"])
            for lbl in ipw_specs if lbl != base_lbl
        ]
        max_prior = max(prior_shifts)
        max_ipw   = max(ipw_shifts)
        pct_prior = max_prior / base_s["hw"] * 100
        pct_ipw   = max_ipw   / base_s["hw"] * 100
        flag = "⚠" if pct_prior > 20 or pct_ipw > 20 else "✓"
        print(f"  {p:<16} {max_prior:>12.4f} {pct_prior:>11.1f}% "
              f"{max_ipw:>12.4f} {pct_ipw:>11.1f}%  {flag}")
    print(f"\n  ✓ = shift < 20% of 89% CI half-width  (robust)")
    print(f"  ⚠ = shift >= 20% of 89% CI half-width  (review required)")

    # =======================================================================
    # STEP 8 -- FOREST PLOT
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 8 -- GENERATING FOREST PLOT")
    print("=" * 70)

    prior_colours = ["#264653", "#2a9d8f", "#e76f51"]   # base, tight, wide
    ipw_colours   = ["#e9c46a", "#264653", "#e76f51"]   # 95, base, 99.9

    prior_plot_specs = [
        ("Base (M6, slope=0.50, IPW 99th)",  "Base (slope σ=0.50)",    prior_colours[0]),
        ("Tight prior  (slope=0.25, IPW 99th)", "Tight (slope σ=0.25)", prior_colours[1]),
        ("Wide prior   (slope=1.00, IPW 99th)", "Wide (slope σ=1.00)",  prior_colours[2]),
    ]
    ipw_plot_specs = [
        ("IPW 95th pct (base prior)",          "95th pct cap",  ipw_colours[0]),
        ("Base (M6, slope=0.50, IPW 99th)",    "99th pct cap (base)", ipw_colours[1]),
        ("IPW 99.9th pct (base prior)",        "99.9th pct cap", ipw_colours[2]),
    ]

    n_p = len(SENS_PARAMS)
    fig, (ax_pr, ax_ipw) = plt.subplots(1, 2, figsize=(14, 5),
                                         sharey=True)

    offsets = np.linspace(-0.22, 0.22, 3)

    for ax, plot_specs, title in [
        (ax_pr,  prior_plot_specs, "Prior specification"),
        (ax_ipw, ipw_plot_specs,   "IPW trim cap"),
    ]:
        for oi, (lbl, short_lbl, col) in enumerate(plot_specs):
            for pi, p in enumerate(SENS_PARAMS):
                s = all_stats[lbl][p]
                y = pi + offsets[oi]
                ax.plot([s["lo"], s["hi"]], [y, y],
                        color=col, linewidth=2.2, alpha=0.8,
                        solid_capstyle="round")
                ax.scatter([s["mean"]], [y],
                           color=col, s=45, zorder=5,
                           label=short_lbl if pi == 0 else "")

        ax.axvline(0, color="#cccccc", linestyle="--", linewidth=0.8)
        ax.set_yticks(np.arange(n_p))
        ax.set_yticklabels([PARAM_LABELS[p] for p in SENS_PARAMS], fontsize=8.5)
        ax.set_xlabel("Posterior mean  +  89% PI", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "M6 Causal Bayes — Sensitivity Analysis\n"
        "Posterior means and 89% PIs across prior and IPW specifications",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_sensitivity_forest.png")

    # =======================================================================
    # SENSITIVITY SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY  —  M6 CAUSAL BAYES")
    print("=" * 70)
    print(f"  Base specification: draws=3000, tune=3000, chains=4,")
    print(f"    target_accept=0.95, CHAIN_SEEDS={CHAIN_SEEDS}")
    print(f"    prior: slope sigma=0.50, scale sigma=1.00, IPW 99th pct cap")
    print(f"\n  New fits: 4  (tight prior, wide prior, IPW-95, IPW-99.9)")
    print(f"  Parameters tested: {SENS_PARAMS}")
    print()
    print(f"  {'Specification':<45} {'Div':>5} {'R-hat':>8} {'ESS':>6}")
    print("  " + "─" * 68)
    for d in diags:
        print(f"  {d['label']:<45} {d['ndiv']:>5} "
              f"{d['rhat']:>8.4f} {d['ess']:>6,}")
    print()
    print(f"  Robustness (shift as % of 89% CI half-width):")
    print(f"  {'Parameter':<16} {'Max prior shift':>16} {'Max IPW shift':>14}  Flag")
    print("  " + "─" * 54)
    for p in SENS_PARAMS:
        base_s  = all_stats[base_lbl][p]
        prior_shifts = [
            abs(all_stats[lbl][p]["mean"] - base_s["mean"])
            for lbl in prior_specs if lbl != base_lbl
        ]
        ipw_shifts = [
            abs(all_stats[lbl][p]["mean"] - base_s["mean"])
            for lbl in ipw_specs if lbl != base_lbl
        ]
        pct_prior = max(prior_shifts) / base_s["hw"] * 100
        pct_ipw   = max(ipw_shifts)   / base_s["hw"] * 100
        flag = "⚠" if pct_prior > 20 or pct_ipw > 20 else "✓"
        print(f"  {p:<16} {pct_prior:>15.1f}% {pct_ipw:>13.1f}%  {flag}")
    print()
    print(f"  Figure saved to: {FIG_DIR}")
    print("=" * 70)
