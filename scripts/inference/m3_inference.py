"""
m3_inference.py
===============
Posterior analysis for M3 Causally-Blind Hierarchical Bayes.

Purpose: examine whether the log-linear Normal likelihood specification
allows M3 to fully exploit its access to the five fuel mediators, or
whether the structural mis-match with the true EPA additive formula
(linear in original scale, not log scale) dilutes the mediator signal.
This directly tests the explanation for why M3 and M6 achieve near-
identical MdAPE despite M3 having access to mediators.

M3 specification:
  Data         : ~21,714 rows (CB pipeline, all rows with observed target)
  Features     : UPSTREAM_COLS + MEDIATOR_COLS (no derived/circular fields)
  Model        : Hierarchical Normal on log1p(GHG), non-centred
  IPW          : none
  Sampling     : 3,000 draws, 3,000 tune, 4 chains, target_accept=0.95
  Mediator priors : Normal(0, 2.0) -- agnostic about EPA emission factors

Steps
-----
1  Fetch data (Socrata API)
2  Preprocess (CB pipeline)
3  Load M3 trace
4  MCMC diagnostics
5  Mediator coefficient posteriors  <-- primary analysis
6  Upstream slope comparison: M3 vs M6

Figures saved to results/inference/m3/
  fig_m3_trace.png              -- MCMC trace + marginals (scalar params)
  fig_m3_rhat_ess.png           -- R-hat and ESS bar charts
  fig_m3_mediator_slopes.png    -- Mediator coefficient posteriors (89% PI)
  fig_m3_upstream_vs_m6.png     -- Upstream slopes M3 vs M6 side-by-side

Run (from drafts/complete/): uv run python scripts/inference/m3_inference.py
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

_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "experiments"))

from pipeline import (
    fetch_data, preprocess,
    SEED, APP_TOKEN, MEDIATOR_COLS,
    _flatten_az_result,
)

FIG_DIR      = os.path.join(_ROOT, "results", "inference", "m3")
TRACE_M3     = os.path.join(_ROOT, "results", "inference", "m3_trace.nc")
TRACE_M6     = os.path.join(_ROOT, "results", "inference", "m6_trace.nc")

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

MEDIATOR_BETA_NAMES = [f"beta_{c}" for c in MEDIATOR_COLS]

MEDIATOR_LABELS = {
    f"beta_{c}": c.replace("_use_kbtu", "").replace("_", " ")
    for c in MEDIATOR_COLS
}

UPSTREAM_LABELS = {
    "beta_A": "log floor area",
    "beta_B": "log # buildings",
    "beta_T": "data year",
    "beta_Y": "year built",
}

COLOUR_M3 = "#e76f51"
COLOUR_M6 = "#2a9d8f"


def _pct89(samples):
    a = np.asarray(samples).flatten()
    return np.percentile(a, 5.5), np.percentile(a, 94.5)


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {p}")


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
    # STEP 2 -- PREPROCESS  (CB pipeline)
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 2 -- PREPROCESSING  (CB pipeline)")
    print("=" * 70)
    ctx   = preprocess(df)
    obs   = ctx["obs_cb"]
    n_obs = ctx["n_cb"]
    y_log = ctx["y_log_cb"]
    y_obs = ctx["y_obs_cb"]
    print(f"  CB training rows : {n_obs:,}")

    # Compute mediator standardisation stats (same logic as run_hbm)
    med_stats = {}
    for c in MEDIATOR_COLS:
        vals = obs[c].to_numpy().astype(float)
        med_stats[c] = {"mean": vals.mean(), "std": max(vals.std(), 1e-9)}

    # =======================================================================
    # STEP 3 -- LOAD M3 TRACE
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 3 -- LOADING M3 TRACE")
    print(f"  Source : {TRACE_M3}")
    print("=" * 70)
    if not os.path.exists(TRACE_M3):
        raise FileNotFoundError(
            f"M3 trace not found at {TRACE_M3}. "
            "Re-run scripts/experiments/models.py first."
        )
    trace_m3 = az.from_netcdf(TRACE_M3)
    print(f"  Loaded : {trace_m3.posterior.sizes['chain']} chains x "
          f"{trace_m3.posterior.sizes['draw']} draws")

    trace_m6 = None
    if os.path.exists(TRACE_M6):
        trace_m6 = az.from_netcdf(TRACE_M6)
        print(f"  M6 trace loaded for upstream comparison.")

    # =======================================================================
    # STEP 4 -- MCMC DIAGNOSTICS
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 4 -- MCMC DIAGNOSTICS")
    print("=" * 70)

    all_diag_params = SCALAR_PARAMS + MEDIATOR_BETA_NAMES
    rhat_vals = _flatten_az_result(az.rhat(trace_m3, var_names=all_diag_params))
    ess_vals  = _flatten_az_result(az.ess(trace_m3,  var_names=all_diag_params))
    n_div     = int(trace_m3.sample_stats.diverging.values.sum())
    rhat_max  = float(rhat_vals.max())
    ess_min   = int(ess_vals.min())

    print(f"  R-hat max : {rhat_max:.4f}")
    print(f"  ESS min   : {ess_min:,}")
    print(f"  Divergences: {n_div}")
    for flag, cond, msg in [
        ("R-hat",       rhat_max > 1.01, f"{rhat_max:.4f} > 1.01"),
        ("ESS",         ess_min  < 400,  f"{ess_min:,} < 400"),
        ("divergences", n_div    > 0,    f"{n_div} divergent transitions"),
    ]:
        if cond:
            print(f"  WARNING {flag}: {msg}")

    # ── fig_m3_trace.png ──────────────────────────────────────────────────────
    _CHAIN_COLS = ["#264653", "#2a9d8f", "#e76f51", "#e9c46a"]
    n_p = len(SCALAR_PARAMS)
    fig_tr, axes_tr = plt.subplots(
        n_p, 2, figsize=(12, 2.5 * n_p),
        gridspec_kw={"width_ratios": [3, 1]})
    for i, p in enumerate(SCALAR_PARAMS):
        samp = np.asarray(trace_m3.posterior[p])
        ax_t, ax_m = axes_tr[i]
        for c in range(samp.shape[0]):
            ax_t.plot(samp[c], alpha=0.6, linewidth=0.4,
                      color=_CHAIN_COLS[c % len(_CHAIN_COLS)])
        ax_t.set_ylabel(p, fontsize=7)
        ax_m.hist(samp.flatten(), bins=60, density=True,
                  color=COLOUR_M3, alpha=0.75, edgecolor="none")
        if i == 0:
            ax_t.set_title("Trace", fontsize=8)
            ax_m.set_title("Marginal", fontsize=8)
    fig_tr.suptitle("M3 CB Bayes — MCMC Trace (scalar parameters)",
                    fontsize=11, fontweight="bold")
    fig_tr.tight_layout()
    _save(fig_tr, "fig_m3_trace.png")

    # ── fig_m3_rhat_ess.png ───────────────────────────────────────────────────
    rhat_ds = az.rhat(trace_m3, var_names=all_diag_params)
    ess_ds  = az.ess(trace_m3,  var_names=all_diag_params)
    rv = {p: float(np.asarray(rhat_ds[p]).flat[0]) for p in all_diag_params}
    ev = {p: float(np.asarray(ess_ds[p]).flat[0])  for p in all_diag_params}

    fig_re, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(14, 5))
    rc = ["#e74c3c" if v > 1.01 else "#2ecc71" for v in rv.values()]
    ax_r.barh(list(rv.keys()), list(rv.values()), color=rc,
              edgecolor="white", height=0.6)
    ax_r.axvline(1.01, color="#e74c3c", linestyle="--", linewidth=0.9,
                 label="threshold 1.01")
    for p, v in rv.items():
        ax_r.text(v + 0.0002, list(rv.keys()).index(p),
                  f"{v:.4f}", va="center", fontsize=7)
    ax_r.set_xlabel("R-hat")
    ax_r.set_title("R-hat  (threshold 1.01)", fontsize=9, fontweight="bold")
    ax_r.legend(fontsize=8)

    ec = ["#e74c3c" if v < 400 else "#2ecc71" for v in ev.values()]
    ax_e.barh(list(ev.keys()), list(ev.values()), color=ec,
              edgecolor="white", height=0.6)
    ax_e.axvline(400, color="#e74c3c", linestyle="--", linewidth=0.9,
                 label="threshold 400")
    for p, v in ev.items():
        ax_e.text(v + 20, list(ev.keys()).index(p),
                  f"{v:,.0f}", va="center", fontsize=7)
    ax_e.set_xlabel("Effective Sample Size")
    ax_e.set_title("ESS  (threshold 400)", fontsize=9, fontweight="bold")
    ax_e.legend(fontsize=8)

    fig_re.suptitle("M3 CB Bayes — Convergence Diagnostics",
                    fontsize=11, fontweight="bold")
    fig_re.tight_layout()
    _save(fig_re, "fig_m3_rhat_ess.png")

    # =======================================================================
    # STEP 5 -- MEDIATOR COEFFICIENT POSTERIORS
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 5 -- MEDIATOR COEFFICIENT POSTERIORS")
    print("=" * 70)
    print("  (coefficients are on standardised mediators in log1p(GHG) space)")
    print()

    post = trace_m3.posterior
    med_stats_out = {}
    for bname in MEDIATOR_BETA_NAMES:
        samp = np.asarray(post[bname]).flatten()
        lo, hi = _pct89(samp)
        med_stats_out[bname] = {"mean": float(samp.mean()), "lo": lo, "hi": hi}
        label = MEDIATOR_LABELS[bname]
        print(f"  {label:<30}  mean = {samp.mean():+.4f}  "
              f"89% PI = [{lo:+.4f}, {hi:+.4f}]")

    print()
    print("  Upstream slopes (for comparison):")
    upstream_stats = {}
    for p in ["beta_A", "beta_B", "beta_T", "beta_Y"]:
        samp = np.asarray(post[p]).flatten()
        lo, hi = _pct89(samp)
        upstream_stats[p] = {"mean": float(samp.mean()), "lo": lo, "hi": hi}
        print(f"  {UPSTREAM_LABELS[p]:<30}  mean = {samp.mean():+.4f}  "
              f"89% PI = [{lo:+.4f}, {hi:+.4f}]")

    # ── fig_m3_mediator_slopes.png ────────────────────────────────────────────
    all_params  = MEDIATOR_BETA_NAMES + ["beta_A", "beta_B", "beta_T", "beta_Y"]
    all_labels  = (
        [MEDIATOR_LABELS[b] for b in MEDIATOR_BETA_NAMES]
        + [UPSTREAM_LABELS[p] for p in ["beta_A", "beta_B", "beta_T", "beta_Y"]]
    )
    all_means   = (
        [med_stats_out[b]["mean"] for b in MEDIATOR_BETA_NAMES]
        + [upstream_stats[p]["mean"] for p in ["beta_A", "beta_B", "beta_T", "beta_Y"]]
    )
    all_lo      = (
        [med_stats_out[b]["lo"] for b in MEDIATOR_BETA_NAMES]
        + [upstream_stats[p]["lo"] for p in ["beta_A", "beta_B", "beta_T", "beta_Y"]]
    )
    all_hi      = (
        [med_stats_out[b]["hi"] for b in MEDIATOR_BETA_NAMES]
        + [upstream_stats[p]["hi"] for p in ["beta_A", "beta_B", "beta_T", "beta_Y"]]
    )
    all_colours = [COLOUR_M3] * len(MEDIATOR_BETA_NAMES) + ["#264653"] * 4

    fig_ms, ax_ms = plt.subplots(figsize=(10, 6))
    n_all = len(all_params)
    ypos  = np.arange(n_all)

    for i in range(n_all):
        ax_ms.plot([all_lo[i], all_hi[i]], [i, i],
                   color=all_colours[i], linewidth=3.0, solid_capstyle="round")
        ax_ms.scatter([all_means[i]], [i],
                      color=all_colours[i], s=60, zorder=5)
        ax_ms.text(all_hi[i] + 0.01, i,
                   f"{all_means[i]:+.4f}  [{all_lo[i]:+.4f}, {all_hi[i]:+.4f}]",
                   va="center", ha="left", fontsize=7.5, color="#444444")

    ax_ms.axhline(len(MEDIATOR_BETA_NAMES) - 0.5,
                  color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax_ms.axvline(0, color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax_ms.set_yticks(ypos)
    ax_ms.set_yticklabels(all_labels, fontsize=9)
    ax_ms.set_xlabel(
        "Posterior mean + 89% PI  "
        "(standardised mediators, log1p GHG scale)", fontsize=9)

    from matplotlib.patches import Patch
    ax_ms.legend(
        handles=[
            Patch(color=COLOUR_M3, label="Fuel mediators"),
            Patch(color="#264653", label="Upstream causal features"),
        ],
        fontsize=8, loc="lower right")
    ax_ms.set_title(
        "M3 CB Bayes — Mediator and Upstream Coefficient Posteriors\n"
        "Coefficients are on standardised inputs in log1p(GHG) space",
        fontsize=10, fontweight="bold")
    xlo, xhi = ax_ms.get_xlim()
    ax_ms.set_xlim(xlo, xhi + (xhi - xlo) * 0.55)
    fig_ms.tight_layout()
    _save(fig_ms, "fig_m3_mediator_slopes.png")

    # =======================================================================
    # STEP 6 -- UPSTREAM SLOPE COMPARISON: M3 vs M6
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 6 -- UPSTREAM SLOPE COMPARISON: M3 vs M6")
    print("=" * 70)

    if trace_m6 is None:
        print("  M6 trace not found — skipping comparison.")
    else:
        post_m6 = trace_m6.posterior
        print(f"  {'Parameter':<20} {'M3 mean':>10} {'M3 89% PI':>22}   "
              f"{'M6 mean':>10} {'M6 89% PI':>22}")
        print("  " + "-" * 90)

        m6_upstream = {}
        for p in ["beta_A", "beta_B", "beta_T", "beta_Y"]:
            s3 = np.asarray(post[p]).flatten()
            s6 = np.asarray(post_m6[p]).flatten()
            lo3, hi3 = _pct89(s3)
            lo6, hi6 = _pct89(s6)
            m6_upstream[p] = {"mean": float(s6.mean()), "lo": lo6, "hi": hi6}
            print(f"  {UPSTREAM_LABELS[p]:<20} "
                  f"{s3.mean():>+10.4f}  [{lo3:+.4f}, {hi3:+.4f}]   "
                  f"{s6.mean():>+10.4f}  [{lo6:+.4f}, {hi6:+.4f}]")

        # ── fig_m3_upstream_vs_m6.png ─────────────────────────────────────────
        upstream_keys = ["beta_A", "beta_B", "beta_T", "beta_Y"]
        n_up  = len(upstream_keys)
        ypos  = np.arange(n_up)
        gap   = 0.18

        fig_cmp, ax_cmp = plt.subplots(figsize=(11, 4))

        for i, p in enumerate(upstream_keys):
            s3 = upstream_stats[p]
            s6 = m6_upstream[p]
            label = UPSTREAM_LABELS[p]

            ax_cmp.plot([s3["lo"], s3["hi"]], [i + gap, i + gap],
                        color=COLOUR_M3, linewidth=3.0, solid_capstyle="round")
            ax_cmp.scatter([s3["mean"]], [i + gap],
                           color=COLOUR_M3, s=60, zorder=5, label="M3 CB Bayes" if i == 0 else "")

            ax_cmp.plot([s6["lo"], s6["hi"]], [i - gap, i - gap],
                        color=COLOUR_M6, linewidth=3.0, solid_capstyle="round")
            ax_cmp.scatter([s6["mean"]], [i - gap],
                           color=COLOUR_M6, s=60, zorder=5, label="M6 Causal Bayes" if i == 0 else "")

        ax_cmp.axvline(0, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax_cmp.set_yticks(ypos)
        ax_cmp.set_yticklabels([UPSTREAM_LABELS[p] for p in upstream_keys], fontsize=9)
        ax_cmp.set_xlabel(
            "Posterior mean + 89% PI  (log1p GHG scale)", fontsize=9)
        ax_cmp.set_title(
            "Upstream Slope Comparison: M3 CB Bayes vs M6 Causal Bayes\n"
            "Same four upstream features; M3 prior sigma=2.5, M6 prior sigma=0.5",
            fontsize=10, fontweight="bold")
        ax_cmp.legend(fontsize=8.5)
        fig_cmp.tight_layout()
        _save(fig_cmp, "fig_m3_upstream_vs_m6.png")

        # ── fig_m3_combined.png ───────────────────────────────────────────────
        fig_comb, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(18, 6), gridspec_kw={"wspace": 0.45},
        )

        # Left panel: all mediator + upstream slopes (mirrors fig_m3_mediator_slopes)
        for i in range(n_all):
            ax_left.plot([all_lo[i], all_hi[i]], [i, i],
                         color=all_colours[i], linewidth=3.0, solid_capstyle="round")
            ax_left.scatter([all_means[i]], [i],
                            color=all_colours[i], s=50, zorder=5)
            ax_left.text(all_hi[i] + 0.01, i,
                         f"{all_means[i]:+.4f}  [{all_lo[i]:+.4f}, {all_hi[i]:+.4f}]",
                         va="center", ha="left", fontsize=7, color="#444444")
        ax_left.axhline(len(MEDIATOR_BETA_NAMES) - 0.5,
                        color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax_left.axvline(0, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax_left.set_yticks(np.arange(n_all))
        ax_left.set_yticklabels(all_labels, fontsize=8)
        ax_left.set_xlabel(
            "Posterior mean + 89% PI  (standardised inputs, log1p GHG scale)",
            fontsize=8)
        from matplotlib.patches import Patch as _Patch
        ax_left.legend(
            handles=[
                _Patch(color=COLOUR_M3, label="Fuel mediators"),
                _Patch(color="#264653", label="Upstream causal features"),
            ],
            fontsize=7.5, loc="lower right")
        xlo_l, xhi_l = ax_left.get_xlim()
        ax_left.set_xlim(xlo_l, xhi_l + (xhi_l - xlo_l) * 0.55)
        ax_left.set_title(
            "(a)  M3 Mediator and Upstream Coefficient Posteriors",
            fontsize=9, fontweight="bold", pad=6)

        # Right panel: upstream slope comparison M3 vs M6 (mirrors fig_m3_upstream_vs_m6)
        gap_c = 0.18
        ypos_c = np.arange(len(upstream_keys))
        for i, p in enumerate(upstream_keys):
            s3 = upstream_stats[p]
            s6 = m6_upstream[p]
            ax_right.plot([s3["lo"], s3["hi"]], [i + gap_c, i + gap_c],
                          color=COLOUR_M3, linewidth=3.0, solid_capstyle="round")
            ax_right.scatter([s3["mean"]], [i + gap_c],
                             color=COLOUR_M3, s=50, zorder=5,
                             label="M3 CB Bayes" if i == 0 else "")
            ax_right.plot([s6["lo"], s6["hi"]], [i - gap_c, i - gap_c],
                          color=COLOUR_M6, linewidth=3.0, solid_capstyle="round")
            ax_right.scatter([s6["mean"]], [i - gap_c],
                             color=COLOUR_M6, s=50, zorder=5,
                             label="M6 Causal Bayes" if i == 0 else "")
        ax_right.axvline(0, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax_right.set_yticks(ypos_c)
        ax_right.set_yticklabels([UPSTREAM_LABELS[p] for p in upstream_keys], fontsize=8)
        ax_right.set_xlabel("Posterior mean + 89% PI  (log1p GHG scale)", fontsize=8)
        ax_right.legend(fontsize=8)
        ax_right.set_title(
            "(b)  Upstream Slopes: M3 CB Bayes vs M6 Causal Bayes",
            fontsize=9, fontweight="bold", pad=6)

        fig_comb.suptitle(
            "M3 CB Bayes — Posterior Inference: "
            "Mediator Dilution and Upstream Comparison",
            fontsize=11, fontweight="bold",
        )
        fig_comb.tight_layout()
        _save(fig_comb, "fig_m3_combined.png")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY  —  M3 MEDIATOR POSTERIORS")
    print("=" * 70)
    print()
    print("  Mediator slopes (standardised inputs, log1p GHG scale):")
    for bname in MEDIATOR_BETA_NAMES:
        s = med_stats_out[bname]
        label = MEDIATOR_LABELS[bname]
        width = s["hi"] - s["lo"]
        print(f"    {label:<30}  mean={s['mean']:+.4f}  "
              f"89% PI width={width:.4f}")
    print()
    print("  Key question: are mediator slopes large relative to upstream slopes?")
    print("  If the log-linear likelihood mis-specification dilutes the mediator")
    print("  signal, mediator posterior means will be small and/or wide relative")
    print("  to the variance they should explain given the EPA formula structure.")
    print()
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 70)
