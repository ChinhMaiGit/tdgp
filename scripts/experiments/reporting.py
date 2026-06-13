"""
reporting.py
============
Printed comparison table, LaTeX paper table, and static visualisations
for the 2x3 TDGP factorial experiment.

Context dict keys expected from pipeline.preprocess()
------------------------------------------------------
obs_cb, y_obs_cb, y_log_cb, bids_cb, splits_cb,
obs_causal, y_obs_causal, y_log_causal, bids_causal, splits_causal,
ipw_w, non_compliant, df_full, n_cb, n_causal, cb_medians
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

from pathlib import Path

from pipeline import (
    MODEL_META, COLOURS, GROUPS, APPROACH, SHORTS,
    CB_MODELS, CAUSAL_MODELS, BAYES_MODELS,
)

# Bundle results directory: drafts/complete/results/experiments/
_OUT = Path(__file__).resolve().parents[2] / "results" / "experiments"
_OUT.mkdir(parents=True, exist_ok=True)

# ── Matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

# ── Model ordering ────────────────────────────────────────────────────────────
CB_ORDER     = ["M1 CB OLS",     "M2 CB XGB",     "M3 CB Bayes"]
CAUSAL_ORDER = ["M4 Causal OLS", "M5 Causal XGB", "M6 Causal Bayes"]
ALL_ORDER    = CB_ORDER + CAUSAL_ORDER

# M4 Causal OLS colour changed to teal to distinguish from M5 dark green
COLOURS_FIXED = {
    **COLOURS,
    "M4 Causal OLS": "#17BECF",
}

APPROACH_LABELS = {
    "M1 CB OLS":       "OLS",
    "M2 CB XGB":       "XGBoost",
    "M3 CB Bayes":     "Hier. Bayes",
    "M4 Causal OLS":   "OLS",
    "M5 Causal XGB":   "XGBoost",
    "M6 Causal Bayes": "Hier. Bayes",
}

LEGEND_LABELS = {
    "M1 CB OLS":
        "M1  CB OLS       [causally-blind | frequentist | ALL features | KFold]",
    "M2 CB XGB":
        "M2  CB XGB       [causally-blind | ML | ALL features | KFold tuned]",
    "M3 CB Bayes":
        "M3  CB Bayes     [causally-blind | Bayesian | ALL features | PSIS-LOO]",
    "M4 Causal OLS":
        "M4  Causal OLS   [causal | frequentist | upstream | GroupKFold]",
    "M5 Causal XGB":
        "M5  Causal XGB   [causal | ML | upstream | GroupKFold tuned] <- ceiling",
    "M6 Causal Bayes":
        "M6  Causal Bayes [causal | Bayesian | upstream | PSIS-LOO] <- structural",
}

LINE_STYLES_DEPLOY = {
    "M1 CB OLS":       ("--",         2.0),
    "M2 CB XGB":       ((0, (3, 1)),  2.0),
    "M3 CB Bayes":     ("-.",          2.0),
    "M4 Causal OLS":   ("-",           2.0),
    "M5 Causal XGB":   ("-",           2.6),
    "M6 Causal Bayes": ("-",           2.0),
}

IPW_PAIRS = [
    ("Frequentist", "M1 CB OLS",   "M4 Causal OLS"),
    ("ML",          "M2 CB XGB",   "M5 Causal XGB"),
    ("Bayesian",    "M3 CB Bayes", "M6 Causal Bayes"),
]

# Shared title padding — pushes titles up, away from bar labels
TITLE_PAD = 18


# ===========================================================================
# 0. CONTEXT VALIDATION
# ===========================================================================

def validate_ctx(ctx: dict) -> None:
    """Verify ctx contains all keys expected by reporting functions."""
    required = {
        "obs_cb",      "y_obs_cb",     "y_log_cb",     "bids_cb",
        "obs_causal",  "y_obs_causal", "y_log_causal", "bids_causal",
        "ipw_w", "non_compliant", "df_full",
        "n_cb", "n_causal", "cb_medians",
        "splits_cb", "splits_causal",
    }
    missing = required - set(ctx.keys())
    if missing:
        raise KeyError(
            f"ctx is missing keys: {sorted(missing)}\n"
            f"Available: {sorted(ctx.keys())}"
        )
    print("  ctx validation passed — all required keys present")


# ===========================================================================
# 1. PRINTED COMPARISON TABLE
# ===========================================================================

def print_comparison(results: dict) -> None:
    """Three-block printed comparison table."""
    w_name, w_col = 22, 13
    full_w = w_name + w_col * 5 + 4
    sep    = "-" * full_w
    hdr    = (f"  {'Model':<{w_name}}"
              f"{'R2(log)':>{w_col}}"
              f"{'MdAPE %':>{w_col}}"
              f"{'Med RMSE':>{w_col}}"
              f"{'89% cov':>{w_col}}"
              f"{'ELPD':>{w_col}}")

    def _row(name, m):
        bayes_mark = " *" if "Bayes" in name else "  "
        return (f"  {name + bayes_mark:<{w_name}}"
                f"{m['r2_mean']:>+{w_col}.4f}"
                f"{m['mape_mean']:>{w_col}.2f}%"
                f"{m['rmse_median']:>{w_col},.0f}"
                f"{m['cov_mean']:>{w_col}.1f}%"
                f"{m['elpd']:>+{w_col}.0f}")

    print("\n" + "=" * full_w)
    print("  2x3 FACTORIAL COMPARISON  (out-of-sample)".center(full_w))
    print("=" * full_w)

    print("\n  -- CAUSALLY-BLIND PIPELINE "
          "(no IPW | ALL features | KFold) --")
    print(hdr); print("  " + sep)
    for n in CB_ORDER:
        if n in results: print(_row(n, results[n]))

    print("\n  -- CAUSAL PIPELINE "
          "(IPW | upstream features | GroupKFold) --")
    print(hdr); print("  " + sep)
    for n in CAUSAL_ORDER:
        if n in results: print(_row(n, results[n]))
    print("  " + sep)

    print("\n  -- IPW EFFECT: causal minus causally-blind "
          "(same approach) --")
    delta_hdr = (f"  {'Approach':<{w_name}}"
                 f"{'Delta R2':>{w_col}}"
                 f"{'Delta MdAPE':>{w_col}}"
                 f"{'Delta RMSE':>{w_col}}"
                 f"{'Delta cov':>{w_col}}"
                 f"{'Delta ELPD':>{w_col}}")
    print(delta_hdr); print("  " + sep)
    for approach, cb_m, c_m in IPW_PAIRS:
        if cb_m not in results or c_m not in results: continue
        dr2   = results[c_m]["r2_mean"]     - results[cb_m]["r2_mean"]
        dmape = results[c_m]["mape_mean"]   - results[cb_m]["mape_mean"]
        drmse = results[c_m]["rmse_median"] - results[cb_m]["rmse_median"]
        dcov  = results[c_m]["cov_mean"]    - results[cb_m]["cov_mean"]
        delpd = results[c_m]["elpd"]        - results[cb_m]["elpd"]
        print(f"  {approach:<{w_name}}"
              f"{dr2:>+{w_col}.4f}"
              f"{dmape:>+{w_col}.2f}%"
              f"{drmse:>+{w_col},.0f}"
              f"{dcov:>+{w_col}.1f}%"
              f"{delpd:>+{w_col}.0f}")
    print("  " + sep)

    print(f"""
  Notes
  {"-" * 60}
  Causally-blind pipeline: ~21,714 rows (all observed target), ALL features, KFold, no IPW.
  Causal pipeline: ~21,406 rows (compliant only), upstream only, GroupKFold, IPW.
  * Bayesian RMSE from PSIS-LOO predictive means — not comparable
    to fold-median RMSE of OLS/XGBoost.
  ELPD  : Normal-approx CV (M1,M2,M4,M5); PSIS-LOO (M3,M6).
  Delta MdAPE: positive = causal has HIGHER error.
    ML (+18.9 pp): EXPECTED — CB exploits post-treatment mediators + circular
                  derived features in CB_ALL_FEATURES, both unavailable at deployment.
  """)


# ===========================================================================
# 2. LATEX TABLE FOR PAPER
# ===========================================================================

def print_latex_table(results: dict) -> None:
    """Generate a publication-ready LaTeX table."""

    def _r2(v):   return f"{v:+.3f}"
    def _mape(v): return f"{v:.1f}\\%"
    def _rmse(v): return f"{v:,.0f}"
    def _cov(v):  return f"{v:.1f}\\%"
    def _elpd(v): return f"{v:,.0f}"

    def _model_row(mname, m):
        lbl = APPROACH_LABELS.get(mname, mname)
        if "Bayes" in mname:
            lbl += r"$^{\dagger}$"
        return (f"& {lbl} & "
                f"{_r2(m['r2_mean'])} & "
                f"{_mape(m['mape_mean'])} & "
                f"{_rmse(m['rmse_median'])} & "
                f"{_cov(m['cov_mean'])} & "
                f"{_elpd(m['elpd'])} \\\\")

    def _delta_row(approach, cb_m, c_m):
        dr2   = results[c_m]["r2_mean"]     - results[cb_m]["r2_mean"]
        dmape = results[c_m]["mape_mean"]   - results[cb_m]["mape_mean"]
        drmse = results[c_m]["rmse_median"] - results[cb_m]["rmse_median"]
        dcov  = results[c_m]["cov_mean"]    - results[cb_m]["cov_mean"]
        delpd = results[c_m]["elpd"]        - results[cb_m]["elpd"]
        return (f"& {approach} & "
                f"{_r2(dr2)} & "
                f"{_mape(dmape)} & "
                f"{_rmse(drmse)} & "
                f"{_cov(dcov)} & "
                f"{_elpd(delpd)} \\\\")

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\small")
    lines.append(
        r"\caption{"
        r"Out-of-sample performance for the 2$\times$3 factorial "
        r"design (realistic comparison). "
        r"Causally-blind pipeline: $\sim$21{,}714 rows (all observed target), "
        r"ALL features (including derived metrics), KFold(5), no IPW. "
        r"Causal pipeline: $\sim$21{,}406 rows (compliant only), "
        r"upstream features only, GroupKFold(5) on building ID, IPW back-door adjustment. "
        r"The large positive $\Delta$MdAPE for ML is "
        r"\emph{expected}: the causally-blind XGBoost exploits post-treatment mediators "
        r"(e.g.\ \texttt{electricity\_use\_kbtu}) and circular derived metrics "
        r"(e.g.\ \texttt{ghg\_intensity\_kg\_co2e\_sq\_ft}) in \texttt{CB\_ALL\_FEATURES}, "
        r"both groups unavailable or invalid at deployment. "
        r"ELPD: Normal-approximation CV for OLS and XGBoost; "
        r"PSIS-LOO for Hierarchical Bayes. "
        r"$^{\ddagger}$~Bayesian RMSE from PSIS-LOO predictive means; "
        r"not comparable to fold-median RMSE."
        r"}"
    )
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Pipeline & Approach "
        r"& $R^2(\log)$ & MdAPE "
        r"& Med.\ RMSE$^{\ddagger}$ "
        r"& 89\% PI cov. & ELPD \\"
    )

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\textit{Panel A: Causally-blind pipeline "
        r"(no IPW, ALL features, KFold)}} \\"
    )
    lines.append(r"\midrule")
    for mname in CB_ORDER:
        if mname in results:
            lines.append(_model_row(mname, results[mname]))

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\textit{Panel B: Causal pipeline "
        r"(IPW, upstream features, GroupKFold)}} \\"
    )
    lines.append(r"\midrule")
    for mname in CAUSAL_ORDER:
        if mname in results:
            lines.append(_model_row(mname, results[mname]))

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\textit{Panel C: IPW effect "
        r"(causal $-$ causally-blind; "
        r"positive $\Delta$MdAPE = causal has higher error)}} \\"
    )
    lines.append(r"\midrule")
    for approach, cb_m, c_m in IPW_PAIRS:
        if cb_m in results and c_m in results:
            lines.append(_delta_row(approach, cb_m, c_m))

    lines.append(r"\bottomrule")
    lines.append(
        r"\multicolumn{7}{l}{"
        r"\footnotesize "
        r"$^{\dagger}$~PSIS-LOO; fold variance not applicable. "
        r"$^{\ddagger}$~Bayesian RMSE from LOO predictive means."
        r"} \\"
    )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    print("\n" + "=" * 65)
    print("  LATEX TABLE")
    print("=" * 65)
    print(latex_str)
    with open(_OUT / "table_comparison.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"\n  Saved -> {_OUT / 'table_comparison.tex'}")


# ===========================================================================
# 3. SHARED PLOT HELPERS
# ===========================================================================

def _make_legend_patches(model_names: list) -> list:
    """Build mpatches.Patch handles using COLOURS_FIXED."""
    patches = []
    for m in model_names:
        colour = COLOURS_FIXED.get(m, COLOURS.get(m))
        if colour is None:
            continue
        patches.append(mpatches.Patch(
            color=colour,
            label=LEGEND_LABELS.get(m, m),
        ))
    return patches


def _style_ax(ax, ylabel: str = "", title: str = ""):
    """Apply consistent spine / tick / label styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.7)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=TITLE_PAD)


def _pipeline_separator(ax, after_idx: int,
                         label_text: str = "causally-blind  |  causal"):
    lo, hi = ax.get_ylim()
    ax.axvline(after_idx + 0.5, color="#aaaaaa", linestyle="--",
               linewidth=0.9, zorder=1)
    ax.text(after_idx + 0.6, hi * 0.97, label_text,
            fontsize=6.5, color="#888888", va="top", ha="left")


def _bottom_legend(fig, model_names: list,
                   n_cols: int = 2, bbox_y: float = 0.01):
    """Place a patch legend at the bottom of the figure."""
    patches = _make_legend_patches(model_names)
    fig.legend(
        handles        = patches,
        loc            = "lower center",
        ncol           = n_cols,
        fontsize       = 7.5,
        frameon        = False,
        bbox_to_anchor = (0.5, bbox_y),
        bbox_transform = fig.transFigure,
        handlelength   = 1.2,
        handleheight   = 0.9,
        columnspacing  = 1.5,
        labelspacing   = 0.50,
    )


# ===========================================================================
# 4. MAIN COMPARISON FIGURE
# ===========================================================================

def plot_main_comparison(results: dict,
                          save_path: str = "fig_comparison.png"):
    """
    4-panel comparison figure.

    P1  R2(log)
    P2  MdAPE %    — M2 hatched, plain bold number above bar
    P3  ELPD       — M2 hatched, bold value at 72% of axis height,
                     upward arrow indicator; NO stub bar
    P4  IPW effect — y=0 is the axis floor (no separate axhline),
                     bottom spine IS the zero reference
    """
    all_models = [m for m in MODEL_META if m in results]
    n_cb       = sum(1 for m in all_models if GROUPS[m] == "causally-blind")
    sep_idx    = n_cb - 1
    colours    = [COLOURS_FIXED.get(m, COLOURS[m]) for m in all_models]
    shorts     = [SHORTS[m] for m in all_models]

    r2s     = [results[m]["r2_mean"]   for m in all_models]
    r2_errs = [results[m]["r2_std"]    for m in all_models]
    mapes   = [results[m]["mape_mean"] for m in all_models]
    m_errs  = [results[m]["mape_std"]  for m in all_models]
    elpds   = [results[m]["elpd"]      for m in all_models]
    e_errs  = [results[m]["elpd_se"]   for m in all_models]

    m2_name = "M2 CB XGB"
    m2_idx  = all_models.index(m2_name) if m2_name in all_models else -1

    # Dynamic M2 ELPD for P3 title
    m2_elpd_str = (
        f"{results[m2_name]['elpd']:+,.0f}"
        if m2_name in results else "off-scale"
    )

    # IPW effect per approach
    ipw_labels, ipw_deltas, ipw_cols = [], [], []
    for approach, cb_m, c_m in IPW_PAIRS:
        if cb_m in results and c_m in results:
            d = results[c_m]["mape_mean"] - results[cb_m]["mape_mean"]
            ipw_labels.append(approach)
            ipw_deltas.append(d)
            ipw_cols.append("#2ecc71" if d < 0 else "#e74c3c")

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.60, wspace=0.32,
                   left=0.07, right=0.97,
                   top=0.88, bottom=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    x   = np.arange(len(all_models))

    # ── P1: R2(log) ──────────────────────────────────────────────────────────
    bars1 = ax1.bar(x, r2s, color=colours, width=0.6,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax1.errorbar(x, r2s, yerr=r2_errs, fmt="none", color="#333333",
                 capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
    ax1.set_ylim(0, 1.22)
    lo1, hi1 = ax1.get_ylim()
    off1 = (hi1 - lo1) * 0.02
    for bar, v in zip(bars1, r2s):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + off1,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7, color="#222222")
    _pipeline_separator(ax1, sep_idx)
    ax1.set_xticks(x); ax1.set_xticklabels(shorts, fontsize=8)
    _style_ax(ax1, ylabel="R\u00b2(log)",
              title="Out-of-sample R\u00b2(log)")

    # ── P2: MdAPE % ──────────────────────────────────────────────────────────
    mapes_excl = [v for i, v in enumerate(mapes) if i != m2_idx]
    y2_max     = max(mapes_excl) * 1.28 if mapes_excl else 50.0

    bars2 = ax2.bar(x, mapes, color=colours, width=0.6,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax2.errorbar(x, mapes, yerr=m_errs, fmt="none", color="#333333",
                 capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
    ax2.set_ylim(0, y2_max)
    lo2, hi2 = ax2.get_ylim()
    off2 = (hi2 - lo2) * 0.02

    for i, (bar, v) in enumerate(zip(bars2, mapes)):
        if i == m2_idx:
            bar.set_hatch("////")
            bar.set_edgecolor(colours[i])
            bar.set_linewidth(1.2)
            label_y = min(v + off2 * 3, hi2 * 0.92)
            ax2.text(bar.get_x() + bar.get_width() / 2, label_y,
                     f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=8, color=colours[i], fontweight="bold")
        else:
            y_label = min(v + off2, hi2 * 0.96)
            ax2.text(bar.get_x() + bar.get_width() / 2, y_label,
                     f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=7, color="#222222")

    _pipeline_separator(ax2, sep_idx)
    ax2.set_xticks(x); ax2.set_xticklabels(shorts, fontsize=8)
    _style_ax(ax2, ylabel="MdAPE (%)",
              title="Out-of-sample MdAPE")

    # ── P3: ELPD ─────────────────────────────────────────────────────────────
    elpds_excl = [v for i, v in enumerate(elpds) if i != m2_idx]
    lo3_data   = min(elpds_excl) if elpds_excl else -25000
    hi3_data   = max(elpds_excl) if elpds_excl else 0
    margin3    = (hi3_data - lo3_data) * 0.28
    y3_lo      = lo3_data - margin3
    y3_hi      = hi3_data + margin3 * 2.2   # headroom for M2 value label

    bars3 = ax3.bar(x, elpds, color=colours, width=0.6,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax3.errorbar(x, elpds, yerr=e_errs, fmt="none", color="#333333",
                 capsize=3, capthick=0.8, linewidth=0.8, zorder=4)
    ax3.set_ylim(y3_lo, y3_hi)
    lo3l, hi3l = ax3.get_ylim()
    yr3        = hi3l - lo3l

    for i, (bar, v) in enumerate(zip(bars3, elpds)):
        if i == m2_idx:
            # Hatch the (invisible) bar — no stub bar added
            bar.set_hatch("////")
            bar.set_edgecolor(colours[i])
            bar.set_linewidth(1.2)

            m2_x = bar.get_x() + bar.get_width() / 2

            # Value label at 72% of axis height — clear of title and bars
            label_y = lo3l + yr3 * 0.72
            ax3.text(
                m2_x, label_y,
                f"{v:,.0f}",
                ha="center", va="bottom",
                fontsize=8, color=colours[i], fontweight="bold",
            )

            # Upward arrow: from below the label to near the axis top
            ax3.annotate(
                "",
                xy     = (m2_x, hi3l * 0.97),
                xytext = (m2_x, label_y + yr3 * 0.08),
                arrowprops=dict(
                    arrowstyle = "-|>",
                    color      = colours[i],
                    lw         = 1.2,
                ),
            )
        else:
            va  = "bottom" if v >= 0 else "top"
            off = yr3 * 0.025
            y   = v + off if v >= 0 else v - off
            label_str = f"{v:,.0f}" if v != 0 else "0"
            if y3_lo < y < y3_hi:
                ax3.text(bar.get_x() + bar.get_width() / 2, y,
                         label_str, ha="center", va=va,
                         fontsize=7, color="#222222")

    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _pipeline_separator(ax3, sep_idx)
    ax3.set_xticks(x); ax3.set_xticklabels(shorts, fontsize=8)
    _style_ax(ax3, ylabel="ELPD",
              title="Expected Log Predictive Density (ELPD)")

    # ── P4: IPW effect ───────────────────────────────────────────────────────
    # All deltas are positive (causal > causally-blind on MdAPE).
    # Set y floor to exactly 0 so the bottom spine IS the zero reference.
    # No axhline drawn — avoids the double horizontal axis visual.
    x4        = np.arange(len(ipw_labels))
    max_delta = max(abs(d) for d in ipw_deltas) if ipw_deltas else 1.0
    min_bar_h = max_delta * 0.04
    rendered_deltas = [
        d if abs(d) >= min_bar_h
        else (min_bar_h if d >= 0 else -min_bar_h)
        for d in ipw_deltas
    ]

    bars4 = ax4.bar(x4, rendered_deltas, color=ipw_cols, width=0.50,
                    edgecolor="white", linewidth=0.5, zorder=3)

    y4_max = max_delta * 1.35
    ax4.set_ylim(0, y4_max)
    lo4l, hi4l = ax4.get_ylim()
    yr4 = hi4l - lo4l

    # Make the bottom spine prominent — it serves as the zero reference
    ax4.spines["bottom"].set_linewidth(1.2)
    ax4.spines["bottom"].set_color("#333333")

    for i, (bar, actual_v) in enumerate(zip(bars4, ipw_deltas)):
        rendered_v = rendered_deltas[i]
        off = yr4 * 0.03
        y   = rendered_v + off
        ax4.text(bar.get_x() + bar.get_width() / 2, y,
                 f"{actual_v:+.1f} pp",
                 ha="center", va="bottom",
                 fontsize=8.5, color="#222222", fontweight="bold")

    ax4.set_xticks(x4)
    ax4.set_xticklabels(ipw_labels, fontsize=10)

    _style_ax(ax4,
              ylabel="MdAPE change (pp)\n(causal minus causally-blind)",
              title="IPW Correction Effect on MdAPE")

    _bottom_legend(fig, all_models, n_cols=2, bbox_y=0.01)
    fig.suptitle(
        "Causally-Blind vs Causal Pipeline \u2014 2\u00d73 Factorial Comparison\n"
        "Chicago Energy Benchmarking Dataset",
        fontsize=11, fontweight="bold", y=0.97,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ===========================================================================
# 5. MAPE BY FLOOR-AREA QUINTILE
# ===========================================================================

def plot_mape_by_quintile(preds: dict, ctx: dict,
                           save_path: str = "fig_quintile.png"):
    """
    MdAPE by floor-area quintile, log-scale y-axis.
    In-plot annotation explains M2 CB XGB near-invisible bars.
    """
    obs_cb   = ctx["obs_cb"];     y_obs_cb = ctx["y_obs_cb"]
    obs_c    = ctx["obs_causal"]; y_obs_c  = ctx["y_obs_causal"]

    def _qmapes(pred_log, y_orig, src_df):
        area     = src_df["gross_floor_area_buildings_sq_ft"].to_numpy()
        q_bounds = np.percentile(area, [0, 20, 40, 60, 80, 100])
        q_idx    = np.digitize(area, q_bounds[1:-1])
        out = []
        for q in range(5):
            mask = q_idx == q
            if mask.sum() == 0:
                out.append(0.0); continue
            out.append(float(np.median(np.abs(
                (y_orig[mask]
                 - np.clip(np.expm1(pred_log[mask]), 0, None))
                / (y_orig[mask] + 1e-6)
            )) * 100))
        return out

    q_labels   = ["Q1\n(smallest 20%)", "Q2", "Q3", "Q4",
                  "Q5\n(largest 20%)"]
    model_list = [m for m in ALL_ORDER if m in preds]
    n_models   = len(model_list)
    x          = np.arange(5)
    width      = 0.78 / n_models

    bot_pad  = 1.90
    fig_h    = 6.0 + bot_pad
    fig, ax  = plt.subplots(figsize=(13, fig_h))
    bottom_f = bot_pad / fig_h
    fig.subplots_adjust(left=0.09, right=0.97,
                        top=0.88, bottom=bottom_f)

    all_mapes, bar_groups = [], []
    for i, mname in enumerate(model_list):
        pred_log = preds[mname]
        if mname in CB_MODELS:
            if len(pred_log) != len(y_obs_cb): continue
            mq = _qmapes(pred_log, y_obs_cb, obs_cb)
        else:
            if len(pred_log) != len(y_obs_c): continue
            mq = _qmapes(pred_log, y_obs_c, obs_c)

        offset = (i - n_models / 2 + 0.5) * width
        colour = COLOURS_FIXED.get(mname, COLOURS.get(mname, "#888888"))
        bars   = ax.bar(x + offset, mq, width=width * 0.88,
                        color=colour, edgecolor="white",
                        linewidth=0.4, zorder=3)
        all_mapes.extend(mq)
        bar_groups.append((mname, bars, mq))

    if not all_mapes:
        plt.close(fig); return

    ax.set_yscale("log")
    y_max = max(v for v in all_mapes if v > 0)
    ax.set_ylim(0.8, y_max * 2.5)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    for mname, bars, mq in bar_groups:
        for bar, v in zip(bars, mq):
            if v < 1.0: continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v * 1.08, f"{v:.1f}%",
                    ha="center", va="bottom",
                    fontsize=6, color="#222222", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=9)
    _style_ax(
        ax,
        ylabel="MdAPE (%, log scale)",
        title="MdAPE by Floor-Area Quintile",
    )

    legend_y = (bot_pad * 0.50) / fig_h
    _bottom_legend(fig, model_list, n_cols=2, bbox_y=legend_y)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ===========================================================================
# 6. DEPLOYMENT TEST FIGURE
# ===========================================================================

def plot_deployment_test(deploy_preds: dict,
                          save_path: str = "fig_deployment.png"):
    """
    Density line chart for all six models on non-compliant records.

    All six models appear in the legend regardless of whether their
    predictions are available. Failed models (deploy_preds[m] is None)
    get a grey entry with an '(not available)' suffix — no note box
    cluttering the plot area.

    X-axis capped at 13 to suppress the M1 CB OLS clip-ceiling spike.
    A small annotation in the corner explains the omission.
    """
    bins    = np.linspace(0, 13, 100)
    centres = (bins[:-1] + bins[1:]) / 2

    bot_pad  = 2.20
    fig_h    = 5.5 + bot_pad
    fig, ax  = plt.subplots(figsize=(10, fig_h))
    bottom_f = bot_pad / fig_h
    fig.subplots_adjust(left=0.09, right=0.97,
                        top=0.87, bottom=bottom_f)

    plotted_models = []
    failed_models  = []

    for mname in ALL_ORDER:
        pred_orig = deploy_preds.get(mname)
        colour    = COLOURS_FIXED.get(mname, COLOURS.get(mname, "#888888"))
        ls, lw    = LINE_STYLES_DEPLOY.get(mname, ("-", 1.8))

        if pred_orig is not None and len(pred_orig) > 0:
            log_clipped = np.clip(np.log1p(pred_orig), 0, 13)
            counts, _   = np.histogram(log_clipped, bins=bins, density=False)
            pct         = counts / max(counts.sum(), 1)
            ax.plot(centres, pct,
                    color=colour, linestyle=ls, linewidth=lw)
            plotted_models.append(mname)
        else:
            failed_models.append(mname)

    ax.set_xlabel(
        "log\u2081\u208a(Predicted GHG, metric tons CO\u2082e)",
        fontsize=8, labelpad=4)
    ax.set_xlim(0, 13)

    # Secondary x-axis: original-scale GHG equivalents
    ref_ghg    = [1, 10, 100, 1000, 10000, 100000]
    ref_log    = [np.log1p(v) for v in ref_ghg]
    ref_labels = ["1", "10", "100", "1 K", "10 K", "100 K"]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ref_log)
    ax2.set_xticklabels(ref_labels, fontsize=7, color="#666666")
    ax2.set_xlabel("Original-scale GHG (metric tons CO\u2082e)",
                   fontsize=7, color="#666666", labelpad=2)
    ax2.tick_params(axis="x", length=2, width=0.5, color="#666666")
    ax2.spines["top"].set_linewidth(0.5)
    ax2.spines["top"].set_color("#cccccc")
    ax2.spines["right"].set_visible(False)

    _style_ax(
        ax,
        ylabel="Proportion of non-compliant buildings",
        title="Predicted GHG Emissions for Non-Compliant Buildings",
    )

    # ── Legend — all six models always appear ─────────────────────────────────
    legend_handles = []
    for mname in ALL_ORDER:
        colour    = COLOURS_FIXED.get(mname, COLOURS.get(mname, "#888888"))
        ls, lw    = LINE_STYLES_DEPLOY.get(mname, ("-", 1.8))
        full_label = LEGEND_LABELS.get(mname, mname)

        if mname in failed_models:
            handle = plt.Line2D(
                [0], [0],
                color="#aaaaaa", linestyle="--", linewidth=1.2,
                label=full_label + "  (not available)",
            )
        else:
            handle = plt.Line2D(
                [0], [0],
                color=colour,
                linestyle=ls if isinstance(ls, str) else "-",
                linewidth=lw,
                label=full_label,
            )
        legend_handles.append(handle)

    legend_y = (bot_pad * 0.45) / fig_h
    fig.legend(
        handles        = legend_handles,
        loc            = "lower center",
        ncol           = 2,
        fontsize       = 7.5,
        frameon        = False,
        bbox_to_anchor = (0.5, legend_y),
        bbox_transform = fig.transFigure,
        handlelength   = 1.4,
        handleheight   = 0.9,
        columnspacing  = 1.5,
        labelspacing   = 0.50,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ===========================================================================
# 7. PREDICTED VS ACTUAL HEATMAPS
# ===========================================================================

def plot_pred_vs_actual(preds: dict, ctx: dict, results: dict,
                         save_path: str = "fig_pred_vs_actual.png",
                         n_bins: int = 60):
    """
    2 x 3 grid of predicted-vs-actual log1p(GHG) density heatmaps.
    Log-normalised colour scale. In-panel R2/MdAPE annotations.
    """
    y_log_cb = ctx["y_log_cb"]
    y_log_c  = ctx["y_log_causal"]

    cb_keys     = [m for m in CB_ORDER     if m in preds]
    causal_keys = [m for m in CAUSAL_ORDER if m in preds]
    all_keys    = cb_keys + causal_keys

    n_cols = 3
    n_rows = 2
    bins   = np.linspace(0, 13, n_bins + 1)
    xe     = (bins[:-1] + bins[1:]) / 2

    cell_w    = 3.5
    cell_h    = 3.5
    top_pad   = 1.30
    bot_pad   = 1.90
    left_pad  = 0.70
    right_pad = 0.20
    fig_w     = n_cols * cell_w + left_pad + right_pad
    fig_h     = n_rows * cell_h + top_pad  + bot_pad

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes      = axes.flatten()

    left_frac   = left_pad  / fig_w
    right_frac  = 1.0 - right_pad / fig_w
    top_frac    = 1.0 - top_pad   / fig_h
    bottom_frac = bot_pad          / fig_h

    fig.subplots_adjust(
        left=left_frac, right=right_frac,
        top=top_frac,   bottom=bottom_frac,
        hspace=0.65, wspace=0.42,
    )

    for idx, key in enumerate(all_keys):
        ax       = axes[idx]
        pred_log = preds[key]
        y_ref    = y_log_cb if key in CB_MODELS else y_log_c
        n_use    = min(len(pred_log), len(y_ref))
        y_use    = y_ref[:n_use]
        p_use    = np.clip(pred_log[:n_use], 0, 13)

        counts, _, _ = np.histogram2d(y_use, p_use, bins=bins)
        ax.pcolormesh(
            xe, xe, counts.T + 0.5,
            cmap    = "Blues",
            norm    = LogNorm(vmin=0.5, vmax=counts.max() + 0.5),
            shading = "auto",
            zorder  = 2,
        )
        ax.plot([0, 13], [0, 13], color="red", linestyle="--",
                linewidth=1.0, zorder=3)

        ax.set_title(
            key,
            fontsize=7.5, fontweight="bold", pad=5,
        )
        ax.set_xlabel("Actual log\u2081\u208a(GHG)", fontsize=7, labelpad=3)
        ax.set_ylabel("Predicted log\u2081\u208a(GHG)", fontsize=7, labelpad=3)
        ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
        ax.set_xlim(0, 13); ax.set_ylim(0, 13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        if results is not None and key in results:
            r2_val   = results[key]["r2_mean"]
            mape_val = results[key]["mape_mean"]
            ax.text(0.04, 0.97,
                    f"R\u00b2={r2_val:.3f}\nMdAPE={mape_val:.1f}%",
                    transform=ax.transAxes,
                    fontsize=6.5, color="#333333",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white",
                              edgecolor="#cccccc",
                              alpha=0.85))

    for idx in range(len(all_keys), len(axes)):
        axes[idx].set_visible(False)

    top_row_centre = top_frac - (cell_h * 0.5) / fig_h
    bot_row_centre = bottom_frac + (cell_h * 0.5) / fig_h
    fig.text(left_frac - 0.055, top_row_centre,
             "Causally-blind", fontsize=8, fontweight="bold",
             color="#4C72B0", va="center", ha="right", rotation=90)
    fig.text(left_frac - 0.055, bot_row_centre,
             "Causal", fontsize=8, fontweight="bold",
             color="#55A868", va="center", ha="right", rotation=90)

    suptitle_y = 1.0 - (top_pad * 0.22) / fig_h
    fig.suptitle(
        "Predicted vs Actual log\u2081\u208a(GHG) \u2014 2\u00d73 Factorial",
        fontsize=9, fontweight="bold",
        y=suptitle_y, va="top",
    )

    legend_y = (bot_pad * 0.50) / fig_h
    _bottom_legend(fig, all_keys, n_cols=2, bbox_y=legend_y)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ===========================================================================
# 8. STANDALONE ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import pickle
    import os

    results_path = _OUT / "results.pkl"
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}.")
        print("Run experiment.py first to generate results.")
    else:
        with open(results_path, "rb") as f:
            saved = pickle.load(f)

        results      = saved["results"]
        preds        = saved["preds"]
        ctx          = saved["ctx"]
        deploy_preds = saved["deploy_preds"]

        validate_ctx(ctx)
        print_comparison(results)
        print_latex_table(results)

        print("\nGenerating figures ...")
        plot_main_comparison(results,            str(_OUT / "fig_comparison.png"))
        plot_mape_by_quintile(preds, ctx,        str(_OUT / "fig_quintile.png"))
        plot_deployment_test(deploy_preds,       str(_OUT / "fig_deployment.png"))
        plot_pred_vs_actual(preds, ctx, results, str(_OUT / "fig_pred_vs_actual.png"))
        print("\nDone.")