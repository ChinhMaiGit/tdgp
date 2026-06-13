"""
diagnostic/m2_shap.py
=====================
SHAP diagnostic for M2 CB XGB — circular feature exploitation analysis.

Purpose
-------
Fits M2 (causally-blind XGBoost on CB_ALL_FEATURES) on the full CB
training dataset and computes SHAP values to quantify which features
drive its near-perfect CV performance (R² > 0.99, MdAPE < 3%).

The central finding this script makes visible: the derived metric
ghg_intensity_kg_co2e_sq_ft (= total_GHG / floor_area) accounts for
the vast majority of M2's predictive signal. Including it as a predictor
of total_GHG is a near-tautology — the model recovers the identity
  GHG ≈ ghg_intensity × floor_area
without being told to. SHAP attribution makes this explicit and
quantified, strengthening the paper's circular-feature-exploitation
argument.

Exact M2 setup (mirrors experiment)
------------------------------------
- Data        : CB pipeline, ~21,714 rows (all observed-target rows)
- Features    : CB_ALL_FEATURES (upstream + spatial + compliance +
                mediators + derived metrics + water use)
- Imputation  : column medians for mediators and derived metrics
                (naive analyst default, same as experiment)
- CV protocol : not repeated here; we train on the full CB dataset
                to maximise SHAP stability (more data = more stable
                attributions). CV metrics are already documented in
                results/docs/experiments/results.txt.
- Seed        : SEED = 20 (matches experiment)

Outputs (saved to results/diagnostic/)
--------------------------------------
fig_shap_beeswarm.png : SHAP beeswarm — per-observation SHAP values
                        coloured by feature value magnitude. Shows
                        which features have the widest impact range.
fig_shap_bar.png      : Mean |SHAP| bar chart, top 25 features, bars
                        coloured by feature category. Red = circular
                        derived, teal = upstream causal.
fig_shap_group.png    : Grouped bar — total mean |SHAP| per feature
                        category. Quantifies the fraction of predictive
                        signal from circular vs causally-valid features.

Run (from drafts/complete/)
---------------------------
  uv run python scripts/diagnostic/m2_shap.py
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import sys
import os
import warnings
import random

warnings.filterwarnings("ignore")

# Add scripts/experiments/ to path — reuse exact same pipeline code so
# preprocessing, feature definitions, and constants are identical to the
# experiment. If the experiment code is updated, this diagnostic picks it up.
# _ROOT resolves to the bundle root (drafts/complete).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "experiments"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
from sklearn.metrics import r2_score

from pipeline import (
    fetch_data, preprocess,
    CB_ALL_FEATURES,
    UPSTREAM_COLS, MEDIATOR_COLS, DERIVED_COLS,
    SPATIAL_COLS, COMPLIANCE_COLS, WATER_COLS,
    build_label_encoded_features,
    APP_TOKEN, SEED, TARGET,
)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR = os.path.join(_ROOT, "results", "diagnostic")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hyperparameters for the full-data M2 fit.
# These replicate the CB XGB regime — shallow-to-moderate trees with light
# regularisation, matching the depth range Optuna selects on this dataset.
# The circular feature exploitation pattern is robust to exact params: any
# reasonable XGBoost configuration will exploit ghg_intensity.
M2_PARAMS = dict(
    learning_rate     = 0.05,
    max_depth         = 5,
    subsample         = 0.8,
    colsample_bytree  = 0.7,
    min_child_weight  = 5,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_estimators      = 800,
    objective         = "reg:squarederror",
    eval_metric       = "rmse",
    tree_method       = "hist",
    random_state      = SEED,
    n_jobs            = 1,         # deterministic accumulation
    verbosity         = 0,
    enable_categorical = False,
)

# Feature group taxonomy — mirrors pipeline.py variable definitions.
# Used to colour-code SHAP plots and compute group-level attribution.
FEATURE_GROUPS = {
    "derived_circular": DERIVED_COLS,     # ghg_intensity, EUIs, etc.
    "mediator":         MEDIATOR_COLS,    # energy use by fuel type
    "upstream":         UPSTREAM_COLS,    # floor area, type, year, etc.
    "spatial":          SPATIAL_COLS,     # ZIP, community area, lat/lon
    "compliance":       COMPLIANCE_COLS,  # reporting status, exempt flag
    "water":            WATER_COLS,       # water_use_kgal
}

# Red for circular, teal for causal, muted tones for the rest.
GROUP_COLORS = {
    "derived_circular": "#e63946",
    "mediator":         "#f4a261",
    "upstream":         "#2a9d8f",
    "spatial":          "#457b9d",
    "compliance":       "#a8dadc",
    "water":            "#8ecae6",
}

GROUP_LABELS = {
    "derived_circular": "Derived / circular  (ghg_intensity, EUIs, …)",
    "mediator":         "Mediators  (energy use by fuel)",
    "upstream":         "Upstream causal  (floor area, type, year, …)",
    "spatial":          "Spatial  (ZIP, community area, lat/lon)",
    "compliance":       "Compliance metadata  (status, exempt flag)",
    "water":            "Water use",
}


def _feature_group(col: str) -> str:
    for group, cols in FEATURE_GROUPS.items():
        if col in cols:
            return group
    return "other"


# ---------------------------------------------------------------------------
# 2. Plotting helpers
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.family":    "sans-serif",
        "font.size":      9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.dpi":     150,
    })


def plot_beeswarm(shap_values: np.ndarray,
                  X_pd: pd.DataFrame,
                  save_path: str) -> None:
    """SHAP beeswarm — shap.summary_plot manages its own figure."""
    _style()
    shap.summary_plot(
        shap_values, X_pd,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    plt.title(
        "M2 CB XGB — SHAP beeswarm (top 20 features)\n"
        "Derived/circular features dominate predictive signal",
        fontsize=12, pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {save_path}")


def plot_bar(shap_df: pd.DataFrame, save_path: str) -> None:
    """Mean |SHAP| bar chart — top 25 features, coloured by group."""
    _style()
    top_n  = min(25, len(shap_df))
    top    = shap_df.head(top_n)
    colors = [GROUP_COLORS.get(g, "#999999") for g in top["group"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top["feature"].values[::-1],
        top["mean_abs_shap"].values[::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Mean |SHAP value|  (log₁₊ GHG scale)")
    ax.set_title(
        f"M2 CB XGB — Top {top_n} features by mean |SHAP|\n"
        "Red = derived/circular (tautological) · teal = upstream causal",
        fontsize=11,
    )

    present_groups = top["group"].unique()
    legend_patches = [
        mpatches.Patch(
            color=GROUP_COLORS.get(g, "#999999"),
            label=GROUP_LABELS.get(g, g),
        )
        for g in GROUP_COLORS if g in present_groups
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
              framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {save_path}")


def plot_group(group_df: pd.DataFrame, save_path: str) -> None:
    """Grouped bar — total mean |SHAP| share per feature category."""
    _style()
    labels = [GROUP_LABELS.get(g, g) for g in group_df["group"]]
    colors = [GROUP_COLORS.get(g, "#999999") for g in group_df["group"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(group_df)), group_df["pct"].values,
        color=colors, edgecolor="white", linewidth=0.4,
    )
    ax.set_ylabel("% of total mean |SHAP|")
    ax.set_title(
        "M2 CB XGB — Predictive signal by feature group\n"
        "Circular derived features account for most of M2's apparent performance",
        fontsize=11,
    )
    ax.set_xticks(range(len(group_df)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)

    for bar, pct in zip(bars, group_df["pct"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {save_path}")


def plot_combined(shap_values: np.ndarray,
                  X_pd: pd.DataFrame,
                  shap_df: pd.DataFrame,
                  group_df: pd.DataFrame,
                  save_path: str) -> None:
    """3-panel combined SHAP figure: (a) group, (b) per-feature bar, (c) beeswarm.

    Vertical single-column stack at near-print width, so fonts remain
    legible when the figure is scaled to text width in the paper. All
    three panels are drawn as vector art into one figure; the beeswarm
    is drawn directly onto its axes (plot_size=None stops summary_plot
    from resizing the figure) rather than embedded as a raster image.
    """
    _style()
    fig = plt.figure(figsize=(8.5, 15))
    gs  = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.0, 1.0], hspace=0.5)
    ax_grp = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_bee = fig.add_subplot(gs[2])

    # ── (a) Group attribution chart ──────────────────────────────────────────
    labels = [GROUP_LABELS.get(g, g) for g in group_df["group"]]
    colors = [GROUP_COLORS.get(g, "#999999") for g in group_df["group"]]
    bars = ax_grp.bar(
        range(len(group_df)), group_df["pct"].values,
        color=colors, edgecolor="white", linewidth=0.4,
    )
    ax_grp.set_ylabel("% of total mean |SHAP|", fontsize=10)
    ax_grp.set_xticks(range(len(group_df)))
    ax_grp.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    for bar, pct in zip(bars, group_df["pct"].values):
        ax_grp.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    ax_grp.spines[["top", "right"]].set_visible(False)
    ax_grp.set_title(
        "(a)  Predictive signal by feature group",
        fontsize=11, fontweight="bold", pad=8, loc="left",
    )

    # ── (b) Per-feature mean |SHAP| bar chart ───────────────────────────────
    top_n      = min(15, len(shap_df))
    top        = shap_df.head(top_n)
    colors_bar = [GROUP_COLORS.get(g, "#999999") for g in top["group"]]
    ax_bar.barh(
        top["feature"].values[::-1],
        top["mean_abs_shap"].values[::-1],
        color=colors_bar[::-1],
        edgecolor="white", linewidth=0.4,
    )
    ax_bar.set_xlabel("Mean |SHAP value|  (log₁₊ GHG scale)", fontsize=10)
    present_groups = top["group"].unique()
    legend_patches = [
        mpatches.Patch(
            color=GROUP_COLORS.get(g, "#999999"),
            label=GROUP_LABELS.get(g, g),
        )
        for g in GROUP_COLORS if g in present_groups
    ]
    ax_bar.legend(handles=legend_patches, fontsize=8.5, loc="lower right",
                  framealpha=0.9)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="y", labelsize=9)
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.set_title(
        f"(b)  Top {top_n} features by mean |SHAP|",
        fontsize=11, fontweight="bold", pad=8, loc="left",
    )

    # ── (c) Beeswarm (vector, drawn onto its own axes) ───────────────────────
    plt.sca(ax_bee)
    shap.summary_plot(
        shap_values, X_pd,
        max_display=12,
        show=False,
        plot_type="dot",
        plot_size=None,
    )
    ax_bee.tick_params(axis="y", labelsize=9)
    ax_bee.tick_params(axis="x", labelsize=9)
    ax_bee.set_xlabel(ax_bee.get_xlabel(), fontsize=10)
    ax_bee.set_title(
        "(c)  SHAP value distribution (beeswarm, top 12 features)",
        fontsize=11, fontweight="bold", pad=8, loc="left",
    )

    fig.suptitle(
        "M2 CB XGBoost SHAP attribution analysis",
        fontsize=12, fontweight="bold", y=0.997,
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {save_path}")


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    random.seed(SEED)
    np.random.seed(SEED)

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 -- FETCHING DATA")
    print("=" * 70)
    df_raw = fetch_data(APP_TOKEN)

    # ── Step 2: Preprocess (CB pipeline only) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 -- PREPROCESSING  (CB pipeline)")
    print("=" * 70)
    ctx      = preprocess(df_raw)
    obs_cb   = ctx["obs_cb"]
    y_log_cb = ctx["y_log_cb"]
    y_obs_cb = ctx["y_obs_cb"]
    n_cb     = ctx["n_cb"]

    # ── Step 3: Build label-encoded feature matrix ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 -- BUILDING FEATURE MATRIX  (CB_ALL_FEATURES)")
    print("=" * 70)
    X_pd, _, _, _ = build_label_encoded_features(obs_cb, CB_ALL_FEATURES)
    feature_names  = list(X_pd.columns)

    print(f"  Training rows : {len(X_pd):,}")
    print(f"  Features      : {len(feature_names)}")
    circular_present = [c for c in DERIVED_COLS if c in feature_names]
    print(f"  Circular derived features present:")
    for c in circular_present:
        print(f"    {c}")

    # ── Step 4: Fit M2 on full CB dataset ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 -- FITTING M2 CB XGB  (full CB dataset, fixed params)")
    print("=" * 70)
    model = xgb.XGBRegressor(**M2_PARAMS)
    model.fit(X_pd, y_log_cb)

    y_pred_log  = model.predict(X_pd)
    y_pred_orig = np.clip(np.expm1(y_pred_log), 0, None)
    r2_train    = r2_score(y_log_cb, y_pred_log)
    mape_train  = float(
        np.median(np.abs((y_obs_cb - y_pred_orig) / (y_obs_cb + 1e-6))) * 100
    )
    print(f"  Train R²(log) : {r2_train:.4f}  "
          f"(high R² confirms circular feature exploitation)")
    print(f"  Train MdAPE   : {mape_train:.2f}%")

    # ── Step 5: SHAP values ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 -- COMPUTING SHAP VALUES")
    print("=" * 70)
    print("  Building TreeExplainer ...")
    explainer   = shap.TreeExplainer(model)
    print("  Computing SHAP values (may take 1–2 minutes) ...")
    shap_values = explainer.shap_values(X_pd)

    # Per-feature attribution table
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap    = mean_abs_shap.sum()
    shap_df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs_shap,
        "pct_of_total":  mean_abs_shap / total_shap * 100,
        "group":         [_feature_group(f) for f in feature_names],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Group-level attribution table
    group_df = (
        shap_df.groupby("group", sort=False)["mean_abs_shap"]
        .sum()
        .reset_index()
        .rename(columns={"mean_abs_shap": "total_shap"})
    )
    group_df["pct"] = group_df["total_shap"] / total_shap * 100
    group_df = group_df.sort_values("total_shap", ascending=False).reset_index(
        drop=True)

    # Print group summary
    print("\n  Feature group attribution (% of total mean |SHAP|):")
    print(f"  {'Group':<22} {'Mean |SHAP|':>12} {'%':>8}")
    print(f"  {'─' * 44}")
    for _, row in group_df.iterrows():
        print(f"  {row['group']:<22} {row['total_shap']:>12.4f} "
              f"{row['pct']:>7.1f}%")

    print(f"\n  Top 15 features:")
    print(f"  {'Feature':<50} {'%':>7}  {'Group'}")
    print(f"  {'─' * 80}")
    for _, row in shap_df.head(15).iterrows():
        print(f"  {row['feature']:<50} {row['pct_of_total']:>6.1f}%  "
              f"{row['group']}")

    # ── Step 6: Generate figures ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 -- GENERATING FIGURES")
    print("=" * 70)

    plot_beeswarm(
        shap_values, X_pd,
        save_path=os.path.join(FIGURES_DIR, "fig_shap_beeswarm.png"),
    )
    plot_bar(
        shap_df,
        save_path=os.path.join(FIGURES_DIR, "fig_shap_bar.png"),
    )
    plot_group(
        group_df,
        save_path=os.path.join(FIGURES_DIR, "fig_shap_group.png"),
    )
    plot_combined(
        shap_values, X_pd, shap_df, group_df,
        save_path=os.path.join(FIGURES_DIR, "fig_shap_combined.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    def _group_pct(name: str) -> float:
        row = group_df.loc[group_df["group"] == name, "pct"]
        return float(row.values[0]) if len(row) else 0.0

    circ_pct = _group_pct("derived_circular")
    med_pct  = _group_pct("mediator")
    ups_pct  = _group_pct("upstream")
    top_feat = shap_df.iloc[0]

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Train R²(log)                    : {r2_train:.4f}")
    print(f"  Train MdAPE                      : {mape_train:.2f}%")
    print(f"  Top feature                      : {top_feat['feature']} "
          f"({top_feat['pct_of_total']:.1f}%)")
    print(f"  Derived/circular group share     : {circ_pct:.1f}%")
    print(f"  Mediator group share             : {med_pct:.1f}%")
    print(f"  Upstream causal group share      : {ups_pct:.1f}%")
    print(f"\n  Interpretation")
    print(f"  {'─' * 60}")
    print(f"  M2's near-perfect CV metrics stem from derived features in")
    print(f"  CB_ALL_FEATURES — particularly ghg_intensity_kg_co2e_sq_ft")
    print(f"  (= total_GHG / floor_area). Including this as a predictor")
    print(f"  of total_GHG is a near-tautology: the model recovers the")
    print(f"  identity GHG ≈ ghg_intensity × floor_area without being")
    print(f"  told to. The {circ_pct:.0f}% SHAP share from derived/circular")
    print(f"  features vs {ups_pct:.0f}% from upstream causal features")
    print(f"  confirms that M2's apparent performance does not reflect")
    print(f"  genuine causal understanding of the GHG DGP.")
    print(f"\n  Figures saved to: {FIGURES_DIR}")
    print("=" * 70)
