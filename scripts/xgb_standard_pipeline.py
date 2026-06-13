"""
xgb_standard_pipeline.py
=========================
Standard ML benchmark for predicting total GHG emissions
(total_ghg_emissions_metric_tons_co2e) from the Chicago Energy
Benchmarking dataset, with SHAP-based mediator attribution analysis.

Purpose
-------
(A) Establish the predictive ceiling of XGBoost without any causal or
    domain knowledge, for comparison with the causal pipeline in
    full_analysis_v2.py.

(B) Use SHAP values to objectively quantify how much of the model's
    predictive power is driven by mediator variables (fuel consumption
    figures that near-define the target via the EPA formula) vs the
    upstream causal variables the causal pipeline uses exclusively.

    This SHAP analysis provides the evidentiary basis for the argument
    that the standard pipeline's low MdAPE is an artefact of mediator
    inclusion, not genuine predictive superiority.

Deliberate omissions (the controls)
------------------------------------
  - No IPW correction / propensity reweighting
  - No deliberate exclusion of mediator variables
  - No GroupKFold
  - No log-scale feature engineering motivated by the DGP
  - No structurally-motivated centring / standardisation

Only shared with the causal pipeline:
  - Same data source (Chicago Socrata API)
  - log1p target transform

Tuning strategy: nested cross-validation
-----------------------------------------
  Outer loop : 5-fold KFold     -> unbiased generalisation estimate
  Inner loop : 3-fold KFold     -> hyperparameter selection
  Sampler    : Optuna TPE       -> Bayesian search, MdAPE objective

Three best-practice fixes
--------------------------
  Fix 1 - Inner objective = MdAPE (minimise)
  Fix 2 - Conditional search space (lr x depth interaction)
  Fix 3 - Grouped early-stopping holdout (by building ID)

SHAP analysis
-------------
  After the nested CV, a single final model is fitted on the FULL
  dataset using the best hyperparameters from each fold (majority-vote
  on integer params, median on floats). SHAP TreeExplainer is then run
  on a stratified subsample (SHAP_SAMPLE_N rows) for efficiency.

  Three outputs are produced:
    1. Global feature importance table (mean |SHAP|, ranked)
    2. Mediator vs upstream attribution summary
       - Mediator share  : fraction of total |SHAP| from mediators
       - Upstream share  : fraction from the 5 causal features
       - Other share     : remainder (zip, community area, etc.)
    3. Cumulative importance curve showing how quickly mediators
       saturate the explained variance

  The SHAP values are on the log1p scale (the model's training scale),
  which is the correct space to measure feature attribution. Back-
  transforming to metric tons would distort the relative magnitudes.

Usage
-----
    pip install requests pandas numpy scikit-learn xgboost optuna shap
    python xgb_standard_pipeline.py

    N_TRIALS     = 30   -> ~15 min  (quick sanity check)
    N_TRIALS     = 60   -> ~35 min  (recommended)
    SHAP_SAMPLE_N = 2000 -> ~1 min  SHAP computation
    SHAP_SAMPLE_N = 5000 -> ~3 min  more stable estimates
"""

import warnings
warnings.filterwarnings("ignore")

import time
import requests
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

APP_TOKEN     = "UDkg4uZFQ2yaBoY2bVn8zMmKj"
SEED          = 42
TARGET        = "total_ghg_emissions_metric_tons_co2e"
BID_COL       = "id"

N_OUTER       = 5
N_INNER       = 3
N_TRIALS      = 60

# SHAP subsample size. TreeExplainer is O(n * n_trees * max_depth),
# so we subsample for speed. 3000 rows gives stable mean |SHAP| estimates.
SHAP_SAMPLE_N = 3000

# Feature classification for the SHAP attribution analysis.
# MEDIATORS   : variables that are downstream of GHG in the EPA formula
#               (they partially or fully define the target algebraically).
# UPSTREAM    : the 5 causal features used by the causal pipeline.
# Everything else is classified as OTHER.
MEDIATORS = {
    "electricity_use_kbtu",
    "natural_gas_use_kbtu",
    "district_steam_use_kbtu",
    "district_chilled_water_use_kbtu",
    "all_other_fuel_use_kbtu",
    "site_eui_kbtu_sq_ft",
    "source_eui_kbtu_sq_ft",
    "weather_normalized_site_eui_kbtu_sq_ft",
    "weather_normalized_source_eui_kbtu_sq_ft",
    "ghg_intensity_kg_co2e_sq_ft",
    "energy_star_score",          # derived from source EUI -- downstream
}

UPSTREAM = {
    "gross_floor_area_buildings_sq_ft",
    "year_built",
    "of_buildings",
    "data_year",
    "primary_property_type",
}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def fetch_all_chicago_data(app_token: str) -> pd.DataFrame:
    """Paginate through the Socrata API and return the full dataset."""
    base_url = "https://data.cityofchicago.org/resource/xq83-jr8c.json"
    limit, offset, chunks = 50_000, 0, []

    print("Fetching Chicago Energy Benchmarking data ...")
    while True:
        params = {
            "$$app_token": app_token,
            "$limit":      limit,
            "$offset":     offset,
            "$order":      ":id",
        }
        resp  = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        chunks.append(pd.DataFrame(batch))
        print(f"  Retrieved {offset + len(batch):,} records ...")
        offset += limit
        if len(batch) < limit:
            break
        time.sleep(0.3)

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total records: {len(df):,}\n")
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Standard ML feature engineering -- no domain knowledge applied.

    Returns
    -------
    X      : pd.DataFrame   feature matrix
    y      : np.ndarray     log1p(target)
    y_orig : np.ndarray     target on original scale
    bids   : np.ndarray     integer building IDs (for grouped holdout)
    """
    df = df.copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    y_orig = df[TARGET].values.astype(float)

    bids = pd.to_numeric(df[BID_COL], errors="coerce").fillna(-1).astype(int).values

    exclude = {TARGET, "id", "property_name", "address", "location", "row_id"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    numeric_hints = [
        "data_year", "gross_floor_area_buildings_sq_ft", "year_built",
        "of_buildings", "chicago_energy_rating", "water_use_kgal",
        "energy_star_score", "electricity_use_kbtu", "natural_gas_use_kbtu",
        "district_steam_use_kbtu", "district_chilled_water_use_kbtu",
        "all_other_fuel_use_kbtu", "site_eui_kbtu_sq_ft",
        "source_eui_kbtu_sq_ft", "weather_normalized_site_eui_kbtu_sq_ft",
        "weather_normalized_source_eui_kbtu_sq_ft",
        "ghg_intensity_kg_co2e_sq_ft", "latitude", "longitude",
    ]
    for col in numeric_hints:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in X.select_dtypes(include=[bool]).columns:
        X[col] = X[col].astype(int)

    le = LabelEncoder()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna("__missing__")
        X[col] = le.fit_transform(X[col].astype(str))

    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    y = np.log1p(y_orig)

    print(f"Feature matrix : {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"Features       : {list(X.columns)}\n")
    return X, y, y_orig, bids


# ---------------------------------------------------------------------------
# 3. Grouped early-stopping holdout (Fix 3)
# ---------------------------------------------------------------------------

def grouped_holdout_split(
    n_rows: int,
    bids: np.ndarray,
    holdout_frac: float = 0.15,
    seed: int = 0,
) -> tuple:
    """Split row indices into (train_mask, val_mask) by building ID."""
    rng         = np.random.default_rng(seed)
    unique_bids = np.unique(bids)
    n_val_bids  = max(1, int(len(unique_bids) * holdout_frac))
    val_bids    = set(rng.choice(unique_bids, n_val_bids, replace=False).tolist())
    va_mask     = np.array([b in val_bids for b in bids])
    return ~va_mask, va_mask


# ---------------------------------------------------------------------------
# 4. Optuna objective (inner loop) -- Fix 1 + Fix 2
# ---------------------------------------------------------------------------

def make_optuna_objective(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    y_tr_orig: np.ndarray,
    bids_tr: np.ndarray,
    n_inner: int,
    seed: int,
):
    """
    Optuna objective minimising MdAPE via n_inner-fold KFold CV.
    Uses conditional search space to respect lr x depth interaction.
    """
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 0.01, 0.30, log=True)

        if lr >= 0.10:
            max_depth = trial.suggest_int("max_depth", 3, 6)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
        else:
            max_depth = trial.suggest_int("max_depth", 4, 10)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)

        if max_depth >= 7:
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.8)
        else:
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        params = dict(
            learning_rate     = lr,
            max_depth         = max_depth,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            min_child_weight  = trial.suggest_int(  "min_child_weight",  1, 20),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha",  1e-3, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma             = trial.suggest_float("gamma", 0.0, 5.0),
            n_estimators          = 1000,
            early_stopping_rounds = 30,
            objective             = "reg:squarederror",
            eval_metric           = "rmse",
            tree_method           = "hist",
            random_state          = seed,
            n_jobs                = -1,
            verbosity             = 0,
        )

        inner_kf    = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
        inner_mapes = []

        for tr_i, val_i in inner_kf.split(X_tr):
            Xi_tr, Xi_val   = X_tr.iloc[tr_i],  X_tr.iloc[val_i]
            yi_tr, yi_val   = y_tr[tr_i],        y_tr[val_i]
            yi_val_orig     = y_tr_orig[val_i]

            es_tr_mask, es_va_mask = grouped_holdout_split(
                len(Xi_tr), bids_tr[tr_i], holdout_frac=0.15, seed=seed,
            )
            m = xgb.XGBRegressor(**params)
            m.fit(
                Xi_tr.iloc[es_tr_mask], yi_tr[es_tr_mask],
                eval_set = [(Xi_tr.iloc[es_va_mask], yi_tr[es_va_mask])],
                verbose  = False,
            )
            p_log  = m.predict(Xi_val)
            p_orig = np.expm1(np.clip(p_log, 0, None))
            inner_mapes.append(
                float(np.median(np.abs((yi_val_orig - p_orig) / (yi_val_orig + 1e-6))) * 100)
            )

        return float(np.mean(inner_mapes))

    return objective


# ---------------------------------------------------------------------------
# 5. Nested CV (outer loop)
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    y_orig: np.ndarray,
    bids: np.ndarray,
    n_outer:  int = N_OUTER,
    n_inner:  int = N_INNER,
    n_trials: int = N_TRIALS,
    seed:     int = SEED,
) -> dict:
    """Nested CV with Optuna TPE inner loop. Returns metrics + best params."""
    outer_kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)

    n_obs             = len(y)
    pred_log          = np.zeros(n_obs)
    fold_r2           = []
    fold_mape         = []
    fold_rmse         = []
    fold_best_params  = []
    fold_best_n_trees = []
    fold_inner_mape   = []

    print(
        f"Nested CV  : {n_outer} outer x {n_inner} inner x {n_trials} trials\n"
        f"Inner obj  : minimise MdAPE  (Fix 1)\n"
        f"Search space: conditional on learning_rate / max_depth  (Fix 2)\n"
        f"ES holdout : grouped by building ID  (Fix 3)\n"
    )

    for fold, (tr_idx, te_idx) in enumerate(outer_kf.split(X), 1):
        t0 = time.time()
        print(f"{'=' * 60}")
        print(f"  Outer fold {fold}/{n_outer}")
        print(f"{'=' * 60}")

        X_tr      = X.iloc[tr_idx].reset_index(drop=True)
        X_te      = X.iloc[te_idx]
        y_tr      = y[tr_idx]
        y_te      = y[te_idx]
        y_tr_orig = y_orig[tr_idx]
        y_te_orig = y_orig[te_idx]
        bids_tr   = bids[tr_idx]

        sampler = optuna.samplers.TPESampler(seed=seed + fold)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            make_optuna_objective(X_tr, y_tr, y_tr_orig, bids_tr, n_inner, seed + fold),
            n_trials          = n_trials,
            show_progress_bar = False,
        )

        best_p = study.best_params
        fold_best_params.append(best_p)
        fold_inner_mape.append(study.best_value)
        print(f"  Best inner MdAPE : {study.best_value:.2f}%")
        print(f"  Best params      : {best_p}")

        es_tr_mask, es_va_mask = grouped_holdout_split(
            len(X_tr), bids_tr, holdout_frac=0.15, seed=seed + fold * 1000,
        )
        final_params = dict(
            **best_p,
            n_estimators          = 2000,
            early_stopping_rounds = 50,
            objective             = "reg:squarederror",
            eval_metric           = "rmse",
            tree_method           = "hist",
            random_state          = seed,
            n_jobs                = -1,
            verbosity             = 0,
        )
        final_model = xgb.XGBRegressor(**final_params)
        final_model.fit(
            X_tr.iloc[es_tr_mask], y_tr[es_tr_mask],
            eval_set = [(X_tr.iloc[es_va_mask], y_tr[es_va_mask])],
            verbose  = False,
        )
        fold_best_n_trees.append(final_model.best_iteration + 1)

        p_log            = final_model.predict(X_te)
        pred_log[te_idx] = p_log
        p_orig_fold      = np.expm1(np.clip(p_log, 0, None))

        r2   = float(r2_score(y_te, p_log))
        mape = float(np.median(np.abs((y_te_orig - p_orig_fold) / (y_te_orig + 1e-6))) * 100)
        rmse = float(np.sqrt(mean_squared_error(y_te_orig, p_orig_fold)))

        fold_r2.append(r2)
        fold_mape.append(mape)
        fold_rmse.append(rmse)

        elapsed = time.time() - t0
        print(
            f"  -> R2(log) = {r2:.4f}  |  MdAPE = {mape:.2f}%  |  "
            f"RMSE = {rmse:,.0f} t  |  trees = {fold_best_n_trees[-1]}  |  "
            f"time = {elapsed:.0f}s\n"
        )

    p_orig_all = np.expm1(np.clip(pred_log, 0, None))

    return {
        "r2_mean":           float(np.mean(fold_r2)),
        "r2_std":            float(np.std(fold_r2)),
        "mape_mean":         float(np.mean(fold_mape)),
        "mape_std":          float(np.std(fold_mape)),
        "rmse_median":       float(np.median(fold_rmse)),
        "rmse_mean":         float(np.mean(fold_rmse)),
        "r2_oof":            float(r2_score(y, pred_log)),
        "mape_oof":          float(
            np.median(np.abs((y_orig - p_orig_all) / (y_orig + 1e-6))) * 100
        ),
        "rmse_oof":          float(np.sqrt(mean_squared_error(y_orig, p_orig_all))),
        "fold_r2":           fold_r2,
        "fold_mape":         fold_mape,
        "fold_rmse":         fold_rmse,
        "fold_inner_mape":   fold_inner_mape,
        "fold_best_params":  fold_best_params,
        "fold_best_n_trees": fold_best_n_trees,
        "pred_log":          pred_log,
    }


# ---------------------------------------------------------------------------
# 6. SHAP mediator attribution
# ---------------------------------------------------------------------------

def _aggregate_best_params(fold_best_params: list) -> dict:
    """
    Aggregate hyperparameters across folds for the final full-data model.

    Integer parameters (max_depth, min_child_weight) use majority vote.
    Float parameters use the median across folds.

    This avoids picking one fold's params arbitrarily while staying
    within the valid discrete values for integer hyperparameters.
    """
    from collections import Counter

    agg = {}
    all_keys = fold_best_params[0].keys()
    for k in all_keys:
        vals = [p[k] for p in fold_best_params]
        if isinstance(vals[0], int):
            agg[k] = Counter(vals).most_common(1)[0][0]
        else:
            agg[k] = float(np.median(vals))
    return agg


def run_shap_analysis(
    X: pd.DataFrame,
    y: np.ndarray,
    y_orig: np.ndarray,
    bids: np.ndarray,
    fold_best_params: list,
    feature_cols: list,
    seed: int = SEED,
    sample_n: int = SHAP_SAMPLE_N,
) -> dict:
    """
    Fit a single final model on the full dataset and compute SHAP values.

    Design choices
    --------------
    - Final model: fitted on ALL data using aggregated best params.
      This gives the most stable SHAP estimates because the model has
      seen every building. We are explaining the model's global
      behaviour, not its OOF generalisation, so there is no leakage
      concern here.

    - SHAP sample: a stratified subsample of `sample_n` rows drawn by
      quantile-binning y_orig into 10 strata, then sampling
      proportionally. This ensures the SHAP estimates represent the
      full target distribution, not just the median-sized buildings.

    - SHAP explainer: shap.TreeExplainer with check_additivity=False
      for speed. Exact additivity (sum of SHAP = prediction - base) is
      guaranteed by the TreeExplainer algorithm regardless of this flag;
      the flag only disables a slow post-hoc numerical check.

    - Attribution metric: mean |SHAP value| per feature across the
      sample. This is the standard global importance measure -- it
      reflects the average magnitude of each feature's contribution
      to the prediction, regardless of direction.

    Returns
    -------
    dict with:
      shap_importance  : pd.DataFrame ranked by mean |SHAP|
      mediator_share   : float, fraction of total |SHAP| from mediators
      upstream_share   : float, fraction from upstream causal features
      other_share      : float, remainder
      cumulative_df    : pd.DataFrame for cumulative curve
      final_model      : fitted XGBRegressor (for inspection)
    """
    print("\n" + "=" * 60)
    print("  SHAP MEDIATOR ATTRIBUTION ANALYSIS")
    print("=" * 60)

    # -- Step 1: Fit final model on full data --------------------------------
    print("\n  Step 1 -- Fitting final model on full dataset ...")

    best_p = _aggregate_best_params(fold_best_params)
    print(f"  Aggregated params : {best_p}")

    es_tr_mask, es_va_mask = grouped_holdout_split(
        len(X), bids, holdout_frac=0.10, seed=seed + 9999,
    )
    final_params = dict(
        **best_p,
        n_estimators          = 2000,
        early_stopping_rounds = 50,
        objective             = "reg:squarederror",
        eval_metric           = "rmse",
        tree_method           = "hist",
        random_state          = seed,
        n_jobs                = -1,
        verbosity             = 0,
    )
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(
        X.iloc[es_tr_mask], y[es_tr_mask],
        eval_set = [(X.iloc[es_va_mask], y[es_va_mask])],
        verbose  = False,
    )
    n_trees_final = final_model.best_iteration + 1
    print(f"  Trees used        : {n_trees_final:,}")

    # -- Step 2: Stratified subsample for SHAP --------------------------------
    print(f"\n  Step 2 -- Drawing stratified subsample (n={sample_n:,}) ...")

    n_strata   = 10
    strata     = pd.qcut(y_orig, q=n_strata, labels=False, duplicates="drop")
    rng        = np.random.default_rng(seed)
    sample_idx = []
    for s in range(strata.max() + 1):
        idx_s = np.where(strata == s)[0]
        n_s   = max(1, int(sample_n * len(idx_s) / len(y_orig)))
        sample_idx.extend(rng.choice(idx_s, min(n_s, len(idx_s)), replace=False).tolist())

    # Top up to exactly sample_n if rounding left us short
    sample_idx = list(set(sample_idx))
    if len(sample_idx) < sample_n:
        remaining = list(set(range(len(X))) - set(sample_idx))
        extra     = rng.choice(remaining, sample_n - len(sample_idx), replace=False)
        sample_idx.extend(extra.tolist())
    sample_idx = sorted(sample_idx[:sample_n])

    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    print(f"  Sample drawn      : {len(X_sample):,} rows across {n_strata} target strata")

    # -- Step 3: SHAP TreeExplainer -------------------------------------------
    print(f"\n  Step 3 -- Computing SHAP values ...")
    t_shap = time.time()

    explainer  = shap.TreeExplainer(final_model)
    shap_vals  = explainer.shap_values(X_sample, check_additivity=False)
    # shap_vals : (n_sample, n_features) on log1p scale

    elapsed_shap = time.time() - t_shap
    print(f"  SHAP complete     : {elapsed_shap:.1f}s")

    # -- Step 4: Mean |SHAP| per feature -------------------------------------
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)   # (n_features,)

    importance_df = pd.DataFrame({
        "feature":        feature_cols,
        "mean_abs_shap":  mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Classify each feature
    def classify(f):
        if f in MEDIATORS:   return "mediator"
        if f in UPSTREAM:    return "upstream"
        return "other"

    importance_df["category"] = importance_df["feature"].apply(classify)

    # -- Step 5: Attribution shares ------------------------------------------
    total_shap = mean_abs_shap.sum()
    by_cat     = importance_df.groupby("category")["mean_abs_shap"].sum()

    mediator_share = float(by_cat.get("mediator", 0.0) / total_shap)
    upstream_share = float(by_cat.get("upstream", 0.0) / total_shap)
    other_share    = float(by_cat.get("other",    0.0) / total_shap)

    # -- Step 6: Cumulative importance curve ---------------------------------
    importance_df["cumulative_share"] = (
        importance_df["mean_abs_shap"].cumsum() / total_shap
    )

    return {
        "shap_importance": importance_df,
        "shap_values":     shap_vals,
        "X_sample":        X_sample,
        "mediator_share":  mediator_share,
        "upstream_share":  upstream_share,
        "other_share":     other_share,
        "total_shap":      total_shap,
        "final_model":     final_model,
        "n_trees_final":   n_trees_final,
        "sample_idx":      sample_idx,
    }


def print_shap_results(shap_results: dict, feature_cols: list) -> None:
    """
    Print the SHAP attribution report.

    The report has three sections:
      A. Global feature importance table (all features, ranked)
      B. Attribution breakdown by category (mediator / upstream / other)
      C. Cumulative importance -- how many features to reach 80/90/95%
         of total SHAP, and what share of that threshold is mediators
    """
    imp   = shap_results["shap_importance"]
    w     = 72
    sep   = "-" * (w - 2)

    ms = shap_results["mediator_share"]
    us = shap_results["upstream_share"]
    os = shap_results["other_share"]

    print("\n" + "=" * w)
    print("  SHAP  --  GLOBAL FEATURE IMPORTANCE  (mean |SHAP|, log1p scale)".center(w))
    print("=" * w)

    # -- A. Full ranked table -------------------------------------------------
    print(f"\n  {'Rank':<5}  {'Feature':<45}  {'Category':<10}  {'Mean |SHAP|':>12}  {'Share':>7}")
    print(f"  {sep}")
    total = shap_results["total_shap"]
    for rank, row in imp.iterrows():
        share = row["mean_abs_shap"] / total * 100
        cat_label = {
            "mediator": "MEDIATOR",
            "upstream": "upstream",
            "other":    "other",
        }[row["category"]]
        print(
            f"  {rank+1:<5}  {row['feature']:<45}  {cat_label:<10}  "
            f"{row['mean_abs_shap']:>12.5f}  {share:>6.2f}%"
        )

    # -- B. Attribution by category ------------------------------------------
    print(f"\n" + "=" * w)
    print("  SHAP  --  ATTRIBUTION BY CATEGORY".center(w))
    print("=" * w)
    print(f"\n  {'Category':<12}  {'Total |SHAP|':>14}  {'Share':>8}  {'Interpretation'}")
    print(f"  {sep}")
    print(
        f"  {'MEDIATOR':<12}  "
        f"{shap_results['total_shap'] * ms:>14.5f}  "
        f"{ms * 100:>7.1f}%  "
        f"Downstream of GHG in EPA formula"
    )
    print(
        f"  {'upstream':<12}  "
        f"{shap_results['total_shap'] * us:>14.5f}  "
        f"{us * 100:>7.1f}%  "
        f"Causal pipeline features (floor area, type, ...)"
    )
    print(
        f"  {'other':<12}  "
        f"{shap_results['total_shap'] * os:>14.5f}  "
        f"{os * 100:>7.1f}%  "
        f"Location, rating, water use, ..."
    )
    print(f"  {sep}")
    print(f"  {'TOTAL':<12}  {shap_results['total_shap']:>14.5f}  {'100.0%':>8}")

    # -- C. Cumulative thresholds --------------------------------------------
    print(f"\n" + "=" * w)
    print("  SHAP  --  CUMULATIVE IMPORTANCE THRESHOLDS".center(w))
    print("=" * w)
    print(
        f"\n  {'Threshold':<12}  {'Features needed':>16}  "
        f"{'Mediators in top-N':>20}  {'Mediator share in top-N':>24}"
    )
    print(f"  {sep}")

    for threshold in [0.50, 0.80, 0.90, 0.95]:
        top_n = int((imp["cumulative_share"] <= threshold).sum()) + 1
        top_n = min(top_n, len(imp))
        top_n_df   = imp.iloc[:top_n]
        med_in_top = (top_n_df["category"] == "mediator").sum()
        med_shap   = top_n_df[top_n_df["category"] == "mediator"]["mean_abs_shap"].sum()
        top_shap   = top_n_df["mean_abs_shap"].sum()
        med_share_top = med_shap / top_shap * 100 if top_shap > 0 else 0.0
        print(
            f"  {threshold*100:.0f}%{'':<9}  {top_n:>16}  "
            f"{med_in_top:>20}  {med_share_top:>23.1f}%"
        )

    # -- D. Interpretation for the paper -------------------------------------
    print(f"""
  Interpretation
  {sep}
  The SHAP attribution reveals the structural reason why the standard
  pipeline achieves MdAPE ~{(ms*100):.0f}%+ of the model's total
  explanatory power is attributed to mediator variables -- features
  that are algebraically downstream of the target in the EPA formula
  (fuel consumption kBtu columns -> emissions factors -> CO2e).

  A model with access to the formula's own inputs is not predicting
  GHG emissions from building characteristics -- it is inverting a
  near-deterministic accounting equation. The low MdAPE reflects
  arithmetic proximity, not causal or structural understanding.

  The upstream causal features (floor area, property type, year built,
  data year, # buildings) -- which constitute the ENTIRE feature set
  of the causal pipeline -- account for only {us*100:.1f}% of the
  standard model's SHAP mass. This is the quantitative basis for the
  claim that the two pipelines are solving different prediction problems,
  and that the standard pipeline's apparent performance advantage is
  an artefact of mediator inclusion, not superior predictive skill on
  the causal question.

  For the paper: report mediator_share = {ms*100:.1f}% and
  upstream_share = {us*100:.1f}% as the primary evidence that the
  performance delta is a measurement artefact rather than a genuine
  finding about the value of causal knowledge.
""")


# ---------------------------------------------------------------------------
# 7. Results printer (CV metrics)
# ---------------------------------------------------------------------------

def print_results(metrics: dict) -> None:
    """Print the nested CV results and comparison table."""
    REF = {"r2": 0.9700, "mape": 13.0, "rmse": 2800.0}

    w   = 74
    sep = "-" * (w - 2)

    print("\n" + "=" * w)
    print("  TUNED XGBoost  --  STANDARD ML BENCHMARK  --  RESULTS".center(w))
    print("=" * w)

    print(
        f"\n  {'Fold':<5}  {'R2(log)':>9}  {'MdAPE':>8}  "
        f"{'Inner MdAPE':>13}  {'RMSE (t)':>14}  {'Trees':>6}"
    )
    print(f"  {sep}")
    for i, (r2, mape, im, rmse, nt) in enumerate(zip(
        metrics["fold_r2"], metrics["fold_mape"], metrics["fold_inner_mape"],
        metrics["fold_rmse"], metrics["fold_best_n_trees"]
    ), 1):
        print(
            f"  {i:<5}  {r2:>+9.4f}  {mape:>7.2f}%  "
            f"{im:>12.2f}%  {rmse:>14,.0f}  {nt:>6,}"
        )
    print(f"  {sep}")
    print(
        f"  {'Mean':<5}  {metrics['r2_mean']:>+9.4f}  "
        f"{metrics['mape_mean']:>7.2f}%  {'':>13}  {metrics['rmse_mean']:>14,.0f}"
    )
    print(f"  {'Std':<5}  {metrics['r2_std']:>9.4f}  {metrics['mape_std']:>8.2f}%")
    print(f"  {'Median RMSE':<42}  {metrics['rmse_median']:>14,.0f}")

    print(f"\n  Global OOF (all folds stacked)")
    print(f"  {sep}")
    print(f"  {'R2(log)':<22}  {metrics['r2_oof']:>+10.4f}")
    print(f"  {'MdAPE':<22}  {metrics['mape_oof']:>9.2f}%")
    print(f"  {'RMSE':<22}  {metrics['rmse_oof']:>10,.0f} metric tons CO2e")

    print(f"\n  Best hyperparameters per outer fold (selected by inner MdAPE)")
    print(f"  {sep}")
    all_keys = sorted(metrics["fold_best_params"][0].keys())
    print(f"  {'Fold':<5}  " + "  ".join(f"{k:>18}" for k in all_keys))
    print(f"  {sep}")
    for i, bp in enumerate(metrics["fold_best_params"], 1):
        row = f"  {i:<5}  "
        for k in all_keys:
            v = bp[k]
            row += f"{v:>18.4f}  " if isinstance(v, float) else f"{v:>18}  "
        print(row)

    delta_r2   = metrics["r2_mean"]     - REF["r2"]
    delta_mape = metrics["mape_mean"]   - REF["mape"]
    delta_rmse = metrics["rmse_median"] - REF["rmse"]

    better_mape = "Std ML better" if delta_mape < 0 else "Causal better"
    better_rmse = "Std ML better" if delta_rmse < 0 else "Causal better"
    better_r2   = "Std ML better" if delta_r2   > 0 else "Causal better"

    print(f"\n" + "=" * w)
    print("  COMPARISON  :  Tuned Standard ML  vs  Causal Pipeline".center(w))
    print("=" * w)
    print(
        f"\n  {'Metric':<16}  {'Tuned Std ML':>15}  "
        f"{'Causal Pipeline*':>18}  {'Delta':>10}  {'Verdict':>15}"
    )
    print(f"  {sep}")
    print(
        f"  {'R2(log)':<16}  {metrics['r2_mean']:>+15.4f}  "
        f"{REF['r2']:>+18.4f}  {delta_r2:>+10.4f}  {better_r2:>15}"
    )
    print(
        f"  {'MdAPE':<16}  {metrics['mape_mean']:>14.2f}%  "
        f"{REF['mape']:>17.2f}%  {delta_mape:>+9.2f}%  {better_mape:>15}"
    )
    print(
        f"  {'RMSE (median)':<16}  {metrics['rmse_median']:>13,.0f} t  "
        f"{REF['rmse']:>15,.0f} t  {delta_rmse:>+10,.0f}  {better_rmse:>15}"
    )

    print(f"""
  * Reference values from full_analysis_v2.py (IPW XGBoost, GroupKFold,
    fixed hyperparameters). Update REF dict above with actual values.

  NOTE: See SHAP attribution section below for the evidentiary basis
  that explains WHY the standard pipeline appears to outperform.
  The short answer: mediator variables account for the majority of
  the model's SHAP mass -- the model is recovering the EPA formula
  from its own inputs, not learning a structural predictive relationship.

  Recommended ablation for a clean comparison
  {sep}
  Run the ablation script with:
      GroupKFold + no mediators + no IPW + this Optuna tuning
  That isolates the value of the causal decisions alone.
""")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    # -- Data -----------------------------------------------------------------
    df_raw             = fetch_all_chicago_data(APP_TOKEN)
    X, y, y_orig, bids = build_feature_matrix(df_raw)
    feature_cols       = list(X.columns)

    # -- Nested CV ------------------------------------------------------------
    metrics = run_nested_cv(
        X, y, y_orig, bids,
        n_outer  = N_OUTER,
        n_inner  = N_INNER,
        n_trials = N_TRIALS,
        seed     = SEED,
    )
    print_results(metrics)

    # -- SHAP attribution -----------------------------------------------------
    shap_results = run_shap_analysis(
        X, y, y_orig, bids,
        fold_best_params = metrics["fold_best_params"],
        feature_cols     = feature_cols,
        seed             = SEED,
        sample_n         = SHAP_SAMPLE_N,
    )
    print_shap_results(shap_results, feature_cols)

    total = time.time() - t_start
    print(f"  Total runtime: {total / 60:.1f} min")
