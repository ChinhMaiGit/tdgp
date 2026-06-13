"""
models.py
=========
All six model runners for the 2x3 TDGP factorial experiment.

Random seed policy
------------------
Full reproducibility requires seeding at multiple levels:

  1. numpy global state  : np.random.seed(SEED) before PyMC sampling
                           controls jitter+adapt_diag initialisation
  2. PyMC chain seeds    : list [SEED, SEED+1, SEED+2, SEED+3] — one
                           per chain — ensures each chain starts from
                           a deterministic position
  3. XGBoost random_state: SEED passed to every XGBRegressor instance
  4. XGBoost n_jobs      : full-data refits use n_jobs=1 for deterministic
                           floating-point accumulation; CV fold models
                           use n_jobs=-1 for speed (metrics only, not
                           the deployed model)
  5. Optuna TPE sampler  : seed=SEED+k per fold, where k is the fold index
  6. Holdout splits      : np.random.default_rng(seed) with explicit seed
                           per call — fully deterministic

Causally-blind pipeline (M1-M3)
---------------------------------
M1  CB OLS    : OLS on CB_ALL_FEATURES, KFold, no IPW
                (~21,714 rows — all rows with observed target)
M2  CB XGB    : Tuned XGBoost on CB_ALL_FEATURES, KFold, no IPW
M3  CB Bayes  : Hierarchical Bayes on upstream + mediator features,
                PSIS-LOO, no IPW. The Bayesian parametric structure
                uses upstream structural features and mediator slopes;
                derived metrics, spatial and compliance columns from
                CB_ALL_FEATURES are not given explicit slope parameters
                in the generative model.

Causal pipeline (M4-M6)
--------------------------
M4  Causal OLS   : IPW WLS on upstream features, GroupKFold
                   (~21,406 rows — compliant rows only)
M5  Causal XGB   : IPW tuned XGBoost on upstream features, GroupKFold
M6  Causal Bayes : IPW hierarchical Bayes on upstream features, PSIS-LOO

Deployment test
---------------
All six models applied to 5,510 non-compliant building-year records.
Log-scale predictions clipped to [-20, 15] before exponentiation.

CB deployment imputation
------------------------
Non-compliant buildings have no mediators or derived metrics (they
never submitted a report). CB models receive median-imputed values from
ctx["cb_medians"] — the same imputation used during training.
Causal models receive structural zeros for mediators (tested DAG).

Row counts
----------
CB pipeline    : ~21,714 rows  (all rows with observed target)
Causal pipeline: ~21,406 rows  (compliant rows only)
The two pipelines have DIFFERENT row counts by design.
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import os
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import polars as pl
import optuna
import pymc as pm
import arviz as az

from numpy.linalg import lstsq
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb

from pipeline import (
    SEED, TARGET, BID_COL, RAW_CAT, RAW_NUMERIC,
    UPSTREAM_COLS, MEDIATOR_COLS, MEDIATOR_FEATURE_COLS,
    CB_ALL_FEATURES,
    N_OUTER, N_INNER, N_TRIALS, Z89,
    MODEL_META, COLOURS, GROUPS, CB_MODELS, CAUSAL_MODELS,
    fold_metrics, aggregate_folds,
    grouped_holdout, random_holdout,
    build_ols_design, build_label_encoded_features,
    _flatten_az_result, _loo_elpd, _loo_se, _pareto_k,
    preprocess,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Log-scale clip bounds ─────────────────────────────────────────────────────
LOG_CLIP_LO = -20.0   # exp(-20) ≈ 2e-9 metric tons  (effectively zero)
LOG_CLIP_HI =  15.0   # exp(15)  ≈ 3.3M metric tons  (physical ceiling)

# ── PyMC chain seeds — one per chain for deterministic initialisation ─────────
CHAIN_SEEDS = [SEED + i for i in range(4)]


def _safe_expm1(log_pred: np.ndarray) -> np.ndarray:
    """
    Clip log-scale predictions then exponentiate.
    Prevents numerical overflow (inf) and underflow that occur when
    OLS or Bayesian models extrapolate to extreme feature values.
    """
    return np.clip(
        np.expm1(np.clip(log_pred, LOG_CLIP_LO, LOG_CLIP_HI)), 0, None)


# ===========================================================================
# 1. OLS RUNNER  (M1 CB OLS and M4 Causal OLS)
# ===========================================================================

def run_ols(X_ols: np.ndarray,
            y_log: np.ndarray,
            y_obs: np.ndarray,
            splits: list,
            weights=None,
            label: str = "OLS") -> tuple:
    """
    Cross-validated OLS (weights=None) or IPW WLS (weights=array).

    OLS is a closed-form estimator — no random state is involved in
    the coefficient computation itself. The only randomness is in the
    CV split construction, which is handled by the caller via seeded
    KFold or GroupKFold objects.

    Returns (metrics_dict, oof_pred_log, full_data_coef).
    """
    n     = len(y_log)
    pred  = np.zeros(n)
    folds = []
    print(f"\n-- {label} --")

    for k, (tr, te) in enumerate(splits, 1):
        X_tr, X_te = X_ols[tr], X_ols[te]
        y_tr, y_te = y_log[tr], y_log[te]

        if weights is not None:
            sw       = np.sqrt(weights[tr])
            coef, *_ = lstsq(X_tr * sw[:, None], y_tr * sw, rcond=None)
        else:
            coef, *_ = lstsq(X_tr, y_tr, rcond=None)

        yp       = X_te @ coef
        pred[te] = yp
        fm       = fold_metrics(y_te, y_obs[te], yp)
        folds.append(fm)
        print(f"  Fold {k}: R2={fm['r2']:.4f}  "
              f"MdAPE={fm['mape']:.2f}%  "
              f"RMSE={fm['rmse']:,.0f}")

    # Full-data refit for deployment test — deterministic (closed-form)
    if weights is not None:
        sw_all    = np.sqrt(weights)
        coef_full, *_ = lstsq(
            X_ols * sw_all[:, None], y_log * sw_all, rcond=None)
    else:
        coef_full, *_ = lstsq(X_ols, y_log, rcond=None)

    return aggregate_folds(folds, n), pred, coef_full


# ===========================================================================
# 2. OPTUNA OBJECTIVE FACTORY
# ===========================================================================

def build_optuna_objective(X_tr_pd, y_tr, y_tr_orig, bids_tr,
                            n_inner, seed, use_groups,
                            sample_weight=None):
    """
    Factory returning an Optuna objective for XGBoost tuning.

    Seeding strategy
    ----------------
    - Optuna TPE sampler receives `seed` (SEED + fold_index)
    - Each XGBRegressor in the inner loop receives random_state=seed
    - Inner holdout splits use np.random.default_rng(seed) — deterministic
    - n_jobs=-1 inside the objective: fold models are used only for
      hyperparameter selection, not for the deployed model, so small
      floating-point non-determinism here does not affect final results

    use_groups=True  -> GroupKFold inner loop (M5 Causal XGB)
    use_groups=False -> KFold inner loop      (M2 CB XGB)
    """
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 0.01, 0.30, log=True)
        if lr >= 0.10:
            depth = trial.suggest_int("max_depth", 3, 6)
            sub   = trial.suggest_float("subsample", 0.6, 1.0)
        else:
            depth = trial.suggest_int("max_depth", 4, 10)
            sub   = trial.suggest_float("subsample", 0.5, 1.0)

        cbt = (trial.suggest_float("colsample_bytree", 0.4, 0.8)
               if depth >= 7
               else trial.suggest_float("colsample_bytree", 0.5, 1.0))

        params = dict(
            learning_rate     = lr,
            max_depth         = depth,
            subsample         = sub,
            colsample_bytree  = cbt,
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            colsample_bylevel = trial.suggest_float(
                "colsample_bylevel", 0.4, 1.0),
            reg_alpha  = trial.suggest_float("reg_alpha",  1e-3, 10.0, log=True),
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma      = trial.suggest_float("gamma", 0.0, 5.0),
            n_estimators          = 1000,
            early_stopping_rounds = 30,
            objective    = "reg:squarederror",
            eval_metric  = "rmse",
            tree_method  = "hist",
            random_state = seed,
            n_jobs       = -1,
            verbosity    = 0,
            enable_categorical = False,
        )

        if use_groups:
            inner_cv   = GroupKFold(n_splits=n_inner)
            split_iter = inner_cv.split(X_tr_pd, groups=bids_tr)
        else:
            inner_cv   = KFold(n_splits=n_inner, shuffle=True,
                               random_state=seed)
            split_iter = inner_cv.split(X_tr_pd)

        mapes = []
        for ti, vi in split_iter:
            Xi_tr, Xi_va = X_tr_pd.iloc[ti], X_tr_pd.iloc[vi]
            yi_tr        = y_tr[ti]
            yi_va_orig   = y_tr_orig[vi]
            es_tr, es_va = (
                grouped_holdout(bids_tr[ti], frac=0.15, seed=seed)
                if use_groups
                else random_holdout(len(ti), frac=0.15, seed=seed)
            )
            fit_kw = dict(
                eval_set=[(Xi_tr.iloc[es_va], yi_tr[es_va])],
                verbose=False,
            )
            if sample_weight is not None:
                fit_kw["sample_weight"] = sample_weight[ti][es_tr]

            m = xgb.XGBRegressor(**params)
            m.fit(Xi_tr.iloc[es_tr], yi_tr[es_tr], **fit_kw)
            po = _safe_expm1(m.predict(Xi_va))
            mapes.append(float(np.median(
                np.abs((yi_va_orig - po) / (yi_va_orig + 1e-6))
            ) * 100))

        return float(np.mean(mapes))

    return objective


# ===========================================================================
# 3. XGB RUNNER  (M2 CB XGB and M5 Causal XGB)
# ===========================================================================

def run_xgb_tuned(X_pd, y_log, y_obs, bids, splits,
                   ipw_w=None, use_groups=True,
                   n_trials=N_TRIALS, label="XGB tuned") -> tuple:
    """
    Nested CV with Optuna TPE.

    Seeding strategy
    ----------------
    - Outer fold final model : random_state=SEED, n_jobs=1 (deterministic)
    - Full-data refit        : random_state=SEED, n_jobs=1 (deterministic)
    - Inner Optuna loop      : TPESampler(seed=SEED+k), n_jobs=-1 (speed)
    - Early-stopping holdout : grouped_holdout / random_holdout with fixed seed

    Using n_jobs=1 on the outer fold models and the full-data refit
    ensures that the deployed model (the one used in deployment_test
    and reported in figures) is fully deterministic. The inner CV
    models use n_jobs=-1 for speed since they only affect hyperparameter
    selection, not the final predictions.

    Returns (metrics_dict, oof_pred_log, final_model).
    """
    n                = len(y_log)
    pred             = np.zeros(n)
    folds            = []
    best_params_list = []
    print(f"\n-- {label} --")

    for k, (tr, te) in enumerate(splits, 1):
        t0        = time.time()
        X_tr      = X_pd.iloc[tr].reset_index(drop=True)
        X_te      = X_pd.iloc[te]
        y_tr      = y_log[tr]
        y_te      = y_log[te]
        y_tr_orig = y_obs[tr]
        y_te_orig = y_obs[te]
        bids_tr   = bids[tr]
        w_tr      = None if ipw_w is None else ipw_w[tr]

        # ── Inner Optuna tuning ───────────────────────────────────────────────
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED + k),
        )
        study.optimize(
            build_optuna_objective(
                X_tr, y_tr, y_tr_orig, bids_tr,
                N_INNER, SEED + k,
                use_groups    = use_groups,
                sample_weight = w_tr,
            ),
            n_trials=n_trials, show_progress_bar=False,
        )
        bp = study.best_params
        best_params_list.append(bp)
        print(f"  Fold {k} inner MdAPE={study.best_value:.2f}%  "
              f"best_lr={bp.get('learning_rate', '?'):.4f}  "
              f"best_depth={bp.get('max_depth', '?')}")

        # ── Outer fold model — n_jobs=1 for deterministic CV metrics ─────────
        es_tr_m, es_va_m = (
            grouped_holdout(bids_tr, frac=0.15, seed=SEED + k * 1000)
            if use_groups
            else random_holdout(len(tr), frac=0.15, seed=SEED + k * 1000)
        )
        final_fold = xgb.XGBRegressor(
            **bp,
            n_estimators          = 2000,
            early_stopping_rounds = 50,
            objective    = "reg:squarederror",
            eval_metric  = "rmse",
            tree_method  = "hist",
            random_state = SEED,
            n_jobs       = 1,
            verbosity    = 0,
            enable_categorical = False,
        )
        fit_kw = dict(
            eval_set=[(X_tr.iloc[es_va_m], y_tr[es_va_m])],
            verbose=False,
        )
        if w_tr is not None:
            fit_kw["sample_weight"] = w_tr[es_tr_m]
        final_fold.fit(X_tr.iloc[es_tr_m], y_tr[es_tr_m], **fit_kw)

        yp       = final_fold.predict(X_te)
        pred[te] = yp
        fm       = fold_metrics(y_te, y_te_orig, yp)
        folds.append(fm)
        print(f"         outer: R2={fm['r2']:.4f}  "
              f"MdAPE={fm['mape']:.2f}%  "
              f"RMSE={fm['rmse']:,.0f}  "
              f"[{time.time() - t0:.0f}s]")

    # ── Full-data refit for deployment test — n_jobs=1 for determinism ────────
    mid_bp = best_params_list[len(best_params_list) // 2]
    X_full = X_pd.reset_index(drop=True)
    es_tr_f, es_va_f = (
        grouped_holdout(bids, frac=0.15, seed=SEED)
        if use_groups
        else random_holdout(n, frac=0.15, seed=SEED)
    )
    final = xgb.XGBRegressor(
        **mid_bp,
        n_estimators          = 2000,
        early_stopping_rounds = 50,
        objective    = "reg:squarederror",
        eval_metric  = "rmse",
        tree_method  = "hist",
        random_state = SEED,
        n_jobs       = 1,
        verbosity    = 0,
        enable_categorical = False,
    )
    fit_kw_f = dict(
        eval_set=[(X_full.iloc[es_va_f], y_log[es_va_f])],
        verbose=False,
    )
    if ipw_w is not None:
        fit_kw_f["sample_weight"] = ipw_w[es_tr_f]
    final.fit(X_full.iloc[es_tr_f], y_log[es_tr_f], **fit_kw_f)

    return aggregate_folds(folds, n), pred, final


# ===========================================================================
# 4. BAYESIAN RUNNER  (M3 CB Bayes and M6 Causal Bayes)
# ===========================================================================

def run_hbm(source_df: pl.DataFrame,
            y_log: np.ndarray,
            y_obs: np.ndarray,
            feature_cols: list,
            n_obs: int,
            ipw_w=None,
            prior_upstream_sigma: float = 0.5,
            label: str = "HBM") -> tuple:
    """
    Hierarchical Bayesian model evaluated via PSIS-LOO.

    Seeding strategy
    ----------------
    - np.random.seed(SEED) before pm.sample() controls the jitter in
      jitter+adapt_diag initialisation, which uses numpy's global state
    - CHAIN_SEEDS = [SEED, SEED+1, SEED+2, SEED+3] seeds each chain
      independently for deterministic NUTS trajectories
    - pm.sample_posterior_predictive() receives CHAIN_SEEDS so the
      posterior predictive draws are also reproducible

    M3 CB Bayes  (ipw_w=None, feature_cols=CB_ALL_FEATURES,
                  prior_upstream_sigma=2.5):
      Detects MEDIATOR_COLS within feature_cols and adds mediator slopes.
      No IPW. Derived metrics, spatial and compliance columns in
      CB_ALL_FEATURES are NOT given explicit slope parameters — the
      Bayesian parametric structure uses upstream features and mediators
      only. M3 is trained on the larger ~21,714-row CB dataset.
      Upstream slope priors are Normal(0, 2.5) — uninformative, reflecting
      the causally-blind analyst's absence of physical prior knowledge.
      Mediator slope priors are Normal(0, 2.0) (hardcoded; see below).

    M6 Causal Bayes (ipw_w=array, feature_cols=UPSTREAM_COLS,
                     prior_upstream_sigma=0.5):
      Upstream only. IPW via pm.Potential. Structural causal model.
      Trained on ~21,406 compliant rows.
      Upstream slope priors are Normal(0, 0.5) — informative, encoding
      the causal analyst's expectation of near-unit elasticities from
      the tested DAG.

    Returns (metrics_dict, trace, loo_pred_log).
    """
    print(f"\n-- {label} --")
    print(f"  n_obs         : {n_obs:,}")
    print(f"  feature_cols  : {feature_cols}")
    print(f"  IPW           : {'yes' if ipw_w is not None else 'no'}")

    all_types   = sorted(source_df[RAW_CAT].unique().to_list())
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    type_idx    = np.array(
        [type_to_idx[t] for t in source_df[RAW_CAT].to_list()])

    log_area   = np.log(
        source_df["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs  = np.log(source_df["of_buildings"].to_numpy().clip(1))
    year_data  = source_df["data_year"].to_numpy().astype(float)
    year_built = source_df["year_built"].to_numpy().astype(float)

    log_area_c  = log_area  - log_area.mean()
    log_bldgs_c = log_bldgs - log_bldgs.mean()
    yr_data_s   = (year_data  - year_data.mean())  / year_data.std()
    yr_built_s  = (year_built - year_built.mean()) / year_built.std()

    include_mediators = any(c in feature_cols for c in MEDIATOR_COLS)
    med_arrays = {}
    if include_mediators:
        for c in MEDIATOR_COLS:
            vals = source_df[c].to_numpy().astype(float)
            std  = vals.std()
            med_arrays[c] = (vals - vals.mean()) / (std if std > 1e-9 else 1.0)

    coords = {"obs": np.arange(n_obs), "type": all_types}

    with pm.Model(coords=coords) as model:
        tidx = pm.Data("type_idx",   type_idx,    dims="obs")
        la   = pm.Data("log_area",   log_area_c,  dims="obs")
        lb   = pm.Data("log_bldgs",  log_bldgs_c, dims="obs")
        yd   = pm.Data("year_data",  yr_data_s,   dims="obs")
        yb   = pm.Data("year_built", yr_built_s,  dims="obs")

        ab    = pm.Normal("alpha_bar",   mu=7.0, sigma=1.0)
        sa    = pm.HalfNormal("sigma_alpha", sigma=1.0)
        dj    = pm.Normal("alpha_offset", mu=0, sigma=1, dims="type")
        alpha = pm.Deterministic("alpha", ab + dj * sa, dims="type")
        bA    = pm.Normal("beta_A", mu=0, sigma=prior_upstream_sigma)
        bB    = pm.Normal("beta_B", mu=0, sigma=prior_upstream_sigma)
        bT    = pm.Normal("beta_T", mu=0, sigma=prior_upstream_sigma)
        bY    = pm.Normal("beta_Y", mu=0, sigma=prior_upstream_sigma)
        sig   = pm.HalfNormal("sigma", sigma=0.5)

        mu = alpha[tidx] + bA*la + bB*lb + bT*yd + bY*yb

        # Mediator slopes — M3 CB Bayes only (prior always Normal(0, 2.0))
        if include_mediators:
            for c in MEDIATOR_COLS:
                med_data = pm.Data(c, med_arrays[c], dims="obs")
                b_med    = pm.Normal(f"beta_{c}", mu=0, sigma=2.0)
                mu       = mu + b_med * med_data

        # IPW correction — M6 Causal Bayes only
        if ipw_w is not None:
            ipw_data = pm.Data("ipw", ipw_w, dims="obs")
            ll       = pm.logp(pm.Normal.dist(mu=mu, sigma=sig), y_log)
            pm.Potential("ipw_weight", (ipw_data - 1.0) * ll)

        pm.Normal("y_hat", mu=mu, sigma=sig,
                  observed=y_log, dims="obs")

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

    sp        = ["alpha_bar", "sigma_alpha",
                 "beta_A", "beta_B", "beta_T", "beta_Y", "sigma"]
    rhat_vals = _flatten_az_result(az.rhat(trace, var_names=sp))
    ess_vals  = _flatten_az_result(az.ess(trace,  var_names=sp))
    rhat_max  = float(rhat_vals.max()) if len(rhat_vals) else float("nan")
    ess_min   = int(ess_vals.min())    if len(ess_vals)  else 0
    n_div     = int(trace.sample_stats.diverging.values.sum())
    loo       = az.loo(trace, pointwise=True)
    pk_vals   = _pareto_k(loo)
    n_bad_k   = int((pk_vals > 0.7).sum()) if len(pk_vals) else -1

    print(f"  R-hat max={rhat_max:.4f}  "
          f"ESS min={ess_min:,}  "
          f"divergences={n_div}  "
          f"Pareto-k>0.7: {n_bad_k}")
    for flag, cond, msg in [
        ("R-hat",       rhat_max > 1.01, f"{rhat_max:.4f} > 1.01"),
        ("ESS",         ess_min  < 400,  f"{ess_min:,} < 400"),
        ("divergences", n_div    > 0,    f"{n_div} divergent transitions"),
    ]:
        if cond:
            print(f"  WARNING {flag}: {msg}")

    ll_flat  = trace.log_likelihood["y_hat"].values.reshape(-1, n_obs)
    pp_draws = trace.posterior_predictive["y_hat"].values.reshape(-1, n_obs)

    loo_pred = np.zeros(n_obs)
    for i in range(n_obs):
        log_w    = -ll_flat[:, i]
        log_w   -= log_w.max()
        w        = np.exp(log_w); w /= w.sum()
        loo_pred[i] = float(np.sum(w * pp_draws[:, i]))

    loo_orig = _safe_expm1(loo_pred)
    pi_lo    = _safe_expm1(np.percentile(pp_draws, 5.5,  axis=0))
    pi_hi    = _safe_expm1(np.percentile(pp_draws, 94.5, axis=0))

    from sklearn.metrics import r2_score, mean_squared_error
    metrics = dict(
        r2_mean     = float(r2_score(y_log, loo_pred)),
        r2_std      = 0.0,
        mape_mean   = float(np.median(
            np.abs((y_obs - loo_orig) / (y_obs + 1e-6))) * 100),
        mape_std    = 0.0,
        rmse_median = float(np.sqrt(np.median((y_obs - loo_orig) ** 2))),
        rmse_mean   = float(np.sqrt(np.mean((y_obs - loo_orig) ** 2))),
        cov_mean    = float(
            np.mean((y_obs >= pi_lo) & (y_obs <= pi_hi)) * 100),
        elpd        = _loo_elpd(loo),
        elpd_se     = _loo_se(loo),
        rhat_max    = rhat_max,
        ess_min     = ess_min,
        n_div       = n_div,
        n_bad_k     = n_bad_k,
    )
    return metrics, trace, loo_pred


# ===========================================================================
# 5. BAYESIAN DEPLOYMENT PREDICTOR
# ===========================================================================

def predict_hbm_deployment(trace,
                            source_df_train: pl.DataFrame,
                            source_df_deploy: pl.DataFrame,
                            feature_cols: list,
                            mediator_fill_vals: dict | None = None) -> np.ndarray:
    """
    Apply a fitted HBM posterior to non-compliant records.

    The posterior is NOT re-sampled. pm.sample_posterior_predictive()
    evaluates p(y_new | theta, x_new) using the stored posterior draws.

    Parameters
    ----------
    trace              : PyMC InferenceData from run_hbm()
    source_df_train    : training DataFrame (for standardisation params)
    source_df_deploy   : deployment DataFrame (non-compliant buildings)
    feature_cols       : feature list (determines include_mediators flag)
    mediator_fill_vals : optional dict {col: fill_value} for mediator
                         imputation at deployment time.
                         - None / not provided -> structural zero
                           (used for M6 Causal Bayes: tested DAG
                           establishes zero fuel for non-compliant)
                         - dict of CB training medians (ctx["cb_medians"])
                           (used for M3 CB Bayes: naive analyst imputes
                           missing mediators with training column median,
                           the same imputation used during training)

    Returns pred_orig : np.ndarray, shape (n_deploy,), metric tons CO2e
    """
    n_deploy = len(source_df_deploy)
    print(f"  Building deployment predictive for {n_deploy:,} buildings ...")

    # ── Recover training standardisation parameters ───────────────────────────
    log_area_tr   = np.log(
        source_df_train["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs_tr  = np.log(
        source_df_train["of_buildings"].to_numpy().clip(1))
    year_data_tr  = source_df_train["data_year"].to_numpy().astype(float)
    year_built_tr = source_df_train["year_built"].to_numpy().astype(float)

    la_mean  = log_area_tr.mean()
    lb_mean  = log_bldgs_tr.mean()
    yd_mean  = year_data_tr.mean();  yd_std  = year_data_tr.std()
    yb_mean  = year_built_tr.mean(); yb_std  = year_built_tr.std()

    # ── Transform deployment features on training scale ───────────────────────
    log_area_d   = np.log(
        source_df_deploy["gross_floor_area_buildings_sq_ft"].to_numpy())
    log_bldgs_d  = np.log(
        source_df_deploy["of_buildings"].to_numpy().clip(1))
    year_data_d  = source_df_deploy["data_year"].to_numpy().astype(float)
    year_built_d = source_df_deploy["year_built"].to_numpy().astype(float)

    log_area_c_d  = log_area_d  - la_mean
    log_bldgs_c_d = log_bldgs_d - lb_mean
    yr_data_s_d   = (year_data_d  - yd_mean) / yd_std
    yr_built_s_d  = (year_built_d - yb_mean) / yb_std

    # ── Build type index ──────────────────────────────────────────────────────
    all_types   = sorted(source_df_train[RAW_CAT].unique().to_list())
    type_to_idx = {t: i for i, t in enumerate(all_types)}
    type_idx_d  = np.array([
        type_to_idx.get(t, 0)
        for t in source_df_deploy[RAW_CAT].fill_null("Unknown").to_list()
    ])

    # ── Mediator standardisation ──────────────────────────────────────────────
    # For M6 Causal Bayes (mediator_fill_vals=None):
    #   Structural zero: deploy value = 0, standardised as (0 - mean) / std.
    #   The tested DAG establishes zero fuel for non-compliant records.
    #
    # For M3 CB Bayes (mediator_fill_vals=ctx["cb_medians"]):
    #   Naive analyst imputation: deploy value = training column median,
    #   standardised as (median - mean) / std.
    #   This is the same imputation used during CB training. The prediction
    #   is not zero-collapsed but is miscalibrated — a different failure
    #   mode that looks more plausible but remains causally unsound.
    include_mediators = any(c in feature_cols for c in MEDIATOR_COLS)
    med_arrays_d = {}
    if include_mediators:
        for c in MEDIATOR_COLS:
            vals_tr = source_df_train[c].to_numpy().astype(float)
            mean_tr = vals_tr.mean()
            std_tr  = vals_tr.std()
            std_tr  = std_tr if std_tr > 1e-9 else 1.0

            if mediator_fill_vals is not None and c in mediator_fill_vals:
                fill = mediator_fill_vals[c]
            else:
                fill = 0.0   # structural zero (causal default)

            standardised        = (fill - mean_tr) / std_tr
            med_arrays_d[c]     = np.full(n_deploy, standardised)

    # ── Rebuild model structure with deployment data ───────────────────────────
    coords_d = {"obs": np.arange(n_deploy), "type": all_types}

    with pm.Model(coords=coords_d):
        pm.Data("type_idx",   type_idx_d,    dims="obs")
        pm.Data("log_area",   log_area_c_d,  dims="obs")
        pm.Data("log_bldgs",  log_bldgs_c_d, dims="obs")
        pm.Data("year_data",  yr_data_s_d,   dims="obs")
        pm.Data("year_built", yr_built_s_d,  dims="obs")

        ab    = pm.Normal("alpha_bar",   mu=7.0, sigma=1.0)
        sa    = pm.HalfNormal("sigma_alpha", sigma=1.0)
        dj    = pm.Normal("alpha_offset", mu=0, sigma=1, dims="type")
        alpha = pm.Deterministic("alpha", ab + dj * sa, dims="type")
        bA    = pm.Normal("beta_A", mu=0, sigma=0.5)
        bB    = pm.Normal("beta_B", mu=0, sigma=0.5)
        bT    = pm.Normal("beta_T", mu=0, sigma=0.5)
        bY    = pm.Normal("beta_Y", mu=0, sigma=0.5)
        sig   = pm.HalfNormal("sigma", sigma=0.5)

        mu = (alpha[type_idx_d]
              + bA * log_area_c_d
              + bB * log_bldgs_c_d
              + bT * yr_data_s_d
              + bY * yr_built_s_d)

        if include_mediators:
            for c in MEDIATOR_COLS:
                pm.Data(c, med_arrays_d[c], dims="obs")
                b_med = pm.Normal(f"beta_{c}", mu=0, sigma=2.0)
                mu    = mu + b_med * med_arrays_d[c]

        pm.Normal("y_hat", mu=mu, sigma=sig,
                  observed=np.zeros(n_deploy), dims="obs")

        np.random.seed(SEED)
        ppc_deploy = pm.sample_posterior_predictive(
            trace,
            var_names   = ["y_hat"],
            random_seed = SEED,
            progressbar = False,
        )

    pp_draws  = ppc_deploy.posterior_predictive["y_hat"].values.reshape(
        -1, n_deploy)
    pred_log  = np.median(pp_draws, axis=0)
    pred_orig = _safe_expm1(pred_log)
    return pred_orig


# ===========================================================================
# 6. FEATURE BUILDER FOR DEPLOYMENT (XGBoost models)
# ===========================================================================

def build_label_encoded_features_deploy(
        source_df: pl.DataFrame,
        feature_cols: list,
        reference_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Build a label-encoded feature DataFrame for XGBoost deployment.
    Categorical vocabulary sourced from reference_pd to handle unseen
    categories without KeyError.
    """
    def _safe_le(train_series: pd.Series) -> LabelEncoder:
        vals = train_series.fillna("__unknown__").astype(str).tolist()
        if "__unknown__" not in vals:
            vals = vals + ["__unknown__"]
        le = LabelEncoder()
        le.fit(vals)
        return le

    def _safe_transform(series: pd.Series,
                         le: LabelEncoder) -> pd.Series:
        known  = set(le.classes_)
        mapped = series.fillna("__unknown__").astype(str)
        mapped = mapped.where(mapped.isin(known), other="__unknown__")
        return pd.Series(le.transform(mapped),
                         index=series.index, dtype=np.int64)

    available = [c for c in feature_cols if c in source_df.columns]
    X = source_df.select(available).to_pandas()

    for c in X.columns:
        if X[c].dtype == bool or str(X[c].dtype) in ("bool", "boolean"):
            X[c] = X[c].astype(float).astype(int)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("object", "string"):
            ref_col = reference_pd[c] if c in reference_pd.columns else X[c]
            le      = _safe_le(ref_col)
            X[c]    = _safe_transform(X[c], le)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med  = X[c].median()
        X[c] = X[c].fillna(0.0 if pd.isna(med) else med)

    return X


# ===========================================================================
# 7. OLS DEPLOYMENT MATRIX BUILDER
# ===========================================================================

def build_ols_deploy(train_df: pl.DataFrame,
                     deploy_df: pl.DataFrame,
                     feature_cols: list) -> np.ndarray:
    """
    Build an OLS deployment design matrix using encoders fitted on training.

    Mirrors build_ols_design() from pipeline.py but fits categorical
    encoders on train_df and applies them to deploy_df, preventing
    vocabulary mismatch between training and deployment.

    Used for M1 CB OLS deployment (CB_ALL_FEATURES has many categorical
    columns: zip_code, community_area, reporting_status, exempt flag,
    primary_property_type). For M4 Causal OLS only property type is
    categorical, so the same pattern still applies correctly.
    """
    # Identify categorical columns present in training data
    cat_cols = [c for c in feature_cols
                if c in train_df.columns
                and train_df[c].dtype in (pl.String, pl.Boolean, pl.Utf8)]
    num_cols = [c for c in feature_cols
                if c in train_df.columns and c not in cat_cols]

    # Numeric columns from deployment
    deploy_num_cols = [c for c in num_cols if c in deploy_df.columns]
    X_num = (deploy_df.select(deploy_num_cols).to_pandas()
             .apply(pd.to_numeric, errors="coerce")
             .fillna(0.0).values) if deploy_num_cols else np.empty((len(deploy_df), 0))

    if cat_cols:
        train_cat_cols  = [c for c in cat_cols if c in train_df.columns]
        deploy_cat_cols = [c for c in cat_cols if c in deploy_df.columns]

        train_cats  = (train_df.select(train_cat_cols).to_pandas()
                       .astype(str).fillna("__missing__"))
        deploy_cats = (deploy_df.select(deploy_cat_cols).to_pandas()
                       .astype(str).fillna("__missing__"))

        # Align column order — deploy may be missing some cat cols
        for c in train_cat_cols:
            if c not in deploy_cats.columns:
                deploy_cats[c] = "__missing__"
        deploy_cats = deploy_cats[train_cat_cols]

        ohe     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(train_cats.values)
        cat_enc = ohe.transform(deploy_cats.values)
        X       = np.hstack([X_num, cat_enc, np.ones((len(deploy_df), 1))])
    else:
        X = np.hstack([X_num, np.ones((len(deploy_df), 1))])

    return X


# ===========================================================================
# 8. DEPLOYMENT TEST  (all six models)
# ===========================================================================

def deployment_test(ctx: dict,
                    trained_models: dict,
                    traces: dict) -> dict:
    """
    Apply all six trained models to the 5,510 non-compliant building-year records.

    CB deployment imputation (M1, M2, M3)
    --------------------------------------
    Non-compliant buildings have no mediators or derived metrics —
    they never filed a report. The naive analyst would impute these
    with the training column median (ctx["cb_medians"]), the same
    imputation applied during CB training. This is consistent but
    causally unsound: predictions are conditioned on the energy
    behaviour of a typical *compliant* building, not on any actual
    information about the non-compliant building.

    The expected CB failure modes under the corrected design:
      M1 CB OLS  : Derived metric columns receive training-median
                   imputation. OLS extrapolation may inflate the mean
                   for some buildings, but the median is meaningful.
      M2 CB XGB  : With ghg_intensity imputed to median, XGBoost
                   predicts near (median_ghg_intensity × floor_area)
                   for each building — a miscalibrated but non-zero
                   prediction. This is a more insidious failure than
                   the near-zero collapse of the old design.
      M3 CB Bayes: Mediators imputed to training median on the
                   standardised scale. Predictions are modestly shifted
                   relative to the zero-imputed causal case.

    Causal deployment (M4, M5, M6)
    --------------------------------
    Non-compliant buildings receive structural zeros for mediators
    (tested DAG). Causal models trained on upstream features only
    remain stable across compliance status.

    All log-scale predictions are clipped via _safe_expm1().
    """
    print("\n-- Deployment test: non-compliant buildings --")
    nc   = ctx["non_compliant"]
    n_nc = len(nc)
    print(f"  Non-compliant buildings : {n_nc:,}")
    print(f"  Log-scale predictions clipped to "
          f"[{LOG_CLIP_LO}, {LOG_CLIP_HI}] before exponentiation")

    cb_medians = ctx["cb_medians"]

    # ── Prepare non-compliant rows with CB median imputation (M1, M2, M3) ─────
    # Fill each CB feature column with the training median stored in cb_medians.
    # This replicates exactly the imputation the naive analyst applied to
    # missing values in the training set.
    nc_cb = nc
    for c, fill_val in cb_medians.items():
        if c in nc_cb.columns:
            nc_cb = nc_cb.with_columns(pl.col(c).fill_null(fill_val))

    # ── Prepare non-compliant rows with structural zeros (M4, M5, M6) ────────
    nc_causal = nc
    for c in MEDIATOR_COLS:
        nc_causal = nc_causal.with_columns(pl.col(c).fill_null(0.0))

    # ── OLS design matrices ───────────────────────────────────────────────────
    # M1 CB OLS: CB_ALL_FEATURES includes categorical string/boolean columns.
    #   build_ols_deploy fits encoders on training data then transforms deploy.
    X_nc_m1 = build_ols_deploy(ctx["obs_cb"],     nc_cb,     CB_ALL_FEATURES)
    X_nc_m4 = build_ols_deploy(ctx["obs_causal"], nc_causal, UPSTREAM_COLS)

    # ── XGBoost feature frames ────────────────────────────────────────────────
    obs_cb_pd = ctx["obs_cb"].select(
        [c for c in CB_ALL_FEATURES if c in ctx["obs_cb"].columns]
    ).to_pandas()
    nc_m2_pd  = build_label_encoded_features_deploy(
        nc_cb, CB_ALL_FEATURES, obs_cb_pd)

    obs_c_pd = ctx["obs_causal"].select(
        [c for c in UPSTREAM_COLS if c in ctx["obs_causal"].columns]
    ).to_pandas()
    nc_m5_pd = build_label_encoded_features_deploy(
        nc_causal, UPSTREAM_COLS, obs_c_pd)

    def _align(X: pd.DataFrame,
               model: xgb.XGBRegressor) -> pd.DataFrame:
        try:
            expected = model.get_booster().feature_names
            if expected is None:
                return X
            return X.reindex(columns=expected, fill_value=0)
        except Exception:
            return X

    # ── OLS and XGBoost predictions ───────────────────────────────────────────
    deploy_preds = {}

    for mname, model_obj in trained_models.items():
        try:
            if mname == "M1 CB OLS":
                pred_log = X_nc_m1 @ model_obj
            elif mname == "M4 Causal OLS":
                pred_log = X_nc_m4 @ model_obj
            elif mname == "M2 CB XGB":
                pred_log = model_obj.predict(_align(nc_m2_pd, model_obj))
            elif mname == "M5 Causal XGB":
                pred_log = model_obj.predict(_align(nc_m5_pd, model_obj))
            else:
                continue

            pred_orig           = _safe_expm1(pred_log)
            deploy_preds[mname] = pred_orig
            print(f"  {mname:<25} "
                  f"median={np.median(pred_orig):>10,.0f}  "
                  f"mean={np.mean(pred_orig):>10,.0f}  "
                  f"metric tons CO2e")

        except Exception as exc:
            print(f"  {mname:<25} FAILED: {exc}")
            deploy_preds[mname] = None

    # ── Bayesian model predictions ────────────────────────────────────────────
    for mname, trace in traces.items():
        try:
            if mname == "M3 CB Bayes":
                print(f"  {mname:<25} computing posterior predictive ...")
                # CB Bayes: impute mediators with training medians (not zeros)
                # so that deployment matches the CB training imputation.
                pred_orig = predict_hbm_deployment(
                    trace              = trace,
                    source_df_train    = ctx["obs_cb"],
                    source_df_deploy   = nc_cb,
                    feature_cols       = CB_ALL_FEATURES,
                    mediator_fill_vals = cb_medians,
                )
            elif mname == "M6 Causal Bayes":
                print(f"  {mname:<25} computing posterior predictive ...")
                # Causal Bayes: structural zeros for mediators (tested DAG)
                pred_orig = predict_hbm_deployment(
                    trace              = trace,
                    source_df_train    = ctx["obs_causal"],
                    source_df_deploy   = nc_causal,
                    feature_cols       = UPSTREAM_COLS,
                    mediator_fill_vals = None,
                )
            else:
                continue

            deploy_preds[mname] = pred_orig
            print(f"  {mname:<25} "
                  f"median={np.median(pred_orig):>10,.0f}  "
                  f"mean={np.mean(pred_orig):>10,.0f}  "
                  f"metric tons CO2e")

        except Exception as exc:
            print(f"  {mname:<25} FAILED: {exc}")
            deploy_preds[mname] = None

    return deploy_preds


# ===========================================================================
# 9. ORCHESTRATOR
# ===========================================================================

def run_all_models(df: pl.DataFrame) -> tuple:
    """
    Run all six models in the 2x3 factorial design and the deployment test.

    Global random seed is set at the top of experiment.py before this
    function is called. Additional model-level seeding is applied within
    each runner function as documented in their docstrings.

    Returns (results, preds, ctx, traces, deploy_preds).

    Row counts
    ----------
    CB pipeline    : n_cb    (~21,714 rows)
    Causal pipeline: n_causal (~21,406 rows)
    The two pipelines intentionally differ — this is part of the design.
    """
    # ── STEP 1: Preprocessing ─────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 -- PREPROCESSING")
    print("=" * 70)
    ctx = preprocess(df)

    obs_cb        = ctx["obs_cb"]
    n_cb          = ctx["n_cb"]
    y_log_cb      = ctx["y_log_cb"]
    y_obs_cb      = ctx["y_obs_cb"]
    bids_cb       = ctx["bids_cb"]
    splits_cb     = ctx["splits_cb"]

    obs_causal    = ctx["obs_causal"]
    n_causal      = ctx["n_causal"]
    y_log_causal  = ctx["y_log_causal"]
    y_obs_causal  = ctx["y_obs_causal"]
    bids_causal   = ctx["bids_causal"]
    splits_causal = ctx["splits_causal"]
    ipw_w         = ctx["ipw_w"]

    # ── STEP 2: M1 CB OLS ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 -- M1  Causally-Blind OLS")
    print("  Features : CB_ALL_FEATURES (upstream + spatial + compliance +")
    print("             mediators + derived metrics + water use)")
    print(f"  Rows     : {n_cb:,} (all rows with observed target)")
    print("  CV       : KFold — no building ID grouping")
    print("  IPW      : none")
    print("=" * 70)
    X_cb_ols = build_ols_design(obs_cb, CB_ALL_FEATURES)
    m1, pred_m1, coef_m1 = run_ols(
        X_cb_ols, y_log_cb, y_obs_cb, splits_cb,
        weights=None, label="M1 CB OLS",
    )

    # ── STEP 3: M2 CB XGB ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 -- M2  Causally-Blind XGB")
    print("  Features : CB_ALL_FEATURES (includes ghg_intensity, EUIs, etc.)")
    print(f"  Rows     : {n_cb:,} (all rows with observed target)")
    print("  CV       : KFold tuned — no building ID grouping")
    print("  IPW      : none")
    print("  Expected : R2 -> 1.0, MdAPE -> 0% (ghg_intensity ~ target / area)")
    print("=" * 70)
    X_cb_xgb, _, _, _ = build_label_encoded_features(
        obs_cb, CB_ALL_FEATURES)
    m2, pred_m2, mod_m2 = run_xgb_tuned(
        X_cb_xgb, y_log_cb, y_obs_cb, bids_cb, splits_cb,
        ipw_w=None, use_groups=False,
        n_trials=N_TRIALS, label="M2 CB XGB",
    )

    # ── STEP 4: M3 CB Bayes ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 -- M3  Causally-Blind Bayes")
    print("  Features : CB_ALL_FEATURES passed; model uses upstream structural")
    print("             features + mediator slopes (parametric structure does")
    print("             not accommodate all of CB_ALL_FEATURES explicitly)")
    print(f"  Rows     : {n_cb:,} (all rows with observed target)")
    print("  CV       : PSIS-LOO")
    print("  IPW      : none")
    print("  Expected : mediator slopes converge toward EPA emission factors")
    print("=" * 70)
    m3, trace_m3, pred_m3 = run_hbm(
        source_df            = obs_cb,
        y_log                = y_log_cb,
        y_obs                = y_obs_cb,
        feature_cols         = CB_ALL_FEATURES,
        n_obs                = n_cb,
        ipw_w                = None,
        prior_upstream_sigma = 2.5,   # CB analyst has no physical prior knowledge
        label                = "M3 CB Bayes",
    )

    # ── STEP 5: M4 Causal OLS ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 -- M4  Causal OLS")
    print("  Features : upstream only (structural zeros, tested DAG)")
    print(f"  Rows     : {n_causal:,} (compliant rows only)")
    print("  CV       : GroupKFold on building ID")
    print("  IPW      : back-door adjustment (WLS)")
    print("=" * 70)
    X_c_ols = build_ols_design(obs_causal, UPSTREAM_COLS)
    m4, pred_m4, coef_m4 = run_ols(
        X_c_ols, y_log_causal, y_obs_causal, splits_causal,
        weights=ipw_w, label="M4 Causal OLS",
    )

    # ── STEP 6: M5 Causal XGB ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 -- M5  Causal XGB  <- causal ceiling")
    print("  Features : upstream only (structural zeros, tested DAG)")
    print(f"  Rows     : {n_causal:,} (compliant rows only)")
    print("  CV       : GroupKFold tuned on building ID")
    print("  IPW      : back-door adjustment (sample weights)")
    print("=" * 70)
    X_c_xgb, _, _, _ = build_label_encoded_features(
        obs_causal, UPSTREAM_COLS)
    m5, pred_m5, mod_m5 = run_xgb_tuned(
        X_c_xgb, y_log_causal, y_obs_causal, bids_causal, splits_causal,
        ipw_w=ipw_w, use_groups=True,
        n_trials=N_TRIALS, label="M5 Causal XGB",
    )

    # ── STEP 7: M6 Causal Bayes ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7 -- M6  Causal Bayes  <- structural model")
    print("  Features : upstream only (structural zeros, tested DAG)")
    print(f"  Rows     : {n_causal:,} (compliant rows only)")
    print("  CV       : PSIS-LOO")
    print("  IPW      : back-door adjustment via pm.Potential")
    print("=" * 70)
    m6, trace_m6, pred_m6 = run_hbm(
        source_df    = obs_causal,
        y_log        = y_log_causal,
        y_obs        = y_obs_causal,
        feature_cols = UPSTREAM_COLS,
        n_obs        = n_causal,
        ipw_w        = ipw_w,
        label        = "M6 Causal Bayes",
    )

    # ── Persist M6 and M3 traces for inference ───────────────────────────────
    # _root resolves to the bundle root (drafts/complete); traces go to results/.
    _exp_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _root      = os.path.dirname(_exp_dir)
    _trace_dir = os.path.join(_root, "results", "inference")
    os.makedirs(_trace_dir, exist_ok=True)
    trace_m6.to_netcdf(os.path.join(_trace_dir, "m6_trace.nc"))
    print(f"  M6 trace saved -> {os.path.join(_trace_dir, 'm6_trace.nc')}")
    trace_m3.to_netcdf(os.path.join(_trace_dir, "m3_trace.nc"))
    print(f"  M3 trace saved -> {os.path.join(_trace_dir, 'm3_trace.nc')}")

    # ── Collect results ───────────────────────────────────────────────────────
    results = {
        "M1 CB OLS":       m1, "M2 CB XGB":       m2,
        "M3 CB Bayes":     m3, "M4 Causal OLS":   m4,
        "M5 Causal XGB":   m5, "M6 Causal Bayes": m6,
    }
    preds = {
        "M1 CB OLS":       pred_m1,
        "M2 CB XGB":       pred_m2,
        "M3 CB Bayes":     pred_m3,
        "M4 Causal OLS":   pred_m4,
        "M5 Causal XGB":   pred_m5,
        "M6 Causal Bayes": pred_m6,
    }
    trained_models = {
        "M1 CB OLS":     coef_m1,
        "M2 CB XGB":     mod_m2,
        "M4 Causal OLS": coef_m4,
        "M5 Causal XGB": mod_m5,
    }
    traces = {
        "M3 CB Bayes":     trace_m3,
        "M6 Causal Bayes": trace_m6,
    }

    # ── STEP 8: Deployment test (all six models) ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8 -- DEPLOYMENT TEST (all 6 models)")
    print("  CB models    : non-compliant buildings receive CB median")
    print("                 imputation for all features (mirrors training).")
    print("  Causal models: non-compliant buildings receive structural zeros")
    print("                 for mediators (tested DAG).")
    print("  Causal models expected to be stable (~850-1,100 metric tons).")
    print("=" * 70)
    deploy_preds = deployment_test(ctx, trained_models, traces=traces)

    # ── Per-pipeline row alignment check ─────────────────────────────────────
    print("\n-- Row-alignment check --")
    print(f"  CB pipeline    ({n_cb:,} rows)")
    for mname in ("M1 CB OLS", "M2 CB XGB", "M3 CB Bayes"):
        p      = preds[mname]
        status = "OK" if len(p) == n_cb else f"MISMATCH ({len(p)})"
        print(f"    {mname:<25} {len(p):,} rows  [{status}]")

    print(f"  Causal pipeline ({n_causal:,} rows)")
    for mname in ("M4 Causal OLS", "M5 Causal XGB", "M6 Causal Bayes"):
        p      = preds[mname]
        status = "OK" if len(p) == n_causal else f"MISMATCH ({len(p)})"
        print(f"    {mname:<25} {len(p):,} rows  [{status}]")

    return results, preds, ctx, traces, deploy_preds
