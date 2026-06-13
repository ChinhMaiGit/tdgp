"""
pipeline.py
===========
Shared infrastructure for the 2x3 TDGP factorial experiment.

Experimental design
-------------------
Two pipelines, three approaches each, giving six models total.

                    Frequentist     ML              Bayesian
                 ─────────────────────────────────────────────────
Causally-blind  │  M1 CB OLS    M2 CB XGB       M3 CB Bayes
                │  no IPW       no IPW           no IPW
                │  ALL features ALL features     ALL features
                │  all obs rows all obs rows     all obs rows
                │  KFold        KFold tuned      PSIS-LOO
                │
Causal          │  M4 Causal    M5 Causal        M6 Causal
                │  OLS          XGB              Bayes
                │  IPW          IPW              IPW
                │  upstream     upstream         upstream
                │  compliant    compliant        compliant
                │  GroupKFold   GroupKFold tuned PSIS-LOO

Realistic comparison design (replaces Option A)
-----------------------------------------------
Each pipeline uses the data as its analyst would actually use it.
Row counts intentionally differ:

  Causally-blind : ~21,714 rows — all rows with an observed GHG
                   target, regardless of compliance status. A naive
                   analyst has no reason to filter on compliance.

  Causal         : ~21,406 rows — compliant records with an
                   observed GHG target. The causal analyst knows
                   about the compliance gate and corrects for it
                   via IPW.

The row count difference is itself part of the paper's argument:
the naive analyst uses *more* data and still gets the wrong answer.

Four intentional differences between pipelines
----------------------------------------------
  1. Feature set   : ALL available features (CB) vs upstream only
                     (causal, restricted per tested DAG)
  2. Row subset    : all observed target rows (CB) vs compliant rows
                     only (causal, corrects for compliance gate)
  3. CV protocol   : KFold (CB) vs GroupKFold on building ID (causal)
  4. IPW correction: none (CB) vs back-door adjustment (causal)

Plus one additional Bayesian difference:
  5. Priors/model  : flat generic (M3 CB Bayes) vs calibrated
                     hierarchical (M6 Causal Bayes)

Mediator fill (documented but not a confound)
----------------------------------------------
  CB    : column median from all observed target rows (naive default)
  Causal: structural zero for district fuels (tested DAG)

Why CB_ALL_FEATURES differs from MEDIATOR_FEATURE_COLS
-------------------------------------------------------
The causally-blind analyst opens the dataset, drops obvious
non-predictors (identifiers, free-text fields, the target itself),
and uses everything left with reasonable coverage and correlation
with the target. This includes:

  - Derived metrics: site_eui, source_eui, weather-normalised EUIs,
    ghg_intensity_kg_co2e_sq_ft, energy_star_score,
    chicago_energy_rating. The causal analyst excludes these because
    the tested DAG establishes they are deterministic downstream
    functions of the mediators — computed from the same fuel data
    that produces GHG emissions. Including ghg_intensity as a
    predictor of total GHG is circular. The naive analyst has no
    reason to exclude them.

  - Spatial metadata: zip_code, community_area, latitude, longitude.
  - Compliance metadata: reporting_status, exempt flag.
  - Water use: water_use_kgal (partially observed).

The consequence is dramatic: M2 CB XGB achieves R² > 0.99 and
MdAPE ≈ 0% in CV because ghg_intensity × floor_area ≈ total GHG
(a near-identity transformation). The deployment test then exposes
the circularity: derived metrics are missing for non-compliant
buildings (they never submitted a report), so the CB models receive
median-imputed derived metrics rather than actual values, and their
predictions become uncalibrated to reality.
"""

# ---------------------------------------------------------------------------
# 0. Imports and configuration
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import time
import requests
import numpy as np
import pandas as pd
import polars as pl
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ────────────────────────────────────────────────────────────────
APP_TOKEN = "UDkg4uZFQ2yaBoY2bVn8zMmKj"
SEED      = 20
TARGET    = "total_ghg_emissions_metric_tons_co2e"
BID_COL   = "id"

N_OUTER  = 5
N_INNER  = 3
N_TRIALS = 60
Z89      = 1.5982   # Phi^{-1}(0.945) for 89% symmetric PI

# ── Variable classification ──────────────────────────────────────────────────
RAW_NUMERIC   = ["year_built", "of_buildings", "data_year",
                  "gross_floor_area_buildings_sq_ft"]
RAW_CAT       = "primary_property_type"
UPSTREAM_COLS = RAW_NUMERIC + [RAW_CAT]

MEDIATOR_COLS = [
    "electricity_use_kbtu",
    "natural_gas_use_kbtu",
    "district_steam_use_kbtu",
    "district_chilled_water_use_kbtu",
    "all_other_fuel_use_kbtu",
]

# Derived metrics: deterministic downstream functions of the mediators.
# The causal analyst excludes these (circular per tested DAG).
# The causally-blind analyst includes them — no reason to exclude.
DERIVED_COLS = [
    "site_eui_kbtu_sq_ft",
    "source_eui_kbtu_sq_ft",
    "weather_normalized_site_eui_kbtu_sq_ft",
    "weather_normalized_source_eui_kbtu_sq_ft",
    "ghg_intensity_kg_co2e_sq_ft",
    "energy_star_score",
    "chicago_energy_rating",
]

# Spatial and compliance metadata available to any analyst who looks.
SPATIAL_COLS     = ["zip_code", "community_area", "latitude", "longitude"]
COMPLIANCE_COLS  = ["reporting_status", "exempt_from_chicago_energy_rating"]
WATER_COLS       = ["water_use_kgal"]

# ── Causally-blind full feature set ─────────────────────────────────────────
# All features a genuinely naive analyst would use: everything with
# reasonable coverage excluding only identifiers (id, property_name,
# address, row_id, location), the target itself, and the compliant
# flag (a derived column added during preprocessing, not in the raw data).
CB_ALL_FEATURES = (
    UPSTREAM_COLS
    + SPATIAL_COLS
    + COMPLIANCE_COLS
    + MEDIATOR_COLS
    + DERIVED_COLS
    + WATER_COLS
)

# Kept for backward compatibility and causal deployment test.
MEDIATOR_FEATURE_COLS = UPSTREAM_COLS + MEDIATOR_COLS

DICT_META = {
    "data_year": pl.Int64, "id": pl.Int64,
    "property_name": pl.String, "reporting_status": pl.String,
    "address": pl.String, "zip_code": pl.String,
    "chicago_energy_rating": pl.Float64,
    "exempt_from_chicago_energy_rating": pl.Boolean,
    "community_area": pl.String, "primary_property_type": pl.String,
    "gross_floor_area_buildings_sq_ft": pl.Float64,
    "year_built": pl.Float64, "of_buildings": pl.Int64,
    "water_use_kgal": pl.Float64, "energy_star_score": pl.Int64,
    "electricity_use_kbtu": pl.Float64, "natural_gas_use_kbtu": pl.Float64,
    "district_steam_use_kbtu": pl.Float64,
    "district_chilled_water_use_kbtu": pl.Float64,
    "all_other_fuel_use_kbtu": pl.Float64,
    "site_eui_kbtu_sq_ft": pl.Float64, "source_eui_kbtu_sq_ft": pl.Float64,
    "weather_normalized_site_eui_kbtu_sq_ft": pl.Float64,
    "weather_normalized_source_eui_kbtu_sq_ft": pl.Float64,
    "total_ghg_emissions_metric_tons_co2e": pl.Float64,
    "ghg_intensity_kg_co2e_sq_ft": pl.Float64,
    "latitude": pl.Float64, "longitude": pl.Float64,
    "location": pl.String, "row_id": pl.String,
}

# ── Visual identity ──────────────────────────────────────────────────────────
# Each entry: (pipeline, approach, colour, short_label)
MODEL_META = {
    "M1 CB OLS":         ("causally-blind", "frequentist", "#4C72B0", "M1"),
    "M2 CB XGB":         ("causally-blind", "ml",          "#DD8452", "M2"),
    "M3 CB Bayes":       ("causally-blind", "bayesian",    "#C44E52", "M3"),
    "M4 Causal OLS":     ("causal",         "frequentist", "#55A868", "M4"),
    "M5 Causal XGB":     ("causal",         "ml",          "#1a7a1a", "M5"),
    "M6 Causal Bayes":   ("causal",         "bayesian",    "#8172B2", "M6"),
}
COLOURS  = {k: v[2] for k, v in MODEL_META.items()}
GROUPS   = {k: v[0] for k, v in MODEL_META.items()}
APPROACH = {k: v[1] for k, v in MODEL_META.items()}
SHORTS   = {k: v[3] for k, v in MODEL_META.items()}

CB_MODELS     = {m for m, v in MODEL_META.items() if v[0] == "causally-blind"}
CAUSAL_MODELS = {m for m, v in MODEL_META.items() if v[0] == "causal"}
BAYES_MODELS  = {"M3 CB Bayes", "M6 Causal Bayes"}

LEGEND_LABELS = {
    "M1 CB OLS":
        "M1  CB OLS       [causally-blind | frequentist | ALL features | KFold]",
    "M2 CB XGB":
        "M2  CB XGB       [causally-blind | ML          | ALL features | KFold tuned]",
    "M3 CB Bayes":
        "M3  CB Bayes     [causally-blind | Bayesian    | ALL features | PSIS-LOO]",
    "M4 Causal OLS":
        "M4  Causal OLS   [causal         | frequentist | upstream     | GroupKFold]",
    "M5 Causal XGB":
        "M5  Causal XGB   [causal         | ML          | upstream     | GroupKFold tuned]  <- ceiling",
    "M6 Causal Bayes":
        "M6  Causal Bayes [causal         | Bayesian    | upstream     | PSIS-LOO]          <- structural",
}


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def fetch_data(app_token: str) -> pl.DataFrame:
    """Paginate the Socrata REST API and return a typed Polars DataFrame."""
    base   = "https://data.cityofchicago.org/resource/xq83-jr8c.json"
    chunks, offset, limit = [], 0, 50_000
    print("Fetching Chicago Energy Benchmarking data ...")
    while True:
        r = requests.get(base, params={
            "$$app_token": app_token,
            "$limit":      limit,
            "$offset":     offset,
            "$order":      ":id",
        }, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        chunks.append(pd.DataFrame(batch))
        offset += limit
        print(f"  {offset:,} records ...")
        if len(batch) < limit:
            break
        time.sleep(0.3)

    df_pd = pd.concat(chunks, ignore_index=True)
    print(f"  Total records: {len(df_pd):,}\n")
    return (
        pl.from_pandas(df_pd)
        .select(DICT_META.keys())
        .with_columns([pl.col(c).cast(t) for c, t in DICT_META.items()])
        .drop("location")
    )


# ===========================================================================
# 2. ARVIZ COMPATIBILITY SHIMS
# ===========================================================================

def _flatten_az_result(az_result) -> np.ndarray:
    import xarray as xr
    if hasattr(az_result, "to_dataset"):
        ds = az_result.to_dataset()
    elif isinstance(az_result, xr.Dataset):
        ds = az_result
    else:
        try:
            return np.concatenate(
                [np.array(v).flatten() for v in az_result.values()])
        except Exception:
            return np.array([float("nan")])
    arrays = [np.array(ds[v].values).flatten() for v in ds.data_vars]
    return np.concatenate(arrays) if arrays else np.array([float("nan")])


def _loo_elpd(loo_result) -> float:
    for attr in ("elpd_loo", "elpd", "looic"):
        if hasattr(loo_result, attr):
            return float(np.asarray(getattr(loo_result, attr)).flat[0])
    d = getattr(loo_result, "__dict__", {})
    for key in ("elpd_loo", "elpd", "looic"):
        if key in d:
            return float(np.asarray(d[key]).flat[0])
    raise AttributeError(f"Cannot find ELPD. Available: {list(d.keys())}")


def _loo_se(loo_result) -> float:
    for attr in ("se", "elpd_loo_se", "se_elpd_loo"):
        if hasattr(loo_result, attr):
            return float(np.asarray(getattr(loo_result, attr)).flat[0])
    d = getattr(loo_result, "__dict__", {})
    for key in ("se", "elpd_loo_se", "se_elpd_loo"):
        if key in d:
            return float(np.asarray(d[key]).flat[0])
    return 0.0


def _pareto_k(loo_result) -> np.ndarray:
    for attr in ("pareto_k", "pareto_k_values", "k"):
        if hasattr(loo_result, attr):
            return np.asarray(getattr(loo_result, attr)).flatten()
    for attr in ("pointwise", "loo_i"):
        if hasattr(loo_result, attr):
            pw = getattr(loo_result, attr)
            for sub in ("pareto_k", "k"):
                if hasattr(pw, sub):
                    return np.asarray(getattr(pw, sub)).flatten()
    return np.array([])


# ===========================================================================
# 3. PREPROCESSING  (Realistic comparison — separate row subsets)
# ===========================================================================

def preprocess(df: pl.DataFrame) -> dict:
    """
    Preprocessing pipeline producing separate, realistically-sized subsets
    for the causally-blind and causal pipelines.

    Design rationale
    ----------------
    This implements the *realistic comparison* design rather than the
    controlled Option A design. Each pipeline uses the data exactly as
    its analyst would actually prepare it:

      Causally-blind analyst:
        Opens the dataset, sees ~21,714 rows where the GHG target is
        observed, and trains on all of them. Has no reason to filter on
        compliance status — understanding the compliance gate requires the
        causal domain knowledge the naive analyst lacks.

      Causal analyst:
        Knows about the compliance gate from Stage 2 of the TDGP pipeline.
        Filters to ~21,406 compliant rows and applies IPW to re-balance
        the compliant sample toward the full city building stock.

    Preprocessing steps shared by BOTH pipelines
    ---------------------------------------------
    1. Compliance flag (from ordinance rules)
    2. Upstream numeric imputation (global median, full 28,329-row dataset)
    3. Upstream categorical imputation ("Unknown" for nulls)

    Preprocessing steps specific to the causally-blind pipeline
    -----------------------------------------------------------
    4a. Row filter: all rows where target is not null (~21,714)
    4b. Mediator nulls filled with column median (naive analyst default)
    4c. Derived metric nulls filled with column median (naive default;
        the naive analyst sees these columns and includes them — they
        have no reason to recognise the circularity)
    4d. String/bool columns for spatial/compliance features coerced
    4e. KFold splits — no building ID grouping (naive analyst does not
        know the data is a longitudinal panel)

    Preprocessing steps specific to the causal pipeline
    ---------------------------------------------------
    4f. Row filter: compliant == 1 AND target is not null (~21,406)
    4g. Mediator nulls filled with structural zero (tested DAG)
    4h. GroupKFold splits on building ID (panel-aware)
    4i. Propensity model fitted on full 28,329-row dataset
    4j. IPW weights computed from propensity scores

    Context dict keys
    -----------------
    obs_cb        : CB Polars DataFrame (~21,714 rows, all imputed)
    n_cb          : number of CB training rows
    y_obs_cb      : target on original scale, aligned to obs_cb
    y_log_cb      : log1p(target), aligned to obs_cb
    bids_cb       : building IDs, aligned to obs_cb
    splits_cb     : KFold splits on obs_cb indices
    cb_medians    : dict of column medians used to impute CB features
                    (needed to fill non-compliant rows at deployment time)

    obs_causal    : causal Polars DataFrame (~21,406 rows)
    n_causal      : number of causal training rows
    y_obs_causal  : target on original scale, aligned to obs_causal
    y_log_causal  : log1p(target), aligned to obs_causal
    bids_causal   : building IDs, aligned to obs_causal
    splits_causal : GroupKFold splits on obs_causal indices
    ipw_w         : normalised IPW weights, aligned to obs_causal

    non_compliant : non-compliant rows for deployment test
    df_full       : full dataset with compliant flag and p_compliant
    """

    # ── Step 1: Compliance flag ───────────────────────────────────────────────
    print("Building compliance flag ...")
    submitted = {"Submitted", "Submitted Data"}
    df = df.with_columns(
        pl.when(
            pl.col("reporting_status").is_in(submitted) &
            (pl.col("exempt_from_chicago_energy_rating")
               .fill_null(False) == False)
        ).then(1).otherwise(0).alias("compliant")
    )

    # ── Step 2: Upstream imputation (shared by both pipelines) ────────────────
    # Both a naive and a causal analyst would impute missing building
    # characteristics before modelling. Medians are computed on the full
    # 28,329-row dataset before any filtering — domain-agnostic and
    # identical for both pipelines.
    print("Imputing upstream features (shared, global median) ...")
    for c in ["gross_floor_area_buildings_sq_ft", "year_built", "of_buildings"]:
        med = df[c].drop_nulls().median()
        df  = df.with_columns(pl.col(c).fill_null(med))
    df = df.with_columns(pl.col(RAW_CAT).fill_null("Unknown"))

    # ── Step 3a: Causally-blind row filter ───────────────────────────────────
    # The naive analyst uses all rows where the target is observed.
    # No compliance filter — they have no reason to apply one.
    obs_cb = df.filter(pl.col(TARGET).is_not_null())
    n_cb   = len(obs_cb)
    print(f"\n  Causally-blind row set   : {n_cb:,} rows "
          f"(all rows with observed target)")

    # ── Step 3b: Causally-blind feature imputation ───────────────────────────
    # The naive analyst's default: fill missing values with column median.
    # Applied to mediators, derived metrics, spatial metadata, water use.
    # Compliance metadata (reporting_status, exempt flag) is used as-is —
    # string/boolean columns that are label-encoded at model-build time.
    #
    # Medians are computed from obs_cb (all observed target rows),
    # which is the only dataset the naive analyst has access to.
    print("Filling CB features with column medians (naive analyst default) ...")
    cb_medians = {}
    for c in MEDIATOR_COLS + DERIVED_COLS + WATER_COLS:
        if c in obs_cb.columns:
            med = obs_cb[c].drop_nulls().median()
            fill_val      = float(med) if med is not None else 0.0
            cb_medians[c] = fill_val
            obs_cb        = obs_cb.with_columns(pl.col(c).fill_null(fill_val))

    y_obs_cb  = obs_cb[TARGET].to_numpy().astype(float)
    y_log_cb  = np.log1p(y_obs_cb)
    bids_cb   = obs_cb[BID_COL].to_numpy()

    # KFold: the naive analyst does not know the data is a longitudinal
    # panel with repeated building observations.
    kf        = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    splits_cb = list(kf.split(np.arange(n_cb)))

    # ── Step 4a: Causal row filter ────────────────────────────────────────────
    # The causal analyst knows the compliance gate and filters to compliant
    # rows with an observed target — a smaller, carefully selected subset.
    obs_causal_base = (
        df.filter(pl.col("compliant") == 1)
          .filter(pl.col(TARGET).is_not_null())
    )
    n_causal = len(obs_causal_base)
    print(f"  Causal row set           : {n_causal:,} rows "
          f"(compliant + observed target)")
    print(f"\n  Row count difference     : {n_cb - n_causal:,} rows — "
          f"naive analyst uses more data and still gets the wrong answer.")

    # ── Step 4b: Causal mediator handling ────────────────────────────────────
    # The tested DAG establishes that absences in district steam, district
    # chilled water, and other fuel use are structural zeros: the building is
    # physically not connected to the district energy network. Electricity and
    # natural gas nulls are also filled with zero for consistency.
    print("Applying structural zeros for causal pipeline (tested DAG) ...")
    obs_causal = obs_causal_base
    for c in MEDIATOR_COLS:
        obs_causal = obs_causal.with_columns(pl.col(c).fill_null(0.0))

    y_obs_causal  = obs_causal[TARGET].to_numpy().astype(float)
    y_log_causal  = np.log1p(y_obs_causal)
    bids_causal   = obs_causal[BID_COL].to_numpy()

    # GroupKFold: the causal analyst knows the data is a longitudinal panel.
    # Repeated building observations must be grouped to prevent leakage.
    gkf           = GroupKFold(n_splits=N_OUTER)
    splits_causal = list(gkf.split(
        np.arange(n_causal), groups=bids_causal))

    # ── Step 5: Propensity model and IPW weights (causal pipeline only) ───────
    # The propensity model estimates P(compliant=1 | upstream features).
    # Fitted on the full 28,329-row dataset to see both compliant and
    # non-compliant records. Scores are then extracted for causal rows only.
    # This step uses knowledge of the compliance gate that the causally-
    # blind analyst does not have.
    print("Fitting propensity model on full dataset ...")
    num_arr = df.select(RAW_NUMERIC).to_numpy().astype(float)
    cat_arr = df[RAW_CAT].to_numpy().reshape(-1, 1)
    ohe     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_enc = ohe.fit_transform(cat_arr)
    X_prop  = np.hstack([num_arr, cat_enc])
    y_comp  = df["compliant"].to_numpy()

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_prop)
    lr_prop = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    lr_prop.fit(X_sc, y_comp)
    p_hat_full = lr_prop.predict_proba(X_sc)[:, 1]
    acc        = (lr_prop.predict(X_sc) == y_comp).mean()
    print(f"  Propensity model accuracy : {acc:.3f}")

    # Attach propensity scores to full dataset
    df = df.with_columns(pl.Series("p_compliant", p_hat_full))

    # Extract propensity scores for causal (compliant+observed) rows only
    obs_causal_with_prop = (
        df.filter(pl.col("compliant") == 1)
          .filter(pl.col(TARGET).is_not_null())
    )
    p_obs = obs_causal_with_prop["p_compliant"].to_numpy()

    # IPW weights: 1/p(compliant), trimmed at 99th percentile, normalised
    raw_w = 1.0 / p_obs
    cap   = np.percentile(raw_w, 99)
    ipw_w = np.clip(raw_w, None, cap)
    ipw_w = ipw_w / ipw_w.mean()
    print(f"  IPW 99th-pct cap     : {cap:.3f}")
    print(f"  IPW max (normalised) : {ipw_w.max():.3f}")

    # ── Step 6: Non-compliant rows for deployment test ────────────────────────
    non_compliant = df.filter(pl.col("compliant") == 0)
    print(f"\n  Non-compliant (deployment test) : {len(non_compliant):,} rows")

    # ── Per-pipeline alignment checks ────────────────────────────────────────
    # The two pipelines have DIFFERENT row counts by design.
    # Each pipeline's arrays must be internally consistent.
    assert len(y_obs_cb)     == n_cb,     "CB target length mismatch"
    assert len(y_log_cb)     == n_cb,     "CB log-target length mismatch"
    assert len(bids_cb)      == n_cb,     "CB bids length mismatch"

    assert len(y_obs_causal) == n_causal, "Causal target length mismatch"
    assert len(y_log_causal) == n_causal, "Causal log-target length mismatch"
    assert len(bids_causal)  == n_causal, "Causal bids length mismatch"
    assert len(ipw_w)        == n_causal, "IPW weight length mismatch"

    print(f"\n  Alignment verified:")
    print(f"    CB pipeline    : {n_cb:,} rows (target, log-target, bids)")
    print(f"    Causal pipeline: {n_causal:,} rows (target, log-target, bids, IPW)")

    return dict(
        # Causally-blind pipeline
        obs_cb        = obs_cb,
        n_cb          = n_cb,
        y_obs_cb      = y_obs_cb,
        y_log_cb      = y_log_cb,
        bids_cb       = bids_cb,
        splits_cb     = splits_cb,
        cb_medians    = cb_medians,   # used at deployment to fill non-compliant
        # Causal pipeline
        obs_causal    = obs_causal,
        n_causal      = n_causal,
        y_obs_causal  = y_obs_causal,
        y_log_causal  = y_log_causal,
        bids_causal   = bids_causal,
        splits_causal = splits_causal,
        ipw_w         = ipw_w,
        # Deployment test
        non_compliant = non_compliant,
        df_full       = df,
    )


# ===========================================================================
# 4. SHARED METRIC HELPERS
# ===========================================================================

def fold_metrics(y_te_log: np.ndarray, y_te_orig: np.ndarray,
                 y_pred_log: np.ndarray) -> dict:
    """Compute per-fold evaluation metrics from log-scale predictions."""
    y_pred_orig = np.clip(np.expm1(y_pred_log), 0, None)
    sigma       = max(float(np.std(y_te_log - y_pred_log)), 1e-6)
    pi_lo       = np.clip(np.expm1(y_pred_log - Z89 * sigma), 0, None)
    pi_hi       = np.expm1(y_pred_log + Z89 * sigma)
    elpd_i      = (
        -0.5 * np.log(2 * np.pi * sigma ** 2)
        - (y_te_log - y_pred_log) ** 2 / (2 * sigma ** 2)
    )
    return dict(
        r2     = float(r2_score(y_te_log, y_pred_log)),
        mape   = float(np.median(
            np.abs((y_te_orig - y_pred_orig) / (y_te_orig + 1e-6))
        ) * 100),
        rmse   = float(np.sqrt(mean_squared_error(y_te_orig, y_pred_orig))),
        cov    = float(np.mean(
            (y_te_orig >= pi_lo) & (y_te_orig <= pi_hi)) * 100),
        sigma  = sigma,
        elpd_i = elpd_i,
    )


def aggregate_folds(fold_list: list, n_obs: int) -> dict:
    """Aggregate a list of per-fold metric dicts into a summary dict."""
    elpd_all = np.concatenate([f["elpd_i"] for f in fold_list])
    return dict(
        r2_mean     = float(np.mean([f["r2"]   for f in fold_list])),
        r2_std      = float(np.std( [f["r2"]   for f in fold_list])),
        mape_mean   = float(np.mean([f["mape"] for f in fold_list])),
        mape_std    = float(np.std( [f["mape"] for f in fold_list])),
        rmse_median = float(np.median([f["rmse"] for f in fold_list])),
        rmse_mean   = float(np.mean(  [f["rmse"] for f in fold_list])),
        cov_mean    = float(np.mean([f["cov"]  for f in fold_list])),
        elpd        = float(elpd_all.sum()),
        elpd_se     = float(np.sqrt(n_obs * np.var(elpd_all))),
    )


def grouped_holdout(bids: np.ndarray, frac: float = 0.20,
                    seed: int = 0) -> tuple:
    """Return (train_mask, val_mask) split by building ID."""
    rng   = np.random.default_rng(seed)
    uniq  = np.unique(bids)
    n_val = max(1, int(len(uniq) * frac))
    val_s = set(rng.choice(uniq, n_val, replace=False).tolist())
    mask  = np.array([b in val_s for b in bids])
    return ~mask, mask


def random_holdout(n: int, frac: float = 0.15, seed: int = 0) -> tuple:
    """Return (train_mask, val_mask) as a random split without grouping."""
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n)
    split = max(1, int(n * frac))
    val_m = np.zeros(n, dtype=bool)
    val_m[perm[:split]] = True
    return ~val_m, val_m


# ===========================================================================
# 5. FEATURE BUILDERS
# ===========================================================================

def build_ols_design(source_df: pl.DataFrame,
                     feature_cols: list) -> np.ndarray:
    """
    OLS design matrix: numeric columns + one-hot property type + intercept.

    Handles the expanded CB feature set (CB_ALL_FEATURES) as well as the
    upstream-only causal set (UPSTREAM_COLS). For CB models, string columns
    (zip_code, community_area, reporting_status) and the boolean exempt flag
    are one-hot encoded alongside property type; all other columns are
    treated as numeric with coerce-and-fill-zero.
    """
    # Identify categorical columns present in this feature set
    cat_cols = [c for c in feature_cols
                if c in source_df.columns
                and source_df[c].dtype in (pl.String, pl.Boolean, pl.Utf8)]

    num_cols = [c for c in feature_cols
                if c in source_df.columns and c not in cat_cols]

    X_num = (source_df.select(num_cols).to_pandas()
             .apply(pd.to_numeric, errors="coerce")
             .fillna(0.0).values) if num_cols else np.empty((len(source_df), 0))

    if cat_cols:
        cat_pd  = source_df.select(cat_cols).to_pandas().astype(str).fillna("__missing__")
        ohe     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_enc = ohe.fit_transform(cat_pd.values)
        X       = np.hstack([X_num, cat_enc, np.ones((len(source_df), 1))])
    else:
        X = np.hstack([X_num, np.ones((len(source_df), 1))])

    return X


def build_label_encoded_features(source_df: pl.DataFrame,
                                   feature_cols: list) -> tuple:
    """
    Label-encoded pandas DataFrame for XGBoost (no category dtype).
    Returns (X_pd, y_log, y_orig, bids) all length len(source_df).

    Handles both the expanded CB feature set (CB_ALL_FEATURES, which
    includes string/boolean columns) and the upstream-only causal set
    (UPSTREAM_COLS). String and boolean columns are label-encoded;
    numeric columns are coerced and median-filled.
    """
    df_pd         = source_df.to_pandas()
    df_pd[TARGET] = pd.to_numeric(df_pd[TARGET], errors="coerce")
    df_pd         = df_pd.dropna(subset=[TARGET]).reset_index(drop=True)
    y_orig        = df_pd[TARGET].values.astype(float)
    bids          = (pd.to_numeric(df_pd[BID_COL], errors="coerce")
                     .fillna(-1).astype(int).values)

    # Only keep columns that exist in the DataFrame
    present_cols = [c for c in feature_cols if c in df_pd.columns]
    X = df_pd[present_cols].copy()

    # Boolean -> int before dtype dispatch
    for c in X.columns:
        if X[c].dtype == bool or str(X[c].dtype) in ("bool", "boolean"):
            X[c] = X[c].astype(float).astype(int)

    # Object / string -> label encode (handles zip_code, community_area,
    # reporting_status, primary_property_type, and any other string cols)
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].fillna("__missing__")
        le   = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))

    # Numeric -> coerce and fill with median
    for c in X.select_dtypes(include=[np.number]).columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    return X, np.log1p(y_orig), y_orig, bids
