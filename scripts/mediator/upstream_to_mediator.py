"""Check the X_s, X_o -> M associations (Condition 2 corroboration) for Section 4.4.

Fits simple OLS of each (log1p) energy mediator on the upstream features and
reports n, R^2, and the log-floor-area elasticity. Two specs per mediator:
  (A) floor area only:        log1p(M) ~ log(floor area)
  (B) full upstream:          log1p(M) ~ log(floor area) + property type
                                         + year built + no. buildings + year FE
Mediators are observed only for (largely) compliant units, so the fit uses that
subsample; conditioning on the upstream features (which drive selection) keeps
E[M | X] unbiased there. ASSOCIATION only -- direction X->M rests on physical /
temporal order, not these fits.

Reuses fetch_dataset / build_compliance from scripts/data/eda_dataset.py.
Run (from drafts/complete/): uv run python scripts/mediator/upstream_to_mediator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# eda_dataset.py lives in the bundle's scripts/data/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
from eda_dataset import fetch_dataset, build_compliance  # noqa: E402

from sklearn.linear_model import LinearRegression  # noqa: E402
from scipy import stats  # noqa: E402

UPSTREAM = [
    "gross_floor_area_buildings_sq_ft",
    "primary_property_type",
    "year_built",
    "of_buildings",
    "data_year",
]

MEDIATORS = {
    "electricity_use_kbtu": "Electricity",
    "natural_gas_use_kbtu": "Natural gas",
    "district_steam_use_kbtu": "District steam (struct. zero)",
    "district_chilled_water_use_kbtu": "District chilled water (struct. zero)",
    "all_other_fuel_use_kbtu": "Other fuel",
}


def _design(df: pd.DataFrame, include_type: bool = False,
            include_other: bool = False) -> pd.DataFrame:
    """Design matrix added in blocks: floor area (always); + property type;
    + other upstream (year built, no. buildings, data-year FE)."""
    X = pd.DataFrame(index=df.index)
    X["log_floor_area"] = np.log(df["gross_floor_area_buildings_sq_ft"].astype(float))
    if include_type:
        pt = df["primary_property_type"].astype(str)
        counts = pt.value_counts()
        rare = counts[counts < 100].index
        pt = pt.where(~pt.isin(rare), "Other")
        X = pd.concat([X, pd.get_dummies(pt, prefix="pt", drop_first=True)], axis=1)
    if include_other:
        X["year_built"] = df["year_built"].astype(float)
        X["of_buildings"] = df["of_buildings"].astype(float)
        X = pd.concat([X, pd.get_dummies(df["data_year"].astype(str), prefix="yr",
                                         drop_first=True)], axis=1)
    return X.astype(float)


def _fit(df: pd.DataFrame, col: str, include_type: bool = False,
         include_other: bool = False):
    y = np.log1p(df[col].astype(float).values)
    X = _design(df, include_type, include_other)
    m = LinearRegression().fit(X.values, y)
    r2 = m.score(X.values, y)
    beta = m.coef_[list(X.columns).index("log_floor_area")]
    return r2, beta


def _adj_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R^2; p = number of regressors (excluding the intercept)."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def _design_parts(df: pd.DataFrame, include_type: bool, include_yb_nb: bool,
                  include_year: bool) -> pd.DataFrame:
    """Design matrix with each upstream block toggled independently, so the
    data-year fixed effects can be isolated from year built / no. buildings."""
    X = pd.DataFrame(index=df.index)
    X["log_floor_area"] = np.log(df["gross_floor_area_buildings_sq_ft"].astype(float))
    if include_type:
        pt = df["primary_property_type"].astype(str)
        counts = pt.value_counts()
        rare = counts[counts < 100].index
        pt = pt.where(~pt.isin(rare), "Other")
        X = pd.concat([X, pd.get_dummies(pt, prefix="pt", drop_first=True)], axis=1)
    if include_yb_nb:
        X["year_built"] = df["year_built"].astype(float)
        X["of_buildings"] = df["of_buildings"].astype(float)
    if include_year:
        X = pd.concat([X, pd.get_dummies(df["data_year"].astype(str), prefix="yr",
                                         drop_first=True)], axis=1)
    return X.astype(float)


def datayear_marginal(pdf: pd.DataFrame) -> None:
    """Isolate the marginal contribution of the data-year fixed effects to each
    mediator. Base model = area + type + year built + no. buildings; full model
    adds data-year FE only. Reports R^2, ADJUSTED R^2, and a partial F-test on
    the year block (does data_year earn its parameters once penalised?)."""
    print("\n" + "=" * 96)
    print("Marginal contribution of data_year to M   (base = area + type + built + nbldg)")
    print("full = base + data-year fixed effects only;  adjR2 penalises the added year dummies")
    print("=" * 96)
    print(f"{'Mediator':<26}{'n':>8}{'R2_base':>9}{'R2_full':>9}"
          f"{'aR2_base':>10}{'aR2_full':>10}{'d_aR2':>9}{'F_year':>9}{'p_val':>10}")
    print("-" * 96)

    targets = list(MEDIATORS.items()) + [("_total_energy", "TOTAL metered energy")]
    for col, label in targets:
        if col not in pdf.columns:
            continue
        keep = UPSTREAM + (["electricity_use_kbtu"] if col == "_total_energy" else [col])
        sub = pdf.dropna(subset=keep).copy()
        sub = sub[sub["gross_floor_area_buildings_sq_ft"].astype(float) > 0]
        tvals = sub[col].astype(float)
        sub = sub[np.isfinite(tvals) & (tvals >= 0)]
        n = len(sub)
        if n < 50:
            continue
        y = np.log1p(sub[col].astype(float).values)
        Xb = _design_parts(sub, True, True, False)
        Xf = _design_parts(sub, True, True, True)
        r2b = LinearRegression().fit(Xb.values, y).score(Xb.values, y)
        r2f = LinearRegression().fit(Xf.values, y).score(Xf.values, y)
        pb, pf = Xb.shape[1], Xf.shape[1]
        q = pf - pb
        adjb, adjf = _adj_r2(r2b, n, pb), _adj_r2(r2f, n, pf)
        denom = (1.0 - r2f) / (n - pf - 1)
        Fval = ((r2f - r2b) / q) / denom if denom > 0 and q > 0 else float("nan")
        pval = float(stats.f.sf(Fval, q, n - pf - 1)) if np.isfinite(Fval) else float("nan")
        print(f"{label:<26}{n:>8,}{r2b:>9.3f}{r2f:>9.3f}{adjb:>10.3f}{adjf:>10.3f}"
              f"{adjf - adjb:>9.4f}{Fval:>9.1f}{pval:>10.2e}")


def main() -> None:
    pdf = build_compliance(fetch_dataset()).to_pandas()

    print("=" * 78)
    print("X_s, X_o -> M  association check   (log1p mediator ~ upstream)")
    print("Sample: rows with the mediator observed AND complete upstream, floor area > 0")
    print("=" * 78)
    print(f"{'Mediator':<34}{'n':>8}{'R2(area)':>10}{'R2(+type)':>11}{'R2(full)':>10}{'beta_area':>11}")
    print("-" * 84)

    five = list(MEDIATORS.keys())
    present_five = [c for c in five if c in pdf.columns]
    pdf["_total_energy"] = pdf[present_five].apply(
        lambda r: np.nansum(r.values.astype(float)), axis=1
    )

    targets = list(MEDIATORS.items()) + [("_total_energy", "TOTAL metered energy")]
    for col, label in targets:
        if col not in pdf.columns:
            print(f"{label:<40}{'absent from API':>38}")
            continue
        keep = UPSTREAM + (["electricity_use_kbtu"] if col == "_total_energy" else [col])
        sub = pdf.dropna(subset=keep).copy()
        sub = sub[sub["gross_floor_area_buildings_sq_ft"].astype(float) > 0]
        # Drop non-finite / negative mediator values (data errors; log1p needs >= 0).
        tcol = "_total_energy" if col == "_total_energy" else col
        tvals = sub[tcol].astype(float)
        sub = sub[np.isfinite(tvals) & (tvals >= 0)]
        if len(sub) < 50:
            print(f"{label:<40}{len(sub):>8,}{'(n too small)':>30}")
            continue
        r2a, _ = _fit(sub, col)
        r2t, _ = _fit(sub, col, include_type=True)
        r2f, beta = _fit(sub, col, include_type=True, include_other=True)
        print(f"{label:<34}{len(sub):>8,}{r2a:>10.3f}{r2t:>11.3f}{r2f:>10.3f}{beta:>11.3f}")

    datayear_marginal(pdf)

    print("\nNotes:")
    print(" - beta_area = elasticity of log1p(mediator) wrt log(floor area), full model.")
    print(" - district steam / chilled water are structural zeros (mostly 0); their")
    print("   fits reflect presence+magnitude mixed and are not the headline links.")
    print(" - Association only; X->M direction rests on physical/temporal order.")
    print("\nDone.")


if __name__ == "__main__":
    main()
