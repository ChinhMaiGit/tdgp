"""M -> Y reconstruction check (mediator contribution to total GHG) for Section 4.4.

Total GHG = sum over fuels of (energy x emission factor), so it should be a
near-exact LINEAR function of the energy mediators. This fits GHG ~ mediators
(linear -- the EPA accounting is additive, NOT multiplicative, so no logs) and
reports: R^2 (reconstruction quality), the recovered effective emission factors
(kg CO2e per MBtu, for comparison with EPA published factors in Stage 3), each
fuel's share of total emissions, and electricity-alone R^2.

Mediator nulls are filled with 0 (unreported / off-network structural zero =
no contribution). Reuses fetch_dataset / build_compliance from scripts/data.
Run (from drafts/complete/): uv run python scripts/mediator/mediator_to_outcome.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
from eda_dataset import fetch_dataset, build_compliance  # noqa: E402

from sklearn.linear_model import LinearRegression  # noqa: E402

Y = "total_ghg_emissions_metric_tons_co2e"
MED = {
    "electricity_use_kbtu": "Electricity",
    "natural_gas_use_kbtu": "Natural gas",
    "district_steam_use_kbtu": "District steam",
    "district_chilled_water_use_kbtu": "District chilled water",
    "all_other_fuel_use_kbtu": "Other fuel",
}


def main() -> None:
    pdf = build_compliance(fetch_dataset()).to_pandas()
    meds = [c for c in MED if c in pdf.columns]

    sub = pdf.dropna(subset=[Y]).copy()
    sub = sub[sub[Y].astype(float) >= 0]
    Xm = sub[meds].astype(float).fillna(0.0)
    y = sub[Y].astype(float).values
    mask = (Xm >= 0).all(axis=1) & np.isfinite(y)
    Xm, y = Xm[mask], y[mask]

    print("=" * 70)
    print("M -> Y reconstruction:  total GHG (t CO2e) ~ energy mediators (kBtu)")
    print(f"n = {len(y):,}   (rows with GHG observed; mediator nulls -> 0)")
    print("=" * 70)

    for fit_int in (True, False):
        m = LinearRegression(fit_intercept=fit_int).fit(Xm.values, y)
        r2 = m.score(Xm.values, y)
        print(f"\nLinear, intercept={fit_int}:  R^2 = {r2:.5f}"
              + (f"   intercept = {m.intercept_:.3f} t" if fit_int else ""))

    # Recovered effective emission factors (no-intercept = physical model).
    m0 = LinearRegression(fit_intercept=False).fit(Xm.values, y)
    print("\nRecovered effective emission factors (no-intercept model):")
    print(f"   {'Fuel':<24}{'kg CO2e / MBtu':>16}{'coef (t/kBtu)':>16}")
    for c, coef in zip(meds, m0.coef_):
        print(f"   {MED[c]:<24}{coef * 1e6:>16.2f}{coef:>16.3e}")

    # Share of total emissions by fuel.
    contrib = (Xm.values * m0.coef_).sum(axis=0)
    tot = y.sum()
    print("\nShare of total emissions by fuel (no-intercept model):")
    for c, ct in zip(meds, contrib):
        print(f"   {MED[c]:<24}{100 * ct / tot:>6.1f}%")

    # Electricity alone.
    e = "electricity_use_kbtu"
    if e in meds:
        me = LinearRegression(fit_intercept=False).fit(Xm[[e]].values, y)
        print(f"\nElectricity alone (no-intercept):  R^2 = {me.score(Xm[[e]].values, y):.5f}")
    # Electricity + natural gas.
    eg = [c for c in ["electricity_use_kbtu", "natural_gas_use_kbtu"] if c in meds]
    if len(eg) == 2:
        meg = LinearRegression(fit_intercept=False).fit(Xm[eg].values, y)
        print(f"Electricity + natural gas (no-intercept):  R^2 = {meg.score(Xm[eg].values, y):.5f}")

    # Year-specific electricity factor (eGRID changes yearly); other fuels constant.
    if e in meds:
        yr = pd.get_dummies(sub.loc[Xm.index, "data_year"].astype(str), prefix="y")
        elec_by_year = yr.values * Xm[e].values[:, None]
        others = [c for c in meds if c != e]
        Xyr = np.hstack([elec_by_year, Xm[others].values])
        myr = LinearRegression(fit_intercept=False).fit(Xyr, y)
        print(f"\nYear-specific electricity factor + constant other fuels (no-intercept):")
        print(f"   R^2 = {myr.score(Xyr, y):.5f}")
        print("   recovered electricity factor by year (kg CO2e / MBtu):")
        for yname, coef in zip(yr.columns, myr.coef_[:yr.shape[1]]):
            print(f"      {yname[2:]}: {coef * 1e6:7.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
