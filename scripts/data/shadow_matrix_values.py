"""Print exact shadow-matrix correlations needed for Section 4.2.

Reuses fetch_dataset() and build_compliance() from eda_dataset.py so the
numbers match Figure 4 (fig_04_shadow_matrix.png) exactly. Groups the
COMPLIANT-vs-variable correlations by SCM role and reports the within-block
co-missingness among the gated mediators/outcomes.

Run (from drafts/complete/): uv run python scripts/data/shadow_matrix_values.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_dataset import fetch_dataset, build_compliance  # noqa: E402


# Variable -> SCM role label, in the order we want to report them.
ROLE = {
    # Upstream selection
    "gross_floor_area_buildings_sq_ft": "Xs (selection)",
    "primary_property_type":            "Xs (selection)",
    # Upstream non-selection
    "year_built":                       "Xo (non-selection)",
    "of_buildings":                     "Xo (non-selection)",
    # Gated energy mediators
    "electricity_use_kbtu":             "M_obs (mediator)",
    "natural_gas_use_kbtu":             "M_obs (mediator)",
    "all_other_fuel_use_kbtu":          "M_obs (mediator)",
    # Structural-zero mediators
    "district_steam_use_kbtu":          "M_obs (struct. zero)",
    "district_chilled_water_use_kbtu":  "M_obs (struct. zero)",
    # Gated derived outcomes
    "total_ghg_emissions_metric_tons_co2e": "Y_obs (target)",
    "ghg_intensity_kg_co2e_sq_ft":      "Y_obs (derived)",
    "site_eui_kbtu_sq_ft":              "Y_obs (derived)",
    "source_eui_kbtu_sq_ft":            "Y_obs (derived)",
    "weather_normalized_site_eui_kbtu_sq_ft": "Y_obs (derived)",
    "energy_star_score":                "Y_obs (derived)",
    "chicago_energy_rating":            "Y_obs (special)",
    # Spatial metadata
    "zip_code":                         "Spatial",
    "community_area":                   "Spatial",
    "latitude":                         "Spatial",
}

# Gated block used for within-block co-missingness (true gate fields only,
# excluding structural zeros and spatial).
GATED_BLOCK = [
    "electricity_use_kbtu",
    "natural_gas_use_kbtu",
    "total_ghg_emissions_metric_tons_co2e",
    "site_eui_kbtu_sq_ft",
    "source_eui_kbtu_sq_ft",
    "ghg_intensity_kg_co2e_sq_ft",
]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    df = build_compliance(fetch_dataset())
    n = df.height

    # Observation indicator: 1 = observed, 0 = missing (matches figure convention).
    def obs(col: str) -> np.ndarray:
        return (~df[col].is_null()).cast(pl.Int8).to_numpy().astype(float)

    compliant = df["C"].to_numpy().astype(float)  # 1 = compliant, 0 = not

    print("=" * 70)
    print(f"SHADOW MATRIX -- key correlations (n = {n:,})")
    print("Indicator: 1 = observed, 0 = missing; COMPLIANT: 1 = C=1, 0 = C=0")
    print("=" * 70)
    print(f"\n{'Variable':<42}{'Role':<22}{'corr(obs, COMPLIANT)':>20}")
    print("-" * 84)
    for col, role in ROLE.items():
        if col not in df.columns:
            print(f"{col:<42}{role:<22}{'(absent from API)':>20}")
            continue
        r = corr(obs(col), compliant)
        print(f"{col:<42}{role:<22}{r:>20.3f}")

    # Within-block co-missingness among the true gated fields.
    print("\n" + "=" * 70)
    print("WITHIN-GATED-BLOCK co-missingness (corr of observation indicators)")
    print("=" * 70)
    present = [c for c in GATED_BLOCK if c in df.columns]
    M = np.column_stack([obs(c) for c in present])
    cm = np.corrcoef(M.T)
    short = [c.replace("_use_kbtu", "").replace("_kbtu_sq_ft", "_EUI")
             .replace("total_ghg_emissions_metric_tons_co2e", "GHG")
             .replace("ghg_intensity_kg_co2e_sq_ft", "GHG_int") for c in present]
    header = "".join(f"{s:>12}" for s in short)
    print(f"{'':<12}{header}")
    for i, s in enumerate(short):
        row = "".join(f"{cm[i, j]:>12.3f}" for j in range(len(short)))
        print(f"{s:<12}{row}")

    # Off-diagonal summary
    iu = np.triu_indices(len(present), k=1)
    offdiag = cm[iu]
    print(f"\nWithin-block off-diagonal correlations: "
          f"min={offdiag.min():.3f}  mean={offdiag.mean():.3f}  max={offdiag.max():.3f}")

    # Spatial / idiosyncratic check: do latitude, longitude, location form an
    # MCAR-like block (mutually co-missing but uncorrelated with everything else)?
    print("\n" + "=" * 70)
    print("SPATIAL / IDIOSYNCRATIC block check (observation-indicator corr)")
    print("=" * 70)
    spatial = [c for c in ["latitude", "longitude", "location",
                           "community_area", "zip_code"] if c in df.columns]
    refs = [c for c in ["electricity_use_kbtu",
                        "total_ghg_emissions_metric_tons_co2e",
                        "gross_floor_area_buildings_sq_ft"] if c in df.columns]
    print("\nMutual correlations among spatial fields:")
    for a in spatial:
        for b in spatial:
            if a < b:
                print(f"  {a:16s} x {b:16s} = {corr(obs(a), obs(b)):.3f}")
    print("\nSpatial vs COMPLIANT and vs substantive fields:")
    hdr = "  {:16s}{:>10}".format("field", "COMPLIANT")
    hdr += "".join(f"{r.split('_')[0]:>12}" for r in refs)
    print(hdr)
    for s in spatial:
        line = f"  {s:16s}{corr(obs(s), compliant):>10.3f}"
        line += "".join(f"{corr(obs(s), obs(r)):>12.3f}" for r in refs)
        print(line)

    print("\nDone.")


if __name__ == "__main__":
    main()
