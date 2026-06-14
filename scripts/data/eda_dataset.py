"""
eda_dataset.py
==============
Fetch the Chicago Energy Benchmarking dataset from the City of Chicago
Socrata API, produce a comprehensive EDA text report, and save diagnostic
plots to results/data/.

Run from drafts/complete/:
    uv run python scripts/data/eda_dataset.py

Redirect text output to a file if needed:
    uv run python scripts/data/eda_dataset.py > results/docs/data/eda_report.txt

Plots are saved to results/data/ regardless of where the script is run from.
"""

import time
import requests
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

matplotlib.use("Agg")   # non-interactive backend -- safe for scripts

# Global type-size settings (Storytelling with Data: small, clean text)
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "normal",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Outputs are routed to the bundle's results/data/ directory.
FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Frozen local data snapshot (the exact vintage used in the paper). When
# present it is used instead of the live Socrata API so results reproduce
# exactly; the API is only a fallback. See data/SNAPSHOT.txt for provenance.
DATA_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "data_full.parquet"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_ENDPOINT = "https://data.cityofchicago.org/resource/xq83-jr8c.json"
APP_TOKEN    = "UDkg4uZFQ2yaBoY2bVn8zMmKj"
PAGE_SIZE    = 1000

SCHEMA: dict = {
    "data_year":                                    pl.Int64,
    "id":                                           pl.Int64,
    "property_name":                                pl.String,
    "reporting_status":                             pl.String,
    "address":                                      pl.String,
    "zip_code":                                     pl.String,
    "chicago_energy_rating":                        pl.Float64,
    "exempt_from_chicago_energy_rating":            pl.Boolean,
    "community_area":                               pl.String,
    "primary_property_type":                        pl.String,
    "gross_floor_area_buildings_sq_ft":             pl.Float64,
    "year_built":                                   pl.Float64,
    "of_buildings":                                 pl.Int64,
    "water_use_kgal":                               pl.Float64,
    "energy_star_score":                            pl.Int64,
    "electricity_use_kbtu":                         pl.Float64,
    "natural_gas_use_kbtu":                         pl.Float64,
    "district_steam_use_kbtu":                      pl.Float64,
    "district_chilled_water_use_kbtu":              pl.Float64,
    "all_other_fuel_use_kbtu":                      pl.Float64,
    "site_eui_kbtu_sq_ft":                         pl.Float64,
    "source_eui_kbtu_sq_ft":                       pl.Float64,
    "weather_normalized_site_eui_kbtu_sq_ft":      pl.Float64,
    "weather_normalized_source_eui_kbtu_sq_ft":    pl.Float64,
    "total_ghg_emissions_metric_tons_co2e":        pl.Float64,
    "ghg_intensity_kg_co2e_sq_ft":                 pl.Float64,
    "latitude":                                    pl.Float64,
    "longitude":                                   pl.Float64,
    "location":                                    pl.String,
    "row_id":                                      pl.String,
}

# SCM role labels and colours for plots
SCM_ROLES: dict = {
    "data_year":                                "Xo (upstream)",
    "id":                                       "Admin",
    "property_name":                            "Admin",
    "address":                                  "Admin",
    "zip_code":                                 "Spatial (CB only)",
    "community_area":                           "Spatial (CB only)",
    "location":                                 "Admin",
    "row_id":                                   "Admin",
    "latitude":                                 "Spatial (CB only)",
    "longitude":                                "Spatial (CB only)",
    "reporting_status":                         "Gate (raw)",
    "exempt_from_chicago_energy_rating":        "Gate (raw)",
    "gross_floor_area_buildings_sq_ft":         "Xs",
    "primary_property_type":                    "Xs",
    "year_built":                               "Xo",
    "of_buildings":                             "Xo",
    "water_use_kgal":                           "Xo (partial)",
    "electricity_use_kbtu":                     "M_obs",
    "natural_gas_use_kbtu":                     "M_obs",
    "district_steam_use_kbtu":                  "M_obs (struct. zero)",
    "district_chilled_water_use_kbtu":          "M_obs (struct. zero)",
    "all_other_fuel_use_kbtu":                  "M_obs",
    "energy_star_score":                        "Y_obs (derived; also CB feature)",
    "site_eui_kbtu_sq_ft":                     "Y_obs (derived)",
    "source_eui_kbtu_sq_ft":                   "Y_obs (derived)",
    "weather_normalized_site_eui_kbtu_sq_ft":  "Y_obs (derived, WN)",
    "weather_normalized_source_eui_kbtu_sq_ft":"Y_obs (derived, WN)",
    "total_ghg_emissions_metric_tons_co2e":    "Y_obs (target)",
    "ghg_intensity_kg_co2e_sq_ft":             "Y_obs (derived)",
    "chicago_energy_rating":                    "Y_obs (special; also CB feature)",
}

ROLE_COLORS: dict = {
    "Admin":               "#aaaaaa",
    "Admin / Xo":          "#cccccc",
    "Gate (raw)":          "#e69f00",
    "Xs":                  "#0072b2",
    "Xo":                  "#56b4e9",
    "Xo (partial)":        "#88ccee",
    "M_obs":               "#009e73",
    "M_obs (struct. zero)":"#66c2a5",
    "Y_obs (target)":      "#d55e00",
    "Y_obs (derived)":     "#cc79a7",
    "Y_obs (derived, WN)": "#ddaacc",
    "Y_obs (special)":     "#e07070",
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_dataset() -> pl.DataFrame:
    if DATA_SNAPSHOT.exists():
        print(f"Loading frozen data snapshot: {DATA_SNAPSHOT}")
        return pl.read_parquet(DATA_SNAPSHOT)

    print("Fetching Chicago Energy Benchmarking data from Socrata API...")
    print("  Endpoint : " + API_ENDPOINT)

    all_records: list = []
    offset = 0

    while True:
        params = {
            "$limit":      PAGE_SIZE,
            "$offset":     offset,
            "$order":      ":id",          # deterministic ordering (matches experiment)
            "$$app_token": APP_TOKEN,
        }
        resp = requests.get(API_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break

        all_records.extend(page)
        print(f"  Page {offset // PAGE_SIZE + 1}: {len(page)} records "
              f"(total: {len(all_records):,})")

        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)

    print(f"\nFetch complete -- {len(all_records):,} records.\n")

    df = pl.DataFrame(all_records)
    known = [c for c in SCHEMA if c in df.columns]
    df = df.select(known)
    df = df.with_columns([
        pl.col(col).cast(dtype, strict=False)
        for col, dtype in SCHEMA.items() if col in df.columns
    ])
    return df


def build_compliance(df: pl.DataFrame) -> pl.DataFrame:
    """Append the compliance indicator C to the dataframe."""
    return df.with_columns(
        pl.when(
            pl.col("reporting_status").is_in(["Submitted", "Submitted Data"])
            & (
                pl.col("exempt_from_chicago_energy_rating").is_null()
                | (pl.col("exempt_from_chicago_energy_rating") == False)
            )
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("C")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title: str = "", width: int = 72) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'-' * pad} {title} {'-' * (width - pad - len(title) - 2)}")
    else:
        print(f"\n{'-' * width}")


def pct(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f}%)"


def num_summary(series: pl.Series) -> str:
    v = series.drop_nulls()
    if len(v) == 0:
        return "no non-null values"
    return (
        f"min={v.min():,.0f}  p25={v.quantile(0.25):,.0f}  "
        f"median={v.median():,.0f}  p75={v.quantile(0.75):,.0f}  "
        f"max={v.max():,.0f}  mean={v.mean():,.1f}"
    )


def savefig(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Style helper  (Storytelling with Data principles)
# ---------------------------------------------------------------------------

# Restrained two-tone palette
C_BLUE   = "#2166ac"   # compliant / positive
C_RED    = "#d73027"   # non-compliant / zero-stars
C_GRAY   = "#bbbbbb"   # null / pre-2018
C_MID    = "#4393c3"   # single-series default
C_DARK   = "#333333"   # annotation lines and text


def _clean(ax: plt.Axes,
           hide_yaxis: bool = True,
           hide_xaxis: bool = False,
           hide_left_spine: bool = True) -> None:
    """
    Apply Storytelling with Data declutter to a matplotlib Axes.
    - Removes grid, top/right spines, tick marks.
    - Optionally hides y-axis or x-axis completely (when values go on bars).
    - Keeps bottom spine for orientation unless hide_xaxis is True.
    """
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left_spine:
        ax.spines["left"].set_visible(False)
    if hide_yaxis:
        ax.yaxis.set_visible(False)
    else:
        ax.tick_params(left=False)
    if hide_xaxis:
        ax.xaxis.set_visible(False)
    else:
        ax.tick_params(bottom=False)
        ax.spines["bottom"].set_color("#cccccc")


def _label_bars_h(ax: plt.Axes, bars, fmt: str = "{:,.0f}",
                  pad: int = 4, color: str = C_DARK) -> None:
    """Place value labels at the right end of horizontal bars."""
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(
                w + pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va="center", ha="left",
                fontsize=8, color=color,
            )


def _label_bars_v(ax: plt.Axes, bars, fmt: str = "{:,.0f}",
                  pad: int = 4, color: str = C_DARK) -> None:
    """Place value labels above vertical bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + pad,
                fmt.format(h), va="bottom", ha="center",
                fontsize=8, color=color,
            )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_records_per_year(df: pl.DataFrame) -> None:
    """Fig 01 -- Records per year, stacked compliant / non-compliant."""
    by = (
        df.group_by(["data_year", "C"])
        .agg(pl.len().alias("n"))
        .sort(["data_year", "C"])
    )
    years = sorted(df["data_year"].unique().to_list())
    c0 = {r["data_year"]: r["n"] for r in by.filter(pl.col("C") == 0).iter_rows(named=True)}
    c1 = {r["data_year"]: r["n"] for r in by.filter(pl.col("C") == 1).iter_rows(named=True)}
    n0 = [c0.get(y, 0) for y in years]
    n1 = [c1.get(y, 0) for y in years]
    totals = [a + b for a, b in zip(n0, n1)]

    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(years, n1, color=C_BLUE,  label="Compliant")
    b0 = ax.bar(years, n0, bottom=n1, color=C_RED, label="Non-compliant")

    # Total label above each bar
    for yr, tot in zip(years, totals):
        ax.text(yr, tot + 15, f"{tot:,}", ha="center", va="bottom",
                fontsize=7.5, color=C_DARK)

    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8)
    ax.set_title("Records per reporting year", pad=12)

    # Legend in upper left -- bars are shortest there (2014 had few records)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handles=[b1, b0])

    # Add headroom so top labels aren't clipped
    ax.set_ylim(0, max(totals) * 1.14)

    _clean(ax, hide_yaxis=True, hide_left_spine=True)
    fig.tight_layout()
    savefig(fig, "fig_01_records_per_year.png")


def plot_ghg_distribution(df: pl.DataFrame) -> None:
    """Fig 02 -- GHG distribution on raw and log1p scale."""
    ghg     = df["total_ghg_emissions_metric_tons_co2e"].drop_nulls().to_numpy()
    log_ghg = np.log1p(ghg)
    med_raw = float(np.median(ghg))
    med_log = float(np.median(log_ghg))
    n       = len(ghg)

    fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(11, 4))

    # -- raw scale --
    ax_raw.hist(ghg, bins=80, color=C_MID, edgecolor="none")
    ax_raw.axvline(med_raw, color=C_DARK, linestyle="--", linewidth=1.2)
    ax_raw.text(med_raw * 1.04, ax_raw.get_ylim()[1] * 0.90,
                f"Median\n{med_raw:,.0f} t",
                fontsize=7.5, color=C_DARK, va="top")
    ax_raw.set_xlabel("GHG (metric tons CO₂e)", fontsize=8)
    ax_raw.set_title("Raw scale  -- right-skewed")
    _clean(ax_raw, hide_yaxis=True, hide_left_spine=True)

    # -- log1p scale --
    ax_log.hist(log_ghg, bins=60, color=C_MID, edgecolor="none")
    ax_log.axvline(med_log, color=C_DARK, linestyle="--", linewidth=1.2)
    ax_log.text(med_log + 0.08, ax_log.get_ylim()[1] * 0.90,
                f"Median\n{med_log:.2f}",
                fontsize=7.5, color=C_DARK, va="top")
    ax_log.set_xlabel("log₁₊₁(GHG)", fontsize=8)
    ax_log.set_title("log₁₊₁ scale  -- approximately Normal")
    _clean(ax_log, hide_yaxis=True, hide_left_spine=True)

    fig.suptitle(
        f"GHG emission distribution  --  {n:,} observed records (76.6% of total)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig_02_ghg_distribution.png")


def plot_missingness_by_variable(df: pl.DataFrame) -> None:
    """Fig 03 -- Null rate per variable, coloured by SCM role."""
    cols  = [c for c in df.columns if c != "C"]
    N     = len(df)
    rates = [100 * int(df[c].is_null().sum()) / N for c in cols]
    roles = [SCM_ROLES.get(c, "Admin") for c in cols]
    clrs  = [ROLE_COLORS.get(r, "#888888") for r in roles]

    # Sort descending so the most-missing variables appear at the top
    order  = sorted(range(len(cols)), key=lambda i: rates[i], reverse=True)
    cols_s = [cols[i]  for i in order]
    rats_s = [rates[i] for i in order]
    clrs_s = [clrs[i]  for i in order]

    fig, ax = plt.subplots(figsize=(9, 9))
    bars = ax.barh(range(len(cols_s)), rats_s, color=clrs_s,
                   edgecolor="none", height=0.7)

    # Labels directly on bars (only for non-zero rates)
    for bar, rate in zip(bars, rats_s):
        if rate > 0.2:
            ax.text(
                bar.get_width() + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%",
                va="center", ha="left", fontsize=7.5, color=C_DARK,
            )

    # Threshold annotation line
    ax.axvline(23.4, color=C_DARK, linestyle="--", linewidth=0.8)
    ax.text(23.8, len(cols_s) - 0.5, "23.4% gate",
            fontsize=7.5, color=C_DARK, va="top")

    # y-axis: variable names (keep left spine for reference)
    ax.set_yticks(range(len(cols_s)))
    ax.set_yticklabels(cols_s, fontsize=7)
    ax.invert_yaxis()

    ax.set_title("Missing-value rate by variable  (coloured by SCM role)", pad=10)

    # Role legend -- place below the chart where there's always space
    legend_handles = [
        mpatches.Patch(color=c, label=r)
        for r, c in ROLE_COLORS.items()
        if any(SCM_ROLES.get(col, "Admin") == r for col in cols)
    ]
    ax.legend(handles=legend_handles, fontsize=7, frameon=False,
              loc="lower right", ncol=2)

    # Hide x-axis values; label is on bars
    ax.xaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(left=False)
    ax.set_xlim(0, max(rats_s) * 1.18)   # headroom for labels

    fig.tight_layout()
    savefig(fig, "fig_03_missingness_by_variable.png")


def plot_shadow_matrix(df: pl.DataFrame) -> None:
    """Fig 04 -- Shadow matrix: missingness-indicator correlations (no colorbar)."""
    # Use only variables with some missingness; add COMPLIANT to show gate alignment
    cols_miss = [
        c for c in df.columns
        if c != "C" and int(df[c].is_null().sum()) > 0
    ]

    # Build indicator matrix  (1 = missing, 0 = observed)
    mat = np.column_stack([
        df[c].is_null().cast(pl.Int8).to_numpy() for c in cols_miss
    ] + [
        (1 - df["C"].to_numpy())   # COMPLIANT as observed=1 / missing=0 inverted
    ]).astype(float)

    labels_raw = cols_miss + ["COMPLIANT"]
    corr = np.corrcoef(mat.T)

    # Short readable labels
    def shorten(c: str) -> str:
        return (c
                .replace("_use_kbtu", "")
                .replace("_kbtu_sq_ft", "_EUI")
                .replace("total_ghg_emissions_metric_tons_co2e", "GHG")
                .replace("ghg_intensity_kg_co2e_sq_ft", "GHG_int")
                .replace("weather_normalized_", "WN_")
                .replace("gross_floor_area_buildings_sq_ft", "floor_area")
                .replace("chicago_energy_rating", "CER")
                .replace("energy_star_score", "ES_score")
                .replace("exempt_from_chicago_energy_rating", "exempt_CER")
                .replace("source_eui_EUI", "source_EUI")
                .replace("site_eui_EUI", "site_EUI"))

    short = [shorten(c) for c in labels_raw]
    n = len(short)
    sz = max(9, n * 0.52)

    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

    # Cell text annotations (no colorbar -- matches workbook coloraxis_showscale=False)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            txt_color = "white" if abs(val) > 0.65 else C_DARK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)

    # Remove all spines -- the cell grid is the frame
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(bottom=False, left=False)

    ax.set_title(
        "Shadow matrix  --  Pearson correlation between missingness indicators\n"
        "1 = observed, 0 = missing.  Block structure = atomic compliance gate.",
        fontsize=9, pad=10,
    )
    fig.tight_layout()
    savefig(fig, "fig_04_shadow_matrix.png")


def plot_floor_area_by_compliance(df: pl.DataFrame) -> None:
    """Fig 05 -- Log floor area: compliant vs. non-compliant."""
    fa_c1 = np.log(
        df.filter(pl.col("C") == 1)["gross_floor_area_buildings_sq_ft"]
        .drop_nulls().to_numpy()
    )
    fa_c0 = np.log(
        df.filter(pl.col("C") == 0)["gross_floor_area_buildings_sq_ft"]
        .drop_nulls().to_numpy()
    )
    bins = np.linspace(
        min(fa_c1.min(), fa_c0.min()),
        max(fa_c1.max(), fa_c0.max()), 55
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(fa_c1, bins=bins, alpha=0.65, color=C_BLUE,  density=True, label=None)
    ax.hist(fa_c0, bins=bins, alpha=0.65, color=C_RED,   density=True, label=None)

    # Direct labels on the distributions (positioned at each peak)
    c1_peak_x = bins[np.histogram(fa_c1, bins=bins, density=True)[0].argmax()]
    c0_peak_x = bins[np.histogram(fa_c0, bins=bins, density=True)[0].argmax()]
    c1_peak_y = np.histogram(fa_c1, bins=bins, density=True)[0].max()
    c0_peak_y = np.histogram(fa_c0, bins=bins, density=True)[0].max()

    ax.text(c1_peak_x, c1_peak_y * 1.06,
            f"Compliant  n={len(fa_c1):,}", ha="center", fontsize=8,
            color=C_BLUE, fontweight="bold")
    ax.text(c0_peak_x, c0_peak_y * 1.06,
            f"Non-compliant  n={len(fa_c0):,}", ha="center", fontsize=8,
            color=C_RED, fontweight="bold")

    ax.set_xlabel("log(gross floor area, sq ft)", fontsize=8)
    ax.set_title("Floor area distribution by compliance status", pad=10)

    _clean(ax, hide_yaxis=True, hide_left_spine=True)
    ax.set_ylim(0, max(c1_peak_y, c0_peak_y) * 1.22)
    fig.tight_layout()
    savefig(fig, "fig_05_floor_area_by_compliance.png")


def plot_chicago_energy_rating_by_year(df: pl.DataFrame) -> None:
    """Fig 06 -- Chicago Energy Rating breakdown per year."""
    by = (
        df.group_by("data_year")
        .agg(
            pl.col("chicago_energy_rating").is_null().sum().alias("null_n"),
            (pl.col("chicago_energy_rating") == 0).sum().alias("zero_n"),
            (pl.col("chicago_energy_rating") > 0).sum().alias("pos_n"),
        )
        .sort("data_year")
    )
    years  = by["data_year"].to_list()
    nulls  = by["null_n"].to_list()
    zeros  = by["zero_n"].to_list()
    pos    = by["pos_n"].to_list()
    totals = [a + b + c for a, b, c in zip(nulls, zeros, pos)]
    bot1   = nulls
    bot2   = [a + b for a, b in zip(nulls, zeros)]

    fig, ax = plt.subplots(figsize=(9, 4))
    b_null = ax.bar(years, nulls, color=C_GRAY,  label="Blank  (pre-2018 / exempt)")
    b_zero = ax.bar(years, zeros, bottom=bot1,  color=C_RED,  label="0 stars  (non-compliant)")
    b_pos  = ax.bar(years, pos,   bottom=bot2,  color=C_BLUE, label="1-4 stars  (compliant)")

    # Total label at top of each bar
    for yr, tot in zip(years, totals):
        ax.text(yr, tot + 12, f"{tot:,}", ha="center", va="bottom",
                fontsize=7.5, color=C_DARK)

    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8)
    ax.set_title("Chicago Energy Rating composition by reporting year", pad=12)
    ax.set_ylim(0, max(totals) * 1.12)

    # Legend at upper left -- bars are shortest in 2014/2015
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handles=[b_null, b_zero, b_pos])

    _clean(ax, hide_yaxis=True, hide_left_spine=True)
    fig.tight_layout()
    savefig(fig, "fig_06_chicago_energy_rating_by_year.png")


def plot_property_type_frequency(df: pl.DataFrame) -> None:
    """Fig 07 -- Top 20 property types by record count (no x-axis)."""
    top = (
        df.group_by("primary_property_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(20)
    )
    types  = top["primary_property_type"].to_list()
    counts = top["n"].to_list()
    max_n  = max(counts)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(types)), counts, color=C_MID,
                   edgecolor="none", height=0.65)

    # Count labels at bar ends
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_width() + max_n * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}", va="center", ha="left",
            fontsize=8, color=C_DARK,
        )

    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Top 20 property types by record count", pad=10)
    ax.set_xlim(0, max_n * 1.15)

    # No x-axis at all -- counts are on the bars
    ax.xaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(left=False)

    fig.tight_layout()
    savefig(fig, "fig_07_property_type_frequency.png")


def save_all_plots(df: pl.DataFrame) -> None:
    sep("PLOTS  (saving to " + str(FIGURES_DIR) + ")")
    plot_records_per_year(df)
    plot_ghg_distribution(df)
    plot_missingness_by_variable(df)
    plot_shadow_matrix(df)
    plot_floor_area_by_compliance(df)
    plot_chicago_energy_rating_by_year(df)
    plot_property_type_frequency(df)


# ---------------------------------------------------------------------------
# EDA text report
# ---------------------------------------------------------------------------

def run_eda(df: pl.DataFrame) -> None:
    N = len(df)
    print("Chicago Energy Benchmarking -- EDA Report")
    print("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total rows fetched: {N:,}")

    sep("1. DATASET OVERVIEW")
    n_buildings = df["id"].n_unique()
    year_min = df["data_year"].min()
    year_max = df["data_year"].max()
    n_cols   = len(df.columns)
    print(f"  Total building-year records : {N:,}")
    print(f"  Unique building IDs         : {n_buildings:,}")
    print(f"  Reporting years             : {year_min}-{year_max} ({year_max - year_min + 1} years)")
    print(f"  Columns in fetched data     : {n_cols}")
    print("\n  Records per data_year:")
    for row in df.group_by("data_year").agg(pl.len().alias("n")).sort("data_year").iter_rows(named=True):
        print(f"    {row['data_year']}: {row['n']:,}")

    sep("2. COMPLIANCE INDICATOR C")
    print("  Rule: C=1  iff  reporting_status IN {'Submitted','Submitted Data'}")
    print("               AND exempt_from_chicago_energy_rating IS NOT TRUE")
    df = build_compliance(df)
    n_c1 = int(df["C"].sum())
    n_c0 = N - n_c1
    print(f"\n  C=1 (compliant)     : {pct(n_c1, N)}")
    print(f"  C=0 (non-compliant) : {pct(n_c0, N)}")
    print("\n  reporting_status value counts:")
    for row in (
        df.group_by("reporting_status")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .iter_rows(named=True)
    ):
        print(f"    {str(row['reporting_status']):<40} {row['n']:,}")

    sep("3. TARGET: total_ghg_emissions_metric_tons_co2e")
    ghg    = df["total_ghg_emissions_metric_tons_co2e"]
    n_obs  = ghg.drop_nulls().len()
    n_miss = int(ghg.is_null().sum())
    print(f"  Observed : {pct(n_obs, N)}")
    print(f"  Missing  : {pct(n_miss, N)}")
    print(f"  Distribution of observed values:")
    print(f"    {num_summary(ghg)}")
    print(f"    std = {ghg.drop_nulls().std():,.1f}")
    n_c1_ghg   = df.filter((pl.col("C") == 1) & ghg.is_not_null()).height
    n_c1_noghg = df.filter((pl.col("C") == 1) & ghg.is_null()).height
    n_c0_ghg   = df.filter((pl.col("C") == 0) & ghg.is_not_null()).height
    n_c0_noghg = df.filter((pl.col("C") == 0) & ghg.is_null()).height
    print("\n  GHG vs. compliance:")
    print(f"    C=1 AND GHG observed : {n_c1_ghg:,}")
    print(f"    C=1 AND GHG missing  : {n_c1_noghg:,}  <- compliant but no GHG")
    print(f"    C=0 AND GHG observed : {n_c0_ghg:,}  <- edge case")
    print(f"    C=0 AND GHG missing  : {n_c0_noghg:,}")

    sep("4. PER-VARIABLE MISSINGNESS")
    print(f"  {'Column':<46} {'Role':<28} {'Non-null':>9} {'Null':>7} {'Null%':>6}")
    print(f"  {'-'*46} {'-'*28} {'-'*9} {'-'*7} {'-'*6}")
    for col in df.columns:
        if col == "C":
            continue
        n_nn  = df[col].drop_nulls().len()
        n_nl  = int(df[col].is_null().sum())
        role  = SCM_ROLES.get(col, "--")
        print(f"  {col:<46} {role:<28} {n_nn:>9,} {n_nl:>7,} {100*n_nl/N:>5.1f}%")

    sep("5. STRUCTURAL ZEROS: district steam & district chilled water")
    for col in ["district_steam_use_kbtu", "district_chilled_water_use_kbtu"]:
        s    = df[col]
        s_c1 = df.filter(pl.col("C") == 1)[col]
        print(f"\n  {col}:")
        print(f"    All records : null={int(s.is_null().sum()):,}  zero={int((s==0).sum()):,}  positive={int((s>0).sum()):,}")
        print(f"    C=1 only    : null={int(s_c1.is_null().sum()):,}  zero={int((s_c1==0).sum()):,}  positive={int((s_c1>0).sum()):,}")

    sep("6. CHICAGO ENERGY RATING")
    cer = df["chicago_energy_rating"]
    n_nl_cer  = int(cer.is_null().sum())
    n_z_cer   = int((cer == 0).sum())
    n_pos_cer = int((cer > 0).sum())
    print(f"  null (blank - pre-2018 or exempt) : {pct(n_nl_cer, N)}")
    print(f"  0    (non-compliant, 2018+)       : {pct(n_z_cer, N)}")
    print(f"  1-4  (compliant, 2018+)           : {pct(n_pos_cer, N)}")
    print(f"\n  {'Year':>5}  {'Total':>7}  {'Null':>7}  {'Zero':>7}  {'1-4':>7}")
    by_cer = (
        df.group_by("data_year")
        .agg(
            pl.len().alias("total"),
            pl.col("chicago_energy_rating").is_null().sum().alias("null_n"),
            (pl.col("chicago_energy_rating") == 0).sum().alias("zero_n"),
            (pl.col("chicago_energy_rating") > 0).sum().alias("pos_n"),
        )
        .sort("data_year")
    )
    for row in by_cer.iter_rows(named=True):
        print(f"  {row['data_year']:>5}  {row['total']:>7,}  "
              f"{row['null_n']:>7,}  {row['zero_n']:>7,}  {row['pos_n']:>7,}")
    print("\n  By compliance (C):")
    for row in (
        df.group_by("C")
        .agg(
            pl.len().alias("total"),
            pl.col("chicago_energy_rating").is_null().sum().alias("null_n"),
            (pl.col("chicago_energy_rating") == 0).sum().alias("zero_n"),
            (pl.col("chicago_energy_rating") > 0).sum().alias("pos_n"),
        )
        .sort("C")
        .iter_rows(named=True)
    ):
        print(f"    C={row['C']}: total={row['total']:,}  "
              f"null={row['null_n']:,}  zero={row['zero_n']:,}  1-4={row['pos_n']:,}")

    sep("7. UPSTREAM VARIABLE COMPLETENESS")
    for col in ["gross_floor_area_buildings_sq_ft", "primary_property_type",
                "year_built", "of_buildings", "data_year"]:
        n_nl = int(df[col].is_null().sum())
        print(f"  {col:<45}  null={n_nl:,}  ({100*n_nl/N:.2f}%)")

    sep("8. PROPERTY TYPES")
    n_types_all = df["primary_property_type"].n_unique()
    n_types_c1  = df.filter(pl.col("C") == 1)["primary_property_type"].n_unique()
    print(f"  Unique types -- all records   : {n_types_all}")
    print(f"  Unique types -- compliant set : {n_types_c1}")
    print("\n  Top 15 by record count:")
    for row in (
        df.group_by("primary_property_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(15)
        .iter_rows(named=True)
    ):
        print(f"    {str(row['primary_property_type']):<45} {row['n']:,}")

    sep("9. CAUSAL TRAINING SET")
    n_all_ghg    = df.filter(pl.col("total_ghg_emissions_metric_tons_co2e").is_not_null()).height
    n_c1_ghg_obs = df.filter(
        (pl.col("C") == 1) & pl.col("total_ghg_emissions_metric_tons_co2e").is_not_null()
    ).height
    causal = (
        df.filter((pl.col("C") == 1) & pl.col("total_ghg_emissions_metric_tons_co2e").is_not_null())
        .with_columns([
            pl.col("district_steam_use_kbtu").fill_null(0),
            pl.col("district_chilled_water_use_kbtu").fill_null(0),
        ])
    )
    n_causal = causal.height
    print(f"  All with GHG observed                        : {n_all_ghg:,}")
    print(f"  C=1 AND GHG observed (pre-structural-zero)   : {n_c1_ghg_obs:,}")
    print(f"  Causal training set (after structural zeros) : {n_causal:,}")
    print(f"  Deployment test set (C=0)                    : {n_c0:,}")
    print(f"  Paper reports: 21,714 / 21,406 / 5,510")

    sep("10. FLOOR AREA BY COMPLIANCE STATUS")
    for c_val, label in [(1, "Compliant (C=1)"), (0, "Non-compliant (C=0)")]:
        s = df.filter(pl.col("C") == c_val)["gross_floor_area_buildings_sq_ft"]
        print(f"\n  {label}:  {num_summary(s)}")
    print("\n  Compliance rate by floor area band:")
    print(f"  {'Band':<22}  {'Total':>7}  {'C=1':>7}  {'C=1 rate':>9}")
    for lo, hi, label in [
        (0,       50_000,    "< 50k sq ft    "),
        (50_000,  75_000,    "50k-75k sq ft  "),
        (75_000,  100_000,   "75k-100k sq ft "),
        (100_000, 200_000,   "100k-200k sq ft"),
        (200_000, 9_999_999, "> 200k sq ft   "),
    ]:
        sub   = df.filter((pl.col("gross_floor_area_buildings_sq_ft") >= lo)
                          & (pl.col("gross_floor_area_buildings_sq_ft") < hi))
        n_b   = sub.height
        n_b1  = sub.filter(pl.col("C") == 1).height
        rate  = 100 * n_b1 / n_b if n_b > 0 else 0
        print(f"  {label:<22}  {n_b:>7,}  {n_b1:>7,}  {rate:>8.0f}%")

    sep("11. KEY FIGURES FOR SECTION 4.1")
    print(f"  Total records                : {N:,}")
    print(f"  Unique buildings             : {n_buildings:,}")
    print(f"  Years                        : {year_min}-{year_max}")
    print(f"  Columns                      : {n_cols}")
    print(f"  Property types (all)         : {n_types_all}")
    print(f"  Property types (C=1)         : {n_types_c1}")
    print()
    print(f"  GHG observed                 : {pct(n_obs, N)}")
    print(f"  GHG missing                  : {pct(n_miss, N)}")
    print(f"  C=1 (compliant)              : {pct(n_c1, N)}")
    print(f"  C=0 (non-compliant)          : {pct(n_c0, N)}")
    print()
    print(f"  C=1 with GHG observed        : {n_c1_ghg_obs:,}")
    print(f"  Causal training set          : {n_causal:,}")
    print(f"  Deployment test set          : {n_c0:,}")
    print()
    print(f"  GHG distribution (observed): {num_summary(ghg)}")
    print()
    print("  Chicago Energy Rating:")
    print(f"    null (pre-2018 or exempt)  : {n_nl_cer:,}")
    print(f"    0 (non-compliant 2018+)    : {n_z_cer:,}")
    print(f"    1-4 stars (compliant 2018+): {n_pos_cer:,}")

    print(f"\n{'='*72}")
    print("EDA complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = fetch_dataset()
    df = build_compliance(df)
    run_eda(df)
    save_all_plots(df)
