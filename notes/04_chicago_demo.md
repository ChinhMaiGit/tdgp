# 4. Empirical Demonstration: Chicago Energy Benchmarking

## 4.1 Dataset and Institutional Context

- Source: City of Chicago Data Portal
- 28,329 building-year records, 2014--2023, 3,852 unique buildings
- Coverage grew from 243 (2014) to ~3,400--3,600/year (2018+)

**Variable groups by SCM role**:
- **Upstream selection (Xs)**: gross floor area (92.2% observed), primary property type (85.0%, 63 levels)
- **Upstream non-selection (Xo)**: year built (82.2%), number of buildings (84.3%), data year (100%)
- **Compliance gate (C)**: derived from reporting_status + exempt flag (100% observed)
- **Energy mediators (M^obs)**: electricity (81.2%), natural gas (76.0%), district steam (20.3%, structural zero), district chilled water (20.9%, structural zero), other fuel (0.3%)
- **Derived outcomes (Y^obs)**: total GHG (76.6%, target), site/source EUI, GHG intensity, ENERGY STAR score, Chicago Energy Rating

**Key data features**:
- Compliance rate (80.5%) != GHG observation rate (76.6%) -- gaps are informative
  - 22,819 compliant, 5,510 non-compliant
  - Among compliant: 21,406 have GHG, 1,413 don't (partial submissions)
  - Among non-compliant: 5,202 no GHG, 308 have GHG (reported in some years)
- Compliant buildings larger (median 131,000 vs 92,966 sq ft)
- Target: 5 orders of magnitude (0 to 185,162 t CO2e), severe right skew (mean 2.4x median)
- Log1+ transformation => approximately Normal

## 4.2 Stage 1 Applied: The Shadow Matrix

**Compliance indicator construction**: C=1 when reporting_status in {Submitted, Submitted Data} AND exempt flag = False. 22,819 compliant (80.5%), 5,510 non-compliant (19.5%).

**Key findings from shadow matrix**:
- **Gate alignment**: gated fields correlate strongly with C -- electricity 0.93, site EUI 0.91, GHG 0.83, natural gas 0.80
- **Block structure**: within gated set, observation indicators co-vary 0.74--1.00 (mean 0.88). Electricity/site-EUI at 0.98, GHG/GHG-intensity at 1.00. Fields appear and vanish as one unit.
- **Structural zeros apart**: district steam/chilled water correlate only 0.22-0.23 with C (blanks track physical configuration)
- **Chicago Energy Rating anomaly**: correlates -0.02 with C (missingness governed by 2018 introduction date)
- **Upstream fields**: moderate C correlation (0.54-0.87), reflect registry completeness
- **Spatial fields**: geographic coords near-zero with C

**Conclusion**: Block-and-gate pattern incompatible with MCAR. Rules out complete-case analysis. Output = observation structure map (gate, gated block, structural zeros, upstream/spatial sets).

## 4.3 Stage 2 Applied: The Candidate DAG

**Step 1 (carry partition)**: from Stage 1 map
**Step 2 (name Y)**: total_ghg_emissions_metric_tons_co2e
**Step 3 (reason M)**: five metered energy quantities (electricity, natural gas, district steam, district chilled water, other fuel)
**Step 4 (reason Xs)**: gross floor area + primary property type (size-based eligibility); Xo = year built, building count, data year
**Step 5 (instantiate edges)**: Definition 2.1 template yields Figure 4.7

**Three predictions for Stage 3**:
- P1 (selection): compliance caused by Xs (size/type), not energy or emissions
- P2 (atomic gate): mediators and outcome observed/withheld as single unit
- P3 (mediator sufficiency): pa(Y) = M, emissions are function of metered fuels alone

## 4.4 Stage 3 Applied: Falsification and Conformance

**Documentary test** (against Chicago Municipal Code Ch.18-14, EPA Portfolio Manager docs):
- **Selection**: Ordinance defines covered buildings by floor area >= 50,000 sq ft AND property class -- exactly matches Xs assigned. Phased rollout (250K->50K sq ft by class, 2014-2016) corroborates.
- **Atomic gate**: Single annual Portfolio Manager submission carries all fields at once.
- **Mediator/outcome split**: Reporting schema distinguishes inputs (metered fuels) from outputs (GHG, EUI, ENERGY STAR).
- **Formula**: EPA computes GHG by multiplying site energy per fuel by emission factor and summing: GHG = sum_f(phi_f * E_f). This IS pa(Y) = M.

**Reconstruction check**: OLS of GHG on 4 metered fuels (no intercept).
- Constant factor per fuel: R² = 0.987
- Year-specific electricity factor: R² = **0.9999** (adjusted identical)
- Recovered factors: natural gas 53.7 vs EPA 53.11; chilled water 49.3 vs EPA 49.31; steam 58.2 vs EPA 66.40 (few buildings, noisy); electricity 169.1 (multi-year avg of declining factor)
- Residual = electricity factor falling year-on-year (grid decarbonisation)

**Upstream-to-mediator check**: nested OLS of log(1+M) on upstream features.
- Floor area dominates: electricity elasticity 1.03, total energy 0.94
- Property type lifts district fuels (steam 0.00 -> 0.41)
- Confirms Condition 2: Xs genuinely among causes of M, bias is real

**Refinement**: Data year lifted out of Xo (Figure 4.7 -> Figure 4.8):
- Gains edge to Y (eGRID factor decline) and to C (phased rollout)
- Ordinance fixes Xs -> C as sharp threshold, not smooth dependence
- No cost to analysis: data year already conditioned on in all models

**Result**: All three predictions survive documentary test + quantitative corroboration. Tested DAG (Figure 4.8) licenses full Pearlian analysis in Sections 5-7.
