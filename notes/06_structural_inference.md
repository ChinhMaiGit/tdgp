# 6. Structural Inference from the Focal Model (M6)

## Setup

- M6 posterior (4 chains, 12,000 post-warmup draws) from Section 5 experiment
- Four **pre-declared** parameters of interest: β_A, β_T, σ_α, σ
- β_B and β_Y are auxiliary controls
- All estimates are reduced-form causal effects (upstream -> GHG through mediator layer)

## 6.1 Posterior Health

- R̂_max = 1.0095, ESS_min = 526 (>400 threshold), zero divergent transitions
- PSIS-LOO ELPD = -12,316 (SE 317.4), 1/21,406 obs with Pareto-k > 0.7
- 92.8% empirical coverage at 89% nominal (slightly conservative)
- Posterior predictive checks: 200 simulated datasets match observed distribution

## 6.2 Global Parameter Posteriors

**Table 6.1 -- All seven parameters (89% CI)**:

| Parameter | Role | Mean | 89% CI |
|---|---|---|---|
| ᾶ | global intercept | 7.409 | [7.304, 7.515] |
| σ_α | between-type SD | **0.483** | [0.409, 0.568] |
| β_A | log floor area | **0.979** | [0.973, 0.985] |
| β_B | log building count | 0.014 | [0.002, 0.026] |
| β_T | data year (std) | **-0.145** | [-0.150, -0.140] |
| β_Y | year built (std) | 0.047 | [0.042, 0.052] |
| σ | residual SD | **0.430** | [0.427, 0.434] |

**Interpretation**:
- **β_A = 0.979**: doubling floor area multiplies GHG by 2^0.979 ≈ 1.97. Near-unit elasticity consistent with OLS from Stage 3 and EPA formula. Interval excludes 1.0 (credibly sub-proportional -- larger buildings slightly more efficient per sq ft).
- **β_T = -0.145**: strongest temporal signal. exp(-0.145) ≈ 0.865 per SD of data year (~5-6%/yr decline). Consistent with eGRID factor decline + stock efficiency gains. Best-identified parameter.
- **β_B = 0.014**: small but credibly positive. More buildings = marginally more emissions (common-area overhead).
- **β_Y = 0.047**: newer construction = slightly HIGHER emissions, controlling for size/type/reporting year. Counterintuitive but consistent with newer buildings housing more intensive operations. β_T and β_Y separated because tested DAG carries them on different nodes.
- **σ_α = 0.483**: substantial type-level heterogeneity beyond size/age.
- **σ = 0.430**: irreducible within-type variation (factor ~0.65-1.54 around conditional mean). This is the floor under MdAPE (~21%).

## 6.3 Property-Type Hierarchy

- 55 types represented in compliant training set
- 22-fold emissions range: Data Center (α=9.352) down to Repair Services (α=6.234) = 3.1 log units spread
- Ordering consistent with engineering knowledge (data centres > labs > supermarkets > ... > repair services)
- Partial pooling in action: rare types heavily shrunk toward global mean; common types precisely estimated
- Flat model (ignoring type) would be systematically wrong across entire range

## 6.4 Deployment Inference

- 5,510 non-compliant records predicted from upstream features only
- Posterior predictive median: **1,043 t** (vs compliant median 1,018 t, 2.5% diff)
- 89% interval for individual building: [495, 2,103] -- deliberately wide (no energy data = full σ + type uncertainty)
- 3,924/5,510 (71%) have Unknown property type -> get global intercept with full between-type uncertainty
- Per-type heterogeneity: K-12 schools 676 t, multifamily 893 t, mixed-use 2,117 t
- Result consistent with selection (compliance determined by gate, not emissions)

## 6.5 Sensitivity Analysis

**Design**: vary two analyst-controlled inputs independently. Criterion: flag if posterior mean shifts >20% of base CI half-width.

**Prior sensitivity** (tight: N(0,0.25) vs base: N(0,0.5) vs wide: N(0,1.0)):
- All four parameters stable. Max shift: 3.2% of half-width (σ_α).
- β_A and β_T identical to 3 decimal places. Likelihood dominates at n=21,406.

**IPW trim cap sensitivity** (95th vs 99th vs 99.9th percentile):
- Max normalised weights: 1.166, 1.322, 2.750
- Three parameters move ≤11.3% of half-width
- β_T identical under every cap (temporal signal can't be redistributed by reweighting)
- β_A at lenient cap: 29% of half-width (flagged), but absolute shift is 0.0018 (0.979->0.977), doubling effect from ×1.971 to ×1.968 (0.15% diff, no consequence)

**Conclusion**: Findings are properties of data + tested structure, not artefacts of prior choice or weight trimming.
