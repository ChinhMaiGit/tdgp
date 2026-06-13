# 7. Counterfactual Analysis

All numbers are analytical functions of saved M6 posterior (12,000 draws x 21,406 compliant records). Linear predictor manipulated; no new sampling.

## 7.1 Decarbonisation Pace

- Convert β_T from standardised to calendar-year rate: r = β_T / s_t = -0.1447 / 2.590 = **-0.0559 log units/yr**
- Exponentiate: **-5.43% per year** [89% CI: -5.61, -5.26]

**Against city target**: Chicago Climate Action Plan (2022) commits to 62% reduction by 2040 from 2017 baseline.
- Required rate: ln(0.38)/23 = -0.0421/yr
- Estimated rate exceeds it: sustained delivers ~72% reduction (vs 62% target)
- Indicative, not conclusive (plan covers all sectors; estimate is benchmarked buildings only)

**Full decarbonisation**: 80% reduction takes ~29 years (arriving ~2052). Net-zero-by-2040 (99% reduction) requires -23.73%/yr (~4.4x estimated pace). Gap = -0.2150 log units/yr.

**Qualifications**: trend assumed to persist; bundles grid emission factor decline + stock efficiency (not building efficiency alone). But β_T is the best-defended parameter (identical under all sensitivity specs).

## 7.2 Counterfactual 1: Compliance Risk in 2030

- Set data year to 2030 for all 21,406 compliant records
- Counterfactual standardised year = (2030 - 2019.1) / 2.590 = 4.21
- Each record's log-scale prediction shifted by β_T * (tilde_t* - tilde_t)
- For each posterior draw: add residual σ, back-transform, check if GHG <= threshold
- Compliance probability = fraction of 12,000 draws falling at or below threshold

**Threshold**: τ = 546 t CO2e (25th percentile of observed compliant GHG). Bands: on-track P > 0.80, high-risk P < 0.20, uncertain between.

**Results** (Table 7.1):
| Classification | Criterion | Records | Share |
|---|---|---|---|
| On-track | P > 0.80 | 7,586 | 35.4% |
| Uncertain | 0.20 <= P <= 0.80 | 5,887 | 27.5% |
| High-risk | P < 0.20 | 7,933 | 37.1% |

Median P = 0.526 (barely better than even odds). Uncertain band = policy-relevant population (marginal cases where intervention could change classification). High-risk block won't reach threshold on trend alone.

**Key contribution**: per-record probability (not binary verdict). Risk-ranked targeting possible. Machinery applies to whatever threshold a regulator specifies.

## 7.3 Counterfactual 2: Vintage Reassignment

- Reassign every building's construction year to 2010 (actual mean = 1964)
- Linear predictor: difference is in vintage term only
- For each draw: μ^CF - μ^F = β_Y * (tilde_y* - tilde_y) = β_Y * (2010 - v_i) / 37.4
- Counterfactual std vintage = (2010 - 1964.3) / 37.4 = 1.22

**Result**: Δ_i (factual - counterfactual):
- Median: **-48 t CO2e** (counterfactual emits MORE)
- 89% CI per typical record: [-53, -42]
- 90.2% of records show credible increase, 8.8% credible reduction, 1.0% no effect
- Mean: -115 t (larger buildings dominate mean)

**Why newer = MORE emissions?**: β_Y = +0.047 positive. β_T absorbs year-on-year efficiency; β_Y captures cross-sectional composition of vintage cohorts. Newer buildings in benchmarked stock house more energy-intensive operations. Reassigning 1960s building to 2010 moves it into more intensive cohort.

**Policy implication**: vintage is poorly targeted instrument. Area-normalised intensity standards by property type are better.

**Method implication**: NOT a retrofit-impact estimator. Retrofit operates through energy mediators (compliance-gated, outside upstream set). Estimating that effect requires mediator layer (beyond paper's scope).

## Why the Climb Matters

Both CF queries could have been computed on an asserted DAG, but the counterintuitive vintage result would be indistinguishable from a specification artefact. On the tested structure, the surprising answer has a warrant: the separation of trend from cohort that produces it is carried by the structure that survived Stage 3.
