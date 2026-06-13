# 8. Conclusion

## Restatement

- **Problem**: DAG specification is the binding constraint in causal inference. Usually asserted, not tested.
- **Solution**: For compliance-gated datasets (3 conditions: threshold selection, endogenous Xs, atomic reporting), the documents that create the reporting obligation also encode the DGP. The TDGP pipeline (shadow matrix -> DAG hypothesis -> documentary falsification) exploits this to produce a tested DAG.
- **Chicago results**: Pipeline ran end-to-end on 28,329 building-year records. EPA formula recovered at R²=0.9999. Stage 3 left DAG unrefuted.

## Core Findings

- **Convergence**: M5 IPW XGBoost (21.0%) vs M6 IPW Hierarchical Bayes (21.4%) -- **0.4 pp MdAPE gap**
- **Elasticity**: β_A = 0.979 (doubling floor area ~ doubles GHG; interval excludes 1.0)
- **Decarbonisation**: -5.4%/yr, survives all sensitivity tests
- **Type heterogeneity**: 22-fold emissions range across 55 property types
- **Deployment**: 1,043 t median for 5,510 non-compliant buildings (consistent with selection, not cleaner stock)
- **Counterfactuals**: 37.1% of records at high risk by 2030; 90.2% of buildings increase emissions under 2010 vintage

## Breiman Resolution

Two-cultures tension dissolves under a specific, checkable condition: the DGP is recoverable from institutional documentation AND survives falsification. When it holds, both families estimate the same quantity and differ only in what they return (number vs mechanism). When it doesn't, algorithmic caution stands.

## Limitations

1. **Selection ignorability** is the residual documents can't reach. 95.1% propensity accuracy, not 100%. Unobserved confounders of compliance (management quality, retrofits) remain uncorrected.

2. **Overlap boundary**: IPW recovers only where compliant units exist. Below 50K sq ft threshold, extrapolation relies on modularity. Deployment predictions for non-reporters lack ground truth.

3. **Temporal trend bundle**: β_T mixes grid decarbonisation + building efficiency + operational shifts. Projections assume persistence.

4. **Pipeline tests structure, not functional form**: M3 couldn't exploit mediator access because log-linear != additive EPA formula. Convergence conditional on chosen form being adequate.

5. **Mechanism-level CFs out of reach**: Mediators are compliance-gated. Retrofit impacts require mediator layer (not identified here).

6. **One city, one decade**: Generalisation to other class members is empirical question.

## Extensions

1. **Replication across class**: Re-run on other cities' benchmarking ordinances, OSHA, TRI, credit data.

2. **Doubly robust estimation**: Survive misspecification of either selection or outcome model. + Mediator layer modelling to enable retrofit counterfactuals.

3. **Regulatory deployment**: Replace point-imputed emission gap-filling with uncertainty-stating estimates. Probabilistic compliance classification with actual enforcement parameters.

## Key Insight

The compliance gate -- ordinarily a reason to distrust administrative data -- is its most valuable feature: the same institution that censors the data documents the process that generated it.

## Data & Code

- Chicago Energy Benchmarking data: publicly available from City of Chicago Data Portal (API). Not redistributed.
- Analysis code: available from author on request.
