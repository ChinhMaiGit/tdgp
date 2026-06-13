# 1. Introduction

## The Compliance-Gated Data Problem

- Mandatory reporting programmes (EPA, OSHA, SEC, municipal benchmarking) share a structural feature: reporting is triggered by an observable characteristic that is also a cause of the outcome.
  - e.g. large buildings must report energy use *because* they are large, and they emit more GHG *because* they are large.
  - This is systematic unit-level selection, not random missingness.

- **Default analysis fails**: standard workflows (regression, ML) treat the observed sample as representative of the population.
  - They estimate P(Y | X, C = 1), not P(Y | X).
  - Cross-validation cannot detect this bias (test folds come from the same compliance-enriched distribution).

- Existing tools are fragmented:
  - Pearl's framework: DAG is assumed, not tested.
  - IPW literature: adjustment set validity is assumed, not tested against data structure.
  - Heckman selection models: built for MNAR (unobservables), distributional assumptions that compliance-gated MAR data doesn't need.
  - ML literature: treats DGP as unknown, can't detect selection bias.

## Four Contributions

1. **Formalise compliance-gated dataset as a distinct class** (3 jointly necessary structural conditions):
   - Unit-level threshold selection on observables
   - Endogenous selection variable (also causes outcome)
   - Atomic reporting (all-or-nothing outcome observation)

2. **TDGP pipeline**: 3-stage procedure (shadow matrix -> DAG hypothesis -> documentary falsification) to establish a tested causal graph from compliance-gated data.

3. **Common causal foundation**: IPW correction enables structural and algorithmic models to answer the same population-level question, enabling genuine cross-paradigm comparison.

4. **Empirical convergence**: M6 (Bayes) matches M5 (XGBoost) to <1 pp MdAPE, resolving Breiman's two-cultures tension under the condition that the DGP is recoverable.

## Paper Roadmap

- Sec 2: Formalise class, selection bias, tractability argument
- Sec 3: TDGP pipeline in dataset-agnostic form
- Sec 4: Pipeline applied to Chicago data (stages 1-3)
- Sec 5: Causal estimation (2x3 factorial, M1-M6 results)
- Sec 6: Structural inference from focal model M6
- Sec 7: Counterfactual analysis (two queries)
- Sec 8: Conclusion, limitations, extensions
