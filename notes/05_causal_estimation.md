# 5. Causal Estimation

## 5.1-5.2 Experimental Setup

**Two analysts, same objective**: model emissions for entire building stock.
- **Causal analyst** (knows tested DGP): recognises P(Y|X,C=1) != P(Y|X). Uses upstream features only, compliant rows, IPW, GroupKFold.
- **Causally-blind analyst**: treats observed sample as population. Uses all features (incl. mediators & derived metrics), all observed rows, no IPW, KFold.

**2x3 factorial design**:

| | Causally-blind | Causal |
|---|---|---|
| **OLS** | M1: CB OLS | M4: IPW OLS |
| **XGBoost** | M2: CB XGB | M5: IPW XGB |
| **Hier. Bayes** | M3: CB Bayes | M6: IPW Bayes |

**Four pipeline differences** (determined by tested DGP):
1. **Feature set**: ALL features (CB) vs upstream only (causal)
2. **Row subset**: all observed target rows (21,714) vs compliant only (21,406) -- asymmetry is deliberate
3. **Selection correction**: none (CB) vs IPW reweighting (causal)
4. **CV protocol**: KFold (CB) vs GroupKFold on building ID (causal)

Plus Bayesian distinction: uninformative priors Normal(0, 2.5) for M3 vs informative Normal(0, 0.5) for M6.

## 5.3 Propensity Model

- Logistic regression: C ~ Xs + Xo + data year (one-hot encoded property types)
- Fitted on all 28,329 records (both compliant and non-compliant)
- Upstream features imputed: global median for numeric, "Unknown" category for property type
- Raw weights w_i = 1/ê(X_i), trimmed at 99th percentile, normalised to mean 1
- Maximum normalised weight: **1.32** (moderate correction)
- Classification accuracy: **95.1%**

**IPW application**:
- OLS/XGB: weights as observation weights in fitting
- Bayesian: likelihood potential: log p*(θ|data) = log p(θ|data) + Σ_i (w_i - 1) * log p(y_i | μ_i, σ)

## 5.4 Model Specifications

**OLS (M1, M4)**: y ~ Normal(μ, σ), μ = x^T β. Closed-form least squares. M1: full CB feature set, KFold, no IPW. M4: upstream only, GroupKFold, WLS with IPW weights.

**XGBoost (M2, M5)**: additive ensemble of T trees. Nested CV: outer 5 folds, inner Optuna TPE 60 trials tuning learning rate, depth, subsampling, regularisation. Objective: MdAPE. M2: full features, KFold, no IPW. M5: upstream only, GroupKFold, IPW as sample weights.

**M3 CB Hierarchical Bayes** (21,714 rows, no IPW):
- y_i ~ Normal(μ_i, σ)
- μ_i = α_{type[i]} + upstream slopes (β_A..β_Y) + mediator slopes (β_E..β_O)
- α_j = ᾶ + δ_j σ_α (non-centred, 63 types)
- Priors: ᾶ ~ Normal(7.0, 1.0), σ_α ~ HalfNormal(1.0), upstream β ~ Normal(0, 2.5), mediator β ~ Normal(0, 2.0), σ ~ HalfNormal(0.5)
- Key: log-linear likelihood CANNOT represent additive EPA formula. Mediator slopes expected small.

**M6 Causal Hierarchical Bayes** (21,406 compliant rows, IPW):
- μ_i = α_{type[i]} + β_A * log(area) + β_B * log(bldgs) + β_T * data year + β_Y * year built
- 55 types (only those in compliant set)
- Upstream β ~ Normal(0, 0.5) -- encodes causal analyst's prior of near-unit log elasticities
- No mediator slopes. IPW via likelihood potential.
- Sampling: 4 chains, 3000 warmup, 3000 draws. MCMC diagnostics: R̂ < 1.01, ESS > 400.

## 5.5 Evaluation Metrics & Ex-Ante Predictions

**Metrics**: R²(log), **MdAPE** (primary), median RMSE, 89% PI coverage, ELPD (PSIS-LOO for Bayes, normal approx for OLS/XGB)

**Four predictions** (stated before results):
1. **Convergence**: M5 and M6 will agree on MdAPE to within ~1 pp
2. **Cost of correction**: causal pipeline shows higher apparent CV error in every approach pair
3. **Accounting artefact**: M2 will attain unusually low CV error via derived-field exploitation
4. **Deployment separation**: causal models produce stable estimates for non-compliant buildings; CB models produce unreliable ones

## 5.6 Results

**Table 5.1 -- Full results:**

| Pipeline | Model | R²(log) | MdAPE | ELPD |
|---|---|---|---|---|
| CB | M1 OLS | 0.756 | 31.5% | -17,075 |
| CB | M2 XGB | **0.993** | **2.1%** | +22,317 |
| CB | M3 Bayes | 0.848 | 20.9% | -11,935 |
| Causal | M4 OLS | 0.557 | 39.5% | -22,716 |
| Causal | M5 XGB | 0.835 | 21.0% | -12,491 |
| Causal | M6 Bayes | 0.838 | 21.4% | -12,316 |
| **IPW effect** | Frequentist | -0.199 | +8.0 pp | |
| | ML | -0.158 | **+18.9 pp** | |
| | Bayesian | -0.010 | +0.4 pp | |

**P1 -- Convergence (corroborated)**: M5 vs M6 gap = **0.4 pp** (well within 1 pp threshold). Holds across all 5 floor-area quintiles. ELPD difference marginal (175 units in M6's favour). Structural model preferred: same accuracy + interpretable parameters + calibrated uncertainty.

**P2 -- Cost of correction (corroborated)**: ΔMdAPE positive in all three pairs. CB advantage reflects circular feature leakage, not generalisation.

**P3 -- Accounting artefact (corroborated)**: M2 R²=0.993, MdAPE=2.1%. SHAP attribution:
- Energy mediators: 57.2% of signal (electricity alone 42.3%)
- Derived/circular features: 15.0%
- Upstream causal features: only 26.1%
- 72.2% from features unavailable or causally invalid at deployment

**P4 -- Deployment separation (corroborated)**: Causal medians: M4=854, M5=1213, M6=1043 t (near compliant median of 1018). CB models span 486-1025 t (539-ton spread). M1 mean=2640 vs median=486 (OLS extrapolation).

**Diagnostic: M3 vs M6 near-zero gap**: M3's log-linear likelihood cannot represent additive EPA formula, so mediator slopes converge to near-zero (electricity β=0.078, floor area β=0.898). Model cannot exploit its mediator access. Near-identical performance from upstream features alone.

**Limits**: IPW corrects observed selection only (max weight 1.32 = moderate). No ground truth for non-compliant buildings. M3's similar deployment doesn't imply causal equivalence.
