# 2. The Compliance-Gated Dataset Class

## 2.1 Formal Definition

- Framework: Structural Causal Model (SCM) from Pearl (2009)
- SCM is a triple M = <U, V, F>: exogenous variables, endogenous variables, structural equations.

**Endogenous variables:**
- **Xs** = selection variables (determine compliance AND cause outcome)
- **Xo** = non-selection features (always observed, may cause outcome)
- **M** = latent mediators (between Xs and Y)
- **Y** = outcome of interest (latent)
- **C** = compliance indicator (binary: 1 if satisfies reporting obligation)
- **M^obs** = observed mediators (M if C=1, missing otherwise)
- **Y^obs** = observed outcome (Y if C=1, missing otherwise)
- **R** = compliance region (C=1 iff Xs in R)

**Structural equations:**
- Xs <- U_Xs
- Xo <- U_Xo
- M <- f_M(Xs, Xo, U_M)
- Y <- f_Y(M, U_Y)
- C <- 1{Xs in R}
- M^obs <- C * M
- Y^obs <- C * Y

First four model the world; last three model the institution (how reporting obligation filters data).

### Three Jointly Necessary Conditions

**Condition 1: Unit-level threshold selection.** C_i = 1{Xs,i in R}. No discretion, no partial compliance. Threshold-based programmes are most common.

**Condition 2: Endogenous selection variable.** Xs appears as causal argument in f_M with non-zero effect. Guarantees path Xs -> M -> Y. Xs is not merely correlated with Y through a common cause; it IS a cause.

**Condition 3: Atomic reporting requirement.** Unit reports all outcome data or none. Partial reporting structurally impossible. Distinguishes compliance-gated from item non-response.

**Remark on necessity**: Condition 1 alone = MAR. Condition 2 makes the truncation analytically consequential (bias exists). Condition 3 distinguishes from item non-response.

**Constitutive assumption**: Documents specify the rules the institution uses. The SCM equations ARE the programme's published rules. Disturbances carry whatever the rules do not. Exogeneity (independence of U terms) is assumed, not read from documents.

## 2.2 Selection Bias This Class Induces

**Proposition 2.1 (Compliance-Gate Selection Bias):** Under Def 2.1, if (i) population has both compliant and non-compliant units, and (ii) the two groups differ in expected outcome, then E[Y | C=1] != E[Y].
- Direct consequence: complete-case analysis computes the wrong quantity.
- The error is inferential: Y^obs is not wrong (matches Y for compliant units), but using it as if it described the full population is the mistake.

**Missingness classification:**
- R_i = 1 - C_i (1 when Y missing, 0 when observed)
- P(R=1 | Y, Xs, Xo) = 1{Xs not in R} -- depends only on observed Xs
- This is **MAR** (Missing At Random): missingness fully explained by observed variables
- Not MCAR (would need near-zero correlations in shadow matrix)
- Not MNAR (selection doesn't depend on unobserved Y)

**MAR is consequential**: missingness mechanism can be modelled from data in hand. Estimate P(C=1 | Xs, Xo), reweight complete cases by its inverse.

**Vs Heckman (1979)**: Heckman targets MNAR (selection on unobservables) with distributional assumptions. Compliance gate needs none of that (deterministic function of observed covariates = MAR).

**Cross-validation blind spot**: both train and test folds drawn from complete cases. CV measures fit on compliant subsample, not population representativeness.

## 2.3 Why DAG Falsification Is Tractable

- The central bottleneck in causal inference is DAG specification (not estimation).
- For compliance-gated data, the docs are **constitutive**, not descriptive -- they are the primary source of the rules that generate the data.

Three doc components map directly onto three DAG components:
1. **Selection rule** (threshold provision) -> Xs, R, structural equation for C
2. **Reporting schema** (required fields) -> M^obs, Y^obs, and through them M and Y
3. **Technical standard** (accounting formula) -> f_Y, pa(Y) = M

**What documents settle**: the gate, the classification of fields, the outcome's parents -- the backbone of the graph.

**What documents don't settle**:
- Xs -> M, Xo -> M edges (mediators are measured, not computed from a rule; confirmed by association in data)
- Realized fidelity (degree to which rule was actually applied, in disturbance terms)
- Selection ignorability (no unobserved confounder of compliance and outcome)

Result: DAG becomes testable. Each conjecture can be set against a documented provision. A contradiction refutes the conjecture; the same provision supplies the correction.

## 2.4 Scope of the Class

Four programmes mapped onto the definition:
- Municipal energy benchmarking (floor area threshold, GHG via fuel use)
- OSHA injury recordkeeping (employee count > 10)
- EPA Toxics Release Inventory (AND-logic: industry + employees + chemical throughput)
- SEC accelerated filer disclosure (boundary case: public float threshold; Condition 3 only holds for individual gated items)

Class membership criterion: a written rule that predates the data, fixes which units are recorded, and turns on a cause of the outcome. Extends beyond government regulation (lender loan books, insurer eligibility, clinical registries).

Key: the document must be obtainable. If proprietary and withheld, DAG can only be asserted.
