# 3. The Tested Data-Generating Process Pipeline

## Overview

Three stages that return a DAG that has survived a falsification test:
1. Read the recording layer from the data (shadow matrix)
2. Hypothesise the world layer that generated the values (candidate DAG)
3. Test that hypothesis against primary documentation (falsification)

Pipeline is dataset-agnostic. Section 4 applies it to Chicago data.

---

## Stage 1: Shadow Matrix Analysis

- For each unit i and variable j, define missingness indicator R_ij = 1{X_ij is missing}
- Shadow matrix = p x p matrix of pairwise correlations among R_1 ... R_p (phi coefficients)

**Three regimes distinguishable**:
- **Idiosyncratic (MCAR)**: off-diagonal correlations near zero -- refutes compliance-gate hypothesis
- **Block structure**: set of variables goes missing in lockstep -- fingerprint of atomic reporting gate (Condition 3). Gated fields should share single missingness pattern, correlate almost perfectly.
- **Gate alignment**: observed C correlates with field missingness. For gated fields, R_ij = 1 - C_i, correlation approaching -1.

**Missingness taxonomy** (multiple mechanisms, told apart by relation to C):
- Gate-missingness: absent exactly when C=0 (MAR; pipeline corrects)
- Selection-side discrepancy: compliance doesn't follow threshold cleanly
- Reporting-side discrepancy: partial filing, errors -- gated fields stop moving in lockstep
- Structural zeros: quantity doesn't exist for unit (physical configuration, not gate)
- Incidental gaps: MCAR, handled by ordinary means

**Key output**: map of observation structure -- gate C, gated block {M^obs, Y^obs}, structural zeros to preserve, always-observed upstream features. Map stops at coarse partition (cannot separate outcome from mediators within gated block).

**Key conclusion**: block-and-gate structure is incompatible with MCAR. Refuting MCAR rules out default workflows (complete-case analysis, unconditional imputation).

---

## Stage 2: DAG Hypothesis Formulation

Build candidate causal graph from shadow matrix + variable descriptions + Definition 2.1 only. Documents stay closed (they are the test in Stage 3).

**Five steps**:
1. **Carry over partition (data)**: gate C, gated block, structural zeros, upstream set from Stage 1
2. **Name Y (choice)**: within gated block, the quantity the study explains. Research question determines this.
3. **Reason M (conjecture)**: remaining gated fields described as intermediate quantities -> mediators feeding Y. Claim pa(Y) = M.
4. **Reason Xs (conjecture)**: within upstream set, feature marked as plausible eligibility criterion (measure of size/scale) -> Xs. Rest -> Xo. Threshold signature (C turns from 0 to 1 at cutoff) can corroborate.
5. **Instantiate edges (template)**: Definition 2.1 supplies every edge: Xs -> C, Xs -> M, Xo -> M, M -> Y, M -> M^obs <- C, Y -> Y^obs <- C.

**Role of data**: settles only the coarse partition (Step 1) and threshold check (Step 4). Everything else is categorisation and argument.

**Sharpest claim**: pa(Y) = M (mediators exhaust causes of Y). This is what Stage 3 tests.

**Implication**: admissible inputs = upstream features only (Xs, Xo). Mediators are inadmissible twice: Y = f_Y(M) makes them a reconstruction of the generating relation; M^obs is gated (absent for non-reporting units).

---

## Stage 3: DGP Falsification and Refinement

- Open the documents Stage 2 held back.
- Compare each conjecture to what the documentation states.
- Three provisions settle three conjectures:
  - **Selection rule** -> actual gate (Xs, R) -- bears on Step 4
  - **Reporting schema** -> classification of gated block (M vs Y) -- bears on Steps 2-3
  - **Technical standard** -> connection from M to Y (f_Y, pa(Y) = M)

- Agreement = assignment stands unrefuted.
- Disagreement = conjecture refuted, document supplies correction.

**What documents settle outright**: the gate, the classification, pa(Y) = M (the backbone).
**What they don't constitute**: upstream-to-mediator edges (data corroborates by association).
**What they guarantee**: structure as written, not degree to which rule was actually applied (fidelity).

**Empirical check**: reconstruction of Y from M (formula predicts outcome from mediators). Clean reconstruction = data follows documented formula.

---

## 3.4 What the Tested DAG Licenses

**Pearl's Ladder of Causation**:

**Rung 1 -- Association (d-separation)**:
- Xs -> M -> Y: chain, cut by conditioning on mediators
- C <- Xs -> M: fork, cut by conditioning on Xs
- M -> M^obs <- C: collider, conditioning OPENS it (selecting on compliant units opens this path)

**Rung 2 -- Intervention (back-door adjustment / IPW)**:
- Xs is the confounder: common cause of selection (Xs -> C) and outcome (Xs -> M -> Y)
- Conditioning on Xs closes the back door: C _||_ Y | Xs
- Xo is safe default: nothing connects it to C, can't confound. Included because it causes outcome.
- IPW: weight each compliant unit by 1/P(C=1 | Xs, Xo). Restores missing mass.
- **Boundary**: reweighting only works where overlap exists (P(C=1 | Xs) > 0). Below threshold, must extrapolate via structural form (modularity of equations).
- **Residual assumption**: selection ignorability (no unobserved driver of compliance correlated with outcome). Not settled by documents; partially auditable via gate-fidelity checks.

**Rung 3 -- Counterfactuals** (abduction-action-prediction):
- Abduction: recover unit's disturbances from observed values
- Action: replace feature with counterfactual value
- Prediction: re-run model with new value
- Requires estimated structural equations (from Rung 2).

Estimators are procedural; any model family works as long as it targets latent Y, uses upstream features only, carries IPW weights.
