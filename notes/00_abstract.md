# Abstract

- **Problem**: In applied causal inference, the causal graph (DAG) is usually asserted, not tested. Most observational data cannot distinguish between incompatible DAGs.

- **Key insight**: For compliance-gated administrative datasets, the documentation defining the reporting obligation also encodes the data-generating process, making the DAG testable.

- **The TDGP pipeline** (3 stages):
  - Stage 1: Shadow matrix analysis -- locate the compliance gate in missingness patterns.
  - Stage 2: DAG hypothesis formulation -- construct a candidate causal graph from the gate and variable descriptions alone.
  - Stage 3: DGP falsification -- test the candidate against primary institutional documentation (ordinances, EPA formulas), which were held out during construction.

- **Demonstrated on**: Chicago Energy Benchmarking dataset (28,329 building-year records, 2014--2023, gated at 50,000 sq ft).
  - EPA GHG accounting identity recovered at R² = 0.9999.

- **2x3 factorial experiment** crosses 2 pipelines (causal vs causally-blind) x 3 model types (OLS, XGBoost, Hierarchical Bayes) = 6 models (M1--M6).

- **Central finding**: IPW-weighted hierarchical Bayesian model (M6) and IPW XGBoost (M5) converge to within 0.4 pp on MdAPE (21.4% vs 21.0%) -- evidence that structural and algorithmic models estimate the same causal quantity once the DGP is tested.

- **Resolution of Breiman's two-cultures tension**: dissolves when the DGP is recoverable from institutional documentation and survives falsification.

- **Additional outputs**: Deployment predictions for 5,510 non-compliant records, counterfactual policy queries.

- **Keywords**: causal inference, directed acyclic graphs, selection bias, missing data, inverse probability weighting, energy benchmarking
