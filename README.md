# DP-CompositionOpt

Research materials for combining **Composite Optimization with Error Feedback: the Dual Averaging Approach** (Paper 3) with client-level central differential privacy.

## Current mainline

The current mechanism is deliberately narrow:

- full client participation, q = 1;
- bounded local EControl states h/e/r;
- Top-K communication;
- release of the current clean aggregate with fresh Gaussian noise each round;
- real iterates as the reported sequence;
- no server-side integration of noisy differences;
- no final uncompressed upload in the privacy mechanism;
- central client-level DP with an explicit sensitivity and RDP accountant.

For a bounded state radius B_h and n clients, the per-round replacement sensitivity of the aggregate is 2 B_h / n. The dense DA baseline uses 2 C_0 / n.

## Main research conclusion so far

The mechanism migrates to several convex composite objectives, but the real-iterate convergence statement must remain conditional.

The key candidate closure is

```
C_T <= kappa_0 + kappa_D sum_t D_t
             + kappa_P sum_t P_t + kappa_Xi sum_t Xi_t
D_t <= K_x M_t + K_P P_t + K_Xi Xi_t
```

where D_t is the local EControl input variation and M_t is real-iterate movement. A candidate absorption condition is

```
mu > 6 kappa_D K_x / gamma_lower.
```

The theory is split into two versions:

1. **Clipped/bounded objective:** beta_t = 0, or the clipped oracle is treated as the objective oracle.
2. **Original objective:** an additional summability/zero-mean/decay assumption or a separate clipping-residual feedback mechanism is required.

No unconditional O(T^{-1/2}) real-iterate theorem is claimed in this repository.

## Repository layout

- `code/`: simulators and adversarial-search code.
- `docs/`: research record, theorem skeleton, closure lemma, and cross-objective conditions.
- `experiments/`: CSV/JSON ledgers and multi-seed diagnostics.

## Reproducibility

The simulators use Python, NumPy, and the local RDP accountant in `code/p3_bounded_sim.py`.

Representative checks already run:

- Python compilation for all simulator modules;
- finite-difference gradient checks;
- simplex feasibility checks;
- three-seed runs for softmax+ℓ1, box least squares, and simplex logistic regression;
- fixed-total-privacy and fixed-per-round-sigma horizon sweeps;
- signed projection-residual diagnostics.

## Important diagnostic

Under a fixed total privacy budget, per-round sigma increases with the number of rounds. In addition, persistent clipping bias can make the cumulative E_t term grow even if sigma is fixed. The repository therefore records:

- accountant mode, q, sensitivity, and sigma;
- state age and clipping residuals;
- c_t, rho_t, beta_t, movement, and E_t diagnostics;
- communication bits including Top-K indices.

## Key references

- Paper 3: https://arxiv.org/abs/2510.03507
- EControl: https://arxiv.org/abs/2311.05645
- DiceSGD: https://arxiv.org/abs/2311.14632
- Fixed-size sampling RDP: https://arxiv.org/abs/1909.03567
