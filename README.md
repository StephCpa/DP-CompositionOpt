# DP-CompositionOpt

Research materials for combining **Composite Optimization with Error Feedback: the Dual Averaging Approach** (Paper 3) with client-level central differential privacy.

## Current mainline

The mechanism is deliberately narrow:

- full client participation, `q = 1`;
- bounded local EControl states `h/e/r`;
- Top-K communication;
- release of the current clean aggregate with fresh Gaussian noise each round;
- real iterates as the reported sequence;
- no server-side integration of noisy differences;
- no final uncompressed upload in the privacy mechanism;
- central client-level DP with explicit replacement sensitivity and an RDP accountant.

For a bounded state radius `B_h` and `n` clients, the per-round replacement sensitivity of the aggregate is `2 B_h / n`. Conditioning on the released history makes the neighboring iterate trajectories identical before the replaced client's bounded state is averaged, so this is a closed deterministic sensitivity bound for the mainline mechanism. The dense DA baseline uses `2 C_0 / n`.

## Main research conclusion so far

The mechanism migrates to several convex composite objectives, but the real-iterate convergence statement remains conditional. The theory is split into two versions:

1. **Clipped/bounded objective:** treat the clipped oracle as the objective oracle, or impose `beta_t = 0`.
2. **Original objective:** add a clipping-bias summability/zero-mean/decay assumption, or add a separate clipping-residual feedback mechanism.

No unconditional `O(T^{-1/2})` real-iterate theorem is claimed in this repository.

## Corrected diagnostics

The simulators and audit scripts now use the theorem-aligned signed decomposition

```text
c_t   = H_t - mean_i(u_i,t)
rho_t = mean_i(u_i,t - v_i,t)
beta_t = mean_i(v_i,t - raw_mean_i,t)
E_t   = sum_{s <= t} (c_s + rho_s + beta_s)
```

The historical `delta - TopK(delta)` quantity and accumulated projection-norm proxy are retained only as legacy stress diagnostics and are superseded for theory claims. They are not guaranteed conservative because signed terms can cancel or reinforce. Projection residual vectors are stored separately as `raw - new`, with the sign convention documented in `docs/p3_Et_closure_lemma_zh.md`. The remaining utility-proof target is the cumulative e/r projection-residual term `P_T`, not the privacy sensitivity.

The horizon driver uses `signed_unclipped` (the old `signed_clipped` label was misleading because `C0=Cg=100` disables input clipping) and `signed_noprojection` for the high-epsilon/no-state-projection diagnostic. Noise-calibrated gamma uses an explicit `--reference-radius`; the default is the generic `Bbox=2.0`, while `1.3` is reserved for historical reproduction.

The superseded JSONs and their interpretation are listed in
`docs/LEGACY_DIAGNOSTICS.md`.

Dense communication counts values only (`T d * 32` bits/client); Top-K additionally sends indices (`T K (32 + ceil(log2 d))` bits/client). The headline manifest includes both the historical dense baseline (`C0=1.5`) and a matched-sensitivity dense baseline (`C0=1.0`).

## Repository layout

- `code/`: simulators, accountant, and adversarial-search code.
- `docs/`: research record, theorem skeleton, closure lemma, cross-objective conditions, and reproducibility review.
- `repro/`: reproducibility driver and independent diagnostic audits.
- `experiments/`: CSV/JSON ledgers, corrected headline manifests, and horizon sweeps.

## Reproducibility

The simulators use Python, NumPy, and the local RDP accountant in `code/p3_bounded_sim.py`.

Regenerate headline runs and both horizon protocols from the repository root:

```bash
python repro/run_reproducibility.py headline \
  --output experiments/repro_headline.json
python repro/run_reproducibility.py horizon \
  --gamma-mode fixed \
  --output experiments/repro_horizon_fixed.json
python repro/run_reproducibility.py horizon \
  --gamma-mode noise_calibrated \
  --output experiments/repro_horizon_noise_calibrated.json
python repro/audit_telescoping.py \
  --output experiments/softmax_simplex_telescoping_audit.json
```

The manifests record `C0`, `Cg`, `Br`, `Bh`, `Be`, `gamma`, `gamma_mode`, `reference_radius`, seeds, `epsilon`, `delta`, accountant, participation rate, sensitivity, per-round sigma, last/uniform-average iterate metrics, iterate protocol, and bit budget. The independent checker `repro/tele_scope_softmax_simplex_audit.py` reconstructs local updates for softmax and simplex and runs a projection sign stress test.

Representative checks already run include Python compilation, finite-difference gradient checks, simplex feasibility checks, three-seed runs for softmax+ℓ1, box least squares, and simplex logistic regression, fixed-total-privacy and fixed-per-round-sigma horizon sweeps, signed projection-residual diagnostics, and matched-sensitivity baselines.

## Important limitation

Under a fixed total privacy budget, per-round sigma increases with the number of rounds. Persistent clipping bias can also make the original-objective cumulative error grow even when sigma is fixed. The corrected clipped-objective diagnostic and the original-objective bias are therefore reported separately. The no-projection horizon diagnostic is a high-epsilon/non-private stress test, not an epsilon=8 headline result; its corrected `Q_T/T` is flat or falling (3.06 to 2.18), while the old proxy is superseded.

## Key references

- Paper 3: https://arxiv.org/abs/2510.03507
- EControl: https://arxiv.org/abs/2311.05645
- DiceSGD: https://arxiv.org/abs/2311.14632
- Fixed-size sampling RDP: https://arxiv.org/abs/1909.03567
