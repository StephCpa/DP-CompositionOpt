# Reproducibility review

## What was checked

The three headline objective families can be regenerated from the committed
simulators with the same settings used in the current research record:

- softmax plus \(\ell_1\): 20 clients, 40 samples/client, \(d=8\), 3 classes,
  80 rounds, Top-K fraction 0.1;
- box-constrained least squares: 20 clients, 40 samples/client, \(d=10\),
  100 rounds, Top-K fraction 0.1;
- simplex-constrained logistic regression: 20 clients, 40 samples/client,
  \(d=8\), 100 rounds, Top-K fraction 0.2.

All runs use seeds 0, 1, and 2, \(\varepsilon=8\),
\(\delta=10^{-5}\), full participation \(q=1\), and the
`without_replacement_bound` accountant.  The state settings are recorded
explicitly: \(C_0=C_g=1.5\), \(B_r=2\), \(B_h=1\), \(B_e=2\), and
\(\gamma=5\).  The driver writes these settings into every output manifest.

The driver is:

```text
repro/run_reproducibility.py
```

Example commands:

```bash
python repro/run_reproducibility.py headline \
  --output experiments/repro_headline.json

python repro/run_reproducibility.py horizon \
  --output experiments/repro_horizon.json

python repro/tele_scope_softmax_simplex_audit.py

python repro/generate_sign_stress.py
```

For the current flat scratch layout, pass `--code-dir .`; in the repository,
the default is `code/`.

## Diagnostic correction

The simulator now logs the theory-aligned signed vectors rather than treating
the Top-K residual as the cumulative theorem error:

\[
c_t = H_t - \frac1n\sum_i u_{i,t},
\qquad
\rho_t = \frac1n\sum_i(u_{i,t}-v_{i,t}),
\qquad
\beta_t = \frac1n\sum_i(v_{i,t}-\text{raw\_mean}_{i,t}).
\]

The full error is

\[
E_t=\sum_{s\le t}(c_s+\rho_s+\beta_s),
\]

with signed vectors retained in each trajectory.  The driver reports both

- `Q_T_clipped`, which accumulates \(c_t+\rho_t\) and corresponds to the
  bounded/clipped-oracle objective; and
- `Q_T`, which also includes \(\beta_t\) and therefore measures the distance
  to the original unclipped objective.

It also checks the telescoping identity by comparing the reconstructed signed
sum with the simulator's stored `E_t_vec`.  In the three-seed box-LS headline
run, the maximum identity error is at machine precision (about \(10^{-14}\)).

The historical `delta - TopK(delta)` proxy is retained only as a legacy
compression diagnostic. It is not the theorem's \(c_t\), and it is not guaranteed
to be a conservative upper bound because signed terms can cancel or reinforce.
All old proxy sweep JSONs are superseded for theory claims.

## Horizon modes

The horizon driver supports:

1. `fixed_total`: fixed total \((\varepsilon,\delta)=(8,10^{-5})\), so the
   per-round Gaussian scale changes with the number of rounds;
2. `fixed_sigma`: fixed per-round \(\sigma\), with the equivalent epsilon
   computed from the same RDP accountant;
3. `signed_unclipped` (legacy driver key: `signed_clipped`):
   \(C_0=C_g=100\), retaining bounded-state projections;
4. `signed_noprojection`: \(C_0=C_g=B_r=B_e=100\), \(B_h=10\), with the
   reported privacy loss in the 64--428 range. Its manifest label is a
   high-epsilon/non-private diagnostic, not a private headline configuration.

The default horizon set is \(T\in\{25,50,100,200\}\).  The optional
`--gamma-mode noise_calibrated` setting uses

\[
\gamma=\max\{5,\sigma\sqrt{dT}/R_{\mathrm{ref}}\},
\]

where `--reference-radius` is explicit; the default is the generic box radius
`Bbox=2.0`, and `1.3` is reserved for reproducing the historical generator-
informed calibration. The manifest records `reference_radius` and the full
calibration note.

## Remaining limitations

- The driver reproduces the simulators; it does not turn empirical
  telescoping into a convergence proof.
- The headline simulator reports the last iterate as the primary metric and
  now also reports a uniform average of the real iterates. The averaged metrics
  are included for direct comparison with the averaged-iterate theorem; they do
  not by themselves prove a convergence rate.
- The vector-level algebraic audit now covers box-LS, softmax, and simplex.
  It is available as `repro/audit_telescoping.py` and checks that stored
  cumulative `E_t` equals the signed `c/rho/beta` sum to floating-point
  precision.  This is an implementation check, not a convergence proof.
- `repro/tele_scope_softmax_simplex_audit.py` resolves `code/` (or the flat
  checkout) automatically and writes to
  `experiments/tele_scope_softmax_simplex_audit.json` by default. It can be
  run directly from the repository root without setting `PYTHONPATH`.
- `repro/generate_sign_stress.py` regenerates the active-projection stress
  manifest. Its rows are explicitly `softmax_nonprivate` and
  `simplex_nonprivate`; `private=False` and \(\sigma=0\) are intentional
  because this experiment tests algebraic residual signs, not privacy.
- Communication accounting charges dense baselines for values only; Top-K additionally sends indices. The simulators implement that rule. Historical sweep files retain legacy labels and are superseded for theory claims.

