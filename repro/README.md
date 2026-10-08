# Reproducibility drivers

Run these commands from the repository root:

```bash
python repro/run_reproducibility.py headline \
  --output experiments/repro_headline.json

python repro/run_reproducibility.py horizon \
  --output experiments/repro_horizon.json

python repro/audit_telescoping.py \
  --output experiments/telescoping_audit.json

python repro/tele_scope_softmax_simplex_audit.py

python repro/generate_sign_stress.py
```

When the simulators are in the repository's `code/` directory, the driver
finds them automatically.  In a flat checkout or scratch directory, pass
`--code-dir .`.

`run_reproducibility.py headline` regenerates the softmax+ℓ1, box least
squares, and simplex logistic headline experiments with seeds 0, 1, and 2,
including the matched-sensitivity dense baseline.

`run_reproducibility.py horizon` regenerates fixed-total-privacy,
fixed-per-round-noise, an unclipped-gradient signed-state diagnostic, and a
no-state-projection diagnostic for `T = 25, 50, 100, 200`.  Use
`--gamma-mode noise_calibrated` to record the horizon-scaled step size.  The
calibration uses an explicit `--reference-radius`; by default it uses the
generic box radius (`Bbox=2.0`).  Pass `--reference-radius 1.3` only when
reproducing the historical generator-informed calibration.

The signed case formerly called `signed_clipped` is now called
`signed_unclipped`: it sets `C0=Cg=100`, so per-example and gradient clipping
are disabled while the `h/e/r` state projections remain active.  The
`signed_noprojection` case disables those state projections as well and is a
high-epsilon diagnostic rather than a utility baseline.

The outputs include the exact settings and the theory-aligned signed
diagnostics:

```text
c_t   = H_t - mean_i(u_i,t)
rho_t = mean_i(u_i,t - v_i,t)
beta_t = mean_i(v_i,t - raw_mean_i,t)
E_t   = sum_{s<=t} (c_s + rho_s + beta_s)
```

The audit checks the stored cumulative vector against that identity.  It is an
algebraic implementation check, not a convergence or privacy proof.

Every simulator reports both the historical last iterate and the uniform
post-update average `T^-1 sum_{t=1}^T x_t` of the real iterates (with `W_t` or
`w_t` for the matrix/simplex tasks).  Headline tables continue to use the
last iterate for backward compatibility; averaged metrics and exact averaged
vectors are included in the manifests for comparison with averaged-iterate
convergence statements.

`tele_scope_softmax_simplex_audit.py` automatically finds the simulator
modules in `code/` (or in the flat checkout) and writes its default result to
`experiments/tele_scope_softmax_simplex_audit.json`. The active-projection
stress generator deliberately runs with `private=False` and `sigma=0`; its
rows are named `softmax_nonprivate` and `simplex_nonprivate` because it tests
projection-residual signs rather than differential privacy.
