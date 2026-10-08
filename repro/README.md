# Reproducibility drivers

Run these commands from the repository root:

```bash
python repro/run_reproducibility.py headline \
  --output experiments/repro_headline.json

python repro/run_reproducibility.py horizon \
  --output experiments/repro_horizon.json

python repro/audit_telescoping.py \
  --output experiments/telescoping_audit.json
```

When the simulators are in the repository's `code/` directory, the driver
finds them automatically.  In a flat checkout or scratch directory, pass
`--code-dir .`.

`run_reproducibility.py headline` regenerates the softmax+ℓ1, box least
squares, and simplex logistic headline experiments with seeds 0, 1, and 2,
including the matched-sensitivity dense baseline.

`run_reproducibility.py horizon` regenerates fixed-total-privacy,
fixed-per-round-noise, signed-state, and no-projection box-LS diagnostics for
`T = 25, 50, 100, 200`.  Use `--gamma-mode noise_calibrated` to record the
horizon-scaled step size.

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
