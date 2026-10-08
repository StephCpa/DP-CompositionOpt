# Reproducibility drivers

Run the following seven commands from the repository root to regenerate the
committed diagnostic outputs:

```bash
python repro/run_reproducibility.py headline \
  --code-dir . \
  --output experiments/repro_headline_20261008.json

python repro/run_reproducibility.py horizon \
  --code-dir . \
  --horizons 25,50,100,200 \
  --gamma-mode fixed \
  --output experiments/repro_horizon_fixed_20261008.json

python repro/run_reproducibility.py horizon \
  --code-dir . \
  --horizons 25,50,100,200 \
  --gamma-mode noise_calibrated \
  --noise-coefficient 2 \
  --minimum-gamma 5 \
  --output experiments/repro_horizon_noise_calibrated_20261008.json

python repro/audit_telescoping.py \
  --code-dir code \
  --rounds 40 \
  --seeds 3 \
  --output experiments/softmax_simplex_telescoping_audit.json

python repro/tele_scope_softmax_simplex_audit.py \
  --output experiments/tele_scope_softmax_simplex_audit.json

python repro/generate_sign_stress.py \
  --output experiments/tele_scope_sign_stress.json

python repro/state_margin_probe.py \
  --output experiments/state_margin_probe_20261008.json
```

The `audit_telescoping.py` command uses `--rounds 40` because that is the
round count in the committed artifact. Its default is 80, which is useful for
an independent longer audit but does not regenerate that file byte for byte.

When the simulators are in the repository's `code/` directory, the driver
finds them automatically. `--code-dir .` is accepted for the repository root
and for the historical flat checkout. The driver records the Python and NumPy
versions and SHA-256 hashes of the driver and imported simulator sources in
each manifest.

## Headline and horizon runs

`run_reproducibility.py headline` regenerates the softmax+ℓ1, box least
squares, and simplex logistic headline experiments with seeds 0, 1, and 2,
including a dense central-DP baseline at matched `C0`.

`run_reproducibility.py horizon` regenerates fixed-total-privacy,
fixed-per-round-noise, an unclipped-gradient signed-state diagnostic, and a
large-threshold signed-state diagnostic for `T = 25, 50, 100, 200`.

The headline and horizon summaries use the uniform post-update average as the
primary iterate:

```text
primary iterate:  x̄_T = T^-1 sum_{t=1}^T x_t
legacy iterate:   x_T   (retained in the existing scalar fields)
```

The existing scalar fields such as `objective`, `test_mse`, `accuracy`, and
parameter norms continue to refer to the last iterate for backward
compatibility. They are never silently overwritten. Each result now includes:

- `primary_summary`: averaged-iterate metrics, with `iterate` set to
  `uniform_average`;
- `legacy_last_iterate_summary`: explicitly labelled last-iterate metrics;
- `average_iterates`: the exact averaged parameter vector or matrix when the
  simulator exposes it.

The averaged-iterate metrics are the quantities to use when comparing against
the theorem's averaged-iterate statement.

## Step-size calibration

Use `--gamma-mode noise_calibrated` to apply the explicit rule

```text
gamma = max(minimum_gamma,
            noise_coefficient * sigma * sqrt(d*T) / reference_radius).
```

The default `noise_coefficient=2` comes from the stationary point of the
two-term proxy

```text
gamma R0^2/(2T) + 2 d sigma^2/gamma.
```

This is an experiment calibration rule, not a claim that the complete
real-iterate theorem is optimized by this expression. The full bound can also
contain error-feedback, initialization, clipping-bias, and other terms.

`reference_radius` is an explicit ℓ2 distance scale. For the box task with
`x0=0` and `x*` in `[-Bbox, Bbox]^d`, the default is

```text
reference_radius = Bbox * sqrt(d).
```

The historical driver used coefficient 1 and `reference_radius=2.0`. To
reproduce that calibration, run:

```bash
python repro/run_reproducibility.py horizon \
  --code-dir . \
  --horizons 25,50,100,200 \
  --gamma-mode noise_calibrated \
  --noise-coefficient 1 \
  --reference-radius 2.0 \
  --output experiments/repro_horizon_noise_calibrated_legacy_driver.json
```

The older generator-informed calibration used the same coefficient 1 with
`--reference-radius 1.3`:

```bash
python repro/run_reproducibility.py horizon \
  --code-dir . \
  --horizons 25,50,100,200 \
  --gamma-mode noise_calibrated \
  --noise-coefficient 1 \
  --reference-radius 1.3 \
  --output experiments/repro_horizon_noise_calibrated_legacy_1p3.json
```

All calibration flags and their resolved values are stored in the horizon
manifest, including `noise_coefficient`, `minimum_gamma`,
`reference_radius`, and the calibration note.

## Signed diagnostics and privacy scope

The signed diagnostics use the theory-aligned vectors

```text
c_t    = H_t - mean_i(u_i,t)
rho_t  = mean_i(u_i,t - v_i,t)
beta_t = mean_i(v_i,t - raw_mean_i,t)
E_t    = sum_{s<=t} (c_s + rho_s + beta_s)
```

The case formerly called `signed_clipped` is now called
`signed_unclipped`. It uses `C0=Cg=100`, which is intended to stress the
unclipped-gradient regime. Those are finite thresholds: clipping remains
mathematically possible, and the `h/e/r` state projections remain active.

The `signed_noprojection` case uses finite large thresholds
`C0=Cg=Br=Be=100` and `Bh=10`. It is intended as a no-state-projection
diagnostic, but the projections are not mathematically removed. The output
records whether they were active on the tested traces. Because this case has
large privacy loss at the reported horizons, it is a diagnostic rather than a
DP utility baseline.

`tele_scope_softmax_simplex_audit.py` independently reconstructs the
softmax/simplex local updates and checks the signed telescoping identities.
`generate_sign_stress.py` deliberately runs with `private=False` and
`sigma=0`; its rows are named `softmax_nonprivate` and `simplex_nonprivate`.
That file tests projection-residual signs and is not a DP result.

The audit is an algebraic implementation check. It does not establish a
convergence theorem, privacy bound, or sensitivity result.

The calibration regression checks can be run without generating artifacts:

```bash
python repro/test_calibration.py -v
```
