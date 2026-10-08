# Legacy diagnostic artifacts

The JSON files below predate the corrected telemetry definitions and are retained for provenance. They report the historical Top-K residual or accumulated projection-norm proxy, not the theorem's cumulative error vector `E_t`. The proxy can under-report when signed terms cancel or reinforce, so these files must not be used as evidence for the clipped-objective bound.

Use the regenerated manifests in `experiments/` for current results:

- `experiments/repro_headline_20261008.json`
- `experiments/repro_horizon_fixed_20261008.json`
- `experiments/repro_horizon_noise_calibrated_20261008.json`
- `experiments/tele_scope_softmax_simplex_audit.json`
- `experiments/tele_scope_sign_stress.json`

Legacy files:

- `experiments/box_T_sweep_summary.json`
- `experiments/box_fixed_sigma_sweep_summary.json`
- `experiments/box_clipped_objective_sweep_summary.json`
- `experiments/box_signed_clipped_sweep_summary.json`
- `experiments/box_signed_noprojection_sweep_summary.json`

The signed no-projection diagnostic is also high-epsilon/effectively non-private in the historical runs. It is useful for checking algebra and projection behavior, not for comparing private utility.
