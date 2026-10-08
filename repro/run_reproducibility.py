#!/usr/bin/env python3
"""Reproduce the Paper 3 DP headline runs and horizon diagnostics.

The repository contains small research simulators rather than a package.  This
driver is the single entry point used to regenerate the headline tables and
the box-LS horizon sweeps.  It records every numerical setting in the output
manifest, including state bounds, accountant, privacy budget, seeds, and
step-size policy.

Run from the repository root after the simulators have been placed in
``code/``::

    python repro/run_reproducibility.py headline --output experiments/repro_headline.json
    python repro/run_reproducibility.py horizon --output experiments/repro_horizon.json

For the current scratch layout, ``--code-dir .`` is also accepted.  The
driver never infers settings from an existing JSON file; all settings are
constructed here and included in the resulting manifest.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


METHODS = (
    "nonprivate_bounded_EControl_TopK",
    "centralDP_bounded_EControl_TopK",
    "centralDP_DA_no_compression",
    "centralDP_DA_no_compression_matched_sensitivity",
)


def _find_code_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"code directory does not exist: {path}")
        return path
    here = Path(__file__).resolve()
    candidates = [here.parent.parent / "code", here.parent.parent, Path.cwd() / "code", Path.cwd()]
    for path in candidates:
        if (path / "p3_box_ls_sim.py").exists():
            return path.resolve()
    raise FileNotFoundError("could not find p3_box_ls_sim.py; pass --code-dir explicitly")


def _load_modules(code_dir: Path) -> Dict[str, ModuleType]:
    sys.path.insert(0, str(code_dir))
    names = ["p3_bounded_sim", "p3_box_ls_sim", "p3_box_ls_signed_sim", "p3_softmax_sim", "p3_simplex_logistic_sim"]
    modules: Dict[str, ModuleType] = {}
    for name in names:
        modules[name] = importlib.import_module(name)
    return modules


def _ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _base_box_args(**overrides: Any) -> SimpleNamespace:
    values = dict(
        n_clients=20,
        samples_per_client=40,
        d=10,
        rounds=100,
        topk_frac=0.1,
        epsilon=8.0,
        delta_dp=1e-5,
        gamma=5.0,
        C0=1.5,
        Cg=1.5,
        Br=2.0,
        Bh=1.0,
        Be=2.0,
        Bbox=2.0,
        heterogeneity=0.2,
        noise_std=0.15,
        seed=0,
        n_seeds=3,
        accountant="without_replacement_bound",
        dense_C0=None,
    )
    values.update(overrides)
    return _ns(**values)


def _base_simplex_args(**overrides: Any) -> SimpleNamespace:
    values = dict(
        n_clients=20,
        samples_per_client=40,
        d=8,
        rounds=100,
        topk_frac=0.2,
        epsilon=8.0,
        delta_dp=1e-5,
        gamma=5.0,
        C0=1.5,
        Cg=1.5,
        Br=2.0,
        Bh=1.0,
        Be=2.0,
        heterogeneity=0.2,
        seed=0,
        n_seeds=3,
        accountant="without_replacement_bound",
        dense_C0=None,
    )
    values.update(overrides)
    return _ns(**values)


def _base_softmax_args(**overrides: Any) -> SimpleNamespace:
    values = dict(
        n_clients=20,
        samples_per_client=40,
        d=8,
        n_classes=3,
        rounds=80,
        topk_frac=0.1,
        epsilon=8.0,
        delta_dp=1e-5,
        gamma=5.0,
        l1=0.002,
        C0=1.5,
        Cg=1.5,
        Br=2.0,
        Bh=1.0,
        Be=2.0,
        heterogeneity=0.2,
        label_noise=0.08,
        seed=0,
        accountant="without_replacement_bound",
        dense_C0=None,
    )
    values.update(overrides)
    return _ns(**values)


def _epsilon_for_sigma(
    accounting_module: ModuleType,
    *,
    sigma: float,
    sensitivity: float,
    rounds: int,
    delta: float,
    accountant: str,
) -> float:
    """Return the RDP epsilon represented by a fixed per-round sigma."""
    orders = np.arange(2.0, 257.0)
    rdp = rounds * accounting_module.rdp_per_round(
        orders, 1.0, sensitivity, sigma, mode=accountant
    )
    return float(np.min(rdp + math.log(1.0 / delta) / (orders - 1.0)))


def _mean_sd(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        key: float(values.mean()),
        f"{key}_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def _summarize_softmax(runs: Mapping[str, List[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    metrics = ("objective", "accuracy", "sigma", "sensitivity", "bits_per_client", "accounted_epsilon")
    out: Dict[str, Dict[str, Any]] = {}
    for method, rows in runs.items():
        result: Dict[str, Any] = {"n_seeds": len(rows)}
        for metric in metrics:
            result.update(_mean_sd(rows, metric))
        out[method] = result
    return out


def _corrected_box_diagnostics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract theory-aligned c/rho/beta/E diagnostics from box runs.

    The simulator logs the exact signed vectors:
      c_t = H_t - mean_i u_i,t
      rho_t = mean_i(u_i,t - v_i,t)
      beta_t = mean_i(v_i,t - raw_mean_i,t)
      E_t = sum_{s<=t}(c_s + rho_s + beta_s).
    Historical ``delta - TopK(delta)`` quantities are intentionally not used.
    """
    out: Dict[str, Any] = {}
    for method, rows in result["runs"].items():
        per_seed: List[Dict[str, float]] = []
        for run in rows:
            history = run.get("history", [])
            if not history:
                continue
            e = np.zeros(len(history[0]["E_t_vec"]))
            e_clipped = np.zeros_like(e)
            identity_error = 0.0
            for h in history:
                c = np.asarray(h["c_t_vec"], dtype=float)
                rho = np.asarray(h["rho_t_vec"], dtype=float)
                beta = np.asarray(h["beta_t_vec"], dtype=float)
                e = e + c + rho + beta
                e_clipped = e_clipped + c + rho
                identity_error = max(identity_error, float(np.linalg.norm(e - np.asarray(h["E_t_vec"], dtype=float))))
            # Q_full includes the original-objective clipping bias beta.  The
            # clipped-objective diagnostic omits beta and is the quantity
            # relevant to the bounded-oracle theorem.
            q_full = float(sum(float(h["E_t_sq"]) for h in history))
            q_clipped = 0.0
            e_tmp = np.zeros_like(e)
            for h in history:
                e_tmp = e_tmp + np.asarray(h["c_t_vec"], dtype=float) + np.asarray(h["rho_t_vec"], dtype=float)
                q_clipped += float(np.dot(e_tmp, e_tmp))
            per_seed.append({
                "seed": int(run.get("seed", -1)),
                "Q_T": q_full,
                "Q_T_over_T": q_full / len(history),
                "Q_T_clipped": q_clipped,
                "Q_T_clipped_over_T": q_clipped / len(history),
                "final_E_clipped_norm": float(np.linalg.norm(e_clipped)),
                "mean_c_t_norm": float(np.mean([h["c_t_norm"] for h in history])),
                "mean_rho_t_norm": float(np.mean([h["rho_t"] for h in history])),
                "mean_beta_t_norm": float(np.mean([h["beta_t_norm"] for h in history])),
                "max_E_t_norm": float(np.max([h["E_t_norm"] for h in history])),
                "max_telescoping_identity_error": identity_error,
            })
        if per_seed:
            summary: Dict[str, Any] = {"n_seeds": len(per_seed)}
            for key in per_seed[0]:
                if key == "seed":
                    continue
                summary.update(_mean_sd(per_seed, key))
            out[method] = {"summary": summary, "per_seed": per_seed}
    return out


def run_headline(modules: Mapping[str, ModuleType], *, keep_histories: bool = True) -> Dict[str, Any]:
    """Run the three objective families with the checked headline settings."""
    # The fourth baseline uses C0=1.0 so its central-DP sensitivity matches
    # the Top-K release's 2*Bh/n when Bh=1.0.  It is reported separately from
    # the historical dense C0=1.5 result.
    box_args = _base_box_args(dense_C0=1.0)
    box_result = modules["p3_box_ls_sim"].run_suite(box_args)
    box_corrected = _corrected_box_diagnostics(box_result)
    simplex_args = _base_simplex_args(dense_C0=1.0)
    simplex_result = modules["p3_simplex_logistic_sim"].run_suite(simplex_args)

    soft_runs: Dict[str, List[Dict[str, Any]]] = {}
    soft_args = _base_softmax_args(dense_C0=1.0)
    for seed in range(3):
        result = modules["p3_softmax_sim"].run_suite(_base_softmax_args(seed=seed, dense_C0=1.0))
        for method, method_result in result.items():
            row = dict(method_result)
            row["seed"] = seed
            soft_runs.setdefault(method, []).append(row)

    if not keep_histories:
        box_result = {k: v for k, v in box_result.items() if k != "runs"}
        simplex_result = {k: v for k, v in simplex_result.items() if k != "runs"}
        soft_runs = {method: [{k: v for k, v in row.items() if k != "history"} for row in rows] for method, rows in soft_runs.items()}

    return {
        "manifest": {
            "driver": "repro/run_reproducibility.py",
            "mode": "headline",
            "seeds": [0, 1, 2],
            "accountant": "without_replacement_bound",
            "participation_rate": 1.0,
            "release": "clean aggregate plus fresh Gaussian noise",
            "sensitivity": "2*Bh/n for Top-K; 2*C0/n for dense DA",
            "step_size": "gamma=5.0; last iterate (matches committed headline runs)",
            "scope_note": "Synthetic, three-seed diagnostics; no unconditional real-iterate theorem is claimed.",
            "box_settings": vars(box_args),
            "simplex_settings": vars(simplex_args),
            "softmax_settings": vars(soft_args),
        },
        "box": {
            "summary": box_result.get("summary"),
            "corrected_diagnostics": box_corrected,
            **({"runs": box_result.get("runs")} if keep_histories else {}),
        },
        "simplex": simplex_result if keep_histories else {"metadata": simplex_result.get("metadata"), "summary": simplex_result.get("summary")},
        "softmax": {"summary": _summarize_softmax(soft_runs), "runs": soft_runs if keep_histories else {method: [{k: v for k, v in row.items() if k != "history"} for row in rows] for method, rows in soft_runs.items()}},
    }


def _run_box_case(
    module: ModuleType,
    args: SimpleNamespace,
    *,
    corrected: bool,
    keep_histories: bool = False,
) -> Dict[str, Any]:
    result = module.run_suite(args)
    payload = {
        "settings": vars(args),
        "summary": result.get("summary"),
        "corrected_diagnostics": _corrected_box_diagnostics(result) if corrected else {},
    }
    if keep_histories:
        payload["runs"] = result.get("runs")
    return payload


def run_horizon(
    modules: Mapping[str, ModuleType],
    *,
    horizons: Sequence[int] = (25, 50, 100, 200),
    target_sigma: float = 0.6931788020546441,
    gamma_mode: str = "fixed",
    keep_histories: bool = False,
) -> Dict[str, Any]:
    """Run fixed-total-budget, fixed-sigma, and signed diagnostics.

    ``fixed_total`` uses epsilon=8 for every horizon.  ``fixed_sigma``
    computes the equivalent epsilon from the same RDP accountant so that the
    released Gaussian scale is fixed at ``target_sigma``.  The signed cases
    use p3_box_ls_signed_sim and retain the same settings as the committed
    diagnostic sweeps.
    """
    cases: Dict[str, List[Dict[str, Any]]] = {name: [] for name in (
        "fixed_total", "fixed_sigma", "signed_clipped", "signed_noprojection"
    )}
    rdp = modules["p3_bounded_sim"]
    for rounds in horizons:
        common: Dict[str, Any] = {"rounds": int(rounds)}
        # Compute the calibrated gamma after determining the relevant sigma.
        total_args = _base_box_args(rounds=rounds, epsilon=8.0)
        total_sigma = rdp.gaussian_sigma(8.0, total_args.delta_dp, 2 * total_args.Bh / total_args.n_clients, rounds, 1.0, total_args.accountant)
        if gamma_mode == "noise_calibrated":
            total_args.gamma = max(5.0, total_sigma * math.sqrt(total_args.d * rounds) / 1.3)
        cases["fixed_total"].append(_run_box_case(modules["p3_box_ls_sim"], total_args, corrected=True, keep_histories=keep_histories))

        fixed_eps = _epsilon_for_sigma(
            rdp, sigma=target_sigma, sensitivity=2 * total_args.Bh / total_args.n_clients,
            rounds=rounds, delta=total_args.delta_dp, accountant=total_args.accountant,
        )
        fixed_args = _base_box_args(rounds=rounds, epsilon=fixed_eps)
        if gamma_mode == "noise_calibrated":
            fixed_args.gamma = max(5.0, target_sigma * math.sqrt(fixed_args.d * rounds) / 1.3)
        cases["fixed_sigma"].append(_run_box_case(modules["p3_box_ls_sim"], fixed_args, corrected=True, keep_histories=keep_histories))

        signed_args = _base_box_args(
            rounds=rounds, epsilon=fixed_eps, C0=100.0, Cg=100.0,
        )
        cases["signed_clipped"].append(_run_box_case(modules["p3_box_ls_signed_sim"], signed_args, corrected=True, keep_histories=keep_histories))

        no_projection_args = _base_box_args(
            rounds=rounds, epsilon=_epsilon_for_sigma(
                rdp, sigma=target_sigma, sensitivity=2 * 10.0 / total_args.n_clients,
                rounds=rounds, delta=total_args.delta_dp, accountant=total_args.accountant,
            ), C0=100.0, Cg=100.0, Br=100.0, Bh=10.0, Be=100.0,
        )
        cases["signed_noprojection"].append(_run_box_case(modules["p3_box_ls_signed_sim"], no_projection_args, corrected=True, keep_histories=keep_histories))

    manifest = {
        "driver": "repro/run_reproducibility.py",
        "mode": "horizon",
        "horizons": [int(t) for t in horizons],
        "target_sigma": float(target_sigma),
        "gamma_mode": gamma_mode,
        "gamma_calibration": "max(5, sigma*sqrt(d*T)/1.3)" if gamma_mode == "noise_calibrated" else "gamma=5.0",
        "seeds": [0, 1, 2],
        "accountant": "without_replacement_bound",
        "participation_rate": 1.0,
        "state_bounds": {"Bh": 1.0, "Be": 2.0, "Br": 2.0},
        "privacy": "replacement adjacency; sensitivity 2*Bh/n for Top-K release",
        "diagnostic_definition": {
            "c_t": "H_t - mean_i(u_i,t)",
            "rho_t": "mean_i(u_i,t - v_i,t)",
            "beta_t": "mean_i(v_i,t - raw_mean_i,t)",
            "E_t": "sum_{s<=t}(c_s + rho_s + beta_s), signed vectors",
        },
        "scope_note": "The sweep is a reproducibility diagnostic; it is not a convergence proof.",
    }
    if not keep_histories:
        for cases_for_mode in cases.values():
            for item in cases_for_mode:
                # _run_box_case already strips full trajectories.
                pass
    return {"manifest": manifest, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("headline", "horizon", "all"))
    parser.add_argument("--code-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default="25,50,100,200")
    parser.add_argument("--target-sigma", type=float, default=0.6931788020546441)
    parser.add_argument("--gamma-mode", choices=("fixed", "noise_calibrated"), default="fixed")
    parser.add_argument("--keep-histories", action="store_true")
    args = parser.parse_args()
    modules = _load_modules(_find_code_dir(args.code_dir))
    payload: Dict[str, Any] = {}
    if args.mode in ("headline", "all"):
        payload["headline"] = run_headline(modules, keep_histories=args.keep_histories)
    if args.mode in ("horizon", "all"):
        horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
        payload["horizon"] = run_horizon(
            modules,
            horizons=horizons,
            target_sigma=args.target_sigma,
            gamma_mode=args.gamma_mode,
            keep_histories=args.keep_histories,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
