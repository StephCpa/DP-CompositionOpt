#!/usr/bin/env python3
"""Probe the state margins behind the Paper 3 EControl closure argument.

This script deliberately wraps the existing box-LS simulator rather than
reimplementing its update.  It records the clean local input ``u``, the raw
and projected state vectors, movement, and the signed e/r projection
residuals.  The output is an empirical diagnostic; it is not a privacy or
convergence theorem.

Run from the repository root::

    python repro/state_margin_probe.py \
      --output experiments/state_margin_probe_20261008.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _load_box_module(code_dir: str | None):
    here = Path(__file__).resolve()
    candidates = []
    if code_dir:
        candidates.append(Path(code_dir).expanduser().resolve())
    candidates.extend([here.parent.parent / "code", here.parent.parent, Path.cwd() / "code", Path.cwd()])
    for path in candidates:
        if (path / "p3_box_ls_sim.py").exists():
            sys.path.insert(0, str(path.resolve()))
            import p3_box_ls_sim as box  # type: ignore

            return box
    raise FileNotFoundError("could not find code/p3_box_ls_sim.py; pass --code-dir")


def _finite(value: Any) -> bool:
    """Return whether a nested diagnostic contains only finite numbers."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(np.isfinite(value))
    if isinstance(value, Mapping):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return True
    return False


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=float))


class RecordingBoxLSEControlDA:
    """Thin recording wrapper around ``BoxLSEControlDA``.

    The wrapped simulator remains responsible for every state update.  The
    wrapper reconstructs raw states from the returned projection residuals:
    ``raw = new + (raw - new)``.  This avoids copying the update rule here.
    """

    def __init__(self, box: Any, clients: Any, config: Any):
        self.box = box
        self.impl = box.BoxLSEControlDA(clients, config)
        self.round_records: List[Dict[str, Any]] = []
        self._client_records: List[Dict[str, Any]] = []
        self._previous_u: List[np.ndarray | None] = [None] * self.impl.n
        self._prefix_pe_minus_pr = np.zeros(self.impl.d)
        self._prefix_pe_total = 0.0
        self._prefix_pr_total = 0.0
        self._prefix_p_norms: List[float] = []
        self._eclip = np.zeros(self.impl.d)
        self._original_local_update = None

    def _local_update(self, i: int):
        old_h = self.impl.h[i].copy()
        old_e = self.impl.e[i].copy()
        old_r = self.impl.r[i].copy()
        if self._original_local_update is None:
            raise RuntimeError("recording wrapper was called outside step()")
        stats = self._original_local_update(i)

        h_new = self.impl.h[i].copy()
        e_new = self.impl.e[i].copy()
        r_new = self.impl.r[i].copy()
        h_raw = h_new + np.asarray(stats["h_clip_vec"], dtype=float)
        e_raw = e_new + np.asarray(stats["e_clip_vec"], dtype=float)
        r_raw = r_new + np.asarray(stats["r_clip_vec"], dtype=float)
        u = np.asarray(stats["u_vec"], dtype=float)
        previous_u = self._previous_u[i]
        u_delta = 0.0 if previous_u is None else float(np.linalg.norm(u - previous_u))
        self._previous_u[i] = u.copy()

        self._client_records.append({
            "client": int(i),
            "u": u.tolist(),
            "u_norm": float(np.linalg.norm(u)),
            "u_delta_norm": u_delta,
            "old_h_norm": float(np.linalg.norm(old_h)),
            "old_e_norm": float(np.linalg.norm(old_e)),
            "old_r_norm": float(np.linalg.norm(old_r)),
            "h_raw_norm": float(np.linalg.norm(h_raw)),
            "h_norm": float(np.linalg.norm(h_new)),
            "e_raw_norm": float(np.linalg.norm(e_raw)),
            "e_norm": float(np.linalg.norm(e_new)),
            "r_raw_norm": float(np.linalg.norm(r_raw)),
            "r_norm": float(np.linalg.norm(r_new)),
            "h_projection_vec": np.asarray(stats["h_clip_vec"], dtype=float).tolist(),
            "e_projection_vec": np.asarray(stats["e_clip_vec"], dtype=float).tolist(),
            "r_projection_vec": np.asarray(stats["r_clip_vec"], dtype=float).tolist(),
        })
        return stats

    def step(self):
        self._client_records = []
        original = self.impl._local_update
        self._original_local_update = original
        self.impl._local_update = self._local_update  # type: ignore[method-assign]
        try:
            self.impl.step()
        finally:
            self.impl._local_update = original  # type: ignore[method-assign]
            self._original_local_update = None

        history = self.impl.history[-1]
        records = self._client_records
        pe_bar = np.mean([np.asarray(r["e_projection_vec"], dtype=float) for r in records], axis=0)
        pr_bar = np.mean([np.asarray(r["r_projection_vec"], dtype=float) for r in records], axis=0)
        self._prefix_pe_minus_pr += pe_bar - pr_bar
        self._prefix_pe_total += float(np.linalg.norm(pe_bar))
        self._prefix_pr_total += float(np.linalg.norm(pr_bar))
        self._prefix_p_norms.append(float(np.linalg.norm(self._prefix_pe_minus_pr)))

        c = np.asarray(history["c_t_vec"], dtype=float)
        rho = np.asarray(history["rho_t_vec"], dtype=float)
        self._eclip += c + rho
        mean_e = np.mean(self.impl.e, axis=0)
        mean_r = np.mean(self.impl.r, axis=0)
        mean_h = np.mean(self.impl.h, axis=0)
        self.round_records.append({
            "round": int(history["round"]),
            "mean_u_norm": float(np.linalg.norm(np.mean([np.asarray(r["u"], dtype=float) for r in records], axis=0))),
            "max_u_norm": float(max(r["u_norm"] for r in records)),
            "max_u_delta_norm": float(max(r["u_delta_norm"] for r in records)),
            "mean_u_delta_norm": float(np.mean([r["u_delta_norm"] for r in records])),
            "max_h_raw_norm": float(max(r["h_raw_norm"] for r in records)),
            "max_h_norm": float(max(r["h_norm"] for r in records)),
            "max_e_raw_norm": float(max(r["e_raw_norm"] for r in records)),
            "max_e_norm": float(max(r["e_norm"] for r in records)),
            "max_r_raw_norm": float(max(r["r_raw_norm"] for r in records)),
            "max_r_norm": float(max(r["r_norm"] for r in records)),
            "mean_h_norm": float(np.linalg.norm(mean_h)),
            "mean_e_norm": float(np.linalg.norm(mean_e)),
            "mean_r_norm": float(np.linalg.norm(mean_r)),
            "h_projection_mean_norm": float(np.linalg.norm(np.mean([
                np.asarray(r["h_projection_vec"], dtype=float) for r in records
            ], axis=0))),
            "e_projection_mean_norm": float(np.linalg.norm(pe_bar)),
            "r_projection_mean_norm": float(np.linalg.norm(pr_bar)),
            "p_prefix_signed_norm": float(np.linalg.norm(self._prefix_pe_minus_pr)),
            "p_prefix_total_variation": float(self._prefix_pe_total + self._prefix_pr_total),
            "eclip_norm": float(np.linalg.norm(self._eclip)),
            "mean_e_identity_error": float(np.linalg.norm(self._eclip - mean_e)),
            "history_movement": float(history["movement"]),
            "noise_norm": float(history["noise_norm"]),
        })

    def run(self, test: Tuple[np.ndarray, np.ndarray], w_true: np.ndarray) -> Dict[str, Any]:
        for _ in range(int(self.impl.cfg.rounds)):
            self.step()
        # Finish the run without calling ``impl.run`` again: that method also
        # performs the update loop, while this wrapper has already recorded
        # every round above.
        Xtest, ytest = test
        test_loss, _ = self.box.ls_loss_and_grad(Xtest, ytest, self.impl.x)
        train_X = np.concatenate([xy[0] for xy in self.impl.clients], axis=0)
        train_y = np.concatenate([xy[1] for xy in self.impl.clients], axis=0)
        train_loss, _ = self.box.ls_loss_and_grad(train_X, train_y, self.impl.x)
        rounds = max(int(self.impl.cfg.rounds), 1)
        x_average = self.impl.x_sum / rounds
        average_test_loss, _ = self.box.ls_loss_and_grad(Xtest, ytest, x_average)
        average_train_loss, _ = self.box.ls_loss_and_grad(train_X, train_y, x_average)
        last = self.impl.history[-1]
        bits_per_value = 32
        index_bits = int(np.ceil(np.log2(max(self.impl.d, 2))))
        values_per_round = self.impl.k if self.impl.cfg.compression else self.impl.d
        bits_per_client = int(self.impl.cfg.rounds * (
            values_per_round * (bits_per_value + index_bits)
            if self.impl.cfg.compression else self.impl.d * bits_per_value
        ))
        result: Dict[str, Any] = {
            "objective": float(test_loss), "test_mse": float(2.0 * test_loss),
            "train_mse": float(2.0 * train_loss),
            "parameter_mse": float(np.mean((self.impl.x - w_true) ** 2)),
            "average_objective": float(average_test_loss),
            "average_test_mse": float(2.0 * average_test_loss),
            "average_train_mse": float(2.0 * average_train_loss),
            "average_parameter_mse": float(np.mean((x_average - w_true) ** 2)),
            "average_x_norm": float(np.linalg.norm(x_average)),
            "average_iterate": x_average.tolist(),
            "iterate_reporting": "last_and_uniform_average",
            "x_norm": float(np.linalg.norm(self.impl.x)),
            "sigma": float(self.impl.sigma), "sensitivity": float(self.impl.sensitivity),
            "epsilon": float(self.impl.cfg.epsilon if self.impl.cfg.private else 0.0),
            "delta": float(self.impl.cfg.delta_dp if self.impl.cfg.private else 0.0),
            "accounted_epsilon": float(self.impl.accounted_epsilon()),
            "n_clients": int(self.impl.n), "d": int(self.impl.d),
            "rounds": int(self.impl.cfg.rounds), "topk": int(self.impl.k),
            "topk_frac": float(self.impl.cfg.topk_frac),
            "bits_per_client": bits_per_client,
            "history": self.impl.history,
        }
        all_h_raw = np.asarray([r["max_h_raw_norm"] for r in self.round_records])
        all_e_raw = np.asarray([r["max_e_raw_norm"] for r in self.round_records])
        all_e = np.asarray([r["max_e_norm"] for r in self.round_records])
        all_r_raw = np.asarray([r["max_r_raw_norm"] for r in self.round_records])
        rounds = len(self.round_records)
        first = max(1, min(10, rounds))
        late_start = max(0, rounds - max(1, rounds // 4))
        cfg = self.impl.cfg
        result["state_margin_probe"] = {
            "max_h_raw_norm": float(np.max(all_h_raw)),
            "h_margin_to_Bh": float(cfg.Bh - np.max(all_h_raw)),
            "max_e_raw_norm": float(np.max(all_e_raw)),
            "max_e_norm": float(np.max(all_e)),
            "e_margin_to_Be": None if not np.isfinite(cfg.Be) else float(cfg.Be - np.max(all_e_raw)),
            "max_r_raw_norm": float(np.max(all_r_raw)),
            "r_margin_to_Br": None if not np.isfinite(cfg.Br) else float(cfg.Br - np.max(all_r_raw)),
            "startup_max_e_raw_norm": float(np.max(all_e_raw[:first])),
            "late_max_e_raw_norm": float(np.max(all_e_raw[late_start:])),
            "max_u_delta_norm": float(max(r["max_u_delta_norm"] for r in self.round_records)),
            "mean_u_delta_norm": float(np.mean([r["mean_u_delta_norm"] for r in self.round_records])),
            "max_x_movement": float(max(r["history_movement"] for r in self.round_records)),
            "sum_x_movement": float(sum(r["history_movement"] for r in self.round_records)),
            "max_p_prefix_signed_norm": float(max(self._prefix_p_norms)),
            "endpoint_p_prefix_signed_norm": float(self._prefix_p_norms[-1]),
            "p_prefix_total_variation": float(self._prefix_pe_total + self._prefix_pr_total),
            "max_e_projection_mean_norm": float(max(r["e_projection_mean_norm"] for r in self.round_records)),
            "max_r_projection_mean_norm": float(max(r["r_projection_mean_norm"] for r in self.round_records)),
            "max_mean_e_identity_error": float(max(r["mean_e_identity_error"] for r in self.round_records)),
            "no_h_projection_observed": bool(np.max(all_h_raw) <= cfg.Bh + 1e-12),
            "no_e_projection_observed": bool(np.max(all_e_raw) <= cfg.Be + 1e-12) if np.isfinite(cfg.Be) else True,
            "no_r_projection_observed": bool(np.max(all_r_raw) <= cfg.Br + 1e-12) if np.isfinite(cfg.Br) else True,
            "rounds": int(rounds),
            "per_round": self.round_records,
        }
        return result


def _base_config(box: Any, **overrides: Any):
    values = dict(
        n_clients=20,
        samples_per_client=40,
        d=10,
        rounds=100,
        topk_frac=0.1,
        eta=0.1,
        C0=1.5,
        Cg=1.5,
        Br=2.0,
        Bh=1.0,
        Be=2.0,
        Bbox=2.0,
        gamma=5.0,
        epsilon=8.0,
        delta_dp=1e-5,
        seed=0,
        private=True,
        compression=True,
        residual_buffer=True,
        accountant="without_replacement_bound",
    )
    values.update(overrides)
    return box.Config(**values)


def _run_case(box: Any, name: str, config: Any, seed: int, *, override_input: np.ndarray | None = None) -> Dict[str, Any]:
    clients, test, w_true = box.make_box_ls_data(
        config.n_clients, config.samples_per_client, config.d,
        seed=seed, heterogeneity=0.2, noise_std=0.15, box_radius=config.Bbox,
    )
    if override_input is not None:
        class ConstantInput(RecordingBoxLSEControlDA):
            def _constant_gradient(self, i: int):
                vec = override_input.copy()
                return {
                    "v_vec": vec,
                    "u_vec": box.clip_vec(vec, self.impl.cfg.Cg),
                    "raw_mean_vec": vec.copy(),
                }

            def _local_update(self, i: int):
                old_h = self.impl.h[i].copy()
                old_e = self.impl.e[i].copy()
                old_r = self.impl.r[i].copy()
                # The actual update still supplies all state transitions.
                # Only the oracle is replaced by a deterministic bounded input
                # for this explicitly labeled stress test.
                original = self.impl._clipped_gradient
                self.impl._clipped_gradient = lambda _i: (
                    override_input.copy(), override_input.copy(), 0.0
                )
                if self._original_local_update is None:
                    raise RuntimeError("constant-input wrapper was called outside step()")
                try:
                    stats = self._original_local_update(i)
                finally:
                    self.impl._clipped_gradient = original
                h_new, e_new, r_new = self.impl.h[i].copy(), self.impl.e[i].copy(), self.impl.r[i].copy()
                h_raw = h_new + np.asarray(stats["h_clip_vec"], dtype=float)
                e_raw = e_new + np.asarray(stats["e_clip_vec"], dtype=float)
                r_raw = r_new + np.asarray(stats["r_clip_vec"], dtype=float)
                u = np.asarray(stats["u_vec"], dtype=float)
                previous_u = self._previous_u[i]
                u_delta = 0.0 if previous_u is None else float(np.linalg.norm(u - previous_u))
                self._previous_u[i] = u.copy()
                self._client_records.append({
                    "client": int(i), "u": u.tolist(), "u_norm": float(np.linalg.norm(u)),
                    "u_delta_norm": u_delta, "old_h_norm": float(np.linalg.norm(old_h)),
                    "old_e_norm": float(np.linalg.norm(old_e)), "old_r_norm": float(np.linalg.norm(old_r)),
                    "h_raw_norm": float(np.linalg.norm(h_raw)), "h_norm": float(np.linalg.norm(h_new)),
                    "e_raw_norm": float(np.linalg.norm(e_raw)), "e_norm": float(np.linalg.norm(e_new)),
                    "r_raw_norm": float(np.linalg.norm(r_raw)), "r_norm": float(np.linalg.norm(r_new)),
                    "h_projection_vec": np.asarray(stats["h_clip_vec"], dtype=float).tolist(),
                    "e_projection_vec": np.asarray(stats["e_clip_vec"], dtype=float).tolist(),
                    "r_projection_vec": np.asarray(stats["r_clip_vec"], dtype=float).tolist(),
                })
                return stats

        recorder = ConstantInput(box, clients, config)
    else:
        recorder = RecordingBoxLSEControlDA(box, clients, config)
    result = recorder.run(test, w_true)
    result["case"] = name
    result["config"] = {
        "rounds": int(config.rounds), "d": int(config.d), "n_clients": int(config.n_clients),
        "gamma": float(config.gamma), "C0": float(config.C0), "Cg": float(config.Cg),
        "Bh": float(config.Bh), "Br": float(config.Br),
        "Be": "inf" if not np.isfinite(config.Be) else float(config.Be),
        "Bbox": float(config.Bbox), "topk_frac": float(config.topk_frac),
        "private": bool(config.private), "epsilon": float(config.epsilon),
        "delta_dp": float(config.delta_dp), "seed": int(seed),
    }
    return result


def _summarize_case(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def vals(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=float)

    first = rows[0]
    keys = [
        "objective", "average_objective", "test_mse", "average_test_mse",
        "sigma", "sensitivity", "accounted_epsilon",
    ]
    probe_keys = [
        "max_h_raw_norm", "h_margin_to_Bh", "max_e_raw_norm", "max_e_norm",
        "startup_max_e_raw_norm", "late_max_e_raw_norm", "max_u_delta_norm",
        "mean_u_delta_norm", "max_x_movement", "max_p_prefix_signed_norm",
        "endpoint_p_prefix_signed_norm", "p_prefix_total_variation",
        "max_e_projection_mean_norm", "max_r_projection_mean_norm",
        "max_mean_e_identity_error",
    ]
    out: Dict[str, Any] = {"n_seeds": len(rows), "config": first["config"]}
    for key in keys + probe_keys:
        if key in first:
            series = vals(key)
        else:
            series = np.asarray([row["state_margin_probe"][key] for row in rows], dtype=float)
        out[key] = float(np.mean(series))
        out[f"{key}_sd"] = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
    out["no_h_projection_observed_all_seeds"] = bool(all(row["state_margin_probe"]["no_h_projection_observed"] for row in rows))
    out["no_e_projection_observed_all_seeds"] = bool(all(row["state_margin_probe"]["no_e_projection_observed"] for row in rows))
    out["no_r_projection_observed_all_seeds"] = bool(all(row["state_margin_probe"]["no_r_projection_observed"] for row in rows))
    return out


def _run_all(box: Any, seeds: Iterable[int]) -> Dict[str, Any]:
    seeds = list(seeds)
    cases: Dict[str, List[Dict[str, Any]]] = {}
    base = _base_config(box)
    # Instantiate once to obtain the exact sigma used by the simulator for
    # the requested public configuration.
    probe_clients, _, _ = box.make_box_ls_data(20, 40, 10, seed=0, box_radius=2.0)
    sigma_reference = RecordingBoxLSEControlDA(box, probe_clients, base).impl.sigma
    old_gamma = max(5.0, sigma_reference * math.sqrt(base.d * base.rounds) / 2.0)
    corrected_radius = base.Bbox * math.sqrt(base.d)
    corrected_gamma = max(5.0, 2.0 * sigma_reference * math.sqrt(base.d * base.rounds) / corrected_radius)

    case_configs = {
        "headline_gamma5_Be2": base,
        "headline_gamma5_BeInf": _base_config(box, Be=float("inf")),
        "old_calibration_radius2_factor1": _base_config(box, gamma=old_gamma),
        "corrected_geometry_radius_Bbox_sqrt_d_factor2": _base_config(box, gamma=corrected_gamma),
        # With full coordinates sent and Cg=C0<Bh, h follows u and the
        # margin is intentionally explicit. This is an empirical check.
        "no_activation_margin_C0_less_Bh": _base_config(
            box, topk_frac=1.0, eta=1.0, C0=0.5, Cg=0.5, Be=float("inf"), Br=2.0,
        ),
    }
    for name, cfg in case_configs.items():
        cases[name] = [_run_case(box, name, cfg, seed) for seed in seeds]

    overload_cfg = _base_config(
        box, n_clients=4, samples_per_client=2, d=1, rounds=40,
        topk_frac=1.0, eta=1.0, C0=2.0, Cg=2.0, Bh=1.0,
        Be=float("inf"), Br=2.0, Bbox=2.0, private=False,
    )
    overload_rows = []
    for seed in seeds:
        overload_rows.append(_run_case(box, "overload_constant_input_norm2_Bh1", overload_cfg, seed, override_input=np.asarray([2.0])))
    cases["overload_constant_input_norm2_Bh1"] = overload_rows

    checks: Dict[str, Any] = {}
    finite_rows = all(_finite(row) for rows in cases.values() for row in rows)
    checks["all_serialized_numeric_values_finite"] = {"passed": bool(finite_rows)}

    finite = cases["headline_gamma5_Be2"][0]
    infinite = cases["headline_gamma5_BeInf"][0]
    checks["Be_does_not_change_sigma"] = {
        "passed": bool(abs(finite["sigma"] - infinite["sigma"]) <= 1e-12),
        "finite_Be_sigma": float(finite["sigma"]), "infinite_Be_sigma": float(infinite["sigma"]),
    }
    checks["Be_does_not_change_sensitivity"] = {
        "passed": bool(abs(finite["sensitivity"] - infinite["sensitivity"]) <= 1e-12),
        "finite_Be_sensitivity": float(finite["sensitivity"]), "infinite_Be_sensitivity": float(infinite["sensitivity"]),
    }
    inf_probe = infinite["state_margin_probe"]
    checks["Eclip_equals_mean_e_when_e_r_projections_inactive"] = {
        "passed": bool(inf_probe["max_mean_e_identity_error"] <= 1e-12 and inf_probe["max_e_projection_mean_norm"] <= 1e-12 and inf_probe["max_r_projection_mean_norm"] <= 1e-12),
        "max_identity_error": float(inf_probe["max_mean_e_identity_error"]),
        "max_e_projection": float(inf_probe["max_e_projection_mean_norm"]),
        "max_r_projection": float(inf_probe["max_r_projection_mean_norm"]),
    }
    margin_rows = cases["no_activation_margin_C0_less_Bh"]
    checks["C0_less_Bh_no_projection_observed"] = {
        "passed": bool(all(row["state_margin_probe"]["no_h_projection_observed"] and row["state_margin_probe"]["no_e_projection_observed"] for row in margin_rows)),
        "C0": 0.5, "Bh": 1.0,
    }
    calibrated_names = [
        "old_calibration_radius2_factor1",
        "corrected_geometry_radius_Bbox_sqrt_d_factor2",
    ]
    checks["calibrated_h_and_r_projections_inactive_observed"] = {
        "passed": bool(all(
            row["state_margin_probe"]["no_h_projection_observed"]
            and row["state_margin_probe"]["no_r_projection_observed"]
            for name in calibrated_names for row in cases[name]
        )),
        "cases": calibrated_names,
        "scope": "empirical finite-seed observation, not a theorem",
    }
    overload = cases["overload_constant_input_norm2_Bh1"][0]["state_margin_probe"]
    e_series = np.asarray([r["max_e_norm"] for r in overload["per_round"]], dtype=float)
    rounds = np.arange(1, len(e_series) + 1, dtype=float)
    slope, intercept = np.polyfit(rounds, e_series, 1)
    fitted = slope * rounds + intercept
    ss_res = float(np.sum((e_series - fitted) ** 2))
    ss_tot = float(np.sum((e_series - np.mean(e_series)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    checks["overload_e_growth_is_linear"] = {
        "passed": bool(slope > 0.5 and r2 > 0.99),
        "slope": float(slope), "r2": float(r2), "final_e_norm": float(e_series[-1]),
    }

    summaries = {name: _summarize_case(rows) for name, rows in cases.items()}
    return {
        "metadata": {
            "task": "Paper 3 DP EControl state-margin and telescoping probe",
            "description": "Wraps the actual BoxLSEControlDA update; all conclusions are empirical diagnostics.",
            "command": "python repro/state_margin_probe.py --output experiments/state_margin_probe_20261008.json",
            "seeds": seeds,
            "state_update_source": "code/p3_box_ls_sim.py::BoxLSEControlDA",
            "mainline": "full participation q=1, fresh Gaussian H_t release, real iterates",
            "old_calibration": {"reference_radius": 2.0, "factor": 1.0, "gamma": float(old_gamma)},
            "corrected_geometry": {"reference_radius": float(corrected_radius), "factor": 2.0, "gamma": float(corrected_gamma)},
            "infinity_encoding": "Be=inf is represented as the string 'inf' in configuration metadata; all numeric output fields are finite.",
            "overload_note": "Controlled nonprivate stress test with deterministic constant input norm 2 > Bh=1; not a private-data claim.",
        },
        "checks": checks,
        "summary": summaries,
        "runs": cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default=None)
    ap.add_argument("--output", default="experiments/state_margin_probe_20261008.json")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()
    box = _load_box_module(args.code_dir)
    result = _run_all(box, range(args.seed_start, args.seed_start + args.n_seeds))
    if not result["checks"]["all_serialized_numeric_values_finite"]["passed"]:
        raise RuntimeError("probe produced a non-finite numeric field")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for name, summary in result["summary"].items():
        probe = summary
        print(
            f"{name:48s} gamma={summary['config']['gamma']:.5f} "
            f"h_raw={probe['max_h_raw_norm']:.4f} "
            f"e_raw={probe['max_e_raw_norm']:.4f} "
            f"Pmax={probe['max_p_prefix_signed_norm']:.4f} "
            f"avg_mse={summary['average_test_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
