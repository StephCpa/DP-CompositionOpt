#!/usr/bin/env python3
"""Audit theorem-aligned c/rho/beta/E vectors for all three simulators.

This is an algebraic implementation check.  It does not establish a
convergence theorem or a sensitivity bound.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np


def load(code_dir: Path):
    sys.path.insert(0, str(code_dir.resolve()))
    return {name: importlib.import_module(name) for name in (
        "p3_box_ls_sim", "p3_softmax_sim", "p3_simplex_logistic_sim"
    )}


def check_run(run: Dict[str, Any]) -> Dict[str, float]:
    hist = run.get("history", [])
    if not hist:
        return {"rounds": 0, "max_E_identity_error": float("nan")}
    e = np.zeros(len(hist[0]["E_t_vec"]))
    e_clip = np.zeros_like(e)
    q_full = 0.0
    q_clipped = 0.0
    max_err = 0.0
    for h in hist:
        c = np.asarray(h["c_t_vec"], dtype=float)
        rho = np.asarray(h["rho_t_vec"], dtype=float)
        beta = np.asarray(h["beta_t_vec"], dtype=float)
        e += c + rho + beta
        e_clip += c + rho
        q_full += float(np.dot(e, e))
        q_clipped += float(np.dot(e_clip, e_clip))
        max_err = max(max_err, float(np.linalg.norm(e - np.asarray(h["E_t_vec"], dtype=float))))
    return {
        "rounds": len(hist),
        "max_E_identity_error": max_err,
        "Q_T": q_full,
        "Q_T_clipped": q_clipped,
        "Q_T_over_T": q_full / len(hist),
        "Q_T_clipped_over_T": q_clipped / len(hist),
        "max_E_t_norm": max(float(h["E_t_norm"]) for h in hist),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    m = load(Path(args.code_dir))
    out: Dict[str, Any] = {"metadata": {
        "purpose": "algebraic audit of theorem-aligned signed diagnostics",
        "definition": {
            "c_t": "H_t - mean_i(u_i,t)",
            "rho_t": "mean_i(u_i,t - v_i,t)",
            "beta_t": "mean_i(v_i,t - raw_mean_i,t)",
            "E_t": "sum_{s<=t}(c_s + rho_s + beta_s)",
        },
        "seeds": list(range(args.seeds)),
        "rounds": args.rounds,
        "scope_note": "Algebraic audit only; no convergence or privacy proof.",
    }, "results": {}}
    # Box and simplex use their existing suite wrappers.
    box_args = SimpleNamespace(n_clients=20, samples_per_client=40, d=10,
        rounds=args.rounds, topk_frac=.1, epsilon=8., delta_dp=1e-5,
        gamma=5., C0=1.5, Cg=1.5, Br=2., Bh=1., Be=2., Bbox=2.,
        heterogeneity=.2, noise_std=.15, seed=0, n_seeds=args.seeds,
        accountant="without_replacement_bound")
    box = m["p3_box_ls_sim"].run_suite(box_args)
    out["results"]["box"] = {name: [check_run(r) for r in rows] for name, rows in box["runs"].items()}
    simplex_args = SimpleNamespace(n_clients=20, samples_per_client=40, d=8,
        rounds=args.rounds, topk_frac=.2, epsilon=8., delta_dp=1e-5,
        gamma=5., C0=1.5, Cg=1.5, Br=2., Bh=1., Be=2., heterogeneity=.2,
        seed=0, n_seeds=args.seeds, accountant="without_replacement_bound")
    simplex = m["p3_simplex_logistic_sim"].run_suite(simplex_args)
    out["results"]["simplex"] = {name: [check_run(r) for r in rows] for name, rows in simplex["runs"].items()}
    soft_runs: Dict[str, list] = {}
    for seed in range(args.seeds):
        soft_args = SimpleNamespace(n_clients=20, samples_per_client=40, d=8,
            n_classes=3, rounds=args.rounds, topk_frac=.1, epsilon=8.,
            delta_dp=1e-5, gamma=5., l1=.002, C0=1.5, Cg=1.5, Br=2.,
            Bh=1., Be=2., heterogeneity=.2, label_noise=.08, seed=seed,
            accountant="without_replacement_bound")
        result = m["p3_softmax_sim"].run_suite(soft_args)
        for name, row in result.items():
            soft_runs.setdefault(name, []).append(check_run(row))
    out["results"]["softmax"] = soft_runs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default="code")
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    payload = run(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
