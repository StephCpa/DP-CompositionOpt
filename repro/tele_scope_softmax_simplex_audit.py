"""Independent audit of c_t, rho_t, beta_t and E_t for Paper 3 simulators.

This script instruments the committed SoftmaxEControlDA and
SimplexLogisticEControlDA classes without changing their source.  At every
round it independently reconstructs each local bounded-EControl update before
calling the simulator's own update, then compares the resulting states.

The theory convention is:
  c_t    = H_t - mean_i u_{i,t} = mean_i(h_{i,t} - u_{i,t})
  rho_t  = mean_i(u_{i,t} - v_{i,t})
  beta_t = mean_i(v_{i,t} - raw_mean_{i,t})
  E_t    = sum_{s<=t}(c_s + rho_s + beta_s)

where raw_mean is the unclipped per-client gradient mean.  The current
softmax and simplex simulators expose these exact quantities in their history
and accumulate the corresponding E vector.  A legacy Top-K-residual proxy is
also reconstructed only to quantify how far the old diagnostic would differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Resolve the simulator directory before importing the modules.  The
# repository keeps simulators under ``code/`` while the historical flat
# checkout keeps them beside this file.  This makes the audit runnable as
# ``python repro/tele_scope_softmax_simplex_audit.py`` from the repository
# root without relying on PYTHONPATH.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "repro" else SCRIPT_DIR
_candidates = []
for _arg_idx, _arg in enumerate(sys.argv[1:]):
    if _arg == "--code-dir" and _arg_idx + 1 < len(sys.argv[1:]):
        _candidates.append(Path(sys.argv[1:][_arg_idx + 1]))
    elif _arg.startswith("--code-dir="):
        _candidates.append(Path(_arg.split("=", 1)[1]))
if os.environ.get("P3_CODE_DIR"):
    _candidates.append(Path(os.environ["P3_CODE_DIR"]))
if (REPO_ROOT / "code").is_dir():
    _candidates.append(REPO_ROOT / "code")
_candidates.extend([SCRIPT_DIR, REPO_ROOT])
CODE_DIR = next(
    (p.resolve() for p in _candidates if (p / "p3_bounded_sim.py").is_file()),
    None,
)
if CODE_DIR is None:
    raise RuntimeError(
        "Could not locate p3_bounded_sim.py. Use P3_CODE_DIR or --code-dir."
    )
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np

from p3_bounded_sim import clip_vec, topk
from p3_softmax_sim import (
    Config as SoftmaxConfig,
    SoftmaxEControlDA,
    make_softmax_data,
)
from p3_simplex_logistic_sim import (
    Config as SimplexConfig,
    SimplexLogisticEControlDA,
    make_simplex_data,
)

DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "tele_scope_softmax_simplex_audit.json"


def _finite_max(vals: List[float]) -> float:
    return float(max(vals)) if vals else 0.0


def _expected_local(sim: Any, i: int) -> Dict[str, Any]:
    """Reconstruct one compressed local update from pre-update state."""
    # Both current simulators expose _clipped_gradient with the same contract:
    # (clipped client mean, unclipped/raw client mean, clipping fraction).
    out = sim._clipped_gradient(i)
    v = out[0]
    raw_mean = out[1]

    residual_buffer = bool(sim.cfg.residual_buffer)
    rbar = (
        clip_vec(sim.r[i], sim.cfg.Br)
        if residual_buffer
        else np.zeros_like(sim.r[i])
    )
    u = clip_vec(v + rbar, sim.cfg.Cg)
    hbar = clip_vec(sim.h[i], sim.cfg.Bh)
    ebar = clip_vec(sim.e[i], sim.cfg.Be)
    delta_unclipped = u - hbar - sim.eta * ebar
    delta = clip_vec(
        delta_unclipped,
        sim.cfg.Cg + sim.cfg.Bh + sim.eta * sim.cfg.Be,
    )
    q = topk(delta, sim.k)
    h_raw = sim.h[i] + q
    h_new = clip_vec(h_raw, sim.cfg.Bh)
    e_raw = ebar + h_new - u
    e_new = clip_vec(e_raw, sim.cfg.Be)
    r_raw = v + rbar - u
    r_new = (
        clip_vec(r_raw, sim.cfg.Br)
        if residual_buffer
        else np.zeros_like(sim.r[i])
    )

    # Projection residual sign used for the telescoping equations below:
    # p^a = a_new - a_raw.  The closure draft also contains the opposite
    # raw-new convention; both signs are reported by the checker.
    return {
        "v": np.array(v, copy=True),
        "raw_mean": np.array(raw_mean, copy=True),
        "u": np.array(u, copy=True),
        "delta": np.array(delta, copy=True),
        "q": np.array(q, copy=True),
        "h_raw": np.array(h_raw, copy=True),
        "h_new": np.array(h_new, copy=True),
        "e_raw": np.array(e_raw, copy=True),
        "e_new": np.array(e_new, copy=True),
        "r_raw": np.array(r_raw, copy=True),
        "r_new": np.array(r_new, copy=True),
        "p_h_new_minus_raw": np.array(h_new - h_raw, copy=True),
        "p_e_new_minus_raw": np.array(e_new - e_raw, copy=True),
        "p_r_new_minus_raw": np.array(r_new - r_raw, copy=True),
        "h_clip_norm": float(np.linalg.norm(h_raw - h_new)),
        "e_clip_norm": float(np.linalg.norm(e_raw - e_new)),
        "r_clip_norm": float(np.linalg.norm(r_raw - r_new)),
    }


def audit_one(sim: Any) -> Dict[str, Any]:
    """Audit a compressed simulator instance for all rounds."""
    if not sim.cfg.compression:
        raise ValueError("Audit requires compression=True")

    e_sum = np.zeros_like(sim.e[0])
    rho_sum = np.zeros_like(sim.r[0])
    beta_sum = np.zeros_like(sim.r[0])
    c_sum = np.zeros_like(sim.h[0])
    c_proxy_sum = np.zeros_like(sim.h[0])
    rho_proxy_sum = np.zeros_like(sim.h[0])
    beta_code_sum = np.zeros_like(sim.h[0])
    prev_e_mean = np.mean(sim.e, axis=0).copy()
    prev_r_mean = np.mean(sim.r, axis=0).copy()

    max_update_mismatch = 0.0
    max_c_tel = 0.0
    max_rho_tel = 0.0
    max_e_identity = 0.0
    max_e_code_identity = 0.0
    max_h_state_identity = 0.0
    max_ebar_old_gap = 0.0
    max_rbar_old_gap = 0.0
    max_wrong_sign_c_gap = 0.0
    max_wrong_sign_rho_gap = 0.0
    max_h_projection = 0.0
    max_e_projection = 0.0
    max_r_projection = 0.0
    max_history_c_norm_gap = 0.0
    max_history_rho_norm_gap = 0.0
    max_history_beta_norm_gap = 0.0
    max_history_E_norm_gap = 0.0
    max_history_E_sq_gap = 0.0
    rows: List[Dict[str, Any]] = []

    for round_idx in range(sim.cfg.rounds):
        local: List[Dict[str, Any]] = []
        original = sim._local_update

        def wrapped(i: int):
            expected = _expected_local(sim, i)
            old_h = sim.h[i].copy()
            old_e = sim.e[i].copy()
            old_r = sim.r[i].copy()
            ret = original(i)
            mismatch = max(
                float(np.max(np.abs(sim.h[i] - expected["h_new"]))),
                float(np.max(np.abs(sim.e[i] - expected["e_new"]))),
                float(np.max(np.abs(sim.r[i] - expected["r_new"]))),
            )
            nonlocal max_update_mismatch
            max_update_mismatch = max(max_update_mismatch, mismatch)
            expected["i"] = int(i)
            expected["old_h"] = old_h
            expected["old_e"] = old_e
            expected["old_r"] = old_r
            expected["sim_diagnostics"] = ret
            local.append(expected)
            return ret

        sim._local_update = wrapped
        sim.step()
        sim._local_update = original

        # The wrapper has one entry per participating client.  All current
        # configurations use q=1, so this is the exact released aggregate.
        if len(local) != sim.n:
            raise AssertionError(f"expected {sim.n} local records, got {len(local)}")

        c_t = np.mean([x["h_new"] - x["u"] for x in local], axis=0)
        rho_t = np.mean([x["u"] - x["v"] for x in local], axis=0)
        beta_true_t = np.mean([x["v"] - x["raw_mean"] for x in local], axis=0)
        beta_code_t = np.mean([x["raw_mean"] - x["v"] for x in local], axis=0)

        # The simulator's proxy: Top-K residual plus a scalar norm of all
        # projection residuals in coordinate 0.  This exists explicitly in
        # simplex and is reconstructed here for softmax as a comparison.
        c_proxy_t = np.mean([x["delta"] - x["q"] for x in local], axis=0)
        rho_scalar_t = float(
            np.mean([
                x["h_clip_norm"] + x["e_clip_norm"] + x["r_clip_norm"]
                for x in local
            ])
        )
        rho_proxy_t = np.zeros_like(c_t)
        rho_proxy_t[0] = rho_scalar_t

        # Cumulative theorem quantities.
        c_sum += c_t
        rho_sum += rho_t
        beta_sum += beta_true_t
        e_sum += c_t + rho_t + beta_true_t
        c_proxy_sum += c_proxy_t
        rho_proxy_sum += rho_proxy_t
        beta_code_sum += beta_code_t

        # Exact algebraic identity: c+rho+beta_true equals h_new-raw_mean.
        e_direct = np.sum(
            np.mean([x["h_new"] - x["raw_mean"] for x in local], axis=0),
            axis=0,
        ) if False else None
        direct_increment = np.mean(
            [x["h_new"] - x["raw_mean"] for x in local], axis=0
        )
        max_e_identity = max(
            max_e_identity,
            float(np.max(np.abs((c_t + rho_t + beta_true_t) - direct_increment))),
        )

        # The c telescoping identity with p_e = e_new - e_raw:
        # sum_s c_s = mean(e_t) - sum_s mean(p_e_s), when ebar=e_old.
        e_mean = np.mean(sim.e, axis=0)
        p_e_t = np.mean([x["p_e_new_minus_raw"] for x in local], axis=0)
        c_tel_rhs = e_mean - np.sum(
            [np.mean([r["p_e_new_minus_raw"] for r in local], axis=0)
             for _ in [0]], axis=0
        )
        # The expression above only includes the current p; use a running
        # sum to avoid ambiguity in numerical reporting.
        if round_idx == 0:
            p_e_sum = np.zeros_like(p_e_t)
            p_r_sum = np.zeros_like(p_e_t)
        p_e_sum += p_e_t
        p_r_t = np.mean([x["p_r_new_minus_raw"] for x in local], axis=0)
        p_r_sum += p_r_t
        c_tel_rhs = e_mean - p_e_sum
        max_c_tel = max(max_c_tel, float(np.max(np.abs(c_sum - c_tel_rhs))))

        r_mean = np.mean(sim.r, axis=0)
        # With p_r = r_new-r_raw and r_0=0:
        # sum_s rho_s = -mean(r_t) + sum_s mean(p_r_s).
        rho_tel_rhs = -r_mean + p_r_sum
        max_rho_tel = max(max_rho_tel, float(np.max(np.abs(rho_sum - rho_tel_rhs))))

        # The closure draft defines p_e/p_r as raw-new, whereas the skeleton
        # writes them as new-raw.  If one keeps the *same* telescoping signs
        # while switching conventions, the following formulas are wrong:
        #   c_sum = e_mean - sum(raw-new)
        #   rho_sum = -r_mean + sum(raw-new).
        # These diagnostics quantify the resulting sign error.
        p_e_raw_minus_new_sum = -p_e_sum
        p_r_raw_minus_new_sum = -p_r_sum
        c_wrong_sign = e_mean - p_e_raw_minus_new_sum
        rho_wrong_sign = -r_mean + p_r_raw_minus_new_sum
        max_wrong_sign_c_gap = max(
            max_wrong_sign_c_gap,
            float(np.max(np.abs(c_sum - c_wrong_sign))),
        )
        max_wrong_sign_rho_gap = max(
            max_wrong_sign_rho_gap,
            float(np.max(np.abs(rho_sum - rho_wrong_sign))),
        )

        # ebar/rbar are clipped versions of old states.  For this simulator
        # states are already bounded, so these gaps should be zero.
        ebar_mean = np.mean([clip_vec(x["old_e"], sim.cfg.Be) for x in local], axis=0)
        rbar_mean = np.mean([clip_vec(x["old_r"], sim.cfg.Br) for x in local], axis=0)
        max_ebar_old_gap = max(max_ebar_old_gap, float(np.max(np.abs(ebar_mean - prev_e_mean))))
        max_rbar_old_gap = max(max_rbar_old_gap, float(np.max(np.abs(rbar_mean - prev_r_mean))))
        prev_e_mean = e_mean.copy()
        prev_r_mean = r_mean.copy()

        max_h_projection = max(
            max_h_projection,
            max(float(np.linalg.norm(x["p_h_new_minus_raw"])) for x in local),
        )
        max_e_projection = max(
            max_e_projection,
            max(float(np.linalg.norm(x["p_e_new_minus_raw"])) for x in local),
        )
        max_r_projection = max(
            max_r_projection,
            max(float(np.linalg.norm(x["p_r_new_minus_raw"])) for x in local),
        )

        # Compare the simulator's published history diagnostics with the
        # independently reconstructed quantities.  Both simulators now use
        # c=h_clean-u_bar, rho=u_bar-v_bar, beta=v_bar-raw_bar, and exact
        # cumulative E in these fields.
        hist = sim.history[-1]
        max_history_c_norm_gap = max(
            max_history_c_norm_gap,
            abs(float(hist["c_t_norm"]) - float(np.linalg.norm(c_t))),
        )
        max_history_rho_norm_gap = max(
            max_history_rho_norm_gap,
            abs(float(hist["rho_t"]) - float(np.linalg.norm(rho_t))),
        )
        max_history_beta_norm_gap = max(
            max_history_beta_norm_gap,
            abs(float(hist["beta_t_norm"]) - float(np.linalg.norm(beta_true_t))),
        )
        max_history_E_norm_gap = max(
            max_history_E_norm_gap,
            abs(float(hist["E_t_norm"]) - float(np.linalg.norm(e_sum))),
        )
        max_history_E_sq_gap = max(
            max_history_E_sq_gap,
            abs(float(hist["E_t_sq"]) - float(np.dot(e_sum, e_sum))),
        )

        # h recursion identity, useful as a secondary state sanity check.
        h_mean = np.mean(sim.h, axis=0)
        q_t = np.mean([x["q"] for x in local], axis=0)
        p_h_t = np.mean([x["p_h_new_minus_raw"] for x in local], axis=0)
        if round_idx == 0:
            q_sum = np.zeros_like(q_t)
            p_h_sum = np.zeros_like(q_t)
        q_sum += q_t
        p_h_sum += p_h_t
        max_h_state_identity = max(
            max_h_state_identity,
            float(np.max(np.abs(h_mean - q_sum - p_h_sum))),
        )

        # The simplex implementation exposes its proxy E directly.
        actual_proxy = getattr(sim, "_E", None)
        proxy_gap = None
        if actual_proxy is not None:
            expected_proxy = c_proxy_sum + rho_proxy_sum + beta_code_sum
            proxy_gap = float(np.max(np.abs(actual_proxy - expected_proxy)))
            max_e_code_identity = max(max_e_code_identity, proxy_gap)

        rows.append({
            "round": int(round_idx + 1),
            "c_norm": float(np.linalg.norm(c_t)),
            "rho_norm": float(np.linalg.norm(rho_t)),
            "beta_true_norm": float(np.linalg.norm(beta_true_t)),
            "beta_code_norm": float(np.linalg.norm(beta_code_t)),
            "E_true_norm": float(np.linalg.norm(e_sum)),
            "E_true_sq": float(np.dot(e_sum, e_sum)),
            "c_cumulative_norm": float(np.linalg.norm(c_sum)),
            "rho_cumulative_norm": float(np.linalg.norm(rho_sum)),
            "beta_cumulative_norm": float(np.linalg.norm(beta_sum)),
            "c_tel_error_inf": float(np.max(np.abs(c_sum - c_tel_rhs))),
            "rho_tel_error_inf": float(np.max(np.abs(rho_sum - rho_tel_rhs))),
            "algebra_error_inf": float(np.max(np.abs((c_t + rho_t + beta_true_t) - direct_increment))),
            "proxy_E_gap_inf": proxy_gap,
            "h_state_error_inf": float(np.max(np.abs(h_mean - q_sum - p_h_sum))),
        })

    final_proxy = getattr(sim, "_E", None)
    result = {
        "class": sim.__class__.__name__,
        "seed": int(sim.cfg.seed),
        "rounds": int(sim.cfg.rounds),
        "n_clients": int(sim.n),
        "dimension": int(sim.D if hasattr(sim, "D") else sim.d),
        "config": {
            "C0": float(sim.cfg.C0), "Cg": float(sim.cfg.Cg),
            "Br": float(sim.cfg.Br), "Bh": float(sim.cfg.Bh),
            "Be": float(sim.cfg.Be), "eta": float(sim.eta),
            "topk_frac": float(sim.cfg.topk_frac),
            "private": bool(sim.cfg.private),
        },
        "max_update_mismatch_inf": max_update_mismatch,
        "max_algebra_identity_error_inf": max_e_identity,
        "max_c_telescoping_error_inf": max_c_tel,
        "max_rho_telescoping_error_inf": max_rho_tel,
        "max_h_state_identity_error_inf": max_h_state_identity,
        "max_ebar_minus_old_e_inf": max_ebar_old_gap,
        "max_rbar_minus_old_r_inf": max_rbar_old_gap,
        "max_wrong_sign_c_gap_inf": max_wrong_sign_c_gap,
        "max_wrong_sign_rho_gap_inf": max_wrong_sign_rho_gap,
        "max_h_projection_residual_norm": max_h_projection,
        "max_e_projection_residual_norm": max_e_projection,
        "max_r_projection_residual_norm": max_r_projection,
        "max_history_c_norm_gap": max_history_c_norm_gap,
        "max_history_rho_norm_gap": max_history_rho_norm_gap,
        "max_history_beta_norm_gap": max_history_beta_norm_gap,
        "max_history_E_norm_gap": max_history_E_norm_gap,
        "max_history_E_sq_gap": max_history_E_sq_gap,
        # This is a legacy diagnostic (Top-K residual + scalar projection
        # norms) retained only to expose what an older proxy would measure;
        # it is not the current simulator's E definition.
        "max_legacy_proxy_reconstruction_error_inf": max_e_code_identity,
        "final_E_true_norm": float(np.linalg.norm(e_sum)),
        "final_E_true_sq": float(np.dot(e_sum, e_sum)),
        "final_c_cumulative_norm": float(np.linalg.norm(c_sum)),
        "final_rho_cumulative_norm": float(np.linalg.norm(rho_sum)),
        "final_beta_true_cumulative_norm": float(np.linalg.norm(beta_sum)),
        "final_beta_code_cumulative_norm": float(np.linalg.norm(beta_code_sum)),
        "final_code_E_norm": None if final_proxy is None else float(np.linalg.norm(final_proxy)),
        "final_code_E_vs_true_gap_inf": None if final_proxy is None else float(np.max(np.abs(final_proxy - e_sum))),
        "trajectory": rows,
    }
    return result


def make_softmax(seed: int, private: bool = True):
    clients, test, _ = make_softmax_data(20, 40, 8, 3, seed=seed)
    cfg = SoftmaxConfig(
        n_clients=20, samples_per_client=40, d=8, n_classes=3,
        rounds=80, topk_frac=0.1, C0=1.5, Cg=1.5, Br=2.0,
        Bh=1.0, Be=2.0, epsilon=8.0, delta_dp=1e-5, seed=seed,
        private=private, compression=True,
        accountant="without_replacement_bound",
    )
    return SoftmaxEControlDA(clients, cfg)


def make_simplex(seed: int, private: bool = True):
    clients, test, w_true = make_simplex_data(20, 40, 8, seed=seed)
    cfg = SimplexConfig(
        n_clients=20, samples_per_client=40, d=8, rounds=100,
        topk_frac=0.2, C0=1.5, Cg=1.5, Br=2.0, Bh=1.0, Be=2.0,
        gamma=5.0, epsilon=8.0, delta_dp=1e-5, seed=seed,
        private=private, compression=True,
        accountant="without_replacement_bound",
    )
    return SimplexLogisticEControlDA(clients, cfg)


def main(output: Path = DEFAULT_OUTPUT):
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for kind, maker in [("softmax", make_softmax), ("simplex", make_simplex)]:
        for private in (False, True):
            key = f"{kind}_{'centralDP' if private else 'nonprivate'}"
            all_results[key] = []
            for seed in (0, 1, 2):
                sim = maker(seed, private=private)
                all_results[key].append(audit_one(sim))
                print(
                    key, seed,
                    "alg=%.3g ctel=%.3g rtel=%.3g Eident=%.3g proxygap=%s" % (
                        all_results[key][-1]["max_update_mismatch_inf"],
                        all_results[key][-1]["max_c_telescoping_error_inf"],
                        all_results[key][-1]["max_rho_telescoping_error_inf"],
                        all_results[key][-1]["max_algebra_identity_error_inf"],
                        str(all_results[key][-1]["final_code_E_vs_true_gap_inf"]),
                    ),
                )
    out = {
        "metadata": {
            "purpose": "independent c/rho/beta/E telescoping audit",
            "source_softmax": "p3_softmax_sim.py",
            "source_simplex": "p3_simplex_logistic_sim.py",
            "seeds": [0, 1, 2],
            "note": "The checker instruments committed classes; it does not modify them.",
        },
        "results": all_results,
    }
    out_path = output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit theorem-aligned signed diagnostics for softmax and simplex."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: experiments/tele_scope_softmax_simplex_audit.json).",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=None,
        help="Optional simulator directory; equivalent to P3_CODE_DIR.",
    )
    args = parser.parse_args()
    if args.code_dir is not None:
        code_dir = args.code_dir.resolve()
        if not (code_dir / "p3_bounded_sim.py").is_file():
            parser.error(f"--code-dir does not contain p3_bounded_sim.py: {code_dir}")
        # The imports above happen before argument parsing, so a custom path
        # is intended for unusual layouts and must be supplied through the
        # environment in those cases.
        if code_dir != CODE_DIR:
            parser.error(
                "For a custom code directory set P3_CODE_DIR before invoking the script."
            )
    main(args.output)
