#!/usr/bin/env python3
"""Empirical EControl gains versus eta, and a Top-K-aware margin probe on box-LS.

Part ``recursion``
    Finite-trace gains of the unprojected EControl recursion in the code's
    convention (``delta_t = u_t - h_{t-1} - eta e_t``, ``h_t = h_{t-1} +
    TopK(delta_t)``, ``e_{t+1} = e_t + h_t - u_t``) under ramp, random-direction
    and greedy-adversarial input-variation sequences with ``||du_t|| = 1``.
    The recursion is positively homogeneous, so these ratios are scale free.
    They are empirical LOWER bounds on the worst-case gains, not certificates.

Part ``box_ls``
    The eta sweep on the real box-LS simulator with ``B_e = inf``, fixed total
    epsilon, noise-calibrated gamma (the driver's ``_calibrated_gamma`` with
    ``R_ref = Bbox*sqrt(d)``) and the averaged iterate as primary metric.  Each
    client's raw h/e state, tracking error ``h_raw - u``, input variation and
    Top-K support are recorded.  Support statistics include the longest run of
    consecutive rounds a coordinate stayed unselected on one client.

Part ``projection``
    Top-K EControl with and without the ``B_h`` projection while ``u_t`` is
    kept inside ``0.95 B_h`` (rotation and random-walk inputs).  This tests
    whether h-saturation by itself destabilises ``e`` when ``||u|| < B_h``.

Run from the repository root::

    python repro/eta_gain_probe.py --output experiments/eta_gain_scan_20261008.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.run_reproducibility import _calibrated_gamma  # noqa: E402


def _load_box_module(code_dir: str | None):
    candidates = [Path(code_dir).expanduser().resolve()] if code_dir else []
    candidates += [ROOT / "code", ROOT, Path.cwd() / "code", Path.cwd()]
    for path in candidates:
        if (path / "p3_box_ls_sim.py").exists():
            sys.path.insert(0, str(path))
            import p3_box_ls_sim as box  # type: ignore

            return box
    raise FileNotFoundError("could not find code/p3_box_ls_sim.py; pass --code-dir")


def _topk(v: np.ndarray, k: int) -> np.ndarray:
    # Same tie handling as code/p3_bounded_sim.topk.
    out = np.zeros_like(v)
    idx = np.argsort(-np.abs(v), kind="stable")[:k]
    out[idx] = v[idx]
    return out


# ---------------------------------------------------------------------------
# Part A: recursion gains
# ---------------------------------------------------------------------------

def _run_du(du_seq: np.ndarray, eta: float, k: int) -> Dict[str, float]:
    d = du_seq.shape[1]
    e = np.zeros(d); dl = np.zeros(d)
    me = mh = 0.0
    for du in du_seq:
        q = _topk(dl, k)
        mh = max(mh, float(np.linalg.norm(q - dl - eta * e)))
        e, dl = (1 - eta) * e + q - dl, (1 + eta) * (dl - q) + eta**2 * e + du
        me = max(me, float(np.linalg.norm(e)))
    return {"e_gain": me, "h_gain": mh}


def _greedy(eta: float, k: int, d: int, T: int, seed: int, target: str) -> Dict[str, float]:
    """One-step-lookahead adversary over coordinate, state-aligned and random unit directions."""
    rng = np.random.default_rng(seed)
    e = np.zeros(d); dl = np.zeros(d); me = mh = 0.0
    eye = np.eye(d)
    for _ in range(T):
        q = _topk(dl, k)
        mh = max(mh, float(np.linalg.norm(q - dl - eta * e)))
        e_new = (1 - eta) * e + q - dl
        base = (1 + eta) * (dl - q) + eta**2 * e
        cands = [s * eye[j] for j in range(d) for s in (1.0, -1.0)]
        for v in (e_new, base, dl - q):
            n = np.linalg.norm(v)
            if n > 0:
                cands += [v / n, -v / n]
        cands += [x / np.linalg.norm(x) for x in rng.normal(size=(8, d))]

        def score(du: np.ndarray) -> float:
            dn = base + du; qn = _topk(dn, k)
            if target == "e":
                return float(np.linalg.norm((1 - eta) * e_new + qn - dn))
            return float(np.linalg.norm(qn - dn - eta * e_new))

        du = max(cands, key=score)
        e, dl = e_new, base + du
        me = max(me, float(np.linalg.norm(e)))
    return {"e_gain": me, "h_gain": mh}


def part_recursion(etas: Sequence[float], d: int = 10, k: int = 1) -> List[Dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(0)
    T = 3000
    for eta in etas:
        v = rng.normal(size=d); v /= np.linalg.norm(v)
        R = rng.normal(size=(T, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
        rows.append({
            "eta": eta,
            "ramp": _run_du(np.tile(v, (T, 1)), eta, k),
            "random_direction": _run_du(R, eta, k),
            "greedy_e": _greedy(eta, k, d, 600, 1, "e"),
            "greedy_h": _greedy(eta, k, d, 600, 2, "h"),
        })
        r = rows[-1]
        print(f"[recursion] eta={eta:<7} ramp e={r['ramp']['e_gain']:.1f} h={r['ramp']['h_gain']:.1f} | "
              f"greedy e={r['greedy_e']['e_gain']:.1f} h={r['greedy_h']['h_gain']:.1f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Part B: box-LS eta sweep with Top-K support logging
# ---------------------------------------------------------------------------

def _make_recorder(box):
    class TopKRecorder(box.BoxLSEControlDA):
        """Records per-client raw states and Top-K supports; the simulator does every update."""

        def __init__(self, clients, config):
            super().__init__(clients, config)
            self.prev_u = np.full((self.n, self.d), np.nan)
            self.last_sel = np.full((self.n, self.d), -1, dtype=int)
            self.longest_unselected = 0
            self.switches = 0
            self.prev_support = [None] * self.n
            self.stat = dict(max_u=0.0, max_h_raw=0.0, max_e_raw=0.0, max_du=0.0, max_track=0.0,
                             h_active=0, client_rounds=0, consistency=0.0)

        def _local_update(self, i: int):
            cfg = self.cfg
            v, _, _ = self._clipped_gradient(i)
            rbar = box.clip_vec(self.r[i], cfg.Br) if cfg.residual_buffer else np.zeros(self.d)
            u = box.clip_vec(v + rbar, cfg.Cg)
            hbar = box.clip_vec(self.h[i], cfg.Bh)
            ebar = box.clip_vec(self.e[i], cfg.Be)
            delta = box.clip_vec(u - hbar - self.eta * ebar, cfg.Cg + cfg.Bh + self.eta * cfg.Be)
            support = tuple(sorted(np.argsort(-np.abs(delta), kind="stable")[: self.k].tolist()))
            h_raw = self.h[i] + box.topk(delta, self.k)
            h_new = box.clip_vec(h_raw, cfg.Bh)
            e_raw = ebar + h_new - u
            out = super()._local_update(i)
            s = self.stat
            s["consistency"] = max(s["consistency"], float(np.abs(self.h[i] - h_new).max()),
                                   float(np.abs(self.e[i] - box.clip_vec(e_raw, cfg.Be)).max()))
            s["max_u"] = max(s["max_u"], float(np.linalg.norm(u)))
            s["max_h_raw"] = max(s["max_h_raw"], float(np.linalg.norm(h_raw)))
            s["max_e_raw"] = max(s["max_e_raw"], float(np.linalg.norm(e_raw)))
            s["max_track"] = max(s["max_track"], float(np.linalg.norm(h_raw - u)))
            if not np.isnan(self.prev_u[i, 0]):
                s["max_du"] = max(s["max_du"], float(np.linalg.norm(u - self.prev_u[i])))
            s["h_active"] += int(np.linalg.norm(h_raw) > cfg.Bh)
            s["client_rounds"] += 1
            self.prev_u[i] = u
            for j in support:
                self.longest_unselected = max(self.longest_unselected, int(self.t - self.last_sel[i, j] - 1))
                self.last_sel[i, j] = self.t
            if self.prev_support[i] is not None and self.prev_support[i] != support:
                self.switches += 1
            self.prev_support[i] = support
            return out

        def finish(self) -> Dict[str, Any]:
            trailing = int((self.t - 1 - self.last_sel).max())
            s = dict(self.stat)
            s["h_active_fraction"] = s.pop("h_active") / max(s.pop("client_rounds"), 1)
            s["longest_unselected_run"] = max(self.longest_unselected, trailing)
            s["never_selected_fraction"] = float(np.mean(self.last_sel < 0))
            s["support_switch_fraction"] = self.switches / max(self.n * (self.t - 1), 1)
            return s

    return TopKRecorder


def part_box_ls(box, etas: Sequence[float], horizons: Sequence[int], seeds: Sequence[int],
                epsilon: float) -> List[Dict[str, Any]]:
    Recorder = _make_recorder(box)
    rows = []
    for T in horizons:
        for eta in etas:
            per_seed = []
            for seed in seeds:
                clients, test, w_true = box.make_box_ls_data(20, 40, 10, seed=seed)
                cfg = box.Config(rounds=T, seed=seed, epsilon=epsilon, eta=eta, Be=float("inf"))
                sim = Recorder(clients, cfg)
                sim.cfg.gamma = _calibrated_gamma(sigma=sim.sigma, dimension=sim.d, rounds=T,
                                                  reference_radius=cfg.Bbox * math.sqrt(sim.d))
                res = sim.run(test, w_true)
                E = np.zeros(sim.d); q_clip = 0.0
                for h in res["history"]:
                    E += np.asarray(h["c_t_vec"]) + np.asarray(h["rho_t_vec"])
                    q_clip += float(E @ E)
                stat = sim.finish()
                per_seed.append({
                    "seed": seed, "gamma": sim.cfg.gamma, "sigma": sim.sigma, "sensitivity": sim.sensitivity,
                    "average_test_mse": res["average_test_mse"], "last_test_mse": res["test_mse"],
                    "Q_T_clip_over_T": q_clip / T,
                    "E_clip_minus_mean_e_inf": float(np.abs(E - sim.e.mean(axis=0)).max()),
                    **stat,
                })
            agg: Dict[str, Any] = {"T": T, "eta": eta, "per_seed": per_seed}
            for key in ("average_test_mse", "last_test_mse", "Q_T_clip_over_T"):
                vals = np.array([r[key] for r in per_seed])
                agg[key] = float(vals.mean()); agg[key + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            for key in ("max_u", "max_h_raw", "max_e_raw", "max_du", "max_track", "h_active_fraction",
                        "longest_unselected_run", "consistency", "E_clip_minus_mean_e_inf"):
                agg[key] = float(max(r[key] for r in per_seed))
            agg["gamma"] = per_seed[0]["gamma"]
            agg["track_over_du"] = agg["max_track"] / agg["max_du"] if agg["max_du"] > 0 else None
            rows.append(agg)
            print(f"[box_ls] T={T} eta={eta:<7} gamma={agg['gamma']:.2f} avg MSE={agg['average_test_mse']:.3f}"
                  f"±{agg['average_test_mse_sd']:.3f} max||e||={agg['max_e_raw']:.2f} Q/T clip={agg['Q_T_clip_over_T']:.3f}"
                  f" max||h_raw||={agg['max_h_raw']:.3f} track/du={agg['track_over_du']:.2f}"
                  f" longest unselected={agg['longest_unselected_run']:.0f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Part C: h-projection with u inside the ball
# ---------------------------------------------------------------------------

def _projected_run(u_seq: np.ndarray, eta: float, k: int, Bh: float, project: bool) -> Dict[str, float]:
    d = u_seq.shape[1]
    h = np.zeros(d); e = np.zeros(d); me = 0.0; active = 0
    for u in u_seq:
        h_raw = h + _topk(u - h - eta * e, k)
        n = np.linalg.norm(h_raw)
        active += int(n > Bh)
        h_new = h_raw * (Bh / n) if (project and n > Bh) else h_raw
        e = e + h_new - u
        h = h_new
        me = max(me, float(np.linalg.norm(e)))
    return {"sup_e": me, "h_projection_active_fraction": active / len(u_seq)}


def part_projection(etas: Sequence[float], d: int = 10, k: int = 1, T: int = 20000,
                    Bh: float = 1.0, radius: float = 0.95) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows = []
    for step in (0.02, 0.1, 0.3):
        a, b = rng.normal(size=d), rng.normal(size=d)
        a /= np.linalg.norm(a); b -= (b @ a) * a; b /= np.linalg.norm(b)
        x = a.copy(); rot, rw = [], []
        for t in range(T):
            ang = step * t / radius
            rot.append(radius * (math.cos(ang) * a + math.sin(ang) * b))
            x = x + step * rng.normal(size=d) / math.sqrt(d); x /= np.linalg.norm(x)
            rw.append(radius * x)
        for name, U in (("rotation", np.array(rot)), ("random_walk", np.array(rw))):
            for eta in etas:
                rows.append({"input": name, "step": step, "eta": eta,
                             "projected": _projected_run(U, eta, k, Bh, True),
                             "unprojected": _projected_run(U, eta, k, Bh, False)})
                r = rows[-1]
                print(f"[projection] {name:11s} step={step:<5} eta={eta:<6} sup||e|| proj={r['projected']['sup_e']:.2f}"
                      f" unproj={r['unprojected']['sup_e']:.2f} active={r['projected']['h_projection_active_fraction']:.2%}",
                      flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code-dir", default=None)
    ap.add_argument("--etas", default="0.1,0.05,0.027,0.018,0.0135")
    ap.add_argument("--horizons", default="100,200")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    box = _load_box_module(args.code_dir)
    etas = [float(x) for x in args.etas.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    out = {
        "metadata": {
            "script": "repro/eta_gain_probe.py",
            "status": "empirical diagnostic; finite traces; not a certificate or theorem",
            "recursion_gain_definition": "sup_t ||e_t|| and sup_t ||h_t - u_t|| over a trace with ||u_{t+1}-u_t|| = 1; empirical lower bounds on worst-case gains",
            "recursion_setting": {"d": 10, "k": 1, "kappa": 0.1, "ramp_and_random_T": 3000, "greedy_T": 600},
            "box_ls_setting": {
                "n_clients": 20, "samples_per_client": 40, "d": 10, "topk_frac": 0.1, "C0": 1.5, "Cg": 1.5,
                "Bh": 1.0, "Be": "inf", "Br": 2.0, "Bbox": 2.0, "epsilon": args.epsilon, "delta_dp": 1e-5,
                "gamma_rule": "repro.run_reproducibility._calibrated_gamma(noise_coefficient=2, minimum_gamma=5, reference_radius=Bbox*sqrt(d))",
                "iterate_reporting": {"primary": "uniform_average", "additional": "last"},
                "seeds": list(range(args.seeds)),
            },
            "box_ls_fields": {
                "max_fields": "max_* and longest_unselected_run are maxima over clients, rounds AND seeds (not seed means)",
                "max_track": "max of ||h_raw - u|| (h_raw before projection)",
                "track_over_du": "max_track / max ||u_t - u_{t-1}|| (t >= 1): effective tracking gain on real inputs",
                "longest_unselected_run": "longest run of consecutive rounds one coordinate of one client was not in the Top-K support",
                "consistency": "max deviation between the recorder's reconstruction and the simulator's own h/e update (should be 0)",
                "E_clip_minus_mean_e_inf": "identity check: with B_e=inf and Cg>=C0, sum(c+rho) equals mean(e_T)",
            },
            "projection_setting": {"d": 10, "k": 1, "T": 20000, "Bh": 1.0, "u_radius": 0.95, "Be": "inf"},
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "recursion": part_recursion(etas),
        "box_ls": part_box_ls(box, etas, horizons, list(range(args.seeds)), args.epsilon),
        "projection": part_projection([0.1, 0.018]),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
