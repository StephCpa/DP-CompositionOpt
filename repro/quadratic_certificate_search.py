#!/usr/bin/env python3
"""Stability certificates for the unprojected EControl recursion of the simulators.

The simulators use ``eta=topk_frac`` and the update order

    delta_t = u_t - h_{t-1} - eta e_t,   q_t = TopK(delta_t),
    h_t = h_{t-1} + q_t,                 e_{t+1} = e_t + h_t - u_t,

which, while no projection is active, is the closure note's EC-rec:

    e_{t+1}     = (1-eta) e_t + q_t - delta_t
    delta_{t+1} = (1+eta)(delta_t - q_t) + eta^2 e_t + (u_{t+1} - u_t)
    h_t - u_t   = q_t - delta_t - eta e_t.

Three bounds are reported for kappa = k/d:

1. ``ec_matrix``: the 2x2 absolute-value recursion of the closure note.  It
   is stable iff ``eta < (1/sqrt(1-kappa) - 1)/2`` (closed form, checked
   numerically here) and then gives explicit steady-state gains.
2. ``quadratic``: a block-scalar quadratic Lyapunov function
   ``V = [e;delta]^T (P kron I_d) [e;delta]`` found by an S-procedure that uses
   two Top-K facts: ``<q, delta-q> = 0`` (disjoint supports) and
   ``||delta-q||^2 <= (1-kappa)||delta||^2``.  A certificate is a point where
   ``V' - rho^2 V - c||du||^2`` plus the multiplier terms is negative
   semidefinite; every reported certificate is re-verified by an eigenvalue
   check.  The search is a deterministic Nelder-Mead heuristic in pure numpy,
   so "not found" is NOT a proof of infeasibility or of instability.
3. ``quadratic_sector``: the same search with the h-ball projection residual
   ``p = h_raw - Pi(h_raw)`` added as a sector nonlinearity,
   ``<p, (h_raw - u) - p> >= 0``, which holds whenever ``||u_t|| <= B_h``.

Gains are steady-state bounds per unit input variation:

    limsup_t ||e_t||         <= e_gain * sup_s ||u_{s+1} - u_s||
    limsup_t ||h_t - u_t||   <= h_gain * sup_s ||u_{s+1} - u_s||

For the quadratic certificate the same constants also bound root-mean-square
quantities (take expectations in ``V' <= rho^2 V + c||du||^2``).  The code's
start ``h_{-1}=0`` gives ``delta_0 = u_0``, which adds a decaying transient
``rho^t sqrt(V_0)`` that is not included in the gains.

Run from the repository root::

    python repro/quadratic_certificate_search.py \
      --output experiments/quadratic_certificate_20261008.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EIG_TOL = 1e-9


# ---------------------------------------------------------------------------
# 1. Closed-form EC-matrix route
# ---------------------------------------------------------------------------

def eta_critical(kappa: float) -> float:
    """Largest eta with rho(M_delta) < 1 for the closure note's EC-matrix."""
    chi = math.sqrt(1.0 - kappa)
    return (1.0 / chi - 1.0) / 2.0


def ec_matrix(kappa: float, eta: float) -> Dict[str, Any]:
    """Spectral radius and steady-state comparison gains of EC-matrix.

    With ``M = [[1-eta, chi], [eta^2, (1+eta) chi]] >= 0`` entrywise and
    ``rho(M) < 1``, the comparison system gives
    ``sup||e|| <= chi/det(I-M) * sup||du||`` and
    ``sup||h-u|| <= chi*b + eta*a <= 2 chi/(1-chi-2 eta chi) * sup||du||``,
    where ``det(I-M) = eta (1 - chi - 2 eta chi)``.
    """
    chi = math.sqrt(1.0 - kappa)
    M = np.array([[abs(1.0 - eta), chi], [eta**2, (1.0 + eta) * chi]])
    rho = float(max(abs(np.linalg.eigvals(M))))
    out: Dict[str, Any] = {"eta": eta, "rho": rho, "stable": rho < 1.0}
    if rho < 1.0 and eta <= 1.0:
        det = eta * (1.0 - chi - 2.0 * eta * chi)
        out["e_gain"] = chi / det
        out["h_gain"] = 2.0 * chi / (1.0 - chi - 2.0 * eta * chi)
    else:
        out["e_gain"] = None
        out["h_gain"] = None
    return out


# ---------------------------------------------------------------------------
# 2. Quadratic S-procedure certificates
# ---------------------------------------------------------------------------

def _blocks(eta: float, kappa: float, sector: bool):
    """Dynamics and constraint matrices on variables (e, delta, q, [p], w).

    All matrices act on scalar blocks; the full form is ``A kron I_d``, so a
    block-level certificate is valid in every dimension d.
    """
    if sector:
        # e' = (1-eta)e - delta + q - p ;  delta' = eta^2 e + (1+eta)(delta - q) + (1+eta)p + w
        B = np.array([[1 - eta, -1.0, 1.0, -1.0, 0.0],
                      [eta**2, 1 + eta, -(1 + eta), 1 + eta, 1.0]])
    else:
        B = np.array([[1 - eta, -1.0, 1.0, 0.0],
                      [eta**2, 1 + eta, -(1 + eta), 1.0]])
    n = B.shape[1]
    E = np.zeros((2, n)); E[0, 0] = E[1, 1] = 1.0
    Q1 = np.zeros((n, n)); Q1[1, 2] = Q1[2, 1] = 0.5; Q1[2, 2] = -1.0          # <q, delta - q> = 0
    Q2 = np.zeros((n, n)); Q2[1, 1] = -kappa; Q2[1, 2] = Q2[2, 1] = 1.0; Q2[2, 2] = -1.0  # (1-k)|d|^2 - |d-q|^2 >= 0
    Q3 = np.zeros((n, n))
    if sector:
        dvec = np.zeros(n); dvec[:3] = [-eta, -1.0, 1.0]                       # h_raw - u = q - delta - eta e
        pvec = np.zeros(n); pvec[3] = 1.0
        Q3 = 0.5 * (np.outer(pvec, dvec - pvec) + np.outer(dvec - pvec, pvec))  # <p, d - p> >= 0
    return B, E, Q1, Q2, Q3


def _form(z: np.ndarray, eta: float, kappa: float, rho: float, sector: bool):
    """Return (A, P, multipliers) for parameters z = (p12, p22, l1, l2[, l3])."""
    B, E, Q1, Q2, Q3 = _blocks(eta, kappa, sector)
    p12, p22, l1, l2 = z[:4]
    l3 = z[4] if sector else 0.0
    P = np.array([[1.0, p12], [p12, p22]])
    A = B.T @ P @ B - rho**2 * E.T @ P @ E + l1 * Q1 + l2 * Q2 + l3 * Q3
    return A, P, (l1, l2, l3)


def _verify(z: np.ndarray, eta: float, kappa: float, rho: float, sector: bool) -> Tuple[bool, float]:
    """Exact check of a candidate: P > 0, inequality multipliers >= 0, w-free block < 0."""
    A, P, (_, l2, l3) = _form(z, eta, kappa, rho, sector)
    m = A.shape[0] - 1
    lmax = float(np.linalg.eigvalsh(A[:m, :m]).max())
    ok = np.linalg.eigvalsh(P).min() > 0 and l2 >= 0 and l3 >= 0 and lmax < -EIG_TOL
    return bool(ok), lmax


def _nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, iters: int, step: float = 0.5):
    n = len(x0)
    S = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    F = [f(s) for s in S]
    for _ in range(iters):
        order = np.argsort(F)
        S = [S[i] for i in order]; F = [F[i] for i in order]
        c = np.mean(S[:-1], axis=0)
        xr = c + (c - S[-1]); fr = f(xr)
        if fr < F[0]:
            xe = c + 2.0 * (c - S[-1]); fe = f(xe)
            S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]:
            S[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (S[-1] - c); fc = f(xc)
            if fc < F[-1]:
                S[-1], F[-1] = xc, fc
            else:
                S = [S[0]] + [S[0] + 0.5 * (s - S[0]) for s in S[1:]]
                F = [F[0]] + [f(s) for s in S[1:]]
    i = int(np.argmin(F))
    return S[i], F[i]


def _find_certificate(eta: float, kappa: float, rho: float, sector: bool,
                      restarts: int, iters: int, seed: int) -> Optional[np.ndarray]:
    """Minimise the largest eigenvalue (with penalties) and return a verified point."""
    dim = 5 if sector else 4

    def objective(z: np.ndarray) -> float:
        A, P, (_, l2, l3) = _form(z, eta, kappa, rho, sector)
        m = A.shape[0] - 1
        pen = 1e3 * (max(0.0, 1e-6 - np.linalg.eigvalsh(P).min()) + max(0.0, -l2) + max(0.0, -l3))
        return float(np.linalg.eigvalsh(A[:m, :m]).max() / max(1.0, abs(P[1, 1])) + pen)

    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        x0 = np.abs(rng.normal(size=dim)) * np.array([1.0, 3.0, 3.0, 3.0, 3.0][:dim])
        z, _ = _nelder_mead(objective, x0, iters)
        if _verify(z, eta, kappa, rho, sector)[0]:
            return z
    return None


def min_certified_rho(eta: float, kappa: float, sector: bool, *, lo: float = 0.9, hi: float = 1.2,
                      steps: int = 12, restarts: int = 4, iters: int = 1200, seed: int = 0) -> Dict[str, Any]:
    """Bisection for the smallest rho at which the search finds a verified certificate."""
    best: Optional[Tuple[float, np.ndarray]] = None
    if _find_certificate(eta, kappa, hi, sector, restarts, iters, seed) is None:
        return {"found": False, "searched_interval": [lo, hi]}
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        z = _find_certificate(eta, kappa, mid, sector, restarts, iters, seed)
        if z is None:
            lo = mid
        else:
            hi, best = mid, (mid, z)
    if best is None:
        best = (hi, _find_certificate(eta, kappa, hi, sector, restarts, iters, seed))
    rho, z = best
    ok, lmax = _verify(z, eta, kappa, rho, sector)
    return {"found": True, "rho": rho, "rho_uncertified_below": lo, "verified": ok,
            "max_eigenvalue": lmax, "params": [float(v) for v in z]}


def certified_gains(eta: float, kappa: float, rhos: Sequence[float], *, restarts: int = 4,
                    iters: int = 2000, seed: int = 0) -> Optional[Dict[str, Any]]:
    """Smallest certified e-gain over a rho grid (no-projection model), plus h-gain.

    For fixed rho the least admissible ``c`` follows from the Schur complement
    of the w-block; ``mu_e = p11 - p12^2/p22`` lower-bounds ``V/||e||^2``.
    """
    rng = np.random.default_rng(seed)
    best = None
    for rho in rhos:
        def c_min(z: np.ndarray) -> float:
            A, P, (_, l2, _) = _form(z, eta, kappa, rho, False)
            if np.linalg.eigvalsh(P).min() <= 0 or l2 < 0:
                return math.inf
            A3, b, aww = A[:3, :3], A[:3, 3], A[3, 3]
            if np.linalg.eigvalsh(A3).max() >= -EIG_TOL:
                return math.inf
            return float(aww - b @ np.linalg.solve(A3, b))

        def objective(z: np.ndarray) -> float:
            c = c_min(z)
            if not math.isfinite(c):
                A, _, _ = _form(z, eta, kappa, rho, False)
                return 1e12 * (1.0 + max(0.0, np.linalg.eigvalsh(A[:3, :3]).max()))
            _, P, _ = _form(z, eta, kappa, rho, False)
            mu = P[0, 0] - P[0, 1] ** 2 / P[1, 1]
            return c / ((1.0 - rho**2) * mu)

        for _ in range(restarts):
            z, f = _nelder_mead(objective, rng.normal(size=4) * 2.0, iters)
            if f < 1e11 and (best is None or f < best[0]):
                best = (f, rho, z, c_min(z))
    if best is None:
        return None
    f, rho, z, c = best
    # A tiny margin on c turns the zero Schur complement into a strict one, so
    # the eigenvalue re-check is not decided by rounding when c is large.
    c = c * (1.0 + 1e-7) + 1e-12
    A, P, (_, l2, _) = _form(z, eta, kappa, rho, False)
    f = c / ((1.0 - rho**2) * (P[0, 0] - P[0, 1] ** 2 / P[1, 1]))
    A_full = A.copy(); A_full[3, 3] -= c
    lmax_full = float(np.linalg.eigvalsh(A_full).max())
    scale = max(1.0, float(np.abs(A_full).max()))

    # h-gain: smallest beta with ||q - delta - eta e||^2 <= beta V on the Top-K constraint set.
    _, _, Q1, Q2, _ = _blocks(eta, kappa, False)
    g = np.array([-eta, -1.0, 1.0]); G = np.outer(g, g)
    Vm = np.zeros((3, 3)); Vm[:2, :2] = P

    def beta_for(y: np.ndarray) -> float:
        lo_b, hi_b = 0.0, 1e8
        for _ in range(90):
            mid = 0.5 * (lo_b + hi_b)
            M = G - mid * Vm + y[0] * Q1[:3, :3] + y[1] ** 2 * Q2[:3, :3]
            lo_b, hi_b = (lo_b, mid) if np.linalg.eigvalsh(M).max() <= 0 else (mid, hi_b)
        return hi_b

    beta_best, y_best = math.inf, None
    for s in range(4):
        y, b = _nelder_mead(beta_for, np.random.default_rng(100 + s).normal(size=2), 300)
        if b < beta_best:
            beta_best, y_best = b, y
    M = G - beta_best * Vm + y_best[0] * Q1[:3, :3] + y_best[1] ** 2 * Q2[:3, :3]
    mu = P[0, 0] - P[0, 1] ** 2 / P[1, 1]
    return {
        "rho": float(rho),
        "e_gain": float(math.sqrt(f)),
        "h_gain": float(math.sqrt(beta_best * c / (1.0 - rho**2))),
        "c": float(c), "mu_e": float(mu), "beta": float(beta_best),
        "P": P.tolist(), "lambda_q_orth": float(z[2]), "lambda_contraction": float(l2),
        "verification": {
            "dissipation_max_eigenvalue": lmax_full,
            "dissipation_matrix_scale": scale,
            "h_bound_max_eigenvalue": float(np.linalg.eigvalsh(M).max()),
            "passed": bool(lmax_full <= 1e-9 * scale and np.linalg.eigvalsh(M).max() <= 1e-9),
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(kappa: float, etas: Sequence[float]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for eta in etas:
        row: Dict[str, Any] = {"eta": eta, "ec_matrix": ec_matrix(kappa, eta)}
        row["quadratic"] = min_certified_rho(eta, kappa, sector=False)
        row["quadratic_sector"] = min_certified_rho(eta, kappa, sector=True)
        q = row["quadratic"]
        if q.get("found") and q["rho"] < 1.0:
            r0 = q["rho"]
            grid = [r0 + (1.0 - r0) * f for f in (0.1, 0.3, 0.5, 0.7)]
            row["quadratic_gains"] = certified_gains(eta, kappa, grid)
        else:
            row["quadratic_gains"] = None
        rows.append(row)
        print(f"eta={eta:<7} EC-matrix rho={row['ec_matrix']['rho']:.4f} | "
              f"quadratic rho={q.get('rho', float('nan')):.4f} | "
              f"sector rho={row['quadratic_sector'].get('rho', float('nan')):.4f} | "
              f"gains={row['quadratic_gains'] and (round(row['quadratic_gains']['e_gain'], 1), round(row['quadratic_gains']['h_gain'], 1))}",
              flush=True)
    chi = math.sqrt(1.0 - kappa)
    crit = eta_critical(kappa)
    return {
        "metadata": {
            "script": "repro/quadratic_certificate_search.py",
            "kappa_topk_fraction": kappa,
            "eta_critical_closed_form": crit,
            "eta_critical_check": {
                "rho_at_crit_minus_1e-4": ec_matrix(kappa, crit - 1e-4)["rho"],
                "rho_at_crit_plus_1e-4": ec_matrix(kappa, crit + 1e-4)["rho"],
            },
            "eta_paper3": kappa / (3.0 * chi * (1.0 + chi)),
            "recursion": "code convention, no projection: e'=(1-eta)e+q-delta; delta'=(1+eta)(delta-q)+eta^2 e+du",
            "gain_definition": "steady-state sup bound per unit sup||u_{t+1}-u_t||; quadratic gains also bound RMS values; transient from delta_0=u_0 excluded",
            "quadratic_constraints": ["<q, delta-q> = 0", "||delta-q||^2 <= (1-kappa)||delta||^2"],
            "sector_constraint": "<p, (h_raw-u) - p> >= 0 for the h-ball projection residual p; valid when ||u_t|| <= B_h",
            "search": "deterministic Nelder-Mead over block-scalar P and S-procedure multipliers; bisection on rho (12 steps in [0.9, 1.2])",
            "certificate_validity": "every reported certificate is re-verified: P > 0, multipliers >= 0, max eigenvalue < -1e-9 (rho search) or <= 1e-9 * matrix scale after a 1e-7 relative margin on c (gain certificate)",
            "not_found_meaning": "the heuristic search found no certificate; this is NOT a proof that none exists and NOT evidence of instability",
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kappa", type=float, default=0.1, help="Top-K fraction k/d")
    ap.add_argument("--etas", type=str, default="0.1,0.05,0.027,0.018,0.0135")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    etas = [float(x) for x in args.etas.split(",") if x.strip()]
    out = run(args.kappa, etas)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
