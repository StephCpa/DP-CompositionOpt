#!/usr/bin/env python3
"""Projected EControl: exact-prefix windup certificate and sufficient closure bound.

The integer-interval prefix is a computer-assisted proof for rational data,
not a floating-point stability claim. Full proofs and scope are in
``docs/p3_projected_econtrol_closure_zh.md``. NumPy is used only for empirical
traces and local-update cross-checks. The interval verifier uses Python ints.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

BASE_COMMIT = "a52d13c51fbb5a422829e6d62f44aed30517e017"


class Interval:
    """Closed dyadic interval [lo,hi]/SCALE with outward integer rounding."""
    BITS = 4096
    SCALE = 1 << BITS

    def __init__(self, lo: int, hi: int | None = None):
        self.lo, self.hi = lo, lo if hi is None else hi
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    @classmethod
    def rational(cls, numerator: int, denominator: int = 1):
        if denominator <= 0:
            raise ValueError("denominator must be positive")
        n = numerator * cls.SCALE
        return cls(n // denominator, -((-n) // denominator))

    def __add__(self, x):
        return Interval(self.lo + x.lo, self.hi + x.hi)

    def __sub__(self, x):
        return Interval(self.lo - x.hi, self.hi - x.lo)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __mul__(self, x):
        p = [a * b for a in (self.lo, self.hi) for b in (x.lo, x.hi)]
        return Interval(min(p) // self.SCALE, -((-max(p)) // self.SCALE))

    def __truediv__(self, x):
        if x.lo <= 0:
            raise ArithmeticError("division requires a strictly positive interval")
        pairs = [(a * self.SCALE, b) for a in (self.lo, self.hi) for b in (x.lo, x.hi)]
        return Interval(min(a // b for a, b in pairs), max(-((-a) // b) for a, b in pairs))

    def square(self):
        low = 0 if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2)
        high = max(self.lo**2, self.hi**2)
        return Interval(low // self.SCALE, -((-high) // self.SCALE))

    def sqrt(self):
        if self.lo < 0:
            raise ArithmeticError("negative square-root interval")
        low = math.isqrt(self.lo * self.SCALE)
        high = math.isqrt(self.hi * self.SCALE)
        high += int(high * high < self.hi * self.SCALE)
        return Interval(low, high)

    def absolute(self):
        if self.lo >= 0:
            return self
        if self.hi <= 0:
            return -self
        return Interval(0, max(-self.lo, self.hi))


def interval_sum(xs):
    value = Interval.rational(0)
    for x in xs:
        value = value + x
    return value


def certify_prefix(bits: int = 4096) -> dict:
    """Enclose all 1000 exact-real updates; certify S_1000 >= 600.

    Top-1 ties are accepted only for coordinates never selected before: those
    coordinates have exactly h_j=0,e_j=-t*31/100, by symmetry. Every other
    ordering and every projection branch must be separated by interval bounds.
    An unresolved comparison raises, rather than returning a certificate.
    """
    if bits < 128:
        raise ValueError("use at least 128 bits; 4096 is the verified default")
    Interval.BITS, Interval.SCALE = bits, 1 << bits
    z, u, eta = Interval.rational(0), Interval.rational(31, 100), Interval.rational(1, 10)
    h, e, selected = [z] * 10, [z] * 10, [False] * 10
    branch_log = bytearray()
    ties = projections = 0
    for t in range(1000):
        delta = [u - hj - eta * ej for hj, ej in zip(h, e)]
        mags = [x.absolute() for x in delta]
        j = max(range(10), key=lambda j: (mags[j].lo + mags[j].hi, -j))
        for l in range(10):
            if l == j or mags[j].lo >= mags[l].hi:
                continue
            symmetric = (not selected[j] and not selected[l] and j < l
                         and (delta[j].lo, delta[j].hi) == (delta[l].lo, delta[l].hi))
            if not symmetric:
                raise ArithmeticError(f"unresolved Top-1 comparison at round {t+1}, {j}/{l}")
            ties += 1
        raw = list(h)
        raw[j] = raw[j] + delta[j]
        n2 = interval_sum([x.square() for x in raw])
        if n2.hi <= Interval.SCALE:
            hn, active = raw, 0
        elif n2.lo > Interval.SCALE:
            norm = n2.sqrt()
            hn, active = [x / norm for x in raw], 1
        else:
            raise ArithmeticError(f"unresolved ball-projection branch at round {t+1}")
        e = [ej + hj - u for ej, hj in zip(e, hn)]
        h = hn
        selected[j] = True
        projections += active
        branch_log.extend((j, active))
    s = -interval_sum(e)
    if s.lo < 600 * Interval.SCALE:
        raise ArithmeticError("prefix did not certify entry into the invariant region")
    # Coarsen OUTWARDS to rational decimal endpoints, without floating point.
    den = 10**12
    lower = s.lo * den // Interval.SCALE
    upper = -((-s.hi * den) // Interval.SCALE)
    width_num = s.hi - s.lo
    return {
        "status": "exact-integer outward-interval prefix plus analytic invariant-region proof",
        "parameters": {"d": 10, "k": 1, "Bh": 1, "eta": "1/10", "u_each": "31/100",
                       "u_norm_squared": "961/1000", "Be": "inf", "h_initial": 0, "e_initial": 0},
        "rounds": 1000, "bits": bits,
        "S_lower_numerator": lower, "S_upper_numerator": upper, "S_denominator": den,
        "internal_interval_width_less_than_2_power": width_num.bit_length() - bits,
        "symbolic_tie_comparisons": ties, "projection_rounds": projections,
        "branch_sha256": hashlib.sha256(branch_log).hexdigest(),
        "entry_threshold": 600, "S_increment_lower_bound_after_entry": "27/20",
        "prefix_entry_certified": True,
        "consequence": "||e_T|| >= [S_1000 + (27/20)(T-1000)]/sqrt(10), T>=1000; Q_T=Theta(T^3)",
        "scope": "ideal-real projected local recursion; not a theorem about measured LS/softmax test utility",
    }


def clip_vec(v, radius):
    n = np.linalg.norm(v)
    if n <= radius or n == 0.0:
        return v.copy()
    return v * (radius / n)


def topk(v, k):
    if k >= v.size:
        return v.copy()
    out = np.zeros_like(v)
    idx = np.argsort(-np.abs(v), kind="stable")[:k]
    out[idx] = v[idx]
    return out


def step(h, e, u, eta, k, Bh=1.0):
    """Matches the repository's Be=inf, r=0 local update, including re-clipping."""
    delta = u - clip_vec(h, Bh) - eta * e
    raw = h + topk(delta, k)
    hn = clip_vec(raw, Bh)
    return hn, e + hn - u, raw


def float_trace(component: float, eta: float, k: int = 1, T: int = 30000, projected: bool = True):
    h, e, u = np.zeros(10), np.zeros(10), np.full(10, component)
    Q = max_e = 0.0
    active = 0
    snapshots = []
    checkpoints = {100, 1000, 10000, T}
    for t in range(1, T + 1):
        h, e, raw = step(h, e, u, eta, k, 1.0 if projected else math.inf)
        enorm = float(np.linalg.norm(e))
        Q += enorm**2
        max_e = max(max_e, enorm)
        active += int(np.linalg.norm(raw) > 1.0)
        if t in checkpoints:
            snapshots.append({"T": t, "e_norm": enorm, "S": float(-e.sum()), "Q_T_over_T": Q/t})
    return {"component": component, "u_norm": float(np.linalg.norm(u)), "eta": eta, "k": k,
            "projected": projected, "T": T, "max_e_norm": max_e, "max_du": 0.0,
            "h_projection_active_fraction" if projected else "would_project_fraction": active/T,
            "snapshots": snapshots}


def sufficient_bound(U: float, Bh: float, kappa: float, eta: float) -> dict:
    """Evaluate the conservative bound proved in the companion note."""
    if not (0 <= U and Bh > 0 and 0 < kappa <= 1 and eta > 0):
        raise ValueError("invalid state-bound parameters")
    a = math.sqrt(kappa)
    mu = Bh*a - U
    if mu <= 0:
        return {"condition_satisfied": False, "reason": "requires U < Bh*sqrt(k/d)"}
    D, W = U + Bh, math.sqrt(U*U + Bh*Bh)
    b = (1+a)*D/eta
    K = Bh*b + 4*Bh*W/(eta*a)
    C = 2*K + D*D
    R0 = max(2*b/a, 2*(Bh+W)/(eta*a))
    R = max(R0, C/(2*mu))
    return {"condition_satisfied": True, "U": U, "Bh": Bh, "kappa": kappa, "eta": eta,
            "mu": mu, "D": D, "W": W, "b": b, "K": K, "C": C, "R0": R0,
            "R": R, "zero_initialization_e_bound": R+D,
            "scope": "analytic sufficient condition, not a sharp stability boundary or optimized constant"}


def check_source_methods(root: Path) -> list:
    """Execute only audited _local_update bodies, NOT full simulator training."""
    specs = [("p3_box_ls_sim.py", "BoxLSEControlDA"),
             ("p3_simplex_logistic_sim.py", "SimplexLogisticEControlDA"),
             ("p3_softmax_sim.py", "SoftmaxEControlDA")]
    rows = []
    for filename, classname in specs:
        path = root / "code" / filename
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == classname)
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_local_update")
        digest = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
        ns = {"np": np, "clip_vec": clip_vec, "topk": topk}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        cfg = SimpleNamespace(C0=1., Cg=1., Bh=1., Be=math.inf, Br=2., residual_buffer=True)
        sim = SimpleNamespace(cfg=cfg, d=10, D=10, eta=.1, k=1, t=0,
                              h=np.zeros((1,10)), e=np.zeros((1,10)), r=np.zeros((1,10)),
                              last_update=np.zeros(1, dtype=int))
        for name in ("prefix_clip_residual", "prefix_e_clip_residual", "prefix_r_clip_residual",
                     "prefix_h_clip", "prefix_e_clip", "prefix_r_clip"):
            setattr(sim, name, np.zeros(1))
        u = np.full(10, .31)
        sim._clipped_gradient = lambda i: (u.copy(), u.copy(), 0.0)
        update = MethodType(ns["_local_update"], sim)
        h, e, worst = np.zeros(10), np.zeros(10), 0.0
        for t in range(1000):
            sim.t = t
            update(0)
            h, e, _ = step(h, e, u, .1, 1)
            worst = max(worst, float(np.max(abs(h-sim.h[0]))), float(np.max(abs(e-sim.e[0]))),
                        float(np.max(abs(sim.r))))
        if worst > 1e-12:
            raise AssertionError(f"local-update mismatch: {filename}: {worst}")
        rows.append({"file": filename, "class": classname, "method_ast_sha256": digest,
                     "steps": 1000, "max_abs_deviation": worst,
                     "source_scope": "audited_method_extract" if "Audited method extract" in text[:200]
                                     else "method_extracted_from_repository_file",
                     "execution_scope": "local-update body with constant oracle; not full simulator tests"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    ap.add_argument("--horizon", type=int, default=30000)
    ap.add_argument("--bits", type=int, default=4096)
    ap.add_argument("--upstream-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--skip-source-check", action="store_true")
    args = ap.parse_args()
    if args.horizon < 1000:
        ap.error("--horizon must be at least 1000")
    prefix = certify_prefix(args.bits)
    source = [] if args.skip_source_check else check_source_methods(args.upstream_root)
    traces = [float_trace(r/math.sqrt(10), eta, T=args.horizon)
              for eta in (.1, .05, .018) for r in (.25, .95, .98, 1.0)]
    traces += [float_trace(.31, .1, T=args.horizon),
               float_trace(.31, .1, T=args.horizon, projected=False),
               float_trace(.31, .1, k=10, T=args.horizon)]
    out = {"metadata": {"base_commit": BASE_COMMIT, "script": "repro/projection_closure_probe.py",
                        "fresh_noise": "not used: constant oracle is independent of the iterate",
                        "scope": "local recursion, analytical proofs and integer-certified finite prefix",
                        "float_traces": "empirical only; exact rational prefix certification is separate",
                        "no_algorithm_or_privacy_defaults_changed": True},
           "strict_interior_certificate": prefix, "source_method_checks": source,
           "sufficient_bounds": [sufficient_bound(.25, 1, .1, eta) for eta in (.1, .05, .018)],
           "float_traces": traces}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")
    print(f"Certified prefix S_1000 >= {prefix['S_lower_numerator']}/{prefix['S_denominator']} > 600")
    print(f"Local-method checks: {len(source)}; wrote {path}")


if __name__ == "__main__":
    main()
