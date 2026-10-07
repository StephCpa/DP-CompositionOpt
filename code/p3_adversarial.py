"""Black-box adversarial search for untruncated Paper-3 EControl.

This is a diagnostic, not a proof.  It searches over two independent bounded
input streams ||g_t||_2 <= 1 and tries to maximize the running difference of
the released estimate h_{t+1}=h_t+TopK(g_t-h_t-eta e_t).

The search uses projected candidate directions, coordinate directions, state
directions, and random unit vectors.  It is deliberately separate from the
uploaded p3_checks.py so that the original file remains unchanged.
"""
from __future__ import annotations

import argparse
import numpy as np


def topk(v: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(v)
    idx = np.argpartition(np.abs(v), -k)[-k:]
    out[idx] = v[idx]
    return out


def unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return None if n < 1e-12 else v / n


def step(h: np.ndarray, e: np.ndarray, g: np.ndarray, k: int, eta: float):
    # Match the sign convention in upload/p3_checks.py: its buffer is
    # e_{t+1}=e_t+g_t-h_{t+1} and the compressed input is g_t-h_t+eta e_t.
    delta = topk(g - h + eta * e, k)
    h_new = h + delta
    e_new = e + g - h_new
    return h_new, e_new, delta


def candidate_pool(d, rng, h1, e1, h2, e2, eta, n_random=16):
    vecs = []
    # Coordinate directions make the pool capable of intentionally changing
    # Top-K support.  They are cheap and useful in low-dimensional searches.
    eye = np.eye(d)
    for j in range(d):
        vecs.extend((eye[j], -eye[j]))
    for v in (h1, -h1, e1, -e1, h2, -h2, e2, -e2,
              h1 - h2, h2 - h1, e1 - e2, e2 - e1,
              h1 + eta * e1, h2 + eta * e2):
        u = unit(v)
        if u is not None:
            vecs.append(u)
    for _ in range(n_random):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        vecs.append(v)
    # Remove near duplicates while keeping deterministic order.
    out = []
    for v in vecs:
        if not any(np.linalg.norm(v - w) < 1e-9 for w in out):
            out.append(v)
    return np.asarray(out)


def search(d=16, k=1, T=250, eta=None, restarts=32, coord_rounds=4,
           random_candidates=16, seed=0):
    if eta is None:
        eta = k / d
    rng = np.random.default_rng(seed)
    best = None
    for restart in range(restarts):
        h1 = np.zeros(d); e1 = np.zeros(d)
        h2 = np.zeros(d); e2 = np.zeros(d)
        max_h = 0.0; max_delta = 0.0; max_e = 0.0
        record = None
        for t in range(T):
            pool = candidate_pool(d, rng, h1, e1, h2, e2, eta,
                                  n_random=random_candidates)
            # Coordinate ascent on the pair (g1, g2), with the other stream
            # fixed at each substep.  Objective is the next released h gap.
            g1 = pool[rng.integers(len(pool))]
            g2 = pool[rng.integers(len(pool))]
            for _ in range(coord_rounds):
                scores = []
                for cand in pool:
                    nh1, _, _ = step(h1, e1, cand, k, eta)
                    nh2, _, _ = step(h2, e2, g2, k, eta)
                    scores.append(np.linalg.norm(nh1 - nh2))
                g1 = pool[int(np.argmax(scores))]
                scores = []
                nh1, _, _ = step(h1, e1, g1, k, eta)
                for cand in pool:
                    nh2, _, _ = step(h2, e2, cand, k, eta)
                    scores.append(np.linalg.norm(nh1 - nh2))
                g2 = pool[int(np.argmax(scores))]
            h1, e1, q1 = step(h1, e1, g1, k, eta)
            h2, e2, q2 = step(h2, e2, g2, k, eta)
            dh = np.linalg.norm(h1 - h2)
            dd = np.linalg.norm(q1 - q2)
            de = np.linalg.norm(e1 - e2)
            if dh > max_h:
                max_h = dh
                record = (t + 1, max_h, dd, de)
            max_delta = max(max_delta, dd)
            max_e = max(max_e, de)
        result = dict(restart=restart, max_h=max_h, max_delta=max_delta,
                      max_e=max_e, eta_times_e=eta * max_e,
                      record=record)
        if best is None or result['max_h'] > best['max_h']:
            best = result
        if restart == 0 or result['max_h'] >= best['max_h']:
            print(f"restart={restart:3d} max||h-h'||={max_h:8.4f} "
                  f"max||q-q'||={max_delta:7.4f} "
                  f"max||e-e'||={max_e:9.3f} eta*e={eta*max_e:8.4f} "
                  f"at={record[0] if record else 0}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=16)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--T', type=int, default=250)
    ap.add_argument('--eta', type=float, default=None)
    ap.add_argument('--restarts', type=int, default=32)
    ap.add_argument('--coord-rounds', type=int, default=4)
    ap.add_argument('--random-candidates', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    print(vars(args))
    best = search(**vars(args))
    print('BEST', best)


if __name__ == '__main__':
    main()
