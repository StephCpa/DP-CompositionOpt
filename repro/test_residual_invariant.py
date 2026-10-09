#!/usr/bin/env python3
"""Regression gate for the residual-buffer invariant used by the closure note.

In all three simulators ``u = clip_Cg(v + clip_Br(r))`` and
``r_next = clip_Br(v + clip_Br(r) - u)``, with ``r_0 = 0`` and ``v`` the mean
of per-example gradients clipped to ``C0`` (so ``||v|| <= C0``).  If
``Cg >= C0`` then by induction ``r`` stays exactly zero and ``u = v``
(``r_next = 0`` iff ``u = v`` when ``r = 0``).  The docs rely on this
(``r-zero`` in ``docs/p3_Et_closure_lemma_zh.md``).  These tests fail if a
change to how ``v`` is formed, how ``r`` is initialised or updated, or the
clipping order silently breaks it.  A ``Cg < ||v||`` control checks that the
buffer can still activate, so the zero result is not vacuous.

Run from the repository root::

    python repro/test_residual_invariant.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import p3_box_ls_sim as box  # noqa: E402
import p3_simplex_logistic_sim as simplex  # noqa: E402
import p3_softmax_sim as softmax  # noqa: E402

ROUNDS = 30
SEEDS = (0, 1)


def _build(task: str, seed: int, **overrides):
    if task == "box":
        clients, _, _ = box.make_box_ls_data(20, 40, 10, seed=seed)
        return box.BoxLSEControlDA(clients, box.Config(seed=seed, rounds=ROUNDS, **overrides))
    if task == "simplex":
        clients, _, _ = simplex.make_simplex_data(20, 40, 8, seed, heterogeneity=0.2)
        return simplex.SimplexLogisticEControlDA(clients, simplex.Config(seed=seed, rounds=ROUNDS, **overrides))
    clients, _, _ = softmax.make_softmax_data(
        n_clients=20, samples_per_client=40, d=8, n_classes=3, seed=seed,
        heterogeneity=0.2, label_noise=0.08,
    )
    return softmax.SoftmaxEControlDA(
        clients, softmax.Config(d=8, n_classes=3, rounds=ROUNDS, seed=seed, **overrides)
    )


def _max_residual_norm(task: str, **overrides) -> float:
    worst = 0.0
    for seed in SEEDS:
        sim = _build(task, seed, **overrides)
        assert sim.cfg.residual_buffer, "the invariant is only meaningful with the buffer enabled"
        # Include r_0: a small nonzero start can be flushed after one step and
        # would otherwise go unnoticed.
        worst = max(worst, float(np.linalg.norm(sim.r, axis=1).max()))
        for _ in range(ROUNDS):
            sim.step()
            worst = max(worst, float(np.linalg.norm(sim.r, axis=1).max()))
    return worst


class ResidualZeroInvariant(unittest.TestCase):
    """``Cg >= C0`` and ``r_0 = 0`` keep the residual buffer exactly zero."""

    def _check_task(self, task: str) -> None:
        for C0, Cg in ((1.5, 1.5), (1.5, 3.0), (0.5, 0.5)):
            with self.subTest(task=task, C0=C0, Cg=Cg):
                self.assertEqual(_max_residual_norm(task, C0=C0, Cg=Cg), 0.0)

    def test_box(self) -> None:
        self._check_task("box")

    def test_simplex(self) -> None:
        self._check_task("simplex")

    def test_softmax(self) -> None:
        self._check_task("softmax")


class ResidualControlActivates(unittest.TestCase):
    """With ``Cg`` below the clipped-gradient norms the buffer must become nonzero."""

    def test_control(self) -> None:
        for task in ("box", "simplex", "softmax"):
            with self.subTest(task=task):
                self.assertGreater(_max_residual_norm(task, C0=1.5, Cg=0.1), 1e-3)


if __name__ == "__main__":
    unittest.main()
