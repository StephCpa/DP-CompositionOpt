"""Bounded private EControl--Dual Averaging on a synthetic multiclass softmax task.

This is a companion diagnostic to ``p3_bounded_sim.py``.  The parameter is a
multiclass weight matrix ``W`` (flattened when communicating), the loss is
multiclass cross-entropy plus an l1 composite term, and the main mechanism is
full-participation bounded EControl + Top-K with a fresh Gaussian perturbation
of the clean aggregate estimate on every round.

The simulator is deliberately small and auditable.  It is intended to test
whether the Paper 3 error-feedback/dual-averaging mechanism survives a second
convex objective; it is not a production DP training implementation.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the tested accountant and vector primitives from the binary simulator.
# Importing the module does not execute its CLI because it is guarded by
# ``if __name__ == '__main__'``.
from p3_bounded_sim import (
    clip_rows,
    clip_vec,
    gaussian_sigma,
    rdp_per_round,
    soft_threshold,
    topk,
)


Array = np.ndarray


def softmax_loss_and_grad(X: Array, y: Array, W: Array, n_classes: int):
    """Return mean cross-entropy and one flattened gradient per example.

    ``W`` has shape ``(d, n_classes)``.  The per-example gradient is
    ``outer(x, p - one_hot(y))`` and is flattened in row-major order so that
    Top-K and the DP sensitivity accounting act on one vector in R^(d*C).
    """
    logits = X @ W
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -60.0, 60.0))
    probs = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
    idx = np.arange(len(y))
    loss = -np.log(np.maximum(probs[idx, y], 1e-12)).mean()
    residual = probs.copy()
    residual[idx, y] -= 1.0
    per = np.einsum("ni,nj->nij", X, residual).reshape(len(y), -1)
    return float(loss), per


def make_softmax_data(
    n_clients: int,
    samples_per_client: int,
    d: int,
    n_classes: int,
    seed: int = 0,
    heterogeneity: float = 0.2,
    label_noise: float = 0.08,
):
    """Create a balanced synthetic multiclass logistic problem.

    A shared random true matrix creates the global decision rule.  Each client
    receives a small feature shift, which gives a mild federated
    heterogeneity while retaining a single convex population objective.
    Labels are argmax softmax classes, with optional symmetric random flips.
    The returned test set is sampled from the global distribution.
    """
    rng = np.random.default_rng(seed)
    W_true = rng.normal(size=(d, n_classes))
    W_true -= W_true.mean(axis=1, keepdims=True)
    W_true /= max(np.linalg.norm(W_true), 1e-12)

    def sample(m: int, client_shift: Optional[Array] = None):
        X = rng.normal(size=(m, d))
        if client_shift is not None:
            X = X + client_shift
        logits = X @ W_true + 0.15 * rng.normal(size=(m, n_classes))
        y = np.argmax(logits, axis=1).astype(int)
        if label_noise > 0:
            flips = rng.random(m) < label_noise
            if np.any(flips):
                offsets = rng.integers(1, n_classes, size=int(flips.sum()))
                y[flips] = (y[flips] + offsets) % n_classes
        return X, y

    clients = []
    for _ in range(n_clients):
        shift = heterogeneity * rng.normal(size=d) / max(np.sqrt(d), 1e-12)
        clients.append(sample(samples_per_client, shift))
    test = sample(max(3000, n_clients * samples_per_client))
    return clients, test, W_true


@dataclass
class Config:
    n_clients: int = 20
    samples_per_client: int = 40
    d: int = 8
    n_classes: int = 3
    rounds: int = 150
    topk_frac: float = 0.1
    eta: Optional[float] = None
    C0: float = 1.5
    Cg: float = 1.5
    Br: float = 2.0
    Bh: float = 1.0
    Be: float = 2.0
    l1: float = 0.002
    gamma: float = 5.0
    epsilon: float = 8.0
    delta_dp: float = 1e-5
    seed: int = 0
    private: bool = True
    compression: bool = True
    residual_buffer: bool = True
    accountant: str = "without_replacement_bound"


class SoftmaxEControlDA:
    """Full-participation bounded EControl + Dual Averaging simulator."""

    def __init__(self, clients: List[Tuple[Array, Array]], config: Config):
        self.clients = clients
        self.cfg = config
        self.n = len(clients)
        self.d = clients[0][0].shape[1]
        self.n_classes = config.n_classes
        self.D = self.d * self.n_classes
        self.k = max(1, int(round(config.topk_frac * self.D)))
        self.eta = config.eta if config.eta is not None else config.topk_frac
        self.rng = np.random.default_rng(config.seed + 19)
        self.h = np.zeros((self.n, self.D))
        self.e = np.zeros((self.n, self.D))
        self.r = np.zeros((self.n, self.D))
        self.W = np.zeros((self.d, self.n_classes))
        self.G = np.zeros(self.D)
        self.A = 0.0
        self.t = 0

        # Full participation: one client replacement changes one bounded local
        # state in an n-way average, hence sensitivity 2 Bh / n.  The dense
        # reference uses the clipped local gradient radius C0 instead.
        release_radius = config.Bh if config.compression else config.C0
        self.sensitivity = 2.0 * release_radius / self.n
        self.accountant_q = 1.0
        self.sigma = (
            gaussian_sigma(
                config.epsilon,
                config.delta_dp,
                self.sensitivity,
                config.rounds,
                sampling_rate=1.0,
                accountant_mode=config.accountant,
            )
            if config.private
            else 0.0
        )
        self.orders = np.arange(2.0, 257.0)
        self.history: List[Dict[str, float]] = []

    def accounted_epsilon(self) -> float:
        if not self.cfg.private:
            return 0.0
        rdp = self.cfg.rounds * rdp_per_round(
            self.orders,
            self.accountant_q,
            self.sensitivity,
            self.sigma,
            mode=self.cfg.accountant,
        )
        return float(
            np.min(rdp + np.log(1.0 / self.cfg.delta_dp) / (self.orders - 1.0))
        )

    def _prox_da(self):
        # psi(W) = l1 * ||W||_1 and Euclidean prox around x_0 = 0.
        return soft_threshold(
            -self.G / max(self.cfg.gamma, 1e-12),
            self.A * self.cfg.l1 / max(self.cfg.gamma, 1e-12),
        )

    def _clipped_gradient(self, i: int):
        X, y = self.clients[i]
        _, per = softmax_loss_and_grad(X, y, self.W, self.n_classes)
        clipped = clip_rows(per, self.cfg.C0)
        v = clipped.mean(axis=0)
        frac = float(np.mean(np.linalg.norm(per, axis=1) > self.cfg.C0))
        return v, per, frac

    def _local_update(self, i: int):
        v, _, clipping_fraction = self._clipped_gradient(i)
        rbar = clip_vec(self.r[i], self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.D)
        u = clip_vec(v + rbar, self.cfg.Cg)
        hbar = clip_vec(self.h[i], self.cfg.Bh)
        ebar = clip_vec(self.e[i], self.cfg.Be)

        delta = u - hbar - self.eta * ebar
        delta_radius = self.cfg.Cg + self.cfg.Bh + self.eta * self.cfg.Be
        delta = clip_vec(delta, delta_radius)
        q = topk(delta, self.k)
        h_raw = self.h[i] + q
        h_new = clip_vec(h_raw, self.cfg.Bh)
        e_raw = ebar + h_new - u
        e_new = clip_vec(e_raw, self.cfg.Be)
        r_raw = v + rbar - u
        r_new = clip_vec(r_raw, self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.D)

        diagnostics = {
            "clipping_fraction": clipping_fraction,
            "message_norm": float(np.linalg.norm(q)),
            "h_clip_residual": float(np.linalg.norm(h_raw - h_new)),
            "e_clip_residual": float(np.linalg.norm(e_raw - e_new)),
            "r_norm": float(np.linalg.norm(r_new)),
            "e_norm": float(np.linalg.norm(e_new)),
        }
        self.h[i], self.e[i], self.r[i] = h_new, e_new, r_new
        return diagnostics

    def step(self):
        active = np.arange(self.n)
        if self.cfg.compression:
            local_stats = [self._local_update(int(i)) for i in active]
            h_clean = self.h.mean(axis=0)
        else:
            local_stats = []
            gs = []
            for i in active:
                v, _, frac = self._clipped_gradient(int(i))
                gs.append(v)
                local_stats.append(
                    {
                        "clipping_fraction": frac,
                        "message_norm": 0.0,
                        "h_clip_residual": 0.0,
                        "e_clip_residual": 0.0,
                        "r_norm": 0.0,
                        "e_norm": 0.0,
                    }
                )
            h_clean = np.mean(gs, axis=0)

        # Fresh aggregate noise is added to the current clean estimate.  The
        # noisy release is never fed into a second server-side accumulator.
        z = self.rng.normal(scale=self.sigma, size=self.D)
        released = h_clean + z
        self.G += released
        self.A += 1.0
        self.W = self._prox_da().reshape(self.d, self.n_classes)

        self.history.append(
            {
                "h_clean_norm": float(np.linalg.norm(h_clean)),
                "noise_norm": float(np.linalg.norm(z)),
                "released_norm": float(np.linalg.norm(released)),
                "clipping_fraction": float(np.mean([s["clipping_fraction"] for s in local_stats])),
                "message_norm": float(np.mean([s["message_norm"] for s in local_stats])),
                "h_clip_residual": float(np.mean([s["h_clip_residual"] for s in local_stats])),
                "e_clip_residual": float(np.mean([s["e_clip_residual"] for s in local_stats])),
                "r_norm": float(np.mean([s["r_norm"] for s in local_stats])),
                "e_norm": float(np.mean([s["e_norm"] for s in local_stats])),
            }
        )
        self.t += 1

    def run(self, test: Tuple[Array, Array]):
        for _ in range(self.cfg.rounds):
            self.step()
        Xtest, ytest = test
        loss, _ = softmax_loss_and_grad(Xtest, ytest, self.W, self.n_classes)
        pred = np.argmax(Xtest @ self.W, axis=1)
        acc = float(np.mean(pred == ytest))
        objective = float(loss + self.cfg.l1 * np.abs(self.W).sum())
        last = self.history[-1]
        bits_per_value = 32
        index_bits = int(np.ceil(np.log2(max(self.D, 2))))
        values_per_round = self.k if self.cfg.compression else self.D
        bits_per_client = self.cfg.rounds * values_per_round * (bits_per_value + index_bits)
        return {
            "objective": objective,
            "cross_entropy": float(loss),
            "accuracy": acc,
            "W_norm": float(np.linalg.norm(self.W)),
            "sigma": float(self.sigma),
            "sensitivity": float(self.sensitivity),
            "epsilon": float(self.cfg.epsilon if self.cfg.private else 0.0),
            "delta": float(self.cfg.delta_dp if self.cfg.private else 0.0),
            "accounted_epsilon": float(self.accounted_epsilon()),
            "n_clients": int(self.n),
            "d_features": int(self.d),
            "n_classes": int(self.n_classes),
            "parameter_dim": int(self.D),
            "topk": int(self.k),
            "bits_total": int(self.n * bits_per_client),
            "bits_per_client": int(bits_per_client),
            "sampling_rate": 1.0,
            # Last-round diagnostics are included at top level for concise
            # tables; complete trajectories remain in ``history``.
            "clipping_fraction": float(last["clipping_fraction"]),
            "message_norm": float(last["message_norm"]),
            "h_clip_residual": float(last["h_clip_residual"]),
            "e_clip_residual": float(last["e_clip_residual"]),
            "r_norm": float(last["r_norm"]),
            "e_norm": float(last["e_norm"]),
            "residuals": {
                "clipping_fraction": float(last["clipping_fraction"]),
                "h_clip_residual": float(last["h_clip_residual"]),
                "e_clip_residual": float(last["e_clip_residual"]),
                "r_norm": float(last["r_norm"]),
                "e_norm": float(last["e_norm"]),
            },
            "history": self.history,
        }


def run_suite(args):
    clients, test, _ = make_softmax_data(
        n_clients=args.n_clients,
        samples_per_client=args.samples_per_client,
        d=args.d,
        n_classes=args.n_classes,
        seed=args.seed,
        heterogeneity=args.heterogeneity,
        label_noise=args.label_noise,
    )
    common = dict(
        n_clients=args.n_clients,
        samples_per_client=args.samples_per_client,
        d=args.d,
        n_classes=args.n_classes,
        rounds=args.rounds,
        topk_frac=args.topk_frac,
        epsilon=args.epsilon,
        delta_dp=args.delta_dp,
        seed=args.seed,
        gamma=args.gamma,
        l1=args.l1,
        C0=args.C0,
        Cg=args.Cg,
        Br=args.Br,
        Bh=args.Bh,
        Be=args.Be,
        accountant=args.accountant,
    )
    configs = {
        "nonprivate_bounded_EControl_TopK": Config(**common, private=False, compression=True),
        "centralDP_bounded_EControl_TopK": Config(**common, private=True, compression=True),
        "centralDP_DA_no_compression": Config(**common, private=True, compression=False),
    }
    results = {}
    for name, cfg in configs.items():
        result = SoftmaxEControlDA(clients, cfg).run(test)
        results[name] = result
        print(
            f"{name:40s} objective={result['objective']:.5f} "
            f"acc={result['accuracy']:.3f} sigma={result['sigma']:.5f} "
            f"sens={result['sensitivity']:.5f} eps={result['accounted_epsilon']:.4f} "
            f"bits/client={result['bits_per_client']:.0f} "
            f"hclip={result['h_clip_residual']:.4f} "
            f"eclip={result['e_clip_residual']:.4f} "
            f"r={result['r_norm']:.4f}"
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=20)
    ap.add_argument("--samples-per-client", type=int, default=40)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--topk-frac", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--delta-dp", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--l1", type=float, default=0.002)
    ap.add_argument("--C0", type=float, default=1.5)
    ap.add_argument("--Cg", type=float, default=1.5)
    ap.add_argument("--Br", type=float, default=2.0)
    ap.add_argument("--Bh", type=float, default=1.0)
    ap.add_argument("--Be", type=float, default=2.0)
    ap.add_argument("--heterogeneity", type=float, default=0.2)
    ap.add_argument("--label-noise", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--accountant",
        choices=["without_replacement_bound", "poisson_surrogate", "conservative"],
        default="without_replacement_bound",
    )
    args = ap.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
