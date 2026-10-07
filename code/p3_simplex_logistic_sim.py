"""Paper 3 bounded EControl--DA on simplex-constrained logistic regression.

The parameter w lies in the probability simplex Delta^d={w>=0,sum(w)=1}.
The simplex indicator is the composite term and is handled by the exact
Euclidean simplex projection in the dual-averaging prox.  The main private
mechanism is full-participation (q=1) bounded EControl + Top-K with a fresh
Gaussian perturbation of the clean aggregate release and real iterates.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from p3_bounded_sim import clip_rows, clip_vec, gaussian_sigma, rdp_per_round, topk

Array = np.ndarray


def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def logistic_loss_and_grad(X: Array, y: Array, w: Array) -> Tuple[float, Array]:
    margin = X @ w
    # Stable binary cross entropy, y in {0,1}.
    loss = np.logaddexp(0.0, margin) - y * margin
    per = (sigmoid(margin) - y)[:, None] * X
    return float(np.mean(loss)), per


def simplex_projection(v: Array) -> Array:
    """Exact Euclidean projection onto {x>=0, sum x=1}."""
    if v.ndim != 1:
        raise ValueError("simplex_projection expects a vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.where(u - cssv / (np.arange(v.size) + 1.0) > 0.0)[0]
    if rho_candidates.size == 0:
        # This branch is only reachable for pathological floating point input.
        return np.full_like(v, 1.0 / max(v.size, 1))
    rho = int(rho_candidates[-1])
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def simplex_projection_check(seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    max_sum_err = 0.0
    max_neg = 0.0
    for _ in range(100):
        v = rng.normal(size=11)
        p = simplex_projection(v)
        max_sum_err = max(max_sum_err, abs(float(p.sum()) - 1.0))
        max_neg = max(max_neg, float(max(0.0, -p.min())))
    # Gradient finite difference sanity check.
    X = rng.normal(size=(13, 7))
    y = rng.integers(0, 2, size=13).astype(float)
    w = simplex_projection(rng.normal(size=7))
    _, per = logistic_loss_and_grad(X, y, w)
    analytic = per.mean(axis=0)
    eps = 1e-6
    numeric = np.zeros_like(w)
    for j in range(w.size):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        numeric[j] = (logistic_loss_and_grad(X, y, wp)[0] - logistic_loss_and_grad(X, y, wm)[0]) / (2 * eps)
    return max(max_sum_err, max_neg), float(np.max(np.abs(analytic - numeric)))


def make_simplex_data(
    n_clients: int,
    samples_per_client: int,
    d: int,
    seed: int = 0,
    heterogeneity: float = 0.2,
) -> Tuple[List[Tuple[Array, Array]], Tuple[Array, Array], Array]:
    """Synthetic binary logistic data with a simplex-valued ground truth."""
    rng = np.random.default_rng(seed)
    w_true = rng.dirichlet(np.ones(d))

    def sample(m: int, shift: Optional[Array] = None):
        X = rng.normal(size=(m, d))
        if shift is not None:
            X = X + shift
        logits = X @ w_true
        y = rng.binomial(1, sigmoid(logits)).astype(float)
        return X, y

    clients: List[Tuple[Array, Array]] = []
    for _ in range(n_clients):
        shift = heterogeneity * rng.normal(size=d) / max(np.sqrt(d), 1e-12)
        clients.append(sample(samples_per_client, shift))
    test = sample(max(4000, n_clients * samples_per_client))
    return clients, test, w_true


@dataclass
class Config:
    n_clients: int = 20
    samples_per_client: int = 40
    d: int = 8
    rounds: int = 100
    topk_frac: float = 0.2
    eta: Optional[float] = None
    C0: float = 1.5
    Cg: float = 1.5
    Br: float = 2.0
    Bh: float = 1.0
    Be: float = 2.0
    gamma: float = 5.0
    epsilon: float = 8.0
    delta_dp: float = 1e-5
    seed: int = 0
    private: bool = True
    compression: bool = True
    residual_buffer: bool = True
    accountant: str = "without_replacement_bound"


class SimplexLogisticEControlDA:
    def __init__(self, clients: List[Tuple[Array, Array]], config: Config):
        self.clients = clients
        self.cfg = config
        self.n = len(clients)
        self.d = clients[0][0].shape[1]
        self.k = max(1, int(round(config.topk_frac * self.d)))
        self.eta = config.eta if config.eta is not None else config.topk_frac
        self.rng = np.random.default_rng(config.seed + 41)
        self.h = np.zeros((self.n, self.d))
        self.e = np.zeros((self.n, self.d))
        self.r = np.zeros((self.n, self.d))
        self.last_update = np.zeros(self.n, dtype=int)
        self.w = np.full(self.d, 1.0 / self.d)
        self.G = np.zeros(self.d)
        self.t = 0
        self.prefix_h_clip = np.zeros(self.n)
        self.prefix_e_clip = np.zeros(self.n)
        self.prefix_r_clip = np.zeros(self.n)
        self._E = np.zeros(self.d)
        release_radius = config.Bh if config.compression else config.C0
        self.sensitivity = 2.0 * release_radius / self.n
        self.accountant_q = 1.0
        self.sigma = (
            gaussian_sigma(config.epsilon, config.delta_dp, self.sensitivity,
                           config.rounds, sampling_rate=1.0,
                           accountant_mode=config.accountant)
            if config.private else 0.0
        )
        self.orders = np.arange(2.0, 257.0)
        self.history: List[Dict[str, float]] = []

    def accounted_epsilon(self) -> float:
        if not self.cfg.private:
            return 0.0
        rdp = self.cfg.rounds * rdp_per_round(
            self.orders, self.accountant_q, self.sensitivity, self.sigma,
            mode=self.cfg.accountant)
        return float(np.min(rdp + np.log(1.0 / self.cfg.delta_dp) / (self.orders - 1.0)))

    def _prox_da(self) -> Array:
        # Composite prox: indicator of Delta^d plus gamma/2 ||w||^2.
        return simplex_projection(-self.G / max(self.cfg.gamma, 1e-12))

    def _clipped_gradient(self, i: int):
        X, y = self.clients[i]
        _, per = logistic_loss_and_grad(X, y, self.w)
        clipped = clip_rows(per, self.cfg.C0)
        v = clipped.mean(axis=0)
        raw_mean = per.mean(axis=0)
        frac = float(np.mean(np.linalg.norm(per, axis=1) > self.cfg.C0))
        return v, raw_mean, frac

    def _local_update(self, i: int):
        v, raw_mean, frac = self._clipped_gradient(i)
        rbar = clip_vec(self.r[i], self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.d)
        u = clip_vec(v + rbar, self.cfg.Cg)
        hbar = clip_vec(self.h[i], self.cfg.Bh)
        ebar = clip_vec(self.e[i], self.cfg.Be)
        delta = u - hbar - self.eta * ebar
        delta = clip_vec(delta, self.cfg.Cg + self.cfg.Bh + self.eta * self.cfg.Be)
        q = topk(delta, self.k)
        h_raw = self.h[i] + q
        h_new = clip_vec(h_raw, self.cfg.Bh)
        e_raw = ebar + h_new - u
        e_new = clip_vec(e_raw, self.cfg.Be)
        r_raw = v + rbar - u
        r_new = clip_vec(r_raw, self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.d)
        h_clip = float(np.linalg.norm(h_raw - h_new))
        e_clip = float(np.linalg.norm(e_raw - e_new))
        r_clip = float(np.linalg.norm(r_raw - r_new))
        self.prefix_h_clip[i] += h_clip
        self.prefix_e_clip[i] += e_clip
        self.prefix_r_clip[i] += r_clip
        self.h[i], self.e[i], self.r[i] = h_new, e_new, r_new
        self.last_update[i] = self.t + 1
        return {
            "clipping_fraction": frac,
            "message_norm": float(np.linalg.norm(q)),
            "h_clip": h_clip, "e_clip": e_clip, "r_clip": r_clip,
            "compression_residual_vec": delta - q,
            "clipping_bias_vec": raw_mean - v,
        }

    def step(self):
        old_w = self.w.copy()
        if self.cfg.compression:
            stats = [self._local_update(i) for i in range(self.n)]
            h_clean = self.h.mean(axis=0)
        else:
            stats = []
            gs = []
            for i in range(self.n):
                v, raw, frac = self._clipped_gradient(i)
                self.last_update[i] = self.t + 1
                gs.append(v)
                stats.append({
                    "clipping_fraction": frac, "message_norm": 0.0,
                    "h_clip": 0.0, "e_clip": 0.0, "r_clip": 0.0,
                    "compression_residual_vec": np.zeros(self.d),
                    "clipping_bias_vec": raw - v,
                })
            h_clean = np.mean(gs, axis=0)
        z = self.rng.normal(scale=self.sigma, size=self.d)
        released = h_clean + z
        self.G += released
        self.w = self._prox_da()

        c_vec = np.mean([s["compression_residual_vec"] for s in stats], axis=0)
        beta_vec = np.mean([s["clipping_bias_vec"] for s in stats], axis=0)
        rho_scalar = float(np.mean([s["h_clip"] + s["e_clip"] + s["r_clip"] for s in stats]))
        self._E += c_vec + beta_vec
        self._E[0] += rho_scalar
        age = self.t + 1 - self.last_update
        self.history.append({
            "round": float(self.t + 1),
            "noise_norm": float(np.linalg.norm(z)),
            "released_norm": float(np.linalg.norm(released)),
            "h_clean_norm": float(np.linalg.norm(h_clean)),
            "clipping_fraction": float(np.mean([s["clipping_fraction"] for s in stats])),
            "message_norm": float(np.mean([s["message_norm"] for s in stats])),
            "h_clip_residual": float(np.mean([s["h_clip"] for s in stats])),
            "e_clip_residual": float(np.mean([s["e_clip"] for s in stats])),
            "r_clip_residual": float(np.mean([s["r_clip"] for s in stats])),
            "prefix_clip_residual": float(np.mean(self.prefix_h_clip)),
            "prefix_e_clip_residual": float(np.mean(self.prefix_e_clip)),
            "prefix_r_clip_residual": float(np.mean(self.prefix_r_clip)),
            "mean_state_age": float(np.mean(age)),
            "p90_state_age": float(np.percentile(age, 90)),
            "max_state_age": float(np.max(age)),
            "movement": float(np.linalg.norm(self.w - old_w)),
            "c_t_norm": float(np.linalg.norm(c_vec)),
            "rho_t": rho_scalar,
            "beta_t_norm": float(np.linalg.norm(beta_vec)),
            "E_t_norm": float(np.linalg.norm(self._E)),
            "E_t_sq": float(np.dot(self._E, self._E)),
            "simplex_sum_error": float(abs(self.w.sum() - 1.0)),
            "simplex_min": float(self.w.min()),
        })
        self.t += 1

    def run(self, test: Tuple[Array, Array], w_true: Array) -> Dict:
        for _ in range(self.cfg.rounds):
            self.step()
        Xtest, ytest = test
        test_loss, _ = logistic_loss_and_grad(Xtest, ytest, self.w)
        Xtrain = np.concatenate([p[0] for p in self.clients])
        ytrain = np.concatenate([p[1] for p in self.clients])
        train_loss, _ = logistic_loss_and_grad(Xtrain, ytrain, self.w)
        pred = (sigmoid(Xtest @ self.w) >= 0.5).astype(float)
        acc = float(np.mean(pred == ytest))
        last = self.history[-1]
        bits_value = 32
        index_bits = int(np.ceil(np.log2(max(self.d, 2))))
        values = self.k if self.cfg.compression else self.d
        bits_per_client = self.cfg.rounds * values * (bits_value + index_bits)
        return {
            "objective": float(test_loss), "accuracy": acc,
            "train_objective": float(train_loss),
            "parameter_mse": float(np.mean((self.w - w_true) ** 2)),
            "simplex_sum_error": float(abs(self.w.sum() - 1.0)),
            "simplex_min": float(self.w.min()),
            "sigma": float(self.sigma), "sensitivity": float(self.sensitivity),
            "epsilon": float(self.cfg.epsilon if self.cfg.private else 0.0),
            "delta": float(self.cfg.delta_dp if self.cfg.private else 0.0),
            "accounted_epsilon": float(self.accounted_epsilon()),
            "n_clients": self.n, "d": self.d, "rounds": self.cfg.rounds,
            "topk": self.k, "topk_frac": self.cfg.topk_frac,
            "bits_per_client": int(bits_per_client),
            "bits_total": int(self.n * bits_per_client), "sampling_rate": 1.0,
            "state_age_mean": float(last["mean_state_age"]),
            "state_age_p90": float(last["p90_state_age"]),
            "state_age_max": float(last["max_state_age"]),
            "prefix_clip_residual": float(last["prefix_clip_residual"]),
            "tail_prefix_clip_residual": float(last["h_clip_residual"]),
            "mean_movement": float(np.mean([h["movement"] for h in self.history])),
            "max_movement": float(np.max([h["movement"] for h in self.history])),
            "mean_E_t_norm": float(np.mean([h["E_t_norm"] for h in self.history])),
            "max_E_t_norm": float(np.max([h["E_t_norm"] for h in self.history])),
            "sum_E_t_sq": float(np.sum([h["E_t_sq"] for h in self.history])),
            "mean_c_t_norm": float(np.mean([h["c_t_norm"] for h in self.history])),
            "mean_rho_t": float(np.mean([h["rho_t"] for h in self.history])),
            "mean_beta_t_norm": float(np.mean([h["beta_t_norm"] for h in self.history])),
            "history": self.history,
        }


def run_one(seed: int, args: argparse.Namespace):
    clients, test, w_true = make_simplex_data(
        args.n_clients, args.samples_per_client, args.d, seed,
        heterogeneity=args.heterogeneity)
    common = dict(n_clients=args.n_clients, samples_per_client=args.samples_per_client,
                  d=args.d, rounds=args.rounds, topk_frac=args.topk_frac,
                  epsilon=args.epsilon, delta_dp=args.delta_dp, seed=seed,
                  gamma=args.gamma, C0=args.C0, Cg=args.Cg, Br=args.Br,
                  Bh=args.Bh, Be=args.Be, accountant=args.accountant)
    configs = {
        "nonprivate_bounded_EControl_TopK": Config(**common, private=False, compression=True),
        "centralDP_bounded_EControl_TopK": Config(**common, private=True, compression=True),
        "centralDP_DA_no_compression": Config(**common, private=True, compression=False),
    }
    return {name: SimplexLogisticEControlDA(clients, cfg).run(test, w_true)
            | {"seed": int(seed)} for name, cfg in configs.items()}


def summarize(all_results):
    metrics = ["objective", "accuracy", "train_objective", "parameter_mse",
               "sigma", "sensitivity", "bits_per_client", "state_age_mean",
               "state_age_p90", "state_age_max", "prefix_clip_residual",
               "tail_prefix_clip_residual", "mean_movement", "max_movement",
               "mean_E_t_norm", "max_E_t_norm", "sum_E_t_sq", "mean_c_t_norm",
               "mean_rho_t", "mean_beta_t_norm", "simplex_sum_error", "simplex_min",
               "accounted_epsilon"]
    out = {}
    for name, rows in all_results.items():
        s = {"n_seeds": len(rows)}
        for metric in metrics:
            v = np.array([r[metric] for r in rows], dtype=float)
            s[metric] = float(v.mean())
            s[metric + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[name] = s
    return out


def run_suite(args):
    proj_err, grad_err = simplex_projection_check(args.seed)
    names = ["nonprivate_bounded_EControl_TopK", "centralDP_bounded_EControl_TopK", "centralDP_DA_no_compression"]
    runs = {name: [] for name in names}
    for seed in range(args.seed, args.seed + args.n_seeds):
        result = run_one(seed, args)
        for name in names:
            runs[name].append(result[name])
    summary = summarize(runs)
    for name, s in summary.items():
        print(f"{name:40s} objective={s['objective']:.6f}±{s['objective_sd']:.6f} "
              f"acc={s['accuracy']:.4f}±{s['accuracy_sd']:.4f} sigma={s['sigma']:.5f} "
              f"sens={s['sensitivity']:.5f} bits/client={s['bits_per_client']:.0f} "
              f"state_age={s['state_age_mean']:.2f} move={s['mean_movement']:.5f} "
              f"max_E={s['max_E_t_norm']:.5f}")
    return {"metadata": {
        "task": "Paper3 DP simplex-constrained logistic regression",
        "mechanism": "fresh Gaussian aggregate release; real iterates; full participation q=1",
        "n_seeds": int(args.n_seeds), "seed_start": int(args.seed),
        "simplex_projection_max_feasibility_error": float(proj_err),
        "gradient_finite_difference_max_error": float(grad_err),
        "n_clients": args.n_clients, "samples_per_client": args.samples_per_client,
        "d": args.d, "rounds": args.rounds, "topk_frac": args.topk_frac,
        "epsilon": args.epsilon, "delta_dp": args.delta_dp,
        "C0": args.C0, "Cg": args.Cg, "Br": args.Br, "Bh": args.Bh, "Be": args.Be,
        "gamma": args.gamma, "heterogeneity": args.heterogeneity,
        "accountant": args.accountant,
        "scope_note": "Empirical diagnostics only; no unconditional real-iterate theorem is claimed.",
    }, "summary": summary, "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=20)
    ap.add_argument("--samples-per-client", type=int, default=40)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--topk-frac", type=float, default=0.2)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--delta-dp", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--C0", type=float, default=1.5)
    ap.add_argument("--Cg", type=float, default=1.5)
    ap.add_argument("--Br", type=float, default=2.0)
    ap.add_argument("--Bh", type=float, default=1.0)
    ap.add_argument("--Be", type=float, default=2.0)
    ap.add_argument("--heterogeneity", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--accountant", choices=["without_replacement_bound", "poisson_surrogate", "conservative"], default="without_replacement_bound")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()
    result = run_suite(args)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
