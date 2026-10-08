"""Paper 3 bounded EControl--dual averaging on box-constrained least squares (signed residual diagnostic).

The mechanism is the mainline client-level central-DP diagnostic:
full participation (q=1), bounded local EControl, Top-K communication,
fresh Gaussian noise on the clean aggregate release, and real iterates.
The noisy release is *not* integrated into the next clean aggregate.

Three configurations are compared:
  nonprivate_bounded_EControl_TopK,
  centralDP_bounded_EControl_TopK,
  centralDP_DA_no_compression.

The objective is convex least squares over a box X=[-Bbox,Bbox]^d.  The
box indicator is the composite term and is handled by the dual-averaging prox.
This is a research simulator rather than a production DP implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from p3_bounded_sim import (
    clip_vec,
    clip_rows,
    gaussian_sigma,
    rdp_per_round,
    topk,
)

Array = np.ndarray


def project_box(x: Array, radius: float) -> Array:
    return np.clip(x, -radius, radius)


def ls_loss_and_grad(X: Array, y: Array, x: Array) -> Tuple[float, Array]:
    """Return mean 1/2 squared loss and one per-example gradient."""
    residual = X @ x - y
    loss = 0.5 * float(np.mean(residual ** 2))
    per = residual[:, None] * X
    return loss, per


def finite_difference_check(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(9, 5))
    y = rng.normal(size=9)
    x = rng.normal(size=5)
    loss, per = ls_loss_and_grad(X, y, x)
    analytic = per.mean(axis=0)
    eps = 1e-6
    numeric = np.zeros_like(x)
    for j in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        numeric[j] = (ls_loss_and_grad(X, y, xp)[0] - ls_loss_and_grad(X, y, xm)[0]) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)))


def make_box_ls_data(
    n_clients: int,
    samples_per_client: int,
    d: int,
    seed: int = 0,
    heterogeneity: float = 0.2,
    noise_std: float = 0.15,
    box_radius: float = 2.0,
) -> Tuple[List[Tuple[Array, Array]], Tuple[Array, Array], Array]:
    """Synthetic heterogeneous linear regression with a bounded optimum."""
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    # Put the population optimum comfortably inside the box.
    w_true *= min(1.0, 0.65 * box_radius / max(np.linalg.norm(w_true), 1e-12))
    clients: List[Tuple[Array, Array]] = []
    for _ in range(n_clients):
        shift = heterogeneity * rng.normal(size=d) / max(np.sqrt(d), 1e-12)
        X = rng.normal(size=(samples_per_client, d)) + shift
        y = X @ w_true + noise_std * rng.normal(size=samples_per_client)
        clients.append((X, y))
    Xtest = rng.normal(size=(max(4000, n_clients * samples_per_client), d))
    ytest = Xtest @ w_true + noise_std * rng.normal(size=Xtest.shape[0])
    return clients, (Xtest, ytest), w_true


@dataclass
class Config:
    n_clients: int = 20
    samples_per_client: int = 40
    d: int = 10
    rounds: int = 100
    topk_frac: float = 0.1
    eta: Optional[float] = None
    C0: float = 1.5
    Cg: float = 1.5
    Br: float = 2.0
    Bh: float = 1.0
    Be: float = 2.0
    Bbox: float = 2.0
    gamma: float = 5.0
    epsilon: float = 8.0
    delta_dp: float = 1e-5
    seed: int = 0
    private: bool = True
    compression: bool = True
    residual_buffer: bool = True
    accountant: str = "without_replacement_bound"


class BoxLSEControlDA:
    def __init__(self, clients: List[Tuple[Array, Array]], config: Config):
        self.clients = clients
        self.cfg = config
        self.n = len(clients)
        self.d = clients[0][0].shape[1]
        self.k = max(1, int(round(config.topk_frac * self.d)))
        # The EControl coefficient is the compression ratio in Paper 3's
        # contractive-error update.  Dense reference does not use it.
        self.eta = config.eta if config.eta is not None else config.topk_frac
        self.rng = np.random.default_rng(config.seed + 29)
        self.h = np.zeros((self.n, self.d))
        self.e = np.zeros((self.n, self.d))
        self.r = np.zeros((self.n, self.d))
        self.last_update = np.zeros(self.n, dtype=int)
        self.x = np.zeros(self.d)
        # Sum of real iterates for theorem-facing averaged-iterate metrics.
        # ``x`` continues to denote the last iterate for compatibility with
        # the historical signed diagnostic outputs.
        self.x_sum = np.zeros(self.d)
        self.G = np.zeros(self.d)
        self.A = 0.0
        self.t = 0
        self.prefix_clip_residual = np.zeros(self.n)
        self.prefix_e_clip_residual = np.zeros(self.n)
        self.prefix_r_clip_residual = np.zeros(self.n)

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
            if config.private else 0.0
        )
        self.orders = np.arange(2.0, 257.0)
        self.history: List[Dict[str, float]] = []
        self._E = np.zeros(self.d)

    def accounted_epsilon(self) -> float:
        if not self.cfg.private:
            return 0.0
        rdp = self.cfg.rounds * rdp_per_round(
            self.orders, self.accountant_q, self.sensitivity, self.sigma,
            mode=self.cfg.accountant,
        )
        return float(np.min(rdp + np.log(1.0 / self.cfg.delta_dp) / (self.orders - 1.0)))

    def _prox_da(self) -> Array:
        # Composite prox of the box indicator plus 1/2 ||x||^2:
        # argmin_{|x_j|<=Bbox} <G,x> + gamma/2 ||x||^2.
        return project_box(-self.G / max(self.cfg.gamma, 1e-12), self.cfg.Bbox)

    def _clipped_gradient(self, i: int):
        X, y = self.clients[i]
        _, per = ls_loss_and_grad(X, y, self.x)
        clipped = clip_rows(per, self.cfg.C0)
        v = clipped.mean(axis=0)
        raw_mean = per.mean(axis=0)
        frac = float(np.mean(np.linalg.norm(per, axis=1) > self.cfg.C0))
        return v, raw_mean, frac

    def _local_update(self, i: int):
        v, raw_mean, clipping_fraction = self._clipped_gradient(i)
        rbar = clip_vec(self.r[i], self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.d)
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
        r_new = clip_vec(r_raw, self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.d)

        h_clip = float(np.linalg.norm(h_raw - h_new))
        e_clip = float(np.linalg.norm(e_raw - e_new))
        r_clip = float(np.linalg.norm(r_raw - r_new))
        self.prefix_clip_residual[i] += h_clip
        self.prefix_e_clip_residual[i] += e_clip
        self.prefix_r_clip_residual[i] += r_clip
        self.h[i], self.e[i], self.r[i] = h_new, e_new, r_new
        self.last_update[i] = self.t + 1

        return {
            "clipping_fraction": clipping_fraction,
            "message_norm": float(np.linalg.norm(q)),
            "h_clip_residual": h_clip,
            "e_clip_residual": e_clip,
            "r_clip_residual": r_clip,
            "h_clip_vec": h_raw - h_new,
            "e_clip_vec": e_raw - e_new,
            "r_clip_vec": r_raw - r_new,
            "r_norm": float(np.linalg.norm(r_new)),
            "e_norm": float(np.linalg.norm(e_new)),
            "compression_residual_vec": delta - q,
            "u_vec": u.copy(),
            "v_vec": v.copy(),
            "raw_mean_vec": raw_mean.copy(),
            # The theory uses beta_t = v_t - raw_mean_t.
            "clipping_bias_vec": v - raw_mean,
        }

    def step(self):
        x_before = self.x.copy()
        if self.cfg.compression:
            local_stats = [self._local_update(i) for i in range(self.n)]
            h_clean = self.h.mean(axis=0)
        else:
            local_stats = []
            gs = []
            for i in range(self.n):
                v, raw_mean, frac = self._clipped_gradient(i)
                # Dense DA has no persistent compressed state; treating the
                # current local oracle as freshly updated keeps the same
                # state-age diagnostic (zero) across the comparison.
                self.last_update[i] = self.t + 1
                gs.append(v)
                local_stats.append({
                    "clipping_fraction": frac,
                    "message_norm": 0.0,
                    "h_clip_residual": 0.0,
                    "e_clip_residual": 0.0,
                    "r_clip_residual": 0.0,
                    "h_clip_vec": np.zeros(self.d),
                    "e_clip_vec": np.zeros(self.d),
                    "r_clip_vec": np.zeros(self.d),
                    "r_norm": 0.0,
                    "e_norm": 0.0,
                    "compression_residual_vec": np.zeros(self.d),
                    "u_vec": v.copy(),
                    "v_vec": v.copy(),
                    "raw_mean_vec": raw_mean.copy(),
                    "clipping_bias_vec": v - raw_mean,
                })
            h_clean = np.mean(gs, axis=0)

        # Fresh aggregate release: no server-side random walk.
        z = self.rng.normal(scale=self.sigma, size=self.d)
        released = h_clean + z
        self.G += released
        self.A += 1.0
        self.x = self._prox_da()
        self.x_sum += self.x

        # Theorem-aligned decomposition:
        # c_t   = H_t - mean_i u_{i,t}
        # rho_t = mean_i(u_{i,t} - v_{i,t})
        # beta_t = mean_i(v_{i,t} - raw_mean_{i,t})
        # E_t accumulates these signed vectors.  Top-K residuals and state
        # projection residuals remain logged separately as diagnostics.
        u_bar = np.mean([s["u_vec"] for s in local_stats], axis=0)
        v_bar = np.mean([s["v_vec"] for s in local_stats], axis=0)
        raw_bar = np.mean([s["raw_mean_vec"] for s in local_stats], axis=0)
        c_vec = h_clean - u_bar
        rho_vec = u_bar - v_bar
        beta_vec = v_bar - raw_bar
        projection_vec = np.mean([
            s["h_clip_vec"] + s["e_clip_vec"] + s["r_clip_vec"]
            for s in local_stats
        ], axis=0)
        projection_norm = float(np.linalg.norm(projection_vec))
        self._E += c_vec + rho_vec + beta_vec

        movement = float(np.linalg.norm(self.x - x_before))
        state_age = self.t + 1 - self.last_update
        self.history.append({
            "round": float(self.t + 1),
            "h_clean_norm": float(np.linalg.norm(h_clean)),
            "noise_norm": float(np.linalg.norm(z)),
            "released_norm": float(np.linalg.norm(released)),
            "clipping_fraction": float(np.mean([s["clipping_fraction"] for s in local_stats])),
            "message_norm": float(np.mean([s["message_norm"] for s in local_stats])),
            "h_clip_residual": float(np.mean([s["h_clip_residual"] for s in local_stats])),
            "e_clip_residual": float(np.mean([s["e_clip_residual"] for s in local_stats])),
            "r_clip_residual": float(np.mean([s["r_clip_residual"] for s in local_stats])),
            "prefix_clip_residual": float(np.mean(self.prefix_clip_residual)),
            "prefix_e_clip_residual": float(np.mean(self.prefix_e_clip_residual)),
            "prefix_r_clip_residual": float(np.mean(self.prefix_r_clip_residual)),
            "mean_state_age": float(np.mean(state_age)),
            "p90_state_age": float(np.percentile(state_age, 90)),
            "max_state_age": float(np.max(state_age)),
            "movement": movement,
            "c_t_norm": float(np.linalg.norm(c_vec)),
            "c_t_vec": c_vec.tolist(),
            "rho_t": float(np.linalg.norm(rho_vec)),
            "rho_t_vec": rho_vec.tolist(),
            "beta_t_norm": float(np.linalg.norm(beta_vec)),
            "beta_t_vec": beta_vec.tolist(),
            "projection_residual_signed_norm": projection_norm,
            "E_t_norm": float(np.linalg.norm(self._E)),
            "E_t_sq": float(np.dot(self._E, self._E)),
            "E_t_vec": self._E.tolist(),
        })
        self.t += 1

    def run(self, test: Tuple[Array, Array], w_true: Array) -> Dict:
        for _ in range(self.cfg.rounds):
            self.step()
        Xtest, ytest = test
        test_loss, _ = ls_loss_and_grad(Xtest, ytest, self.x)
        train_X = np.concatenate([xy[0] for xy in self.clients], axis=0)
        train_y = np.concatenate([xy[1] for xy in self.clients], axis=0)
        train_loss, _ = ls_loss_and_grad(train_X, train_y, self.x)
        x_average = self.x_sum / max(self.cfg.rounds, 1)
        average_test_loss, _ = ls_loss_and_grad(Xtest, ytest, x_average)
        average_train_loss, _ = ls_loss_and_grad(train_X, train_y, x_average)
        last = self.history[-1]
        bits_per_value = 32
        index_bits = int(np.ceil(np.log2(max(self.d, 2))))
        values_per_round = self.k if self.cfg.compression else self.d
        bits_per_client = self.cfg.rounds * (
            values_per_round * (bits_per_value + index_bits)
            if self.cfg.compression else self.d * bits_per_value
        )
        hist = self.history
        out = {
            "objective": float(test_loss),
            "test_mse": float(2.0 * test_loss),
            "train_mse": float(2.0 * train_loss),
            "parameter_mse": float(np.mean((self.x - w_true) ** 2)),
            "average_objective": float(average_test_loss),
            "average_test_mse": float(2.0 * average_test_loss),
            "average_train_mse": float(2.0 * average_train_loss),
            "average_parameter_mse": float(np.mean((x_average - w_true) ** 2)),
            "average_x_norm": float(np.linalg.norm(x_average)),
            "average_iterate": x_average.tolist(),
            "iterate_reporting": "last_and_uniform_average",
            "x_norm": float(np.linalg.norm(self.x)),
            "sigma": float(self.sigma),
            "sensitivity": float(self.sensitivity),
            "epsilon": float(self.cfg.epsilon if self.cfg.private else 0.0),
            "delta": float(self.cfg.delta_dp if self.cfg.private else 0.0),
            "accounted_epsilon": float(self.accounted_epsilon()),
            "n_clients": int(self.n),
            "d": int(self.d),
            "rounds": int(self.cfg.rounds),
            "topk": int(self.k),
            "topk_frac": float(self.cfg.topk_frac),
            "bits_total": int(self.n * bits_per_client),
            "bits_per_client": int(bits_per_client),
            "sampling_rate": 1.0,
            "state_age_mean": float(last["mean_state_age"]),
            "state_age_p90": float(last["p90_state_age"]),
            "state_age_max": float(last["max_state_age"]),
            "prefix_clip_residual": float(last["prefix_clip_residual"]),
            "tail_prefix_clip_residual": float(last["h_clip_residual"]),
            "mean_movement": float(np.mean([h["movement"] for h in hist])),
            "sum_movement": float(np.sum([h["movement"] for h in hist])),
            "mean_E_t_norm": float(np.mean([h["E_t_norm"] for h in hist])),
            "max_E_t_norm": float(np.max([h["E_t_norm"] for h in hist])),
            "sum_E_t_sq": float(np.sum([h["E_t_sq"] for h in hist])),
            "mean_c_t_norm": float(np.mean([h["c_t_norm"] for h in hist])),
            "mean_rho_t": float(np.mean([h["rho_t"] for h in hist])),
            "mean_beta_t_norm": float(np.mean([h["beta_t_norm"] for h in hist])),
            "clipping_fraction": float(last["clipping_fraction"]),
            "history": hist,
        }
        return out


def run_one(seed: int, args: argparse.Namespace) -> Dict[str, Dict]:
    clients, test, w_true = make_box_ls_data(
        args.n_clients, args.samples_per_client, args.d, seed=seed,
        heterogeneity=args.heterogeneity, noise_std=args.noise_std,
        box_radius=args.Bbox,
    )
    common = dict(
        n_clients=args.n_clients, samples_per_client=args.samples_per_client,
        d=args.d, rounds=args.rounds, topk_frac=args.topk_frac,
        epsilon=args.epsilon, delta_dp=args.delta_dp, seed=seed,
        gamma=args.gamma, C0=args.C0, Cg=args.Cg, Br=args.Br,
        Bh=args.Bh, Be=args.Be, Bbox=args.Bbox,
        accountant=args.accountant,
    )
    configs = {
        "nonprivate_bounded_EControl_TopK": Config(**common, private=False, compression=True),
        "centralDP_bounded_EControl_TopK": Config(**common, private=True, compression=True),
        "centralDP_DA_no_compression": Config(**common, private=True, compression=False),
    }
    dense_c0 = getattr(args, "dense_C0", None)
    if dense_c0 is not None:
        dense_common = dict(common)
        dense_common["C0"] = float(dense_c0)
        configs["centralDP_DA_no_compression_matched_sensitivity"] = Config(
            **dense_common, private=True, compression=False
        )
    results: Dict[str, Dict] = {}
    for name, cfg in configs.items():
        result = BoxLSEControlDA(clients, cfg).run(test, w_true)
        result["seed"] = int(seed)
        results[name] = result
    return results


def summarize(all_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    metrics = [
        "objective", "test_mse", "train_mse", "parameter_mse",
        "average_objective", "average_test_mse", "average_train_mse",
        "average_parameter_mse", "average_x_norm", "sigma",
        "sensitivity", "bits_per_client", "state_age_mean", "state_age_p90",
        "state_age_max", "prefix_clip_residual", "tail_prefix_clip_residual",
        "mean_movement", "sum_movement", "mean_E_t_norm", "max_E_t_norm",
        "sum_E_t_sq", "mean_c_t_norm", "mean_rho_t", "mean_beta_t_norm",
        "accounted_epsilon",
    ]
    for name, rows in all_results.items():
        out = {"n_seeds": len(rows)}
        for m in metrics:
            vals = np.asarray([r[m] for r in rows], dtype=float)
            out[m] = float(vals.mean())
            out[m + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[name] = out
    return summary


def run_suite(args: argparse.Namespace) -> Dict:
    grad_err = finite_difference_check(args.seed)
    all_results: Dict[str, List[Dict]] = {
        "nonprivate_bounded_EControl_TopK": [],
        "centralDP_bounded_EControl_TopK": [],
        "centralDP_DA_no_compression": [],
    }
    if getattr(args, "dense_C0", None) is not None:
        all_results["centralDP_DA_no_compression_matched_sensitivity"] = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        one = run_one(seed, args)
        for name, result in one.items():
            all_results[name].append(result)
    summary = summarize(all_results)
    for name, s in summary.items():
        print(
            f"{name:40s} objective={s['objective']:.6f}±{s['objective_sd']:.6f} "
            f"test_mse={s['test_mse']:.6f}±{s['test_mse_sd']:.6f} "
            f"param_mse={s['parameter_mse']:.6f} sigma={s['sigma']:.5f} "
            f"sens={s['sensitivity']:.5f} bits/client={s['bits_per_client']:.0f} "
            f"state_age={s['state_age_mean']:.2f} prefix_clip={s['prefix_clip_residual']:.4f} "
            f"mean_move={s['mean_movement']:.5f} max_E={s['max_E_t_norm']:.5f}"
        )
    return {
        "metadata": {
            "task": "Paper3 DP box-constrained least squares",
            "mechanism": "fresh Gaussian aggregate release; real iterates; full participation q=1",
            "iterate_reporting": "last and uniform-average real iterates",
            "n_seeds": int(args.n_seeds),
            "seed_start": int(args.seed),
            "gradient_finite_difference_max_error": float(grad_err),
            "n_clients": int(args.n_clients),
            "samples_per_client": int(args.samples_per_client),
            "d": int(args.d),
            "rounds": int(args.rounds),
            "topk_frac": float(args.topk_frac),
            "epsilon": float(args.epsilon),
            "delta_dp": float(args.delta_dp),
            "C0": float(args.C0), "Cg": float(args.Cg), "Br": float(args.Br),
            "Bh": float(args.Bh), "Be": float(args.Be), "Bbox": float(args.Bbox),
            "gamma": float(args.gamma), "heterogeneity": float(args.heterogeneity),
            "noise_std": float(args.noise_std),
            "accountant": args.accountant,
            "scope_note": "The compression/top-k and box projection diagnostics are empirical; no unconditional real-iterate theorem is claimed.",
        },
        "summary": summary,
        "runs": all_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=20)
    ap.add_argument("--samples-per-client", type=int, default=40)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--topk-frac", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--delta-dp", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--C0", type=float, default=1.5)
    ap.add_argument("--dense-C0", type=float, default=None,
                    help="optional matched-sensitivity C0 for the dense DP baseline")
    ap.add_argument("--Cg", type=float, default=1.5)
    ap.add_argument("--Br", type=float, default=2.0)
    ap.add_argument("--Bh", type=float, default=1.0)
    ap.add_argument("--Be", type=float, default=2.0)
    ap.add_argument("--Bbox", type=float, default=2.0)
    ap.add_argument("--heterogeneity", type=float, default=0.2)
    ap.add_argument("--noise-std", type=float, default=0.15)
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
