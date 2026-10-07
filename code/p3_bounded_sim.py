"""Minimal convex-composite simulator for bounded private EControl-DA.

The simulator is intentionally small and auditable.  It uses synthetic
binary logistic regression with an l1 composite term and compares:

  * non-private bounded EControl + Top-K;
  * central client-level DP bounded EControl + Top-K;
  * central client-level DP Dual Averaging without compression.

The curator releases the current clean aggregate estimate plus fresh Gaussian
noise.  It never integrates noisy releases into the next released estimate.
This is a research diagnostic, not a production DP training library.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def clip_vec(v: np.ndarray, radius: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= radius or n == 0.0:
        return v.copy()
    return v * (radius / n)


def clip_rows(v: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return v * scale


def topk(v: np.ndarray, k: int) -> np.ndarray:
    if k >= v.size:
        return v.copy()
    out = np.zeros_like(v)
    # Stable tie handling makes runs reproducible.  Values are clipped before
    # this call, so the message norm is deterministically bounded.
    idx = np.argsort(-np.abs(v), kind="stable")[:k]
    out[idx] = v[idx]
    return out


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def logistic_loss_and_grad(X: np.ndarray, y: np.ndarray, x: np.ndarray):
    margin = y * (X @ x)
    # Stable log(1 + exp(-margin)).
    loss = np.logaddexp(0.0, -margin).mean()
    probs = -y / (1.0 + np.exp(np.clip(margin, -60.0, 60.0)))
    per_example = probs[:, None] * X
    return float(loss), per_example


def make_data(n_clients: int, samples_per_client: int, d: int,
              seed: int = 0, heterogeneity: float = 0.2):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= max(np.linalg.norm(w), 1e-12)
    clients = []
    for i in range(n_clients):
        shift = heterogeneity * rng.normal(size=d) / np.sqrt(d)
        X = rng.normal(size=(samples_per_client, d)) + shift
        y = np.where(X @ w + 0.25 * rng.normal(size=samples_per_client) >= 0, 1.0, -1.0)
        clients.append((X, y))
    Xtest = rng.normal(size=(max(2000, n_clients * samples_per_client), d))
    ytest = np.where(Xtest @ w + 0.25 * rng.normal(size=Xtest.shape[0]) >= 0, 1.0, -1.0)
    return clients, (Xtest, ytest), w


def make_digits_data(n_clients: int, samples_per_client: int, seed: int = 0):
    """Small image-feature proxy available without a network download.

    The sklearn digits data are converted to a balanced binary task: digits
    0--4 versus 5--9.  The 64 standardized pixels are split across clients;
    this is a smoke test for image-shaped features before moving to
    FashionMNIST.
    """
    from sklearn.datasets import load_digits

    data = load_digits()
    X = data.data.astype(float)
    y = np.where(data.target < 5, 1.0, -1.0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_train = min(len(X) - 300, n_clients * samples_per_client)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if len(test_idx) == 0:
        test_idx = perm[-300:]
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std
    train_X, train_y = X[train_idx], y[train_idx]
    clients = []
    for i in range(n_clients):
        lo = i * samples_per_client
        hi = lo + samples_per_client
        clients.append((train_X[lo:hi], train_y[lo:hi]))
    return clients, (X[test_idx], y[test_idx]), np.zeros(X.shape[1])


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def _logaddexp_list(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def rdp_without_replacement_bound(orders: np.ndarray, sampling_rate: float,
                                  sensitivity: float, sigma: float) -> np.ndarray:
    """Integer-order upper bound for uniform fixed-size sampling.

    This is the Gaussian specialization of the without-replacement RDP bound
    of Wang, Balle, and Kasiviswanathan (Theorem 9, 2019).  It is deliberately
    implemented as a bound rather than as an exact accountant; the theorem's
    finite-order expression is conservative for some parameter regimes.
    """
    q = float(sampling_rate)
    if q >= 1.0:
        return orders * sensitivity**2 / (2.0 * sigma**2)
    rho2 = sensitivity**2 / sigma**2
    first_min = min(4.0 * np.expm1(min(rho2, 700.0)),
                    2.0 * np.exp(min(rho2, 700.0)))
    out = []
    for alpha_f in orders:
        alpha = int(round(alpha_f))
        if alpha < 2:
            out.append(0.0)
            continue
        logs = [0.0]
        logs.append(
            2.0 * math.log(q)
            + _log_comb(alpha, 2)
            + math.log(max(first_min, 1e-300))
        )
        for j in range(3, alpha + 1):
            # Gaussian RDP has no finite infinity-order term, so the
            # min{2, (exp(rho_inf)-1)^j} factor is 2.
            logs.append(
                j * math.log(q)
                + _log_comb(alpha, j)
                + (j - 1.0) * (j * sensitivity**2 / (2.0 * sigma**2))
                + math.log(2.0)
            )
        bound = _logaddexp_list(logs) / (alpha - 1.0)
        # Subsampling cannot make the bound worse than releasing the same
        # Gaussian mechanism without sampling.
        # Do not index ``rho`` by alpha: callers may pass a non-consecutive
        # integer-order grid. The unsampled Gaussian RDP bound is available
        # directly as alpha*S^2/(2*sigma^2).
        no_sampling = alpha * sensitivity**2 / (2.0 * sigma**2)
        out.append(min(float(no_sampling), bound))
    return np.asarray(out)


def rdp_per_round(orders: np.ndarray, sampling_rate: float,
                  sensitivity: float, sigma: float,
                  mode: str = "without_replacement_bound") -> np.ndarray:
    """Integer-order RDP for the simulator's sampling modes."""
    if sampling_rate >= 1.0:
        return orders * sensitivity**2 / (2.0 * sigma**2)
    if mode == "without_replacement_bound":
        return rdp_without_replacement_bound(orders, sampling_rate, sensitivity, sigma)
    if mode == "conservative":
        return orders * sensitivity**2 / (2.0 * sigma**2)
    q = float(sampling_rate)
    mu2 = (sensitivity / sigma) ** 2
    out = []
    for alpha_f in orders:
        alpha = int(round(alpha_f))
        logs = []
        for k in range(alpha + 1):
            if q == 0.0 and k > 0:
                continue
            if q == 1.0 and k < alpha:
                continue
            if (1.0 - q) == 0.0 and k < alpha:
                continue
            log_weight = _log_comb(alpha, k)
            if k > 0:
                log_weight += k * math.log(q)
            if alpha - k > 0:
                log_weight += (alpha - k) * math.log1p(-q)
            # Exact integer-order term for the Poisson-subsampled Gaussian
            # mechanism under the normalized sensitivity convention.
            logs.append(log_weight + (k * (k - 1) / 2.0) * mu2)
        m = max(logs)
        out.append((m + math.log(sum(math.exp(v - m) for v in logs))) / (alpha - 1.0))
    return np.asarray(out)


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float,
                   rounds: int, sampling_rate: float = 1.0,
                   accountant_mode: str = "without_replacement_bound") -> float:
    """Choose sigma by an integer-order RDP accountant."""
    if epsilon <= 0 or sensitivity <= 0 or rounds <= 0:
        return 0.0
    orders = np.arange(2.0, 257.0)
    log_delta = np.log(1.0 / delta)

    def eps_for_sigma(sigma):
        rdp = rounds * rdp_per_round(
            orders, sampling_rate, sensitivity, sigma, mode=accountant_mode
        )
        return float(np.min(rdp + log_delta / (orders - 1.0)))

    lo, hi = 1e-10, max(sensitivity, 1e-6)
    while eps_for_sigma(hi) > epsilon:
        hi *= 2.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if eps_for_sigma(mid) <= epsilon:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class Config:
    n_clients: int = 20
    samples_per_client: int = 40
    d: int = 20
    rounds: int = 200
    topk_frac: float = 0.1
    eta: Optional[float] = None
    C0: float = 1.5
    Cg: float = 1.5
    Br: float = 2.0
    Bh: float = 5.0
    Be: float = 5.0
    l1: float = 0.002
    gamma: float = 5.0
    epsilon: float = 8.0
    delta_dp: float = 1e-5
    seed: int = 0
    private: bool = True
    compression: bool = True
    residual_buffer: bool = True
    participation_rate: float = 1.0
    accountant: str = "without_replacement_bound"
    release_mode: str = "gradient"  # gradient, or bounded prefix for FTRL


class BoundedPrivateEControlDA:
    def __init__(self, clients, config: Config):
        self.clients = clients
        self.cfg = config
        self.n = len(clients)
        self.d = clients[0][0].shape[1]
        self.k = max(1, int(round(config.topk_frac * self.d)))
        self.eta = config.eta if config.eta is not None else config.topk_frac
        self.rng = np.random.default_rng(config.seed + 19)
        self.m = self.n if config.participation_rate >= 1.0 else max(
            1, int(round(config.participation_rate * self.n))
        )
        self.h = np.zeros((self.n, self.d))
        self.e = np.zeros((self.n, self.d))
        self.r = np.zeros((self.n, self.d))
        self.x = np.zeros(self.d)
        self.G = np.zeros(self.d)
        self.prefix = np.zeros(self.d)
        self.prefix_clip_residual = 0.0
        self.A = 0.0
        self.prev_h_clean = np.zeros(self.d)
        # One-client replacement changes one bounded *local* state in an
        # active average. This gives 2 Bh / m for EControl and 2 C0 / m for
        # the dense gradient reference. The prefix reference is different:
        # ``self.prefix`` is one persistent curator-side state, not an
        # average of per-client states. Even though its input increment is
        # an active average, clipping the accumulated prefix to Bh only
        # bounds the released value in a ball of radius Bh; two neighboring
        # runs can therefore end at opposite points of that ball. Its
        # deterministic sensitivity is 2 Bh, with no additional 1/m factor.
        if config.release_mode == "prefix":
            release_radius = config.Bh
            sensitivity = 2.0 * release_radius
        else:
            release_radius = config.Bh if config.compression else config.C0
            sensitivity = 2.0 * release_radius / self.m
        sampling_rate = self.m / self.n
        # q=1 is exact. For the per-round active-average releases, q<1 can
        # use Wang--Balle--Kasiviswanathan's fixed-size
        # without-replacement RDP upper bound. The prefix reference is
        # different: once a client has affected the persistent curator-side
        # prefix, that difference remains visible even on rounds where the
        # client is inactive. Applying per-round subsampling amplification to
        # this stateful release would therefore be unsound, so account it
        # without amplification (q_accountant=1). The Poisson mode is kept
        # for comparison; conservative also disables amplification.
        accountant_q = (
            1.0
            if config.accountant == "conservative" or config.release_mode == "prefix"
            else sampling_rate
        )
        self.sigma = (
            gaussian_sigma(
                config.epsilon, config.delta_dp, sensitivity, config.rounds,
                sampling_rate=accountant_q,
                accountant_mode=config.accountant,
            )
            if config.private else 0.0
        )
        self.sensitivity = sensitivity
        self.accountant_q = accountant_q
        self.orders = np.arange(2.0, 257.0)
        self.history = []
        self.last_active = np.full(self.n, -1, dtype=int)
        self.t = 0

    def accounted_epsilon(self) -> float:
        if not self.cfg.private:
            return 0.0
        rdp = self.cfg.rounds * rdp_per_round(
            self.orders, self.accountant_q, self.sensitivity, self.sigma,
            mode=self.cfg.accountant,
        )
        return float(np.min(rdp + np.log(1.0 / self.cfg.delta_dp) / (self.orders - 1.0)))

    def _prox_da(self):
        return soft_threshold(
            -self.G / max(self.cfg.gamma, 1e-12),
            self.A * self.cfg.l1 / max(self.cfg.gamma, 1e-12),
        )

    def _clipped_gradient(self, i: int):
        X, y = self.clients[i]
        _, per = logistic_loss_and_grad(X, y, self.x)
        v = clip_rows(per, self.cfg.C0).mean(axis=0)
        return v, per, float(np.mean(np.linalg.norm(per, axis=1) > self.cfg.C0))

    def _local_update(self, i: int):
        v, per, clipping_fraction = self._clipped_gradient(i)
        rbar = clip_vec(self.r[i], self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.d)
        u = clip_vec(v + rbar, self.cfg.Cg)
        hbar = clip_vec(self.h[i], self.cfg.Bh)
        ebar = clip_vec(self.e[i], self.cfg.Be)
        delta = u - hbar - self.eta * ebar
        Cdelta = self.cfg.Cg + self.cfg.Bh + self.eta * self.cfg.Be
        delta = clip_vec(delta, Cdelta)
        q = topk(delta, self.k)
        h_raw = self.h[i] + q
        h_new = clip_vec(h_raw, self.cfg.Bh)
        e_raw = ebar + h_new - u
        e_new = clip_vec(e_raw, self.cfg.Be)
        r_raw = v + rbar - u
        r_new = clip_vec(r_raw, self.cfg.Br) if self.cfg.residual_buffer else np.zeros(self.d)
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
        if self.m == self.n:
            active = np.arange(self.n)
        else:
            active = self.rng.choice(self.n, size=self.m, replace=False)
        self.last_active[active] = self.t
        if self.compression:
            local_stats = [self._local_update(int(i)) for i in active]
        else:
            # The dense reference is a separate central-DP DA path; it does
            # not update EControl states that it never releases.
            local_stats = []
            for i in active:
                _, _, clipping_fraction = self._clipped_gradient(int(i))
                local_stats.append({
                    "clipping_fraction": clipping_fraction,
                    "message_norm": 0.0,
                    "h_clip_residual": 0.0,
                    "e_clip_residual": 0.0,
                    "r_norm": 0.0,
                    "e_norm": 0.0,
                })
        if self.compression:
            h_clean = self.h[active].mean(axis=0)
            z = self.rng.normal(scale=self.sigma, size=self.d)
            released = h_clean + z
            self.G += released
        else:
            # A no-compression DP-DA reference uses the current average clipped
            # stochastic gradient rather than a persistent EControl estimate.
            gs = []
            for i in active:
                gs.append(self._clipped_gradient(int(i))[0])
            h_clean = np.mean(gs, axis=0)
            if self.cfg.release_mode == "prefix":
                prefix_raw = self.prefix + h_clean
                self.prefix = clip_vec(prefix_raw, self.cfg.Bh)
                self.prefix_clip_residual = float(np.linalg.norm(prefix_raw - self.prefix))
                z = self.rng.normal(scale=self.sigma, size=self.d)
                released = self.prefix + z
                # FTRL uses the released prefix directly.  It does not add a
                # fresh noisy prefix to a second cumulative accumulator.
                self.G = released.copy()
            else:
                z = self.rng.normal(scale=self.sigma, size=self.d)
                released = h_clean + z
                self.G += released
        self.A += 1.0
        self.x = self._prox_da()
        self.prev_h_clean = h_clean
        age = np.where(self.last_active >= 0, self.t - self.last_active, self.t + 1)
        self.history.append({
            "x": self.x.copy(),
            "h_clean_norm": float(np.linalg.norm(h_clean)),
            "noise_norm": float(np.linalg.norm(z)),
            "released_norm": float(np.linalg.norm(released)),
            "clipping_fraction": float(np.mean([s["clipping_fraction"] for s in local_stats])),
            "message_norm": float(np.mean([s["message_norm"] for s in local_stats])),
            "h_clip_residual": float(np.mean([s["h_clip_residual"] for s in local_stats])),
            "e_clip_residual": float(np.mean([s["e_clip_residual"] for s in local_stats])),
            "r_norm": float(np.mean([s["r_norm"] for s in local_stats])),
            "e_norm": float(np.mean([s["e_norm"] for s in local_stats])),
            "active_count": int(self.m),
            "mean_state_age": float(np.mean(age)),
            "p90_state_age": float(np.percentile(age, 90)),
            "max_state_age": int(np.max(age)),
            "prefix_clip_residual": self.prefix_clip_residual,
        })
        self.t += 1

    @property
    def compression(self):
        return self.cfg.compression

    def run(self, test):
        for _ in range(self.cfg.rounds):
            self.step()
        Xtest, ytest = test
        pred = np.where(Xtest @ self.x >= 0, 1.0, -1.0)
        acc = float(np.mean(pred == ytest))
        loss = float(logistic_loss_and_grad(Xtest, ytest, self.x)[0] + self.cfg.l1 * np.abs(self.x).sum())
        return {
            "objective": loss,
            "accuracy": acc,
            "x_norm": float(np.linalg.norm(self.x)),
            "sigma": self.sigma,
            "sensitivity": self.sensitivity,
            "epsilon": self.cfg.epsilon if self.cfg.private else 0.0,
            "delta": self.cfg.delta_dp if self.cfg.private else 0.0,
            "accounted_epsilon": self.accounted_epsilon(),
            "active_count": self.m,
            "bits_total": self.cfg.rounds * self.m
            * (self.k if self.compression else self.d)
            * (32 + int(np.ceil(np.log2(self.d)))),
            "bits_per_client": self.cfg.rounds * self.m / self.n
            * (self.k if self.compression else self.d)
            * (32 + int(np.ceil(np.log2(self.d)))),
            "sampling_rate": self.m / self.n,
            "accountant_q": self.accountant_q,
            "history": self.history,
        }


def run_suite(args):
    if args.dataset == "digits":
        clients, test, _ = make_digits_data(
            args.n_clients, args.samples_per_client, args.seed
        )
        actual_d = clients[0][0].shape[1]
    else:
        clients, test, _ = make_data(
            args.n_clients, args.samples_per_client, args.d, args.seed
        )
        actual_d = args.d
    common = dict(
        n_clients=args.n_clients, samples_per_client=args.samples_per_client,
        d=actual_d, rounds=args.rounds, topk_frac=args.topk_frac,
        epsilon=args.epsilon, delta_dp=args.delta_dp, seed=args.seed,
        gamma=args.gamma, l1=args.l1, C0=args.C0, Cg=args.Cg,
        Br=args.Br, Bh=args.Bh, Be=args.Be,
        participation_rate=args.participation_rate,
        accountant=args.accountant,
    )
    configs = {
        "nonprivate_bounded_EControl_TopK": Config(**common, private=False, compression=True),
        "centralDP_bounded_EControl_TopK": Config(**common, private=True, compression=True),
        "centralDP_DA_no_compression": Config(**common, private=True, compression=False),
        "centralDP_bounded_prefix_FTRL": Config(
            **common, private=True, compression=False, release_mode="prefix"
        ),
    }
    results = {}
    for name, cfg in configs.items():
        result = BoundedPrivateEControlDA(clients, cfg).run(test)
        results[name] = result
        tail = result["history"][-1]
        print(
            f"{name:40s} objective={result['objective']:.5f} "
            f"acc={result['accuracy']:.3f} sigma={result['sigma']:.5f} "
            f"sens={result['sensitivity']:.5f} q={result['sampling_rate']:.2f} "
            f"qacc={result['accountant_q']:.2f} "
            f"eps={result['accounted_epsilon']:.4f} "
            f"bits/client={result['bits_per_client']:.0f} "
            f"hclip={tail['h_clip_residual']:.4f} "
            f"pclip={tail['prefix_clip_residual']:.4f} "
            f"age={tail['mean_state_age']:.2f} r={tail['r_norm']:.4f}"
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "digits"], default="synthetic")
    ap.add_argument("--n-clients", type=int, default=20)
    ap.add_argument("--samples-per-client", type=int, default=40)
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--topk-frac", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--delta-dp", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--l1", type=float, default=0.002)
    ap.add_argument("--C0", type=float, default=1.5)
    ap.add_argument("--Cg", type=float, default=1.5)
    ap.add_argument("--Br", type=float, default=2.0)
    ap.add_argument("--Bh", type=float, default=5.0)
    ap.add_argument("--Be", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--participation-rate", type=float, default=1.0)
    ap.add_argument(
        "--accountant",
        choices=["without_replacement_bound", "poisson_surrogate", "conservative"],
        default="without_replacement_bound",
        help="Fixed-size RDP bound, Poisson surrogate, or conservative no-amplification.",
    )
    args = ap.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
