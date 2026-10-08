#!/usr/bin/env python3
"""Generate the active-projection sign-convention stress manifest.

The stress test deliberately sets ``private=False`` and therefore has
``sigma=0``.  It is an algebraic diagnostic, not a DP experiment.  Earlier
versions duplicated the same clean run under ``*_DP`` labels; this generator
uses explicit ``*_nonprivate`` names and records the privacy status in the
manifest so those rows cannot be mistaken for private results.

Run from the repository root::

    python repro/generate_sign_stress.py

Use ``P3_CODE_DIR`` when the simulator modules are in a non-standard path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "repro" else SCRIPT_DIR
if os.environ.get("P3_CODE_DIR"):
    code_dir = Path(os.environ["P3_CODE_DIR"]).resolve()
elif (REPO_ROOT / "code" / "p3_bounded_sim.py").is_file():
    code_dir = (REPO_ROOT / "code").resolve()
elif (REPO_ROOT / "p3_bounded_sim.py").is_file():
    code_dir = REPO_ROOT.resolve()
else:
    raise RuntimeError(
        "Could not locate p3_bounded_sim.py; set P3_CODE_DIR to the simulator directory."
    )
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

# The audit module performs the same path resolution and exposes the
# independent trajectory checker.  Importing it does not run its CLI.
from tele_scope_softmax_simplex_audit import audit_one  # noqa: E402
from p3_softmax_sim import Config as SoftmaxConfig  # noqa: E402
from p3_softmax_sim import SoftmaxEControlDA, make_softmax_data  # noqa: E402
from p3_simplex_logistic_sim import Config as SimplexConfig  # noqa: E402
from p3_simplex_logistic_sim import (  # noqa: E402
    SimplexLogisticEControlDA,
    make_simplex_data,
)


def make_softmax(seed: int) -> SoftmaxEControlDA:
    clients, _, _ = make_softmax_data(20, 40, 8, 3, seed=seed)
    cfg = SoftmaxConfig(
        n_clients=20,
        samples_per_client=40,
        d=8,
        n_classes=3,
        rounds=40,
        topk_frac=0.1,
        C0=1.5,
        Cg=0.25,
        Br=0.05,
        Bh=0.10,
        Be=0.05,
        gamma=5.0,
        epsilon=8.0,
        delta_dp=1e-5,
        seed=seed,
        private=False,
        compression=True,
        accountant="without_replacement_bound",
    )
    return SoftmaxEControlDA(clients, cfg)


def make_simplex(seed: int) -> SimplexLogisticEControlDA:
    clients, _, _ = make_simplex_data(20, 40, 8, seed=seed)
    cfg = SimplexConfig(
        n_clients=20,
        samples_per_client=40,
        d=8,
        rounds=40,
        topk_frac=0.2,
        C0=1.5,
        Cg=0.25,
        Br=0.05,
        Bh=0.10,
        Be=0.05,
        gamma=5.0,
        epsilon=8.0,
        delta_dp=1e-5,
        seed=seed,
        private=False,
        compression=True,
        accountant="without_replacement_bound",
    )
    return SimplexLogisticEControlDA(clients, cfg)


def generate(seeds: tuple[int, ...] = (0,), output: Path | None = None) -> dict:
    rows = {}
    for name, maker in (
        ("softmax_nonprivate", make_softmax),
        ("simplex_nonprivate", make_simplex),
    ):
        rows[name] = [audit_one(maker(seed)) for seed in seeds]
    payload = {
        "metadata": {
            "purpose": "active-projection sign-convention stress check",
            "privacy_status": "nonprivate algebraic diagnostic",
            "private": False,
            "sigma": 0.0,
            # Retain this explicit field for compatibility with earlier
            # manifests, but never label these rows as DP.
            "sigma_override": 0.0,
            "rounds": 40,
            "C0": 1.5,
            "Cg": 0.25,
            "Bh": 0.10,
            "Be": 0.05,
            "Br": 0.05,
            "seeds": list(seeds),
            "note": (
                "Both rows use private=False. sigma_override=0 records the zero-noise "
                "diagnostic; it is not a DP result."
            ),
        },
        "results": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "tele_scope_sign_stress.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    generate(tuple(args.seeds), args.output)


if __name__ == "__main__":
    main()
