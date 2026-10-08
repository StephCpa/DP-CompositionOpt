#!/usr/bin/env python3
"""Small regression checks for the explicit horizon gamma calibration."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.run_reproducibility import _calibrated_gamma  # noqa: E402


class CalibrationTests(unittest.TestCase):
    def test_default_two_term_coefficient_and_box_l2_scale(self) -> None:
        # Bbox=2, d=10, sigma=.693..., T=25.  The coefficient-2 rule is
        # below the explicit floor, so this checks the resolved minimum.
        value = _calibrated_gamma(
            sigma=0.6931788020546441,
            dimension=10,
            rounds=25,
            reference_radius=2.0 * math.sqrt(10.0),
        )
        self.assertEqual(value, 5.0)

    def test_previous_driver_compatibility(self) -> None:
        value = _calibrated_gamma(
            sigma=0.6931788020546441,
            dimension=10,
            rounds=25,
            reference_radius=2.0,
            noise_coefficient=1.0,
        )
        self.assertAlmostEqual(value, 5.4800596005992, places=12)

    def test_invalid_scales_are_rejected(self) -> None:
        kwargs = dict(
            sigma=1.0,
            dimension=10,
            rounds=10,
            reference_radius=1.0,
        )
        with self.assertRaises(ValueError):
            _calibrated_gamma(**kwargs, noise_coefficient=0.0)
        with self.assertRaises(ValueError):
            _calibrated_gamma(**{**kwargs, "reference_radius": 0.0})


if __name__ == "__main__":
    unittest.main()
