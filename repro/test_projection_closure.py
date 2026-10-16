#!/usr/bin/env python3
"""Regression checks for the projected-EControl closure note and exact verifier."""
from fractions import Fraction
import math
import unittest

import numpy as np

from projection_closure_probe import Interval, certify_prefix, float_trace, step, sufficient_bound


class IntegerIntervalTests(unittest.TestCase):
    def contains(self, interval, value):
        self.assertLessEqual(Fraction(interval.lo, Interval.SCALE), value)
        self.assertGreaterEqual(Fraction(interval.hi, Interval.SCALE), value)

    def test_signed_rational_arithmetic(self):
        for a in (Fraction(-17, 13), Fraction(-1, 3), Fraction(0), Fraction(4, 7)):
            for b in (Fraction(-7, 11), Fraction(2, 3), Fraction(5, 2)):
                x = Interval.rational(a.numerator, a.denominator)
                y = Interval.rational(b.numerator, b.denominator)
                self.contains(x+y, a+b)
                self.contains(x-y, a-b)
                self.contains(x*y, a*b)
                self.contains(x.square(), a*a)
                if b > 0:
                    self.contains(x/y, a/b)

    def test_sqrt_outward_rounding(self):
        for n, d in ((0, 1), (1, 1), (2, 1), (961, 1000), (7, 13)):
            x = Interval.rational(n, d).sqrt()
            target = Fraction(n, d)
            self.assertLessEqual(Fraction(x.lo, Interval.SCALE)**2, target)
            self.assertGreaterEqual(Fraction(x.hi, Interval.SCALE)**2, target)

    def test_exact_prefix_two_precisions(self):
        a, b = certify_prefix(4096), certify_prefix(4608)
        self.assertTrue(a['prefix_entry_certified'] and b['prefix_entry_certified'])
        self.assertEqual(a['branch_sha256'], b['branch_sha256'])
        self.assertEqual(a['S_lower_numerator'], b['S_lower_numerator'])
        self.assertEqual(a['S_upper_numerator'], b['S_upper_numerator'])
        self.assertGreater(a['S_lower_numerator'], 600*a['S_denominator'])


class ProjectionDynamicsTests(unittest.TestCase):
    def test_sector_fixed_ray_witness(self):
        u = np.zeros(10); u[0] = 1.
        e = -3*u
        for eta in (.1, .05, .018):
            h_next, e_next, _ = step(u, e, u, eta, 1)
            np.testing.assert_allclose(h_next, u, atol=1e-14, rtol=0)
            np.testing.assert_allclose(e_next, e, atol=1e-14, rtol=0)

    def test_linear_growth_region(self):
        rng = np.random.default_rng(17)
        for _ in range(200):
            h = rng.normal(size=10); h *= rng.uniform(0, 1)/np.linalg.norm(h)
            e = rng.normal(size=10)*100
            S = rng.uniform(600, 2000)
            e -= (e.sum()+S)/10
            hn, en, _ = step(h, e, np.full(10, .31), .1, 1)
            self.assertLessEqual(np.linalg.norm(hn), 1+1e-14)
            self.assertGreaterEqual(float(-en.sum()), S+1.35-1e-10)

    def test_positive_lemma_drift(self):
        rng = np.random.default_rng(23)
        for eta in (.1, .05, .018):
            b = sufficient_bound(.25, 1., .1, eta)
            for _ in range(100):
                h = rng.normal(size=10); h *= rng.uniform(0, 1)/np.linalg.norm(h)
                u = rng.normal(size=10); u *= rng.uniform(0, .25)/np.linalg.norm(u)
                e = rng.normal(size=10); e *= rng.uniform(b['R'], 2*b['R'])/np.linalg.norm(e)
                _, en, _ = step(h, e, u, eta, 1)
                rhs = -2*b['mu']*np.linalg.norm(e)+b['C']
                self.assertLessEqual(float(en@en-e@e), rhs+1e-7)

    def test_dense_control(self):
        row = float_trace(.31, .1, k=10, T=1000)
        self.assertLessEqual(row['max_e_norm'], 1e-12)

    def test_bound_not_silently_extended(self):
        self.assertFalse(sufficient_bound(.98, 1., .1, .1)['condition_satisfied'])
        with self.assertRaises(ValueError):
            sufficient_bound(.25, 1., .1, 0.)


if __name__ == '__main__':
    unittest.main()
