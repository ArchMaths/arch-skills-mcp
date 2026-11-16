"""Numerical contracts for hairjoin; no generator, Arch, or hair.py imports.

Run from any directory with::

    python path/to/test_hairjoin.py -v

Only the standard library and NumPy are required.  Endpoint derivatives use
local fifth-order polynomial extrapolation on a densely sampled bridge, not a
first finite difference (which would confuse curvature with a discontinuity).
"""
import unittest

import numpy as np
from numpy.polynomial import polynomial as poly

from hairjoin import attach_root, fit_c1, legacy_end_jets


def lock_samples(count=81, bottom=9.2):
    """A curved, varying-width six-column lock, quadratic in downward height."""
    z = np.linspace(11.9, bottom, count)
    s = 11.9 - z
    return np.column_stack((
        0.3 + 0.12*s + 0.04*s*s,
        0.7 + 0.10*s - 0.015*s*s,
        z,
        0.08 + 0.018*s + 0.003*s*s,
        0.025 + 0.005*s + 0.001*s*s,
        0.18 - 0.09*s + 0.025*s*s,
    ))


def lock_derivative(s):
    """Exact tail derivative with respect to downward height, not sample index."""
    return np.array((0.12 + 0.08*s, 0.10 - 0.03*s, -1.0,
                     0.018 + 0.006*s, 0.005 + 0.002*s,
                     -0.09 + 0.05*s))


def bridge_and_derivatives(joined, original, end, anchor):
    # The unchanged tail includes the last bridge point, so count that once.
    count = len(joined) - len(original[end:]) + 1
    bridge = joined[:count]
    height = anchor[2] - original[end, 2]
    u = np.linspace(0.0, 1.0, count)
    left = poly.polyfit(u[:8], bridge[:8], 5)[1] / height
    right = poly.polyfit(u[-8:] - 1.0, bridge[-8:], 5)[1] / height
    return bridge, left, right


def curved_values(s):
    """Non-polynomial, six-dimensional curve, and its exact derivative."""
    s = np.asarray(s)
    values = np.column_stack((
        0.3 + 0.2*s + 0.08*np.sin(2.4*s),
        -0.4 + 0.07*s*s + 0.06*np.cos(3.1*s + 0.2),
        12.4 - 2.7*s + 0.04*np.sin(1.7*s),
        0.12 + 0.025*np.cos(2.2*s + 0.1) + 0.01*s*s,
        0.035 + 0.007*np.sin(1.3*s + 0.4),
        -0.2 + 0.35*s + 0.14*np.sin(2.8*s),
    ))
    derivatives = np.column_stack((
        0.2 + 0.192*np.cos(2.4*s),
        0.14*s - 0.186*np.sin(3.1*s + 0.2),
        -2.7 + 0.068*np.cos(1.7*s),
        -0.055*np.sin(2.2*s + 0.1) + 0.02*s,
        0.0091*np.cos(1.3*s + 0.4),
        0.35 + 0.392*np.cos(2.8*s),
    ))
    return values, derivatives


def endpoint_jets(coefficients):
    """Evaluate the returned (already rounded) power-basis coefficients."""
    positions = np.column_stack([
        poly.polyval([-1.0, 1.0], c) for c in coefficients
    ])
    derivatives = np.column_stack([
        poly.polyval([-1.0, 1.0], poly.polyder(c)) for c in coefficients
    ])
    return positions, derivatives


def rounding_bounds(degree):
    # Each power coefficient changes by <= 0.5e-5 on five-decimal rounding.
    # At |t|=1, position accumulates sum(1), derivative accumulates sum(k).
    return ((degree + 1)*0.5e-5 + 1e-9,
            degree*(degree + 1)*0.25e-5 + 1e-9)


class AttachRootTests(unittest.TestCase):
    def setUp(self):
        # Dense samples make fifth-order endpoint extrapolation accurate even
        # when positive width/thickness easing is not itself a polynomial.
        self.points = lock_samples(count=801)
        self.anchor = np.array((0.15, 0.73, 12.55))
        self.tangent = np.array((0.22, -0.13, -0.8))

    def test_anchor_and_root_tangent_and_zero_size_slopes(self):
        for end_height in (11.4, 10.95, 10.2):
            with self.subTest(end_height=end_height):
                joined, end, tangent = attach_root(
                    self.points, self.anchor, self.tangent, end_height)
                bridge, left, _ = bridge_and_derivatives(
                    joined, self.points, end, self.anchor)
                np.testing.assert_array_equal(joined[0, :3], self.anchor)
                np.testing.assert_array_equal(joined[0, 3:], np.zeros(3))
                expected = self.tangent / -self.tangent[2]
                np.testing.assert_allclose(tangent, expected, atol=1e-13, rtol=0)
                np.testing.assert_allclose(left[:3], expected, atol=2e-9, rtol=0)
                np.testing.assert_allclose(left[3:5], 0.0, atol=2e-9, rtol=0)
                # The bridge's parameter is downward height, not arc length.
                np.testing.assert_allclose(
                    bridge[:, 2], np.linspace(self.anchor[2],
                                              self.points[end, 2], len(bridge)),
                    atol=1e-13, rtol=0)

    def test_bridge_end_is_c1_for_xyz_width_thickness_and_twist(self):
        for end_height in (11.4, 10.95, 10.2):
            with self.subTest(end_height=end_height):
                joined, end, _ = attach_root(
                    self.points, self.anchor, self.tangent, end_height)
                bridge, _, right = bridge_and_derivatives(
                    joined, self.points, end, self.anchor)
                np.testing.assert_array_equal(bridge[-1], self.points[end])
                expected = lock_derivative(11.9 - self.points[end, 2])
                np.testing.assert_allclose(right, expected, atol=2e-8, rtol=0)

    def test_tail_is_pointwise_identical_and_inputs_are_not_mutated(self):
        points_before = self.points.copy()
        anchor_before = self.anchor.copy()
        tangent_before = self.tangent.copy()
        for end_height in (11.4, 10.95, 10.2):
            with self.subTest(end_height=end_height):
                joined, end, _ = attach_root(
                    self.points, self.anchor, self.tangent, end_height)
                np.testing.assert_array_equal(
                    joined[-len(self.points[end:]):], points_before[end:])
                np.testing.assert_array_equal(self.points, points_before)
                np.testing.assert_array_equal(self.anchor, anchor_before)
                np.testing.assert_array_equal(self.tangent, tangent_before)
                self.assertTrue(np.isfinite(joined).all())

    def test_smooth_positive_width_and_thickness_stay_nonnegative(self):
        for bottom in (9.2, 11.0):
            with self.subTest(bottom=bottom):
                points = lock_samples(bottom=bottom)
                joined, end, _ = attach_root(points, self.anchor, self.tangent)
                self.assertGreaterEqual(joined[:, 3:5].min(), 0.0)
                # In particular, a short lock still has an unchanged tail.
                self.assertGreaterEqual(len(points) - end, 2)
                np.testing.assert_array_equal(
                    joined[-len(points[end:]):], points[end:])

    def test_steep_positive_width_and_thickness_stay_nonnegative(self):
        # A bounded, smooth positive size profile, not negative input data.
        # Its steep rising slope exercises overshoot in the root easing.
        # For the positive easing w=q*u**3*exp(g), g=k*(u-1)*u**p,
        # p>=1: w~q*u**3 at 0, while w'(1)=q*(3+k)=db.  The derivative
        # tests above verify those jets numerically without reusing the formula.
        points = lock_samples()
        end = np.flatnonzero(points[:, 2] <= 10.95)[0]
        s = 11.9 - points[:, 2]
        midpoint = s[end] + 0.2
        growth = 1.0 / (1.0 + np.exp(-12.0*(s - midpoint)))
        points[:, 3] = 0.12*growth
        points[:, 4] = 0.035*growth
        self.assertTrue((points[:, 3:5] > 0.0).all())
        joined, _, _ = attach_root(points, self.anchor, self.tangent)
        for column, name in ((3, "width"), (4, "thickness")):
            with self.subTest(quantity=name):
                self.assertGreaterEqual(
                    joined[:, column].min(), 0.0,
                    f"positive input {name} became negative in the bridge")

    def test_non_downward_tangent_is_rejected(self):
        for z in (0.0, 0.2, -1e-6):
            with self.subTest(z=z):
                with self.assertRaisesRegex(ValueError, "downwards"):
                    attach_root(self.points, self.anchor, (0.1, 0.2, z))

    def test_anchor_below_bridge_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "above"):
            attach_root(self.points, (0.0, 0.0, 10.0), self.tangent)


class FitC1Tests(unittest.TestCase):
    def test_quadratic_endpoint_jets_are_exact_even_at_low_degree(self):
        t = np.linspace(-1.0, 1.0, 101)
        coefficients = np.array(((0.12345, 12.5, 0.04),
                                 (0.23456, -1.7, 0.01234),
                                 (0.34567, 0.035, 0.00321)))
        values = poly.polyval(t, coefficients).T
        expected_pos = poly.polyval(np.array([-1.0, 1.0]), coefficients).T
        expected_der = poly.polyval(
            np.array([-1.0, 1.0]), poly.polyder(coefficients)).T
        for degree in (3, 5, 8):
            with self.subTest(degree=degree):
                cs, fitted, error = fit_c1(t, values, degree)
                pos, der = endpoint_jets(cs)
                np.testing.assert_allclose(pos, expected_pos, atol=2e-10, rtol=0)
                np.testing.assert_allclose(der, expected_der, atol=2e-10, rtol=0)
                np.testing.assert_allclose(fitted, values, atol=2e-10, rtol=0)
                self.assertLess(error, 2e-10)

    def test_rounded_nontrivial_curve_endpoint_errors_across_segmentations(self):
        partitions = ((0.0, 1.0),
                      (0.0, 0.5, 1.0),
                      (0.0, 0.12, 0.43, 0.7, 1.0))
        t = np.linspace(-1.0, 1.0, 513)
        for degree in (3, 5, 8):
            pos_bound, der_bound = rounding_bounds(degree)
            # Independently known analytic derivatives also include the small
            # O(dt**2) bias of a quadratic endpoint derivative estimate.
            analytic_der_bound = der_bound + 2e-5
            for edges in partitions:
                previous = None
                for a, b in zip(edges[:-1], edges[1:]):
                    with self.subTest(degree=degree, edges=edges, segment=(a, b)):
                        s = a + (t + 1.0)*(b - a)/2.0
                        values, exact_global_der = curved_values(s)
                        cs, fitted, error = fit_c1(t, values, degree)
                        self.assertEqual(fitted.shape, values.shape)
                        self.assertTrue(np.isfinite(fitted).all())
                        for c in cs:
                            np.testing.assert_array_equal(c, np.round(c, 5))
                        recomputed = np.column_stack([poly.polyval(t, c) for c in cs])
                        np.testing.assert_array_equal(fitted, recomputed)
                        self.assertAlmostEqual(
                            error, np.linalg.norm(fitted - values, axis=1).max(),
                            places=13)
                        pos, local_der = endpoint_jets(cs)
                        np.testing.assert_allclose(
                            pos, values[[0, -1]], atol=pos_bound, rtol=0,
                            err_msg="endpoint position error after five-place rounding")
                        # Check both the estimated jets constrained by fit_c1
                        # and the independent analytic jets of this curve.
                        estimated = np.array([
                            poly.polyfit(t[:7] - t[0], values[:7], 2)[1],
                            poly.polyfit(t[-7:] - t[-1], values[-7:], 2)[1],
                        ])
                        np.testing.assert_allclose(
                            local_der, estimated, atol=der_bound, rtol=0,
                            err_msg="constrained derivative error after rounding")
                        exact_local_der = exact_global_der[[0, -1]]*(b - a)/2.0
                        np.testing.assert_allclose(
                            local_der, exact_local_der, atol=analytic_der_bound, rtol=0,
                            err_msg="analytic derivative error after rounding")
                        global_der = local_der*2.0/(b - a)
                        global_bound = analytic_der_bound*2.0/(b - a)
                        if previous is not None:
                            old_pos, old_der, old_bound = previous
                            np.testing.assert_allclose(
                                old_pos, pos[0], atol=2*pos_bound, rtol=0)
                            np.testing.assert_allclose(
                                old_der, global_der[0],
                                atol=old_bound + global_bound, rtol=0,
                                err_msg="C1 seam error in the common global parameter")
                        previous = pos[-1], global_der[-1], global_bound


class LegacyEndJetsTests(unittest.TestCase):
    @staticmethod
    def original_raw_pieces(total, series_at, degree, nseg):
        """Direct original Chebyshev.fit reference, without any C0/C1 repair."""
        from numpy.polynomial import Chebyshev, Polynomial
        t = np.linspace(-1.0, 1.0, 1201)
        errors = []
        for piece in range(nseg):
            st = total*(piece + (t + 1)/2)/nseg
            last = []
            error = 0.0
            for series in series_at(st):
                strand = []
                for values in series:
                    cs = [np.round(Chebyshev.fit(t, values[:, axis], degree)
                                   .convert(kind=Polynomial).coef, 5)
                          for axis in range(3)]
                    fitted = np.array([poly.polyval(t, c) for c in cs]).T
                    error = max(error, np.linalg.norm(fitted-values, axis=1).max())
                    strand.append(cs)
                last.append(strand)
            errors.append(error)
        return last, errors

    def assert_same_coefficients_and_jets(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_strand, expected_strand in zip(actual, expected):
            self.assertEqual(len(actual_strand), 3)
            for actual_vector, expected_vector in zip(actual_strand, expected_strand):
                self.assertEqual(len(actual_vector), 3)
                for c, original in zip(actual_vector, expected_vector):
                    np.testing.assert_array_equal(c, original)
                    np.testing.assert_array_equal(c, np.round(c, 5))
                    # These are LOCAL final-piece t derivatives.  Exact raw
                    # coefficient equality preserves the old curl end jets.
                    self.assertEqual(poly.polyval(1.0, c), poly.polyval(1.0, original))
                    self.assertEqual(poly.polyval(1.0, poly.polyder(c)),
                                     poly.polyval(1.0, poly.polyder(original)))

    def test_default_degree_and_short_long_segment_start_match_original(self):
        def series_at(s):
            c = np.column_stack((0.2 + 0.03*s*s, 0.6 + 0.04*np.sin(s), 12.5-s))
            u = np.column_stack((0.1 + 0.01*s, 0.02*np.cos(s), 0.005*s))
            v = np.column_stack((0.01*np.sin(0.7*s), 0.03 + 0.002*s, 0.003*s*s))
            return [(c, u, v), (c+0.5*u, 0.7*u, 0.6*v), (c-0.3*u, 0.4*u, 0.8*v)]

        for total, expected_count in ((2.49, 1), (2.5, 2), (4.0, 2)):
            with self.subTest(total=total):
                calls = []

                def recorded(s):
                    calls.append(s.copy())
                    return series_at(s)

                actual, count = legacy_end_jets(total, recorded)
                self.assertEqual(count, expected_count)
                self.assertEqual(len(calls), expected_count)
                t = np.linspace(-1.0, 1.0, 1201)
                for k, samples in enumerate(calls):
                    np.testing.assert_array_equal(
                        samples, total*(k+(t+1)/2)/expected_count)
                expected, errors = self.original_raw_pieces(total, series_at, 20, count)
                self.assertLess(max(errors), 0.006)
                self.assert_same_coefficients_and_jets(actual, expected)

    def test_child_thickness_controls_adaptive_count_across_all_pieces(self):
        def series_at(s):
            zero = np.zeros((len(s), 3))
            child_v = zero.copy()
            child_v[:, 0] = 0.08*np.sin(8*s)
            return [(zero, zero, zero), (zero, zero, child_v)]

        actual, count = legacy_end_jets(1.8, series_at, degree=4)
        self.assertEqual(count, 3)
        for candidate in range(1, count):
            _, errors = self.original_raw_pieces(1.8, series_at, 4, candidate)
            self.assertGreaterEqual(max(errors), 0.006)
        expected, errors = self.original_raw_pieces(1.8, series_at, 4, count)
        self.assertLess(max(errors), 0.006)
        self.assert_same_coefficients_and_jets(actual, expected)

    def test_eight_segment_fallback_returns_last_raw_fit_without_c1_refit(self):
        calls = []

        def series_at(s):
            calls.append((len(s), s[0], s[-1]))
            zero = np.zeros((len(s), 3))
            c = np.column_stack((s, 2*s, -s))
            return [(c, zero, zero)]

        actual, count = legacy_end_jets(1.8, series_at, degree=0)
        self.assertEqual(count, 8)
        self.assertEqual(len(calls), sum(range(1, 9)))
        self.assertTrue(all(length == 1201 for length, _, _ in calls))
        self.assertAlmostEqual(calls[-1][1], 1.8*7/8)
        self.assertAlmostEqual(calls[-1][2], 1.8)
        expected, errors = self.original_raw_pieces(1.8, series_at, 0, count)
        self.assertGreater(max(errors), 0.006)
        self.assert_same_coefficients_and_jets(actual, expected)
        # A degree-zero raw end derivative is zero, unlike the source line's
        # nonzero derivative: do not silently replace it with a C1 endpoint jet.
        np.testing.assert_array_equal(
            [poly.polyval(1.0, poly.polyder(c)) for c in actual[0][0]], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
