"""Synthetic CrownContact tests: unittest + NumPy, with no generator or Arch.

Run with ``python test_haircontact.py -v`` or unittest discovery.  Every expected
finite depth comes from an analytic plane, sphere, or circular tube.  The sphere
keeps both positive-y branches, so selecting the outermost branch is necessary.
"""
import unittest

import numpy as np

from haircontact import CrownContact


COLOUR = (180, 100, 40)
CACHE_STEP = 0.004
# Smooth curved cache cells interpolate with O(pitch**2) error.  This bound
# allows the bounded curvature of these fixtures, not arbitrary silhouettes.
# Exact native inversion and scalar/batch consistency retain separate bounds.
CACHE_DEPTH_ATOL = 6*CACHE_STEP**2
EXACT_DEPTH_ATOL = 2e-7


def plane(u, v):
    return u, 1.2 + 0.2*u - 0.1*v, v


def lower_plane(u, v):
    return u, 0.85 + 0.1*u, v


def upper_plane(u, v):
    # A scalar coordinate must broadcast against the parameter arrays.
    return u, 1.65, v


def sphere(latitude, longitude):
    radius = 0.8
    ring = radius*np.cos(latitude)
    return (ring*np.sin(longitude),
            1.5 + ring*np.cos(longitude),
            radius*np.sin(latitude))


def tube(longitudinal, angle):
    return (0.25 + 0.32*np.cos(angle),
            1.1 + 0.1*longitudinal + 0.32*np.sin(angle),
            longitudinal)


def tube_swapped(angle, longitudinal):
    return tube(longitudinal, angle)


def contact(surfaces):
    return CrownContact(surfaces, seed_step=0.1, residual=1e-7, iterations=20,
                        cache_step=CACHE_STEP)


class CrownContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plane_contact = contact([
            ('testPlane', COLOUR, plane, (-1.0, 1.0, -1.0, 1.0)),
        ])
        cls.cap_contact = contact([
            # Non-tube hairCap must retain BOTH full parameter domains.  Front
            # points with x<0 require longitude<0; z<0 requires latitude<0.
            ('hairCap', COLOUR, sphere,
             (-np.pi/2, np.pi/2, -np.pi, np.pi)),
        ])
        cls.tube_contacts = (
            contact([('backTube', COLOUR, tube,
                      (-1.4, 1.6, -np.pi, np.pi))]),
            contact([('sideTube', COLOUR, tube_swapped,
                      (-np.pi, np.pi, -1.4, 1.6))]),
        )
        cls.layers = [
            ('lowerPlane', COLOUR, lower_plane, (-1.0, 1.0, -1.0, 1.0)),
            ('upperPlane', COLOUR, upper_plane, (-0.45, 0.45, -0.6, 0.6)),
        ]
        cls.layer_contact = contact(cls.layers)

    def assert_scalar_equivalent(self, query, x, z, atol=2e-7, method='depth'):
        bx, bz = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        evaluate = getattr(query, method)
        vector = evaluate(x, z)
        scalars = np.array([
            evaluate(float(a), float(b))
            for a, b in zip(bx.ravel(), bz.ravel())
        ]).reshape(bx.shape)
        np.testing.assert_allclose(
            vector, scalars, atol=atol, rtol=0, equal_nan=True,
            err_msg='broadcast depth differs from pointwise scalar queries')

    def test_analytic_plane_broadcasting_and_scalar_equivalence(self):
        x = np.array([-1.0, -0.35, 0.22, 1.0])[:, None]
        z = np.array([-1.0, -0.47, 0.31, 1.0])[None, :]
        depth = self.plane_contact.depth(x, z)
        self.assertEqual(depth.shape, (4, 4))
        np.testing.assert_allclose(depth, 1.2 + 0.2*x - 0.1*z,
                                   atol=2e-8, rtol=0)
        self.assert_scalar_equivalent(self.plane_contact, x, z, atol=2e-8)
        result = self.plane_contact.depth(0.13, -0.27)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.2 + 0.2*0.13 + 0.1*0.27, places=8)
        # Broadcasting works with one scalar, and with empty array dimensions.
        np.testing.assert_allclose(
            self.plane_contact.depth(0.2, z), 1.24 - 0.1*z,
            atol=2e-8, rtol=0)
        self.assertEqual(
            self.plane_contact.depth(np.empty((0, 3)), 0.0).shape, (0, 3))

    def test_two_layers_choose_upper_envelope_and_ignore_patch_order(self):
        # Queries immediately either side of patch edges must retain the
        # discontinuous upper envelope, not average it with the lower plane.
        x = np.array([-0.8, -0.3, 0.0, 0.4, 0.8, 0.0,
                      -0.451, -0.449, 0.449, 0.451, 0.0, 0.0, 0.0, 0.0])
        z = np.array([0.0, -0.3, 0.0, 0.4, 0.0, 0.85,
                      0.0, 0.0, 0.0, 0.0, -0.601, -0.599, 0.599, 0.601])
        in_upper = (np.abs(x) <= 0.45) & (np.abs(z) <= 0.6)
        expected = np.where(in_upper, 1.65, 0.85 + 0.1*x)
        np.testing.assert_allclose(self.layer_contact.depth(x, z), expected,
                                   atol=2e-8, rtol=0)
        reverse = contact(self.layers[::-1])
        np.testing.assert_allclose(reverse.depth(x, z), expected, atol=2e-8, rtol=0)
        self.assert_scalar_equivalent(self.layer_contact, x, z, atol=2e-8)

    def test_haircap_full_domain_sphere_and_outer_branch(self):
        x = np.array([-0.55, -0.15, 0.0, 0.2, 0.55])[:, None]
        z = np.array([-0.4, -0.1, 0.0, 0.35])[None, :]
        expected = 1.5 + np.sqrt(0.8**2 - x*x - z*z)
        for method, tolerance in (('depth', CACHE_DEPTH_ATOL),
                                  ('depth_exact', EXACT_DEPTH_ATOL)):
            with self.subTest(method=method):
                depth = getattr(self.cap_contact, method)(x, z)
                self.assertTrue(np.isfinite(depth).all())
                np.testing.assert_allclose(depth, expected, atol=tolerance, rtol=0)
                self.assert_scalar_equivalent(self.cap_contact, x, z, method=method)

    def test_tubes_keep_full_longitudinal_domain_in_both_parameter_orders(self):
        x = 0.25 + np.array([-0.28, -0.1, 0.05, 0.27])[:, None]
        # The longitudinal domain is NOT [0,1] or [-1,1].  Both ends and both
        # signs must remain queryable when the angle is clipped to outer half.
        z = np.array([-1.4, -1.19, -0.1, 1.31, 1.6])[None, :]
        expected = 1.1 + 0.1*z + np.sqrt(0.32**2 - (x - 0.25)**2)
        for order, query in enumerate(self.tube_contacts):
            for method, tolerance in (('depth', CACHE_DEPTH_ATOL),
                                      ('depth_exact', EXACT_DEPTH_ATOL)):
                with self.subTest(angle_parameter=1-order, method=method):
                    depth = getattr(query, method)(x, z)
                    self.assertTrue(np.isfinite(depth).all())
                    np.testing.assert_allclose(depth, expected, atol=tolerance, rtol=0)
                    self.assert_scalar_equivalent(query, x, z, method=method)

    def test_missing_plane_projection_and_nonfinite_inputs_return_nan(self):
        x = np.array([1.2, -1.2, 0.0, 0.0, np.nan, np.inf, 0.1, 0.1])
        z = np.array([0.0, 0.0, -1.2, 1.2, 0.0, 0.0, np.nan, -np.inf])
        self.assertTrue(np.isnan(self.plane_contact.depth(x, z)).all())
        self.assertTrue(np.isnan(self.plane_contact.depth(1.2, 0.0)))
        self.assert_scalar_equivalent(self.plane_contact, x, z)

    def test_no_projection_inside_sphere_bounding_box_is_not_filled(self):
        # These are inside the sampled X/Z rectangle, but outside its disk.
        x = np.array([0.72, -0.72, 0.7, -0.7])
        z = np.array([0.6, 0.6, -0.6, -0.6])
        self.assertTrue((x*x + z*z > 0.8**2).all())
        self.assertTrue(np.isnan(self.cap_contact.depth(x, z)).all())
        self.assert_scalar_equivalent(self.cap_contact, x, z)

    def test_empty_surface_collection_has_no_fixed_depth_fallback(self):
        empty = contact([])
        value = empty.depth(0.0, 0.0)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isnan(value))
        result = empty.depth(np.array([-1.0, 0.0, 1.0])[:, None], [0.0, 0.4])
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.isnan(result).all())

    def test_smooth_support_preserves_interior_plane_and_broadcasting(self):
        x = np.array([-0.4, 0.1, 0.4])[:, None]
        z = np.array([-0.3, 0.2])[None, :]
        result = self.plane_contact.depth_smooth(x, z)
        np.testing.assert_allclose(result, 1.2 + 0.2*x - 0.1*z,
                                   atol=2e-8, rtol=0)
        self.assert_scalar_equivalent(self.plane_contact, x, z, atol=2e-8,
                                      method='depth_smooth')
        self.assertIsInstance(self.plane_contact.depth_smooth(0.1, 0.2), float)

    def test_smooth_layer_step_is_continuous_without_changing_actual_depth(self):
        x = np.linspace(0.4, 0.5, 101)
        actual_before = self.layer_contact.depth(x, 0.0).copy()
        smooth = self.layer_contact.depth_smooth(x, 0.0)
        self.assertTrue(np.isfinite(smooth).all())
        # A 0.75-unit actual step becomes a smooth transition at sigma=.035;
        # this .001-spaced trace must not retain the original discontinuity.
        self.assertLess(np.max(np.abs(np.diff(smooth))), 0.015)
        self.assertGreater(smooth[50], (0.85 + 0.1*x[50]) + 0.2)
        self.assertLess(smooth[50], 1.65 - 0.2)
        self.assert_scalar_equivalent(self.layer_contact, x[::10], 0.0,
                                      method='depth_smooth')
        np.testing.assert_array_equal(self.layer_contact.depth(x, 0.0), actual_before)

    def test_smoothing_does_not_fill_missing_support_or_nonfinite_queries(self):
        x = np.array([0.72, -0.72, 0.7, -0.7, np.nan, np.inf])
        z = np.array([0.6, 0.6, -0.6, -0.6, 0.0, 0.0])
        self.assertTrue(np.isnan(self.cap_contact.depth_smooth(x, z)).all())
        self.assertTrue(np.isnan(contact([]).depth_smooth(0.0, 0.0)))
        self.assertEqual(
            self.plane_contact.depth_smooth(np.empty((0, 3)), 0.0).shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
