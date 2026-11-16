"""AST-isolated _back_sections tests; never import or execute all of hair.py.

Synthetic Under uses complete/late-starting analytic planes or is absent.
No _sub_at is needed because subs=[]; only NumPy and the standard library are used.
"""
import ast
import hashlib
import math
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial import polynomial as poly


class UnderContact:
    """Broadcasting analytic y=0 plane, or an entirely missing projection."""
    def __init__(self, missing=False):
        self.value = np.nan if missing else 0.0

    def depth(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        value = np.full(x.shape, self.value)
        return float(value) if not x.shape else value

    def gradient(self, x, z):
        return self.depth(x, z), self.depth(x, z)


class LateSlopedUnder:
    """A plane with gradient (1.375,.12), visible only after s=.25.

    The fixture's root is at z=12.5.  Once entry finishes, width=.08 and
    half of the plane's across-width height difference is exactly .11.  The
    old absolute-offset interpolation incorrectly copied that .11 to s=0.
    """
    def depth(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        value = np.where(z <= 12.25, 0.02 + 1.375*(x-0.2) + 0.12*(z-12.5), np.nan)
        return float(value) if not x.shape else value

    def gradient(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        gx = np.where(z <= 12.25, 1.375, np.nan)
        gz = np.where(z <= 12.25, 0.12, np.nan)
        return (float(gx), float(gz)) if not x.shape else (gx, gz)


def load_back_sections(contact):
    path = Path(__file__).with_name('hair.py')
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == '_back_sections']
    if len(functions) != 1:
        raise AssertionError('Expected exactly one _back_sections FunctionDef')
    namespace = {'np': np, 'math': math, 'HEAD_UNDER_CONTACT': contact}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, hashlib.sha256(source).hexdigest()


def quintic_entry(u):
    u = np.clip(u, 0.0, 1.0)
    return 6*u**5 - 15*u**4 + 10*u**3


def section_samples(c, u, v, angles):
    """Evaluate the actual crescent formula, not just C+V or its centreline."""
    samples = []
    for angle in angles:
        sine = math.sin(angle)
        crescent = ((0.65*sine + 0.35*abs(sine))
                    *(0.78 + 0.22*math.cos(2*math.pi*math.cos(angle))))
        samples.append(c + math.cos(angle)*u + crescent*v)
    return np.array(samples)


class HairSectionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sig = np.linspace(0.0, 1.4, 2801)
        cls.root_length = 1.0
        entry = quintic_entry(cls.sig/0.18)
        cls.p = np.column_stack((np.full_like(cls.sig, 0.2),
                                 0.04 + 0.11*quintic_entry(cls.sig),
                                 12.5-cls.sig))
        cls.width = 0.08*entry
        cls.thickness = 0.05*entry
        cls.twist = np.zeros_like(cls.sig)
        cls.tangents = np.tile([0.0, 0.0, -1.0], (len(cls.sig), 1))
        cls.normal = np.array([0.0, 1.0, 0.0])
        cls.material = cls.sig.copy()
        cls.expected = (cls.p.copy(),
                        cls.width[:, None]*np.array([1.0, 0.0, 0.0]),
                        cls.thickness[:, None]*np.array([0.0, 1.0, 0.0]))
        cls.inputs = (cls.sig, cls.p, cls.width, cls.thickness, cls.twist,
                      cls.tangents, cls.normal, cls.material)
        cls.before = tuple(array.copy() for array in cls.inputs)
        cls.namespaces, cls.outputs, cls.source_hashes = {}, {}, {}
        for name, contact in (('plane', UnderContact()), ('missing', UnderContact(True))):
            namespace, sha = load_back_sections(contact)
            cls.namespaces[name], cls.source_hashes[name] = namespace, sha
            result = namespace['_back_sections'](
                cls.sig, cls.p, cls.width, cls.thickness, cls.twist, cls.tangents,
                [], cls.normal, cls.root_length, cls.material)
            if len(result) != 1:
                raise AssertionError('subs=[] must return only the mother strand')
            cls.outputs[name] = result[0]

    def test_only_requested_function_is_executed_and_all_results_are_finite(self):
        for mode, namespace in self.namespaces.items():
            with self.subTest(mode=mode):
                self.assertEqual(set(namespace),
                                 {'np', 'math', 'HEAD_UNDER_CONTACT', '__builtins__',
                                  '_back_sections'})
                self.assertNotIn('_sub_at', namespace)
                for vector in self.outputs[mode]:
                    self.assertEqual(vector.shape, (len(self.sig), 3))
                    self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(self.source_hashes['plane'], self.source_hashes['missing'])

    def test_native_root_values_and_first_jets_are_preserved(self):
        expected_derivatives = ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0])
        for mode, series in self.outputs.items():
            for vector, (actual, expected, derivative) in enumerate(
                    zip(series, self.expected, expected_derivatives)):
                with self.subTest(mode=mode, vector=vector):
                    np.testing.assert_array_equal(actual[0], expected[0])
                    # Dense local extrapolation checks a jet, not a secant
                    # across the rapidly opening width/thickness entry.
                    actual_derivative = poly.polyfit(self.sig[:8], actual[:8], 5)[1]
                    np.testing.assert_allclose(actual_derivative, derivative,
                                               atol=1e-6, rtol=0)

    def test_tail_samples_and_input_arrays_are_pointwise_unchanged(self):
        tail = self.sig >= self.root_length
        self.assertTrue(tail.any())
        for mode, series in self.outputs.items():
            for vector, (actual, original) in enumerate(zip(series, self.expected)):
                with self.subTest(mode=mode, vector=vector):
                    np.testing.assert_array_equal(actual[tail], original[tail])
        for actual, original in zip(self.inputs, self.before):
            np.testing.assert_array_equal(actual, original)

    def test_root_has_real_thickness_before_slow_bridge_weight_grows(self):
        index = int(np.argmin(np.abs(self.sig-0.2)))
        self.assertAlmostEqual(self.sig[index], 0.2)
        c, _, v = self.outputs['plane']
        real_thickness = np.linalg.norm(v[index])
        self.assertGreater(real_thickness, 0.03)
        self.assertGreater(real_thickness, 0.6*self.thickness[index])
        self.assertLessEqual(real_thickness, self.thickness[index]+1e-13)
        # The old V*b paper-thin result would be only 0.002896 here.
        old_flattened = self.thickness[index]*quintic_entry(self.sig[index])
        self.assertGreater(real_thickness, 10*old_flattened)
        self.assertGreater(c[index, 1] + v[index, 1], 0.04)

    def test_all_seventeen_outer_section_angles_clear_the_under_plane(self):
        # Native visible outer crescent is w in [0, pi], matching its declared
        # outer-section support domain; its hidden underside is not this test.
        angles = np.linspace(0.0, math.pi, 17)
        surface = section_samples(*self.outputs['plane'], angles)
        root = self.sig < self.root_length
        self.assertGreaterEqual(surface[:, root, 1].min(), -1e-12)
        index = int(np.argmin(np.abs(self.sig-0.2)))
        # Check a useful positive entry clearance without assuming the exact
        # lift value: its smoothed majorant can anticipate nearby contact.
        self.assertGreaterEqual(surface[:, index, 1].min(), 0.007-1e-12)

    def test_missing_under_projection_keeps_native_hermite_without_a_floor(self):
        for actual, original in zip(self.outputs['missing'], self.expected):
            np.testing.assert_array_equal(actual, original)
        # Distinguish absence-of-support from the known-plane modification.
        self.assertGreater(np.max(np.abs(self.outputs['plane'][0][:, 1]-self.p[:, 1])), 0.01)

    def test_late_sloped_projection_cannot_open_the_zero_width_native_root(self):
        contact = LateSlopedUnder()
        self.assertTrue(np.isnan(contact.depth(self.p[0, 0], self.p[0, 2])))
        self.assertTrue(all(np.isnan(value) for value in contact.gradient(
            self.p[0, 0], self.p[0, 2])))
        probe = int(np.argmin(np.abs(self.sig-0.4)))
        self.assertTrue(np.isfinite(contact.depth(self.p[probe, 0], self.p[probe, 2])))
        namespace, _ = load_back_sections(contact)
        result = namespace['_back_sections'](
            self.sig, self.p, self.width, self.thickness, self.twist, self.tangents,
            [], self.normal, self.root_length, self.material)
        self.assertEqual(len(result), 1)
        c, u, v = result[0]
        # Check the actual cross-section, not just its centre.  At a zero-width
        # root no angle may leave the anchor, even if support appears later.
        np.testing.assert_array_equal(c[0], self.p[0])
        np.testing.assert_array_equal(u[0], np.zeros(3))
        np.testing.assert_array_equal(v[0], np.zeros(3))
        root_surface = section_samples(c[:1], u[:1], v[:1],
                                       np.linspace(0.0, 2*math.pi, 33))[:, 0]
        np.testing.assert_array_equal(root_surface, np.tile(self.p[0], (33, 1)))
        for vector, derivative in zip((c, u, v),
                                      ([0.0, 0.0, -1.0], np.zeros(3), np.zeros(3))):
            self.assertTrue(np.isfinite(vector).all())
            np.testing.assert_allclose(
                poly.polyfit(self.sig[:8], vector[:8], 5)[1], derivative,
                atol=1e-6, rtol=0)
        # This must still reconstruct a genuinely tilted section later, rather
        # than fixing the regression by suppressing every y slope.
        self.assertGreater(abs(u[probe, 1]), 0.04)
        tail = self.sig >= self.root_length
        for actual, original in zip((c, u, v), self.expected):
            np.testing.assert_array_equal(actual[tail], original[tail])

    def test_tilted_native_normal_keeps_traced_xz_width_and_original_tail(self):
        tilted_normal = np.array([0.6, 0.8, 0.0])
        namespace, _ = load_back_sections(UnderContact())
        result = namespace['_back_sections'](
            self.sig, self.p, self.width, self.thickness, self.twist, self.tangents,
            [], tilted_normal, self.root_length, self.material)
        self.assertEqual(len(result), 1)
        c, u, v = result[0]
        projected_width = np.linalg.norm(u[:, [0, 2]], axis=1)
        # W is a back-view XZ half-width, not length along a tilted normal's
        # tangent plane.  The previous cross(T,nativeN) lost about 20% here.
        np.testing.assert_allclose(projected_width, self.width, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(u[:, [0, 2]], self.outputs['plane'][1][:, [0, 2]])
        probe = int(np.argmin(np.abs(self.sig-0.2)))
        self.assertAlmostEqual(projected_width[probe], 0.08, places=13)
        # The native tilt still affects thickness; do not pass by ignoring it.
        self.assertGreater(abs(v[probe, 0]), 0.01)
        tail = self.sig >= self.root_length
        for actual, original in zip((c, u, v), self.expected):
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual[tail], original[tail])
        np.testing.assert_array_equal(tilted_normal, [0.6, 0.8, 0.0])


if __name__ == '__main__':
    unittest.main()
