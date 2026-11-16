"""Exercise production hanging-fit functions without importing/running hair.py.

Only _poly_fit and _hang_pieces FunctionDefs are AST-extracted and compiled;
no surface generation, module statements, or Arch commands are executed.
Run ``python test_hair_fit.py -v`` or use standard unittest discovery.
"""
import ast
import hashlib
import math
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial import polynomial as poly


HAIR_PATH = Path(__file__).with_name('hair.py')


def load_fit_functions():
    source = HAIR_PATH.read_bytes()
    tree = ast.parse(source, filename=str(HAIR_PATH))
    names = ('_poly_fit', '_hang_pieces')
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(node.name for node in functions) != sorted(names):
        raise AssertionError('Expected exactly the two production fit FunctionDefs')
    isolated = ast.Module(body=functions, type_ignores=[])
    namespace = {'np': np, 'math': math, 'CH_DEG': 20}
    exec(compile(isolated, str(HAIR_PATH), 'exec'), namespace)
    return namespace, hashlib.sha256(source).hexdigest()


def joined_vectors(coefficients, previous_end, current_start):
    # Match the caller's C0 seam callback, not the endpoint-fix implementation.
    fade = np.array([0.5, -0.75, 0.0, 0.25])
    return [np.round(poly.polyadd(c, (end-start)*fade), 5)
            for c, end, start in zip(coefficients, previous_end, current_start)]


def samples_at(s, child=0):
    """Low-curvature C/U/V with several identically-zero XYZ coordinates."""
    c = np.column_stack((np.zeros_like(s),
                         0.8 + 0.025*s + 0.001*s*s,
                         12.4 - 0.4*s + 0.002*s*s))
    u = np.column_stack((0.12 + 0.004*s, np.zeros_like(s), 0.008*s))
    v = np.column_stack((np.zeros_like(s), 0.035 + 0.001*s, np.zeros_like(s)))
    if child:
        c = c + 0.3*u
        u, v = 0.7*u, 0.8*v
    return c, u, v


def analytic_derivatives(s, child=0):
    c = np.array([0.0, 0.025 + 0.002*s, -0.4 + 0.004*s])
    u, v = np.array([0.004, 0.0, 0.008]), np.array([0.0, 0.001, 0.0])
    if child:
        c = c + 0.3*u
        u, v = 0.7*u, 0.8*v
    return c, u, v


class HairFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.source_sha256 = load_fit_functions()
        cls.total = 3.6
        cls.end_span = 1.2  # Old last-piece physical span, deliberately unequal.
        cls.t = np.linspace(-1.0, 1.0, 1201)
        cls.sig = np.linspace(0.0, cls.total, 2401)
        parent = samples_at(cls.sig)
        # An identical first child exposes every parent-piece error through
        # returned parts; the second child independently varies its frame.
        cls.prepared = [parent, tuple(v.copy() for v in parent), samples_at(cls.sig, 1)]
        cls.subs = [{}, {}]
        cls.targets = []
        cls.desired_values, cls.desired_derivatives = [], []
        for strand, child in enumerate((0, 0, 1)):
            values = np.array([v[-1] for v in samples_at(cls.sig, child)])
            derivatives = np.array(analytic_derivatives(cls.total, child))
            values += np.array([[0.0002, -0.0001, 0.0],
                                [0.0, -0.0001, 0.0],
                                [0.0001, 0.0, 0.00015]])
            derivatives += np.array([[0.006, 0.0, 0.04 if strand < 2 else 0.055],
                                     [0.0, 0.0015, 0.0],
                                     [0.0, 0.0, -0.002]])
            # Old target polynomials are expressed in their OWN local t.
            # d/ds = (2/old_span)*d/dt, not (2/new_span)*d/dt.
            old_local_derivative = derivatives*cls.end_span/2.0
            cls.targets.append([
                [np.array([value-derivative, derivative])
                 for value, derivative in zip(vector, old_derivative)]
                for vector, old_derivative in zip(values, old_local_derivative)
            ])
            cls.desired_values.append(values)
            cls.desired_derivatives.append(derivatives)
        cls.runs = {}
        for count in (1, 2):
            arguments = (count, cls.t, cls.sig, cls.prepared, joined_vectors, cls.subs)
            baseline = cls.namespace['_hang_pieces'](*arguments)
            corrected = cls.namespace['_hang_pieces'](
                *arguments, end_targets=cls.targets, end_span=cls.end_span)
            cls.runs[count] = baseline, corrected

    @staticmethod
    def rounding_bounds(coefficients):
        length = len(coefficients)
        return (length*0.5e-5 + 2e-9,
                length*(length-1)*0.25e-5 + 2e-9)

    def test_ast_namespace_contains_only_the_two_production_functions(self):
        self.assertEqual(
            set(self.namespace),
            {'np', 'math', 'CH_DEG', '__builtins__', '_poly_fit', '_hang_pieces'})
        self.assertEqual(self.namespace['CH_DEG'], 20)

    def test_exact_zero_coordinates_are_short_before_fix_and_padded_after(self):
        values = np.column_stack((np.zeros_like(self.t), 0.1 + 0.02*self.t,
                                  np.zeros_like(self.t)))
        coefficients, _, _ = self.namespace['_poly_fit'](self.t, values, 20)
        for axis in (0, 2):
            np.testing.assert_array_equal(coefficients[axis], [0.0])
        for count, (baseline, corrected) in self.runs.items():
            with self.subTest(nseg=count):
                # Parent C.x is exactly zero before correction, but receives
                # both a nonzero end value and a nonzero end derivative.
                raw_cx = baseline[2][0][0][0]
                fixed_cx = corrected[2][0][0][0]
                np.testing.assert_array_equal(raw_cx, [0.0])
                self.assertGreaterEqual(len(fixed_cx), 20)
                self.assertTrue(np.any(fixed_cx != 0.0))

    def test_last_values_and_physical_derivatives_match_old_targets(self):
        for count, (_, corrected) in self.runs.items():
            last = corrected[2]
            new_span = self.total/count
            for strand, coefficients in enumerate(last):
                for vector, xyz in enumerate(coefficients):
                    for axis, c in enumerate(xyz):
                        with self.subTest(nseg=count, strand=strand,
                                          vector=vector, axis=axis):
                            position_bound, derivative_bound = self.rounding_bounds(c)
                            self.assertAlmostEqual(
                                poly.polyval(1.0, c),
                                self.desired_values[strand][vector, axis],
                                delta=position_bound)
                            physical_derivative = poly.polyval(1.0, poly.polyder(c))*2/new_span
                            self.assertAlmostEqual(
                                physical_derivative,
                                self.desired_derivatives[strand][vector, axis],
                                delta=derivative_bound*2/new_span)

    def test_end_fix_preserves_start_jets_earlier_pieces_and_locality(self):
        for count, (baseline, corrected) in self.runs.items():
            with self.subTest(nseg=count):
                for old_strand, new_strand in zip(baseline[2], corrected[2]):
                    for old_vector, new_vector in zip(old_strand, new_strand):
                        for before, after in zip(old_vector, new_vector):
                            pos_bound, der_bound = self.rounding_bounds(after)
                            self.assertAlmostEqual(poly.polyval(-1.0, after),
                                                   poly.polyval(-1.0, before),
                                                   delta=pos_bound)
                            self.assertAlmostEqual(
                                poly.polyval(-1.0, poly.polyder(after)),
                                poly.polyval(-1.0, poly.polyder(before)), delta=der_bound)
                for old_parts, new_parts in zip(baseline[0], corrected[0]):
                    for previous, fixed in zip(old_parts[:-1], new_parts[:-1]):
                        for old_vector, new_vector in zip(previous[1], fixed[1]):
                            for before, after in zip(old_vector, new_vector):
                                np.testing.assert_array_equal(after, before)
                # C.z has no requested value shift.  A spread-out cubic
                # derivative bump moves its midpoint by ~.018 (one piece);
                # the actual degree-19 bump keeps this midpoint essentially fixed.
                self.assertLess(abs(poly.polyval(0.0, corrected[2][0][0][2])
                                    - poly.polyval(0.0, baseline[2][0][0][2])), 1e-4)

    def test_rounded_finite_parts_and_returned_errors_recompute(self):
        for count, (baseline, corrected) in self.runs.items():
            for fixed, result in ((False, baseline), (True, corrected)):
                with self.subTest(nseg=count, corrected=fixed):
                    parts, errors, last = result
                    self.assertEqual(len(parts), len(self.subs))
                    self.assertEqual(len(errors), count)
                    self.assertLess(max(errors), 0.006)
                    for strand in last:
                        for vector in strand:
                            for c in vector:
                                self.assertTrue(np.isfinite(c).all())
                                np.testing.assert_array_equal(c, np.round(c, 5))
                    for piece in range(count):
                        st = self.total*(piece + (self.t+1)/2)/count
                        recomputed = 0.0
                        for child, child_parts in enumerate(parts):
                            self.assertEqual(len(child_parts), count)
                            kind, coefficients, precision = child_parts[piece]
                            self.assertEqual(kind, 'poly')
                            self.assertEqual(precision, max(1, math.ceil(self.total/count/1.5)))
                            source = [np.column_stack([
                                np.interp(st, self.sig, vector[:, axis]) for axis in range(3)])
                                for vector in self.prepared[child+1]]
                            fitted = []
                            for xyz in coefficients:
                                for c in xyz:
                                    self.assertTrue(np.isfinite(c).all())
                                    np.testing.assert_array_equal(c, np.round(c, 5))
                                fitted.append(np.column_stack([poly.polyval(self.t, c) for c in xyz]))
                            for angle in np.linspace(0, 2*np.pi, 17):
                                sine = math.sin(angle)
                                crescent = ((0.65*sine + 0.35*abs(sine))
                                            *(0.78 + 0.22*math.cos(2*math.pi*math.cos(angle))))
                                actual_surface = fitted[0] + math.cos(angle)*fitted[1] + crescent*fitted[2]
                                source_surface = source[0] + math.cos(angle)*source[1] + crescent*source[2]
                                recomputed = max(recomputed, np.linalg.norm(
                                    actual_surface-source_surface, axis=1).max())
                        self.assertAlmostEqual(errors[piece], recomputed, places=11)
                    # The identical child makes its stored final coefficients
                    # an independent observable for the otherwise-hidden parent.
                    for mother_vector, child_vector in zip(last[0], last[1]):
                        for mother, child in zip(mother_vector, child_vector):
                            np.testing.assert_array_equal(mother, child)


class NativeRootJetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.source_sha256 = load_fit_functions()
        cls.total = 3.6
        cls.t = np.linspace(-1.0, 1.0, 1201)
        cls.sig = np.linspace(0.0, cls.total, 2401)
        cls.root_tangent = np.array([0.6, 0.3, -2.0])  # Deliberately NOT unit length.
        cls.unit_tangent = cls.root_tangent/np.linalg.norm(cls.root_tangent)
        normal = np.array([0.0, 1.0, 0.0]) - cls.unit_tangent[1]*cls.unit_tangent
        normal /= np.linalg.norm(normal)
        side = np.cross(cls.unit_tangent, normal)
        side /= np.linalg.norm(side)
        phase = np.clip(cls.sig/0.06, 0.0, 1.0)
        entry = 6*phase**5 - 15*phase**4 + 10*phase**3
        c = (np.array([0.2, 0.8, 12.4]) + cls.sig[:, None]*cls.unit_tangent
             + (0.02*entry)[:, None]*normal
             + cls.sig[:, None]**2*np.array([0.004, 0.003, -0.002]))
        u = (entry*(0.08 + 0.01*cls.sig))[:, None]*side
        v = (entry*(0.05 + 0.004*cls.sig))[:, None]*normal
        # All centres have the same native jet; child offsets open with zero
        # first derivative.  The identical child exposes the parent's first fit.
        parent = (c, u, v)
        cls.prepared = [parent, tuple(a.copy() for a in parent),
                        (c+0.15*u, 0.65*u, 0.75*v)]
        cls.subs = [{}, {}]
        cls.runs = {}
        for count in (1, 2):
            arguments = (count, cls.t, cls.sig, cls.prepared, joined_vectors, cls.subs)
            default = cls.namespace['_hang_pieces'](*arguments)
            native = cls.namespace['_hang_pieces'](*arguments, root_tangent=cls.root_tangent)
            cls.runs[count] = default, native

    def interpolated(self, strand, piece, count):
        s = self.total*(piece+(self.t+1)/2)/count
        return [np.column_stack([np.interp(s, self.sig, a[:, axis]) for axis in range(3)])
                for a in self.prepared[strand]]

    def assert_rounded_finite(self, coefficients):
        for xyz in coefficients:
            for c in xyz:
                self.assertTrue(np.isfinite(c).all())
                np.testing.assert_array_equal(c, np.round(c, 5))

    def test_fit_c1_override_beats_coarse_ramp_estimate_and_default_none_is_unchanged(self):
        from hairjoin import fit_c1
        values = self.interpolated(0, 0, 1)[0]
        native_left = self.unit_tangent*self.total/2
        coarse_left = poly.polyfit(self.t[:7]-self.t[0], values[:7], 2)[1]
        derivative_bound = 20*21*0.25e-5 + 2e-9
        # Six coarse fitting cells span a material portion of the .06 ramp;
        # the quadratic sample estimate is NOT the actual root derivative.
        self.assertGreater(np.max(np.abs(coarse_left-native_left)), 10*derivative_bound)
        default = fit_c1(self.t, values, 20)
        explicit_none = fit_c1(self.t, values, 20, start_derivative=None)
        native = fit_c1(self.t, values, 20, start_derivative=native_left)
        for actual, expected in zip(explicit_none[0], default[0]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(explicit_none[1], default[1])
        self.assertEqual(explicit_none[2], default[2])
        estimated_right = poly.polyfit(self.t[-7:]-self.t[-1], values[-7:], 2)[1]
        for coefficients, expected_left in ((default[0], coarse_left), (native[0], native_left)):
            for axis, c in enumerate(coefficients):
                pos_bound, der_bound = HairFitTests.rounding_bounds(c)
                self.assertAlmostEqual(poly.polyval(-1.0, poly.polyder(c)),
                                       expected_left[axis], delta=der_bound)
                self.assertAlmostEqual(poly.polyval(1.0, poly.polyder(c)),
                                       estimated_right[axis], delta=der_bound)
                np.testing.assert_allclose(poly.polyval([-1.0, 1.0], c),
                                           values[[0, -1], axis], atol=pos_bound, rtol=0)
                self.assertTrue(np.isfinite(c).all())
                np.testing.assert_array_equal(c, np.round(c, 5))
        np.testing.assert_array_equal(native_left, self.unit_tangent*self.total/2)
        self.assertAlmostEqual(native[2], np.linalg.norm(native[1]-values, axis=1).max(), places=12)

    def test_hanging_first_piece_restores_native_c_and_zero_uv_jets_with_physical_scale(self):
        for count, (default, native) in self.runs.items():
            span = self.total/count
            for child, pieces in enumerate(native[0]):
                coefficients = pieces[0][1]
                source = self.interpolated(child+1, 0, count)
                self.assert_rounded_finite(coefficients)
                self.assertTrue(np.isfinite(native[1]).all())
                for vector, xyz in enumerate(coefficients):
                    wanted = self.unit_tangent if vector == 0 else np.zeros(3)
                    for axis, c in enumerate(xyz):
                        with self.subTest(nseg=count, child=child, vector=vector, axis=axis):
                            pos_bound, der_bound = HairFitTests.rounding_bounds(c)
                            self.assertAlmostEqual(poly.polyval(-1.0, c), source[vector][0, axis],
                                                   delta=pos_bound)
                            derivative = poly.polyval(-1.0, poly.polyder(c))*2/span
                            self.assertAlmostEqual(derivative, wanted[axis], delta=der_bound*2/span)
                # Ensure the unmodified default path still fits its actual
                # sample-estimated UV jets; it is not silently native-clamped.
                coarse = poly.polyfit(self.t[:7]-self.t[0], source[1][:7], 2)[1]
                self.assertGreater(np.linalg.norm(coarse), 0.02)
                for c, expected in zip(default[0][child][0][1][1], coarse):
                    _, bound = HairFitTests.rounding_bounds(c)
                    self.assertAlmostEqual(poly.polyval(-1.0, poly.polyder(c)), expected, delta=bound)
            for mother, duplicate in zip(native[2][0], native[2][1]):
                for c, d in zip(mother, duplicate):
                    np.testing.assert_array_equal(c, d)
        np.testing.assert_array_equal(self.root_tangent, [0.6, 0.3, -2.0])

    def test_nonfirst_pieces_keep_default_fits_and_only_the_normal_c0_seam_adjustment(self):
        count = 2
        _, native = self.runs[count]
        for child, pieces in enumerate(native[0]):
            source = self.interpolated(child+1, 1, count)
            # Reconstruct the second piece with the DEFAULT _poly_fit API.
            # Its only dependence on the first piece is the usual C0 seam
            # offset, including any five-place rounding at that seam.
            raw = [self.namespace['_poly_fit'](self.t, vector, 20)[0] for vector in source]
            previous = pieces[0][1]
            expected = [joined_vectors(c, [poly.polyval(1.0, a) for a in before],
                                        [poly.polyval(-1.0, a) for a in c])
                        for c, before in zip(raw, previous)]
            actual = pieces[1][1]
            self.assert_rounded_finite(actual)
            for actual_xyz, expected_xyz in zip(actual, expected):
                for c, d in zip(actual_xyz, expected_xyz):
                    np.testing.assert_array_equal(c, d)
            # Later widths vary with s, so incorrectly imposing zero UV jets
            # on every piece is observably different, not a vacuous equality.
            later_uv_jet = poly.polyfit(self.t[:7]-self.t[0], source[1][:7], 2)[1]
            self.assertGreater(np.linalg.norm(later_uv_jet), 0.005)

    def test_hanging_explicit_none_preserves_default_api_coefficients_and_errors(self):
        count = 2
        original = self.runs[count][0]
        explicit = self.namespace['_hang_pieces'](
            count, self.t, self.sig, self.prepared, joined_vectors, self.subs,
            root_tangent=None)
        np.testing.assert_array_equal(explicit[1], original[1])
        for actual_parts, original_parts in zip(explicit[0], original[0]):
            for actual, expected in zip(actual_parts, original_parts):
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[2], expected[2])
                for actual_xyz, expected_xyz in zip(actual[1], expected[1]):
                    for c, d in zip(actual_xyz, expected_xyz):
                        np.testing.assert_array_equal(c, d)
        for actual, expected in zip(explicit[2], original[2]):
            for actual_xyz, expected_xyz in zip(actual, expected):
                for c, d in zip(actual_xyz, expected_xyz):
                    np.testing.assert_array_equal(c, d)


if __name__ == '__main__':
    unittest.main()
