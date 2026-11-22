"""AST-isolated head_root regression tests; never import all of hair.py.

Two narrow crown ribbons provide flow seeds at x=+/-.7.  A separate analytic
contact plane supports traced targets between them, as a cap/under-layer can.
The function's unused sheetfit import is replaced by an empty module stub, so
no real sheetfit or generator module is executed.
"""
import ast
import hashlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def plane_height(x, z):
    return 1.1 + 0.2*np.asarray(x) - 0.15*(np.asarray(z)-12.0)


class PlaneContact:
    """Exact sloped plane H=1.1+.2*x-.15*(z-12), only for |x|<.9."""
    def depth(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        valid = (np.abs(x) < 0.9) & np.isfinite(x) & np.isfinite(z)
        result = np.where(valid, plane_height(x, z), np.nan)
        return float(result) if not x.shape else result

    def depth_smooth(self, x, z):
        # A plane has no smoothing bias in its supported interior.  This mock
        # advertises regular neighbourhoods, without snapping to crown seeds.
        return self.depth(x, z)

    def gradient(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x), np.asarray(z))
        valid = (np.abs(x) < 0.9) & np.isfinite(x) & np.isfinite(z)
        gx, gz = np.where(valid, 0.2, np.nan), np.where(valid, -0.15, np.nan)
        return (float(gx), float(gz)) if not x.shape else (gx, gz)


def crown_ribbon(centre_x):
    def surface(t, w):
        t, w = np.broadcast_arrays(np.asarray(t), np.asarray(w))
        x = centre_x + 0.06*np.cos(w)
        z = 12.5 - 1.2*t
        return x, plane_height(x, z), z
    return surface


def load_head_root(contact, surfaces):
    path = Path(__file__).with_name('hair.py')
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'head_root']
    if len(functions) != 1:
        raise AssertionError('Expected exactly one head_root FunctionDef')
    namespace = {'np': np, 'math': math, 'HEAD_CONTACT': contact,
                 'HEAD_ROOT_SURFACES': surfaces}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, hashlib.sha256(source).hexdigest()


class HairRootsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contact = PlaneContact()
        cls.surfaces = [
            ('backNeg', (180, 100, 40), crown_ribbon(-0.7), (0.0, 1.0, 0.0, math.pi)),
            ('backPos', (180, 100, 40), crown_ribbon(0.7), (0.0, 1.0, 0.0, math.pi)),
        ]
        cls.namespace, cls.source_sha256 = load_head_root(cls.contact, cls.surfaces)
        cls.sheetfit_stub = types.ModuleType('sheetfit')

    def root(self, x, z):
        with patch.dict(sys.modules, {'sheetfit': self.sheetfit_stub}):
            return self.namespace['head_root'](x, z)

    def assert_native_plane_jet(self, point, tangent, normal):
        for value in (point, tangent, normal):
            self.assertEqual(value.shape, (3,))
            self.assertTrue(np.isfinite(value).all())
        self.assertAlmostEqual(point[1], plane_height(point[0], point[2]), places=12)
        expected_normal = np.array([-0.2, 1.0, 0.15])
        expected_normal /= np.linalg.norm(expected_normal)
        np.testing.assert_allclose(normal, expected_normal, atol=2e-12, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=12)
        self.assertAlmostEqual(float(tangent @ normal), 0.0, delta=2e-12)
        self.assertAlmostEqual(tangent[1], 0.2*tangent[0]-0.15*tangent[2], places=12)
        self.assertLess(tangent[2], -0.01)
        self.assertGreater(normal[1], 0.0)

    def test_only_head_root_is_loaded_without_other_hair_module_code(self):
        self.assertEqual(set(self.namespace),
                         {'np', 'math', 'HEAD_CONTACT', 'HEAD_ROOT_SURFACES',
                          '__builtins__', 'head_root'})
        self.assertEqual(set(vars(self.sheetfit_stub)),
                         {'__name__', '__doc__', '__package__', '__loader__', '__spec__'})

    def test_supported_traced_targets_keep_xz_instead_of_snapping_to_crests(self):
        targets = ((-0.5, 11.8), (-0.38, 11.9), (0.38, 12.0),
                   (0.47, 11.85), (0.5, 12.1))
        for x, z in targets:
            with self.subTest(target=(x, z)):
                point, tangent, normal, name, uv = self.root(x, z)
                np.testing.assert_array_equal(point[[0, 2]], [x, z])
                self.assert_native_plane_jet(point, tangent, normal)
                self.assertEqual(name, 'backPos' if x > 0 else 'backNeg')
                self.assertAlmostEqual(uv[0], (12.5-z)/1.2, places=10)
                self.assertAlmostEqual(uv[1], math.pi/2, places=12)
                # UV remains a crown FLOW seed, not a claim that the target
                # XZ lies on the narrow seed ribbon's centre crest.
                selected = next(surface for label, _, surface, _ in self.surfaces if label == name)
                seed = np.array(selected(*uv))
                self.assertAlmostEqual(seed[0], math.copysign(0.7, x), places=12)
                self.assertGreater(abs(seed[0]-point[0]), 0.15)
                self.assertAlmostEqual(seed[2], z, places=10)

    def test_unsupported_outer_target_falls_back_to_real_supported_crest(self):
        for x, z in ((-1.2, 11.85), (1.2, 12.05)):
            with self.subTest(target=(x, z)):
                self.assertTrue(np.isnan(self.contact.depth(x, z)))
                point, tangent, normal, name, uv = self.root(x, z)
                self.assertAlmostEqual(point[0], math.copysign(0.7, x), places=12)
                self.assertAlmostEqual(point[2], z, places=10)
                self.assertLess(abs(point[0]), 0.9)
                self.assertTrue(np.isfinite(self.contact.depth(point[0], point[2])))
                self.assert_native_plane_jet(point, tangent, normal)
                self.assertEqual(name, 'backPos' if x > 0 else 'backNeg')
                selected = next(surface for label, _, surface, _ in self.surfaces if label == name)
                np.testing.assert_allclose(point, selected(*uv), atol=2e-11, rtol=0,
                                           err_msg='fallback invented a floor instead of using actual support')


if __name__ == '__main__':
    unittest.main()
