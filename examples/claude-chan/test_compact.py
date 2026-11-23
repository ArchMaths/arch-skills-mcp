"""compact.py: surfaces rewritten as polynomials only where they look the same.

Run ``python test_compact.py -v`` or use standard unittest discovery.
"""
import math
import unittest

import numpy as np

import compact


def _pts(line, ns_lines, n=9):
    """A surface line's points on an n x n grid of its own parameter ranges."""
    ns = compact._cmp_namespace(ns_lines)
    X, Y, Z, u, u0, u1, v, v0, v1, _ = compact._cmp_parse(line)
    lim = [float(eval(compact._cmp_py(t), ns)) for t in (u0, u1, v0, v1)]
    A = np.linspace(0, 1, n)
    U, V = np.meshgrid(lim[0] + (lim[1] - lim[0]) * A, lim[2] + (lim[3] - lim[2]) * A, indexing='ij')
    return compact._cmp_eval(ns, X, Y, Z, u, v, U, V)


FUNCS = ['bump(a,b)=0.2*exp(-((a-0.3)^2+(b+0.1)^2)/0.4)+0.05*sin(3*a)*cos(2*b)',
         'zz(a)=(2/pi)*asin(0.97*sin(a))']


class CompactTest(unittest.TestCase):
    def test_smooth_surface_is_rewritten_within_tolerance_same_grid_and_side(self):
        line = '(1.5*s+0.2*t^2,t,bump(s,t)+0.01*s^3,s,-1,2,t,0.5,1.5) | ref=patch color=#ffffff'
        lines = FUNCS + [line]
        before = _pts(line, lines)
        report = compact.compact_lines(lines, {'patch': dict(tol=0.0005, gain=100)}, log=lambda s: None)
        self.assertTrue(report[0][2], report)
        after = _pts(lines[-1], lines)
        # (not when it would cost more to open than before: a short formula is left as it is)
        self.assertIsNone(compact.compact_lines([line] + FUNCS, {'patch': dict(tol=0.0005)}, log=lambda s: None)[0][2])
        # the same points (each parameter runs over its range in the same direction, so Arch's grid and the side it
        # lights are the same), within the tolerance
        self.assertLess(float(np.max(np.linalg.norm(after - before, axis=-1))), 0.0005)
        self.assertNotIn('^', lines[-1].split(' | ')[0])
        self.assertNotIn('bump', lines[-1])
        self.assertTrue(lines[-1].endswith('| ref=patch color=#ffffff'))

    def test_surface_with_a_hole_is_kept(self):
        line = '(s,t,sqrt(0.25-s^2-t^2),s,-0.6,0.6,t,-0.6,0.6) | ref=dome color=#ffffff'
        lines = [line]
        report = compact.compact_lines(lines, {'dome': dict(tol=0.001, gain=100)}, log=lambda s: None)
        self.assertIsNone(report[0][2])
        self.assertEqual(lines[0], line)

    def test_creased_surface_is_kept_rather_than_rounded(self):
        line = '(s,t,0.3*abs(s-0.1)+0.05*t,s,-1,1,t,-1,1) | ref=crease color=#ffffff'
        lines = [line]
        report = compact.compact_lines(lines, {'crease': dict(tol=0.0003, gain=100)}, log=lambda s: None)
        self.assertIsNone(report[0][2])
        self.assertEqual(lines[0], line)

    def test_band_with_zigzag_edges_keeps_its_zigzag(self):
        # a band across a smooth sheet G(w, T), its edges zigzagging: T = 0.5 + 0.1 q (0.3 + 0.7 |zz(5 w)|)
        T = '(0.5+0.1*q*(0.3+0.7*abs(zz(5*w+0.4))))'
        line = f'(cos(w)*(1+0.2*bump({T},w)),sin(w)*(1+0.2*bump({T},w)),{T}+0.1*w^2,q,-1,1,w,0.2,2.9) | ref=band color=#ffffff'
        lines = FUNCS + [line]
        before = _pts(line, lines, 33)
        report = compact.compact_lines(lines, {'band': dict(sub=(T, 0.5, 0.1, 'w'), tol=0.0005, gain=100)}, log=lambda s: None)
        self.assertTrue(report[0][2], report)
        after = _pts(lines[-1], lines, 33)
        self.assertLess(float(np.max(np.linalg.norm(after - before, axis=-1))), 0.0005)
        self.assertIn('zz(', lines[-1])
        self.assertNotIn('bump', lines[-1])


if __name__ == '__main__':
    unittest.main()
