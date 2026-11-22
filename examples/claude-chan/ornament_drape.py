"""Lay the hair ornament's hanging parts (the cream ribbons, the black tails and the gold tassel) on the hair.

The hi-res portrait fixes where they are seen from its viewpoint, not how far back they hang. So every point of them
slides along the portrait's line of sight onto what lies behind it there, GAP above it: the ribbons onto the hair, the
tails onto the hair and the ribbons, the tassel (one piece, hanging straight down) onto all of these. Near the bow's
knot they keep their place, and below it they come down onto the hair no more steeply than SLOPE.

    python3 ornament_drape.py            # print the new ORN_TAILS, ORN_RIBBONS and ORN_TASSEL_TOP
    python3 ornament_drape.py --write    # and rewrite them in generate.py

Run it after changing the hair under the ornament (about a minute: it runs the generator for the head, hair, ornament,
earrings, body and capelet, and samples them into a depth map along the portrait's view)."""
import contextlib
import io
import math
import os
import re
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIEW_AZ, VIEW_EL = -92, 12          # the portrait's viewpoint (CLAUDE.md, "Matching the hi-res portrait")
GAP_RIBBON, GAP_TAIL, GAP_TASSEL = 0.022, 0.018, 0.018      # above what lies behind (world units)
SLOPE = 1.0                         # how much a part may come back toward the hair per unit of its length
KEEP = {'ribbon': 0.12, 'tail': 0.2}   # the part of each from the knot that keeps its place (fraction of its length)
DEG = 5                             # degree of the Bezier written for each


def run_generator():
    sys.path.insert(0, HERE)
    argv, cwd = sys.argv, os.getcwd()
    sys.argv = ['generate.py', '--parts', 'head,hair,ornament,earrings,body,cape', '--prec', '3', '-o', os.devnull]
    os.chdir(HERE)
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(os.path.join(HERE, 'generate.py'), run_name='__main__')
    finally:
        sys.argv = argv
        os.chdir(cwd)


G = run_generator()
import sheetfit                     # noqa: E402  (after the generator, which puts HERE on the path)

sc, T = G['sc'], G['HEAD_TURN']
TURN = (T['pivot'], T['axis'], T['angle'], None, T['scale'])


def to_world(P):
    return np.array(sc.turn_point(tuple(float(c) for c in P), TURN))


O_ = to_world((0, 0, 0))
B_ = np.array([to_world(e) - O_ for e in np.eye(3)]).T
B_INV = np.linalg.inv(B_)


def to_head(Pw):
    return B_INV @ (np.asarray(Pw) - O_)


az, el = math.radians(VIEW_AZ), math.radians(VIEW_EL)
D = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])     # toward the viewer
E1 = np.cross((0, 0, 1), D); E1 /= np.linalg.norm(E1)
E2 = np.cross(D, E1)
RES, U0, V0, NU, NV = 0.006, 0.2, 9.0, 260, 560     # the depth map: across (E1) and up (E2) the ornament's side
DEPTH = np.full((NU, NV), -np.inf)
ORNAMENT = ('hairFlower', 'hairBow', 'hairTail', 'hairRibbon', 'hairTassel')


def cells(P):
    P = np.atleast_2d(P)
    return np.floor((P @ E1 - U0) / RES).astype(int), np.floor((P @ E2 - V0) / RES).astype(int)


def splat(P):
    iu, iv = cells(P)
    m = (iu >= 0) & (iv >= 0) & (iu < NU) & (iv < NV)
    np.maximum.at(DEPTH, (iu[m], iv[m]), P[m] @ D)


def grid(f, lim, nu, nv):
    A, B = np.meshgrid(np.linspace(lim[0], lim[1], nu), np.linspace(lim[2], lim[3], nv), indexing='ij')
    return np.stack([np.broadcast_to(np.asarray(c, float), A.shape) for c in f(A, B)], -1)


def splat_scene():
    lo, hi = np.array([U0, V0]), np.array([U0 + NU * RES, V0 + NV * RES])
    for name, _, f, lim in sheetfit.surfaces_from_lines(sc.out):
        if name.startswith(ORNAMENT):
            continue
        Q = grid(f, lim, 24, 24)
        if not np.isfinite(Q).all(-1).any():
            continue
        q = np.stack([Q @ E1, Q @ E2], -1)
        if (np.nanmax(q.reshape(-1, 2), 0) < lo).any() or (np.nanmin(q.reshape(-1, 2), 0) > hi).any():
            continue
        su = np.nanmax(np.linalg.norm(np.diff(q, axis=0), axis=-1))
        sv = np.nanmax(np.linalg.norm(np.diff(q, axis=1), axis=-1))
        nu, nv = int(np.clip(su / RES * 48, 8, 3000)), int(np.clip(sv / RES * 48, 8, 3000))
        while nu * nv > 6e6:
            nu, nv = int(nu * 0.8), int(nv * 0.8)
        P = grid(f, lim, nu, nv).reshape(-1, 3)
        splat(P[np.isfinite(P).all(1)])


def front():
    """What lies frontmost in each cell, one-cell holes closed."""
    Dp = np.pad(DEPTH, 1, constant_values=-np.inf)
    return np.max([Dp[1 + a:1 + a + NU, 1 + b:1 + b + NV] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)


def bernstein(n, s):
    return np.array([math.comb(n, i) * s ** i * (1 - s) ** (n - i) for i in range(n + 1)]).T


def bezier(pts, s):
    return bernstein(len(pts) - 1, s) @ np.asarray(pts, float)


def drape(pts, half_width, gap, keep, F, n=40):
    """A ribbon (head-frame Bezier control points, its half-width) laid on what lies behind it: new control points,
    the shifts along the view, the least clearance of the result and its fit error."""
    pts = np.asarray(pts, float)
    c_ = np.array(G['unit'](G['view']))
    wdir = np.cross(c_, pts[-1] - pts[0]); wdir /= np.linalg.norm(wdir)
    s = np.linspace(0, 1, n)
    C = bezier(pts, s)
    across = np.linspace(-1, 1, 7)

    def clearance_need(P):
        Pw = np.array([to_world(P + a * half_width * wdir) for a in across])
        iu, iv = cells(Pw)
        ok = (iu >= 0) & (iv >= 0) & (iu < NU) & (iv < NV)
        d = F[iu[ok], iv[ok]] + gap - Pw[ok] @ D
        d = d[np.isfinite(d)]
        return d.max() if len(d) else -np.inf

    need = np.array([clearance_need(C[i]) for i in range(n)])
    need[~np.isfinite(need)] = -1.0
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff([to_world(c) for c in C], axis=0), axis=1))]
    sh = np.max(need[None, :] - SLOPE * np.abs(arc[:, None] - arc[None, :]), axis=1)    # at least what is needed
    for _ in range(4):
        sh = np.maximum(sh, np.convolve(np.pad(sh, 2, mode='edge'), np.ones(5) / 5, mode='valid'))
    w = np.clip(s / keep, 0, 1)
    sh = sh * w * w * (3 - 2 * w)
    Q = np.array([to_head(to_world(C[i]) + sh[i] * D) for i in range(n)])
    Bm = bernstein(DEG, s)
    inner, *_ = np.linalg.lstsq(Bm[:, 1:-1], Q - np.outer(Bm[:, 0], Q[0]) - np.outer(Bm[:, -1], Q[-1]), rcond=None)
    ctrl = np.vstack([Q[0], inner, Q[-1]]).round(3)
    s2 = np.linspace(keep, 1, 100)
    worst = min(gap - clearance_need(P) for P in bezier(ctrl, s2))
    return ctrl, sh, worst, np.abs(bezier(ctrl, s) - Q).max(), wdir


def splat_ribbon(ctrl, wdir, half_width):
    F_ = bezier(ctrl, np.linspace(0, 1, 300))
    splat(np.array([to_world(P + a * half_width * wdir) for P in F_ for a in np.linspace(-1, 1, 25)]))


def literal(pts):
    return '(' + ', '.join('(' + ', '.join(f'{c:g}' for c in p) + ')' for p in pts) + ')'


def main():
    splat_scene()
    report = []
    ribbons = []
    for k, pts in enumerate(G['ORN_RIBBONS']):         # behind the tails: first, each on the hair alone
        ctrl, sh, worst, fe, wdir = drape(pts, 0.085, GAP_RIBBON, KEEP['ribbon'], front())
        ribbons.append((ctrl, wdir))
        report.append(f'ribbon {k + 1}: moved {sh.min():+.3f} to {sh.max():+.3f} along the view; clearance >= {worst:.3f}; fit {fe:.4f}')
    for ctrl, wdir in ribbons:
        splat_ribbon(ctrl, wdir, 0.085)
    ribbons = [ctrl for ctrl, _ in ribbons]
    tails = []
    for k, (pts, w_) in enumerate(G['ORN_TAILS']):
        ctrl, sh, worst, fe, wdir = drape(pts, w_ * 1.18, GAP_TAIL, KEEP['tail'], front())     # with the gold rim
        tails.append((ctrl, w_))
        report.append(f'tail {k + 1}: moved {sh.min():+.3f} to {sh.max():+.3f} along the view; clearance >= {worst:.3f}; fit {fe:.4f}')
        splat_ribbon(ctrl, wdir, w_ * 1.18)
    F = front()
    top0 = np.array(G['ORN_TASSEL_TOP'], float)
    L, rr = 0.62, 0.043 * G['HEAD_SCALE'] * 1.12        # as tassel() draws it: its length and widest radius
    need = -np.inf
    for z in np.linspace(0, 0.25 * L + 0.09 + 0.7 * L, 80):
        for a in np.linspace(-1, 1, 7):
            P = top0 + (0, 0, -z) + a * rr * E1
            iu, iv = cells(P)
            if 0 <= iu[0] < NU and 0 <= iv[0] < NV and np.isfinite(F[iu[0], iv[0]]):
                need = max(need, F[iu[0], iv[0]] + GAP_TASSEL + rr * math.sqrt(1 - a * a) - P @ D)
    top = (top0 + (need if np.isfinite(need) else 0) * D).round(4)
    report.append(f'tassel: moved {need:+.3f} along the view')
    text = (f"ORN_TAILS = (({literal(tails[0][0])}, {tails[0][1]:g}),\n"
            f"             ({literal(tails[1][0])}, {tails[1][1]:g}))\n"
            f"ORN_RIBBONS = ({literal(ribbons[0])},\n"
            f"               {literal(ribbons[1])})\n"
            f"ORN_TASSEL_TOP = ({', '.join(f'{c:g}' for c in top)})\n")
    print('\n'.join(report), file=sys.stderr)
    print(text, end='')
    if '--write' in sys.argv:
        path = os.path.join(HERE, 'generate.py')
        src = open(path, encoding='utf-8').read()
        new, n = re.subn(r"ORN_TAILS = .*?\nORN_TASSEL_TOP = \([^)]*\)\n", lambda m: text, src, count=1, flags=re.S)
        if n != 1:
            sys.exit('ornament_drape: ORN_TAILS ... ORN_TASSEL_TOP not found in generate.py')
        open(path, 'w', encoding='utf-8').write(new)
        print('rewrote them in generate.py', file=sys.stderr)


if __name__ == '__main__':
    main()
