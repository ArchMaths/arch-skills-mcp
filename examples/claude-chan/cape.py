# Capelet (run inside generate.py): the user's Blender capelet, made after their reference drawing of it
# (clients/capelet/capelet.py, in metres), fitted to Claude Chan at about 4.6 units a metre, its front opened round her
# ascot and jabot (the user's choice: closed as drawn, it hid them).
#
# A stand collar, open at the front in a V; shoulders that slope gently to a corner and then flare out in a bell to
# pointed tips over the arms; the back hanging longest (from the front its beige lining shows under the side flares);
# godet folds that deepen toward the hem, the front ones running diagonally from beside the collar down to a step in the
# hem; front panels whose edges pass beside the brooch and round the jabot, then cut out to a point (the drawing's notch)
# and back to a pointed tab; double gold lines along the front edges, the hem and the collar; one orange sunburst with a
# gold button on her left front panel.
#
# The shape is designed here in numpy (ck_surface: u across the cloth, -1 at her right front edge .. 1 at her left; s
# down it, 0 at the neck seam .. 1 at the hem), every kink of it on a piece boundary (the creases, the sides, the notch's
# rows), and each piece is written for Arch as a short polynomial in its own parameters (within CK_TOL).
import sys
import numpy as np
from numpy.polynomial import chebyshev as _ck_ch
import compact as _ck_cmp

# ---------------------------------------------------------------------------------------------------
#  The design
# ---------------------------------------------------------------------------------------------------
CP_Y0 = 0.02                     # the body's centre line (y)
CK_ZN = 10.62                    # neck seam at the back
CK_NECK = (0.33, 0.305, 0.305)   # neck seam: half-width, depth behind, depth in front
CK_NECK_DROP = 0.16              # the seam dips this much at the centre front
CK_SH = (0.81, 0.52, 0.52, 2.8)  # the shoulder corner (superellipse): half-width, behind, in front, exponent
CK_SH_DROP = (0.156, 0.46)       # its drop below CK_ZN at the sides, and in front and behind
CK_ROUND = (0.22, 0.42)          # how far the shoulder's Bezier handles reach back from the corner: sides, front/back
CK_HEM = (1.46, 1.12, 0.72)      # the hem (superellipse): half-width, behind, in front
CK_HEM_N = (1.7, 2.4)            # its exponent behind (pointed sides) and in front
CK_DROP = (0.975, 1.56, 1.14, 1.37)   # hem below CK_ZN: sides, centre back, the step (front crease), the front tips
# the godets' creases (angle from the back, her left positive): at the hem and at the collar seam
CK_CREASE_HEM = [-1.64, -1.02, -0.37, 0.34, 1.05, 1.63]
CK_CREASE_NECK = [-1.72, -1.10, -0.37, 0.34, 1.12, 1.71]
CK_FOLD = (0.12, 0.055, 0.025, 0.032)    # godet depth, on the front panels, secondary folds, hem lift on a crest
# front edge, |x| by height: beside the brooch and out round the jabot (clearing the ascot and jabot by 0.05), then the
# drawing's notch (out to a point and back) and the pointed tab
CK_EDGE_XZ = [(10.535, 0.25), (10.45, 0.29), (10.35, 0.36), (10.25, 0.45), (10.15, 0.55), (10.05, 0.63), (9.95, 0.67),
              (9.82, 0.70), (9.63, 0.93), (9.47, 0.75), (9.22, 0.77)]
CK_NOTCH_Z = (9.82, 9.63, 9.47)  # where the front edge turns (the front panels are cut at these rows)
CK_COLLAR_H = 0.14               # collar top above CK_ZN at the back
CK_COLLAR_EDGE = [(0.0, 0.25), (0.3, 0.26), (1.0, 0.325)]   # the collar's front edge: |x| by height up it
CK_TRIM = (0.024, 0.044)         # the gold lines' distance in from the edges
CK_LIFT = 0.006                  # drawn lines lie this far out from the cloth
CK_TOL = 0.0015                  # largest error of the written polynomials
CK_EMB = (1.0, 9.9, 0.19)        # the sunburst: where the front view sees its centre (x, z), and its radius
CK_LINING_COL, CK_SUN, CK_SUN_LT = '#decdb8', '#c46a38', '#dc8a55'   # (Arch lights a face-on surface by 1.3)
FOLD_LINE = '#b8ac9a'            # drawn fold lines, a soft grey-beige (the sleeves' too, arms.py)


def ck_super(phi, a, b_back, b_front, n):
    s, c = np.sin(phi), np.cos(phi)
    b = np.where(c >= 0, b_back, b_front)
    return (np.abs(s / a) ** n + np.abs(c / b) ** n) ** (-1.0 / n)


def ck_fw(phi):
    """0 at the back, 1 at the centre front, smooth."""
    return ((1 - np.cos(phi)) / 2) ** 3


def ck_neck(phi):
    r = ck_super(phi, CK_NECK[0], CK_NECK[1], CK_NECK[2], 2.0)
    return r, CK_ZN - CK_NECK_DROP * ck_fw(phi)


def ck_hem_drop(ph):
    """Hem below CK_ZN for the column whose hem is at angle ph: long at the back, short at the sides, rising from the
    sides to the step at the front crease, then down to the tips at the opening."""
    side, back, step, tip = CK_DROP
    ap = np.abs(ph)
    b = side + (back - side) * np.clip(np.cos(ap), 0, 1) ** 0.7
    c = CK_CREASE_HEM[-1]
    sd = side + (step - side) * np.clip((ap - np.pi / 2) / (c - np.pi / 2), 0, 1) ** 1.3
    fr = step + (tip - step) * np.clip((ap - c) / (CK_TIP_HEM - c), 0, 1)
    return np.where(ap <= np.pi / 2, b, np.where(ap <= c, sd, fr))


def ck_profile(phi, s, ph):
    """The drape without folds: a cubic Bezier in s from the neck seam round the shoulder corner to the hem."""
    rN, zN = ck_neck(phi)
    c2 = np.cos(phi) ** 2
    rC = ck_super(phi, *CK_SH)
    zC = CK_ZN - (CK_SH_DROP[0] + (CK_SH_DROP[1] - CK_SH_DROP[0]) * c2)
    rH = ck_super(phi, CK_HEM[0], CK_HEM[1], CK_HEM[2], np.where(np.cos(phi) >= 0, *CK_HEM_N))
    zH = CK_ZN - ck_hem_drop(ph)
    k = CK_ROUND[0] + (CK_ROUND[1] - CK_ROUND[0]) * c2
    r1, z1 = rC + k * (rN - rC), zC + k * (zN - zC)
    r2, z2 = rC + k * (rH - rC) * 0.45, zC + k * (zH - zC) * 0.45
    b0, b1, b2, b3 = (1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * s * s * (1 - s), s ** 3
    return b0 * rN + b1 * r1 + b2 * r2 + b3 * rH, b0 * zN + b1 * z1 + b2 * z2 + b3 * zH


def ck_creases(s):
    w = (1 - np.asarray(s, float))[..., None] ** 2
    return np.array(CK_CREASE_HEM) * (1 - w) + np.array(CK_CREASE_NECK) * w


def ck_fold(phi, s):
    """-1 in a crease, +1 on the crests (-cos across each sector, so smooth); and the folds' depth there."""
    C = ck_creases(s)
    k = (phi[..., None] > C).sum(-1)
    n = C.shape[-1]
    inner = (k > 0) & (k < n)
    ka = np.clip(k - 1, 0, n - 2)
    ca = np.take_along_axis(C, ka[..., None], -1)[..., 0]
    cb = np.take_along_axis(C, ka[..., None] + 1, -1)[..., 0]
    F_in = -np.cos(2 * np.pi * (phi - ca) / (cb - ca))
    d = np.abs(np.where(phi < 0, C[..., 0] - phi, phi - C[..., -1]))
    g = np.exp(-(d / 0.2) ** 2)
    # the folds' depth: the godets' in the creases and on the back, easing to the front panels' own past the crease
    amp = np.where(inner, CK_FOLD[0], CK_FOLD[1] + (CK_FOLD[0] - CK_FOLD[1]) * g)
    return np.where(inner, F_in, 1 - 2 * g), amp


def ck_point(phi, s, ph):
    """The cloth's point at angle phi (from the back) and level s, in the column whose hem is at angle ph."""
    phi, s, ph = np.broadcast_arrays(np.asarray(phi, float), np.asarray(s, float), np.asarray(ph, float))
    r, z = ck_profile(phi, s, ph)
    F, amp = ck_fold(phi, s)
    r = r + amp * F * s * s * (3 - 2 * s)
    r = r + CK_FOLD[2] * np.cos(13 * phi + 1.3 * np.sin(2 * phi) + 0.4) * (1 - np.exp(-(s / 0.13) ** 2))
    z = z - CK_FOLD[3] * F * s ** 4
    return np.stack([r * np.sin(phi), CP_Y0 + r * np.cos(phi), z], -1)


def _ck_delta(s):
    """Half-gap angle of the front opening at level s (solved below)."""
    s = np.asarray(s, float)
    return np.where(s <= CK_SPLIT[0], np.polyval(CK_EDGE_C, s), np.interp(s, CK_SPLIT + (1.0,), CK_NOTCH_D))


def _ck_edge_solve(s, tip):
    """The edge's angle at levels s: where the edge column's point has the |x| that CK_EDGE_XZ gives at its height."""
    zs, xs = np.array(CK_EDGE_XZ)[::-1].T
    lo, hi = np.full(np.shape(s), 0.3), np.full(np.shape(s), 1.5)
    for _ in range(40):
        mid = (lo + hi) / 2
        P = ck_point(np.pi - mid, s, np.pi - tip)
        big = P[..., 0] > np.interp(P[..., 2], zs, xs)
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return (lo + hi) / 2


# solve the edge: the tip, then the rows where the edge turns, then the angle down it (a smooth curve above the notch,
# straight between its corners), a few rounds since each depends on the others
CK_TIP_HEM, CK_SPLIT, CK_EDGE_C, CK_NOTCH_D = np.pi - 1.0, (0.55, 0.7, 0.82), np.array([1.0]), (1.0, 1.0, 1.0, 1.0)
for _ck_round in range(4):
    _ck_t = 1.0
    for _ck_k in range(30):
        _ck_t = float(_ck_edge_solve(np.array(1.0), _ck_t))
    CK_TIP_HEM = np.pi - _ck_t
    _ck_s = np.linspace(0, 1, 401)
    _ck_z = ck_point(np.pi - _ck_delta(_ck_s), _ck_s, CK_TIP_HEM)[:, 2]
    CK_SPLIT = tuple(float(np.interp(-z_, -_ck_z, _ck_s)) for z_ in CK_NOTCH_Z)
    _ck_st = np.linspace(0, CK_SPLIT[0], 60)
    CK_EDGE_C = np.polyfit(_ck_st, _ck_edge_solve(_ck_st, _ck_t), 6)
    _ck_d = _ck_edge_solve(np.array(CK_SPLIT[1:]), _ck_t)
    CK_NOTCH_D = (float(np.polyval(CK_EDGE_C, CK_SPLIT[0])), float(_ck_d[0]), float(_ck_d[1]), _ck_t)

# the cloth's columns (u): the creases, the sides and the front edges run along fixed u
CK_KNOT_HEM = np.r_[-CK_TIP_HEM, CK_CREASE_HEM[0], -np.pi / 2, CK_CREASE_HEM[1:-1], np.pi / 2, CK_CREASE_HEM[-1], CK_TIP_HEM]
CK_UK = CK_KNOT_HEM / CK_TIP_HEM
CK_CREASE_U = [CK_UK[k] for k in (1, 3, 4, 5, 6, 8)]


def ck_knots(s):
    s = np.asarray(s, float)
    e = (np.pi - _ck_delta(s))[..., None]
    C = ck_creases(s)
    side = np.full(np.shape(s) + (1,), np.pi / 2)
    return np.concatenate([-e, C[..., :1], -side, C[..., 1:-1], side, C[..., -1:], e], -1)


def ck_phi(u, s):
    u, s = np.broadcast_arrays(np.asarray(u, float), np.asarray(s, float))
    K = ck_knots(s)
    k = np.clip(np.searchsorted(CK_UK, u, side='right') - 1, 0, len(CK_UK) - 2)
    ua, ub = CK_UK[k], CK_UK[k + 1]
    pa = np.take_along_axis(K, k[..., None], -1)[..., 0]
    pb = np.take_along_axis(K, k[..., None] + 1, -1)[..., 0]
    return pa + (u - ua) / (ub - ua) * (pb - pa)


def ck_surface(u, s):
    """The capelet's outside at cloth position u (-1 her right front edge .. 1 her left) and level s (0 neck .. 1 hem)."""
    return ck_point(ck_phi(u, s), s, ck_phi(u, np.ones_like(np.asarray(s, float))))


CK_COLLAR_C = np.polyfit(np.array(CK_COLLAR_EDGE)[:, 0], np.array(CK_COLLAR_EDGE)[:, 1], 2)


def ck_collar_point(phi, v):
    rN, zN = ck_neck(phi)
    r = rN * (1 + 0.03 * v) + 0.01 * np.sin(np.pi * v)
    zt = CK_ZN + CK_COLLAR_H - 0.02 * ck_fw(phi)
    return np.stack([r * np.sin(phi), CP_Y0 + r * np.cos(phi), zN + v * (zt - zN)], -1)


def ck_collar(u, v):
    """The collar's outside at u (-1 her right front edge .. 1 her left) and v (0 the neck seam .. 1 its top)."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    g = np.polyval(CK_COLLAR_C, v)
    lo, hi = np.full(np.shape(v), 0.05), np.full(np.shape(v), 1.55)
    for _ in range(44):
        mid = (lo + hi) / 2
        big = ck_collar_point(np.pi - mid, v)[..., 0] > g
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return ck_collar_point(u * (np.pi - (lo + hi) / 2), v)


def ck_normal(f, U, S, h=1e-4):
    """Unit normal of f(u, s) pointing away from the body's axis."""
    Pu = (f(U + h, S) - f(U - h, S)) / (2 * h)
    Ps = (f(U, S + h) - f(U, S - h)) / (2 * h)
    n = np.cross(Pu, Ps)
    n /= np.linalg.norm(n, axis=-1)[..., None]
    P = f(U, S)
    out = np.mean(np.sum(n[..., :2] * (P[..., :2] - np.array([0, CP_Y0])), -1))    # one side for all the points
    return n if out >= 0 else -n


def cape_samples(nu=1400, ns=320):
    """The capelet's outside sampled on a grid (x, y, z arrays): for the hair, which hangs on it."""
    U, S = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(0, 1, ns), indexing='ij')
    P = ck_surface(U, S)
    return P[..., 0], P[..., 1], P[..., 2]


# the capelet's reach for the hair: largest distance from the axis (0, 0.03) by angle from the back and height
def _ck_env_table():
    X, Y, Z = cape_samples(600, 160)
    Uc, Vc = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(0, 1, 12), indexing='ij')
    P = np.concatenate([np.stack([X, Y, Z], -1).reshape(-1, 3), ck_collar(Uc, Vc).reshape(-1, 3)])
    psi = np.abs(np.arctan2(P[:, 0], P[:, 1] - 0.03))
    r = np.hypot(P[:, 0], P[:, 1] - 0.03)
    zs, ps = np.arange(8.7, 10.9, 0.04), np.linspace(0, np.pi, 61)
    T = np.zeros((len(ps), len(zs)))
    iz = np.clip(np.round((P[:, 2] - zs[0]) / 0.04).astype(int), 0, len(zs) - 1)
    ip = np.clip(np.round(psi / (np.pi / 60)).astype(int), 0, 60)
    np.maximum.at(T, (ip, iz), r)
    Tp = np.pad(T, ((0, 0), (1, 1)))
    T = np.maximum.reduce([Tp[:, :-2], Tp[:, 1:-1], Tp[:, 2:]])
    return zs, ps, T


CAPE_ENV = _ck_env_table()


def cape_env(psi, z):
    zs, ps, T = CAPE_ENV
    if z < zs[0] or z > zs[-1]:
        return 0.0
    i = int(round(abs(psi) / (np.pi / 60)))
    j = int(round((z - zs[0]) / 0.04))
    return float(T[min(i, 60), min(j, len(zs) - 1)])


# ---------------------------------------------------------------------------------------------------
#  Polynomials for Arch
# ---------------------------------------------------------------------------------------------------
_CK_A = np.cos(np.pi * (np.arange(65) + 0.5) / 65)[::-1]       # Chebyshev nodes: where the fits sample
_CK_G = np.linspace(-1, 1, 63)                                  # where they are checked (Arch's mesh and between)


def ck_pieces():
    """(u0, u1, s0, s1) of each piece: the sectors between the creases and the sides, the front panels cut at the notch."""
    out = []
    for k in range(len(CK_UK) - 1):
        cuts = [0.0, *CK_SPLIT, 1.0] if k in (0, len(CK_UK) - 2) else [0.0, 1.0]
        out += [(float(CK_UK[k]), float(CK_UK[k + 1]), a, b) for a, b in zip(cuts[:-1], cuts[1:])]
    return out


def ck_fit(f, u0, u1, s0, s1, tol=CK_TOL, deg=(20, 20)):
    """f(u, s) over the piece as a polynomial in u and v (each -1..1), the cheapest that is within tol where checked:
    Arch expressions (Horner form), the largest error, the degrees."""
    mu, hu, ms, hs = (u0 + u1) / 2, (u1 - u0) / 2, (s0 + s1) / 2, (s1 - s0) / 2
    U, S = np.meshgrid(mu + hu * _CK_A, ms + hs * _CK_A, indexing='ij')
    F = f(U, S)
    R = f(*np.meshgrid(mu + hu * _CK_G, ms + hs * _CK_G, indexing='ij'))

    def err(m, k):
        C = _ck_cmp._cmp_fit_grid(F, _CK_A, _CK_A, m, k)
        E = np.stack([_ck_ch.chebgrid2d(_CK_G, _CK_G, C[c]) for c in range(3)], -1)
        return float(np.max(np.linalg.norm(E - R, axis=-1))), C
    M, K = deg
    with np.errstate(all='ignore'):
        nm = next((m for m in range(2, M + 1) if err(m, K)[0] < 0.7 * tol), M)
        nk = next((k for k in range(2, K + 1) if err(M, k)[0] < 0.7 * tol), K)
        best = None
        for m, k in sorted(((m, k) for m in range(nm, M + 1) for k in range(nk, K + 1)), key=lambda mk: (mk[0] + 1) * (mk[1] + 1)):
            e, C = err(m, k)
            if e < tol:
                best = (e, C, m, k)
                break
        if best is None:
            best = (*err(M, K), M, K)
    e, C, m, k = best
    Pw = [_ck_cmp._cmp_to_power(C[c]) for c in range(3)]
    GU, GV = np.meshgrid(_CK_G, _CK_G, indexing='ij')
    W = np.stack([np.polynomial.polynomial.polyval2d(GU, GV, Pw[c]) for c in range(3)], -1)
    e = float(np.max(np.linalg.norm(W - R, axis=-1)))         # as written (rounded to 1e-8)
    return tuple(_ck_cmp._cmp_formula(Pw[c], 'u', 'v') for c in range(3)), e, (m, k)


def ck_fit1(P, tol=0.002, deg=16):
    """Points along a curve (n, 3), evenly spaced, as a polynomial in t -1..1: power-basis coefficients per coordinate."""
    t = np.linspace(-1, 1, len(P))
    for d in range(2, deg + 1):
        C = [np.polynomial.chebyshev.chebfit(t, P[:, c], d) for c in range(3)]
        E = np.stack([np.polynomial.chebyshev.chebval(t, C[c]) for c in range(3)], -1)
        if np.max(np.linalg.norm(E - P, axis=-1)) < tol:
            break
    return [np.round(np.polynomial.chebyshev.cheb2poly(C[c]), 8) for c in range(3)]


def ck_curve_expr(pieces, var='t'):
    """Several polynomial pieces drawn as one curve: piece k for var in k..k+1 (nested conditions)."""
    out = []
    for c in range(3):
        e = None
        for k in reversed(range(len(pieces))):
            h = _ck_cmp._cmp_horner(pieces[k][c], f"(2*{var}-{2 * k + 1})")
            e = h if e is None else f"({var}<{k + 1}?{h}:{e})"
        out.append(e)
    return tuple(out)


def _ck_resample(P, n=90):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, c]) for c in range(3)], -1)


def _ck_offset(f, Q, d, inward):
    """Points d in from a boundary through the parameter points Q (n, 2), on the surface f: along the surface's
    tangent plane, square to the boundary, toward the side where the parameters move by `inward` (a 2-vector)."""
    h = 1e-4
    U, S = Q[:, 0], Q[:, 1]
    P = f(U, S)
    Pu = (f(U + h, S) - f(U - h, S)) / (2 * h)
    Ps = (f(U, S + h) - f(U, S - h)) / (2 * h)
    T = np.gradient(P, axis=0)
    N = np.cross(Pu, Ps)
    m = np.cross(N, T)
    m /= np.linalg.norm(m, axis=-1)[..., None]
    out = []
    for i in range(len(Q)):
        J = np.stack([Pu[i], Ps[i]], 1)
        duv = np.linalg.lstsq(J, d * m[i], rcond=None)[0]
        if np.dot(duv, inward) < 0:
            duv = -duv
        out.append(Q[i] + duv)
    return np.array(out)


def _ck_meet(A, B):
    """Trim two parameter-space polylines (A ends where B starts) where they cross (at a corner that turns toward the
    lines' side they overlap). Where they do not cross (the notch's point, smooth joins) they are left as they are and
    the curve joins them straight. (Extended to meet, in parameter space, they met outside the cloth at the notch:
    the edge is straight in u and s there and turns only in space.)"""
    Ae, Be = A, B
    na, nb = len(Ae), len(Be)
    best = None
    for i in range(max(0, na - 60), na - 1):
        p, r = Ae[i], Ae[i + 1] - Ae[i]
        for j in range(0, min(60, nb - 1)):
            q, s_ = Be[j], Be[j + 1] - Be[j]
            den = r[0] * s_[1] - r[1] * s_[0]
            if abs(den) < 1e-14:
                continue
            t = ((q - p)[0] * s_[1] - (q - p)[1] * s_[0]) / den
            w = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
            if 0 <= t <= 1 and 0 <= w <= 1:
                best = (i, j, p + t * r)
    if best is None:
        return A, B
    i, j, X = best
    return np.vstack([Ae[:i + 1], X]), np.vstack([X, Be[j + 1:]])


def ck_inset_path(f, segs, d):
    """The line d in from a chain of boundary segments (each (n, 2) parameter points with the inward direction),
    mitred at the corners: a list of parameter polylines, one a segment."""
    L = [_ck_offset(f, Q, d, inw) for Q, inw in segs]
    for k in range(len(L) - 1):
        L[k], L[k + 1] = _ck_meet(L[k], L[k + 1])
    return L


def ck_trim_segments():
    """The cloth's trimmed boundary as parameter segments, in order: down her right front edge (cut at the notch's
    corners), along the hem (cut at the side tips and the creases), up her left front edge."""
    n = 120
    cuts = [0.0, *CK_SPLIT, 1.0]
    right = [(np.c_[np.full(n, -1.0), np.linspace(a, b, n)], np.array([1.0, 0])) for a, b in zip(cuts[:-1], cuts[1:])]
    left = [(np.c_[np.full(n, 1.0), np.linspace(b, a, n)], np.array([-1.0, 0])) for a, b in zip(cuts[:-1], cuts[1:])][::-1]
    hk = [CK_UK[0], *CK_UK[2:-2], CK_UK[-1]]          # the front creases are smooth on the hem: no cut there
    hem = [(np.c_[np.linspace(a, b, n), np.ones(n)], np.array([0, -1.0])) for a, b in zip(hk[:-1], hk[1:])]
    return right, hem, left


# ---------------------------------------------------------------------------------------------------
#  Arch
# ---------------------------------------------------------------------------------------------------
def _ck_lift(f, Q, d=CK_LIFT):
    """3D points at the parameter points Q, lifted d off the cloth."""
    return f(Q[:, 0], Q[:, 1]) + d * ck_normal(f, Q[:, 0], Q[:, 1])


CK_LINING_K = 0.988             # the lining: the cloth drawn toward the body's axis by this and lowered a little


def _ck_sheet(pieces):
    """The cloth, each piece a polynomial; and its lining, the same polynomials scaled toward the axis (exact)."""
    worst, terms = 0.0, 0
    for i, (u0, u1, s0, s1) in enumerate(pieces):
        (X, Y, Z), e, (m, k) = ck_fit(ck_surface, u0, u1, s0, s1)
        worst, terms = max(worst, e), terms + (m + 1) * (k + 1)
        n = ck_normal(ck_surface, np.array((u0 + u1) / 2), np.array((s0 + s1) / 2))
        sc.surface((X, Y, Z), ('u', -1, 1), ('v', -1, 1), CREAM, f'cape{i + 1}', ('dir', tuple(n)))
        k_ = fmt(CK_LINING_K)
        lin = (f"{k_}*({X})", f"{fmt(CP_Y0 * (1 - CK_LINING_K))}+{k_}*({Y})", f"{Z}-0.006")
        sc.surface(lin, ('u', -1, 1), ('v', -1, 1), CK_LINING_COL, f'capeLining{i + 1}', ('dir', tuple(-n)))
    return worst, terms


def _ck_emit():
    sc.blank()
    sc.note('Capelet: stand collar, flared bell, open front with notched edges and pointed tabs, gold trim, sunburst')
    sc.comment("""The capelet after the user's Blender model of it: each piece of the cloth (between its creases and sides;
the front panels cut where the edge turns) is a polynomial in its own u (across) and v (down), fitted in cape.py.""")
    pieces = ck_pieces()
    e1, n1 = _ck_sheet(pieces)
    # the collar: one piece, its lining inside it
    col_f = lambda u, v: ck_collar(u, (v + 1) / 2)
    ex, e3, _ = ck_fit(col_f, -1, 1, -1, 1)
    sc.surface(ex, ('u', -1, 1), ('v', -1, 1), CREAM, 'capeCollar', ('tube', 'v'))
    col_l = lambda u, v: ck_collar(u, (v + 1) / 2) - 0.01 * ck_normal(col_f, u, v)
    ex, e4, _ = ck_fit(col_l, -1, 1, -1, 1)
    sc.surface(ex, ('u', -1, 1), ('v', -1, 1), CK_LINING_COL, 'capeCollarLining', ('tube', 'v', 'in'))
    # double gold lines along the front edges and the hem, mitred at the corners: each line in four curves (her right
    # front edge, the hem in two halves, her left front edge), each curve a polynomial a segment
    right, hem, left = ck_trim_segments()
    nr, nh = len(right), len(hem)
    for j, d in enumerate(CK_TRIM):
        L = ck_inset_path(ck_surface, right + hem + left, d)
        hm = nr + nh // 2                               # the hem in two halves (one curve's formula would be too long)
        for tag, part in (('R', L[:nr]), ('HemR', L[nr:hm]), ('HemL', L[hm:nr + nh]), ('L', L[nr + nh:])):
            polys = [ck_fit1(_ck_resample(_ck_lift(ck_surface, Q))) for Q in part]
            sc.curve(ck_curve_expr(polys), ('t', 0, len(polys)), GOLD, f'capeTrim{tag}{j + 1}', thickness=2)
    # and round the collar: up her right front edge, along the top, down her left front edge
    n = 80
    csegs = [(np.c_[np.full(n, -1.0), np.linspace(-1, 1, n)], np.array([1.0, 0])),
             (np.c_[np.linspace(-1, 1, 3 * n), np.ones(3 * n)], np.array([0, -1.0])),
             (np.c_[np.full(n, 1.0), np.linspace(1, -1, n)], np.array([-1.0, 0]))]
    for j, d in enumerate((0.02, 0.038)):
        L = ck_inset_path(col_f, csegs, d)
        polys = [ck_fit1(_ck_resample(_ck_lift(col_f, Q, 0.005))) for Q in L]
        sc.curve(ck_curve_expr(polys), ('t', 0, len(polys)), GOLD, f'capeCollarTrim{j + 1}', thickness=2)
    # drawn fold lines down the creases, from the shoulders to just above the gold lines
    for k, uc in enumerate(CK_CREASE_U):
        s = np.linspace(0.2, 0.955, 60)
        P = _ck_lift(ck_surface, np.c_[np.full_like(s, uc), s])
        sc.curve(ck_curve_expr([ck_fit1(_ck_resample(P))]), ('t', 0, 1), FOLD_LINE, f'capeFold{k + 1}', thickness=1)
    _ck_sunburst()
    print(f"  capelet: {len(pieces)} pieces, {n1} polynomial terms a coordinate; largest error {e1:.4f}, "
          f"collar {max(e3, e4):.4f}; notch rows s {', '.join(f'{v:.3f}' for v in CK_SPLIT)}", file=sys.stderr)


def _ck_sunburst():
    """The orange sunburst on her left front panel: twenty pointed petals of two lengths in turn, a lighter stroke down
    each, a gold button. Drawn in sectors of four petals, each starting and ending between petals (one surface a ring
    drew the petals as spikes)."""
    x0, z0, R = CK_EMB
    U, S = np.meshgrid(np.linspace(CK_UK[-2] + 0.01, 1, 160), np.linspace(0, 1, 200), indexing='ij')
    P = ck_surface(U, S)
    i = np.unravel_index(np.argmin((P[..., 0] - x0) ** 2 + (P[..., 2] - z0) ** 2), U.shape)
    u0, s0 = U[i], S[i]
    n = ck_normal(ck_surface, np.array(u0), np.array(s0))
    C = tuple(ck_surface(np.array(u0), np.array(s0)) + 0.022 * n)
    fr = frame(tuple(n), (0, 0, 1))
    sc.func('ksr(r,w,sz,b,p)', 'sz*(0.2+0.8*r)*(b+(1-b)*abs(cos(10*w))^p)*(0.9+0.1*cos(10*w))')
    for li, (sz, b, p, lift, col) in enumerate(((R, 0.3, 2.2, 0.0, CK_SUN), (0.8 * R, 0.08, 14, 0.003, CK_SUN_LT))):
        L = (f"ksr(r,w,{fmt(sz)},{fmt(b)},{fmt(p)})*cos(w)", f"ksr(r,w,{fmt(sz)},{fmt(b)},{fmt(p)})*sin(w)", fmt(lift))
        for k in range(5):
            w0 = np.pi / 20 + 2 * np.pi * k / 5
            sc.surface(place(L, C, fr), ('r', 0, 1), ('w', w0, w0 + 2 * np.pi / 5), col, f'capeSun{"OI"[li]}{k + 1}', ('dir', fr[2]))
    Cc = add(C, mul(fr[2], 0.006))
    L = (f"{fmt(0.2 * R)}*sin(u)*cos(v)", f"{fmt(0.2 * R)}*sin(u)*sin(v)", f"{fmt(0.08 * R)}*cos(u)")
    sc.surface(place(L, Cc, fr), ('u', 0, 'pi'), ('v', 0, '2*pi'), GOLD, 'capeSunMid', 'closed')
    torus(Cc, fr, 0.205 * R, 0.022 * R, GOLD_DK, 'capeSunRing')


if want('cape'):
    _ck_emit()
