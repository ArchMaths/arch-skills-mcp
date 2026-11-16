# Run inside generate.py after outfit.py: all of Claude Chan's hair.
import numpy as np

ZHC = ZE + 0.42                          # height of the centre of the cranium
HDX, HDY, HDZ = 0.725, 0.815, 0.825      # the dome of the hair around the cranium
HY0 = 0.14
ZBOT = 5.3                               # where the longest locks end
TB = 0.09                                # how far down a lock eases from turning with the head to hanging straight
T0 = 1 / (1 + (ZHC - ZBOT) / (HDZ * math.pi / 2))   # part of a lock's length that lies on the dome
VF = HDZ * math.pi / 2 / T0              # how fast z falls with t below the dome

# ARM_ENV, ARM_PSI (arms.py): how far out the sleeves, cuffs and hands reach by height, and round which angles
_bk = BOOK[2]
BOOK_Z = (min(P[2] for P in _bk) - 0.05, max(P[2] for P in _bk) + 0.05)
BOOK_R = max(math.hypot(P[0], P[1] - 0.03) for P in _bk) + 0.04
BOOK_PSI = (min(math.atan2(P[0], P[1] - 0.03) for P in _bk) - 0.1, max(math.atan2(P[0], P[1] - 0.03) for P in _bk) + 0.1)


def ell_r(a, b, psi):
    return 1 / math.sqrt((math.sin(psi) / a) ** 2 + (math.cos(psi) / b) ** 2)


def envelope(psi, z):
    """Distance from the body axis to the outside of the figure at angle psi from the back."""
    r = 0.3
    r = max(r, cape_env(psi, z))                # the capelet (cape.py), standing out at the back
    if 7.6 <= z <= 9.3:
        r = max(r, ell_r(0.7, 0.55, psi))
    if z < SK_TOP - 0.2:
        th = math.pi - abs(psi)                     # angle from the front
        s = sk_s(z)
        a = (OSR(z) + 0.12) if OV_F - 0.2 < th < OV_B + 0.2 and z > 2.5 else SA(s) + 0.16
        r = max(r, a * math.hypot(math.sin(th), DK(s, th) * math.cos(th)))
        for th_b, z_b in ((OV_B, BOW_ZB), (OV_F, BOW_ZF)):     # the bows, their tails and tassels
            if abs(th - th_b) < 0.4 and z_b - 1.0 < z < z_b + 0.35:
                r = max(r, (OSR(z_b) + 0.1) * math.hypot(math.sin(th_b), DK(sk_s(z_b), th_b) * math.cos(th_b)) + 0.3)
    if ARM_PSI[0] < abs(psi) < ARM_PSI[1] and ARM_ENV[-1][0] < z < ARM_ENV[0][0]:      # arms and hands below the capelet
        r = max(r, float(np.interp(z, [e[0] for e in ARM_ENV[::-1]], [e[1] for e in ARM_ENV[::-1]])))
    if BOOK_PSI[0] < psi < BOOK_PSI[1] and BOOK_Z[0] < z < BOOK_Z[1]:   # the book in her left hand
        r = max(r, BOOK_R)
    return r


def dome(th, ph, k=1.0):
    """Point on the hair dome: th from the crown, ph from the front (+ toward her left)."""
    return (HDX * k * math.sin(th) * math.sin(ph), HY0 - HDY * k * math.sin(th) * math.cos(ph), ZHC + HDZ * k * math.cos(th))


def dome_normal(P):
    return unit((P[0] / HDX ** 2, (P[1] - HY0) / HDY ** 2, (P[2] - ZHC) / HDZ ** 2))


def interp_bezier(P):
    """Control points of the Bezier curve that passes through the points P at t = i/n."""
    n = len(P) - 1
    M = [[math.comb(n, k) * (i / n) ** k * (1 - i / n) ** (n - k) for k in range(n + 1)] for i in range(n + 1)]
    return [tuple(r) for r in np.linalg.solve(np.array(M), np.array(P, dtype=float))]


# ---------- how far out the falling hair must hang to clear the body ----------
US = np.linspace(0, 1 - T0, 300)
ZS = ZHC - VF * US
PSIS = np.linspace(0, 2.4, 40)


def clear_factor(rx, ry, z, margin):
    k = 1.0
    for p_ in PSIS:
        for _ in range(80):
            x_, y_ = k * rx * math.sin(p_), HY0 + k * ry * math.cos(p_)
            if math.hypot(x_, y_ - 0.03) >= envelope(math.atan2(x_, y_ - 0.03), z) + margin:
                break
            k *= 1.01
    return k


side_req, back_req = [], []
for z in ZS:
    rx = max(HDX, max(envelope(p_, z) for p_ in (1.45, 1.6, 1.75)) + 0.1)
    ry = max(HDY, max(envelope(p_, z) for p_ in (0.0, 0.3)) + 0.0)
    k = clear_factor(rx, ry, z, 0.08)
    flare = 0.5 * max(0, (8.0 - z) / 2.7) ** 1.3
    side_req.append(rx * k + flare)
    back_req.append(ry * k + flare * 0.8)
side_req = np.maximum.accumulate(np.array(side_req))
back_req = np.maximum.accumulate(np.array(back_req))
KNOTS = [0, 0.03, 0.06, 0.09, 0.115, 0.14, 0.165, 0.19, 0.215, 0.24, 0.27, 0.31, 0.36, 0.43, 0.52, 0.62, 0.72]


def fit_profile(req, base):
    """base + sum c_i max(0, u - k_i)^2: leaves the dome smoothly and stays outside req."""
    A = np.column_stack([np.maximum(0, US - k) ** 2 for k in KNOTS])
    ramp = np.minimum(1, US / 0.05)
    target = req - base + 0.05 * ramp
    for _ in range(80):
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        fit = A @ coef
        deficit = req - base + 0.025 * ramp - fit
        if deficit.max() <= 0:
            break
        target = target + np.maximum(deficit, 0) * 1.5
    margin = float(np.min((fit - (req - base))[US > 0.05]))
    return ''.join(f"+{fmt(c)}*max(0,t-{fmt(T0 + k)})^2" for c, k in zip(coef, KNOTS)), margin, coef


HP_X, HP_Y = fit_profile(side_req, HDX), fit_profile(back_req, HDY)    # the falling hair's ellipse (Arch text, margin, coefficients)


def _hp(t, coef):
    return sum(c * max(0.0, t - T0 - k) ** 2 for c, k in zip(coef, KNOTS))


def hrx_py(t): return HDX * math.sin(1.5708 * min(1, t / T0)) + _hp(t, HP_X[2])
def hry_py(t): return HDY * math.sin(1.5708 * min(1, t / T0)) + _hp(t, HP_Y[2])
def hdw_py(t): return 1.54 / (1.54 + _hp(t, HP_X[2] + HP_Y[2]))
def hu_py(t): return max(0.0, t - T0) / (1 - T0)
def t_of_z(z): return T0 + (ZHC - z) / VF          # down the fall, where the hair no longer turns with the head


FRONT_BLADES = []      # the locks in front of the face, for the brows drawn over them


# ---------- the two locks that fall over the chest ----------
# Traced by the user over the sheet's third expression panel and read as numbers: the strokes-only drawing
# registered onto the marked-up panel (ECC), the panel onto the sheet (template matching, 4.2x), and into model
# units by the capelet emblems for x (+-1.10, 318 panel px a unit) and the chin, brooch and emblems for z (10.96,
# 10.5, 9.67, 290 px a unit, consistent to 0.01): the panel draws the chest about 9% wider than the full-body
# front view, so the locks are placed by the emblems and the jabot. Each lock is drawn as sub-strands along the
# inner lines of the tracing: (x, z, half-width) down each centre line, from where the panel's frame cuts it
# below the jaw to the tip.
CH_STRANDS = {
    # her right: R1 thin and outermost, R2 curls in at the bottom, R3 hooks in toward the jabot, R4 a short
    # strand on top that ends in a point
    'R1': [(-0.475, 11.27, .025), (-0.48, 11.0, .025), (-0.51, 10.6, .028), (-0.60, 10.3, .03), (-0.70, 10.05, .03), (-0.80, 9.8, .03),
           (-0.875, 9.6, .028), (-0.945, 9.4, .027), (-0.99, 9.2, .03), (-1.02, 9.05, .02), (-1.04, 8.95, 0)],
    'R2': [(-0.44, 11.27, .04), (-0.45, 11.0, .045), (-0.47, 10.6, .045), (-0.565, 10.3, .05), (-0.665, 10.05, .055), (-0.74, 9.8, .06),
           (-0.81, 9.6, .045), (-0.845, 9.4, .07), (-0.86, 9.2, .08), (-0.84, 9.03, .06), (-0.77, 8.93, .04), (-0.66, 8.9, .025), (-0.54, 8.91, 0)],
    'R3': [(-0.39, 11.27, .03), (-0.39, 11.0, .03), (-0.395, 10.6, .03), (-0.42, 10.3, .03), (-0.52, 10.0, .04), (-0.62, 9.8, .05),
           (-0.70, 9.65, .04), (-0.74, 9.45, .033), (-0.735, 9.25, .03), (-0.70, 9.11, .035), (-0.62, 9.035, .03), (-0.53, 9.02, .02), (-0.44, 9.035, 0)],
    'R4': [(-0.40, 11.27, .035), (-0.40, 11.0, .035), (-0.405, 10.75, .035), (-0.43, 10.5, .035), (-0.50, 10.3, .03), (-0.56, 10.12, .025), (-0.61, 9.97, 0)],
    # her left: L1 outermost, curls in at the bottom; L2 sweeps out and curls back in; L3 innermost, curls in
    'L1': [(0.465, 11.3, .03), (0.47, 11.0, .03), (0.475, 10.6, .03), (0.51, 10.3, .03), (0.575, 10.0, .03), (0.635, 9.8, .03),
           (0.695, 9.6, .03), (0.735, 9.4, .035), (0.75, 9.2, .04), (0.76, 9.02, .03), (0.745, 8.9, .03), (0.69, 8.8, .025), (0.62, 8.75, .02), (0.54, 8.71, 0)],
    'L2': [(0.435, 11.3, .04), (0.435, 11.0, .04), (0.43, 10.6, .04), (0.45, 10.3, .04), (0.52, 10.0, .04), (0.56, 9.8, .05),
           (0.60, 9.6, .07), (0.62, 9.4, .083), (0.655, 9.2, .06), (0.69, 9.05, .045), (0.655, 8.92, .045), (0.585, 8.83, .035), (0.50, 8.785, 0)],
    'L3': [(0.40, 11.3, .025), (0.40, 11.0, .025), (0.395, 10.6, .025), (0.41, 10.3, .03), (0.465, 10.0, .025), (0.47, 9.8, .03),
           (0.48, 9.6, .04), (0.49, 9.4, .048), (0.53, 9.22, .045), (0.595, 9.05, .06), (0.54, 8.92, .045), (0.48, 8.83, .025), (0.425, 8.78, 0)],
}
# colour and thickness; they are laid on in this order, each on the body and the strands already laid
CH_LOOK = {'R2': (HAIR, 0.04), 'R1': (HAIR_LT, 0.03), 'R3': (HAIR_DK, 0.035), 'R4': (HAIR_LT, 0.032),
           'L2': (HAIR, 0.04), 'L1': (HAIR_LT, 0.03), 'L3': (HAIR_DK, 0.035)}
CH_GAP = 0.02           # between a strand's underside and the cloth it lies on
CH_ON = 0.006           # and a strand it lies on
CH_JAW = 0.12           # how far forward of the body's centre line a strand is where it comes out from under the jaw
CH_ROOT = (0.34, 0.0, 11.66)   # where a strand starts, hidden inside the head behind the cheek (|x|, y, z)
CH_DEG = 20             # degree of the polynomials (in t from -1 to 1) that Arch draws a strand with
CH_DENSE = 0.5          # the strand's mesh is (1 + CH_DENSE) / (1 - CH_DENSE) times finer at the tip than at the top


def _nat_spline(d, P, s):
    """Natural cubic spline through the rows of P at the knots d, evaluated at s."""
    n, h = len(d), np.diff(d)
    A, B = np.zeros((n, n)), np.zeros((n, P.shape[1]))
    A[0, 0] = A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1:i + 2] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        B[i] = 6 * ((P[i + 1] - P[i]) / h[i] - (P[i] - P[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, B)
    k = np.clip(np.searchsorted(d, s) - 1, 0, n - 2)
    a, b, hk = (d[k + 1] - s)[:, None], (s - d[k])[:, None], h[k][:, None]
    return (M[k] * a ** 3 + M[k + 1] * b ** 3) / (6 * hk) + (P[k] / hk - M[k] * hk / 6) * a + (P[k + 1] / hk - M[k + 1] * hk / 6) * b


# how far forward (-y) the capelet reaches, on a 0.01 grid of (x, z), from its point model (folds included)
_CG = (-1.5, 8.5, 0.01)
_cu, _cq = np.meshgrid(np.linspace(0, 1, 320), np.linspace(0, 1, 1400), indexing='ij')
_cx, _cy, _cz = kp_point(_cu, kp_q(_cu, _cq))
CAPE_FWD = np.zeros((301, 301))
_ok = (_cy < CP_Y0) & (np.abs(_cx) < 1.5) & (_cz > 8.5) & (_cz < 11.5)
np.maximum.at(CAPE_FWD, (np.round((_cx[_ok] - _CG[0]) / _CG[2]).astype(int), np.round((_cz[_ok] - _CG[1]) / _CG[2]).astype(int)), -_cy[_ok])
for _ in range(2):      # close the gaps between samples
    CAPE_FWD = np.maximum.reduce([np.roll(np.roll(CAPE_FWD, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)])


def _grid_at(G, x, z):
    i, j = np.round((np.asarray(x) - _CG[0]) / _CG[2]).astype(int), np.round((np.asarray(z) - _CG[1]) / _CG[2]).astype(int)
    ok = (i >= 0) & (i < G.shape[0]) & (j >= 0) & (j < G.shape[1])
    return np.where(ok, G[np.clip(i, 0, G.shape[0] - 1), np.clip(j, 0, G.shape[1] - 1)], 0.0)


def body_fwd(x, z):
    """How far forward (-y) the outfit reaches at (x, z) where the chest locks hang: the capelet, the blouse over
    the bust, the corset and the high collar with its ruffle."""
    f = float(_grid_at(CAPE_FWD, x, z))
    if 9.1 <= z <= 10.45:
        f = max(f, -chest_y(x, z))
    if z < 9.15 and abs(x) < corset_ab(z)[0]:
        f = max(f, -corset_front(x, z))
    if 10.4 <= z <= 11.05:
        r = 0.41 if z > 10.83 else 0.26
        if abs(x) < r:
            f = max(f, math.sqrt(r * r - x * x) - CP_Y0)
    return f


def chest_strands():
    """Each strand's centre line, side and thickness vectors as polynomials in t (-1..1), and the fit errors.
    Down the tracing the strand hangs: it lies on whatever is under it and, below the fullest point, falls
    straight (how far forward it is never decreases going down), smoothed; above the tracing it rises from under
    the jaw into the head, behind the cheek."""
    under = np.zeros((301, 301))      # how far forward the strands already laid reach
    out = {}
    for key, (col, th) in CH_LOOK.items():
        sd = 1 if key[0] == 'L' else -1
        D = np.array(CH_STRANDS[key], float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(D[:, :2], axis=0).T))]
        s = np.linspace(0, d[-1], 500)
        Q = _nat_spline(d, D, s)
        T2 = np.gradient(Q[:, :2], axis=0)
        T2 /= np.linalg.norm(T2, axis=1)[:, None]
        N2 = np.c_[-T2[:, 1], T2[:, 0]]                       # across the strand in the front view
        hw = np.maximum(Q[:, 2], 0)
        fb, fh = np.zeros(len(s)), np.zeros(len(s))
        for k in np.linspace(-1, 1, 7):
            px, pz = Q[:, 0] + k * hw * N2[:, 0], Q[:, 1] + k * hw * N2[:, 1]
            fb = np.maximum(fb, [body_fwd(a, b) for a, b in zip(px, pz)])
            fh = np.maximum(fh, _grid_at(under, px, pz))
        lo = np.maximum.accumulate(np.r_[CH_JAW, np.maximum(fb + CH_GAP, fh + CH_ON)])[1:]
        g, ker = lo.copy(), np.exp(-0.5 * (np.arange(-24, 25) / 8) ** 2)
        for _ in range(60):
            g = np.maximum(np.convolve(np.pad(g, 24, mode='edge'), ker / ker.sum(), 'valid'), lo)
        tip = np.minimum(1, (d[-1] - s) / 0.15) ** 0.6           # the thickness tapers over the last 0.15
        y = -(g + 0.3 * th * tip)
        # lay it down: the strand's front, for the strands laid on it
        for k in np.linspace(-1, 1, 9):
            px, pz = Q[:, 0] + k * hw * N2[:, 0], Q[:, 1] + k * hw * N2[:, 1]
            i, j = np.round((px - _CG[0]) / _CG[2]).astype(int), np.round((pz - _CG[1]) / _CG[2]).astype(int)
            np.maximum.at(under, (i, j), -y + th * tip * (1 - 0.8 * k * k))
        # one smooth curve, from inside the head behind the cheek, out under the jaw and down the traced part: a
        # natural cubic spline through that point and the traced part every 0.05 (with the width and thickness)
        idx = np.unique(np.r_[np.arange(0, len(s), max(1, int(0.05 / (s[1] - s[0])))), len(s) - 1])
        root = [[CH_ROOT[0] * sd, CH_ROOT[1], CH_ROOT[2]]]
        K = np.c_[np.r_[root, np.c_[Q[idx, 0], y[idx], Q[idx, 1]]], np.r_[hw[0], hw[idx]], np.r_[th, th * tip[idx]]]
        dk = np.r_[0, np.cumsum(np.linalg.norm(np.diff(K[:, :3], axis=0), axis=1))]
        Kd = _nat_spline(dk, K, np.linspace(0, dk[-1], 4000))
        P, W, TH = Kd[:, :3], np.maximum(Kd[:, 3], 0), np.maximum(Kd[:, 4], 0)
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        # resample by t, finer toward the tip, and build the side (across, in the cloth) and thickness (outward) vectors
        t = np.linspace(-1, 1, 1201)
        u = (t + 1) / 2
        st = sig[-1] * (u + CH_DENSE * u * (1 - u))
        Pt = np.c_[[np.interp(st, sig, P[:, i]) for i in range(3)]].T
        Wt, THt = np.interp(st, sig, W), np.interp(st, sig, TH)
        T = np.gradient(Pt, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None]
        n = np.array([0, -1.0, 0]) + T[:, 1:2] * T
        n /= np.linalg.norm(n, axis=1)[:, None]
        side = np.cross(T, n)
        side /= np.linalg.norm(side, axis=1)[:, None]
        polys, err = [], 0.0
        for V in (Pt, side * Wt[:, None], n * THt[:, None]):
            cs = [np.polynomial.chebyshev.Chebyshev.fit(t, V[:, i], CH_DEG).convert(kind=np.polynomial.Polynomial).coef for i in range(3)]
            cs = [np.round(c, 5) for c in cs]        # as written
            fit = np.c_[[np.polynomial.polynomial.polyval(t, c) for c in cs]].T
            err = max(err, float(np.max(np.linalg.norm(fit - V, axis=1))))
            polys.append(cs)
        out[key] = (polys, err, col, Pt, sig[-1])
    return out


# ---------- the two locks down the back ----------
# Traced by the user over the sheet's back view (a crop at 9.3x, found by template matching; the strokes-only
# drawing registered onto it by ECC) and put into model units by the back view's calibration (sheetfit.py). On
# the sheet these two locks are nearly all the hair down the middle of her back: they part below the braid,
# the corset shows between them and the capelet's back beside them, and the rest of the hair falls over the
# arms. Each lock is drawn as sub-strands along the inner lines of the tracing: (x, z, half-width) in the back
# view down each centre line, from under the braid to where its curl starts (or its tip).
BK_STRANDS = {
    # her left: BL1 and BL2 spread out at the bottom (BL2 hooks back in), BL3 and BL4 bend in S-curves and wind
    # together into a corkscrew ringlet
    'BL4': [(0.04, 11.35, .035), (0.05, 10.9, .04), (0.08, 10.3, .045), (0.18, 9.75, .05), (0.33, 9.25, .055), (0.44, 8.8, .055),
            (0.47, 8.4, .05), (0.6, 8.08, .05), (0.76, 7.9, .05), (0.6, 7.79, .05), (0.52, 7.64, .05), (0.55, 7.47, .05),
            (0.66, 7.3, .05), (0.81, 7.17, .05)],
    'BL3': [(0.12, 11.45, .04), (0.15, 11.0, .05), (0.21, 10.4, .06), (0.3, 9.85, .065), (0.42, 9.35, .07), (0.52, 8.95, .07),
            (0.6, 8.55, .065), (0.68, 8.26, .06), (0.53, 8.0, .06), (0.34, 7.82, .055), (0.27, 7.6, .055), (0.28, 7.32, .05)],
    'BL2': [(0.2, 11.5, .04), (0.25, 11.0, .05), (0.33, 10.4, .06), (0.43, 9.85, .065), (0.56, 9.35, .07), (0.67, 8.95, .07),
            (0.8, 8.55, .065), (0.83, 8.25, .06), (0.75, 7.97, .055), (0.73, 7.76, .05), (0.86, 7.5, .05), (1.03, 7.22, .045),
            (1.15, 7.0, .04), (1.14, 6.82, .035), (1.04, 6.7, .025), (0.94, 6.67, 0)],
    'BL1': [(0.27, 11.55, .035), (0.33, 11.05, .045), (0.44, 10.45, .05), (0.54, 9.85, .05), (0.68, 9.35, .05), (0.81, 8.95, .05),
            (0.94, 8.55, .05), (0.98, 8.2, .045), (1.0, 7.95, .045), (1.08, 7.75, .045), (1.19, 7.52, .04), (1.3, 7.3, .04),
            (1.39, 7.08, .035), (1.42, 6.9, .025), (1.4, 6.78, 0)],
    # her right: BR3 runs down the parting and along the bottom of the lock into the spiral, BR2 sweeps out and
    # hooks up, BR1 comes down the outside and coils into a spiral that faces back; BR0 a short strand by the braid
    'BR3': [(-0.01, 11.3, .035), (-0.02, 10.8, .04), (-0.06, 10.1, .045), (-0.2, 9.65, .05), (-0.33, 9.3, .05), (-0.39, 8.95, .05),
            (-0.4, 8.5, .05), (-0.36, 8.2, .05), (-0.33, 7.85, .05), (-0.34, 7.6, .05), (-0.4, 7.43, .05), (-0.52, 7.34, .05),
            (-0.68, 7.31, .05), (-0.84, 7.33, .05)],
    'BR2': [(-0.04, 11.35, .04), (-0.06, 11.0, .05), (-0.15, 10.45, .06), (-0.3, 10.05, .07), (-0.45, 9.65, .075), (-0.6, 9.2, .08),
            (-0.72, 8.8, .08), (-0.77, 8.45, .075), (-0.8, 8.1, .07), (-0.82, 7.85, .065), (-0.9, 7.62, .06), (-1.02, 7.5, .05),
            (-1.12, 7.5, .035), (-1.18, 7.58, 0)],
    'BR1': [(-0.07, 11.4, .035), (-0.11, 11.0, .045), (-0.25, 10.45, .05), (-0.43, 10.05, .05), (-0.62, 9.65, .05), (-0.76, 9.2, .05),
            (-0.92, 8.8, .05), (-1.01, 8.45, .05), (-1.04, 8.1, .05), (-1.12, 7.85, .05), (-1.17, 7.6, .05), (-1.23, 7.35, .05),
            (-1.3, 7.12, .05), (-1.34, 6.95, .05)],
    'BR0': [(-0.42, 11.35, .03), (-0.45, 11.1, .04), (-0.47, 10.85, .03), (-0.47, 10.74, 0)],
}
# the curls, in 3D: a helix about an axis through the given centre (x, and z for an axis that is not upright),
# starting where the hanging part ends; r1 the radius it closes to, turns, drop (along a vertical axis) or advance
# (along a horizontal one), and the sense of the turn
BK_CURL = {
    'BL4': dict(axis=(0, 0, 1), cx=0.55, r1=0.15, turns=1.75, drop=0.75, sense=1),
    'BL3': dict(axis=(0, 0, 1), cx=0.55, r1=0.16, turns=2.0, drop=0.9, sense=1),
    'BR3': dict(axis=(0, 1, 0), cx=-0.92, cz=6.88, r1=0.13, turns=1.5, drop=0.1, sense=-1),
    'BR1': dict(axis=(0, 1, 0), cx=-0.92, cz=6.88, r1=0.2, turns=1.75, drop=0.16, sense=-1),
}
# colour and thickness, in the order they are laid on
BK_LOOK = {'BL4': (HAIR_DK, 0.05), 'BL3': (HAIR, 0.055), 'BL2': (HAIR_LT, 0.05), 'BL1': (HAIR, 0.045),
           'BR3': (HAIR_DK, 0.05), 'BR2': (HAIR, 0.055), 'BR1': (HAIR_LT, 0.05), 'BR0': (HAIR_LT, 0.04)}
BK_ROOT = (0.88, 11.87)     # the strands start inside the braid at the back of the head (y, z; measured: the braid's
                            # outside is at y 0.96 and z 11.84 to 11.9 in the middle)
BK_TOP = 0.96               # how far back a strand is where it comes out under the braid
BK_NAPE = 10.45             # the hair down the back above the two locks ends here (back locks, hairUnder)
BK_WIDE = 1.5               # the traced lines are the strands' borders, drawn apart; the strands overlap their
                            # neighbours so that each lock reads as one
BK_WAVE = (0.06, 0.7, 8.6)  # below z 8.6 the strands wave back and forth by 0.06, 0.7 a wave, so they curl in 3D


def body_back(x, z):
    """How far back (+y) what the back locks hang on reaches at (x, z): the capelet, the corset, the high collar,
    and the hair above the nape (the back locks, which end at BK_NAPE)."""
    f = float(_grid_at(CAPE_BACK, x, z))
    if 7.9 < z < 9.15:                                # corset_ab holds only on the corset
        a, b = corset_ab(z)
        if abs(x) < a:
            f = max(f, b * math.sqrt(1 - (x / a) ** 2))
    if 10.4 <= z <= 11.05:
        r = 0.41 if z > 10.83 else 0.26
        if abs(x) < r:
            f = max(f, CP_Y0 + math.sqrt(r * r - x * x))
    if BK_NAPE - 0.05 < z < 11.5:
        t = t_of_z(z)
        ax, ay = hrx_py(t) + 0.14, hry_py(t) + 0.14
        if abs(x) < ax:
            f = max(f, HY0 + ay * math.sqrt(1 - (x / ax) ** 2))
    return f


_ok = (_cy > CP_Y0) & (np.abs(_cx) < 1.5) & (_cz > 8.5) & (_cz < 11.5)
CAPE_BACK = np.zeros((301, 301))
np.maximum.at(CAPE_BACK, (np.round((_cx[_ok] - _CG[0]) / _CG[2]).astype(int), np.round((_cz[_ok] - _CG[1]) / _CG[2]).astype(int)), _cy[_ok])
for _ in range(2):
    CAPE_BACK = np.maximum.reduce([np.roll(np.roll(CAPE_BACK, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)])


def _lin(terms):
    """Sum of coefficient * expression, leaving out the zero terms."""
    out = [f"{fmt(c)}*{e}" if e else fmt(c) for c, e in terms if abs(c) > 5e-6]
    return '+'.join(out) if out else '0'


def curl_xyz(P, T, Sv, Nv, spec):
    """A curl written exactly for Arch, as a function of t from -1 to 1 (its angle th = Th (t + 1) / 2), starting at
    the end P of the hanging part with its tangent T, side vector Sv and thickness vector Nv. Upright axis: a
    corkscrew ringlet hanging from P, falling fast as it leaves the hanging part (matching its slope, at most 0.8 a
    radian) and then evenly; its section turns to face out from the axis over the first half turn and its width
    turns to lie along the axis. Axis facing back: a spiral in the plane of the back view whose section keeps
    facing back, drifting back as it closes. The radius closes linearly; width and thickness taper over the last
    half turn."""
    e3 = np.array(spec['axis'], float)
    up = e3[2] > 0.5
    Th = 2 * math.pi * spec['turns']
    O = np.array([spec['cx'], P[1], P[2] if up else spec['cz']])
    n0 = P - O
    n0 -= e3 * (n0 @ e3)
    r0 = float(np.linalg.norm(n0))
    n0 /= r0
    m0 = spec['sense'] * np.cross(e3, n0)
    hw, th = float(np.linalg.norm(Sv)), float(np.linalg.norm(Nv))
    sh, nh = Sv / hw, Nv / th
    TH = f"({fmt(Th / 2)}*(t+1))"
    CS, SN = f"cos{TH}", f"sin{TH}"
    R = f"({fmt(r0)}+{fmt((spec['r1'] - r0) / Th)}*{TH})"
    if up:
        ps = min(0.8, r0 * abs(T @ e3) / max(abs(T @ m0), 1e-3))
        k = 0.6
        ex = k * (1 - math.exp(-Th / k))
        pt = max(0.0, (spec['drop'] - ps * ex) / (Th - ex))
        A = f"(-({fmt(pt)}*{TH}+{fmt((ps - pt) * k)}*(1-exp(-{TH}/{fmt(k)}))))"
        B = f"(0.5+0.5*tanh(({TH}-1.5708)/0.6283))"
    else:
        A = f"({fmt(spec['drop'] / Th)}*{TH})"
    TP = f"tanh({fmt(Th / 2)}*(1-t)/1.5708)^0.6"         # (Th - th), written so it never goes below 0 at t = 1
    xyz = []
    for i in range(3):
        rad = f"({_lin([(n0[i], CS), (m0[i], SN)])})"
        centre = _lin([(O[i], ''), (1.0, f"{R}*{rad}") if abs(n0[i]) + abs(m0[i]) > 5e-6 else (0, ''), (e3[i], A)])
        if up:
            side = f"{fmt(hw)}*{TP}*((1-{B})*{fmt(sh[i])}+{B}*{fmt(e3[i])})"
            thick = f"{fmt(th)}*(0.3+0.7*{TP})*((1-{B})*{fmt(nh[i])}+{B}*{rad})"
        else:
            side = f"{fmt(hw * np.sign(sh @ n0))}*{TP}*{rad}"
            thick = f"{fmt(th * nh[i])}*(0.3+0.7*{TP})"
        xyz.append(f"{centre}+cos(w)*({side})+lcs(w)*({thick})")
    return tuple(xyz)


def _poly_fit(t, V, deg):
    cs = [np.round(np.polynomial.chebyshev.Chebyshev.fit(t, V[:, i], deg).convert(kind=np.polynomial.Polynomial).coef, 5) for i in range(V.shape[1])]
    fit = np.c_[[np.polynomial.polynomial.polyval(t, c) for c in cs]].T
    return cs, fit, float(np.max(np.linalg.norm(fit - V, axis=1)))


def _hang_pieces(n_seg, t, sig, P, W, TH, joined):
    """The hanging part fitted in n_seg polynomial pieces that join exactly; returns the parts, their errors and
    (the last piece's coefficients, its samples)."""
    parts, errs, prev = [], [], None
    for k_ in range(n_seg):
        st = sig[-1] * (k_ + (t + 1) / 2) / n_seg
        Pt = np.c_[[np.interp(st, sig, P[:, i]) for i in range(3)]].T
        Wt, THt = np.interp(st, sig, W), np.interp(st, sig, TH)
        T = np.gradient(Pt, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None]
        nrm = np.array([0, 1.0, 0]) - T[:, 1:2] * T
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        side = np.cross(T, nrm)
        side /= np.linalg.norm(side, axis=1)[:, None]
        fits = [_poly_fit(t, V, CH_DEG) for V in (Pt, side * Wt[:, None], nrm * THt[:, None])]
        cs = [f[0] for f in fits]
        if prev is not None:
            cs = [joined(c, [np.polynomial.polynomial.polyval(1.0, q) for q in pc], [np.polynomial.polynomial.polyval(-1.0, q) for q in c])
                  for c, pc in zip(cs, prev)]
        parts.append(('poly', cs, max(1, int(math.ceil(sig[-1] / n_seg / 1.5)))))
        errs.append(max(f[2] for f in fits))
        prev = cs
    return parts, errs, (prev, (Pt, T, nrm, side, Wt, THt))


def back_strands():
    """Each back strand as parts for Arch: ('poly', [C, U, V], pieces) for the hanging part, polynomials in t
    (-1..1) for the centre line, side and thickness vectors, in as many pieces as keep it within 0.006 (joined
    exactly); ('curl', xyz, pieces) for a curl, written exactly (curl_xyz). The hanging part lies on what is behind it and falls straight below the fullest
    point (how far back it is never decreases going down), like the chest locks."""
    under = np.zeros((301, 301))
    out = {}
    for key, (col, th) in BK_LOOK.items():
        sd = 1 if key[1] == 'L' else -1
        D = np.array(BK_STRANDS[key], float) * [1, 1, BK_WIDE]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(D[:, :2], axis=0).T))]
        s = np.linspace(0, d[-1], 600)
        Q = _nat_spline(d, D, s)
        T2 = np.gradient(Q[:, :2], axis=0)
        T2 /= np.linalg.norm(T2, axis=1)[:, None]
        N2 = np.c_[-T2[:, 1], T2[:, 0]]
        hw = np.maximum(Q[:, 2], 0)
        fb, fh = np.zeros(len(s)), np.zeros(len(s))
        for k in np.linspace(-1, 1, 7):
            px, pz = Q[:, 0] + k * hw * N2[:, 0], Q[:, 1] + k * hw * N2[:, 1]
            fb = np.maximum(fb, [body_back(a, b) for a, b in zip(px, pz)])
            fh = np.maximum(fh, _grid_at(under, px, pz))
        lo = np.maximum.accumulate(np.r_[BK_TOP, np.maximum(fb + CH_GAP, fh + CH_ON)])[1:]
        g, ker = lo.copy(), np.exp(-0.5 * (np.arange(-24, 25) / 8) ** 2)
        for _ in range(60):
            g = np.maximum(np.convolve(np.pad(g, 24, mode='edge'), ker / ker.sum(), 'valid'), lo)
        curl = BK_CURL.get(key)
        tip = np.ones(len(s)) if curl else np.minimum(1, (d[-1] - s) / 0.15) ** 0.6
        y = g + 0.3 * th * tip
        amp, lam, zw = BK_WAVE
        ramp = np.clip((zw - Q[:, 1]) / 0.5, 0, 1)
        y = y + amp * ramp * np.sin(2 * math.pi * s / lam + 2.4 * len(out))      # each strand its own phase
        for k in np.linspace(-1, 1, 9):
            px, pz = Q[:, 0] + k * hw * N2[:, 0], Q[:, 1] + k * hw * N2[:, 1]
            i, j = np.round((px - _CG[0]) / _CG[2]).astype(int), np.round((pz - _CG[1]) / _CG[2]).astype(int)
            ok = (i >= 0) & (i < 301) & (j >= 0) & (j < 301)
            np.maximum.at(under, (i[ok], j[ok]), (y + th * tip * (1 - 0.8 * k * k))[ok])
        # the hanging part: a natural spline from the root inside the braid through the traced part every 0.05
        idx = np.unique(np.r_[np.arange(0, len(s), max(1, int(0.05 / (s[1] - s[0])))), len(s) - 1])
        root = [[0.5 * D[0, 0], BK_ROOT[0], BK_ROOT[1]]] if key != 'BR0' else [[sd * 0.36, 0.7, 11.7]]
        K = np.c_[np.r_[root, np.c_[Q[idx, 0], y[idx], Q[idx, 1]]], np.r_[hw[0], hw[idx]], np.r_[th, th * tip[idx]]]
        dk = np.r_[0, np.cumsum(np.linalg.norm(np.diff(K[:, :3], axis=0), axis=1))]
        Kd = _nat_spline(dk, K, np.linspace(0, dk[-1], 4000))
        P, W, TH = Kd[:, :3], np.maximum(Kd[:, 3], 0), np.maximum(Kd[:, 4], 0)
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        t = np.linspace(-1, 1, 1201)
        fade = np.polynomial.polynomial.polypow([0.5, -0.5], 3)      # ((1 - t) / 2)^3: 1 at t = -1, 0 at t = 1

        def joined(cs, end, start):
            """Polynomials moved so that they start exactly where the previous part ends, the move fading out."""
            return [np.round(np.polynomial.polynomial.polyadd(c, (e - s0) * fade), 5) for c, e, s0 in zip(cs, end, start)]

        # the hanging part, in two polynomial pieces (the lower one bends in S-curves) that join exactly
        for n_seg in range(1 if sig[-1] < 2.5 else 2, 5):      # as few pieces as keep the fit within 0.006
            parts, errs, prev = _hang_pieces(n_seg, t, sig, P, W, TH, joined)
            if max(errs) < 0.006:
                break
        Pt, T, nrm, side, Wt, THt = prev[1]
        prev = prev[0]
        if curl:
            # starts exactly where the hanging part's polynomials end
            ev = lambda cs, d=0: np.array([np.polynomial.polynomial.polyval(1.0, np.polynomial.polynomial.polyder(c, d)) for c in cs])
            Te = ev(prev[0], 1)
            parts.append(('curl', curl_xyz(ev(prev[0]), Te / np.linalg.norm(Te), ev(prev[1]), ev(prev[2]), curl), int(math.ceil(curl['turns']))))
        out[key] = (parts, max(errs), col, errs)
    return out


def run_hair():
    sc.blank()
    sc.note('Hair')
    sc.comment(f"""Her hair flows from the crown. A lock is described by t, from the crown (0) down to the end of
the longest hair (1): up to t = {fmt(T0)} it lies on a dome around the head (ha is the angle from the
crown), below that it falls. hzz(t) is the height, hrx(t) and hry(t) are the half-axes of the ellipse the
hair lies on at that height: they follow the dome, then keep clear of the shoulders, capelet, arms, book
and skirt and flare out toward the ends (fitted by the generator). hu(t) is 0 on the dome and grows to 1
down the fall. hdw keeps a lock's width steady as the hair spreads.""")
    (gx, m1, cx_), (gy, m2, cy_) = HP_X, HP_Y
    # hdw = min(1, 1.54 / (hrx + hry)), exactly: on the dome hrx + hry = 1.54 sin(ha) and the profiles' knot terms
    # are 0; on the fall sin(ha) = 1. So hdw = 1.54 / (1.54 + both profiles' knot terms), one spline (their
    # coefficients summed, the knots are shared) instead of min(...), which Arch writes with both profiles twice.
    gs = ''.join(f"+{fmt(a + b)}*max(0,t-{fmt(T0 + k)})^2" for a, b, k in zip(cx_, cy_, KNOTS))
    print(f'  hair clearance margins: side {m1:.3f}, back {m2:.3f}', file=sys.stderr)
    sc.func('ha(t)', f"1.5708*min(1,t/{fmt(T0)})")
    sc.func('hu(t)', f"max(0,t-{fmt(T0)})/{fmt(1 - T0)}")
    sc.func('hzz(t)', f"{fmt(ZHC)}+{fmt(HDZ)}*cos(ha(t))-{fmt(VF)}*max(0,t-{fmt(T0)})")
    sc.func('hrx(t)', f"{fmt(HDX)}*sin(ha(t)){gx}")
    sc.func('hry(t)', f"{fmt(HDY)}*sin(ha(t)){gy}")
    sc.func('hdw(t)', f"1.54/(1.54{gs})")
    sc.comment(f"""Her head is tilted and turned (see the generator); a lock turns with it where it lies on the head and
hangs straight below. hb(t) is 1 on the head and eases to 0 by t = {fmt(T0 + TB)}; hs and hk are the sine and
one minus the cosine of the head's turn times hb, and hm scales the lock with the head where it lies on it.""")
    sc.func('hb(t)', f"1-min(1,max(0,(t-{fmt(T0)})/{fmt(TB)}))^2*(3-2*min(1,max(0,(t-{fmt(T0)})/{fmt(TB)})))")
    sc.func('hs(t)', f"sin({fmt(HEAD_TURN['angle'])}*hb(t))")
    sc.func('hk(t)', f"1-cos({fmt(HEAD_TURN['angle'])}*hb(t))")
    sc.func('hm(t)', f"1-{fmt(1 - HEAD_TURN['scale'])}*hb(t)")
    HB = ('hs(t)', 'hk(t)', 'hm(t)')
    sc.comment("""A lock is a tube around its own flow line. lpc is the angle of that line around the head: a starting
angle pb+p0 (0 = straight back, + toward her left) whose part p0 closes up by sg as the hair spreads (sg = 1
falls straight down, sg = 0 fans out over the shoulders; pb is 0 for the back and pi for the front), a drift dr, and a wave of size am that grows down the
lock. lrc is how far the line stands off the hair surface: its layer la, and the wave again a quarter turn
later, so the lock snakes in and out as well as side to side. Across the lock (w runs around it) lwd is
the half-width and lth the thickness, a crescent lcs with three ridges that is flat underneath. ltp tapers
the lock toward its end te, and lq runs 0..1 over the last stretch, where the tip turns through three
quarters of a circle: sideways like a hook, or outward like a scroll.""")
    sc.func('ltp(t,te)', 'min(1,max(0,te-t)/0.13)^0.6')
    sc.func('lq(t,te)', 'max(0,1-max(0,te-t)/0.08)')
    sc.func('lcs(w)', '(0.65*sin(w)+0.35*abs(sin(w)))*(0.78+0.22*cos(2*pi*cos(w)))')
    sc.func('lpc(t,pb,p0,sg,dr,am,kk,ph)', 'pb+p0*(1-sg*(1-hdw(t)))+dr*hu(t)+am*hu(t)^1.5*sin(kk*hu(t)+ph)')
    sc.func('lrc(t,la,am,kk,ph)', 'la*min(1,t/0.04)+0.6*am*hu(t)^1.5*cos(kk*hu(t)+ph)')
    sc.func('lwd(t,te,hw)', 'hw*ltp(t,te)*hdw(t)')
    sc.func('lth(t,w,te,th)', 'th*ltp(t,te)*lcs(w)*min(1,t/0.04)')

    def lock(ref, p0, sg, dr, hw, te, la, th, am, kk, ph, col, curl=('side', 1, 0.12), start=0.012, pb=0.0):
        """One lock: main part, then its curled tip as a second surface (so the curl gets a fine mesh)."""
        PC = f"lpc(t,{fmt(pb)},{fmt(p0 - pb)},{fmt(sg)},{fmt(dr)},{fmt(am)},{fmt(kk)},{fmt(ph)})"
        RC = f"lrc(t,{fmt(la)},{fmt(am)},{fmt(kk)},{fmt(ph)})"
        WD = f"lwd(t,{fmt(te)},{fmt(hw)})"
        TH = f"lth(t,w,{fmt(te)},{fmt(th)})"
        end = te if curl is None else te - 0.08
        with sc.turned(**HEAD_TURN, blend=HB):
            sc.surface((f"(hrx(t)+({RC}+{TH})*sin(ha(t)))*sin({PC}+{WD}*cos(w))", f"{fmt(HY0)}+(hry(t)+({RC}+{TH})*sin(ha(t)))*cos({PC}+{WD}*cos(w))",
                        f"hzz(t)+({RC}+{TH})*cos(ha(t))"), ('t', start, end), ('w', 0, '2*pi'), col, ref, ('tube', 'u'))
        if curl is None:
            return
        AL = f"(4.712*lq(t,{fmt(te)}))"
        ZC = f"hzz(t)+{fmt(VF * 0.08)}*lq(t,{fmt(te)})"
        if curl[0] == 'side':
            _, sgn, rc = curl
            PSI = f"{PC}+{fmt(sgn * rc / 1.6)}*(1-cos{AL})+{WD}*cos(w)*cos{AL}"
            xyz = (f"(hrx(t)+{RC}+{TH})*sin({PSI})", f"{fmt(HY0)}+(hry(t)+{RC}+{TH})*cos({PSI})",
                   f"{ZC}-{fmt(rc)}*sin{AL}+{fmt(sgn * 0.77 * hw)}*ltp(t,{fmt(te)})*cos(w)*sin{AL}")
        else:
            _, rc = curl
            RAD = f"{RC}+{fmt(rc)}*(1-cos{AL})+{TH}*cos{AL}"
            xyz = (f"(hrx(t)+{RAD})*sin({PC}+{WD}*cos(w))", f"{fmt(HY0)}+(hry(t)+{RAD})*cos({PC}+{WD}*cos(w))",
                   f"{ZC}-{fmt(rc)}*sin{AL}+{TH}*sin{AL}")
        sc.surface(xyz, ('t', te - 0.08, te), ('w', 0, '2*pi'), col, ref + 'Curl', ('tube', 'u'))

    sc.comment("""A dark cap and an inner sheet under the locks, so no gap shows skin. The cap stops higher above her left
eye and a little higher above her right eye, where the forehead shows between the bangs as on the sheet.""")
    CAPM = "(1.75-0.42*exp(-((v+1.3)/0.1)^2)-0.2*exp(-((v+1.15)/0.15)^2)-0.1*exp(-((v+2.12)/0.12)^2))"
    with sc.turned(**HEAD_TURN):
        sc.surface((f"{fmt(HDX * 0.98)}*sin(u*{CAPM})*cos(v)", f"{fmt(HY0)}+{fmt(HDY * 0.98)}*sin(u*{CAPM})*sin(v)", f"{fmt(ZHC)}+{fmt(HDZ * 0.98)}*cos(u*{CAPM})"),
               ('u', 0, 1), ('v', '-pi', 'pi'), HAIR_DK, 'hairCap', ('point', (0, HY0, ZHC)))
    with sc.turned(**HEAD_TURN, blend=HB):
        sc.surface(("(hrx(t)-0.03)*sin(k*(1-0.64*(1-hdw(t))))", f"{fmt(HY0)}+(hry(t)-0.03)*cos(k*(1-0.64*(1-hdw(t))))", "hzz(t)"),
                   ('t', T0 * 0.9, t_of_z(BK_NAPE) - 0.03), ('k', -1.2, 1.2), HAIR_DK, 'hairUnder', ('point', (0, 0, 9)))

    rng = np.random.default_rng(11)
    R = rng.random
    def cdir(rc):
        return ('out', rc) if R() < 0.3 else ('side', 1 if R() < 0.5 else -1, rc)
    T_NAPE = t_of_z(BK_NAPE)

    def nape_lock(ref, p0, sg, hw, la, th, am, kk, ph, col):
        """A lock from the crown that ends at the nape, closing in as it falls so that it ends under the two back locks."""
        p_c = p0 * (1 - sg * (1 - hdw_py(T_NAPE)))
        dr = (math.copysign(min(abs(p_c), 0.42), p0) - p_c) / hu_py(T_NAPE)
        lock(ref, p0, sg, dr, hw, T_NAPE, la, th, am, kk, ph, col, curl=None)

    sc.comment(f"""Long locks. On the sheet's back view two locks (drawn below) are nearly all the hair down the middle of
her back, with the corset showing between them and the capelet's back beside them. So the locks from the crown
down the back (three layers: wide ones underneath, the main layer, thin lighter strands on top) end at the
nape (z {fmt(BK_NAPE)}), closing in under those two, except the outermost of each layer, which fall over the
arms with the locks at the sides that fan out over the shoulders.""")
    # the random draws are made as before the back was parted, so the side locks keep theirs. The outermost locks of
    # each layer stay long and are moved out over the arms, at least 1.32 from the back so they pass outside the
    # capelet's back emblems (which the sheet shows), where the sheet's back view shows the rest of the hair falling.
    def back_layer(tag, p0s, sg, hw, la, th, col, draw, arms):
        for i, p0 in enumerate(p0s):
            dr_, te_, am_, kk_, ph_, cd_ = draw()
            rank = sorted((abs(p) for p in p0s if p * p0 > 0), reverse=True).index(abs(p0)) if p0 != 0 else 99
            if rank < len(arms):        # the outermost: long, out over the arms (beyond the capelet's back emblems)
                lock(f'{tag}{i + 1}', math.copysign(arms[rank], p0), 0.3, dr_, hw, te_, la, th, am_, kk_, ph_, col, curl=cd_)
            else:
                nape_lock(f'{tag}{i + 1}', p0, sg, hw, la, th, am_, kk_, ph_, col)
    back_layer('backU', (-1.0, -0.68, -0.34, 0.0, 0.34, 0.68, 1.0), 0.66, 0.28, 0.0, 0.05, HAIR_DK,
               lambda: (0.05 * (R() - 0.5), 0.86 + 0.1 * R(), 0.07 + 0.04 * R(), 8 + 3 * R(), 6.28 * R(), cdir(0.11)), (1.5, 1.36))
    back_layer('back', (-0.85, -0.51, -0.17, 0.17, 0.51, 0.85), 0.68, 0.24, 0.04, 0.055, HAIR,
               lambda: (0.06 * (R() - 0.5), 0.9 + 0.1 * R(), 0.09 + 0.05 * R(), 9 + 4 * R(), 6.28 * R(), cdir(0.13)), (1.44, 1.32))
    back_layer('backT', (-0.7, -0.3, 0.1, 0.45, 0.8), 0.68, 0.1, 0.085, 0.04, HAIR_LT,
               lambda: (0.12 * (R() - 0.5), 0.8 + 0.18 * R(), 0.12 + 0.06 * R(), 10 + 5 * R(), 6.28 * R(), cdir(0.12)), (1.4,))
    side = [  # p0 (her right; mirrored for her left), layer, half-width, colour
        (1.3, 'U', 0.3, HAIR_DK), (1.62, 'U', 0.28, HAIR_DK), (1.18, 'M', 0.24, HAIR), (1.5, 'M', 0.24, HAIR), (1.8, 'M', 0.2, HAIR),
        (1.35, 'T', 0.1, HAIR_LT), (1.68, 'T', 0.1, HAIR_LT)]
    for sd, tag in ((-1, 'R'), (1, 'L')):
        for i, (p0, lay, hw_, col) in enumerate(side):
            if sd > 0 and p0 > 1.25:
                # her left ear shows: the hair above it is swept back under the ornament
                lock(f'tuck{i + 1}', p0, 0.0, -1.2, hw_, T0 + 0.05, {'U': 0.0, 'M': 0.035, 'T': 0.07}[lay], 0.05, 0.0, 1, 0, col, curl=None)
                continue
            la_, th_ = {'U': (0.0, 0.05), 'M': (0.04, 0.055), 'T': (0.085, 0.04)}[lay]
            lock(f'side{tag}{i + 1}', sd * p0, 0.25, -sd * (0.1 + 0.1 * R()), hw_, 0.74 + 0.16 * R(), la_, th_, 0.1 + 0.06 * R(), 9 + 4 * R(), 6.28 * R(), col, curl=cdir(0.13))
    sc.comment("""The two locks that fall over the chest, from under the jaw to curled tips at the top of the corset, as
traced on the sheet's portrait: each is several strands, a centre line with side and thickness vectors written
as polynomials in t (-1..1) and the crescent section lcs(w) (fitted by the generator: they lie on the capelet
and the blouse, fall straight below the bust and rise from under the jaw into the head). The lower half of each is
a separate surface so that its curl gets a fine mesh.""")
    CH = chest_strands()
    print(f"  chest locks: degree {CH_DEG}, largest fit error {max(v[1] for v in CH.values()):.4f}", file=sys.stderr)
    for key, (polys, err, col, _, _) in CH.items():
        C, U, V = polys
        xyz = tuple(f"{horner(C[i], 't')}+cos(w)*({horner(U[i], 't')})+lcs(w)*({horner(V[i], 't')})" for i in range(3))
        sc.surface(xyz, ('t', -1, 0), ('w', 0, '2*pi'), col, f'chestLock{key}', ('tube', 'u'))
        sc.surface(xyz, ('t', 0, 1), ('w', 0, '2*pi'), col, f'chestLock{key}Tip', ('tube', 'u'))
    sc.comment("""The two locks down the back, from under the braid, as traced on the sheet's back view: on her left four
strands, two spreading out at the bottom and two winding together into a corkscrew ringlet; on her right
strands that sweep out and coil into a spiral facing back. The hanging part of each strand is written like the
chest locks (in pieces); a curl is a helix written as it is, in its angle th = Th (t + 1) / 2, its radius closing
as it turns, one surface a turn.""")
    BK = back_strands()
    print(f"  back locks: largest fit error {max(v[1] for v in BK.values()):.4f}", file=sys.stderr)
    for key, (parts, err, col, _) in BK.items():
        letters = iter('abcdefgh')
        for part in parts:
            if part[0] == 'poly':
                _, (C, U, V), n = part
                xyz = tuple(f"{horner(C[i], 't')}+cos(w)*({horner(U[i], 't')})+lcs(w)*({horner(V[i], 't')})" for i in range(3))
                name = f'backLock{key[2:]}{key[1]}'
            else:
                _, xyz, n = part
                name = f'backLock{key[2:]}{key[1]}Curl'
            for j in range(n):
                tag = next(letters) if part[0] == 'poly' else 'abcd'[j]
                sc.surface(xyz, ('t', -1 + 2 * j / n, -1 + 2 * (j + 1) / n), ('w', 0, '2*pi'), col, f'{name}{tag}', ('tube', 'u'))

    # ---------- bangs and the locks that frame the face ----------
    sc.blank()
    sc.note('Bangs')
    sc.comment("""Bangs and the locks framing the face are blades: a Bezier centre line (written as a polynomial in t)
that leaves the crown, follows the dome to the hairline and hangs to a point just in front of the face.
Across it (w) the blade has the same crescent section as the long locks; its width and thickness
vectors are polynomials too.""")
    fyf = sc.ns['fy']

    def blade(ref, pts, nrms, wid, thk, color, sheen=None):
        """sheen=(t0, t1): a lighter streak down the front of the blade between t0 and t1, as anime hair is shaded."""
        n = len(pts) - 1
        sides = []
        for i in range(n + 1):
            T = unit(sub(pts[min(i + 1, n)], pts[max(i - 1, 0)]))
            s_ = cross(T, nrms[i])
            sides.append(unit(s_))
        C = bezier_poly(interp_bezier(pts))
        U = bezier_poly(interp_bezier([mul(s_, w_) for s_, w_ in zip(sides, wid)]))
        Vv = bezier_poly(interp_bezier([mul(n_, t_) for n_, t_ in zip(nrms, thk)]))
        xyz = tuple(f"{horner(C[i], 't')}+cos(w)*({horner(U[i], 't')})+lcs(w)*({horner(Vv[i], 't')})" for i in range(3))
        FRONT_BLADES.append((ref, xyz))
        with sc.turned(**HEAD_TURN):
            sc.surface(xyz, ('t', 0, 1), ('w', 0, '2*pi'), color, ref, ('tube', 'u'))
        if sheen:
            t0, t1, col = sheen
            tp = f"max(0,sin(pi*(t-{fmt(t0)})/{fmt(t1 - t0)}))^1.2"
            sx = tuple(f"{horner(C[i], 't')}+0.22*cos(w)*{tp}*({horner(U[i], 't')})+1.12*({horner(Vv[i], 't')})" for i in range(3))
            FRONT_BLADES.append((ref + 'Sheen', sx))
            with sc.turned(**HEAD_TURN):
                sc.surface(sx, ('t', t0, t1), ('w', 0, 'pi'), col, ref + 'Sheen', ('dir', (0, -1, 0.3)))

    def bang(ref, phis, tip, w0, color, ths=(0.14, 0.75, 1.32), fwd_=0.05, off=0.04, tk=1.0, wp=(0.18, 0.95, 1.0, 0.6), lift=0.0):
        """lift stacks a lock on top of the ones it overlaps, so they lie on each other instead of crossing."""
        p0, p1, p2 = dome(ths[0], phis[0], 1.005 + lift), dome(ths[1], phis[1], 1.035 + lift), dome(ths[2], phis[2], 1.045 + lift)
        xt, zt = tip
        p4 = (xt, fyf(xt, zt) - off - 0.5 * lift, zt)
        x3, z3 = p2[0] + (xt - p2[0]) * 0.55, p2[2] + (zt - p2[2]) * 0.5
        y3 = min(0.5 * (p2[1] + p4[1]) - fwd_, fyf(x3, z3) - 0.075) - lift
        p3 = (x3, y3, z3)
        n3 = unit((0.4 * x3, -1, 0.15))
        n4 = unit((0.4 * xt, -1, 0.0))
        blade(ref, [p0, p1, p2, p3, p4], [dome_normal(p0), dome_normal(p1), dome_normal(p2), n3, n4],
              [wp[0] * w0, wp[1] * w0, wp[2] * w0, wp[3] * w0, 0.0], [0.008 * tk, 0.04 * tk, 0.045 * tk, 0.03 * tk, 0.0], color)

    def strand(ref, xtop, mid, tip, hw, color, lift=0.0, tk=0.75, off=0.04, sweep=-0.12, sheen=None, hug=False):
        """One strand of the bangs: from the crown over the dome to the hairline at x = xtop, then through mid and
        down to a pointed tip (face x, height above the pupils) just in front of the face. hw: half-widths at the
        hairline and at mid. It is narrow and round in section, so the light runs across it like hair. hug keeps
        its middle close to the face (for strands that fall past the cheeks)."""
        ph2 = math.asin(max(-0.98, min(0.98, xtop / (HDX * math.sin(1.32) * (1.045 + lift)))))
        p0, p1, p2 = dome(0.14, ph2 + 2 * sweep, 1.005 + lift), dome(0.75, ph2 + sweep, 1.035 + lift), dome(1.32, ph2, 1.045 + lift)
        (x3, z3), (xt, zt) = (mid[0], ZE + mid[1]), (tip[0], ZE + tip[1])
        p4 = (xt, fyf(xt, zt) - off - 0.5 * lift, zt)
        y3 = fyf(x3, z3) - 0.05 if hug else min(0.5 * (p2[1] + p4[1]) - 0.03, fyf(x3, z3) - 0.07)
        p3 = (x3, y3 - lift, z3)
        blade(ref, [p0, p1, p2, p3, p4], [dome_normal(p0), dome_normal(p1), dome_normal(p2), unit((0.4 * x3, -1, 0.15)), unit((0.4 * xt, -1, 0.0))],
              [0.2 * hw[0], 0.9 * hw[0], hw[0], hw[1], 0.0], [0.008 * tk, 0.038 * tk, 0.042 * tk, 0.034 * tk, 0.0], color,
              sheen=None if sheen is None else (0.47, sheen, HAIR_HI))

    # As on the sheet's portrait: one wide lock sweeps down to a point at the bridge of the nose, the forehead
    # shows in a tall triangle above her left eye, a lock sweeps over her left temple, and above her right eye a
    # smaller gap is framed by the locks at the side (bang5-6). The wide lock and the temple lock are made of
    # strands along the strand lines drawn on the sheet, with staggered pointed tips; a thin dark layer under the
    # wide lock keeps the forehead from showing between its strands. Strands are stacked (lift) so that they lie
    # on each other, and the main ones carry a lighter streak (sheen).
    strand('bangUnder', -0.08, (-0.09, 0.25), (-0.05, -0.03), (0.23, 0.16), HAIR_SH, tk=0.3)
    strand('bang1', -0.23, (-0.23, 0.25), (-0.14, -0.08), (0.07, 0.055), HAIR, lift=0.018, sheen=0.8)
    strand('bang2', -0.31, (-0.32, 0.3), (-0.28, 0.05), (0.045, 0.035), HAIR_DK, lift=0.03)
    strand('bang3', -0.13, (-0.12, 0.25), (-0.06, -0.09), (0.065, 0.05), HAIR_LT, lift=0.032, sheen=0.78)
    strand('bang4', -0.03, (-0.04, 0.25), (-0.01, -0.1), (0.07, 0.055), HAIR, lift=0.015, sheen=0.82)
    strand('bang10', 0.08, (0.06, 0.25), (0.04, -0.06), (0.07, 0.05), HAIR_LT, lift=0.03, sheen=0.78)
    strand('bang11', 0.16, (0.135, 0.28), (0.07, 0.02), (0.045, 0.033), HAIR_DK, lift=0.042)
    strand('bang12', -0.17, (-0.13, 0.42), (-0.08, 0.2), (0.05, 0.035), HAIR_LT, lift=0.05)
    strand('bang13', 0.12, (0.11, 0.38), (0.06, 0.13), (0.045, 0.03), HAIR, lift=0.056)
    bang('bang5', (-0.6, -0.8, -0.85), (-0.5, ZE + 0.1), 0.16, HAIR_LT, lift=0.03)
    bang('bang6', (-0.7, -1.05, -1.27), (-0.6, ZE - 0.02), 0.18, HAIR)
    strand('bang7', 0.3, (0.4, 0.3), (0.55, 0.07), (0.07, 0.05), HAIR, sweep=-0.1, sheen=0.8)
    strand('bang9', 0.42, (0.51, 0.3), (0.62, 0.0), (0.08, 0.065), HAIR_LT, lift=0.02, sweep=-0.1)
    strand('bang14', 0.52, (0.55, 0.38), (0.53, 0.14), (0.05, 0.035), HAIR_DK, lift=0.04, sweep=-0.1)
    bang('bang8', (1.0, 1.26, 1.42), (0.66, ZE + 0.1), 0.16, HAIR_LT, lift=0.03)
    for sd, tag in ((1, 'L'), (-1, 'R')):
        pts = [dome(0.3, 1.5 * sd, 1.005), dome(1.15, 1.62 * sd, 1.04), (0.71 * sd, -0.07, ZE + 0.06), (0.69 * sd, -0.13, ZE - 0.2)]
        xt, zt = 0.6 * sd, ZE - 0.38
        pts.append((xt, fyf(xt, zt) - 0.045, zt))
        nr = [dome_normal(pts[0]), dome_normal(pts[1]), unit((sd, -0.25, 0)), unit((sd, -0.3, 0)), unit((sd, -0.5, 0))]
        blade(f'frame{tag}', pts, nr, [0.04, 0.15, 0.14, 0.1, 0.0], [0.008, 0.04, 0.04, 0.03, 0.0], HAIR)

    sc.comment("""Thin wisps from the bangs that cross the eyes, and locks that fall from the temples to below the jaw:
beside the face on her right, so the jaw line shows, and over the outer cheek on her left, as in the portrait.""")
    bang('wisp1', (0.22, 0.02, -0.14), (-0.25, ZE - 0.07), 0.035, HAIR_LT, off=0.042, tk=0.45)
    bang('wisp2', (0.3, 0.22, 0.16), (0.2, ZE + 0.02), 0.03, HAIR_LT, off=0.042, tk=0.45)
    bang('wisp3', (-0.45, -0.78, -0.98), (-0.5, ZE - 0.1), 0.032, HAIR, off=0.042, tk=0.45)
    # thin strands that fall past her left cheek and flick inward, and one beside her right cheek, as drawn on the sheet
    strand('wisp4', 0.62, (0.55, -0.1), (0.34, -0.34), (0.04, 0.03), HAIR_LT, lift=0.02, off=0.03, sweep=-0.05, hug=True)
    strand('wisp5', 0.63, (0.54, -0.24), (0.31, -0.47), (0.035, 0.026), HAIR, lift=0.025, off=0.03, sweep=-0.05, hug=True)
    strand('wisp6', 0.62, (0.5, -0.4), (0.35, -0.56), (0.03, 0.022), HAIR_LT, lift=0.03, off=0.03, sweep=-0.05, hug=True)
    strand('wisp7', -0.6, (-0.62, 0.2), (-0.63, -0.22), (0.05, 0.04), HAIR_LT, lift=0.03, sweep=0.05, hug=True)

    def face_normal(x, z, e=1e-4):
        return unit(((fyf(x + e, z) - fyf(x - e, z)) / (2 * e), -1, (fyf(x, z + e) - fyf(x, z - e)) / (2 * e)))

    def cheek_lock(ref, sd, dx, x2, x3, x4, z4, w0, color, gap=0.045):
        p0 = dome(0.6, (1.12 + dx) * sd, 1.01)
        p1 = dome(1.22, (1.18 + 0.5 * dx) * sd, 1.045)
        z2, z3 = ZE + 0.02, ZE - 0.28
        fw_ = sc.ns['fwd']
        p2 = (x2 * sd, fyf(min(x2, fw_(z2) - 0.02) * sd, z2) - gap, z2)
        p3 = (x3 * sd, fyf(min(x3, fw_(z3) - 0.02) * sd, z3) - gap, z3)     # beside the jaw where it is outside it
        p4 = (x4 * sd, -0.3, z4)
        # keep clear of the face: wherever the centre line is over the face it stays gap/2 in front of it;
        # if it does not, bring the lower part of the lock forward
        for _ in range(12):
            pts = [p0, p1, p2, p3, p4]
            Cb = interp_bezier(pts)
            worst = min([fyf(P[0], P[2]) - P[1] for P in (bezier_eval(Cb, u) for u in np.linspace(0, 1, 300))
                         if HC + 0.01 < P[2] < ZE + 0.6 and abs(P[0]) < sc.ns['fwd'](P[2])] or [1])
            if worst > gap * 0.5:
                break
            p3, p4 = sub(p3, (0, 0.006, 0)), sub(p4, (0, 0.012, 0))
        assert worst > gap * 0.5, (ref, worst)
        nr = [dome_normal(p0), dome_normal(p1), face_normal(p2[0], z2), face_normal(p3[0], z3), unit((0.6 * sd, -1, 0))]
        blade(ref, pts, nr, [0.035, 0.9 * w0, w0, 0.85 * w0, 0.0], [0.008, 0.03, 0.028, 0.024, 0.0], color)

    for sd, tag in ((1, 'L'), (-1, 'R')):
        if sd < 0:      # her right: the hair falls beside the face, so the whole jaw line shows
            cheek_lock(f'cheekLock{tag}1', sd, 0.0, 0.63, 0.6, 0.5, ZE - 0.86, 0.075, HAIR)
            cheek_lock(f'cheekLock{tag}2', sd, -0.07, 0.61, 0.57, 0.45, ZE - 0.74, 0.05, HAIR_LT)
            cheek_lock(f'cheekLock{tag}3', sd, 0.06, 0.645, 0.63, 0.56, ZE - 0.98, 0.065, HAIR_DK)
        else:           # her left: the hair covers the outer cheek, leaving about 0.42 of the face showing
            cheek_lock(f'cheekLock{tag}1', sd, 0.0, 0.6, 0.47, 0.36, ZE - 0.82, 0.08, HAIR)
            cheek_lock(f'cheekLock{tag}2', sd, -0.07, 0.575, 0.42, 0.3, ZE - 0.68, 0.05, HAIR_LT)
            cheek_lock(f'cheekLock{tag}3', sd, 0.06, 0.615, 0.53, 0.44, ZE - 0.96, 0.065, HAIR_DK)

    sc.blank()
    sc.note('Brows')
    with sc.turned(**HEAD_TURN):
        draw_brows(FRONT_BLADES)

    # ---------- the braid crown ----------
    sc.blank()
    sc.note('Braid')
    sc.comment("""A braided crown around the back of the head. The braid follows an oval ring on the dome: bdx, bdy, bdz
give the direction from the head's centre to the ring at angle a, with ro scaling the ring (1 = its
centre line). Its three strands weave as sin(tau) across the ring and sin(2 tau) in height, tau a third
of a turn apart: the classic three-strand braid.""")
    THC, TH_, TV_ = 68 * math.pi / 180, math.tan(66 * math.pi / 180), math.tan(32 * math.pi / 180)
    sc.func('bdx(a,ro)', f"ro*{fmt(TH_)}*cos(a)")
    sc.func('bdy(a,ro)', f"{fmt(math.sin(THC))}+ro*{fmt(TV_ * math.cos(THC))}*sin(a)")
    sc.func('bdz(a,ro)', f"{fmt(math.cos(THC))}-ro*{fmt(TV_ * math.sin(THC))}*sin(a)")
    sc.func('bdn(a,ro)', 'sqrt(bdx(a,ro)^2+bdy(a,ro)^2+bdz(a,ro)^2)')
    NB = 13
    with sc.turned(**HEAD_TURN):
        for j in range(3):
            tau = f"({NB}*a+{fmt(2 * math.pi * j / 3)})"
            RO = f"(1+0.1*sin{tau}+0.085*cos(v))"
            RR = f"(1.13+0.04*sin(2*{tau})+0.07*sin(v))"
            for q in range(4):
                sc.surface((f"{fmt(HDX)}*{RR}*bdx(a,{RO})/bdn(a,{RO})", f"{fmt(HY0)}+{fmt(HDY)}*{RR}*bdy(a,{RO})/bdn(a,{RO})", f"{fmt(ZHC)}+{fmt(HDZ)}*{RR}*bdz(a,{RO})/bdn(a,{RO})"),
                           ('a', q * math.pi / 2, (q + 1) * math.pi / 2), ('v', 0, '2*pi'), (HAIR, HAIR_LT, HAIR_DK)[j], f'braid{j + 1}{"abcd"[q]}', ('tube', 'u'))
    return THC, TH_, TV_


if want('hair') or want('ornament'):
    if want('hair'):
        THC, TH_, TV_ = run_hair()
    else:
        THC, TH_, TV_ = 68 * math.pi / 180, math.tan(66 * math.pi / 180), math.tan(32 * math.pi / 180)
    # the ornament sits on the side of the head, above and just behind her left ear
    P_ = dome(1.03, 1.5, 1.1)
    ORN_N = unit(add(mul(dome_normal(P_), 0.9), (0.1, -0.4, 0)))
    ORN_P = add(P_, mul(ORN_N, 0.1))
