# Run inside generate.py after outfit.py: all of Claude Chan's hair.
import numpy as np

ZHC = ZE + 0.42                          # height of the centre of the cranium
HDX, HDY, HDZ = 0.79, 0.815, 0.76       # the dome of the hair around the cranium: as wide and as low as the hi-res
                                         # portrait draws her hair (measured from its viewpoint: the crown 0.1 pupil
                                         # distances lower than before, the sides fuller)
HY0 = 0.14
ZBOT = 5.3                               # where the longest locks end
TB = 0.09                                # how far down a lock eases from turning with the head to hanging straight
T0 = 1 / (1 + (ZHC - ZBOT) / (HDZ * math.pi / 2))   # part of a lock's length that lies on the dome
VF = HDZ * math.pi / 2 / T0              # how fast z falls with t below the dome

# the book in her left hand, which the side hair clears
_bk = BOOK[2]
BOOK_Z = (min(P[2] for P in _bk) - 0.05, max(P[2] for P in _bk) + 0.05)
BOOK_R = max(math.hypot(P[0], P[1] - 0.03) for P in _bk) + 0.04
BOOK_PSI = (min(math.atan2(P[0], P[1] - 0.03) for P in _bk) - 0.1, max(math.atan2(P[0], P[1] - 0.03) for P in _bk) + 0.1)


# HAIR_ARM_R: how far out from the body axis the arms and hands reach round each angle from the back (every 0.05) and
# at each height (every 0.1 from z 5), both arms folded onto one side, from arms.py's points: the side hair clears
# the arms' own outline, not a band at their furthest reach round every angle they cover (ARM_ENV, ARM_PSI)
_harm_p = arm_points()
HAIR_ARM_R = np.zeros((65, 45))
np.maximum.at(HAIR_ARM_R, (np.clip(np.round(np.arctan2(np.abs(_harm_p[:, 0]), _harm_p[:, 1] - 0.03) / 0.05).astype(int), 0, 64),
                           np.clip(np.round((_harm_p[:, 2] - 5.0) / 0.1).astype(int), 0, 44)),
              np.hypot(_harm_p[:, 0], _harm_p[:, 1] - 0.03) + 0.05)
def _grow(G, pad=0.0):
    """Each cell the largest of itself and its eight neighbours (padded at the edges, not wrapped round)."""
    Gp = np.pad(G, 1, constant_values=pad)
    return np.max([Gp[1 + a:1 + a + G.shape[0], 1 + b:1 + b + G.shape[1]] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)


HAIR_ARM_R = _grow(HAIR_ARM_R)


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
    if 5.0 <= z <= 9.4:                                 # arms and hands below the capelet, round their own angles
        r = max(r, float(HAIR_ARM_R[min(64, int(round(abs(psi) / 0.05))), int(round((z - 5.0) / 0.1))]))
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
PSIS = np.linspace(0, 1.85, 32)      # round to the side, as far as the long locks hang (not round to the book in front)


def clear_factor(rx, ry, z, margin):
    k = 1.0
    for p_ in PSIS:
        for _ in range(80):
            x_, y_ = k * rx * math.sin(p_), HY0 + k * ry * math.cos(p_)
            if math.hypot(x_, y_ - 0.03) >= envelope(math.atan2(x_, y_ - 0.03), z) + margin:
                break
            k *= 1.01
    return k


# The locks at her sides (the only ones below the nape) lie as close to her clothes as they can (the user's choice;
# on the sheet's back view they fall over the arms within 1.45 to 2.8 of her middle): the shell clears the shoulders,
# capelet, arms and hands by a little and does not flare; below its widest it falls straight.
side_req, back_req = [], []
for z in ZS:
    rx = max(HDX, max(envelope(p_, z) for p_ in (1.45, 1.6, 1.75)) + 0.06)
    ry = max(HDY, max(envelope(p_, z) for p_ in (0.0, 0.3)) + 0.0)
    k = clear_factor(rx, ry, z, 0.04)
    side_req.append(rx * k)
    back_req.append(ry * k)
side_req = np.maximum.accumulate(np.array(side_req))      # below its widest the hair falls straight, as hair does
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
# The hi-res portrait draws much more hair in front: besides these two locks, locks that come from behind the jaw, fall
# forward over the collarbone and the shoulder and down the capelet outside its emblems, framing the neck (as an artist
# would place them: (x, z, half-width) in the front view, like the traced ones). S: shoulder; 1 the main, wide lock, 2 a
# narrower one inside it ending above the emblem, 3 a thin outer strand.
CH_STRANDS.update({
    'RS1': [(-0.62, 11.15, .05), (-0.69, 10.88, .06), (-0.8, 10.58, .07), (-0.95, 10.28, .075), (-1.1, 10.0, .07), (-1.23, 9.74, .062),
            (-1.31, 9.5, .052), (-1.34, 9.33, .04), (-1.32, 9.22, .025), (-1.28, 9.16, 0)],
    'RS2': [(-0.57, 11.15, .035), (-0.62, 10.88, .04), (-0.71, 10.58, .045), (-0.83, 10.3, .045), (-0.95, 10.08, .04), (-1.04, 9.94, .03),
            (-1.08, 9.88, 0)],
    'RS3': [(-0.66, 11.1, .02), (-0.78, 10.82, .022), (-0.95, 10.52, .025), (-1.12, 10.22, .025), (-1.28, 9.96, .022), (-1.39, 9.72, .018),
            (-1.45, 9.52, .012), (-1.46, 9.4, 0)],
    'LS1': [(0.66, 11.1, .05), (0.73, 10.84, .06), (0.84, 10.55, .068), (0.98, 10.26, .072), (1.12, 9.98, .068), (1.24, 9.72, .06),
            (1.31, 9.5, .05), (1.33, 9.34, .038), (1.3, 9.24, .022), (1.26, 9.19, 0)],
    'LS2': [(0.61, 11.12, .035), (0.66, 10.86, .04), (0.75, 10.57, .042), (0.86, 10.3, .042), (0.97, 10.09, .036), (1.05, 9.96, .026),
            (1.09, 9.9, 0)],
    'LS3': [(0.7, 11.05, .02), (0.83, 10.78, .022), (0.99, 10.48, .024), (1.15, 10.2, .024), (1.3, 9.94, .02), (1.4, 9.7, .016),
            (1.45, 9.52, .01), (1.46, 9.42, 0)],
})
CH_LOOK = {'R2': (HAIR, 0.04), 'R1': (HAIR_LT, 0.03), 'R3': (HAIR_DK, 0.035), 'R4': (HAIR_LT, 0.032),
           'L2': (HAIR, 0.04), 'L1': (HAIR_LT, 0.03), 'L3': (HAIR_DK, 0.035),
           'RS1': (HAIR, 0.075), 'RS2': (HAIR_LT, 0.06), 'RS3': (HAIR_LT, 0.035),
           'LS1': (HAIR, 0.075), 'LS2': (HAIR_LT, 0.06), 'LS3': (HAIR, 0.035)}
# the shoulder locks are full clumps: their widths grown by a third (about three times as wide as they are thick)
for _k in ('RS1', 'RS2', 'LS1', 'LS2'):
    CH_STRANDS[_k] = [(x_, z_, w_ * 1.3) for x_, z_, w_ in CH_STRANDS[_k]]
CH_GAP = 0.02           # between a strand's underside and the cloth it lies on
CH_ON = 0.006           # and a strand it lies on
CH_JAW = 0.12           # how far forward of the body's centre line a strand is where it comes out from under the jaw
CH_ROOT = (0.34, 0.0, 11.66)   # where a strand starts, hidden inside the head behind the cheek (|x|, y, z)
CH_ROOTS = {k: (0.5, 0.12, 11.62) for k in ('RS1', 'RS2', 'RS3', 'LS1', 'LS2', 'LS3')}    # the shoulder locks: further back
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
        rt = CH_ROOTS.get(key, CH_ROOT)
        root = [[rt[0] * sd, rt[1], rt[2]]]
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


# ---------- the hair down the back ----------
# Laid out as the user's back-view picture of her draws it, with the strands made as the chest locks are (bundles of
# strands of different widths, curling at their ends) and the bangs (blades with pointed tips): below the braid the
# hair falls straight and narrows down her back, and parts above the corset into two groups beside the sash that swing
# out a little, back in, and end in C-curls about the knees (her left group a little longer); the rest, from the
# braid's ends behind the ears, drapes over the shoulders and the backs of the upper arms and fans out beyond the
# forearms, the widest at the ends, curling at its tips (BK_SIDE). All of it lies facing back, as drawn (round her
# sides, as the long side locks were, it was seen edge-on from behind, as tentacles). The waves and curls lie flat in
# the sheet of the hair, as drawn; none twists about its own line or winds into a corkscrew (the user found those
# twisted and weird; seven clumps evenly fanned read as piled up). Each strand is (x, z, half-width) in the back view
# down its centre line, from under the braid to its tip, read off the picture (calibrated on the head's top, the
# waist and her hands: 207 px a unit), as the chest locks' CH_STRANDS.
# The long hair is modelled as the bangs are (the user found the bangs perfectly anime and the back, waving, hooked and
# outlined, not): blades that hang in one smooth curve, no S-waves, each tapering to a sharp point, with the bangs'
# ridged crescent section (lcs),
# painted as they are (see paint_hair): thin dark strand lines down the lower part, a light peach sheen lens in the
# middle and the shade under the braid; no outlines. (The waves' code is kept with amp 0: its phase at height z is
# 2 pi (ZHC - z) / lambda + phase, and it still picks which way a tip flicks.)
HAIR_WAVE = dict(lam=1.6, phase=0.6, amp=0.0, z_top=9.6, z_full=8.0, narrow=0.0)
HAIR_OUTLINE, OUTLINE_W, OUTLINE_LIFT = '#7a3020', 0, 0.004     # (line art: OUTLINE_W pixels; none, as the bangs)


# The long locks' tips curl in, as the chest locks' do (the user's reference for natural hair): over the last 0.45 the
# lock turns about 85 degrees in toward the back in the sheet it lies in, in a smooth C (its curvature growing gently,
# as s^0.5), still wide where it starts to turn and closing to a point (ltl). (Hooks of 100 to 130 degrees that
# tightened at the very end, on a thin tip, read as fish hooks; flicks of 40 degrees on long points, with the locks
# piled side by side, as stiff.)
HAIR_CURL = dict(turn=1.5, length=0.45, p=0.5)


def _hook(turn, length, p=None, n=97):
    """A hook of this length turning `turn` radians in all, its bend growing as (s / length)^p: arc lengths s, headings
    th (from straight on) and the offsets across (lat) and along (fwd) from where it starts."""
    p = HAIR_CURL['p'] if p is None else p
    s = np.linspace(0, length, n)
    th = turn * (s / length) ** (p + 1)
    mid = 0.5 * (th[1:] + th[:-1])
    lat = np.r_[0, np.cumsum(np.sin(mid) * np.diff(s))]
    fwd = np.r_[0, np.cumsum(np.cos(mid) * np.diff(s))]
    return s, th, lat, fwd


from hairflow import flow as _flow_strand     # each strand from its sweep, widths, wave and curl, as drawn

# the waves (hairflow: amplitude, wavelength, phase, from z, full by z), one to a group so that its strands flow together
_WL = (0.13, 1.5, 0.4, 8.4, 7.2)       # her left group beside the sash
_WR = (0.13, 1.5, 2.0, 8.4, 7.2)
_WO = (0.16, 1.5, 0.6, 8.6, 7.2)       # the side hair's
BK_FLOW = {
    'BLm': (dict(pts=[(0.02, 11.42), (0.02, 10.6), (0.05, 9.8), (0.06, 9.15)], widths=[(11.42, .2), (10.6, .26), (9.9, .24), (9.5, .14), (9.15, 0)])),
    'BRm': (dict(pts=[(-0.1, 11.4), (-0.12, 10.6), (-0.15, 9.8), (-0.13, 9.28)], widths=[(11.4, .18), (10.6, .22), (9.9, .2), (9.6, .12), (9.28, 0)])),
    # her left inner group
    'BLu': dict(pts=[(0.15, 11.45), (0.3, 10.4), (0.55, 9.3), (0.75, 8.5), (0.9, 7.8), (0.95, 7.2), (0.9, 6.8)],
                widths=[(11.45, .2), (10.4, .3), (9.3, .27), (7.5, .26), (7.1, .2), (6.8, 0)]),
    'BLe': dict(pts=[(0.25, 11.5), (0.38, 10.5), (0.62, 9.4), (0.85, 8.7), (1.05, 8.0), (1.25, 7.5)],
                widths=[(11.5, .07), (10.5, .1), (9.4, .12), (8.0, .14)], wave=_WL, curl=(-1, 0.25, 4.0)),
    'BLb': dict(pts=[(0.3, 11.5), (0.45, 10.5), (0.7, 9.4), (0.95, 8.7), (1.15, 8.1), (1.4, 7.5), (1.52, 7.0)],
                widths=[(11.5, .07), (10.5, .11), (9.4, .13), (8.0, .15), (7.0, .15)], wave=_WL, curl=(-1, 0.3, 3.9)),
    'BLa': dict(pts=[(0.12, 11.45), (0.2, 10.5), (0.42, 9.4), (0.62, 8.8), (0.8, 8.2), (1.05, 7.6), (1.2, 7.1), (1.18, 6.7)],
                widths=[(11.45, .1), (10.5, .15), (9.4, .18), (8.2, .2), (6.8, .19)], wave=_WL, curl=(-1, 0.288, 3.6)),
    'BLd': dict(pts=[(0.2, 11.5), (0.3, 10.5), (0.55, 9.4), (0.72, 8.7), (0.85, 8.0), (0.95, 7.4), (0.85, 6.9), (0.72, 6.5)],
                widths=[(11.5, .07), (10.5, .1), (9.4, .12), (8.0, .13), (6.6, .13)], wave=_WL, curl=(1, 0.25, 3.7)),
    'BLc': dict(pts=[(0.05, 11.4), (0.1, 10.5), (0.32, 9.3), (0.45, 8.7), (0.48, 8.0), (0.42, 7.4), (0.38, 6.9), (0.4, 6.6)],
                widths=[(11.4, .07), (10.5, .11), (9.3, .13), (7.4, .13), (6.6, .12)], wave=_WL, curl=(1, 0.225, 3.5)),
    # her right inner group
    'BRu': dict(pts=[(-0.15, 11.4), (-0.3, 10.4), (-0.55, 9.3), (-0.75, 8.5), (-0.85, 7.8), (-0.9, 7.3), (-0.88, 6.95)],
                widths=[(11.4, .2), (10.4, .3), (9.3, .27), (7.8, .26), (7.25, .2), (6.95, 0)]),
    'BRe': dict(pts=[(-0.25, 11.45), (-0.38, 10.5), (-0.62, 9.4), (-0.82, 8.6), (-1.0, 7.9), (-1.18, 7.45)],
                widths=[(11.45, .07), (10.5, .1), (9.4, .12), (8.0, .14)], wave=_WR, curl=(1, 0.25, 3.9)),
    'BRb': dict(pts=[(-0.28, 11.45), (-0.42, 10.5), (-0.68, 9.4), (-0.9, 8.7), (-1.05, 8.0), (-1.3, 7.45), (-1.48, 7.0), (-1.5, 6.85)],
                widths=[(11.45, .07), (10.5, .11), (9.4, .13), (8.0, .14), (7.0, .14)], wave=_WR, curl=(1, 0.275, 3.7)),
    'BRa': dict(pts=[(-0.1, 11.4), (-0.18, 10.5), (-0.42, 9.4), (-0.62, 8.8), (-0.72, 8.1), (-0.95, 7.5), (-1.15, 7.1), (-1.2, 6.85)],
                widths=[(11.4, .1), (10.5, .15), (9.4, .18), (8.1, .2), (6.9, .19)], wave=_WR, curl=(1, 0.263, 3.5)),
    'BRd': dict(pts=[(-0.2, 11.42), (-0.3, 10.5), (-0.55, 9.4), (-0.75, 8.6), (-0.85, 7.9), (-0.78, 7.35), (-0.8, 6.95), (-0.9, 6.75)],
                widths=[(11.42, .07), (10.5, .1), (9.4, .12), (7.9, .12), (6.8, .11)], wave=_WR, curl=(-1, 0.225, 3.5)),
    'BRc': dict(pts=[(-0.02, 11.38), (-0.06, 10.5), (-0.3, 9.3), (-0.42, 8.6), (-0.4, 7.9), (-0.48, 7.2), (-0.55, 6.8), (-0.55, 6.6)],
                widths=[(11.38, .07), (10.5, .11), (9.3, .13), (7.2, .12), (6.6, .11)], wave=_WR, curl=(-1, 0.213, 3.4)),
    # her left side hair: thin over the capelet, a mass below, waving, curling at different heights
    'BLou': dict(pts=[(0.58, 11.3), (0.85, 10.75), (1.2, 10.1), (1.6, 9.4), (2.0, 8.7), (2.4, 8.0), (2.7, 7.3), (2.85, 6.6), (2.8, 6.0)],
                 widths=[(11.3, .14), (10.5, .16), (10.1, .12), (9.3, .18), (8.7, .3), (7.0, .34), (6.4, .3), (6.0, 0)]),
    'BLo3': dict(pts=[(0.6, 11.3), (0.88, 10.7), (1.25, 10.0), (1.65, 9.3), (2.05, 8.6), (2.5, 7.9), (2.9, 7.2), (3.1, 6.6), (3.1, 6.15)],
                 widths=[(11.3, .04), (10.0, .07), (9.3, .12), (8.6, .2), (7.0, .22), (6.2, .2)], wave=_WO, curl=(-1, 0.312, 3.8)),
    'BLo2': dict(pts=[(0.55, 11.35), (0.8, 10.8), (1.1, 10.2), (1.45, 9.5), (1.85, 8.9), (2.25, 8.2), (2.55, 7.5), (2.75, 6.8), (2.8, 6.2), (2.74, 5.9)],
                 widths=[(11.35, .05), (10.2, .08), (9.5, .13), (8.9, .22), (7.5, .26), (6.0, .24)], wave=_WO, curl=(-1, 0.325, 3.7)),
    'BLo6': dict(pts=[(0.56, 11.32), (0.84, 10.76), (1.15, 10.12), (1.52, 9.45), (1.95, 8.75), (2.35, 8.05), (2.62, 7.4)],
                 widths=[(11.32, .04), (10.1, .07), (9.4, .12), (8.7, .19), (7.5, .2)], wave=_WO, curl=(1, 0.275, 3.9)),
    'BLo5': dict(pts=[(0.52, 11.33), (0.78, 10.78), (1.05, 10.15), (1.38, 9.55), (1.75, 8.9), (2.1, 8.25), (2.4, 7.6), (2.55, 7.0), (2.55, 6.55)],
                 widths=[(11.33, .04), (10.15, .07), (9.55, .11), (8.9, .18), (7.6, .2), (6.6, .19)], wave=_WO, curl=(1, 0.275, 3.5)),
    'BLo4': dict(pts=[(0.5, 11.35), (0.75, 10.8), (1.0, 10.2), (1.3, 9.6), (1.6, 9.0), (1.95, 8.3), (2.15, 7.6), (2.3, 6.9), (2.35, 6.3)],
                 widths=[(11.35, .04), (10.2, .07), (9.6, .1), (9.0, .16), (7.6, .19), (6.3, .18)], wave=_WO, curl=(-1, 0.3, 3.8)),
    'BLo1': dict(pts=[(0.62, 11.3), (0.9, 10.8), (1.2, 10.2), (1.6, 9.5), (2.05, 9.0), (2.5, 8.5), (2.9, 8.0), (3.2, 7.4), (3.35, 6.9)],
                 widths=[(11.3, .03), (10.0, .05), (8.0, .07), (6.9, .07)], wave=_WO, curl=(-1, 0.275, 3.9)),
    'BLo7': dict(pts=[(0.6, 11.3), (0.9, 10.72), (1.28, 10.02), (1.7, 9.3), (2.15, 8.6), (2.62, 7.9), (3.0, 7.3), (3.18, 6.95)],
                 widths=[(11.3, .03), (10.0, .06), (9.3, .1), (8.6, .16), (7.4, .17)], wave=_WO, curl=(1, 0.3, 3.9)),
    # her right side hair
    'BRou': dict(pts=[(-0.58, 11.3), (-0.88, 10.75), (-1.25, 10.1), (-1.65, 9.4), (-2.05, 8.7), (-2.42, 8.0), (-2.68, 7.3), (-2.8, 6.6), (-2.74, 6.05)],
                 widths=[(11.3, .14), (10.5, .16), (10.1, .12), (9.3, .18), (8.7, .3), (7.0, .34), (6.45, .3), (6.05, 0)]),
    'BRo3': dict(pts=[(-0.6, 11.3), (-0.9, 10.7), (-1.3, 10.0), (-1.7, 9.3), (-2.1, 8.6), (-2.5, 7.9), (-2.85, 7.2), (-3.05, 6.6), (-3.0, 6.2)],
                 widths=[(11.3, .04), (10.0, .07), (9.3, .12), (8.6, .2), (7.0, .22), (6.2, .2)], wave=_WO, curl=(1, 0.312, 3.7)),
    'BRo2': dict(pts=[(-0.55, 11.35), (-0.85, 10.8), (-1.2, 10.15), (-1.55, 9.5), (-1.95, 8.85), (-2.35, 8.1), (-2.65, 7.4), (-2.8, 6.7), (-2.78, 6.2), (-2.7, 5.95)],
                 widths=[(11.35, .05), (10.15, .08), (9.5, .13), (8.85, .22), (7.4, .26), (6.0, .24)], wave=_WO, curl=(1, 0.325, 3.6)),
    'BRo6': dict(pts=[(-0.56, 11.32), (-0.86, 10.76), (-1.2, 10.12), (-1.58, 9.45), (-2.0, 8.75), (-2.38, 8.0), (-2.62, 7.35)],
                 widths=[(11.32, .04), (10.1, .07), (9.4, .12), (8.7, .19), (7.5, .2)], wave=_WO, curl=(-1, 0.275, 3.9)),
    'BRo5': dict(pts=[(-0.52, 11.33), (-0.8, 10.78), (-1.1, 10.15), (-1.42, 9.55), (-1.8, 8.9), (-2.15, 8.2), (-2.42, 7.5), (-2.55, 6.9), (-2.5, 6.5)],
                 widths=[(11.33, .04), (10.15, .07), (9.55, .11), (8.9, .18), (7.5, .2), (6.5, .19)], wave=_WO, curl=(-1, 0.275, 3.5)),
    'BRo4': dict(pts=[(-0.5, 11.35), (-0.78, 10.8), (-1.05, 10.2), (-1.35, 9.6), (-1.65, 9.0), (-1.95, 8.3), (-2.1, 7.6), (-2.2, 6.9), (-2.25, 6.4)],
                 widths=[(11.35, .04), (10.2, .07), (9.6, .1), (9.0, .16), (7.6, .19), (6.4, .18)], wave=_WO, curl=(1, 0.288, 3.7)),
    'BRo1': dict(pts=[(-0.62, 11.3), (-0.95, 10.75), (-1.3, 10.1), (-1.75, 9.4), (-2.2, 8.8), (-2.65, 8.2), (-3.0, 7.6), (-3.25, 7.1), (-3.3, 6.75)],
                 widths=[(11.3, .03), (10.0, .05), (8.0, .07), (6.75, .07)], wave=_WO, curl=(1, 0.275, 3.8)),
    'BRo7': dict(pts=[(-0.6, 11.3), (-0.92, 10.72), (-1.32, 10.02), (-1.75, 9.3), (-2.2, 8.6), (-2.65, 7.9), (-3.0, 7.25), (-3.15, 6.9)],
                 widths=[(11.3, .03), (10.0, .06), (9.3, .1), (8.6, .16), (7.4, .17)], wave=_WO, curl=(-1, 0.3, 3.9)),
}
BK_STRANDS = {k: _flow_strand(**v) for k, v in BK_FLOW.items()}
BK_SIDE = tuple(k for k in BK_STRANDS if k[2] == 'o')     # the side hair (from the braid's ends, draped: back_strands)
# where each curl starts (the index of its first point in BK_STRANDS, the sweep's last) and its sense
BK_CURL_AT = {k: (len(_flow_strand(**{**v, 'curl': None})) - 1, v['curl'][0]) for k, v in BK_FLOW.items() if v.get('curl')}
BK_STRANDS['BR0'] = [(-0.42, 11.35, .055), (-0.45, 11.1, .074), (-0.47, 10.85, .055), (-0.47, 10.74, 0)]   # a short strand by the braid
# the curls: ringlets, each a helix about an upright axis through x cx, starting where the strand's hanging part
# ends; r0 the radius it starts at (where the strand ends at least that far from the axis across, it starts beside
# the axis, else in front of it), r1 the radius it closes to, turns, how far it drops (about 0.6 a turn, so that the
# turns of two strands round one axis stand apart), and the sense of the turn (seen from above: 1 anticlockwise).
# The strands of a lock with the same cx share the axis, set back far enough for every turn to clear the skirt
# (ringlet_axes). (The user found the corkscrew ringlets unnatural; none is used. The ringlet code is kept for a
# strand given a curl here.)
BK_CURL = {}


def _under_braid(x0, sd, z_tip, k=1.0):
    """A short strand under the braid at x0 (sd 1 her left, -1 her right), k times the usual width: it starts right
    under the braid (back_strands adds the top), runs down the back of the head moving out 0.12 a unit of height, and
    ends in a tip at z_tip."""
    pts = [(x0 + sd * 0.12 * (11.5 - z), z, k * hw) for z, hw in zip((11.5, 11.2, 10.9, 10.6, 10.35, 10.1), (.055, .06, .06, .055, .05, .045))
           if z > z_tip + 0.12]
    return pts + [(x0 + sd * 0.12 * (11.5 - z_tip), z_tip, 0)]


# and short strands under the braid, over the back of the head beside the curtain (the user's highlight: the crown
# locks close in toward the middle there and left the dark under-layer showing at |x| 0.1 to 0.65, from the braid down
# to the capelet), two a side, wide, to tips on the capelet
BK_STRANDS.update({f'BL{9 + i}': _under_braid(x0, 1, zt, 2.96) for i, (x0, zt) in enumerate(((0.4, 10.0), (0.62, 10.45)))})
BK_STRANDS.update({f'BR{8 + i}': _under_braid(x0, -1, zt, 2.96) for i, (x0, zt) in enumerate(((-0.2, 9.85), (-0.48, 10.3)))})
# colour and thickness, in the order they are laid on: the clumps under, darker (so the folds between the strands
# read), then the strands of each group, each lying on the ones before it, in shades of their own
BK_LOOK = {'BL10': (HAIR_DK, 0.04), 'BL9': (HAIR, 0.04), 'BR9': (HAIR_DK, 0.04), 'BR8': (HAIR, 0.04),
           'BLu': (HAIR_DK, 0.042), 'BRu': (HAIR_DK, 0.042), 'BLou': (HAIR_DK, 0.042), 'BRou': (HAIR_DK, 0.042),
           'BR0': (HAIR_LT, 0.04), 'BLm': (HAIR_LT, 0.042), 'BRm': (HAIR, 0.04),
           'BLe': (HAIR, 0.04), 'BRe': (HAIR_LT, 0.04),
           'BLb': (HAIR_LT, 0.042), 'BRb': (HAIR, 0.042), 'BLa': (HAIR, 0.045), 'BRa': (HAIR_LT, 0.045),
           'BLd': (HAIR_DK, 0.04), 'BRd': (HAIR_DK, 0.04), 'BLc': (HAIR_LT, 0.04), 'BRc': (HAIR, 0.04),
           'BLo3': (HAIR_DK, 0.042), 'BRo3': (HAIR, 0.042), 'BLo2': (HAIR, 0.045), 'BRo2': (HAIR_LT, 0.045),
           'BLo6': (HAIR, 0.042), 'BRo6': (HAIR_LT, 0.042),
           'BLo7': (HAIR_LT, 0.038), 'BRo7': (HAIR, 0.038),
           'BLo5': (HAIR_LT, 0.04), 'BRo5': (HAIR_DK, 0.04), 'BLo4': (HAIR_LT, 0.04), 'BRo4': (HAIR, 0.04),
           'BLo1': (HAIR_LT, 0.03), 'BRo1': (HAIR_LT, 0.03)}
BK_LOOK = {k: (c, round(th * 2.3, 4)) for k, (c, th) in BK_LOOK.items()}
BK_ROOT = (0.88, 11.87)     # the strands start inside the braid at the back of the head (y, z; measured: the braid's
                            # outside is at y 0.96 and z 11.84 to 11.9 in the middle)
BK_TOP = 0.96               # how far back a strand is where it comes out under the braid
BK_NAPE = 10.45             # the hair down the back above the two locks ends here (back locks, hairUnder)
BK_WIDE = 1.0               # the half-widths are as drawn (before, the traced ones were drawn 1.85 times as wide)
BK_WAVE = (0.0, 0.7, 8.6)   # (below z 8.6 a strand may wave back and forth by the first, the second a wave: none, as
                            # neat anime hair falls smooth)
BK_NOISE = (0.005, 0.03, 0.03, 0.005)  # so that no strand is machine-smooth: each wanders across by up to 0.005, its
                            # width and thickness vary by up to 3%, and it lifts off what it lies on by up to 0.005 (more
                            # read as lumps along the big clumps)
BK_TWIST = {}               # turns a strand twists about its own line below the nape (none: neat hair lies flat)
BK_UNDER = ('BL9', 'BL10', 'BR8', 'BR9')   # the strands under the braid
BK_DRAPE_Z = 10.0          # the side hair lies on what is under it above this (coming forward from the head onto the
                            # shoulders); below it, it falls straight (never coming forward: draped down to 9.0, it
                            # creased where it left the capelet's hem)
BK_WRAP = (0.1, 10.4, 9.4)   # in 3D the long hair wraps round her back and arms (the user: drawn from the back view
                            # alone, every strand lay in one plane, which was unnatural): below z 10.4, fully by 9.4, a
                            # strand comes forward by 0.1 (x^2 - x0^2) from where it would hang, x0 where it starts down,
                            # so that it curves round as it fans out (the side hair's ends at |x| 3, 0.75 forward), and
                            # faces out from the curve
BK_DWAVE = (0.07, 2.1, 9.4, 8.6)   # and it waves in depth too: by up to 0.07 either way, a wave 2.1 long, from z 9.4,
                            # full by 8.6; a phase to each group (BK_DPHASE) and 0.7 more to each strand in it, so that
                            # the strands of a group cross over and under each other (never into the ones beneath)
BK_DPHASE = {'BL': 0.3, 'BR': 2.1, 'BLo': 1.2, 'BRo': 4.0}
BK_CURL_TILT = math.radians(45)    # the curls turn out of the sheet, back and away from her, by this (from the side,
                            # loops; from the back, still C-curls): from 0.3 above the curl over 0.5, turning about the
                            # upright through that point
BK_SUBS = {k: 1 for k in BK_STRANDS}   # sub-strands: each strand is drawn as these, side by side across it, each with
                            # its own wander, depth, width, thickness and shade; they fan out at a free tip and wind
                            # together in a ringlet. One each: the locks are bundles of strands, as the chest locks are
BK_WOB = 0.02               # a ringlet's radius wanders by up to twice this
BK_STACK = (0.03, 0.2)      # how far the strands of a lock may rise over each other: on the head and nape, and below


_KG = (-2.8, 4.8, 0.02, 281, 341)       # the clothes grid: x from -2.8, z from 4.8, every 0.02


def _back_grid(surfs, spec):
    """How far back (largest y) the surfaces [(ref, colour, f, lim)] reach at each (x, z) of the grid spec (x0, z0,
    step, nx, nz), -9 where none: each surface sampled about every 0.012, the grid then closed over two cells."""
    import sheetfit
    x0, z0, st, nx, nz = spec
    G = np.full((nx, nz), -9.0)
    for _, _, f, lim in surfs:
        P = sheetfit._grid(f, lim, 20, 20)
        Q = P[np.isfinite(P).all(-1)]
        if not len(Q) or Q[:, 2].max() < z0 or Q[:, 2].min() > z0 + st * nz or Q[:, 1].max() < -0.3:
            continue
        lu = np.nanmax(np.nansum(np.linalg.norm(np.diff(P, axis=0), axis=-1), axis=0))
        lv = np.nanmax(np.nansum(np.linalg.norm(np.diff(P, axis=1), axis=-1), axis=1))
        P = sheetfit._grid(f, lim, int(np.clip(lu / 0.012, 8, 500)), int(np.clip(lv / 0.012, 8, 500))).reshape(-1, 3)
        P = P[np.isfinite(P).all(1)]
        i, j = np.round((P[:, 0] - x0) / st).astype(int), np.round((P[:, 2] - z0) / st).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
        np.maximum.at(G, (i[ok], j[ok]), P[ok, 1])
    for _ in range(2):
        G = _grow(G, -9.0)
    return G


def _clothes_back():
    """How far back (largest y) her clothes reach at each (x, z) of the grid _KG, from the clothes' own surfaces:
    the generator run for the clothes alone (sheetfit.model_surfaces, about 3 s), each surface sampled about every
    0.012, so that the hair is the same whichever parts are drawn. Gaps in the cloth across a row (between the
    corset and the arms) take the depth across them, between the cloth on either side; outside it, -9."""
    import sheetfit, contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):        # its own report of the expanded formulas
        clothes = sheetfit.model_surfaces('body,cape,corset,skirt,arms,book')
    G = _back_grid(clothes, _KG)
    x0, z0, st, nx, nz = _KG
    for j in range(nz):
        on = np.nonzero(G[:, j] > -9)[0]
        if len(on) > 1:
            G[on[0]:on[-1] + 1, j] = np.interp(np.arange(on[0], on[-1] + 1), on, G[on, j])
    return G


CLOTHES_BACK = _clothes_back()


def clothes_y(x, z):
    """How far back her clothes reach at (x, z) (arrays), -9 where there are none."""
    x0, z0, st, nx, nz = _KG
    i, j = np.round((np.asarray(x) - x0) / st).astype(int), np.round((np.asarray(z) - z0) / st).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
    return np.where(ok, CLOTHES_BACK[np.clip(i, 0, nx - 1), np.clip(j, 0, nz - 1)], -9.0)


BRAID_RING = (68 * math.pi / 180, math.tan(66 * math.pi / 180), math.tan(32 * math.pi / 180))   # the braid's ring (run_hair)


def braid_back(x):
    """Where the braid crosses the back of the head at x (arrays): the height of its centre line, how far back it is,
    and the height of its lower edge (0.09 below the centre line), from its ring as run_hair draws it, turned with
    the head."""
    thc, th_, tv_ = BRAID_RING
    a = np.linspace(0, 2 * math.pi, 721)
    d = np.array([th_ * np.cos(a), math.sin(thc) + tv_ * math.cos(thc) * np.sin(a), math.cos(thc) - tv_ * math.sin(thc) * np.sin(a)])
    d /= np.linalg.norm(d, axis=0)
    turn = (HEAD_TURN['pivot'], HEAD_TURN['axis'], HEAD_TURN['angle'], None, HEAD_TURN['scale'])
    P = np.array([sc.turn_point((HDX * 1.13 * d[0, i], HY0 + HDY * 1.13 * d[1, i], ZHC + HDZ * 1.13 * d[2, i]), turn) for i in range(len(a))])
    P = P[P[:, 1] > HY0]                                 # the half round the back of the head
    xs = np.arange(-1.0, 1.001, 0.02)
    low = []                                             # seen from behind the ring is an oval: at each x its lower crossing
    for xb in xs:
        m = np.abs(P[:, 0] - xb) < 0.02
        if m.any():
            i = np.argmin(np.where(m, P[:, 2], np.inf))
            low.append((xb, P[i, 1], P[i, 2]))
    L = np.array(low)
    z = np.interp(x, L[:, 0], L[:, 2])
    return z, np.interp(x, L[:, 0], L[:, 1]), z - 0.09


_HG = (-1.0, 10.2, 0.02, 101, 96)      # the grid of the hair already on the back of the head
HEAD_BACK = None
HEAD_ROOT_SURFACES = []
HEAD_CONTACT = None
HEAD_UNDER_CONTACT = None
BK_JOIN_DIAGNOSTICS = {}


def head_back_grid():
    """How far back the hair already drawn on the back of the head reaches (the under-layer, cap, crown and side
    locks: the generator's own lines so far, evaluated by sheetfit), so that the back locks lie on what is there."""
    import sheetfit, re
    hair = sheetfit.surfaces_from_lines(sc.out, ref=r'(hairUnder|hairCap|back[UT]?\d+|side[LR]\d+|tuck\d+)(Curl)?')
    global HEAD_ROOT_SURFACES, HEAD_CONTACT, HEAD_UNDER_CONTACT
    HEAD_ROOT_SURFACES = [s for s in hair if re.fullmatch(r'back[UT]?\d+', s[0])]
    from haircontact import CrownContact
    HEAD_CONTACT = CrownContact(hair, zmin=10.7)
    HEAD_UNDER_CONTACT = CrownContact([s for s in hair if s[0]=='hairUnder'], zmin=10.7)
    global LOCKS_BACK               # the long locks over the shoulders and arms, for the side hair to lie on
    LOCKS_BACK = _back_grid(sheetfit.surfaces_from_lines(sc.out, ref=r'(back[UT]?\d+|side[LR]\d+|earBackLock\d)(Tip)?'), _LG)
    return _back_grid(hair, _HG)


def locks_back(x, z):
    """How far back the long locks over the shoulders and arms reach at (x, z) (arrays), -9 where there are none."""
    if LOCKS_BACK is None:
        return np.full(np.shape(x), -9.0)
    x0, z0, st, nx, nz = _LG
    i, j = np.round((np.asarray(x) - x0) / st).astype(int), np.round((np.asarray(z) - z0) / st).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
    return np.where(ok, LOCKS_BACK[np.clip(i, 0, nx - 1), np.clip(j, 0, nz - 1)], -9.0)


def head_root(x, z):
    """Actual crown-hair point and downward tangent near a braid attachment.

    Seed the flow from a nearby crown crest, but retain the traced X/Z wherever
    actual hair supports it. Evaluate depth and tangent plane on that actual
    outer envelope, not a dilated depth grid; UV records the flow seed.
    """
    import sheetfit
    solutions = []
    for name, _, f, lim in HEAD_ROOT_SURFACES:
        # w in [0, pi] is the outer crescent of a crown lock.
        # Attach to the crest of an upper strand, never a nearly vertical cut
        # edge of its crescent. An edge's X/Z inverse is ill-conditioned and can
        # make a lower strand visibly grow sideways before it drops.
        tt = np.linspace(lim[0]+1e-5, lim[1]-1e-5, 160)
        P = np.array(np.broadcast_arrays(*f(tt, np.full_like(tt, math.pi/2)))).T
        score = (P[:, 0]-x)**2 + (P[:, 2]-z)**2
        uv = np.array([tt[np.nanargmin(score)], math.pi/2])
        def point(v): return np.array(f(float(v[0]), float(v[1])), float)
        def deriv(v, axis):
            d = np.zeros(2); d[axis] = 1e-5
            return (point(v+d)-point(v-d))/(2e-5)
        for _ in range(8):
            p, du = point(uv), deriv(uv, 0)
            if abs(du[2]) < 1e-6: break
            uv[0] = np.clip(uv[0]-(p[2]-z)/du[2], lim[0]+1e-5, lim[1]-1e-5)
        p, tangent = point(uv), deriv(uv, 0)
        if tangent[2] < -0.01:
            normal = np.cross(tangent, deriv(uv, 1)); normal /= np.linalg.norm(normal)
            if normal[1] < 0: normal = -normal
            solutions.append((float(np.linalg.norm(p[[0, 2]]-[x, z])), p, tangent, normal, name, uv))
    if not solutions:
        raise ValueError('No downward crown hair surface at braid root')
    best = min(s[0] for s in solutions)
    _, p, tangent, normal, name, uv = max((s for s in solutions if s[0] <= best+0.008), key=lambda s: s[1][1])
    # Keep the traced attachment's X/Z when actual upper hair exists there.
    # Snapping every filler to a nearby crest clusters their roots and leaves
    # whole uncovered bands (e.g. x=.48 and -.50) below the braid.
    if np.isfinite(HEAD_CONTACT.depth_smooth(x, z)):
        p[[0, 2]] = [x, z]
    # Avoid an X/Z silhouette cell with no two-dimensional neighbourhood: a
    # root can be on the actual surface yet its depth graph be singular there.
    # Move at most a few grid cells inward, then reproject to the actual hair.
    for inset in (0., .004, .008, .012, .02, .03):
        px = math.copysign(max(0., abs(p[0])-inset), p[0])
        if np.isfinite(HEAD_CONTACT.depth_smooth(px, p[2])):
            p[0] = px
            break
    else:
        raise ValueError(f'No regular crown neighbourhood near {(x,z)}')
    # A cap/side patch can lie outside the chosen crown strand. Anchor to the
    # full visible envelope, not to a patch hidden underneath it.
    H = HEAD_CONTACT.depth(p[0], p[2])
    hx, hz = HEAD_CONTACT.gradient(p[0], p[2])
    if np.all(np.isfinite([H, hx, hz])):
        p[1] = H
        tangent[1] = hx*tangent[0]+hz*tangent[2]
        normal = np.array([-hx, 1., -hz]); normal /= np.linalg.norm(normal)
    else:
        raise ValueError(f'Non-finite native crown jet at {p.tolist()}')
    return p, tangent, normal, name, uv


def body_back(x, z, head=True):
    """How far back (+y) what the back locks hang on reaches at (x, z) (arrays): her clothes, and (with head) the
    hair above the nape (the back locks, which end at BK_NAPE): as drawn (HEAD_BACK), else its shell and a margin."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    f = clothes_y(x, z)
    if not head:
        return f
    if HEAD_BACK is not None:
        x0, z0, st, nx, nz = _HG
        i, j = np.round((x - x0) / st).astype(int), np.round((z - z0) / st).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
        h = np.where(ok, HEAD_BACK[np.clip(i, 0, nx - 1), np.clip(j, 0, nz - 1)], -9.0)
        return np.where(z > BK_NAPE - 0.05, np.maximum(f, h), f)
    t = T0 + (ZHC - z) / VF
    hp = lambda c: sum(a * np.maximum(0.0, t - T0 - k) ** 2 for a, k in zip(c, KNOTS))
    ax, ay = HDX + hp(HP_X[2]) + 0.1, HDY + hp(HP_Y[2]) + 0.1          # the outside of the hair there (measured)
    nape = HY0 + ay * np.sqrt(np.maximum(0, 1 - (x / ax) ** 2))
    return np.where((z > BK_NAPE - 0.05) & (z < 11.95) & (np.abs(x) < ax), np.maximum(f, nape), f)


_UG = (-3.6, 4.8, 0.01, 721, 711)      # the grid of how far back the back strands already laid reach
_LG = (-3.6, 8.0, 0.02, 361, 181)      # and of the long locks over the shoulders and arms (LOCKS_BACK)
LOCKS_BACK = None


def ringlet_axes():
    """The ringlets' axes (x, y), one for each lock's strands with the same cx: as close to the clothes as every turn
    of every one of them allows (its radius closing from r0 to r1 as it drops, 0.07 for its ribbon and a gap), and
    far enough back for each strand, falling straight from the fullest point it hangs over, to start its ringlet
    (beside the axis, or in front of it at radius r0)."""
    need = {}
    for key, c in BK_CURL.items():
        D = np.array(BK_STRANDS[key], float)
        dd = np.r_[0, np.cumsum(np.hypot(*np.diff(D[:, :2], axis=0).T))]
        ss = np.linspace(0, dd[-1], 200)
        px, pz, hw = (np.interp(ss, dd, D[:, i]) for i in range(3))
        hang = max(BK_TOP, max(float(body_back(px + k * hw * BK_WIDE, pz).max()) for k in (-1, 0, 1)) + CH_GAP) + 0.1
        dx = abs(D[-1, 0] - c['cx'])
        g = (key[1], c['cx'])
        need[g] = max(need.get(g, -9.0), hang + (math.sqrt(c['r0'] ** 2 - dx * dx) if dx < 0.9 * c['r0'] else 0.0))
        z1 = BK_STRANDS[key][-1][1]
        zz = np.linspace(z1 - c['drop'] - 0.1, z1 + 0.05, 40)
        rr = np.interp(zz, [z1 - c['drop'], z1], [c['r1'], c['r0']])
        X = c['cx'] + np.linspace(-1, 1, 41)[None, :] * rr[:, None]
        Z = zz[:, None] + 0 * X
        y = clothes_y(X, Z) + np.sqrt(np.maximum(0, rr[:, None] ** 2 - (X - c['cx']) ** 2)) + 0.07
        g = (key[1], c['cx'])
        need[g] = max(need.get(g, -9.0), float(y.max()))
    return {key: (c['cx'], need[(key[1], c['cx'])]) for key, c in BK_CURL.items()}


def _lin(terms):
    """Sum of coefficient * expression, leaving out the zero terms."""
    out = [f"{fmt(c)}*{e}" if e else fmt(c) for c, e in terms if abs(c) > 5e-6]
    return '+'.join(out) if out else '0'


def curl_xyz(P, T, Sv, Nv, spec, subs):
    """A ringlet written exactly for Arch, as a function of t from -1 to 1 (its angle th = Th (t + 1) / 2): a helix
    about the upright axis through spec['at'] (x, y), starting at the end P of the strand's hanging part with its
    tangent T, side vector Sv and thickness vector Nv. It falls fast as it leaves the hanging part (matching its
    slope, at most 0.8 a radian, but over no more than 35% of the drop) and then evenly, spec['drop'] in all. Its
    section turns (the short way) to face out from the axis over the first half turn and its width turns to lie
    along the axis, narrowing to 0.7 so that the turns stand apart. The radius closes linearly to r1; width and
    thickness taper over the last half turn. Returns one (x, y, z) for each sub-strand in subs: placed across the
    strand (d, in its half-widths) and out from it (dep, in its thicknesses) as it is where its hanging part ends,
    starting exactly there (the rest r of its offset fades out over the first half turn), with its own width w,
    thickness t and a radius that wanders a little (by up to 2 BK_WOB at its own rate wob)."""
    e3 = np.array([0, 0, 1.0])
    Th = 2 * math.pi * spec['turns']
    O = np.array([spec['at'][0], spec['at'][1], P[2]])
    n0 = P - O
    n0[2] = 0
    r0 = float(np.linalg.norm(n0))
    n0 /= r0
    m0 = spec['sense'] * np.cross(e3, n0)
    hw, th = float(np.linalg.norm(Sv)), float(np.linalg.norm(Nv))
    sh, nh = Sv / hw, Nv / th
    sgs, sgn = (1.0 if sh @ e3 >= 0 else -1.0), (1.0 if nh @ n0 >= 0 else -1.0)
    TH = f"({fmt(Th / 2)}*(t+1))"
    CS, SN = f"cos{TH}", f"sin{TH}"
    ps = min(0.8, r0 * abs(T @ e3) / max(abs(T @ m0), 1e-3))
    k = min(0.6, 0.35 * spec['drop'] / max(ps, 1e-3))      # the steep start takes at most 35% of the drop
    ex = k * (1 - math.exp(-Th / k))
    pt = max(0.0, (spec['drop'] - ps * ex) / (Th - ex))
    A = f"(-({fmt(pt)}*{TH}+{fmt((ps - pt) * k)}*(1-exp(-{TH}/{fmt(k)}))))"
    b0 = 0.5 + 0.5 * math.tanh(-2.5)
    B = f"({fmt((0.5 - b0) / (1 - b0))}+{fmt(0.5 / (1 - b0))}*tanh(({TH}-1.5708)/0.6283))"   # 0 at the start, 1 a half turn on
    TP = f"tanh({fmt(Th / 2)}*(1-t)/1.5708)^0.6"         # (Th - th), written so it never goes below 0 at t = 1
    TK = f"(0.3+0.7*{TP})"
    out = []
    for sb in subs:
        R = f"({fmt(r0)}+{fmt((spec['r1'] - r0) / Th)}*{TH}+{fmt(BK_WOB)}*(1-cos({fmt(sb['wob'])}*{TH})))"
        xyz = []
        for i in range(3):
            rad = f"({_lin([(n0[i], CS), (m0[i], SN)])})"
            S = f"((1-{B})*{fmt(sh[i])}+{B}*{fmt(0.7 * sgs)})" if i == 2 else f"(1-{B})*{fmt(sh[i])}"
            N = f"((1-{B})*{fmt(nh[i])}+{B}*{fmt(sgn)}*{rad})"
            centre = _lin([(O[i], ''), (1.0, f"{R}*{rad}") if abs(n0[i]) + abs(m0[i]) > 5e-6 else (0, ''), (e3[i], A),
                           (sb['r'][i], f"(1-{B})"), (sb['d'] * hw, f"{TP}*{S}"), (sb['dep'] * th, f"{TK}*{N}")])
            xyz.append(f"{centre}+cos(w)*({fmt(sb['w'] * hw)}*{TP}*{S})+lcs(w)*({fmt(sb['t'] * th)}*{TK}*{N})")
        out.append(tuple(xyz))
    return out


def _poly_fit(t, V, deg, start_derivative=None):
    from hairjoin import fit_c1
    return fit_c1(t, V, deg, start_derivative=start_derivative)


def _back_sections(sig, P, W, TH, PH, TS, subs, root_normal, root_length, material_sig, psi=None):
    """Build finite sections from a native crown root onto the actual hair base (psi: how far each section turns
    about the upright from facing straight back, +y; none if not given).

    As the new section gains thickness its centre moves into the old section's
    depth, rather than piling another layer on top. Whole-section contact with
    hairUnder prevents a tangent crest from leaving bare patches underneath.
    """
    T = TS / np.linalg.norm(TS, axis=1)[:, None]
    psi = np.zeros(len(T)) if psi is None else np.asarray(psi, float)
    ref = np.c_[-np.sin(psi), np.cos(psi), np.zeros(len(T))]
    nrm = ref - np.sum(ref*T, axis=1)[:, None]*T
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    # W is a traced back-view (XZ) half-width, not a geodesic width on the
    # tilted crown. Preserve its projection; using T x native_normal here
    # would narrow outer roots by 10-40% and uncover the under-layer.
    side = np.cross(T, nrm); side /= np.linalg.norm(side, axis=1)[:, None]
    u = np.clip(sig/root_length, 0, 1)
    b = (6*u**5 - 15*u**4 + 10*u**3)[:, None]
    normal = (1-b)*root_normal + b*nrm
    nrm = normal - np.sum(normal*T, axis=1)[:, None]*T
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    nrm, side = np.cos(PH[:, None])*nrm+np.sin(PH[:, None])*side, np.cos(PH[:, None])*side-np.sin(PH[:, None])*nrm
    series = [(P.copy(), side*W[:, None], nrm*TH[:, None])]
    for sb in subs:
        lat, dep, wk, tk = _sub_at(sb, material_sig, material_sig/material_sig[-1])
        series.append((P+(lat*W)[:, None]*side+(dep*TH)[:, None]*nrm, side*(wk*W)[:, None], nrm*(tk*TH)[:, None]))
    root = sig < root_length
    anchor = P[0]
    base0 = HEAD_UNDER_CONTACT.depth(anchor[0], anchor[2])
    gx, gz = HEAD_UNDER_CONTACT.gradient(anchor[0], anchor[2])
    regular_base = np.all(np.isfinite([base0, gx, gz]))
    if not regular_base:
        # Some outer crests extend beyond the under-layer's angular domain.
        # There is no support to invent there; keep their native Hermite flow.
        base0 = anchor[1]
        gx, gz = -root_normal[[0, 2]]/root_normal[1]
    gradient_delta = -root_normal[[0, 2]]/root_normal[1] - [gx, gz]
    beta = b[root, 0]
    entry = np.clip(sig[root]/0.06, 0, 1)
    entry = 6*entry**5-15*entry**4+10*entry**3
    thick = np.clip(sig[root]/.08, 0, 1)
    thick = 6*thick**5-15*thick**4+10*thick**3
    for C, Uv, V in series:
        original_y = C[root, 1].copy()
        B = HEAD_UNDER_CONTACT.depth(C[root, 0], C[root, 2])
        valid = np.isfinite(B)
        # Continue freely beyond the silhouette, not on a fabricated y floor.
        B = np.interp(sig[root], sig[root][valid], B[valid]) if valid.any() else original_y.copy()
        dx, dz = C[root, 0]-anchor[0], C[root, 2]-anchor[2]
        offset = anchor[1]-base0 + (dx*gradient_delta[0]+dz*gradient_delta[1])*np.exp(-(dx*dx+dz*dz)/0.2**2)
        # Replace the upper lock's thickness, rather than adding a new layer on
        # top or flattening the entire lower lock into a paper-thin crest.
        # The actual crown height provides the available root section depth.
        available = np.maximum(offset, 0)*(1-thick)+thick*V[root, 1]
        scale = np.minimum(1., np.maximum(0., available/np.maximum(V[root, 1], 1e-12)))
        if regular_base:
            V[root] *= scale[:, None]
            C[root, 1] = B + (1-beta)*(offset-V[root, 1]) + beta*(original_y-B)
        plus, minus = C[root]+Uv[root], C[root]-Uv[root]
        hp = HEAD_UNDER_CONTACT.depth(plus[:, 0], plus[:, 2])
        hm = HEAD_UNDER_CONTACT.depth(minus[:, 0], minus[:, 2])
        ok = np.isfinite(hp) & np.isfinite(hm)
        if ok.any():
            # Continue a slope per unit width, not an absolute height offset.
            # An outer root may enter this support only later; carrying that
            # later offset backwards would give its zero-width root a y span.
            width = np.linalg.norm(Uv[root], axis=1)
            rate = .5*(hp[ok]-hm[ok])/np.maximum(width[ok], 1e-12)
            slope = np.interp(sig[root], sig[root][ok], rate)*width
            Uv[root, 1] = (1-beta)*slope + beta*Uv[root, 1]
        # Enforce contact on the whole outer section, not just its centre peak.
        # Missing projections are ignored: no head means no artificial floor.
        needed = np.zeros(root.sum())
        for angle in np.linspace(0, math.pi, 17):
            cs = (0.65*math.sin(angle)+0.35*abs(math.sin(angle)))*(0.78+0.22*math.cos(2*math.pi*math.cos(angle)))
            S = C[root] + math.cos(angle)*Uv[root]+cs*V[root]
            H = HEAD_UNDER_CONTACT.depth(S[:, 0], S[:, 2])
            required = np.where(np.isfinite(H), H+0.012*entry-S[:, 1], 0.)
            needed = np.maximum(needed, required)
        # A support's angular boundary can appear abruptly under a wide lock.
        # Anticipate contact with a smooth majorant instead of introducing a
        # vertical step there. Padded max growth, never cyclic np.roll.
        su = np.linspace(sig[root][0], sig[root][-1], len(needed))
        nu = np.interp(su, sig[root], needed)
        radius = max(1, int(math.ceil(.06/(su[1]-su[0]))))
        expanded = np.lib.stride_tricks.sliding_window_view(np.pad(nu, radius, mode='edge'), 2*radius+1).max(axis=1)
        kernel = np.exp(-.5*(np.arange(-radius, radius+1)/(radius/3))**2)
        majorant = np.convolve(np.pad(expanded, radius, mode='edge'), kernel/kernel.sum(), 'valid')
        needed = np.interp(sig[root], su, majorant)
        leave = np.clip((root_length-sig[root])/.12, 0, 1)
        leave = 6*leave**5-15*leave**4+10*leave**3
        needed *= entry*leave
        # C2 positive-part ramp, at most .001 below the strict desired clearance.
        eps = .002
        v = np.clip(needed/eps, 0., 1.)
        lift = np.where(needed>=eps, needed-.5*eps, eps*(v**3-.5*v**4))
        if lift[0] > 1e-8:
            raise ValueError('Under-layer contact would move the native root')
        C[root, 1] += lift
        if not all(np.all(np.isfinite(v)) for v in (C, Uv, V)):
            raise ValueError('Non-finite back strand contact section')
    return series


def _hang_pieces(n_seg, t, sig, prepared, joined, subs, end_targets=None, end_span=1, root_tangent=None):
    """The hanging part fitted in n_seg polynomial pieces that join exactly, for the strand itself (not drawn: its
    ringlet starts from its end) and for each of its sub-strands (_sub_at); returns the sub-strands' parts, the
    largest error of each piece and the last coefficients of the strand and of each sub-strand. The precomputed
    sections already include the crown contact and gradual transition to the old hanging/twisting frame."""
    parts, errs, prev = [[] for _ in subs], [], [None] * (len(subs) + 1)
    for k_ in range(n_seg):
        st = sig[-1] * (k_ + (t + 1) / 2) / n_seg
        series = [tuple(np.array([np.interp(st, sig, V[:, i]) for i in range(3)]).T for V in ser) for ser in prepared]
        err = 0.0
        for m, ser in enumerate(series):
            if k_ == 0 and root_tangent is not None:
                # The native jet is known analytically; do not estimate it
                # from samples across the short, braid-hidden width ramp.
                jet = np.asarray(root_tangent)/np.linalg.norm(root_tangent) * (sig[-1]/(2*n_seg))
                fits = [_poly_fit(t, V, CH_DEG, jet if j == 0 else np.zeros(3)) for j, V in enumerate(ser)]
            else:
                fits = [_poly_fit(t, V, CH_DEG) for V in ser]
            cs = [f[0] for f in fits]
            if prev[m] is not None:
                cs = [joined(c, [np.polynomial.polynomial.polyval(1.0, q) for q in pc], [np.polynomial.polynomial.polyval(-1.0, q) for q in c])
                      for c, pc in zip(cs, prev[m])]
            if end_targets is not None and k_ == n_seg-1:
                # Only the head-side join changes. Preserve the old ringlet's
                # entrance point/frame/tangent instead of letting a new fit
                # alter its entire exponential drop profile.
                scale = (sig[-1]/n_seg)/end_span
                fixed = []
                for cc, old in zip(cs, end_targets[m]):
                    corrected = []
                    for c, q in zip(cc, old):
                        value = np.polynomial.polynomial.polyval(1., q)-np.polynomial.polynomial.polyval(1., c)
                        derivative = scale*np.polynomial.polynomial.polyval(1., np.polynomial.polynomial.polyder(q))-np.polynomial.polynomial.polyval(1., np.polynomial.polynomial.polyder(c))
                        c = np.pad(c, (0, max(0, 20-len(c))))
                        c[:4] += value*np.array([.5,.75,0,-.25])
                        # Preserve the old entrance derivative only very near
                        # the end. A cubic correction spreads its old fit noise
                        # over a whole piece and unnecessarily deforms the hair.
                        bump = np.polynomial.polynomial.polymul([-1.,1.], np.polynomial.polynomial.polypow([.5,.5],18))
                        c[:len(bump)] += derivative*bump
                        corrected.append(np.round(c, 5))
                    fixed.append(corrected)
                cs = fixed
            prev[m] = cs
            if m:
                parts[m - 1].append(('poly', cs, max(1, int(math.ceil(sig[-1] / n_seg / 1.5)))))
            # Validate the final rounded, joined surface, not only the three
            # independent fits before seam correction.
            delta = [np.array([np.polynomial.polynomial.polyval(t, c) for c in cc]).T - V for cc, V in zip(cs, ser)]
            for w_check in np.linspace(0, 2*math.pi, 17):
                crescent = (0.65*math.sin(w_check)+0.35*abs(math.sin(w_check)))*(0.78+0.22*math.cos(2*math.pi*math.cos(w_check)))
                err = max(err, float(np.max(np.linalg.norm(delta[0]+math.cos(w_check)*delta[1]+crescent*delta[2], axis=1))))
        errs.append(err)
    return parts, errs, prev


def _taut(s, y, sigma=0.25):
    """A strand's depth y along its length s hung as a taut sheet, as long hair hangs off the back of the head and the
    shoulder blades (the user's reference: anime's long hair is a smooth mass that need not trace the body, like the
    bangs): the least concave function above y (its upper hull: straight from where it leaves the braid to the fullest
    point it passes over, then on down), its corners rounded over about sigma and lifted just enough to stay above y."""
    hull = []
    for i in range(len(s)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (s[b] - s[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (s[i] - s[a]) >= 0:    # b under the chord a-i: drop it
                hull.pop()
            else:
                break
        hull.append(i)
    h = np.interp(s, s[hull], y[hull])
    step = max(1e-9, float(s[1] - s[0]))
    r = max(1, int(3 * sigma / step))
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) * step / sigma) ** 2)
    hs = np.convolve(np.pad(h, r, mode='edge'), ker / ker.sum(), 'valid')
    return hs + max(0.0, float(np.max(y - hs)))


def _bk_wobble(s, n, j):
    """A smooth wobble between -1 and 1 along a strand (arc length s), its own for strand n and quantity j: three
    waves of incommensurate lengths, their phases a formula in n and j."""
    return sum(a * np.sin(2 * np.pi * s / lam + 2.39 * (5 * n + 3 * j + i) + 0.7)
               for i, (lam, a) in enumerate(((0.9, 0.55), (0.53, 0.3), (0.31, 0.15))))


def _sub_at(sb, s, u):
    """Where a sub-strand is in its strand at arc length s (u: the fraction of the strand's length): across it, in
    the strand's half-widths (its place d, fanning out over the last fifth of a free strand, and a wander of up to
    0.08 of the spacing); out from it, in its thicknesses (0.17 to 0.23, never into what the strand lies on); and its
    width and thickness as fractions of the strand's. Below the end of a free strand, a sub-strand ends a little
    early (at u = end; a strand drawn as one runs to its own point)."""
    j = sb['j']
    if sb['free']:
        f = np.clip((u - 0.8) / 0.2, 0, 1)
        e = np.clip((sb['end'] - u) / 0.04, 0, 1)
        fan, tip = f * f * (3 - 2 * f), 1 - (1 - e) ** 2       # (smoothly: a power under 1 has an endless slope at the end)
    else:
        fan, tip = 0.0, 1.0
    lat = sb['d'] * (1 + 0.25 * fan) + 0.08 * sb['sp'] * _bk_wobble(s, j, 5)
    dep = 0.2 + 0.03 * _bk_wobble(s, j, 6)                # hardly in and out: neat hair lies smooth
    return lat, dep, sb['w'] * (1 + 0.03 * _bk_wobble(s, j, 7)) * tip, (0.9 + 0.03 * _bk_wobble(s, j, 8)) * tip


def back_strands():
    """Each back strand as parts for Arch: ('poly', [C, U, V], pieces) for the hanging part, polynomials in t
    (-1..1) for the centre line, side and thickness vectors, in as many pieces as keep it within 0.006 (joined
    exactly); ('curl', xyz, pieces) for a ringlet, written exactly (curl_xyz); a list of these for each sub-strand
    (BK_SUBS), with their colours. The hanging part lies on the clothes, falls straight where they fall away, and lies on the
    strands beneath. Each strand wanders, varies in width and thickness (BK_NOISE) and twists about its own line below
    the nape (BK_TWIST), lifted where it twists so that its edges (and its sub-strands') stay clear of what it lies on.
    A strand that ends in a ringlet rises over its last 0.6 to where the ringlet starts (ringlet_axes)."""
    ux0, uz0, ust, unx, unz = _UG
    under = np.zeros((unx, unz))
    out, axes = {}, ringlet_axes()

    def at_under(px, pz):
        i, j = np.round((px - ux0) / ust).astype(int), np.round((pz - uz0) / ust).astype(int)
        ok = (i >= 0) & (i < unx) & (j >= 0) & (j < unz)
        return i, j, ok
    for key, (col, th) in BK_LOOK.items():
        sd, n_ = (1 if key[1] == 'L' else -1), list(BK_STRANDS).index(key)      # its noise and wave are its own
        D = np.array(BK_STRANDS[key], float) * [1, 1, BK_WIDE]
        side = key in BK_SIDE
        if key in BK_UNDER or side:                          # it starts right under the braid
            D = np.r_[[[D[0, 0], float(braid_back(D[0, 0])[2]) - 0.02, 0.8 * D[0, 2]]], D]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(D[:, :2], axis=0).T))]
        s = np.linspace(0, d[-1], 600)
        Q = _nat_spline(d, D, s)
        on = np.clip(s / 0.4, 0, 1)                         # the noise comes in below the braid
        nx, nw, nt, ny = BK_NOISE

        def across(Q):
            T2 = np.gradient(Q[:, :2], axis=0)
            T2 /= np.linalg.norm(T2, axis=1)[:, None]
            return np.c_[-T2[:, 1], T2[:, 0]]
        Q[:, :2] += (nx * on * _bk_wobble(s, n_, 0))[:, None] * across(Q)
        N2 = across(Q)
        hw = np.maximum(Q[:, 2], 0) * (1 + nw * on * _bk_wobble(s, n_, 1))
        thv = th * (1 + nt * on * _bk_wobble(s, n_, 2))
        phi = np.zeros(len(s))
        low = np.nonzero(Q[:, 1] < BK_NAPE - 0.1)[0]
        if key in BK_TWIST and len(low):
            u = np.clip((s - s[low[0]]) / (s[-1] - s[low[0]]), 0, 1)
            phi = 2 * np.pi * BK_TWIST[key] * u * u * (3 - 2 * u) + 0.25 * np.sin(np.pi * u) * _bk_wobble(s, n_, 4)
        lift = 0.6 * hw * (np.sqrt(np.sin(phi) ** 2 + 0.0025) - 0.05)   # a twisted strand rises from what it lies on
        fb, fh = np.full(len(s), -9.0), np.zeros(len(s))
        for k in np.linspace(-1, 1, 7):                      # on the head only the middle of its width rests on the hair,
            px, pz = Q[:, 0] + k * hw * N2[:, 0], Q[:, 1] + k * hw * N2[:, 1]     # its edges following it round
            fb = np.maximum(fb, body_back(px, pz, head=abs(k) < 0.4))
            if side:                                         # the side hair lies on the long locks there too
                fb = np.maximum(fb, locks_back(px, pz))
            i, j, ok = at_under(px, pz)
            fh = np.maximum(fh, np.where(ok, under[np.clip(i, 0, unx - 1), np.clip(j, 0, unz - 1)], 0))
        # its depth: on the clothes, and where they fall away (under the capelet's hem) it falls straight, as hair
        # does (how far back it is never decreases going down); and on the strands beneath where they are (easing
        # onto them over its first 0.4, under the braid, where they all begin)
        first = fb[fb > -9][0] + CH_GAP if (fb > -9).any() else BK_TOP      # from the first thing it lies on
        # the wrap (BK_WRAP): how far forward of where it would hang straight it comes, so that wherever this says it
        # falls straight below, it falls straight on the wrap's curve instead
        kw, zw0, zw1 = BK_WRAP
        wrapped = key not in BK_UNDER and key != 'BR0'
        rw = np.clip((zw0 - Q[:, 1]) / (zw0 - zw1), 0, 1)
        rw = rw * rw * (3 - 2 * rw)
        x_ref = Q[int(np.argmax(Q[:, 1] < zw0)), 0] if (Q[:, 1] < zw0).any() else Q[-1, 0]
        wsh = -kw * (Q[:, 0] ** 2 - x_ref ** 2) * rw if wrapped else np.zeros(len(s))
        body = np.r_[0, wsh] + np.maximum.accumulate(np.r_[first, np.where(fb > -9, fb + CH_GAP - wsh, -np.inf)])
        fall = np.nonzero(Q[:, 1] < BK_DRAPE_Z)[0]
        fall = fall[0] if len(fall) else len(s)
        if side:
            # the side hair comes forward from the head onto the shoulders and the backs of the arms, lying on what is
            # there (where nothing is, as before), and below BK_DRAPE_Z falls straight
            raw = np.where(fb > -9, fb + CH_GAP, np.nan) - wsh     # (measured from the wrap's curve)
            raw[0] = first if not np.isfinite(raw[0]) else raw[0]
            raw = raw[np.maximum.accumulate(np.where(np.isfinite(raw), np.arange(len(raw)), 0))]
            raw[fall:] = np.maximum.accumulate(raw[fall:])
            body = np.r_[first, raw + wsh]
        curl = BK_CURL.get(key)
        # its thickness closes to the point over the last 0.25, smoothly (a power under 1 there has an endless slope at
        # the point, which the pieces' polynomials cannot follow)
        tip = np.ones(len(s)) if curl else 1 - (1 - np.minimum(1, (d[-1] - s) / 0.25)) ** 2
        body = body[1:] + 0.3 * thv * tip + lift + ny * on * (1 + _bk_wobble(s, n_, 3)) / 2   # its line over the clothes
        ker = np.exp(-0.5 * (np.arange(-24, 25) / 8) ** 2)
        fh = np.where(fh > BK_TOP, BK_TOP + (fh - BK_TOP) * np.clip(s / 0.4, 0, 1) ** 2, fh)
        g = fh + CH_ON
        g = np.max([np.roll(np.pad(g, 20, mode='edge'), i)[20:-20] for i in range(-20, 21)], axis=0)  # rounded off
        g = np.convolve(np.pad(g, 24, mode='edge'), ker / ker.sum(), 'valid')                    # without sinking
        # on the head and the nape the strands lie side by side on the hair, as they come out of the braid (the
        # user's tracing): over each other they rise by no more than BK_STACK, which grows to its full height below
        f = np.clip((BK_NAPE - Q[:, 1]) / 0.6, 0, 1)
        lo = np.minimum(np.maximum(body, np.where(fh > 0, g, -9.0)), body + BK_STACK[0] + (BK_STACK[1] - BK_STACK[0]) * f * f * (3 - 2 * f))
        y = lo.copy()
        for _ in range(12):                                  # smoothed a little, so as to stay close
            y = np.maximum(np.convolve(np.pad(y, 24, mode='edge'), ker / ker.sum(), 'valid'), lo)
        if side:                                             # the side hair eases over the edges it drapes across
            kd = np.exp(-0.5 * (np.arange(-60, 61) / 20) ** 2)
            for _ in range(30):
                y = np.maximum(np.convolve(np.pad(y, 60, mode='edge'), kd / kd.sum(), 'valid'), lo)
        if side:                                             # past the strands beneath it, it falls straight too
            y[fall:] = wsh[fall:] + np.maximum.accumulate(y[fall:] - wsh[fall:])
        else:
            y = wsh + np.maximum.accumulate(y - wsh)
        if key not in BK_UNDER and not side:
            y = wsh + _taut(s, y - wsh)                      # and hangs as a smooth sheet, not tracing the body
        amp, lam, zw = BK_WAVE
        ramp = np.clip((zw - Q[:, 1]) / 0.5, 0, 1)
        y = y + amp * ramp * np.sin(2 * math.pi * s / lam + 2.4 * n_)      # each strand its own phase
        if wrapped:
            # the wave in depth (BK_DWAVE), kept out of the strands beneath (as far as they may pile: BK_STACK) and the
            # clothes, and smoothed where it rests on them
            a_, lam_, zd0, zd1 = BK_DWAVE
            grp = key[:3] if side else key[:2]
            members = [k for k in BK_FLOW if (k in BK_SIDE) == side and k[1] == key[1]]
            ph_ = BK_DPHASE[grp] + 0.7 * (members.index(key) if key in members else 0)
            rd = np.clip((zd0 - Q[:, 1]) / (zd0 - zd1), 0, 1)
            y = y + a_ * rd * rd * (3 - 2 * rd) * np.sin(2 * math.pi * s / lam_ + ph_)
            cap = BK_STACK[0] + (BK_STACK[1] - BK_STACK[0]) * f * f * (3 - 2 * f)
            floor = np.maximum(np.where(fh > 0, np.minimum(g, body + cap), -9.0),
                               np.where(fb > -9, fb + CH_GAP + 0.3 * thv * tip, -9.0))
            y = np.maximum(y, floor)
            for _ in range(6):
                y = np.maximum(np.convolve(np.pad(y, 24, mode='edge'), ker / ker.sum(), 'valid'), floor)
        if curl:
            # where the ringlet starts: beside its axis if the strand ends at least r0 from it across, else in front
            ox, oy = axes[key]
            dx = abs(Q[-1, 0] - ox)
            ye = oy - math.sqrt(max(0.0, curl['r0'] ** 2 - dx * dx)) if dx < 0.9 * curl['r0'] else oy
            f = np.clip((s - (s[-1] - 0.6)) / 0.6, 0, 1)
            y = y + (max(ye, y[-1]) - y[-1]) * f * f * (3 - 2 * f)
        for k in np.linspace(-1.2, 1.2, 11):                 # across the twisted section and its sub-strands
            kc = k * hw * np.cos(phi)
            px, pz = Q[:, 0] + kc * N2[:, 0], Q[:, 1] + kc * N2[:, 1]
            i, j, ok = at_under(px, pz)
            np.maximum.at(under, (i[ok], j[ok]), (y + 0.1 * thv * tip)[ok])    # the strands over it lie just above
                                                    # its line, sinking into it rather than piling up
        # it faces out from the wrap's curve (psi: how far its section turns about the upright, from facing straight
        # back), and its curl turns out of the sheet (BK_CURL_TILT: about the upright through a point 0.3 above the
        # curl, back and away from her; the strands over it lie on the sheet, as written to `under` above)
        psi = -np.arctan(2 * kw * Q[:, 0] * rw) if wrapped else np.zeros(len(s))
        if wrapped and key in BK_CURL_AT:
            ic, sense = BK_CURL_AT[key]
            ic += len(D) - len(BK_STRANDS[key])              # (the point added under the braid)
            sp = max(0.0, d[ic] - 0.3)
            ip = int(np.searchsorted(s, sp))
            bt = np.clip((s - sp) / 0.5, 0, 1)
            tl = sense * BK_CURL_TILT * bt * bt * (3 - 2 * bt)
            dx_, dy_ = Q[:, 0] - Q[ip, 0], y - y[ip]
            Q[:, 0] = np.where(s >= sp, Q[ip, 0] + dx_ * np.cos(tl) - dy_ * np.sin(tl), Q[:, 0])
            y = np.where(s >= sp, y[ip] + dx_ * np.sin(tl) + dy_ * np.cos(tl), y)
            psi = psi + tl
        # the hanging part: a natural spline from the root inside the braid through the traced part every 0.05
        idx = np.unique(np.r_[np.arange(0, len(s), max(1, int(0.05 / (s[1] - s[0])))), len(s) - 1])
        if key in BK_UNDER or side:                          # its root inside the braid, where the braid crosses it
            zb, yb, _ = braid_back(0.95 * D[0, 0])
            root = [[0.95 * D[0, 0], float(yb) - 0.06, float(zb)]]
        else:
            root = [[0.5 * D[0, 0], BK_ROOT[0], BK_ROOT[1]]] if key != 'BR0' else [[sd * 0.36, 0.7, 11.7]]
        K = np.c_[np.r_[root, np.c_[Q[idx, 0], y[idx], Q[idx, 1]]], np.r_[hw[0], hw[idx]], np.r_[th, (thv * tip)[idx]],
                  np.r_[0, phi[idx]], np.r_[0, psi[idx]]]
        dk = np.r_[0, np.cumsum(np.linalg.norm(np.diff(K[:, :3], axis=0), axis=1))]
        Kd = _nat_spline(dk, K, np.linspace(0, dk[-1], 4000))
        old_values = Kd.copy()
        old_section = Kd[:, 3:5].copy()
        old_sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Kd[:, :3], axis=0), axis=1))]
        old_h = int(0.1 / (old_sig[-1] / len(old_sig)))
        old_ker = np.exp(-0.5 * (np.arange(-3*old_h, 3*old_h+1) / old_h)**2)
        old_ts = np.gradient(np.array([np.convolve(np.pad(Kd[:, i], 3*old_h, mode='edge'), old_ker/old_ker.sum(), 'valid') for i in range(3)]).T, axis=0)
        old_ts_raw = old_ts.copy()
        old_ts /= np.linalg.norm(old_ts, axis=1)[:, None]
        # Replace the hidden fixed root and its outward step, not the traced tail.
        # The new root lies on the actual crown surface under the braid, with
        # that surface's own downward tangent; its section grows from zero.
        from hairjoin import attach_root
        root_z = float(braid_back(D[0, 0])[2]) + 0.06
        anchor, root_tangent, root_normal, root_ref, root_uv = head_root(D[0, 0], root_z)
        Kd, old_end, tangent = attach_root(Kd, anchor, root_tangent)
        P, W, TH, PH, PSI = Kd[:, :3], np.maximum(Kd[:, 3], 0), np.maximum(Kd[:, 4], 0), Kd[:, 5], Kd[:, 6]
        sig = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        bridge_end = len(Kd) - (4000-old_end)
        root_length = sig[bridge_end]
        # Preserve the old material coordinate: changing a root must not shift
        # the wobble, free tips or curl phase all the way down an unchanged lock.
        material_sig = np.r_[np.linspace(0, old_sig[old_end], bridge_end+1), old_sig[old_end+1:]]
        # Restore width quickly for coverage; the crest, not a piled-up full
        # thickness, is what is kept tangent to the upper hair below.
        entry = np.clip(sig/0.06, 0, 1)
        entry = 6*entry**5 - 15*entry**4 + 10*entry**3
        W = np.maximum(0, np.interp(material_sig, old_sig, old_section[:, 0]))*entry
        TH = np.maximum(0, np.interp(material_sig, old_sig, old_section[:, 1]))*entry
        BK_JOIN_DIAGNOSTICS[key] = dict(anchor=anchor, tangent=tangent, normal=root_normal, source=root_ref,
                                        uv=root_uv, root_length=root_length, end=P[bridge_end].copy())
        h = int(0.1 / (sig[-1] / len(sig)))                 # the line smoothed over about 0.1, for the section's turn
        ker = np.exp(-0.5 * (np.arange(-3 * h, 3 * h + 1) / h) ** 2)
        TS = np.gradient(np.c_[[np.convolve(np.pad(P[:, i], 3 * h, mode='edge'), ker / ker.sum(), 'valid') for i in range(3)]].T, axis=0)
        TS /= np.linalg.norm(TS, axis=1)[:, None]
        restore = np.clip((sig-root_length)/0.15, 0, 1)[:, None]
        restore = restore*restore*(3-2*restore)
        tail_ts = np.array([np.interp(material_sig, old_sig, old_ts[:, i]) for i in range(3)]).T
        TS = (1-restore)*TS + restore*tail_ts
        TS /= np.linalg.norm(TS, axis=1)[:, None]
        t = np.linspace(-1, 1, 1201)
        fade = np.array([0.5, -0.75, 0, 0.25])     # 1 -> 0, zero derivative at both endpoints

        def joined(cs, end, start):
            """Polynomials moved so that they start exactly where the previous part ends, the move fading out."""
            return [np.round(np.polynomial.polynomial.polyadd(c, (e - s0) * fade), 5) for c, e, s0 in zip(cs, end, start)]

        # the sub-strands, side by side across the strand, each with its own wander, depth, width and thickness
        n = BK_SUBS[key]
        subs = [dict(j=10 * n_ + k, d=0.8 * (2 * k - (n - 1)) / n, sp=1.6 / n if n > 1 else 0.0, w=min(1.0, 1.2 / n + 0.13),
                     free=not curl, end=1 - 0.06 * ((0.618 * (k + n_)) % 1) if n > 1 else 1.0) for k in range(n)]
        cols = [col] + [c for c in (HAIR, HAIR_LT, HAIR_DK) if c != col]
        # Prepare contact once; trying different polynomial partitions must not
        # repeat the surface inversion or shift the material texture.
        prepared = _back_sections(sig, P, W, TH, PH, TS, subs, root_normal, root_length, material_sig, PSI)
        end_targets, end_span = None, 1.
        if curl:
            # Reconstruct the original fitted ringlet entrance from the unchanged
            # old samples. Preserving only their XYZ is insufficient: a different
            # high-order endpoint tangent changes the helix's drop everywhere.
            from hairjoin import legacy_end_jets
            def original_at(st):
                vals = np.array([np.interp(st, old_sig, old_values[:, i]) for i in range(6)]).T
                Ot = np.array([np.interp(st, old_sig, old_ts_raw[:, i]) for i in range(3)]).T
                Ot /= np.linalg.norm(Ot, axis=1)[:, None]
                On = np.array([0., 1., 0.])-Ot[:, 1:2]*Ot
                On /= np.linalg.norm(On, axis=1)[:, None]
                Os = np.cross(Ot, On); Os /= np.linalg.norm(Os, axis=1)[:, None]
                ph = vals[:, 5:6]
                On, Os = np.cos(ph)*On+np.sin(ph)*Os, np.cos(ph)*Os-np.sin(ph)*On
                ow, oh = np.maximum(vals[:, 3], 0), np.maximum(vals[:, 4], 0)
                result = [(vals[:, :3], Os*ow[:, None], On*oh[:, None])]
                for sb in subs:
                    la, de, wk, tk = _sub_at(sb, st, st/old_sig[-1])
                    result.append((vals[:, :3]+(la*ow)[:, None]*Os+(de*oh)[:, None]*On, Os*(wk*ow)[:, None], On*(tk*oh)[:, None]))
                return result
            end_targets, old_segments = legacy_end_jets(old_sig[-1], original_at, CH_DEG)
            end_span = old_sig[-1]/old_segments
        # the hanging part, in as few polynomial pieces as keep the fit within 0.006, joined exactly
        for n_seg in range(1 if sig[-1] < 2.5 else 2, 17):
            parts, errs, prev = _hang_pieces(n_seg, t, sig, prepared, joined, subs, end_targets, end_span, root_tangent)
            if max(errs) < 0.006:
                break
        if max(errs) >= 0.006:
            raise ValueError(f'{key}: joined surface fit error {max(errs):.6f} exceeds 0.006')
        BK_JOIN_DIAGNOSTICS[key]['fit_error'] = max(errs)
        if curl:
            # each sub-strand's ringlet starts exactly where its hanging part's polynomials end
            ev = lambda cs, d=0: np.array([np.polynomial.polynomial.polyval(1.0, np.polynomial.polynomial.polyder(c, d)) for c in cs])
            Pe, Te, Se, Ne = ev(end_targets[0][0]), ev(end_targets[0][0], 1), ev(end_targets[0][1]), ev(end_targets[0][2])
            at = axes[key]
            ends = []
            for sb, pc in zip(subs, end_targets[1:]):
                lat, dep, wk, tk = _sub_at(sb, material_sig[-1], 1.0)
                ends.append(dict(r=ev(pc[0]) - (Pe + lat * Se + dep * Ne), d=lat, dep=dep, w=wk, t=tk, wob=1.7 + 0.37 * n_ + 0.29 * sb['j'] % 1.3))
            for part, xyz in zip(parts, curl_xyz(Pe, Te / np.linalg.norm(Te), Se, Ne, dict(curl, at=at), ends)):
                part.append(('curl', xyz, int(math.ceil(curl['turns']))))
        out[key] = (parts, max(errs), [cols[k % len(cols)] for k in range(n)], errs)
    return out


def run_hair():
    sc.blank()
    sc.note('Hair')
    sc.comment(f"""Her hair flows from the crown. A lock is described by t, from the crown (0) down to the end of
the longest hair (1): up to t = {fmt(T0)} it lies on a dome around the head (ha is the angle from the
crown), below that it falls. hzz(t) is the height, hrx(t) and hry(t) are the half-axes of the ellipse the
hair lies on at that height: they follow the dome, then keep just clear of the shoulders, capelet, arms, book
and skirt, so that the locks at her sides lie close to her clothes (fitted by the generator). hu(t) is 0 on the dome and grows to 1
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
lock. lrc is how far the line stands off the hair surface: its layer la, and a fifth of the wave, so the wave's
plane tilts a little off the hair (a quarter turn later, as before, the lock wound like a corkscrew). Across the lock (w runs around it) lwd is
the half-width and lth the thickness, a crescent lcs with three ridges that is flat underneath. ltp tapers
the lock toward its end te, and lq runs 0..1 over the last stretch, where the tip turns through three
quarters of a circle: sideways like a hook, or outward like a scroll.""")
    sc.func('ltp(t,te)', 'min(1,max(0,te-t)/0.13)^0.6')
    sc.func('lq(t,te)', 'max(0,1-max(0,te-t)/0.08)')
    sc.func('lcs(w)', '(0.65*sin(w)+0.35*abs(sin(w)))*(0.78+0.22*cos(2*pi*cos(w)))')
    sc.comment("""The long hair down the back and over the shoulders has the bangs' ridged crescent, and tapers (ltl)
into its curled tip, as the chest locks do.""")
    sc.func('ltl(t,te)', 'min(1,max(0,te-t)/0.1)^0.7')
    sc.func('lpc(t,pb,p0,sg,dr,am,kk,ph)', 'pb+p0*(1-sg*(1-hdw(t)))+dr*hu(t)+am*hu(t)^1.5*sin(kk*hu(t)+ph)')
    sc.func('lrc(t,la,am,kk,ph)', 'la*min(1,t/0.04)+0.2*am*hu(t)^1.5*sin(kk*hu(t)+ph)')
    sc.func('lwd(t,te,hw)', 'hw*ltp(t,te)*hdw(t)')
    sc.func('lth(t,w,te,th)', 'th*ltp(t,te)*lcs(w)*min(1,t/0.04)')
    sc.func('lwl(t,te,hw)', 'hw*ltl(t,te)*hdw(t)')
    sc.func('lts(t,w,te,th)', 'th*ltl(t,te)*lcs(w)*min(1,t/0.04)')

    sc.func('hzg(a)', '(2/pi)*asin(0.97*sin(a))')       # a zigzag, for the highlights' serrated edges
    HL_SPIKES = 7.5         # the zigzag's rate across a clump (w from 0.3 to pi - 0.3): about three spikes a clump

    def lock_parts(tv, pb, p0, sg, dr, hw, te, la, th, am, kk, ph, full, smooth=False):
        """A lock's angle, standoff, half-width and thickness, as Arch expressions in the expression tv for t (smooth:
        the long hair, tapering over a longer stretch to a sharp point)."""
        PC = f"lpc({tv},{fmt(pb)},{fmt(p0 - pb)},{fmt(sg)},{fmt(dr)},{fmt(am)},{fmt(kk)},{fmt(ph)})"
        GR = f"(0.55+0.45*min(1,4*hu({tv})))" if full else None
        LA = f"({fmt(la)}*{GR})" if full and la else fmt(la)
        THK = f"({fmt(th)}*{GR})" if full else fmt(th)
        WD = f"{'lwl' if smooth else 'lwd'}({tv},{fmt(te)},{fmt(hw)})"
        if am >= 0.04:          # a wavy lock (HAIR_WAVE) narrows where the wave turns it, as a ribbon does
            WD += f"*(1-{fmt(HAIR_WAVE['narrow'])}*min(1,2*hu({tv})^1.5)*cos({fmt(kk)}*hu({tv})+{fmt(ph)})^2)"
        return PC, f"lrc({tv},{LA},{fmt(am)},{fmt(kk)},{fmt(ph)})", WD, f"{'lts' if smooth else 'lth'}({tv},w,{fmt(te)},{THK})"

    def lock_band(ref, args, full, tc, h, ph0=0.0, smooth=False):
        """A highlight across a lock, as anime hair is lit: a band across the clump at tc whose upper and lower edges are
        zigzags of spikes pointing along the hair, the band's half-length h (in t) at the spikes' tips; drawn twice, a
        light peach and, half as long, a bright core (two crisp steps, as the guides draw it)."""
        zc = ZHC + HDZ * math.cos(1.5708 * min(1, tc / T0)) - VF * max(0.0, tc - T0)
        for k, (hk, col, lift) in enumerate(((h, HAIR_HI, 1.06), (0.45 * h, HAIR_GLOSS, 1.1))):
            T = f"({fmt(tc)}+{fmt(hk)}*q*(0.3+0.7*abs(hzg({fmt(HL_SPIKES)}*w+{fmt(ph0)}))))"
            PC, RC, WD, TH = lock_parts(T, *args, full, smooth)
            R = f"({RC}+{fmt(lift)}*{TH})"
            with sc.turned(**HEAD_TURN, blend=(f'hs({T})', f'hk({T})', f'hm({T})')):
                sc.surface((f"(hrx({T})+{R}*sin(ha({T})))*sin({PC}+{WD}*cos(w))", f"{fmt(HY0)}+(hry({T})+{R}*sin(ha({T})))*cos({PC}+{WD}*cos(w))",
                            f"hzz({T})+{R}*cos(ha({T}))"), ('w', 0.3, math.pi - 0.3), ('q', -1, 1), col, f'{ref}Hl{k + 1}',
                           ('point', (0, HY0, zc - 0.3)))
            # written out again as a polynomial in w and the band's own t (compact.py): pasting the whole lock in at T
            # made each of these the costliest object to open (about 180 KB of formula)
            COMPACT[f'{ref}Hl{k + 1}'] = dict(sub=(T, tc, hk, 'w'), tol=0.0003)

    RING_POLAR = 1.5708 * 0.085 / T0       # the angel ring's angle from the crown (locks' crown_shine is at t 0.085)
    BAND_Z = 10.55          # the second ring of highlights, across the long hair where it curves over the shoulders
    BAND_T = T0 + (ZHC - BAND_Z) / VF

    def lock(ref, p0, sg, dr, hw, te, la, th, am, kk, ph, col, curl=('side', 1, 0.12), start=0.012, pb=0.0, full=None, band=False,
             smooth=False, outline=None, paint=False):
        """One lock: main part, then its curled tip as a second surface (so the curl gets a fine mesh).
        full: the lock's own volume, its standoff and thickness grown by this factor where it falls (from 0.55 of it on
        the head, so the crown keeps its size): a round clump about three times as wide as it is thick, which Arch's light
        shades round, instead of a flat ribbon. band: a highlight across it where it curves over the shoulders. smooth:
        the long hair's long taper (ltl). outline: its line art (HAIR_OUTLINE) this many pixels wide,
        along both edges from where it leaves the dome (the crown stays clean). paint: strand strokes, a shadow wedge
        and a light streak painted near its end (paint_hair)."""
        args = (pb, p0, sg, dr, hw, te, la, th, am, kk, ph)
        PC, RC, WD, TH = lock_parts('t', *args, full, smooth)
        end = te if curl is None else te - (0.1 if curl[0] == 'flick' else curl[2] / VF if curl[0] == 'hook' else 0.08)

        def edges(P, R, t0, t1, tag):
            if not outline or t1 <= t0:
                return
            RR = f"({R}+{fmt(OUTLINE_LIFT)})"
            for sgn_, nm in ((1, 'A'), (-1, 'B')):        # the edges at w = 0 and pi, where the section's thickness is 0
                ang = f"({P}{'+' if sgn_ > 0 else '-'}{WD})"
                sc.curve((f"(hrx(t)+{RR}*sin(ha(t)))*sin({ang})", f"{fmt(HY0)}+(hry(t)+{RR}*sin(ha(t)))*cos({ang})",
                          f"hzz(t)+{RR}*cos(ha(t))"), ('t', t0, t1), HAIR_OUTLINE, f'{ref}{tag}Line{nm}', thickness=outline)
        with sc.turned(**HEAD_TURN, blend=HB):
            sc.surface((f"(hrx(t)+({RC}+{TH})*sin(ha(t)))*sin({PC}+{WD}*cos(w))", f"{fmt(HY0)}+(hry(t)+({RC}+{TH})*sin(ha(t)))*cos({PC}+{WD}*cos(w))",
                        f"hzz(t)+({RC}+{TH})*cos(ha(t))"), ('t', start, end), ('w', 0, '2*pi'), col, ref, ('tube', 'u'))
            edges(PC, RC, max(start, T0), end, '')
        if paint:
            PAINT.append(dict(kind='lock', ref=ref, p0=p0, seed=3.7 * p0 + 11 * te))
        if band and te > BAND_T + 0.03:
            lock_band(ref + 'Band', args, full, BAND_T + 0.011 * math.sin(3.1 * p0 + 0.7), 0.022, ph0=2.3 * p0, smooth=smooth)    # staggered a little
        if curl is None:
            return
        if curl[0] == 'hook':
            # the tip ends in a small hook, as anime draws long hair's ends (HAIR_CURL): over its last lc of length the
            # line bends ever tighter round the body (in the sheet the lock lies in), the way its wave was already
            # bending there (else in toward the back); the lock's width turns with it so it stays a ribbon, and the
            # taper makes the hook fine. Written in sg, from 0 to 1 along the hook: its heading th = turn sg^(p+1) and its
            # offsets across and along, polynomials fitted to the hook (_hook).
            _, turn, lc = curl
            dT = lc / VF
            t0 = te - dT
            hu0 = (t0 - T0) / (1 - T0)
            bend = -math.sin(kk * hu0 + ph) if am >= 0.04 else 0.0
            sgn = (1.0 if bend > 0 else -1.0) if abs(bend) > 0.35 else -math.copysign(1, p0)
            hs_, th_, lat_, fwd_ = _hook(turn, lc)
            cl = np.round(np.polynomial.polynomial.polyfit(hs_ / lc, lat_ / lc, 7), 6)
            cf = np.round(np.polynomial.polynomial.polyfit(hs_ / lc, fwd_ / lc, 7), 6)
            SG = f"min(1,max(0,(t-{fmt(t0)})/{fmt(dT)}))"
            TT = f"({fmt(turn)}*{SG}^{fmt(HAIR_CURL['p'] + 1)})"
            rr = f"(0.5*(hrx(t)+hry(t))+{RC})"
            PH_ = f"{PC}+{fmt(sgn * lc)}*({horner(list(cl), SG)})/{rr}"
            DZ = f"{fmt(lc)}*({SG}-({horner(list(cf), SG)}))"

            def hook_xyz(cw, R):
                ang = f"({PH_}+{WD}*{cw}*cos{TT})"
                return (f"(hrx(t)+{R}*sin(ha(t)))*sin({ang})", f"{fmt(HY0)}+(hry(t)+{R}*sin(ha(t)))*cos({ang})",
                        f"hzz(t)+{R}*cos(ha(t))+{DZ}+{fmt(sgn)}*{WD}*{rr}*{cw}*sin{TT}")
            with sc.turned(**HEAD_TURN, blend=HB):
                sc.surface(hook_xyz('cos(w)', f"({RC}+{TH})"), ('t', t0, te), ('w', 0, '2*pi'), col, ref + 'Tip', ('tube', 'u'))
                if outline:
                    for cw, nm in (('1', 'A'), ('(-1)', 'B')):
                        sc.curve(hook_xyz(cw, f"({RC}+{fmt(OUTLINE_LIFT)})"), ('t', t0, te), HAIR_OUTLINE, f'{ref}TipLine{nm}', thickness=outline)
            return
        if curl[0] == 'flick':
            # the tip: the lock tapers to a point and flicks gently aside and out over its last stretch, as anime hair
            # ends (in its own surface so that the point gets a fine mesh)
            _, sgn, dpsi, dout = curl
            if sgn == 0:        # in toward the back
                sgn = -math.copysign(1, p0) if p0 else 0
            LF = f"max(0,1-({fmt(te)}-t)/0.1)^2"
            PF = f"{PC}+{fmt(sgn * dpsi)}*{LF}"
            RF = f"{RC}+{fmt(dout)}*{LF}"
            with sc.turned(**HEAD_TURN, blend=HB):
                sc.surface((f"(hrx(t)+({RF}+{TH})*sin(ha(t)))*sin({PF}+{WD}*cos(w))", f"{fmt(HY0)}+(hry(t)+({RF}+{TH})*sin(ha(t)))*cos({PF}+{WD}*cos(w))",
                            f"hzz(t)+({RF}+{TH})*cos(ha(t))"), ('t', te - 0.1, te), ('w', 0, '2*pi'), col, ref + 'Tip', ('tube', 'u'))
                edges(PF, RF, te - 0.1, te, 'Tip')
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

    def poly_at(cs, t, d=0):
        return np.array([np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(np.asarray(c, float), d) if d else c) for c in cs])

    def poly_cross(C, zc, lo=-1.0, hi=1.0):
        """The first t in [lo, hi] where the strand's centre line comes down through height zc, or None."""
        ts = np.linspace(lo, hi, 801)
        z = np.polynomial.polynomial.polyval(ts, np.asarray(C[2], float))
        k = np.nonzero((z[:-1] - zc) * (z[1:] - zc) <= 0)[0]
        if not len(k):
            return None
        a, b = ts[k[0]], ts[k[0] + 1]
        za, zb = z[k[0]], z[k[0] + 1]
        return float(a + (b - a) * (za - zc) / (zb - za)) if zb != za else float(a)

    def poly_band(ref, C, U, V, tc, h_len, ph0=0.0, outward=None, sec='lcs'):
        """A highlight across a strand drawn as polynomials C + cos(w) U + lcs(w) V (as lock_band, for the traced
        strands): a band across it at tc with zigzag edges, h_len long at the spikes' tips. It is short, so it is laid
        on the strand's tangent plane there: small expressions, the error far below the strand's own thickness."""
        P0, dP, U0, V0 = poly_at(C, tc), poly_at(C, tc, 1), poly_at(U, tc), poly_at(V, tc)
        ht = h_len / max(1e-9, float(np.linalg.norm(dP)))
        for k, (hk, col, lift) in enumerate(((1.0, HAIR_HI, 1.07), (0.45, HAIR_GLOSS, 1.11))):
            Sw = f"q*(0.3+0.7*abs(hzg({fmt(HL_SPIKES)}*w+{fmt(ph0)})))"
            xyz = tuple(f"{fmt(P0[i])}+{fmt(dP[i] * ht * hk)}*{Sw}+{fmt(0.96 * U0[i])}*cos(w)+{fmt(lift * V0[i])}*{sec}(w)" for i in range(3))
            sc.surface(xyz, ('w', 0.3, math.pi - 0.3), ('q', -1, 1), col, f'{ref}Hl{k + 1}', ('dir', tuple(outward if outward is not None else V0)))

    def paint_plan(seed):
        """A clump's painting, as blade() paints the bangs, its own a little: strand lines as (place across, as cos w,
        from -1 at one edge to 1 at the other; from, to: fractions of its length from the tip), and its sheen (place
        across, from, to)."""
        h = [(math.sin(12.9898 * seed + 78.233 * k) * 43758.5453) % 1 for k in range(5)]
        lines = ((-0.42 + 0.08 * h[0], 0.03, 0.42 + 0.12 * h[1]), (0.38 - 0.08 * h[2], 0.06, 0.52 + 0.1 * h[3]),
                 (0.04 + 0.12 * (h[4] - 0.5), 0.16 + 0.08 * h[0], 0.34 + 0.08 * h[2]))
        return lines, (0.12 * (h[1] - 0.5), 0.5 + 0.06 * h[4], 0.8 + 0.06 * h[3])

    PAINT = []          # the clumps to paint (paint_hair), as they are drawn

    def paint_hair():
        """Paint the clumps in PAINT as blade() paints the bangs: two or three thin dark strand lines (HAIR_LINE) down the
        lower part of the clump, each widest in its middle and closing to the centre at both ends; a light peach sheen
        lens (HAIR_HI) in its middle above them; and on the curtain the translucent shade under the braid, full width at
        the top and closing to a point (HAIR_SHADE, in two steps). Cheaply: each shape is sampled on its clump as drawn (a
        lock's surfaces read back from the scene, a back strand's polynomials evaluated), lifted a little off it, and
        written as short polynomials (the surface between its two edges, bowed through its middle). Written as the
        clump's own formulas with the place across put in, they made the scene too slow to build."""
        import re, sheetfit
        refs = [p['ref'] for p in PAINT if p['kind'] == 'lock']
        S = {}
        if refs:
            pat = '|'.join(re.escape(r) + '(Tip)?' for r in refs)
            S = {nm: (f, lim) for nm, _, f, lim in sheetfit.surfaces_from_lines(sc.out, ref=pat)}

        def poly_eval(polys):
            def E(tau, w):              # tau from 0 to len(polys), each piece over a unit of it
                i = np.clip(np.floor(tau).astype(int), 0, len(polys) - 1)
                t = 2 * (tau - i) - 1
                out = np.zeros(tau.shape + (3,))
                for k, (C, U, V) in enumerate(polys):
                    m = i == k
                    if m.any():
                        ev = lambda cs: np.stack([np.polynomial.polynomial.polyval(t[m], c) for c in cs], -1)
                        sw = (0.65 * np.sin(w[m]) + 0.35 * np.abs(np.sin(w[m]))) * (0.78 + 0.22 * np.cos(2 * np.pi * np.cos(w[m])))
                        cw, sw = np.cos(w[m])[:, None], sw[:, None]      # lcs, the bangs' section
                        out[m] = ev(C) + cw * ev(U) + sw * ev(V)
                return out
            return E, 0.0, float(len(polys))

        def lock_eval(ref):
            fm, lm = S[ref]
            ft, lt = S.get(ref + 'Tip', (None, None))
            t1 = lt[1] if ft else lm[1]

            def E(tau, w):
                out = np.zeros(tau.shape + (3,))
                for f, lim in ((fm, lm), (ft, lt)):
                    if f is None:
                        continue
                    m = (tau >= lim[0] - 1e-9) & (tau <= lim[1] + 1e-9)
                    if m.any():
                        out[m] = np.stack([np.broadcast_to(np.asarray(c, float), tau[m].shape) for c in f(tau[m], w[m])], -1)
                return out
            return E, lm[0], t1

        def fit_poly(P):
            """Short polynomials in t (-1..1) through the points P (n, 3), within 0.003 if it can."""
            tt = np.linspace(-1, 1, len(P))
            for deg in (8, 11, 14):
                cs = [np.polynomial.polynomial.polyfit(tt, P[:, i], deg) for i in range(3)]
                err = max(float(np.max(np.abs(np.polynomial.polynomial.polyval(tt, c) - P[:, i]))) for i, c in enumerate(cs))
                if err < 0.003:
                    break
            return [horner(list(np.round(c, 6)), 't') for c in cs]

        for req in PAINT:
            E, ta, tb = lock_eval(req['ref']) if req['kind'] == 'lock' else poly_eval(req['polys'])
            name = req['ref']
            # distance from the tip along the crest (w = pi/2), and the clump's length
            tg = np.linspace(ta, tb, 600)
            crest = E(tg, np.full_like(tg, math.pi / 2))
            dtip = np.r_[0, np.cumsum(np.linalg.norm(np.diff(crest[::-1], axis=0), axis=1))][::-1]
            L = float(dtip[0])
            t_at = lambda d: np.interp(d, dtip[::-1], tg[::-1])
            lines, (sc0, s0, s1) = paint_plan(req['seed'])

            def lifted(tt, ww, lift):
                P = E(tt, ww)
                mid = 0.5 * (E(tt, np.zeros_like(tt)) + E(tt, np.full_like(tt, math.pi)))
                up = E(tt, np.full_like(tt, math.pi / 2)) - mid
                up /= np.maximum(np.linalg.norm(up, axis=1), 1e-9)[:, None]
                return P + lift * up, mid

            def strip(d0, d1, ca, cb, color, ref, opacity=None, lift=0.006, n=48):
                """The shape between the places ca(d) and cb(d) across the clump (as cos w) from d0 to d1 from the tip."""
                dd = np.linspace(d1, d0, n)
                tt = t_at(dd)
                wa, wb = np.arccos(np.clip(ca(dd), -1, 1)), np.arccos(np.clip(cb(dd), -1, 1))
                A, mid = lifted(tt, wa, lift)
                B, _ = lifted(tt, wb, lift)
                M, _ = lifted(tt, 0.5 * (wa + wb), lift)
                L_, R_, Mb = fit_poly(A), fit_poly(B), fit_poly(M - 0.5 * (A + B))
                xyz = tuple(f"({l})*(1-q)+({r})*q+4*q*(1-q)*({m})" for l, r, m in zip(L_, R_, Mb))
                c = mid[len(mid) // 2]
                sc.surface(xyz, ('t', -1, 1), ('q', 0, 1), color, ref, ('point', tuple(float(v) for v in c)), opacity=opacity)
            if req['kind'] == 'poly':       # the shade under the braid, as under the crown's overhang on the bangs
                for k, (ln, op) in enumerate(((0.95, 0.32), (0.65, 0.32))):     # just below the braid (by length, not a
                    d0 = max(0.0, L - ln)                                       # share of it: the long hair's ran down the back)
                    tp = lambda d, d0=d0: np.minimum(1, np.maximum(0, d - d0) / (0.45 * (L - d0))) ** 0.7
                    strip(d0, L - 0.01, lambda d, tp=tp: 0.96 * tp(d), lambda d, tp=tp: -0.96 * tp(d), HAIR_SHADE,
                          f'{name}Shade{k + 1}', op, lift=0.004 + 0.001 * k)
            # the sheen: a lens 0.22 of the half-width either side of its place, pointed at both ends
            tp = lambda d: np.maximum(0, np.sin(np.pi * (d / L - s0) / (s1 - s0))) ** 1.2
            strip(s0 * L, s1 * L, lambda d: sc0 * tp(d) + 0.22 * tp(d), lambda d: sc0 * tp(d) - 0.22 * tp(d), HAIR_HI,
                  f'{name}Sheen', lift=0.007)
            # the strand lines, closing to the centre at both ends (as the bangs' max(0, sin(...))^0.7)
            for k, (a_, f0, f1) in enumerate(lines):
                tl = lambda d, f0=f0, f1=f1: np.maximum(0, np.sin(np.pi * (d / L - f0) / (f1 - f0))) ** 0.7
                strip(f0 * L, f1 * L, lambda d, a_=a_, tl=tl: (a_ + 0.045) * tl(d), lambda d, a_=a_, tl=tl: (a_ - 0.045) * tl(d),
                      HAIR_LINE, f'{name}Line{k + 1}', lift=0.009)

    def crown_shine(ref, p0, sg, dr, hw, te, la, th, am, kk, ph, pb=0.0, smooth=False):
        """The ring of light across the crown (anime's 'angel ring'): a zigzag highlight band across the lock where it
        crosses the upper dome, at the same height as on every other clump so that they read as one ring."""
        lock_band(ref + 'Ring', (pb, p0, sg, dr, hw, te, la, th, am, kk, ph), None, 0.085, 0.011, ph0=1.7 * p0, smooth=smooth)

    sc.comment("""A dark cap and an inner sheet under the locks, so no gap shows skin. The cap stops higher above her left
eye and a little higher above her right eye, where the forehead shows between the bangs as on the sheet.""")
    CAPM = "(1.75-0.42*exp(-((v+1.3)/0.1)^2)-0.2*exp(-((v+1.15)/0.15)^2)-0.1*exp(-((v+2.12)/0.12)^2))"
    with sc.turned(**HEAD_TURN):
        sc.surface((f"{fmt(HDX * 0.98)}*sin(u*{CAPM})*cos(v)", f"{fmt(HY0)}+{fmt(HDY * 0.98)}*sin(u*{CAPM})*sin(v)", f"{fmt(ZHC)}+{fmt(HDZ * 0.98)}*cos(u*{CAPM})"),
               ('u', 0, 1), ('v', '-pi', 'pi'), HAIR, 'hairCap', ('point', (0, HY0, ZHC)))
    # Arch lights one side of a surface, so the under-layer is two sheets 0.005 apart: the outer one lit from behind
    # (where it shows between the back locks), the inner one from the front (where it shows beside her face and neck)
    with sc.turned(**HEAD_TURN, blend=HB):
        for ref_, d_, ori_ in (('hairUnder', 0.03, ('point', (0, 0, 9))), ('hairUnderIn', 0.035, ('tube', 'u', 'in'))):
            sc.surface((f"(hrx(t)-{fmt(d_)})*sin(k*(1-0.64*(1-hdw(t))))", f"{fmt(HY0)}+(hry(t)-{fmt(d_)})*cos(k*(1-0.64*(1-hdw(t))))", "hzz(t)"),
                       ('t', T0 * 0.9, t_of_z(BK_NAPE) - 0.03), ('k', -1.2, 1.2), HAIR_DK if ref_ == 'hairUnder' else HAIR, ref_, ori_)

    rng = np.random.default_rng(11)
    R = rng.random
    def cdir(rc):
        """How a long lock ends: in a small hook that tightens toward its point, as the user's reference draws long
        hair's ends (HAIR_CURL; the earlier hooks and scrolls of a fixed three quarters of a turn read as unnatural)."""
        R()
        return ('hook', HAIR_CURL['turn'] * (0.85 + 0.3 * R()), HAIR_CURL['length'] * (0.9 + 0.2 * R()))
    T_NAPE = t_of_z(BK_NAPE)

    def nape_lock(ref, p0, sg, hw, la, th, am, kk, ph, col):
        """A lock from the crown that ends at the nape, closing in as it falls so that it ends under the two back locks."""
        p_c = p0 * (1 - sg * (1 - hdw_py(T_NAPE)))
        dr = (math.copysign(min(abs(p_c), 0.42), p0) - p_c) / hu_py(T_NAPE)
        lock(ref, p0, sg, dr, hw, T_NAPE, la, th, am, kk, ph, col, curl=None, smooth=True, outline=OUTLINE_W)

    sc.comment(f"""Long locks. The hair down the middle of her back is one curtain from under the braid (drawn below). So
the locks from the crown down the back (three layers: wide ones underneath, the main layer, thin lighter strands
on top) end at the nape (z {fmt(BK_NAPE)}), closing in under it, except the outermost of each layer, which fall
over the arms with the locks at the sides that fan out over the shoulders. Each long lock falls in one gentle
curve and ends in a point that turns in toward the back.""")
    # the random draws are made as before the back was parted, so the side locks keep theirs. The outermost locks of
    # each layer stay long and are moved out over the arms, at least 1.32 from the back so they pass outside the
    # capelet's back emblems (which the sheet shows), where the sheet's back view shows the rest of the hair falling.
    W_KK = 2 * math.pi * VF * (1 - T0) / HAIR_WAVE['lam']      # the long hair's shared wave (HAIR_WAVE), in hu
    W_AM = 0.0                  # radians round the body (none: the long hair falls in one smooth curve, as the bangs)
    # the long locks at her sides end over the upper arm (z 9.2 to 9.7), under the side hair from the braid's ends,
    # which carries the hair down behind the arms facing back, as the user's picture draws it (round her sides the
    # long locks were seen edge-on from behind, as tentacles); the outermost locks of the back layers end at the nape
    SIDE_TE = (T0 + (ZHC - 9.7) / VF, 0.5 / VF)

    def wave_ph(p0):
        return HAIR_WAVE['phase'] + 0.15 * p0       # drifting a little round the body, as across the curtain

    def back_layer(tag, p0s, sg, hw, la, th, col, draw, arms):
        for i, p0 in enumerate(p0s):
            dr_, te_, am_, kk_, ph_, cd_ = draw()
            rank = sorted((abs(p) for p in p0s if p * p0 > 0), reverse=True).index(abs(p0)) if p0 != 0 else 99
            if rank < len(arms):        # the outermost: long, out over the arms (beyond the capelet's back emblems)
                # one big smooth clump a layer and side (the user asked for fewer, bigger clumps), as thick as a third of
                # its width where it falls, waving with the rest of the long hair
                pa = math.copysign(arms[rank], p0)
                lock(f'{tag}{i + 1}', pa, 0.3, dr_, hw * 1.25, te_, la * 1.8, max(th, 0.105), W_AM, W_KK, wave_ph(pa), col,
                     curl=cd_, full=True, band=False, smooth=True, outline=OUTLINE_W, paint=True)
            else:
                nape_lock(f'{tag}{i + 1}', p0, sg, hw, la, th, am_, kk_, ph_, col)
    back_layer('backU', (-1.0, -0.68, -0.34, 0.0, 0.34, 0.68, 1.0), 0.66, 0.28, 0.0, 0.05, HAIR_DK,
               lambda: (0.05 * (R() - 0.5), 0.86 + 0.1 * R(), 0.015 + 0.01 * R(), 3 + 1.5 * R(), 6.28 * R(), cdir(0.11)), ())
    back_layer('back', (-0.85, -0.51, -0.17, 0.17, 0.51, 0.85), 0.68, 0.24, 0.04, 0.055, HAIR,
               lambda: (0.06 * (R() - 0.5), 0.9 + 0.1 * R(), 0.018 + 0.01 * R(), 3 + 1.5 * R(), 6.28 * R(), cdir(0.13)), ())
    back_layer('backT', (-0.6, 0.0, 0.6), 0.68, 0.17, 0.085, 0.04, HAIR_LT,
               lambda: (0.12 * (R() - 0.5), 0.8 + 0.18 * R(), 0.02 + 0.012 * R(), 3 + 1.5 * R(), 6.28 * R(), cdir(0.12)), ())
    # the long locks at her sides, over the shoulders and arms: a few big smooth clumps in three layers (the user asked
    # for fewer, bigger clumps: ten narrow ones a side read as stripes). p0 (her right), layer, half-width, colour.
    side = [(1.28, 'U', 0.36, HAIR_DK), (1.72, 'U', 0.34, HAIR_DK),
            (1.15, 'M', 0.3, HAIR), (1.5, 'M', 0.3, HAIR), (1.85, 'M', 0.27, HAIR),
            (1.35, 'T', 0.22, HAIR_LT)]
    side_left = [(1.2, 'U', 0.36, HAIR_DK), (0.98, 'M', 0.3, HAIR), (1.1, 'T', 0.22, HAIR_LT)]    # her left: the few
                        # long locks behind the ear, spread round instead of overlapping
    # her left ear shows: the hair above it is swept back under the ornament (as many as before, so the head is covered)
    tucks = [(2, 1.5, 'U', 0.26, HAIR_DK), (3, 1.78, 'U', 0.25, HAIR_DK), (5, 1.36, 'M', 0.21, HAIR), (6, 1.6, 'M', 0.21, HAIR),
             (7, 1.85, 'M', 0.19, HAIR), (9, 1.5, 'T', 0.15, HAIR_LT), (10, 1.74, 'T', 0.14, HAIR_LT)]
    for n, p0, lay, hw_, col in tucks:
        lock(f'tuck{n}', p0, 0.0, -1.2, hw_, T0 + 0.05, {'U': 0.0, 'M': 0.035, 'T': 0.07}[lay], 0.05, 0.0, 1, 0, col, curl=None)
        if lay != 'U':
            crown_shine(f'tuck{n}', p0, 0.0, -1.2, hw_, T0 + 0.05, {'M': 0.035, 'T': 0.07}[lay], 0.05, 0.0, 1, 0)
    for sd, tag, locks in ((-1, 'R', side), (1, 'L', side_left)):
        for i, (p0, lay, hw_, col) in enumerate(locks):
            la_, th_ = {'U': (0.0, 0.105), 'M': (0.08, 0.105), 'T': (0.16, 0.085)}[lay]
            dr_, te_, am_ = -sd * (0.1 + 0.1 * R()), SIDE_TE[0] + SIDE_TE[1] * R(), W_AM * (0.9 + 0.2 * R())
            R(), R()            # (the wave's rate and phase are shared now; still drawn, so the later draws stay the same)
            args_ = (sd * p0, 0.25, dr_, hw_, te_, la_, th_, am_, W_KK, wave_ph(sd * p0))
            cdir(0.13)          # (its draws, so the later ones stay the same; it ends in a point under the side hair)
            lock(f'side{tag}{i + 1}', *args_, col, curl=None, full=True, band=False, smooth=True, outline=OUTLINE_W, paint=True)
            if lay != 'U':
                crown_shine(f'side{tag}{i + 1}', *args_[:5], la_ * 0.55, th_ * 0.55, *args_[7:], smooth=True)      # on the head the lock is 0.55 as full
    # behind her left ear, below the swept-back hair: locks falling past the ear, as the portrait shows them (two big
    # round clumps, staggered in depth)
    for i, (p0, lay, hw_, col) in enumerate(((1.45, 'M', 0.28, HAIR), (1.72, 'M', 0.26, HAIR))):
        la_, th_ = 0.08 + 0.05 * i, 0.105
        lock(f'earBackLock{i + 1}', p0, 0.25, -(0.05 + 0.05 * i), hw_, SIDE_TE[0] + 0.04 * i, la_, th_, W_AM, W_KK, wave_ph(p0), col,
             curl=None, full=True, band=False, smooth=True,
             outline=OUTLINE_W, paint=True)
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
        tc = poly_cross(C, BAND_Z + 0.08 * math.sin(1.9 * len(key) + ord(key[-1])))     # the ring at the shoulders, staggered
        if tc is not None:
            poly_band(f'chestLock{key}', C, U, V, tc, 0.18, ph0=1.3 * len(key) + (0.7 if key[0] == 'L' else 0.0))
    sc.comment("""The hair down the back, from under the braid, laid out as the chest locks are: three locks, each a
bundle of a wide strand, a narrower one on it and a thin one beside it, that sweep together, part toward their
ends and curl in toward the middle of the back at their own heights, with wide darker clumps under the gaps.
Short strands under the braid cover the back of the head beside them. Each is written like the chest locks (in
pieces); each wanders and varies in width and thickness a little, so that none is machine-smooth.""")
    global HEAD_BACK
    HEAD_BACK = head_back_grid()
    BK = back_strands()
    print(f"  back locks: largest fit error {max(v[1] for v in BK.values()):.4f}", file=sys.stderr)
    for key, (subparts, err, cols, _) in BK.items():
        for k, parts in enumerate(subparts):
            letters = iter('abcdefghijklmnop')
            for part in parts:
                if part[0] == 'poly':
                    _, (C, U, V), n = part
                    xyz = tuple(f"{horner(C[i], 't')}+cos(w)*({horner(U[i], 't')})+lcs(w)*({horner(V[i], 't')})" for i in range(3))
                    name = f'backLock{key[2:]}{key[1]}{k + 1}'
                else:
                    _, xyz, n = part
                    name = f'backLock{key[2:]}{key[1]}{k + 1}Curl'
                for j in range(n):
                    tag = next(letters) if part[0] == 'poly' else 'abcd'[j]
                    sc.surface(xyz, ('t', -1 + 2 * j / n, -1 + 2 * (j + 1) / n), ('w', 0, '2*pi'), cols[k], f'{name}{tag}', ('tube', 'u'))
                    if part[0] == 'poly' and OUTLINE_W:     # its line art: both edges (w = 0 and pi, where lcs is 0), lifted a little
                        for sgn_, nm in ((1, 'A'), (-1, 'B')):
                            lv = OUTLINE_LIFT / max(1e-6, BK_LOOK[key][1])
                            exy = tuple(horner(list(np.round(np.polynomial.polynomial.polyadd(np.polynomial.polynomial.polyadd(C[i], sgn_ * np.asarray(U[i], float)),
                                                                                                lv * np.asarray(V[i], float)), 6)), 't')
                                        for i in range(3))      # (one polynomial: C + U + lift V summed)
                            sc.curve(exy, ('t', -1 + 2 * j / n, -1 + 2 * (j + 1) / n), HAIR_OUTLINE, f'{name}{tag}Line{nm}', thickness=OUTLINE_W)
        # painted as the bangs are (strand lines, a sheen, the shade under the braid: paint_hair)
        if key not in BK_UNDER and max(w for _, _, w in BK_STRANDS[key]) * BK_WIDE >= 0.1:     # (not the short or thin strands)
            for k, parts in enumerate(subparts):
                polys = [p[1] for p in parts if p[0] == 'poly']
                PAINT.append(dict(kind='poly', ref=f'backLock{key[2:]}{key[1]}{k + 1}', polys=polys,
                                  seed=5.1 * list(BK_STRANDS).index(key) + 1.7 * k))

    paint_hair()

    # ---------- bangs and the locks that frame the face ----------
    sc.blank()
    sc.note('Bangs')
    sc.comment("""Bangs and the locks framing the face are blades: a Bezier centre line (written as a polynomial in t)
that leaves the crown, follows the dome to the hairline and hangs to a point just in front of the face.
Across it (w) the blade has the same crescent section as the long locks; its width and thickness
vectors are polynomials too.""")
    fyf = sc.ns['fy']

    def blade(ref, pts, nrms, wid, thk, color, sheen=None, lines=(0.3,), crown=(0.16, 0.36), shade=None):
        """sheen=(t0, t1): a lighter streak down the front of the blade between t0 and t1, as anime hair is shaded.
        lines: dark strand lines down its front, at these fractions of its half-width, as the portrait draws hair."""
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
        for k, a in enumerate(lines):
            tl = "max(0,sin(pi*(t-0.42)/0.55))^0.7"
            lx = tuple(f"{horner(C[i], 't')}+({fmt(a)}+0.045*cos(w))*{tl}*({horner(U[i], 't')})+{fmt(1.06 * math.sqrt(1 - a * a))}*({horner(Vv[i], 't')})" for i in range(3))
            FRONT_BLADES.append((f'{ref}Line{k + 1}', lx))
            with sc.turned(**HEAD_TURN):
                sc.surface(lx, ('t', 0.42, 0.97), ('w', 0, 'pi'), HAIR_LINE, f'{ref}Line{k + 1}', ('dir', (0, -1, 0.3)))
        if shade:       # the shade where the lock passes under the crown's overhang at the hairline, painted on
            t0, t1 = shade
            # full width where it comes from under the crown, narrowing to a point toward the tip; translucent, in two steps
            for k, (te, op) in enumerate(((t1, 0.32), (t0 + 0.6 * (t1 - t0), 0.32))):
                tp = f"min(1,({fmt(te)}-t)/{fmt(0.45 * (te - t0))})^0.7"
                sx = tuple(f"{horner(C[i], 't')}+0.96*cos(w)*{tp}*({horner(U[i], 't')})+{fmt(1.08 + 0.02 * k)}*({horner(Vv[i], 't')})" for i in range(3))
                with sc.turned(**HEAD_TURN):
                    sc.surface(sx, ('t', t0, te), ('w', 0, 'pi'), HAIR_SHADE, f'{ref}Shade{k + 1}', ('dir', (0, -1, 0.3)), opacity=op)
        if crown:       # the angel ring: a zigzag band across the blade where it crosses the ring's height on the dome
            ts = np.linspace(0.0, 0.7, 141)
            Pc = np.array([poly_at(C, t_) for t_ in ts])
            polar = np.arccos(np.clip(((Pc[:, 2] - ZHC) / HDZ) / np.linalg.norm(np.c_[Pc[:, 0] / HDX, (Pc[:, 1] - HY0) / HDY, (Pc[:, 2] - ZHC) / HDZ], axis=1), -1, 1))
            k = np.nonzero(polar >= RING_POLAR)[0]
            if len(k) and k[0] > 0:
                with sc.turned(**HEAD_TURN):
                    poly_band(ref, C, U, Vv, float(ts[k[0]]), 0.07, ph0=3.1 * pts[0][0])
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

    def strand(ref, xtop, mid, tip, hw, color, lift=0.0, tk=0.75, off=0.04, sweep=-0.12, sheen=None, hug=False, crown=(0.16, 0.36)):
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
        k_ = 1.0 if ref.startswith('wisp') or ref == 'bangUnder' else 1.4      # clumps, not strips (the wisps stay fine)
        blade(ref, [p0, p1, p2, p3, p4], [dome_normal(p0), dome_normal(p1), dome_normal(p2), unit((0.4 * x3, -1, 0.15)), unit((0.4 * xt, -1, 0.0))],
              [0.2 * hw[0] * k_, 0.9 * hw[0] * k_, hw[0] * k_, hw[1] * k_, 0.0], [0.008 * tk, 0.05 * tk, 0.055 * tk, 0.045 * tk, 0.0], color,
              sheen=None if sheen is None else (0.52, sheen, HAIR_HI), crown=crown, shade=None if k_ == 1.0 else (0.37, 0.62))

    # As on the sheet's portrait: one wide lock sweeps down to a point at the bridge of the nose, the forehead
    # shows in a tall triangle above her left eye, a lock sweeps over her left temple, and above her right eye a
    # smaller gap is framed by the locks at the side (bang5-6). The wide lock and the temple lock are made of
    # strands along the strand lines drawn on the sheet, with staggered pointed tips; a thin dark layer under the
    # wide lock keeps the forehead from showing between its strands. Strands are stacked (lift) so that they lie
    # on each other, and the main ones carry a lighter streak (sheen).
    strand('bangUnder', -0.08, (-0.09, 0.25), (-0.05, -0.03), (0.23, 0.16), HAIR_SH, tk=0.3, crown=None)
    strand('bang1', -0.23, (-0.23, 0.25), (-0.14, -0.08), (0.07, 0.055), HAIR, lift=0.018, sheen=0.8)
    strand('bang2', -0.31, (-0.34, 0.3), (-0.36, 0.17), (0.045, 0.035), HAIR_DK, lift=0.03, sheen=0.8)
    strand('bang3', -0.13, (-0.12, 0.25), (-0.06, -0.09), (0.065, 0.05), HAIR_LT, lift=0.032, sheen=0.78)
    strand('bang4', -0.03, (-0.04, 0.25), (-0.01, -0.1), (0.07, 0.055), HAIR, lift=0.015, sheen=0.82)
    strand('bang10', 0.08, (0.06, 0.25), (0.04, -0.06), (0.07, 0.05), HAIR_LT, lift=0.03, sheen=0.78)
    strand('bang11', 0.145, (0.115, 0.28), (0.045, 0.025), (0.045, 0.033), HAIR_DK, lift=0.042)
    strand('bang12', -0.17, (-0.13, 0.42), (-0.08, 0.2), (0.05, 0.035), HAIR_LT, lift=0.05, sheen=0.75)
    strand('bang13', 0.12, (0.11, 0.38), (0.06, 0.13), (0.045, 0.03), HAIR, lift=0.056, sheen=0.75)
    bang('bang5', (-0.6, -0.8, -0.88), (-0.56, ZE - 0.02), 0.17, HAIR_LT, lift=0.03)
    bang('bang6', (-0.7, -1.05, -1.27), (-0.6, ZE - 0.02), 0.18, HAIR)
    strand('bang7', 0.3, (0.4, 0.3), (0.55, 0.07), (0.07, 0.05), HAIR, sweep=-0.1, sheen=0.8)
    strand('bang9', 0.42, (0.51, 0.3), (0.62, 0.0), (0.08, 0.065), HAIR_LT, lift=0.02, sweep=-0.1, sheen=0.8)
    strand('bang14', 0.52, (0.55, 0.38), (0.53, 0.14), (0.05, 0.035), HAIR_DK, lift=0.04, sweep=-0.1)
    strand('bang15', -0.43, (-0.47, 0.32), (-0.5, 0.06), (0.08, 0.06), HAIR, lift=0.022, sheen=0.8)
    bang('bang8', (1.0, 1.26, 1.42), (0.66, ZE + 0.1), 0.16, HAIR_LT, lift=0.03)
    for sd, tag in ((1, 'L'), (-1, 'R')):
        pts = [dome(0.3, 1.5 * sd, 1.005), dome(1.15, 1.62 * sd, 1.04), (0.71 * sd, -0.07, ZE + 0.06), (0.69 * sd, -0.13, ZE - 0.2)]
        xt, zt = 0.6 * sd, ZE - 0.38
        pts.append((xt, fyf(xt, zt) - 0.045, zt))
        if sd > 0:      # her left ear shows on the portrait: this lock passes behind it
            pts = [pts[0], pts[1], (0.8, 0.3, ZE + 0.06), (0.78, 0.36, ZE - 0.2), (0.72, 0.34, ZE - 0.38)]
        nr = [dome_normal(pts[0]), dome_normal(pts[1]), unit((sd, -0.25, 0)), unit((sd, -0.3, 0)), unit((sd, -0.5, 0))]
        blade(f'frame{tag}', pts, nr, [0.04, 0.15, 0.14, 0.1, 0.0], [0.008, 0.04, 0.04, 0.03, 0.0], HAIR)

    # her right: a wide lock from the temple down past the jaw, outside the cheek locks, so the side of her head is front
    # hair (as drawn), not the inside of the long hair seen edgewise
    pts = [dome(0.35, -1.6, 1.01), dome(1.1, -1.7, 1.05), (-0.8, -0.02, ZE - 0.05), (-0.82, -0.06, ZE - 0.45), (-0.8, -0.1, ZE - 0.88)]
    nr = [dome_normal(pts[0]), dome_normal(pts[1]), unit((-0.35, -1, 0)), unit((-0.3, -1, 0)), unit((-0.3, -1, 0))]
    blade('templeR', pts, nr, [0.04, 0.16, 0.17, 0.15, 0.0], [0.008, 0.04, 0.045, 0.035, 0.0], HAIR, lines=(-0.45, 0.0, 0.4))
    sc.comment("""Thin wisps from the bangs that cross the eyes, and locks that fall from the temples to below the jaw:
beside the face on her right, so the jaw line shows, and over the outer cheek on her left, as in the portrait.""")
    bang('wisp1', (0.22, 0.02, -0.14), (-0.25, ZE - 0.07), 0.035, HAIR_LT, off=0.042, tk=0.45)
    bang('wisp2', (0.3, 0.22, 0.16), (0.2, ZE + 0.02), 0.03, HAIR_LT, off=0.042, tk=0.45)
    bang('wisp3', (-0.45, -0.78, -0.98), (-0.5, ZE - 0.1), 0.032, HAIR, off=0.042, tk=0.45)
    # thin strands that fall past her left cheek and flick inward, and one beside her right cheek, as drawn on the sheet
    strand('wisp4', 0.62, (0.56, -0.1), (0.4, -0.34), (0.04, 0.03), HAIR_LT, lift=0.02, off=0.03, sweep=-0.05, hug=True, crown=None)
    strand('wisp5', 0.63, (0.54, -0.24), (0.31, -0.47), (0.035, 0.026), HAIR, lift=0.025, off=0.03, sweep=-0.05, hug=True, crown=None)
    strand('wisp6', 0.62, (0.5, -0.4), (0.35, -0.56), (0.03, 0.022), HAIR_LT, lift=0.03, off=0.03, sweep=-0.05, hug=True, crown=None)
    strand('wisp7', -0.6, (-0.62, 0.2), (-0.63, -0.22), (0.05, 0.04), HAIR_LT, lift=0.03, sweep=0.05, hug=True, crown=None)

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

    # her left: the hair over the temple swept back above the ear under the ornament, as drawn (from her left the side of
    # her forehead showed bare between the bangs and the ornament): three strands one above another, on the skin over
    # the temple, then round the side of the head above the ear
    for k, (dz, w_, col) in enumerate(((0.1, 0.075, HAIR_LT), (0.0, 0.085, HAIR), (-0.09, 0.075, HAIR_DK))):
        za, zb = ZE + 0.42 + dz, ZE + 0.28 + dz
        pts = [(0.55, fyf(0.55, za) - 0.035, za), (0.615, fyf(0.615, zb) - 0.035, zb), (0.675, -0.06, ZE + 0.19 + dz),
               (0.725, 0.14, ZE + 0.13 + 0.6 * dz), (0.7, 0.36, ZE + 0.08 + 0.5 * dz)]
        nr = [face_normal(0.55, za), face_normal(0.615, zb), unit((1, -0.3, 0.15)), unit((1, 0.05, 0.1)), unit((0.85, 0.45, 0.1))]
        nr = [unit((-n_[0], -n_[1], -n_[2])) if n_[1] > 0 else n_ for n_ in nr[:2]] + nr[2:]
        blade(f'sweptL{k + 1}', pts, nr, [0.035, 0.85 * w_, w_, 0.9 * w_, 0.03], [0.008, 0.03, 0.035, 0.03, 0.012], col,
              sheen=(0.15, 0.55, HAIR_HI), crown=None)
    for sd, tag in ((1, 'L'), (-1, 'R')):
        if sd < 0:      # her right: the hair falls beside the face, so the whole jaw line shows
            cheek_lock(f'cheekLock{tag}1', sd, 0.0, 0.63, 0.6, 0.5, ZE - 0.86, 0.075, HAIR)
            cheek_lock(f'cheekLock{tag}2', sd, -0.07, 0.61, 0.57, 0.45, ZE - 0.74, 0.05, HAIR_LT)
            cheek_lock(f'cheekLock{tag}3', sd, 0.06, 0.645, 0.63, 0.56, ZE - 0.98, 0.065, HAIR_DK)
        else:           # her left: the hair covers the outer cheek, leaving about 0.42 of the face showing
            cheek_lock(f'cheekLock{tag}1', sd, 0.0, 0.6, 0.5, 0.41, ZE - 0.82, 0.08, HAIR)
            cheek_lock(f'cheekLock{tag}2', sd, -0.07, 0.58, 0.47, 0.36, ZE - 0.68, 0.05, HAIR_LT)
            cheek_lock(f'cheekLock{tag}3', sd, 0.06, 0.615, 0.53, 0.44, ZE - 0.96, 0.065, HAIR_DK)

    sc.blank()
    sc.note('Brows')
    with sc.turned(**HEAD_TURN):
        draw_brows(FRONT_BLADES)
        sc.blank()
        sc.note('Shadows of the bangs')
        draw_hair_shadows(FRONT_BLADES)

    # ---------- the braid crown ----------
    sc.blank()
    sc.note('Braid')
    sc.comment("""A braided crown around the back of the head. The braid follows an oval ring on the dome: bdx, bdy, bdz
give the direction from the head's centre to the ring at angle a, with ro scaling the ring (1 = its
centre line). Its three strands weave as sin(tau) across the ring and sin(2 tau) in height, tau a third
of a turn apart: the classic three-strand braid.""")
    THC, TH_, TV_ = BRAID_RING
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
        THC, TH_, TV_ = BRAID_RING
    # the ornament sits on the side of the head above her left ear, level with her temple, its flower turned toward
    # the viewer: placed where the hi-res portrait draws it, seen from the portrait's viewpoint
    P_ = dome(1.66, 1.51, 1.14)
    ORN_N = unit(add(mul(dome_normal(P_), 0.5), (0.059, -0.978, 0.202)))
    ORN_P = add(P_, mul(ORN_N, 0.12))
