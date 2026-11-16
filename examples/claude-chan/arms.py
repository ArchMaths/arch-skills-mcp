# Arms and sleeves, built from outlines traced over the sheet's five views of them (run inside generate.py).
import sys
import numpy as np

# The sleeve: a balloon puff from under the capelet to the elbow, gathered into a small frilled band; a fitted
# forearm sleeve laced on its outer front; a flared frill cuff whose hem is square to the forearm; the hand.
#
# Measured on the arm outlines traced over the sheet's front view (both arms), side view (her left arm) and
# back view (both arms). The views disagree a little on heights (the side view draws the forearm about 0.25
# longer than the front view); each view's landmarks (capelet hem, widest puff, elbow band, forearm top,
# cuff top, cuff hem, fingertips) are mapped to their mean over the five views. The side view draws the arms
# turned about 30 degrees, as it does the dress's bows: read at azimuth -30 the puff is centred beside the
# shoulder (read at 0 it would hang behind her back), and the forearm, cuff and hand come forward and out.
# The back view draws them about 8% closer to the body than the front view; the model follows the front.

# ---------------------------------------------------------------------------------------------------
#  The drawing: her left arm (the right arm is its mirror image), horizontal sections at each height
#  (centre cx, cy and half-widths A across, B front to back), from the front view (x) and the side view
#  read at azimuth -30 (y). Above 8.8 the side view shows the capelet's back, so there the puff is taken
#  round; above 9.07 it is under the capelet.
# ---------------------------------------------------------------------------------------------------
SL_Z = [9.62, 9.45, 9.25, 9.04, 8.92, 8.80, 8.68, 8.56, 8.47, 8.38, 8.32, 8.26, 8.20, 8.14, 8.02, 7.84, 7.66, 7.48, 7.30, 7.12, 6.95, 6.75]
SL_CX = [0.98, 1.04, 1.10, 1.132, 1.165, 1.20, 1.238, 1.263, 1.266, 1.257, 1.25, 1.264, 1.29, 1.355, 1.40, 1.445, 1.49, 1.53, 1.57, 1.605, 1.646, 1.695]
SL_CY = [0.0, -0.02, -0.03, -0.03, -0.02, 0.0, 0.06, 0.05, 0.035, 0.045, 0.08, 0.08, 0.05, -0.02, -0.065, -0.13, -0.18, -0.225, -0.27, -0.31, -0.36, -0.415]
SL_A = [0.15, 0.20, 0.235, 0.25, 0.285, 0.32, 0.366, 0.424, 0.428, 0.41, 0.37, 0.315, 0.279, 0.232, 0.228, 0.228, 0.226, 0.23, 0.235, 0.24, 0.24, 0.24]
SL_B = [0.16, 0.22, 0.26, 0.28, 0.32, 0.38, 0.438, 0.428, 0.43, 0.425, 0.376, 0.342, 0.299, 0.25, 0.232, 0.215, 0.205, 0.195, 0.19, 0.19, 0.19, 0.19]
# the drawn fold lines' tops: up the 16 hollows (odd k) to scattered heights under the capelet, up the crests (even
# k) only through the gathers above the band
SL_FOLD_TOP = [8.45 + 0.06 * ((math.sin(1.7 * k) + 1) / 2) if k % 2 == 0 else 8.95 + 0.4 * ((math.sin(2.39 * k + 0.7) + 1) / 2) for k in range(32)]
SL_TOP, SL_END = 9.62, 6.95          # the sleeve's top (inside the capelet) and its end (inside the cuff)
FORE = (0.243, -0.284)               # the forearm's centre moves this far out (x) and forward (y) per unit drop
# elbow band: a small frill under the puff's gathers
BAND_Z = (8.29, 8.15)
# lacing: crossed cord on the outer front of the forearm, centred 30 degrees round from the front (the front
# view draws it 0.45 of the half-width out from the centre; the side and back views draw it facing them)
LACE_T, LACE_W, LACE_Z, LACE_N = -1.05, 0.33, (8.04, 7.22), 6
# cuff: a frill sewn on square to the forearm whose hem hangs level; its inner side falls straight (the
# front view's inner edge stays at 1.35), the outer side flares to the hem, which is 0.07 higher on the outer side
# than the inner (35% of square to the forearm) and level front to back; radii across and front to back
CUFF_Z = (7.13, 6.74)
CUFF_HEM = (1.77, -0.44)             # the hem's centre (x from the front view, y from the side view)
CUFF_R0, CUFF_R1 = (0.25, 0.215), (0.415, 0.41)
CUFF_N, CUFF_P = 12, 1.5             # waves at the hem; how late the frill flares (v^CUFF_P)
CUFF_DENT = 0.3                      # the hem's inner back quarter lies against the overskirt, pressed in
# hand: on a line 0.1 out from the forearm's (the wrist bends a little), through the measured centre
# HAND_C, hanging 0.38 out and 0.21 forward per unit drop; back of the hand toward the front and out, palm
# toward the body and back; fingertips at 5.94
HAND_C = (1.861, -0.505, 6.58)
HAND_OUT = 0.08                      # the hand's line sits this far out of the outline's centre (thumb and curl are inward)
HAND_D = (0.38, -0.21, -1.0)
HAND_BREADTH = (0.707, 0.707, 0.0)   # across the knuckles, little finger side: out and back (0.36 wide in both views)
HAND_TIP = 5.94
SL_LACE, SL_SKIN = '#8c513e', SKIN


# ---------------------------------------------------------------------------------------------------
#  Fit: least-squares cubic splines in the height q, written as truncated powers
# ---------------------------------------------------------------------------------------------------
_SK = [9.25, 8.92, 8.68, 8.47, 8.32, 8.20, 8.08, 7.6, 7.12]


def _sbasis(q):
    q = np.atleast_1d(np.asarray(q, float))
    d = q - 8.0
    return np.stack([np.ones_like(q), d, d ** 2, d ** 3] + [np.maximum(0, q - k) ** 3 for k in _SK], 1)


def _sfit(vals):
    c, *_ = np.linalg.lstsq(_sbasis(SL_Z), np.asarray(vals, float), rcond=None)
    return c


def _sev(c, q):
    return _sbasis(q) @ c


def _sarch(c, var='q'):
    terms = [fmt(c[0]), f"{fmt(c[1])}*({var}-8)", f"{fmt(c[2])}*({var}-8)^2", f"{fmt(c[3])}*({var}-8)^3"]
    terms += [f"{fmt(ck)}*max(0,{var}-{fmt(k)})^3" for ck, k in zip(c[4:], _SK)]
    return '+'.join(terms).replace('+-', '-')


SL_FIT = {k: _sfit(v) for k, v in (('slx', SL_CX), ('sly', SL_CY), ('sla', SL_A), ('slb', SL_B))}
SL_ERR = {k: float(np.max(np.abs(_sev(c, SL_Z) - v))) for (k, c), v in zip(SL_FIT.items(), (SL_CX, SL_CY, SL_A, SL_B))}


# The puff alone (where the fold lines run, 8.22 to 9.35), as polynomials in x = (q - 8.825) / 0.625: the lines paste their
# height expression into every use of q, and the splines' max(0, q - k) terms use it twice each.
_PQ = np.linspace(8.2, 9.45, 400)
SL_PUFF = {k: np.polynomial.chebyshev.Chebyshev.fit((_PQ - 8.825) / 0.625, _sev(c, _PQ), 12).convert(kind=np.polynomial.Polynomial).coef
           for k, c in SL_FIT.items()}
SL_PUFF_ERR = max(float(np.max(np.abs(np.polynomial.polynomial.polyval((_PQ - 8.825) / 0.625, SL_PUFF[k]) - _sev(c, _PQ))))
                  for k, c in SL_FIT.items())
print(f'  sleeve puff polynomials (for the fold lines): max error {SL_PUFF_ERR:.4f}', file=sys.stderr)


def slx_(q): return _sev(SL_FIT['slx'], q)
def sly_(q): return _sev(SL_FIT['sly'], q)
def sla_(q): return _sev(SL_FIT['sla'], q)
def slb_(q): return _sev(SL_FIT['slb'], q)


def slg_(q):
    """Gathers: deep at the elbow band, light folds over the puff, none on the forearm."""
    q = np.asarray(q, float)
    return 0.035 * np.exp(-((q - 8.27) / 0.09) ** 2) + 0.007 * (1 + np.tanh((q - 8.4) / 0.07))


# (a smooth ramp onto the puff's folds: Arch pastes min's and max's arguments twice, and the fold lines paste their
# whole height expression into this)
SLG_EXPR = '0.035*exp(-((q-8.27)/0.09)^2)+0.007*(1+tanh((q-8.4)/0.07))'


# ---------------------------------------------------------------------------------------------------
#  Hand: a frame from the measured direction; palm, wrist, four fingers and a thumb as capsules
# ---------------------------------------------------------------------------------------------------
def hand_parts(side=1):
    """[(kind, args, ref suffix)] for one hand; her left hand for side 1, mirrored for -1."""
    d = unit(HAND_D)
    eb = unit(sub(HAND_BREADTH, mul(d, dot(HAND_BREADTH, d))))      # across, toward the little finger
    en = unit(cross(d, eb))                                            # palm normal
    if en[0] > 0:                                                      # the palm faces the body
        en = mul(en, -1)
    W = (0.0, 0.0, 0.0)
    parts = []

    def P(s, b, n):
        return add(add(add(W, mul(d, s)), mul(eb, b)), mul(en, n))

    def bent(base, dirn, length, curl, r0, r1, tag):
        """A finger: two segments, the second turned further toward the palm."""
        k1 = unit(add(mul(dirn, math.cos(curl)), mul(en, math.sin(curl))))
        m = add(base, mul(k1, 0.55 * length))
        k2 = unit(add(mul(dirn, math.cos(2.4 * curl)), mul(en, math.sin(2.4 * curl))))
        tip = add(m, mul(k2, 0.45 * length))
        parts.append(('capsule', (base, add(m, mul(k1, 0.01)), r0, 0.5 * (r0 + r1)), tag + 'a'))
        parts.append(('capsule', (m, tip, 0.5 * (r0 + r1), r1), tag + 'b'))
        return tip

    parts.append(('capsule', (P(-0.34, 0, 0), P(0.12, 0, 0.0), 0.11, 0.105), 'Wrist'))
    parts.append(('ellipsoid', (P(0.27, 0.0, 0.0), (0.168, 0.065, 0.27), (eb, en, d)), 'Palm'))
    tips = []
    for k, (b, ln, spread, curl) in enumerate(((-0.112, 0.40, -0.03, 0.14), (-0.037, 0.44, -0.01, 0.17),
                                               (0.037, 0.41, 0.01, 0.2), (0.108, 0.33, 0.035, 0.24))):
        base = P(0.47 + (0.02 if k in (1, 2) else 0.0), b, 0.0)
        dirn = unit(add(d, mul(eb, spread)))
        tips.append(bent(base, dirn, ln, curl, 0.039, 0.027, f'Finger{k + 1}'))
    tdir = unit(add(add(d, mul(eb, -0.2)), mul(en, 0.3)))
    bent(P(0.13, -0.11, 0.04), tdir, 0.34, 0.25, 0.046, 0.032, 'Thumb')
    # place the wrist on the hand's line so that the lowest fingertip is at HAND_TIP
    low = min(t[2] for t in tips) - 0.026
    zw = HAND_TIP - low
    C = (HAND_C[0] + HAND_OUT + (HAND_C[2] - zw) * HAND_D[0], HAND_C[1] + (HAND_C[2] - zw) * HAND_D[1], zw)
    out = []
    for kind, args, tag in parts:
        if kind == 'capsule':
            A, B, r0, r1 = args
            A, B = add(A, C), add(B, C)
            out.append((kind, ((side * A[0], A[1], A[2]), (side * B[0], B[1], B[2]), r0, r1), tag))
        else:
            Cc, R, (a, b, c) = args
            Cc = add(Cc, C)
            mir = lambda v: (side * v[0], v[1], v[2])
            out.append((kind, (mir(Cc), R, (mir(a), mir(b), mir(c))), tag))
    return out, mir_pt(C, side), d


def mir_pt(p, side):
    return (side * p[0], p[1], p[2])


HANDS = {tag: hand_parts(side) for side, tag in ((1, 'L'), (-1, 'R'))}
HAND_PALM = {tag: next(a[0] for k, a, t in HANDS[tag][0] if t == 'Palm') for tag in HANDS}


def hand_reach(tag, n):
    """How far the hand reaches from its palm centre in direction n (for props held against it)."""
    n, C = unit(n), HAND_PALM[tag]
    best = 0.0
    for kind, a_, _ in HANDS[tag][0]:
        if kind == 'capsule':
            A, B, r0, r1 = a_
            best = max(best, dot(n, sub(A, C)) + r0, dot(n, sub(B, C)) + r1)
        else:
            Cc, R, fr_ = a_
            best = max(best, dot(n, sub(Cc, C)) + math.sqrt(sum((R[i] * dot(n, fr_[i])) ** 2 for i in range(3))))
    return best


def arm_points(side=1, n=40):
    """Points over the sleeve, cuff and hand (for the hair's clearance)."""
    pts = []
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for q in np.linspace(SL_END, 9.1, 60):
        a, b = sla_(q)[0] * (1 + slg_(q)), slb_(q)[0] * (1 + slg_(q))
        pts += [(side * (slx_(q)[0] + a * math.cos(w)), sly_(q)[0] + b * math.sin(w), q) for w in t]
    for v in np.linspace(0, 1, 8):
        pts += [mir_pt(p, side) for p in cuff_ring(v, t)]
    for kind, args, _ in HANDS['L' if side > 0 else 'R'][0]:
        if kind == 'capsule':
            A, B, r0, r1 = args
            pts += [add(A, mul(sub(B, A), s)) for s in np.linspace(0, 1, 6)]
    return np.array(pts)


CUFF_O = (CUFF_HEM[0] - float(slx_(CUFF_Z[1])[0]), CUFF_HEM[1] - float(sly_(CUFF_Z[1])[0]))   # hem centre off the forearm's line


def cuff_ring(v, t):
    zc = CUFF_Z[0] - v * (CUFF_Z[0] - CUFF_Z[1])
    f = v ** CUFF_P
    ra = CUFF_R0[0] + (CUFF_R1[0] - CUFF_R0[0]) * f
    rb = CUFF_R0[1] + (CUFF_R1[1] - CUFF_R0[1]) * f
    w = (1 + 0.07 * v ** 2 * np.cos(CUFF_N * t)) * (1 - CUFF_DENT * v ** 2 * np.maximum(0, np.sin(t)) * np.maximum(0, -np.cos(t)))
    dx, dy = ra * np.cos(t) * w, rb * np.sin(t) * w
    return list(zip(slx_(zc)[0] + f * CUFF_O[0] + dx, sly_(zc)[0] + f * CUFF_O[1] + dy, zc + (1 - 0.65 * v) * FORE[0] * dx + (1 - v) * FORE[1] * dy))


# how far out the arms reach at each height, and round which angles from the back (for the hair)
_ap = arm_points()
ARM_PSI = (float(np.min(np.arctan2(_ap[:, 0], _ap[:, 1] - 0.03))) - 0.12, float(np.max(np.arctan2(_ap[:, 0], _ap[:, 1] - 0.03))) + 0.12)
ARM_ENV = []
for _z in np.arange(9.1, HAND_TIP - 0.05, -0.15):
    _m = np.abs(_ap[:, 2] - _z) < 0.1
    if _m.any():
        ARM_ENV.append((round(float(_z), 3), round(float(np.max(np.hypot(_ap[_m, 0], _ap[_m, 1] - 0.03))) + 0.05, 3)))


# ---------------------------------------------------------------------------------------------------
#  Arch
# ---------------------------------------------------------------------------------------------------
def _emit_arms():
    sc.blank()
    sc.note('Arms: puffed sleeves gathered at the elbow, laced forearm sleeves, frill cuffs, hands')
    sc.comment("""Sleeve sections at each height q (measured on the sheet's front, side and back views): centre
slx, sly and half-widths sla across, slb front to back; slg the gathers at the elbow band and the puff's folds.""")
    for k, c in SL_FIT.items():
        sc.func(f'{k}(q)', _sarch(c))
    sc.func('slg(q)', SLG_EXPR)
    sc.func('slt(s)', '1-2*acos(cos(pi*s))/pi')          # triangle wave for the lacing, 1 at whole s
    for side, tag in ((1, 'L'), (-1, 'R')):
        S = fmt(side)
        g = f"(1+slg(q)*cos(16*t))"
        sc.surface((f"{S}*(slx(q)+sla(q)*cos(t)*{g})", f"sly(q)+slb(q)*sin(t)*{g}", 'q'),
                   ('q', SL_END, SL_TOP), ('t', 0, '2*pi'), CREAM, f'sleeve{tag}', ('tube', 'u'))
        # Drawn fold lines on the puff (soft grey-beige), one stroke a sleeve: a long line up each of the 16 hollows to a
        # scattered height under the capelet and a short one up each crest, where the gathers bunch above the band;
        # between lines the stroke runs round the sleeve under the band.
        if side > 0:
            sc.func('sfa(k)', "(1+cos(pi*k))/2*(8.45+0.03*(sin(1.7*k)+1))+(1-cos(pi*k))/2*(8.95+0.2*(sin(2.39*k+0.7)+1))")   # SL_FOLD_TOP
            for k_, c_ in SL_PUFF.items():
                sc.func(f'{k_}p(q)', horner(list(c_), '((q-8.825)/0.625)'))
            sc.func('sfq(t)', "8.22+(sfa(floor(t))-8.22)*max(0,1-abs(t-floor(t)-0.4)/0.4)")
            sc.func('sft(t)', f"{fmt(math.pi / 16)}*(floor(t)+max(0,(t-floor(t)-0.8)/0.2))")
        G_ = "(1.012+slg(sfq(t))*cos(16*sft(t)))"
        sc.curve((f"{S}*(slxp(sfq(t))+slap(sfq(t))*cos(sft(t))*{G_})", f"slyp(sfq(t))+slbp(sfq(t))*sin(sft(t))*{G_}", "sfq(t)"),
                 ('t', 0, 31.8), FOLD_LINE, f'sleeveFoldLines{tag}', thickness=1)
        # elbow band: a short frill below the puff's gathers, flaring a little, with 14 waves
        z0, z1 = BAND_Z
        qv = f"({fmt(z0)}-{fmt(z0 - z1)}*v)"
        fr = f"(1.04+0.13*v^1.5+0.05*v*cos(14*t))"
        sc.surface((f"{S}*(slx({qv})+sla({qv})*{fr}*cos(t))", f"sly({qv})+slb({qv})*{fr}*sin(t)", qv),
                   ('v', 0, 1), ('t', 0, '2*pi'), CREAM, f'sleeveBand{tag}', ('tube', 'u'))
        # lacing: two cords crossing between the eyelet rows, just off the sleeve
        za, zb = LACE_Z
        for j, sg in enumerate((1, -1)):
            qs = f"({fmt(za)}-{fmt((za - zb) / LACE_N)}*s)"
            tt = f"({fmt(LACE_T)}+{fmt(sg * LACE_W)}*slt(s))"
            sc.curve((f"{S}*(slx({qs})+(sla({qs})+0.012)*cos({tt}))", f"sly({qs})+(slb({qs})+0.012)*sin({tt})", qs),
                     ('s', 0, LACE_N), SL_LACE, f'sleeveLace{tag}{j + 1}', thickness=2)
        # cuff: sewn on square to the forearm (higher on the outer and front side), hem level, 12 waves
        qc = f"({fmt(CUFF_Z[0])}-{fmt(CUFF_Z[0] - CUFF_Z[1])}*v)"
        f_ = f"v^{fmt(CUFF_P)}"
        ra = f"({fmt(CUFF_R0[0])}+{fmt(CUFF_R1[0] - CUFF_R0[0])}*{f_})"
        rb = f"({fmt(CUFF_R0[1])}+{fmt(CUFF_R1[1] - CUFF_R0[1])}*{f_})"
        w = f"(1+0.07*v^2*cos({CUFF_N}*t))*(1-{fmt(CUFF_DENT)}*v^2*max(0,sin(t))*max(0,-cos(t)))"
        dx, dy = f"{ra}*cos(t)*{w}", f"{rb}*sin(t)*{w}"
        sc.surface((f"{S}*(slx({qc})+{fmt(CUFF_O[0])}*{f_}+{dx})", f"sly({qc})+{fmt(CUFF_O[1])}*{f_}+{dy}",
                    f"{qc}+(1-0.65*v)*{fmt(FORE[0])}*{dx}-(1-v)*{fmt(-FORE[1])}*{dy}"),
                   ('v', 0, 1), ('t', 0, '2*pi'), CREAM, f'cuff{tag}', ('tube', 'u'))
        for kind, hargs, part in HANDS[tag][0]:
            if kind == 'capsule':
                A, B, r0, r1 = hargs
                capsule(A, B, r0, r1, SL_SKIN, f'hand{part}{tag}')
            else:
                Cc, R, fr_ = hargs
                ellipsoid(Cc, R, SL_SKIN, f'hand{part}{tag}', fr=fr_)


if want('arms'):
    _emit_arms()
