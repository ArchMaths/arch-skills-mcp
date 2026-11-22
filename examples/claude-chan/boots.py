# Lace-up ankle boots, built from the three-view drawing measured on the sheet (run inside generate.py).
import numpy as np

# The drawing works in boot coordinates: s runs back from the toe tip, x across, z up from the ground.
# A boot is placed with its toe toward -y: model y = BOOT_Y0 + s.

# ---------------------------------------------------------------------------------------------------
#  The drawing: profiles measured on the sheet's three views (model units)
# ---------------------------------------------------------------------------------------------------
B_RIM = 1.10                 # rim height; on the sheet the petticoat frill hides the boots above about 1.0
B_SOLE = 0.05                # sole thickness
B_HEEL = 0.33                # ground to the heel breast
B_TREAD, B_BREAST = 0.72, 1.06       # end of the forefoot tread, front face of the heel
B_EYE_X = 0.16               # eyelets either side of the lacing's centre line
B_EYE_Z = [0.42 + 0.098 * k for k in range(7)]     # 7 pairs, as far apart as the front view's lace crossings
B_FACING = 0.205             # stitched facing either side of the eyelets
B_CAP = (0.30, 0.28)         # toe-cap seam: how far back it is on top, and how far forward it comes per unit drop

# side view: the upper's front edge F(z) and back edge B(z), s from the toe tip
_F_Z = [0.05, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.31, 0.35, 0.39, 0.43, 0.47, 0.51, 0.55, 0.63, 0.71, 0.79, 0.87, 0.94, 1.10]
_F_S = [0.035, 0.008, 0.0, 0.008, 0.02, 0.042, 0.088, 0.19, 0.315, 0.385, 0.42, 0.455, 0.485, 0.512, 0.56, 0.60, 0.635, 0.67, 0.69, 0.715]
_B_Z = [0.30, 0.38, 0.43, 0.47, 0.55, 0.63, 0.71, 0.79, 0.87, 0.94, 1.02, 1.10]
_B_S = [1.40, 1.425, 1.432, 1.437, 1.443, 1.438, 1.43, 1.40, 1.36, 1.317, 1.30, 1.298]
# front view: full width at each height; back view: width over the heel counter
_W_W = [0.52, 0.55, 0.568, 0.583, 0.585, 0.578, 0.567, 0.555, 0.528, 0.515, 0.50, 0.492, 0.489, 0.485, 0.48, 0.477, 0.469, 0.462, 0.458, 0.455]
_WC_Z = [0.30, 0.38, 0.47, 0.55, 0.63, 0.71, 0.79, 0.87, 0.94, 1.10]
_WC_W = [0.52, 0.51, 0.512, 0.505, 0.49, 0.478, 0.465, 0.458, 0.455, 0.455]
# plan: footprint of the upper where it is widest (z about 0.18)
_FP_S = [0, 0.03, 0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.28, 1.36, 1.41, 1.445]
_FP_W = [0, 0.13, 0.205, 0.25, 0.28, 0.2925, 0.285, 0.262, 0.24, 0.235, 0.24, 0.228, 0.19, 0.13, 0]


def _pchip(xs, ys):
    """Monotone cubic (PCHIP) interpolation, held constant outside the data."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    h, d = np.diff(xs), np.diff(ys) / np.diff(xs)
    m = np.zeros_like(ys)
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = d[0], d[-1]

    def f(t):
        t = np.clip(np.asarray(t, float), xs[0], xs[-1])
        i = np.clip(np.searchsorted(xs, t) - 1, 0, len(h) - 1)
        q = (t - xs[i]) / h[i]
        return ((1 + 2 * q) * (1 - q) ** 2 * ys[i] + q * (1 - q) ** 2 * h[i] * m[i]
                + q * q * (3 - 2 * q) * ys[i + 1] + q * q * (q - 1) * h[i] * m[i + 1])
    return f


_F, _B, _W, _WC = _pchip(_F_Z, _F_S), _pchip(_B_Z, _B_S), _pchip(_F_Z, _W_W), _pchip(_WC_Z, _WC_W)
_FPN = _pchip(_FP_S, np.array(_FP_W) / max(_FP_W))
_btz = np.linspace(0.12, B_RIM, 500)
_tf = _F(_btz)


def _T(s):
    """Top of the upper at s on the side profile: the toe box, the instep, then the rim."""
    return np.where(np.asarray(s) >= _tf[-1], B_RIM, np.interp(s, _tf, _btz))


def _sstep(a, b, t):
    t = np.clip((np.asarray(t, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _round(d, r):
    a = np.clip(d / r, 0, 1)
    return np.sqrt(1 - (1 - a) ** 2)


def _hw(s, z):
    """Half-width of the upper at (s, z): the footprint low down, an oval shaft higher up (vamp width in
    front, heel-counter width behind), rounded at the front and back of each level and domed over the toe."""
    F, B = _F(z), _B(z)
    lam = _sstep(0.30, 0.50, z)
    sig = _sstep(0.4, 0.8, (s - F) / np.maximum(B - F, 1e-6))
    shaft = _W(z) + (_WC(z) - _W(z)) * sig
    base = 0.5 * ((1 - lam) * _W(z) * _FPN(s) + lam * shaft)
    hw = base * _round(s - F, np.minimum(0.22, 0.03 + 0.8 * F)) * _round(B - s, 0.22)
    rv = 0.13 * (1 - _sstep(0.40, 0.62, s)) + 1e-4
    top = _T(s)
    return np.maximum(hw * np.where(top >= B_RIM - 1e-9, 1.0, _round(top - z, rv)), 0)     # the rim is cut square


def _waist(s):
    """Rise of the sole from the end of the tread to the heel breast."""
    t = np.clip((np.asarray(s, float) - B_TREAD) / (B_BREAST - B_TREAD), 0, 1)
    return B_HEEL * (1 - (1 - t) ** 2.4)


def _scut(z):
    """Where the sole meets level z along the waist (the upper's bottom edge there)."""
    q = np.clip((np.asarray(z, float) - B_SOLE) / B_HEEL, 0, 1)
    return B_TREAD + (B_BREAST - B_TREAD) * (1 - (1 - q) ** (1 / 2.4))


# ---------------------------------------------------------------------------------------------------
#  Fitting the upper for Arch
# ---------------------------------------------------------------------------------------------------
# Each level of the upper is an oval from s = c - a to s = c + a, traced by an angle p:
#   s = c - a cos p,   x = sin p * sum_k b_k cos(k p)
# Low down a level is cut where the sole rises under the arch, so it is drawn for |p| up to an angle that
# grows with height; between the heel seat (z 0.38) and z 0.40 the cut runs round the heel cup and the
# levels close into whole ovals. So the upper is one surface with no seam.
# One parameter u (0..1) runs up the boot, u = 0.5 where the levels close. z, c, a, the cut angle and the
# b_k are cubic splines in u fitted to the drawing; u is spaced so the toe box gets as many levels as the shaft.
B_SEAT = B_SOLE + B_HEEL          # top of the sole at the heel seat
B_CLOSE = 0.39                    # the levels are whole ovals from here up
B_ZS = 0.40                       # where the shaft's levels start rising; all below is inside the sole
B_UC, B_US = 0.40, 0.52           # u at B_SEAT and B_ZS: a wide band of u for the heel cup, so even a coarse
                                  # mesh closes it inside the sole
_KU = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.38, 0.41, 0.44, 0.47, 0.5, 0.53, 0.58, 0.66, 0.74, 0.82, 0.9]
B_K = 7


def _basis(u):
    u = np.asarray(u, float)[..., None]
    return np.concatenate([u ** 0, u, u ** 2, u ** 3] + [np.maximum(0, u - k) ** 3 for k in _KU], -1)


def _spline_fit(u, y, w=None):
    A = _basis(u)
    if w is not None:
        A, y = A * w[:, None], y * w
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _spline_ev(c, u):
    return _basis(u) @ c


def _spline_arch(c, var='u'):
    t = [fmt(c[0]), f"{fmt(c[1])}*{var}", f"{fmt(c[2])}*{var}^2", f"{fmt(c[3])}*{var}^3"]
    t += [f"{fmt(c[4 + i])}*max(0,{var}-{fmt(k)})^3" for i, k in enumerate(_KU)]
    return '+'.join(t)


def _phi(z, c, a):
    """How far round a level reaches: to the waist cut, then closing round the heel cup (past pi)."""
    z = np.asarray(z, float)
    pw = np.arccos(np.clip((c - _scut(np.minimum(z, B_SEAT))) / a, -1, 1))
    return np.where(z <= B_SEAT, pw, pw + (np.pi + 0.3 - pw) * _sstep(B_SEAT, B_CLOSE, z))


def _fit_upper():
    # spacing of the levels below the heel seat: equal steps of the largest movement of their front three
    # quarters between neighbours (the heel cup closes inside the sole, so it gets only a narrow band of u)
    zz = np.linspace(B_SOLE, B_SEAT, 331)
    w = np.linspace(0, 0.75, 31)
    P = []
    for z in zz:
        F, B = float(_F(z)), float(_B(z))
        c, a = (F + B) / 2, (B - F) / 2
        ph = float(_phi(z, c, a)) * w
        s = c - a * np.cos(ph)
        P.append(np.c_[s, _hw(s, z), np.full_like(s, z)])
    P = np.array(P)
    step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1)
    cum = np.r_[0, np.cumsum(step)]
    uf = B_UC * cum / cum[-1]
    u_all = np.r_[uf, np.linspace(B_UC, B_US, 30)[1:], np.linspace(B_US, 1, 200)[1:]]
    z_all = np.r_[zz, np.linspace(B_SEAT, B_ZS, 30)[1:], np.linspace(B_ZS, B_RIM, 200)[1:]]
    wz = np.where(np.isclose(u_all, 0) | np.isclose(u_all, 1), 60.0, 1.0)
    cz = _spline_fit(u_all, z_all, wz)      # exact at the bottom and the rim
    U = np.linspace(0, 1, 241)
    Z = _spline_ev(cz, U)
    F, B = _F(Z), _B(Z)
    cc = _spline_fit(U, (F + B) / 2)
    ca = _spline_fit(U, (B - F) / 2)
    C, A = _spline_ev(cc, U), _spline_ev(ca, U)
    cp = _spline_fit(U, _phi(Z, C, A))
    PHf = np.minimum(np.pi, _spline_ev(cp, U))          # Arch: min(pi, spline)
    # the shape of every level: x = sin p * sum_k b_k(u) cos(k p), fitted jointly over u and p
    rows, tgt, wts = [], [], []
    N = _basis(U)
    for i in range(len(U)):
        p = np.linspace(0, PHf[i], 64)
        s = C[i] - A[i] * np.cos(p)
        hw = _hw(s, Z[i])
        G = np.sin(p)[:, None, None] * np.cos(np.arange(B_K + 1)[None, :, None] * p[:, None, None]) * N[i][None, None, :]
        rows.append(G.reshape(len(p), -1))
        tgt.append(hw)
        wts.append(np.ones(len(p)))
    M, y, wt = np.concatenate(rows), np.concatenate(tgt), np.concatenate(wts)
    M = np.r_[M * wt[:, None], 1e-4 * np.eye(M.shape[1])]
    y = np.r_[y * wt, np.zeros(M.shape[1])]
    beta = np.linalg.lstsq(M, y, rcond=None)[0].reshape(B_K + 1, -1)
    # how far the fitted levels are from the drawing's (largest distance between the outlines)
    err = 0.0
    for i in range(0, len(U), 4):
        p = np.linspace(0, PHf[i], 90)
        bi = np.stack([_spline_ev(beta[k], U[i]) for k in range(B_K + 1)])
        S, Xf = C[i] - A[i] * np.cos(p), np.sin(p) * (bi[None, :] * np.cos(np.arange(B_K + 1)[None, :] * p[:, None])).sum(1)
        sd = np.linspace(F[i], B[i], 700)
        xd = _hw(sd, Z[i])
        err = max(err, np.sqrt((S[:, None] - sd[None]) ** 2 + (Xf[:, None] - xd[None]) ** 2).min(1).max())
    return dict(cz=cz, cc=cc, ca=ca, cp=cp, beta=beta, err=err)


BOOT_FIT = _fit_upper()


def bz_(u): return _spline_ev(BOOT_FIT['cz'], u)
def bc_(u): return _spline_ev(BOOT_FIT['cc'], u)
def ba_(u): return _spline_ev(BOOT_FIT['ca'], u)
def bp_(u): return np.minimum(np.pi, _spline_ev(BOOT_FIT['cp'], u))


def bx_(u, p):
    u, p = np.broadcast_arrays(np.asarray(u, float), np.asarray(p, float))
    b = np.stack([_spline_ev(BOOT_FIT['beta'][k], u) for k in range(B_K + 1)], -1)
    return np.sin(p) * (b * np.cos(np.arange(B_K + 1) * p[..., None])).sum(-1)


def by_(u, p): return bc_(u) - ba_(u) * np.cos(p)


def bu_of_z(z):
    uu = np.linspace(0, 1, 4001)
    return float(np.interp(z, bz_(uu), uu))


def bpoint(u, p):
    """(s, x, z) on the fitted upper."""
    return np.array([by_(u, p), bx_(u, p), bz_(u)], float)


def bnormal(u, p):
    """Outward unit normal of the fitted upper, in (s, x, z)."""
    e = 1e-4
    du = (bpoint(u + e, p) - bpoint(u - e, p)) / (2 * e)
    dp = (bpoint(u, p + e) - bpoint(u, p - e)) / (2 * e)
    n = np.cross(dp, du)                       # outward on the side x > 0
    if np.dot(n, bpoint(u, p) - np.array([bc_(u), 0.0, bz_(u)])) < 0:     # away from the level's centre
        n = -n
    return n / np.linalg.norm(n)


def bp_at_x(u, x):
    """Angle on level u, on the front half, where the upper's half-width reaches |x| (signed like x)."""
    ps = np.linspace(0, np.pi / 2, 2001)
    xs = bx_(np.full_like(ps, u), ps)
    i = np.argmin(np.abs(xs - abs(x)))
    return float(np.sign(x) * ps[i]) if x else 0.0


# ---------------------------------------------------------------------------------------------------
#  Arch
# ---------------------------------------------------------------------------------------------------
def cheb_arch(vals, deg, var='t'):
    """Chebyshev series through vals (sampled on t = -cos(pi k/(n-1))) as an Arch expression in var."""
    n = len(vals)
    t = -np.cos(np.pi * np.arange(n) / (n - 1))
    c = np.polynomial.chebyshev.chebfit(t, vals, deg)
    return '+'.join([fmt(c[0])] + [f"{fmt(c[k])}*cos({k}*acos({var}))" for k in range(1, deg + 1)])


def curve_arch(P, deg=10):
    """A curve through the points P (n x 3, along t from -1 to 1), as three Chebyshev series."""
    P = np.asarray(P, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    tt = -np.cos(np.pi * np.arange(161) / 160)
    q = (tt + 1) / 2 * d[-1]
    Q = np.stack([np.interp(q, d, P[:, k]) for k in range(3)], 1)
    return [cheb_arch(Q[:, k], deg) for k in range(3)]


BOOT_Y0 = -0.10 - float(bc_(1.0))          # the rim's centre stays where the old shaft was (y = -0.10)
BOOT_X = 0.38                               # the front view's boot centres, 0.76 apart
B_UPPER, B_SOLE_C, B_HEEL_C, B_LIFT = '#f0e0d8', '#62483e', '#4e3731', '#2f2420'
B_LACE, B_LINING, B_SEAM, B_LEG = '#80534a', '#a88776', '#c8b0a5', '#f1e8e4'

if want('boots'):
    sc.blank()
    sc.note('Cream lace-up ankle boots with block heels, from the three-view drawing')
    sc.comment(f"Boots. A level of the upper at u (0..1, up the boot) is an oval: s = boc(u)-boa(u)*cos(p) back from\n"
               f"the toe tip, x = sin(p)*sum bok_k(u)*cos(k*p) across, z = boz(u). Fitted to the drawing within {BOOT_FIT['err']:.3f}.")
    for name, c in (('boz', BOOT_FIT['cz']), ('boc', BOOT_FIT['cc']), ('boa', BOOT_FIT['ca'])):
        sc.func(f'{name}(u)', _spline_arch(c))
    sc.func('bop(u)', f"min(pi,{_spline_arch(BOOT_FIT['cp'])})")
    for k in range(B_K + 1):
        sc.func(f'bok{k}(u)', _spline_arch(BOOT_FIT['beta'][k]))
    sc.func('box(u,p)', 'sin(p)*(' + '+'.join(['bok0(u)'] + [f"bok{k}(u)*cos({k}*p)" for k in range(1, B_K + 1)]) + ')')
    sc.func('boy(u,p)', 'boc(u)-boa(u)*cos(p)')
    sc.comment('Sole: underside (toe spring, flat tread, the waist rising to the heel breast) and top, along s.\n'
               'The top is 0.006 above the upper\'s bottom edge, and 0.025 higher round the heel cup, so no gap shows.')
    rise = f"{fmt(B_HEEL)}*(1-(1-min(1,max(0,(q-{fmt(B_TREAD)})/{fmt(B_BREAST - B_TREAD)})))^2.4)"
    sc.func('bozu(q)', f"0.05*max(0,1-q/0.1)^2+{rise}")
    # the top follows the upper's bottom edge as fitted (0.006 above it), and covers the heel cup
    ue = np.linspace(0, 0.6, 1500)
    ue = ue[bp_(ue) < np.pi - 1e-6]
    e_s, e_z = by_(ue, bp_(ue)), bz_(ue)
    o = np.argsort(e_s)
    sq = np.linspace(0.0, 1.47, 400)
    top = np.interp(sq, e_s[o], np.maximum.accumulate(e_z[o]))
    top = np.where(sq < 0.6, B_SOLE + 0.001, top + 0.006)     # level with the upper's edge on the forefoot
    top = np.maximum(top, (B_ZS + 0.01) * _sstep(1.02, 1.12, sq))
    _KS = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    Bs = np.concatenate([sq[:, None] ** 0, sq[:, None]] + [np.maximum(0, sq[:, None] - k) ** 3 for k in _KS], 1)
    cs = np.linalg.lstsq(Bs, top, rcond=None)[0]
    fit_top = Bs @ cs
    lack = max(0.0, float((top - fit_top)[(sq > 0.6) & (sq < 1.42)].max()))     # lift the arch part so it is never
    BOOT_SOLE_TOP = (fit_top + lack * np.clip((sq - 0.6) / 0.1, 0, 1), sq)      # below the edge; the forefoot stays level
    sc.func('bozt(q)', '+'.join([fmt(cs[0]), f"{fmt(cs[1])}*q"] + [f"{fmt(cs[2 + i])}*max(0,q-{fmt(k)})^3" for i, k in enumerate(_KS)]
                                + [f"{fmt(lack)}*min(1,max(0,(q-0.6)/0.1))"]))

    # sole outline: the upper's bottom edge plus a 0.012 welt, as sin(w) * cosine series along the sole
    uu = np.linspace(0, B_US, 800)
    uu = uu[bp_(uu) < np.pi - 1e-6]
    cut_s, cut_x = by_(uu, bp_(uu)), bx_(uu, bp_(uu))     # the upper's bottom edge along the waist and heel cup
    pf = np.linspace(0, bp_(0.0), 300)
    toe_s, toe_x = by_(0.0, pf), bx_(0.0, pf)            # ... and round the forefoot
    es, ex = np.r_[toe_s, cut_s], np.r_[toe_x, cut_x]
    S0, S1 = 0.02, float(es.max()) + 0.012
    ss = np.linspace(S0, S1, 600)
    o = np.argsort(es)
    edge = np.interp(ss, es[o], ex[o], left=0, right=0)
    sole_hw = np.where((ss > es.min()) & (ss < es.max()), edge + 0.012, 0)
    ww = np.arccos(1 - 2 * (ss - S0) / (S1 - S0))
    good = (ww > 0.02) & (ww < np.pi - 0.02)
    Ms = np.sin(ww[good])[:, None] * np.cos(np.arange(9)[None, :] * ww[good][:, None])
    qs = np.linalg.lstsq(Ms, sole_hw[good], rcond=None)[0]
    sole_w = 'sin(w)*(' + '+'.join([fmt(qs[0])] + [f"{fmt(qs[k])}*cos({k}*w)" for k in range(1, 9)]) + ')'
    sc.func('bosw(w)', sole_w)
    sole_s = f"{fmt(S0)}+{fmt(S1 - S0)}*(1-cos(w))/2"

    # front of the upper over the lacing: s = g0(z) + g2(z) x^2 + g4(z) x^4, for the laces and the bow
    zl = np.linspace(B_EYE_Z[0] - 0.04, B_RIM, 40)
    xl = np.linspace(-0.2, 0.2, 21)
    R, T = [], []
    for z in zl:
        u = bu_of_z(z)
        for x in xl:
            p = bp_at_x(u, x)
            s = float(by_(u, abs(p)))
            R.append([z ** i * x ** j for j in (0, 2, 4) for i in range(4)])
            T.append(s)
    gq = np.linalg.lstsq(np.array(R), np.array(T), rcond=None)[0]

    def g_arch(xe, ze, lift):
        terms = []
        for jj, j in enumerate((0, 2, 4)):
            poly = '+'.join(f"{fmt(gq[jj * 4 + i])}*({ze})^{i}" if i else fmt(gq[jj * 4]) for i in range(4))
            terms.append(f"({poly})" + (f"*({xe})^{j}" if j else ''))
        return f"{fmt(BOOT_Y0 - lift)}+" + '+'.join(terms)

    def g_py(x, z):
        return sum(gq[jj * 4 + i] * z ** i * x ** j for jj, j in enumerate((0, 2, 4)) for i in range(4))

    # the lining: inside the shaft from z 0.7 up, and the padded collar round the rim
    u_lin = bu_of_z(0.70)
    rim_b = [float(_spline_ev(BOOT_FIT['beta'][k], 1.0)) for k in range(B_K + 1)]
    rim_x = 'sin(v)*(' + '+'.join([fmt(rim_b[0])] + [f"{fmt(rim_b[k])}*cos({k}*v)" for k in range(1, B_K + 1)]) + ')'
    rc, ra = float(bc_(1.0)), float(ba_(1.0))
    rhw = float(bx_(1.0, np.linspace(0, np.pi, 400)).max())

    # toe-cap seam: the upper cut by a plane that leans forward as it goes down the sides
    ztop = float(_T(B_CAP[0]))
    cap = []
    for z in np.linspace(B_SOLE + 0.006, ztop - 0.004, 60):
        u = bu_of_z(z)
        s_target = B_CAP[0] - B_CAP[1] * (ztop - z)
        ps = np.linspace(0, float(bp_(u)), 3000)
        sv = by_(np.full_like(ps, u), ps)
        if sv[0] > s_target:
            continue
        p = float(ps[np.argmin(np.abs(sv - s_target))])
        cap.append((u, p))
    capP = [bpoint(u, p) + 0.004 * bnormal(u, p) for u, p in cap]
    capP = np.array([q * [1, -1, 1] for q in capP] + capP[::-1])
    # facing stitches round the lacing, and the back seam
    fac = []
    for z in np.linspace(B_RIM - 0.035, B_EYE_Z[0] - 0.01, 30):
        fac.append((-B_FACING, z))
    for t in np.linspace(np.pi, 2 * np.pi, 25)[1:-1]:
        fac.append((B_FACING * np.cos(t), B_EYE_Z[0] - 0.01 + 0.06 * np.sin(t)))
    for z in np.linspace(B_EYE_Z[0] - 0.01, B_RIM - 0.035, 30):
        fac.append((B_FACING, z))
    facP = []
    for x, z in fac:
        u = bu_of_z(z)
        p = bp_at_x(u, x)
        facP.append(bpoint(u, p) + 0.003 * bnormal(u, p))
    facP = np.array(facP)

    for side, tag in ((1, 'L'), (-1, 'R')):
        xc = BOOT_X * side
        X0, Y0 = fmt(xc), fmt(BOOT_Y0)
        # upper: one surface from the sole to the rim
        sc.surface((f"{X0}+box(u,(v*bop(u)))", f"{Y0}+boy(u,(v*bop(u)))", "boz(u)"), ('u', 0, 1), ('v', -1, 1),
                   B_UPPER, f'bootUpper{tag}', 'closed')
        sc.surface((f"{X0}+0.86*box(u,v)", f"{Y0}+boc(u)-(boa(u)-0.035)*cos(v)", "boz(u)-0.002"), ('u', u_lin, 1),
                   ('v', '-pi', 'pi'), B_LINING, f'bootLining{tag}', ('tube', 'u', 'in'))
        r = 0.016
        sc.surface((f"{X0}+({rim_x})*(1+{fmt(r / rhw)}*cos(u))", f"{Y0}+{fmt(rc)}-({fmt(ra)}+{fmt(r)}*cos(u))*cos(v)",
                    f"{fmt(B_RIM - r + 0.002)}+{fmt(r)}*sin(u)"), ('u', 0, '2*pi'), ('v', '-pi', 'pi'),
                   B_UPPER, f'bootCollar{tag}', ('tube', 'v'))
        # sole, heel and top lift
        sc.surface((f"{X0}+bosw(w)*sqc(v,0.3)", f"{Y0}+{sole_s}",
                    f"(bozu(({sole_s}))+bozt(({sole_s})))/2+(bozt(({sole_s}))-bozu(({sole_s})))/2*sqs(v,0.3)"),
                   ('w', 0, 'pi'), ('v', 0, '2*pi'), B_SOLE_C, f'bootSole{tag}', ('tube', 'u'))
        # heel: tapered block, 0.40 x 0.34 under the heel seat, 0.32 x 0.22 on the ground; front face upright
        sc.surface((f"{X0}+(0.18+0.02*sqc(u,0.1))*sqs(u,0.1)*sqs(v,0.35)",
                    f"{Y0}+1.2+0.03*sqc(u,0.1)-(0.14+0.03*sqc(u,0.1))*sqs(u,0.1)*sqc(v,0.35)",
                    "0.1675+0.1675*sqc(u,0.1)"), ('u', 0, 'pi'), ('v', 0, '2*pi'), B_HEEL_C, f'bootHeel{tag}', 'closed')
        sc.surface((f"{X0}+0.164*sqs(u,0.15)*sqs(v,0.35)", f"{Y0}+1.17-0.114*sqs(u,0.15)*sqc(v,0.35)",
                    "0.015+0.0155*sqc(u,0.15)"), ('u', 0, 'pi'), ('v', 0, '2*pi'), B_LIFT, f'bootLift{tag}', 'closed')
        if want('legs'):        # her leg in a pale stocking, from inside the shaft up under the skirt
            capsule((xc, -0.10, 0.80), (xc, -0.06, 2.3), 0.17, 0.21, B_LEG, f'leg{tag}')
        # eyelets: gold rings on both sides of the lacing
        for k, z in enumerate(B_EYE_Z):
            u = bu_of_z(z)
            for sx, st in ((-1, 'a'), (1, 'b')):
                p = bp_at_x(u, sx * B_EYE_X)
                Pp = bpoint(u, p)
                n = bnormal(u, p)
                C = (xc + Pp[1] + 0.003 * n[1], BOOT_Y0 + Pp[0] + 0.003 * n[0], Pp[2] + 0.003 * n[2])
                nm = (n[1], n[0], n[2])
                ellipsoid(C, (0.019, 0.019, 0.006), GOLD, f'bootEyelet{tag}{k + 1}{st}', fr=frame(nm))
        # laces: two zigzags crossing between the pairs, a bar across the lowest pair, a bow at the top
        pitch = B_EYE_Z[1] - B_EYE_Z[0]
        for k, sgn in enumerate((-1, 1)):
            xe = f"{fmt(sgn * B_EYE_X)}*asin(cos(pi*t))*2/pi"
            ze = f"{fmt(B_EYE_Z[0])}+{fmt(pitch)}*t"
            sc.curve((f"{X0}+{xe}", g_arch(xe, ze, 0.013), ze), ('t', 0, len(B_EYE_Z) - 1), B_LACE, f'bootLace{tag}{k + 1}', thickness=3)
        xe, ze = f"{fmt(B_EYE_X)}*t", fmt(B_EYE_Z[0])
        sc.curve((f"{X0}+{xe}", g_arch(xe, ze, 0.013), ze), ('t', -1, 1), B_LACE, f'bootLaceBar{tag}', thickness=3)
        zk = B_EYE_Z[-1] + 0.02
        ellipsoid((xc, BOOT_Y0 + g_py(0, zk) - 0.022, zk), (0.024, 0.013, 0.017), B_LACE, f'bootKnot{tag}')
        for j, sx in enumerate((-1, 1)):
            th = math.radians(12 * sx)
            xe = f"({fmt(sx * 0.08)}+0.08*cos(t)*{fmt(math.cos(th))}-0.028*sin(t)*{fmt(math.sin(th))})"
            ze = f"({fmt(zk + 0.012)}+0.08*cos(t)*{fmt(math.sin(th))}+0.028*sin(t)*{fmt(math.cos(th))})"
            sc.curve((f"{X0}+{xe}", g_arch(xe, ze, 0.025), ze), ('t', 0, '2*pi'), B_LACE, f'bootBowLoop{tag}{j + 1}', thickness=3)
        for j, (sx, x1, z1) in enumerate(((-1, -0.07, 0.86), (1, 0.05, 0.83))):
            xe = f"({fmt(sx * 0.01)}+{fmt(x1 - sx * 0.01)}*t+{fmt(0.02 * sx)}*sin(pi*t))"
            ze = f"({fmt(zk)}+{fmt(z1 - zk)}*t)"
            sc.curve((f"{X0}+{xe}", g_arch(xe, ze, 0.03), ze), ('t', 0, 1), B_LACE, f'bootBowTail{tag}{j + 1}', thickness=3)
            ellipsoid((xc + x1, BOOT_Y0 + g_py(x1, z1) - 0.03, z1 - 0.012), (0.009, 0.009, 0.022), GOLD, f'bootAglet{tag}{j + 1}')
        # seams
        for nm_, P in (('Cap', capP), ('Facing', facP)):
            Xs, Ys, Zs = curve_arch(P, deg=14)
            sc.curve((f"{X0}+{Ys}", f"{Y0}+{Xs}", Zs), ('t', -1, 1), B_SEAM, f'boot{nm_}Seam{tag}', thickness=1)
        sc.curve((X0, f"{fmt(BOOT_Y0 + 0.003)}+boc(u)+boa(u)", "boz(u)"), ('u', bu_of_z(0.40), 1), B_SEAM, f'bootBackSeam{tag}', thickness=1)
