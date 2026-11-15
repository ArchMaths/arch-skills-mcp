# Capelet, measured on outlines traced over the sheet's front, side and back views and its detail panel
# (run inside generate.py).
import numpy as np

# A short bell over the shoulders, open at the front in a wide V: each front edge runs from the collar down
# across the bust and round to an outer corner at the side, with a turned-back facing along its upper part.
# It stands out at the back; its hem is lowest at the centre back, with a lace frill there; double gold
# lines follow the front edges and the hem; soft folds fan out from the shoulders; a standing collar, open at
# the front; four embroidered chrysanthemums, two on the front panels and two at the back.
#
# Measured on the user's tracing over the sheet's three views (calibrated as in sheetfit.py). The side view
# is a true profile for the capelet (azimuth 0): its front edge reaches the bust where the front view's front
# edge, on an oval section hugging the chest, puts it. Its lower front is cut higher than the front view
# shows, and its emblem (y 0.8) fits no one place on the capelet; the model follows the front and back views
# there. The detail panel (披肩) gives the collar, the lining, the trim, the folds and the emblem's colour.

# ---------------------------------------------------------------------------------------------------
#  The drawing
# ---------------------------------------------------------------------------------------------------
CP_Y0 = 0.02                 # the body's centre line (y)
CP_ZN, CP_H0 = 10.62, 9.20   # neckline, and the hem's mean height
# outline half-width a(z): the front and back views averaged (their silhouettes must agree)
CP_AZ = [10.62, 10.55, 10.45, 10.33, 10.21, 10.09, 9.97, 9.85, 9.73, 9.65, 9.57, 9.45, 9.33, 9.25, 9.17]
CP_A = [0.31, 0.47, 0.78, 0.965, 1.035, 1.105, 1.15, 1.25, 1.33, 1.41, 1.47, 1.50, 1.54, 1.58, 1.58]
# depth behind the centre line at the back, from the side view's back edge (near the neck, under the hair,
# it closes in to the neckline)
CP_BZ = [10.62, 10.52, 10.40, 10.28, 10.16, 10.04, 9.92, 9.80, 9.68, 9.56, 9.44, 9.32, 9.20]
CP_BB = [0.30, 0.60, 0.92, 1.01, 1.07, 1.11, 1.12, 1.14, 1.15, 1.16, 1.18, 1.21, 1.25]
# depth in front of the centre line: over the chest (the body plus a little), 0.47 at the bust as the side
# view's front edge shows
CP_FZ = [10.62, 10.45, 10.30, 10.10, 9.90, 9.70, 9.50, 9.20]
CP_BF = [0.29, 0.33, 0.38, 0.43, 0.47, 0.49, 0.50, 0.50]
# front edge on the front view: |x| at each height, from the collar to the outer corner (1.44, 9.12)
FE_X = [0.14, 0.15, 0.16, 0.19, 0.25, 0.30, 0.345, 0.39, 0.43, 0.46, 0.49, 0.53, 0.565, 0.605, 0.655, 0.70, 0.765, 0.81, 0.87, 0.90,
        0.95, 1.0, 1.11, 1.21, 1.32, 1.36, 1.40, 1.43, 1.44]
FE_Z = [10.55, 10.42, 10.30, 10.17, 10.07, 10.02, 9.97, 9.93, 9.89, 9.85, 9.81, 9.77, 9.73, 9.69, 9.65, 9.61, 9.57, 9.53, 9.49, 9.45,
        9.41, 9.37, 9.33, 9.29, 9.25, 9.21, 9.17, 9.13, 9.12]
# hem height round from the front (degrees): the front view's corner, the back view's hem (|x| -> angle)
HEM_PHI = [65, 90, 105, 130, 141, 160, 180]
HEM_Z = [9.12, 9.22, 9.26, 9.28, 9.22, 9.14, 9.09]
# lace below the hem at the back: 0.25 deep at the centre back, 0.15 at the back corners, from 100 degrees
LACE_PHI0, LACE_N = 100, 44
# facing: the fold line of the turned-back front, from the shoulder corner to the front edge (front view)
FACING = [(0.92, 10.31), (0.87, 10.25), (0.83, 10.19), (0.79, 10.13), (0.75, 10.07), (0.72, 10.01), (0.69, 9.95), (0.67, 9.89),
          (0.64, 9.85), (0.585, 9.80), (0.53, 9.77)]
# emblems: the circled places; the front pair on the front view, the back pair on the back view (|x|, z)
EMB_FRONT, EMB_BACK, EMB_R = (1.10, 9.67), (1.23, 9.66), 0.17
# folds fanning out from the shoulders: how many across the cloth (front edge to front edge, so they fan with the
# front edges), how deep at the hem, how far the hem rises on each crest
FOLD_N, FOLD_A, FOLD_Z = 16, 0.045, 0.0
FOLD_LINE = '#b8ac9a'        # the drawn fold lines: a soft grey-beige
CP_P = 1.6                   # u spacing: the shoulder slope (wide, little drop) gets more of u
CP_COLLAR = (10.58, 0.26, 0.305, 24)   # standing collar: bottom, height, radius, half-gap at the front (degrees)
CP_LINING, CP_LACE, CP_EMB = '#decdb8', '#fbf6ee', '#d78a5e'


# ---------------------------------------------------------------------------------------------------
#  Fit
# ---------------------------------------------------------------------------------------------------
def _zb(u):
    """Height of level u before the hem's variation."""
    return CP_ZN - (CP_ZN - CP_H0) * np.asarray(u, float) ** CP_P


def _u_of_z(z):
    return np.clip((CP_ZN - np.asarray(z, float)) / (CP_ZN - CP_H0), 0, None) ** (1 / CP_P)


_KS = [0.15, 0.3, 0.45, 0.6, 0.75]                        # spline knots in u: the sections
_KE = [0.04, 0.08, 0.12, 0.17, 0.24, 0.33, 0.45, 0.6, 0.75, 0.9]   # the front edge, which turns fast near the collar


def _ub(u, K):
    u = np.atleast_1d(np.asarray(u, float))
    return np.stack([np.ones_like(u), u, u ** 2, u ** 3] + [np.maximum(0, u - k) ** 3 for k in K], -1)


def _ufit(z, v, K=_KS):
    """Least-squares cubic spline in u through dense samples of a measured table v(z)."""
    o = np.argsort(z)
    zz = np.linspace(min(z), max(z), 300)
    vv = np.interp(zz, np.asarray(z, float)[o], np.asarray(v, float)[o])
    c, *_ = np.linalg.lstsq(_ub(_u_of_z(zz), K), vv, rcond=None)
    return c, K


def _uev(fit, u):
    c, K = fit
    r = _ub(u, K) @ c
    return r if np.ndim(u) else float(r[0])


def _uarch(fit, var='u'):
    c, K = fit
    t = [fmt(c[0]), f"{fmt(c[1])}*{var}", f"{fmt(c[2])}*{var}^2", f"{fmt(c[3])}*{var}^3"]
    t += [f"{fmt(ck)}*max(0,{var}-{fmt(k)})^3" for ck, k in zip(c[4:], K)]
    return '+'.join(t).replace('+-', '-')


CP_FIT = {'kpa': _ufit(CP_AZ, CP_A), 'kpb': _ufit(CP_BZ, CP_BB), 'kpf': _ufit(CP_FZ, CP_BF)}
CP_ERR = {k: float(np.max(np.abs(_uev(c, _u_of_z(z)) - np.asarray(v)))) for (k, c), (z, v) in
          zip(CP_FIT.items(), ((CP_AZ, CP_A), (CP_BZ, CP_BB), (CP_FZ, CP_BF)))}
# hem: offset below CP_H0 as a polynomial in cos(phi)
HEM_C = np.polyfit(np.cos(np.radians(HEM_PHI)), CP_H0 - np.array(HEM_Z), 4)
HEM_ERR = float(np.max(np.abs(np.polyval(HEM_C, np.cos(np.radians(HEM_PHI))) - (CP_H0 - np.array(HEM_Z)))))


def kp_a(u): return _uev(CP_FIT['kpa'], u)
def kp_b(u): return _uev(CP_FIT['kpb'], u)
def kp_f(u): return _uev(CP_FIT['kpf'], u)
def kp_h(p): return np.polyval(HEM_C, np.cos(p))


def kp_point(u, p, fold=True, s=1.0, dz=0.0):
    """Point on the capelet at level u (0 neckline, 1 hem) and angle p from the front (toward her left);
    s scales the section (the lining), dz lowers it."""
    u, p = np.asarray(u, float), np.asarray(p, float)
    w = kp_w(u, p) if fold else 0.0
    r = 1 + FOLD_A * u ** 2 * w
    B = kp_f(u) + (kp_b(u) - kp_f(u)) * (1 - np.cos(p)) / 2
    return (s * kp_a(u) * np.sin(p) * r, CP_Y0 - s * B * np.cos(p) * r, _zb(u) - u ** 4 * kp_h(p) + FOLD_Z * u ** 8 * w - dz)


def kp_w(u, p):
    """The folds: 1 on a crest, -1 in a hollow; FOLD_N of them across the cloth from front edge to front edge."""
    e = kp_e(u)
    return np.cos(2 * np.pi * FOLD_N * (p - e) / (2 * np.pi - 2 * e))


def fold_lines():
    """The drawn fold lines: (q across the cloth, level u where each starts). With many folds a line runs down
    each hollow; with a few deep ones a line runs down each hollow and a shorter one down each crest between."""
    h = lambda k: (np.sin(2.39 * k + 0.7) + 1) / 2          # a fixed scatter, so the lines start at varied heights
    if FOLD_N >= 12:
        return [((j + 0.5) / FOLD_N, 0.12 + 0.3 * h(j)) for j in range(FOLD_N)]
    return [((k + 1) / (2 * FOLD_N), (0.12 + 0.3 * h(k)) if k % 2 == 0 else (0.5 + 0.2 * h(k))) for k in range(2 * FOLD_N - 1)]


def _front_edge():
    """Angle of the front edge at each level: where the section meets the front view's front edge."""
    fz, fx = np.array(FE_Z)[::-1], np.array(FE_X)[::-1]
    us, ps = np.linspace(0, 1, 101), []
    for u in us:
        p = np.linspace(0.02, 1.6, 4000)
        x, _, z = kp_point(u, p, fold=False)
        d = x - np.interp(z, fz, fx)
        k = np.nonzero(np.diff(np.sign(d)))[0]
        ps.append(p[k[0]] if len(k) else np.nan)
    ps = np.array(ps)
    ok = np.isfinite(ps)
    c, *_ = np.linalg.lstsq(_ub(us[ok], _KE), ps[ok], rcond=None)
    return (c, _KE), float(np.max(np.abs(_ub(us[ok], _KE) @ c - ps[ok])))


FE_C, FE_ERR = _front_edge()


def kp_e(u): return _uev(FE_C, u)


def kp_q(u, q):
    """Angle of the point q (0..1) round the capelet from her left front edge, over the back, to her right one."""
    e = kp_e(u)
    return e + np.asarray(q, float) * (2 * np.pi - 2 * e)


def _normal(u, p, h=1e-4):
    P0 = np.array(kp_point(u, p))
    du = np.array(kp_point(u + h, p)) - P0
    dp = np.array(kp_point(u, p + h)) - P0
    n = np.cross(dp, du)
    n = n / np.linalg.norm(n)
    if n[0] * np.sin(p) + (-np.cos(p)) * n[1] < 0:          # outward
        n = -n
    return tuple(P0), tuple(n), tuple(-du / np.linalg.norm(du))


def _place(x_abs, z, back, side):
    """Level and angle where the front (or back) view sees the point (|x|, z) on the capelet."""
    best = None
    for u in np.linspace(0.3, 1, 701):
        p0 = np.arcsin(np.clip(x_abs / kp_a(u), -1, 1))
        p = np.pi - p0 if back else p0
        z_ = kp_point(u, p, fold=False)[2]
        if best is None or abs(z_ - z) < best[0]:
            best = (abs(z_ - z), u, p)
    u, p = best[1], best[2]
    return u, (p if side > 0 else 2 * np.pi - p)


EMBLEMS = {}
for _side, _t in ((1, 'L'), (-1, 'R')):
    for _back, _w in ((False, 'F'), (True, 'B')):
        _u, _p = _place(*(EMB_BACK if _back else EMB_FRONT), _back, _side)
        EMBLEMS[_w + _t] = _normal(_u, _p)

# facing line on the surface: (u, p) of each traced point
_fac = []
for _x, _z in FACING:
    _u = float(np.interp(-_z, -_zb(np.linspace(0, 1, 400)), np.linspace(0, 1, 400)))
    _fac.append((_u, float(np.arcsin(min(1.0, _x / kp_a(_u))))))
FAC_T = np.linspace(-1, 1, len(_fac))
FAC_CU = np.polyfit(FAC_T, [f[0] for f in _fac], 3)
FAC_CP = np.polyfit(FAC_T, [f[1] for f in _fac], 3)


# the capelet's reach for the hair: largest distance from the axis (0, 0.03) by angle from the back and height
def _env_table():
    pts = []
    for u in np.linspace(0, 1, 60):
        q = np.linspace(0, 1, 240)
        pts.append(np.stack(kp_point(u, kp_q(u, q)), 1))
    for v in np.linspace(0, 1, 6):
        pts.append(np.stack(lace_point(np.linspace(0, 1, 160), v), 1))
    P = np.concatenate(pts)
    psi = np.abs(np.arctan2(P[:, 0], P[:, 1] - 0.03))
    r = np.hypot(P[:, 0], P[:, 1] - 0.03)
    zs, ps = np.arange(8.7, 10.9, 0.04), np.linspace(0, np.pi, 61)
    T = np.zeros((len(ps), len(zs)))
    iz = np.clip(np.round((P[:, 2] - zs[0]) / 0.04).astype(int), 0, len(zs) - 1)
    ip = np.clip(np.round(psi / (np.pi / 60)).astype(int), 0, 60)
    np.maximum.at(T, (ip, iz), r)
    T = np.maximum(T, np.roll(T, 1, 1))
    T = np.maximum(T, np.roll(T, -1, 1))
    return zs, ps, T


def lace_point(q, v):
    """Lace frill below the hem at the back: q round from 100 degrees on her left to 100 on her right, v down."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    p0 = np.radians(LACE_PHI0)
    p = p0 + q * (2 * np.pi - 2 * p0)
    x, y, z = kp_point(1.0, p)
    depth = (0.15 + 0.1 * np.maximum(0, -np.cos(p)) ** 6) * np.minimum(1, np.sin(np.pi * q) * 3)
    k = 1 + 0.04 * v
    return x * k, CP_Y0 + (y - CP_Y0) * k, z - v * depth * (1 - 0.12 * (1 + np.cos(LACE_N * p)) / 2)


CAPE_ENV = _env_table()


def cape_env(psi, z):
    zs, ps, T = CAPE_ENV
    if z < zs[0] or z > zs[-1]:
        return 0.0
    i = int(round(abs(psi) / (np.pi / 60)))
    j = int(round((z - zs[0]) / 0.04))
    return float(T[min(i, 60), min(j, len(zs) - 1)])


# ---------------------------------------------------------------------------------------------------
#  Arch
# ---------------------------------------------------------------------------------------------------
def _emit_cape():
    sc.blank()
    sc.note('Capelet: open in a V at the front, gold-trimmed, lace at the back, standing collar, chrysanthemum emblems')
    sc.comment(f"""Capelet sections at each level u (0 at the neckline, 1 at the hem; measured on the sheet's front, side
and back views): half-width kpa, depth kpf in front of the centre line and kpb behind it; kph the hem below
{fmt(CP_H0)} by angle; kpe the front edge's angle; kpp the angle of point q round from her left front edge.""")
    for k, c in CP_FIT.items():
        sc.func(f'{k}(u)', _uarch(c))
    sc.func('kph(c)', '+'.join(f"{fmt(v)}*c^{len(HEM_C) - 1 - i}" for i, v in enumerate(HEM_C)).replace('+-', '-').replace('*c^0', ''))
    sc.func('kpe(u)', _uarch(FE_C))
    sc.func('kpp(u,q)', 'kpe(u)+q*(2*pi-2*kpe(u))')
    sc.func('kpw(u,p)', f"cos({fmt(2 * np.pi * FOLD_N)}*(p-kpe(u))/(2*pi-2*kpe(u)))")
    sc.func('kpr(u,p)', f"1+{fmt(FOLD_A)}*u^2*kpw(u,p)")
    sc.func('kpx(u,p,s)', 's*kpa(u)*sin(p)*kpr(u,p)')
    sc.func('kpy(u,p,s)', f"{fmt(CP_Y0)}-s*(kpf(u)+(kpb(u)-kpf(u))*(1-cos(p))/2)*cos(p)*kpr(u,p)")
    sc.func('kpz(u,p)', f"{fmt(CP_ZN)}-{fmt(CP_ZN - CP_H0)}*u^{fmt(CP_P)}-u^4*kph(cos(p))" + (f"+{fmt(FOLD_Z)}*u^8*kpw(u,p)" if FOLD_Z else ""))

    def S(X, Y, Z, U, V, col, ref, orient):
        sc.surface((X, Y, Z), U, V, col, ref, orient)

    P = "(kpp(u,q))"
    S(f"kpx(u,{P},1)", f"kpy(u,{P},1)", f"kpz(u,{P})", ('u', 0, 1), ('q', 0, 1), CREAM, 'cape', ('tube', 'u'))
    S(f"kpx(u,{P},0.988)", f"kpy(u,{P},0.988)", f"kpz(u,{P})-0.004", ('u', 0.02, 1), ('q', 0, 1), CP_LINING, 'capeLining',
      ('tube', 'u', 'in'))
    # lace frill below the hem at the back, flaring a little, scalloped
    p0 = np.radians(LACE_PHI0)
    Pl = f"({fmt(p0)}+q*{fmt(2 * np.pi - 2 * p0)})"
    dep = f"(0.15+0.1*max(0,-cos({Pl}))^6)*min(1,3*sin(pi*q))"
    sc.surface((f"kpx(1,{Pl},1+0.04*v)", f"kpy(1,{Pl},1+0.04*v)", f"kpz(1,{Pl})-v*{dep}*(1-0.06*(1+cos({LACE_N}*{Pl})))"),
               ('q', 0, 1), ('v', 0, 1), CP_LACE, 'capeLace', ('point', (0, CP_Y0, 9.6)))
    # double gold lines along the front edges and the hem, just off the surface
    for j, d in enumerate((0.035, 0.075)):
        for side, tag in ((1, 'L'), (-1, 'R')):
            # an inset of d along the section: the angle step d over the section's local radius at the edge
            rad = f"sqrt((kpa(u)*cos(kpe(u)))^2+(kpf(u)*sin(kpe(u)))^2+0.01)"
            pe = f"(kpe(u)+{fmt(d)}/{rad})" if side > 0 else f"(2*pi-kpe(u)-{fmt(d)}/{rad})"
            sc.curve((f"kpx(u,{pe},1.006)", f"kpy(u,{pe},1.006)", f"kpz(u,{pe})"), ('u', 0.03, 1 - d / 2.2), GOLD,
                     f'capeEdgeTrim{tag}{j + 1}', thickness=2)
        uh = fmt(1 - d / 2.25)
        sc.curve((f"kpx({uh},(kpp({uh},q)),1.006)", f"kpy({uh},(kpp({uh},q)),1.006)", f"kpz({uh},(kpp({uh},q)))"),
                 ('q', 0, 1), GOLD, f'capeHemTrim{j + 1}', thickness=2)
    # the facing's fold line, from the shoulder corner to the front edge
    cu = '+'.join(f"{fmt(v)}*t^{len(FAC_CU) - 1 - i}" for i, v in enumerate(FAC_CU)).replace('+-', '-').replace('*t^0', '')
    cp = '+'.join(f"{fmt(v)}*t^{len(FAC_CP) - 1 - i}" for i, v in enumerate(FAC_CP)).replace('+-', '-').replace('*t^0', '')
    for side, tag in ((1, 'L'), (-1, 'R')):
        pf = f"({cp})" if side > 0 else f"(2*pi-({cp}))"
        sc.curve((f"kpx(({cu}),{pf},1.006)", f"kpy(({cu}),{pf},1.006)", f"kpz(({cu}),{pf})"), ('t', -1, 1), GOLD,
                 f'capeFacingLine{tag}', thickness=2)
        # a frill hangs from the fold line (close-up of the chest): 0.12 deep, ruffled
        sf = "(1.012+0.03*v*(1+0.8*sin(40*t)))"
        sc.surface((f"kpx(({cu}),{pf},{sf})", f"kpy(({cu}),{pf},{sf})", f"kpz(({cu}),{pf})-0.12*v*(1+0.12*cos(40*t))"), ('t', -1, 1), ('v', 0, 1), CP_LACE,
                   f'capeFacingFrill{tag}', ('dir', (0, -1, 0.3)))
    # Drawn fold lines (soft grey-beige), all in one stroke: up each line from the hem to where it starts and back
    # down, then along the hem to the next line. kfa(k) is where line k starts.
    FL = fold_lines()
    q0, dq = FL[0][0], FL[1][0] - FL[0][0]
    sc.func('kfa(k)', '+'.join(f"{fmt(a)}*max(0,1-abs(k-{i}))" for i, (_, a) in enumerate(FL)))
    sc.func('kfu(t)', "1-(1-kfa(floor(t)))*max(0,1-abs(t-floor(t)-0.4)/0.4)")
    sc.func('kfq(t)', f"{fmt(q0)}+{fmt(dq)}*(floor(t)+max(0,(t-floor(t)-0.8)/0.2))")
    sc.curve(("kpx(kfu(t),kpp(kfu(t),kfq(t)),1.006)", "kpy(kfu(t),kpp(kfu(t),kfq(t)),1.006)", "kpz(kfu(t),kpp(kfu(t),kfq(t)))"),
             ('t', 0, len(FL) - 0.2), FOLD_LINE, 'capeFoldLines', thickness=1)
    # Gathers where the front meets the collar: four short lines fanning down and out from the collar's corner, in one
    # stroke out along each and back.
    G = ((0.2, 0.022), (0.27, 0.038), (0.31, 0.055), (0.25, 0.072))
    sc.func('kgu(t)', "0.035+(" + '+'.join(f"{fmt(u_ - 0.035)}*max(0,1-abs(floor(t)-{i}))" for i, (u_, _) in enumerate(G)) + ")*max(0,1-abs(t-floor(t)-0.5)/0.5)")
    sc.func('kgq(t)', "0.008+(" + '+'.join(f"{fmt(q_ - 0.008)}*max(0,1-abs(floor(t)-{i}))" for i, (_, q_) in enumerate(G)) + ")*max(0,1-abs(t-floor(t)-0.5)/0.5)")
    for side, tag in ((1, 'L'), (-1, 'R')):
        Q = "kgq(t)" if side > 0 else "(1-kgq(t))"
        sc.curve((f"kpx(kgu(t),kpp(kgu(t),{Q}),1.008)", f"kpy(kgu(t),kpp(kgu(t),{Q}),1.008)", f"kpz(kgu(t),kpp(kgu(t),{Q}))"),
                 ('t', 0, len(G)), FOLD_LINE, f'capeGathers{tag}', thickness=1)
    # standing collar, open at the front, gold along its top and front edges
    z0, hgt, rc, gap = CP_COLLAR
    g = np.radians(gap)
    pc = f"({fmt(g)}+q*{fmt(2 * np.pi - 2 * g)})"
    rr = f"({fmt(rc)}+0.02*v)"
    sc.surface((f"{rr}*sin({pc})", f"{fmt(CP_Y0)}-0.95*{rr}*cos({pc})", f"{fmt(z0)}+{fmt(hgt)}*v"), ('q', 0, 1), ('v', 0, 1), CREAM,
               'capeCollar', ('point', (0, CP_Y0, z0 + hgt / 2)))
    sc.curve((f"{fmt(rc + 0.025)}*sin({pc.replace('q', 't')})", f"{fmt(CP_Y0)}-0.95*{fmt(rc + 0.025)}*cos({pc.replace('q', 't')})",
              fmt(z0 + hgt - 0.015)), ('t', 0, 1), GOLD, 'capeCollarTrim', thickness=3)
    for side, tag in ((1, 'L'), (-1, 'R')):
        pg = fmt(g + 0.06) if side > 0 else fmt(2 * np.pi - g - 0.06)
        sc.curve((f"{fmt(rc + 0.02)}*sin({pg})", f"{fmt(CP_Y0)}-0.95*{fmt(rc + 0.02)}*cos({pg})", f"{fmt(z0)}+{fmt(hgt)}*t"), ('t', 0.05, 0.95),
                 GOLD, f'capeCollarEdge{tag}', thickness=2)
    # embroidered chrysanthemums at the circled places
    for k, (C, n, up) in EMBLEMS.items():
        flower(add(C, mul(n, 0.01)), n, EMB_R, f'capeEmblem{k}', up=up, layers=1, flat=True, colors=(CP_EMB, CP_EMB, CP_EMB, GOLD))


if want('cape'):
    _emit_cape()
