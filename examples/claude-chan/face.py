# Run inside generate.py: Claude Chan's head, sculpted from separate parts.
import numpy as np

sc.blank()
sc.note('Head')

ZE = 11.73          # height of the pupils
HC = ZE - 0.77      # underside of the chin (the lower face is long and V-shaped, as on the sheet)
Y0 = 0.10           # the plane just in front of the ears, where the face meets the side of the head


def spline_fn(name, pts, var='h'):
    """Natural cubic spline through pts as an Arch function: a0 + a1 d + sum c_i max(0, var - k_i)^3."""
    ks = [p[0] for p in pts]
    n = len(pts) - 1
    A = np.zeros((n + 2, n + 2))
    b = np.zeros(n + 2)
    for r, (k, v) in enumerate(pts):
        A[r, 0], A[r, 1] = 1, k - ks[0]
        for i in range(n):
            A[r, 2 + i] = max(0.0, k - ks[i]) ** 3
        b[r] = v
    for i in range(n):
        A[n + 1, 2 + i] = 6 * (ks[n] - ks[i])
    c = np.linalg.solve(A, b)
    body = f"{fmt(c[0])}+{fmt(c[1])}*({var}-{fmt(ks[0])})" + ''.join(
        f"+{fmt(c[2 + i])}*max(0,{var}-{fmt(ks[i])})^3" for i in range(n))
    sc.func(f'{name}({var})', body)


sc.comment("""The head is sculpted from parts: a face mask, the cranium under the hair, the underside of the jaw,
the neck and the ears. The face mask is a stack of horizontal arcs, one per height h. Each arc runs from the
side of the head around the front to the other side: half-width fwd(h), depth fd(h), and a superellipse
exponent fq(h) that makes it boxy across the eyes (a flat front that turns quickly at the cheekbones) and
rounder along the jaw. fpl(h) is the profile of the face seen from the side (forehead, the dip between the
eyes, the plane of the cheeks and mouth, chin), and fws(h) the outline seen from the front: a blunt chin,
straight jaw lines up to the cheekbones, level sides. fr adds what is narrower than the whole face: the nose, lips, chin, the
round of each cheek, brow and eye sockets. fy(p,h) is the y of the skin at x=p, z=h.""")
sc.func('smn(a,b,c)', '(a+b-sqrt((a-b)^2+c^2))/2')
R = lambda dz, y: (ZE + dz, y)
spline_fn('fpl', [R(-0.77, -0.44), R(-0.749, -0.478), R(-0.701, -0.515), R(-0.633, -0.54), R(-0.545, -0.56),
                 R(-0.47, -0.585), R(-0.38, -0.607), R(-0.265, -0.618), R(-0.15, -0.6),
                 R(-0.05, -0.583), R(0.03, -0.578), R(0.12, -0.592), R(0.22, -0.607), R(0.38, -0.613),
                 R(0.52, -0.596), R(0.66, -0.555), R(0.82, -0.48)])
T_ = lambda t, w: (HC + t, w)
# Seen from the front the jaw runs nearly straight from the cheekbones to a blunt, rounded chin: the sheet's
# V-shaped jaw, a little softer so it reads well in 3D.
spline_fn('fws', [T_(0, 0.06), T_(0.03, 0.155), T_(0.07, 0.214), T_(0.12, 0.284), T_(0.17, 0.346), T_(0.22, 0.398), T_(0.27, 0.441),
                  T_(0.32, 0.479), T_(0.37, 0.513), T_(0.42, 0.541), T_(0.48, 0.585), T_(0.58, 0.624),
                  T_(0.68, 0.647), T_(0.77, 0.651), T_(0.96, 0.655), T_(1.26, 0.645), T_(1.55, 0.56), T_(1.62, 0.525)])
sc.func('fwd(h)', f"fws(h)*sqrt(max(0,h-{fmt(HC)})/(h-{fmt(HC - 0.035)}))")
sc.func('fd(h)', f"smn({fmt(Y0)}-fpl(h),1.18*fwd(h),0.08)")
# fq: boxy across the eyes (0.19) and, below them, flatter still across the cheeks than a real face (0.23), turning
# quickly at the outline as anime faces are drawn; Arch's light (from the camera) then stays even over the cheeks
# and falls off only near the edge (measured from the portrait's viewpoint: 0.93 of full at f 0.8, from 0.85)
sc.func('fq(h)', f"0.25-0.06*exp(-((h-{fmt(ZE)})/0.14)^2)-0.03/(1+exp((h-{fmt(ZE - 0.3)})/0.07))")
NT = ZE - 0.255     # height of the tip of the nose
sc.func('fno(h)', f"0.036*exp(-(max(0,h-{fmt(NT)})/0.07)^1.5-(max(0,{fmt(NT)}-h)/0.035)^2)")
sc.func('fr(p,h)',
        f"-fno(h)*exp(-(p/(0.034+0.024*exp(-((h-{fmt(NT - 0.01)})/0.05)^2)))^2)"
        f"-0.008*exp(-((h-{fmt(NT - 0.015)})/0.06)^2-(p/0.1)^2)"
        f"-0.008*exp(-((h-{fmt(ZE - 0.425)})/0.03)^2-(p/0.085)^2)"
        f"+0.006*exp(-((h-{fmt(ZE - 0.46)})/0.012)^2-(p/0.08)^2)"
        f"-0.008*exp(-((h-{fmt(ZE - 0.5)})/0.028)^2-(p/0.065)^2)"
        f"-0.006*exp(-((h-{fmt(ZE - 0.66)})/0.065)^2-(p/0.1)^2)"
        f"+0.02*exp(-((abs(p)-0.33)/0.17)^2-((h-{fmt(ZE + 0.02)})/0.1)^2)"
        f"-0.008*exp(-((abs(p)-0.3)/0.2)^2-((h-{fmt(ZE + 0.24)})/0.07)^2)"
        f"-0.04*exp(-((abs(p)-0.3)/0.2)^2-((h-{fmt(ZE - 0.22)})/0.17)^2)"
        f"-0.02*exp(-((abs(p)-0.26)/0.18)^2-((h-{fmt(ZE - 0.44)})/0.13)^2)")
sc.func('fy(p,h)', 'fpl(h)+fd(h)*(1-max(0,1-(p/fwd(h))^2)^fq(h))+fr(p,h)')

HN = ZE + 0.14      # the lower patches (nose, mouth, chin) end here
HT = ZE + 0.78
FS = 0.36
MASK = ('fwd(h)*sin(f)', 'fpl(h)+fd(h)*(1-abs(cos(f))^(2*fq(h)))+fr(fwd(h)*sin(f),h)', 'h')
HK = ZE - 0.07      # the cheek patches end here, just under the eyes
sc.comment("""The mask is drawn as nine patches of the same surface, three across and three up: cheek, nose and
mouth, cheek; eye, bridge of the nose, eye; and the brow. So the cheeks, nose, lips and chin each get
their own fine mesh. Patches share their edges exactly, so no seam shows.""")
for row, hr in (('', (HC, HK)), ('Eye', (HK, HN)), ('Brow', (HN, HT))):
    for col, fr_ in (('Right', ('-pi/2', -FS)), ('Mid', (-FS, FS)), ('Left', (FS, 'pi/2'))):
        nm = {'Right': 'cheekRight', 'Mid': 'noseMouth', 'Left': 'cheekLeft'}[col] if not row else f'face{row}{col}'
        sc.surface(MASK, ('h', hr[0], hr[1]), ('f', fr_[0], fr_[1]), SKIN, nm, ('point', (0, 0.2, ZE + 0.1)))

sc.comment("Underside of the jaw: chords between the two sides of the jawline, arched up a little in the middle.")
HJ = ZE - 0.12
sc.surface(('q*fwd(h)', 'fpl(h)+fd(h)', f"h+0.05*(1-q^2)*min(1,(h-{fmt(HC)})/0.2)"), ('h', HC, HJ), ('q', -1, 1), SKIN_SH, 'jawUnder', ('dir', (0, 0.6, -1)))
sc.comment("Cranium (under the hair) and neck.")
CR_C, CR_R = (0, 0.14, ZE + 0.42), (0.655, 0.745, 0.7)
ellipsoid(CR_C, CR_R, SKIN, 'cranium')
sc.surface(('fwd(h)*cos(g)', f"{fmt(Y0)}+0.3*sin(g)", 'h'), ('h', ZE - 0.3, ZE + 0.5), ('g', 0, 'pi'), SKIN, 'headSide', ('point', (0, 0.2, ZE)))
# The jaw's lowest outline as seen from the front once the head is turned: z of its bottom at each x.
_jaw = []
for h in np.linspace(HC + 0.001, ZE - 0.2, 90):        # at HC itself the face is 0 wide
    for q in np.linspace(-1, 1, 81):
        x = q * sc.ns['fwd'](h)
        _jaw.append(sc.turn_point((x, sc.ns['fy'](x, h), h)))
_jaw = np.array(_jaw)
_bins = np.linspace(-0.3, 0.3, 31)
_jx = [(a + b) / 2 for a, b in zip(_bins, _bins[1:])]
_jz = [_jaw[(_jaw[:, 0] >= a) & (_jaw[:, 0] < b), 2].min() for a, b in zip(_bins, _bins[1:])]
_jc = np.polynomial.polynomial.polyfit(np.array(_jx) / 0.3, _jz, 6)
print(f'  jaw outline: max fit error {np.max(np.abs(np.polynomial.polynomial.polyval(np.array(_jx) / 0.3, _jc) - _jz)):.4f}', file=sys.stderr)
with sc.unturned():       # the neck stays with the body; the head turns on top of it
    capsule((0, 0.04, 10.2), (0, -0.05, 11.52), 0.2, 0.205, SKIN_SH, 'neck')
    sc.comment("""The head's shadow on the neck: a band whose lower edge follows the jaw's outline (after the head's tilt
and turn) moved down, in four steps for a soft edge. Arch casts no shadows; the portrait shades the neck so.""")
    for k, (dn, f) in enumerate(((0.135, 0.25), (0.115, 0.5), (0.095, 0.75), (0.075, 1.0))):
        rk = 0.2065 + 0.0012 * k
        Zs = f"{horner(list(_jc), f'({fmt(rk / 0.3)}*sin(a))')}-{fmt(dn)}+0.3*v"
        sc.surface((f"{fmt(rk)}*sin(a)", f"0.04-{fmt(0.09 / 1.32)}*({Zs}-10.2)-{fmt(rk)}*cos(a)", Zs), ('a', -1.9, 1.9), ('v', 0, 1),
                   mix(SKIN_SH, unlit('#cb9088'), f), f'neckShadow{k + 1}', ('dir', (0, -1, 0)))


EAR_LOBES = {}
sc.comment("Ears: a dished shell with a rolled helix, an inner fold, the concha, tragus and lobe.")
EK = 0.8
for side, tag in ((1, 'L'), (-1, 'R')):
    n = unit((side, 0.42, 0))
    up0 = (0, 0.33, 1)
    b = unit(sub(up0, mul(n, dot(up0, n))))
    a = cross(b, n)
    if dot(a, (0, 1, 0)) < 0:
        a = mul(a, -1)          # a points toward the back of the head
    C = (side * 0.612, 0.23, ZE - 0.045)
    fr = (a, b, n)
    k = lambda v: fmt(v * EK)
    sc.surface(place((f"{k(0.105)}*r*cos(w)*(1+0.25*sin(w))-{k(0.01)}", f"{k(0.2)}*r*sin(w)", f"{k(0.014)}+{k(0.03)}*r^6-{k(0.012)}*(1-r^2)"), C, fr), ('r', 0, 1), ('w', 0, '2*pi'), SKIN, f'ear{tag}', ('dir', n))
    sc.surface(place((f"{k(0.105)}*r*cos(w)*(1+0.25*sin(w))-{k(0.01)}", f"{k(0.2)}*r*sin(w)", f"-{k(0.004)}-{k(0.012)}*(1-r^2)"), C, fr), ('r', 0, 1), ('w', 0, '2*pi'), SKIN_SH, f'earBack{tag}', ('dir', mul(n, -1)))
    sc.surface(place((f"({k(0.105)}+{k(0.02)}*cos(v))*cos(w)*(1+0.25*sin(w))-{k(0.01)}", f"({k(0.2)}+{k(0.02)}*cos(v))*sin(w)", f"{k(0.042)}+{k(0.02)}*sin(v)"), C, fr), ('w', -1.75, 2.05), ('v', 0, '2*pi'), SKIN, f'earHelix{tag}', ('tube', 'u'))
    sc.surface(place((f"({k(0.06)}+{k(0.012)}*cos(v))*cos(w)*(1+0.25*sin(w))+{k(0.005)}", f"({k(0.12)}+{k(0.012)}*cos(v))*sin(w)+{k(0.01)}", f"{k(0.03)}+{k(0.012)}*sin(v)"), C, fr), ('w', -1.1, 1.7), ('v', 0, '2*pi'), SKIN, f'earFold{tag}', ('tube', 'u'))
    sc.surface(place((f"{k(0.05)}*r*cos(w)-{k(0.005)}", f"{k(0.075)}*r*sin(w)-{k(0.035)}", k(0.016)), C, fr), ('r', 0, 1), ('w', 0, '2*pi'), unlit('#e8a798'), f'earConcha{tag}', ('dir', n))
    ellipsoid(add(C, add(mul(a, -0.1 * EK), mul(b, -0.03 * EK))), (0.022 * EK, 0.03 * EK, 0.022 * EK), SKIN, f'earTragus{tag}', fr=fr)
    lobe = add(C, mul(b, -0.19 * EK))
    EAR_LOBES[tag] = lobe
    ellipsoid(lobe, (0.045 * EK, 0.05 * EK, 0.03 * EK), SKIN, f'earLobe{tag}', fr=fr)


def fdec(X, Z, d, U, V, color, ref, opacity=None, q='fy'):
    """Decal on the face: (X, Z) in terms of the two parameters, d in front of the skin.
    q names the face function used: fy itself, or a cheaper local fit of it (see qfit)."""
    sc.surface((X, f"{q}({X},{Z})-{fmt(d)}", Z), U, V, color, ref, ('dir', (0, -1, 0)), opacity=opacity)


def qfit(name, x0, x1, z0, z1, deg=(6, 3), keep=None):
    """Least-squares fit of the face surface over a small region, written as a cheap polynomial.
    Arch expands user functions textually, so painting many decals with the exact fy is slow;
    a local fit keeps each decal to a few dozen operations. keep(x, z) limits the region."""
    fyf = sc.ns['fy']
    xc, zc = (x0 + x1) / 2, (z0 + z1) / 2
    terms = [(i, j) for j in range(deg[1] + 1) for i in range(deg[0] + 1 - j)]
    rows, vals = [], []
    for x in np.linspace(x0, x1, 61):
        for z in np.linspace(z0, z1, 41):
            if keep and not keep(x, z):
                continue
            a_, b_ = x - xc, z - zc
            rows.append([a_ ** i * b_ ** j for i, j in terms])
            vals.append(fyf(x, z))
    c, *_ = np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)
    err = float(np.max(np.abs(np.array(rows) @ c - np.array(vals))))
    A, B = f"(p-{fmt(xc)})", f"(h-{fmt(zc)})"
    def poly_a(j):
        cs = [c[k] for k, (i, jj) in enumerate(terms) if jj == j]
        return horner(cs, A)
    body = poly_a(deg[1])
    for j in range(deg[1] - 1, -1, -1):
        body = f"{poly_a(j)}+{B}*({body})"
    sc.func(f'{name}(p,h)', body)
    print(f'  fit {name}: max error {err:.4f}', file=sys.stderr)
    return name


# ---------------- eyes ----------------
sc.blank()
sc.note('Eyes')


def nat_spline(pts):
    """Natural cubic spline through pts, as a Python function (same basis as spline_fn)."""
    ks = [p[0] for p in pts]
    n = len(pts) - 1
    A = np.zeros((n + 2, n + 2))
    b = np.zeros(n + 2)
    for r, (k, v) in enumerate(pts):
        A[r, 0], A[r, 1] = 1, k - ks[0]
        for i in range(n):
            A[r, 2 + i] = max(0.0, k - ks[i]) ** 3
        b[r] = v
    for i in range(n):
        A[n + 1, 2 + i] = 6 * (ks[n] - ks[i])
    c = np.linalg.solve(A, b)
    return lambda x: c[0] + c[1] * (x - ks[0]) + sum(c[2 + i] * np.maximum(0.0, x - ks[i]) ** 3 for i in range(n))


def curve_fn(name, pts, deg=6, rng=None):
    """A smooth curve through pts (s, z), written for Arch as one cheap polynomial in s over the points' range
    (or over rng, a part of it)."""
    f = nat_spline(pts)
    lo, hi = rng or (pts[0][0], pts[-1][0])
    m, h = (lo + hi) / 2, (hi - lo) / 2
    xs = np.linspace(lo, hi, 300)
    c = np.polynomial.polynomial.polyfit((xs - m) / h, f(xs), deg)
    err = float(np.max(np.abs(np.polynomial.polynomial.polyval((xs - m) / h, c) - f(xs))))
    sc.func(f'{name}(s)', horner(list(c), f"({fmt(1 / h)}*(s-{fmt(m)}))"))
    print(f'  curve {name}: max error {err:.4f}', file=sys.stderr)


sc.comment("""Her eyes, drawn from the sheet's natural-smile portrait: the strokes were projected onto this face from the
portrait's angle, so each eye has its own shape, as in the drawing. Each eye has its own frame: s runs outward
from the centre of the iris (toward the ear), heights are measured from the centre of the iris.
eu{L,R}(s) is the upper lid (the lower edge of the lash line), et{L,R}(s) the top of the lash line, el{L,R}(s)
the lower lid. The opening is open at both corners: the lash line and the lower lid end apart, with white
between them, and the lash line runs on past the outer corner into a long flat wing.
Layers, back to front: eye white, the pink inner corner, the lid's shadow, iris (dark rim, red-brown, a warm
amber glow below, dark mauve under the lid, light streaks, two white glints on the rim), an upright pupil
ringed lighter with a four-pointed sparkle, the lash line with four lashes curving out of it, the lower lid
line with a few short lashes, the lid's tint and its fold; as on the hi-res natural-smile portrait. Local
polynomial fits of the face (qe{L,R}) keep each decal cheap.""")
# Measured on the sheet (see the comment above), then re-traced on the hi-res portrait as Arch renders her from the
# portrait's own viewpoint (azimuth -92, elevation 12): each eye was matched in place on its iris, through a grid of
# marker points rendered on the face, so the numbers are in this face's (s, z) frame. Iris centre (x, z); the inner and
# outer corners, where the lids meet in a point; the end of the wing; points on the upper lid (the lower edge of the
# lash line), the top of the lash line, the lower lid and the crease, all relative to the iris centre.
# wing and stroke: (start, control, tip, half-width at the start) of tapering strokes in the same frame. The portrait's
# wings run on to s 0.31-0.34, onto the side of the face, which turns away there; they end at 0.27 here.
EYES = {
    'L': dict(side=1, c=(0.324, ZE - 0.030), s_in=-0.173, s_out=0.176, tip=0.27,
              lid=[(-0.173, -0.026), (-0.16, -0.042), (-0.145, -0.058), (-0.13, -0.066), (-0.11, -0.07),
                   (-0.09, -0.073), (-0.06, -0.078), (-0.03, -0.082), (0.0, -0.086), (0.05, -0.09), (0.08, -0.09), (0.1, -0.085),
                   (0.12, -0.076), (0.14, -0.06), (0.16, -0.036), (0.176, -0.012)],
              up=[(-0.173, -0.026), (-0.16, -0.017), (-0.15, -0.008), (-0.13, 0.007), (-0.11, 0.017), (-0.1, 0.024), (-0.05, 0.04),
                  (0.0, 0.043), (0.05, 0.044), (0.1, 0.036), (0.13, 0.022), (0.15, 0.008), (0.165, -0.003), (0.176, -0.012)],
              top=[(-0.173, -0.02), (-0.16, 0.02), (-0.15, 0.035), (-0.13, 0.05), (-0.1, 0.068), (-0.05, 0.08), (0.0, 0.083),
                   (0.05, 0.081), (0.1, 0.08), (0.13, 0.08), (0.15, 0.07), (0.165, 0.062), (0.176, 0.056)],
              crease=[(-0.17, 0.07), (-0.12, 0.095), (-0.06, 0.108), (0.0, 0.112), (0.06, 0.11), (0.11, 0.104), (0.15, 0.095)],
              wing=[((0.15, 0.035), (0.21, 0.027), (0.27, 0.021), 0.022), ((0.18, 0.058), (0.21, 0.064), (0.245, 0.071), 0.0035)],
              stroke=[((-0.14, 0.035), (-0.17, 0.022), (-0.2, 0.008), 0.004), ((-0.15, 0.0), (-0.175, -0.018), (-0.196, -0.032), 0.0035)]),
    'R': dict(side=-1, c=(-0.330, ZE - 0.031), s_in=-0.127, s_out=0.2, tip=0.27,
              lid=[(-0.127, -0.035), (-0.115, -0.047), (-0.1, -0.052), (-0.08, -0.056), (-0.06, -0.061),
                   (-0.03, -0.067), (0.0, -0.077), (0.05, -0.088), (0.09, -0.092), (0.12, -0.09), (0.15, -0.08), (0.175, -0.066),
                   (0.19, -0.056), (0.2, -0.045)],
              up=[(-0.127, -0.035), (-0.12, -0.022), (-0.11, -0.004), (-0.1, 0.012), (-0.08, 0.03), (-0.05, 0.044), (0.0, 0.05),
                  (0.05, 0.051), (0.1, 0.04), (0.125, 0.028), (0.15, 0.014), (0.175, -0.012), (0.19, -0.03), (0.2, -0.045)],
              top=[(-0.127, -0.03), (-0.12, -0.01), (-0.11, 0.008), (-0.1, 0.025), (-0.08, 0.05), (-0.04, 0.07), (0.0, 0.078),
                   (0.05, 0.083), (0.1, 0.082), (0.14, 0.078), (0.17, 0.065), (0.19, 0.05), (0.2, 0.045)],
              crease=[(-0.1, 0.085), (-0.05, 0.097), (0.0, 0.103), (0.05, 0.107), (0.1, 0.108), (0.14, 0.104), (0.17, 0.095)],
              wing=[((0.17, 0.03), (0.22, 0.008), (0.27, -0.015), 0.02), ((0.2, 0.052), (0.235, 0.052), (0.265, 0.05), 0.003)],
              stroke=[((-0.08, 0.045), (-0.11, 0.02), (-0.135, 0.002), 0.0035)]),
}
IRX, IRZ = 0.098, 0.104       # iris half-width and half-height: the hi-res portrait's (0.092 and 0.105 wide, averaged)
LASH = '#2e1712'
EYE_F = {}
for tag, E in EYES.items():
    curve_fn(f'eu{tag}', E['up'], deg=8)
    curve_fn(f'et{tag}', E['top'], deg=8)
    curve_fn(f'el{tag}', E['lid'], deg=8)
    curve_fn(f'ec{tag}', E['crease'], deg=4)
    euf, etf, elf = (nat_spline(E[k]) for k in ('up', 'top', 'lid'))
    EYE_F[tag] = dict(up=euf, top=etf, lid=elf)
    cx, cz, sd = E['c'][0], E['c'][1], E['side']

    def keep(x, z, E=E, euf=euf, etf=etf, elf=elf, cx=cx, cz=cz, sd=sd):
        s_ = sd * (x - cx)
        if not (E['s_in'] - 0.03 < s_ < E['tip'] + 0.02):
            return False
        sl = min(max(s_, E['lid'][0][0]), E['lid'][-1][0])
        su = min(max(s_, E['up'][0][0]), E['up'][-1][0])
        return elf(sl) - 0.03 < z - cz < max(etf(su), E['crease'][len(E['crease']) // 2][1]) + 0.03
    xa, xb = sorted((cx + sd * (E['s_in'] - 0.03), cx + sd * (E['tip'] + 0.02)))
    qfit(f'qe{tag}', xa, xb, cz - 0.13, cz + 0.17, deg=(8, 4), keep=keep)
    # cheaper versions for the layers inside the iris: the face over the iris alone, and the lids over its width
    qfit(f'qi{tag}', cx - 0.11, cx + 0.11, cz - 0.11, cz + 0.11, deg=(5, 3))
    curve_fn(f'eui{tag}', E['up'], deg=4, rng=(-0.1, 0.1))
    curve_fn(f'eli{tag}', E['lid'], deg=4, rng=(-0.1, 0.1))

# The iris from top to bottom, as measured on the hi-res portrait (lit colours): a dark mauve-brown top third, red-brown
# at the centre line, then amber in the lower third. Each step covers the iris below its height.
IRIS_STEPS = [(0.012, '#5c3331'), (0.002, '#743326'), (-0.008, '#9a4831'), (-0.018, '#c06a45'), (-0.028, '#da8b5e'),
              (-0.038, '#e49e76'), (-0.048, '#eaaa80')]


def eye_layers(tag):
    E = EYES[tag]
    cx, cz, sd = E['c'][0], E['c'][1], E['side']
    QE = f'qe{tag}'
    eu, et, el, ec = f'eu{tag}', f'et{tag}', f'el{tag}', f'ec{tag}'
    X = f"{fmt(cx)}+s" if sd > 0 else f"{fmt(cx)}-s"
    Zc = fmt(cz)
    si, so = E['s_in'], E['s_out']
    span = ('s', si, so)
    ends = f"min(1,(s-({fmt(si)}))/0.05)^0.5*min(1,({fmt(so)}-s)/0.05)^0.5"
    # the white, and the grey-mauve shade the upper lid casts on it, in two steps (the portrait's is 0.03-0.04 deep)
    fdec(X, f"{Zc}+{el}(s)+eta*({eu}(s)-{el}(s))", 0.003, span, ('eta', 0, 1), unlit('#f2ebeb'), f'eyeWhite{tag}', q=QE)
    for k, (depth, col) in enumerate(((0.042, '#d3c1c1'), (0.027, '#b39999'))):
        fdec(X, f"{Zc}+max({el}(s),{eu}(s)-eta*{fmt(depth)}*{ends})", 0.0038 + 0.0004 * k, span, ('eta', 0, 1),
             unlit(col), f'lidShadow{tag}{k + 1}', q=QE)

    QI, eui, eli = f'qi{tag}', f'eui{tag}', f'eli{tag}'

    def cap(zk, rs, d, color, ref):
        """The iris below height zk (scale rs), clipped by the lower lid; zk None: the whole iris, clipped by both lids.
        Over the iris the upper lid is always above the steps (0.019 against 0.012 at most), and each step is drawn
        only as wide as its top lies inside the circle, so its top is simply zk."""
        a, b = IRX * rs, IRZ * rs
        S = f"sqrt(abs(1-(s/{fmt(a)})^2))"
        lo = f"max({eli}(s),-{fmt(b)}*{S})"
        if zk is None:
            hi, w = f"min({eui}(s),{fmt(b)}*{S})", a
        else:
            hi, w = fmt(zk), a * math.sqrt(max(0.0, 1 - (zk / b) ** 2)) if zk < 0 else a
        fdec(X, f"{Zc}+{lo}+eta*({hi}-{lo})", d, ('s', -w, w), ('eta', 0, 1), color, ref, q=QI)

    def polar(s0, z0, ax, az, d, color, ref):
        Xp = f"{fmt(cx + sd * s0)}+{fmt(ax)}*r*cos(w)"
        fdec(Xp, f"{fmt(cz + z0)}+{fmt(az)}*r*sin(w)", d, ('r', 0, 1), ('w', 0, '2*pi'), color, ref, q=QI)

    def stroke(P0, P1, P2, w0, d, color, ref):
        """A stroke tapering from half-width w0 to a point, along a quadratic Bezier in the eye's (s, z) frame."""
        ns, nz = -(P2[1] - P0[1]), P2[0] - P0[0]
        n = math.hypot(ns, nz); ns, nz = ns / n, nz / n
        B = lambda i: f"(1-t)^2*{fmt(P0[i])}+2*t*(1-t)*{fmt(P1[i])}+t^2*{fmt(P2[i])}"
        W = f"{fmt(w0)}*(1-t)^0.9*q"
        fdec(f"{fmt(cx)}+{fmt(sd)}*({B(0)}+{fmt(ns)}*{W})", f"{Zc}+{B(1)}+{fmt(nz)}*{W}", d, ('t', 0, 1), ('q', -1, 1), color, ref, q=QE)

    # the iris: a dark rim, a mauve-brown body, and steps of red-brown to amber covering it from below
    cap(None, 1.0, 0.0052, unlit('#361614'), f'irisRim{tag}')
    cap(None, 0.94, 0.0056, unlit('#4d2c2b'), f'iris{tag}')
    for k, (zk, col) in enumerate(IRIS_STEPS):
        cap(zk, 0.94, 0.0059 + 0.0003 * k, unlit(col), f'irisStep{tag}{k + 1}')
    # a thin dark shade right under the lash line, inside the iris
    S = f"sqrt(abs(1-(s/{fmt(IRX * 0.94)})^2))"
    fdec(X, f"{Zc}+min({eui}(s),{fmt(IRZ * 0.94)}*{S})-eta*0.011", 0.0084, ('s', -IRX * 0.94, IRX * 0.94), ('eta', 0, 1),
         unlit('#1e0a08'), f'irisTop{tag}', q=QI)
    # streaks: lavender in the shaded upper iris, light gold radiating through the lower iris
    def streak(th, r0, r1, wd, col, ref):
        c, s_ = math.cos(th), math.sin(th)
        fdec(f"{fmt(cx)}+{fmt(sd * IRX * c)}*rho-{fmt(sd * s_)}*{fmt(wd)}*q*(1.1-rho)", f"{Zc}+{fmt(IRZ * s_)}*rho+{fmt(c)}*{fmt(wd)}*q*(1.1-rho)",
             0.008, ('rho', r0, r1), ('q', -1, 1), unlit(col), ref, q=QI)
    for k, deg in enumerate((30, 45, 60, 75, 105, 120, 135, 150)):
        streak(math.radians(deg), 0.45 if k % 2 else 0.55, 0.9, 0.0045, '#a08c9c', f'irisSheen{tag}{k + 1}')
    for k in range(13):
        r0, r1 = (0.42, 0.76) if k % 2 == 0 else (0.52, 0.88)
        streak(-math.pi / 2 + (k - 6) * 0.21, r0, r1, 0.0024, '#f6cc9c', f'irisStreak{tag}{k + 1}')
    # two white glints on the rim, low on each side: short level dashes, as drawn
    for nm, sg in (('Out', 1), ('In', -1)):
        fdec(f"{fmt(cx)}+{fmt(sd * sg)}*(0.075+0.019*t)", f"{Zc}-0.028+q*0.0017*(0.4+0.6*t)", 0.0091, ('t', 0, 1), ('q', -1, 1),
             '#f4ecea', f'irisGlint{nm}{tag}', q=QI)
    # the pupil: a dark outline round a reddish-brown fill, small and nearly round, with a sparkle at its top
    polar(0.0, -0.005, 0.026, 0.027, 0.0101, unlit('#3a1512'), f'pupilRing{tag}')
    polar(0.0, -0.006, 0.0195, 0.0205, 0.0105, unlit('#5a2b25'), f'pupil{tag}')
    star = "(0.55+0.45*cos(2*w)^2)"
    fdec(f"{fmt(cx + 0.004)}+0.013*r*cos(w)*{star}", f"{fmt(cz + 0.02)}+0.015*r*sin(w)*{star}", 0.0118, ('r', 0, 1), ('w', 0, '2*pi'),
         '#fbf6f4', f'shine{tag}', q=QI)
    # the lash line, from corner to corner: thin where it meets the lower lid at each end, heavy across the eye
    fdec(X, f"{Zc}+{eu}(s)+q*max(0,{et}(s)-{eu}(s))", 0.014, span, ('q', 0, 1), LASH, f'lash{tag}', q=QE)
    # its feathered top edge: a comb of short teeth leaning outward, 4 mesh steps apart
    a_, b_ = si + 0.03, so - 0.01
    D = (b_ - a_) / 31
    L = f"0.008*((1+cos({fmt(2 * math.pi / (4 * D))}*(s-({fmt(a_)}))))/2)^5"
    fdec(f"{fmt(cx)}+{fmt(sd)}*(s+0.6*{L}*q)", f"{Zc}+{et}(s)-0.002+{L}*q", 0.0141, ('s', a_, b_), ('q', 0, 1), LASH, f'lashFringe{tag}', q=QE)
    # the wing past the outer corner, a thin strand above it, and a stroke sweeping into the inner corner
    for k, (P0, P1, P2, w0) in enumerate(E['wing']):
        stroke(P0, P1, P2, w0, 0.0141, LASH, f'lashWing{tag}{k + 1}')
    for k, (P0, P1, P2, w0) in enumerate(E['stroke']):
        stroke(P0, P1, P2, w0, 0.0141, LASH, f'lashInner{tag}{k + 1}')
    # four separate lashes curving up and out of it, where the hi-res portrait draws them (s -0.085 to 0.146): a comb
    # whose teeth are the peaks of ((1+cos)/2)^12, 10 mesh steps apart at precision 10 (31 steps a side) so that each
    # tooth is sampled at its tip
    D, s0 = 0.0077, -0.085
    L = f"(0.02+0.06*(s+{fmt(-s0)}))*((1+cos({fmt(2 * math.pi / (10 * D))}*(s+{fmt(-s0)})))/2)^12"
    fdec(f"{fmt(cx)}+{fmt(sd)}*(s+{L}*q*(0.15+0.75*q))", f"{Zc}+{et}(s)-0.003+{L}*q", 0.0142, ('s', s0 - D, s0 + 30 * D), ('q', 0, 1),
         LASH, f'lashUp{tag}', q=QE)
    # seven short lower lashes leaning down and out, 5 mesh steps apart
    p0 = -0.09 if tag == 'L' else -0.07
    D = 0.008
    L = f"0.015*((1+cos({fmt(2 * math.pi / (5 * D))}*(s-({fmt(p0)}))))/2)^8"
    fdec(f"{fmt(cx)}+{fmt(sd)}*(s+0.5*{L}*q)", f"{Zc}+{el}(s)-0.004-{L}*q", 0.013, ('s', p0 - D, p0 + 30 * D), ('q', 0, 1),
         unlit('#9a5446'), f'lashLow{tag}', q=QE)
    # the lower lid: a thin reddish line, fading at both corners, and a soft shade under it
    taper = f"min(1,(s-({fmt(si)}))/0.03)^0.5*min(1,({fmt(so)}-s)/0.04)^0.7"
    fdec(X, f"{Zc}+{el}(s)+0.002-q*0.0062*{taper}", 0.013, span, ('q', 0, 1), unlit('#c66a5d'), f'lidLow{tag}', q=QE)
    fdec(X, f"{Zc}+{el}(s)-0.004-q*0.012*{taper}", 0.0035, span, ('q', 0, 1), unlit('#eec7bf'), f'lidLowShade{tag}', q=QE)
    # a soft pink-brown tint on the lid above the lash line, and the fold of the lid along its own measured line
    a_, b_ = si + 0.02, so - 0.01
    mid, half = (a_ + b_) / 2, (b_ - a_) / 2
    fdec(X, f"{Zc}+{et}(s)-0.002+q*0.016*(1-((s-{fmt(mid)})/{fmt(half + 0.02)})^2)", 0.0035, ('s', a_, b_), ('q', 0, 1),
         unlit('#e6b8ac'), f'lidTint{tag}', q=QE)
    c0, c1 = E['crease'][0][0], E['crease'][-1][0]
    fdec(X, f"{Zc}+{ec}(s)+q*0.0045*(1-((s-{fmt((c0 + c1) / 2)})/{fmt((c1 - c0) / 2)})^2)^0.6", 0.004, ('s', c0, c1), ('q', 0, 1),
         unlit('#c98b7e'), f'crease{tag}', q=QE)

for tag in ('L', 'R'):
    eye_layers(tag)

sc.comment("""Blush: a wide soft flush under each eye, strongest toward the outer cheek, as on the hi-res natural-smile
portrait (measured there: centred 0.34-0.41 out and 0.22-0.24 below the eyes, about 0.1 across and 0.08 high to its
half strength). Arch draws a surface in one colour, so it fades in ten nested ovals, each a level of a Gaussian
stepping from the skin to the peak; they reach further toward the nose than toward the ear and stop short of the jaw, where
the face turns away steeply, and under the lower lid. qk{L,R} are local fits of the cheek.""")
BLUSH = unlit('#ecb4a9')       # the portrait's at its strongest
NB = 10
for side, tag in ((1, 'L'), (-1, 'R')):
    bx, bz = 0.35 * side, ZE - 0.24
    r1 = math.sqrt(-2 * math.log(0.5 / NB))                 # the outermost oval

    def keep(x, z, side=side, bx=bx, bz=bz, r1=r1):
        dx, dz = side * (x - bx), z - bz
        return (dx / ((0.066 if dx > 0 else 0.094) * r1)) ** 2 + (dz / ((0.044 if dz > 0 else 0.056) * r1)) ** 2 < 1.1
    xa, xb = sorted((side * 0.08, side * 0.58))
    qfit(f'qk{tag}', xa, xb, ZE - 0.44, ZE - 0.11, deg=(6, 4), keep=keep)
    for k in range(1, NB + 1):
        r = math.sqrt(-2 * math.log((k - 0.5) / NB))     # where the Gaussian falls to this level, in standard deviations
        out, inn, up, dn = 0.066 * r, 0.094 * r, 0.044 * r, 0.056 * r
        X = f"{fmt(bx)}+{fmt(side)}*r*cos(w)*({fmt((out + inn) / 2)}+{fmt((out - inn) / 2)}*cos(w))"
        Z = f"{fmt(bz)}+r*sin(w)*({fmt((up + dn) / 2)}+{fmt((up - dn) / 2)}*sin(w))"
        fdec(X, Z, 0.0018 + 0.0003 * k, ('r', 0, 1), ('w', 0, '2*pi'), mix(SKIN, BLUSH, k / NB), f'blush{tag}{k}', q=f'qk{tag}')


# ---------------- brows ----------------
sc.comment("""Brows: long and nearly straight, sloping gently down toward the outside, fuller at the inner end, as on the
hi-res portrait (measured there, unwarped onto this face: 0.26-0.27 above the pupils at the inner end, 0.22-0.24 at
the outer, 0.023 thick inside and 0.016 outside), where they are drawn over the hair. Each is a decal on the skin along bw{L,R}(s) (s outward from the middle of the face); where a lock of the
bangs crosses it, a short strip carries the brow on over the front of that lock (drawn after the hair).""")
BROWS = {
    'L': dict(side=1, pts=[(0.134, 0.232), (0.174, 0.237), (0.224, 0.237), (0.274, 0.231), (0.324, 0.226), (0.374, 0.219),
                           (0.424, 0.209), (0.474, 0.199), (0.51, 0.19)]),
    'R': dict(side=-1, pts=[(0.28, 0.234), (0.33, 0.235), (0.38, 0.235), (0.43, 0.231), (0.48, 0.219), (0.51, 0.209),
                            (0.56, 0.193)]),
}
BROW_C = unlit('#c96c45')      # the portrait's soft orange-brown
for tag, B in BROWS.items():
    curve_fn(f'bw{tag}', B['pts'], deg=4)
    B['f'] = nat_spline(B['pts'])
    s0, s1 = B['pts'][0][0], B['pts'][-1][0]
    xa, xb = sorted((B['side'] * (s0 - 0.03), B['side'] * (s1 + 0.03)))
    qfit(f'qb{tag}', xa, xb, ZE + 0.15, ZE + 0.3, deg=(7, 3))


def brow_th(s0, s1, u):
    """Brow thickness at u (an Arch expression in the brow's s): fuller at the inner end, thin at the outer."""
    t = f"(({u})-({fmt(s0)}))/{fmt(s1 - s0)}"
    return f"(0.023-0.007*{t})*min(1,{t}/0.07)^0.5*min(1,(1-{t})/0.15)^0.7"


def draw_brows(blades):
    """Draw both brows; blades are (ref, xyz) of the locks in front of the face (unturned), for the strips over them."""
    H = []
    for ref, xyz in blades:
        g = sc._f(*xyz, 't', 'w')
        for t in np.linspace(0, 1, 170):
            for w in np.linspace(0, 2 * math.pi, 36, endpoint=False):
                P = g(t, w)
                if P and abs(P[2] - (ZE + 0.215)) < 0.06:
                    H.append(P)
    H = np.array(H) if H else np.zeros((0, 3))
    for tag, B in BROWS.items():
        sd = B['side']
        s0, s1 = B['pts'][0][0], B['pts'][-1][0]
        X = 's' if sd > 0 else '-s'
        fdec(X, f"{fmt(ZE)}+bw{tag}(s)+(q-0.5)*{brow_th(s0, s1, 's')}", 0.004, ('s', s0, s1), ('q', 0, 1), BROW_C, f'brow{tag}', q=f'qb{tag}')
        ss = np.linspace(s0, s1, 100)
        xs, zs = sd * ss, ZE + B['f'](ss)
        yh = np.full(len(ss), np.nan)
        for i in range(len(ss)):
            m = (np.abs(H[:, 0] - xs[i]) < 0.009) & (np.abs(H[:, 2] - zs[i]) < 0.009) if len(H) else np.zeros(0, bool)
            if m.any():
                yh[i] = H[m, 1].min()
        skin = np.array([sc.ns['fy'](x, z) for x, z in zip(xs, zs)])
        over = np.isfinite(yh) & (yh < skin - 0.012)
        runs, i = [], 0
        while i < len(ss):
            if over[i]:
                j = i
                while j + 1 < len(ss) and over[j + 1]:
                    j += 1
                runs.append((max(0, i - 1), min(len(ss) - 1, j + 1)))
                i = j + 1
            else:
                i += 1
        for k, (i0, i1) in enumerate(runs):
            if i1 - i0 < 2:
                continue
            yr = np.where(np.isfinite(yh[i0:i1 + 1]), yh[i0:i1 + 1], skin[i0:i1 + 1])
            yr = np.array([yr[max(0, j - 2):j + 3].min() for j in range(len(yr))]) - 0.008
            u = np.linspace(0, 1, len(yr))
            c = np.polynomial.polynomial.polyfit(u, yr, min(3, len(yr) - 1))
            c[0] -= max(0.0, float(np.max(np.polynomial.polynomial.polyval(u, c) - yr)))
            sa, sb = ss[i0], ss[i1]
            S = f"({fmt(sa)}+{fmt(sb - sa)}*u)"
            sc.surface((f"{fmt(sd)}*{S}", horner(list(c), 'u'), f"{fmt(ZE)}+bw{tag}({S})+(q-0.5)*{brow_th(s0, s1, S)}"),
                       ('u', 0, 1), ('q', 0, 1), BROW_C, f'brow{tag}Hair{k + 1}', ('dir', (0, -1, 0)))


HS_TOP, HS_CUT = ZE + 0.68, ZE + 0.12      # the bangs' shadows lie on the forehead, from the hairline to above the eyes
HS_LAYERS = ((1.45, 0.042, 0.5, 0.0021), (1.18, 0.024, 1.0, 0.0026))    # widening, drop, strength, offset: a soft edge


def draw_hair_shadows(blades):
    """Arch casts no shadows, so each lock lying on the forehead gets a darker strip on the skin under it, as the
    portrait shades the skin under the bangs: the lock's own outline seen from the front, widened and moved down
    a little, so a thin rim shows along its edges and a point under its tip. Two strips per lock give a soft edge.
    Where a lock runs on past the hairline, the eyes or the side of the face, its strip narrows back to the lock's
    own outline (hidden under it), so it never ends in a hard line. blades are (ref, xyz) of the locks, unturned.
    Each strip's depth is fitted as a polynomial in its own parameters (u along, q across), so Arch does not
    evaluate the face for it."""
    fyf, fwd_ = sc.ns['fy'], sc.ns['fwd']
    ws = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    uu, qq = np.meshgrid(np.linspace(-1, 1, 61), np.linspace(-1, 1, 9))
    uu, qq = uu.ravel(), qq.ravel()
    terms = [(i, j) for j in range(5) for i in range(9)]
    deep = []
    worst, wdepth, n = (0.0, ''), (0.0, ''), 0
    for ref, xyz in blades:
        if ref.endswith('Sheen') or 'Line' in ref:
            continue
        g = sc._f(*xyz, 't', 'w')
        ts = np.linspace(0, 1, 241)
        rows = [np.array([p for p in (g(t, w) for w in ws) if p]) for t in ts]
        cen = np.array([r.mean(0) for r in rows])
        tan = np.gradient(cen[:, [0, 2]], axis=0)
        tan /= np.linalg.norm(tan, axis=1)[:, None]
        nrm = np.stack([tan[:, 1], -tan[:, 0]], 1)
        cx, cz = cen[:, 0], cen[:, 2]
        on = np.array([HS_CUT < z < HS_TOP and abs(x) < 0.84 * fwd_(z) and fyf(x, z) - y < 0.16 for x, y, z in cen])
        # the longest run of the lock that lies on the forehead
        best, i = (0, -1), 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                if j - i > best[1] - best[0]:
                    best = (i, j)
                i = j + 1
            else:
                i += 1
        i0, i1 = best
        if i1 - i0 < 10:
            continue
        sl = slice(i0, i1 + 1)
        lo = np.array([((r[:, [0, 2]] - c[[0, 2]]) @ nv).min() for r, c, nv in zip(rows[sl], cen[sl], nrm[sl])])
        hi = np.array([((r[:, [0, 2]] - c[[0, 2]]) @ nv).max() for r, c, nv in zip(rows[sl], cen[sl], nrm[sl])])
        x, z, nv = cx[sl], cz[sl], nrm[sl]
        # 1 where the shadow shows in full, easing to 0 toward the hairline, the eyes and the side of the face
        sstep = lambda v: (lambda c: c * c * (3 - 2 * c))(np.clip(v, 0, 1))
        ramp = sstep((z - HS_CUT) / 0.07) * sstep((HS_TOP - z) / 0.07) * \
            sstep((0.84 * np.array([fwd_(v) for v in z]) - np.abs(x)) / 0.06)
        u = np.linspace(-1, 1, len(x))
        n += 1
        for k, (wk, dk, fk, ok) in enumerate(HS_LAYERS):
            W, dz = 1 + (wk - 1) * ramp, dk * ramp
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * W
            M = np.stack([x + mid * nv[:, 0], z + mid * nv[:, 1] - dz], 1)
            Dv = half[:, None] * nv
            cs, polys = [], []
            for col in (M[:, 0], M[:, 1], Dv[:, 0], Dv[:, 1]):
                c = np.polynomial.polynomial.polyfit(u, col, 8)
                worst = max(worst, (float(np.max(np.abs(np.polynomial.polynomial.polyval(u, c) - col))), ref))
                cs.append(c)
                polys.append(horner(list(c), 'u'))
            pv = lambda c: np.polynomial.polynomial.polyval(uu, c)
            Xs, Zs = pv(cs[0]) + qq * pv(cs[2]), pv(cs[1]) + qq * pv(cs[3])
            Ys = np.array([fyf(a, b) for a, b in zip(Xs, Zs)]) - ok
            A = np.array([uu ** i * qq ** j for i, j in terms]).T
            cy, *_ = np.linalg.lstsq(A, Ys, rcond=None)
            e = float(np.max(np.abs(A @ cy - Ys)))
            wdepth = max(wdepth, (e, ref))
            if e > 0.001:
                deep.append(f'{ref}{k + 1} {e:.4f}')
            Pj = [horner([cy[m] for m, (i, j) in enumerate(terms) if j == jj], 'u') for jj in range(5)]
            Y = Pj[4]
            for jj in (3, 2, 1, 0):
                Y = f"{Pj[jj]}+q*({Y})"
            sc.surface((f"{polys[0]}+q*({polys[2]})", Y, f"{polys[1]}+q*({polys[3]})"),
                       ('u', -1, 1), ('q', -1, 1), mix(SKIN, unlit('#e8c4bb'), fk), f'{ref}Shadow{k + 1}', ('dir', (0, -1, 0)))
    print(f'  hair shadows: {n} locks, outline fit max error {worst[0]:.4f} ({worst[1]}), depth {wdepth[0]:.4f} ({wdepth[1]})'
          + (f'; over 0.001: {", ".join(deep)}' if deep else ''), file=sys.stderr)

sc.blank()
sc.note('Nose and mouth')
sc.comment("""As on the sheet: a thin shadow line down her right side of the nose that turns under its tip (an L), a
highlight above the tip, and a calm closed smile, slightly lower on her left.""")
fdec('-0.024-0.012*t^1.5+0.0028*q', f"{fmt(NT + 0.075)}-0.083*t", 0.0016, ('t', 0, 1), ('q', -1, 1), unlit('#dba79b'), 'noseLine')
fdec('-0.034+0.044*t', f"{fmt(NT - 0.012)}-0.008*t+0.006*t^2+0.0028*q*(1-0.6*t)", 0.0016, ('t', 0, 1), ('q', -1, 1), unlit('#d09a8e'), 'noseUnder')
fdec('0.004+0.009*r*cos(w)', f"{fmt(NT + 0.035)}+0.028*r*sin(w)", 0.0015, ('r', 0, 1), ('w', 0, '2*pi'), unlit('#fff6f0'), 'noseShine')
MZ = ZE - 0.46
sc.comment("""Lips, as on the hi-res portrait: a soft pink lower lip just under the smile (fading in three ovals), a
faint tint above it and a small gloss highlight; the smile itself is a warm red-brown.""")
LIP = unlit('#efb5ab')
for k, (ax, az, f) in enumerate(((0.068, 0.019, 0.35), (0.054, 0.0145, 0.7), (0.038, 0.01, 1.0))):
    fdec(f"-0.012+{fmt(ax)}*r*cos(w)", f"{fmt(MZ - 0.019)}+{fmt(az)}*r*sin(w)", 0.0014 + 0.0004 * k, ('r', 0, 1), ('w', 0, '2*pi'),
         mix(SKIN, LIP, f), f'lipLower{k + 1}')
fdec('-0.03+0.06*r*cos(w)', f"{fmt(MZ + 0.011)}+0.0075*r*sin(w)", 0.0014, ('r', 0, 1), ('w', 0, '2*pi'), mix(SKIN, LIP, 0.4), 'lipUpper')
fdec('0.008+0.009*r*cos(w)', f"{fmt(MZ - 0.017)}+0.0026*r*sin(w)", 0.0028, ('r', 0, 1), ('w', 0, '2*pi'), unlit('#fff3ef'), 'lipGloss')
fdec('p', f"{fmt(MZ)}-0.12*(p+0.03)*0.085+0.011*((p+0.03)/0.1)^2+q*0.0085*(1-((p+0.03)/0.102)^2)^0.6", 0.003, ('p', -0.13, 0.07), ('q', -0.45, 0.45),
     unlit('#b45a50'), 'smile')
