# Run inside generate.py: Claude Chan's head, sculpted from separate parts.
import numpy as np

sc.blank()
sc.note('Head')

ZE = 11.73          # height of the pupils
HC = ZE - 0.82      # underside of the chin (the lower face is long and V-shaped, as on the hi-res portrait)
# Against the hi-res portrait the face read as tipped forward, looking down: the chin was short (the mouth too close
# to it) and the eyes low under the forehead. So the face below the nose is stretched to put the chin 0.05 lower
# (lowstretch maps the heights the profile and outline were measured at), the mouth sits 0.025 higher, nearer the
# nose, and the eyes, brows and blush 0.02 higher (EYE_UP). Eyes to nose, nose to mouth, mouth to chin are then about
# 0.35 : 0.25 : 0.40 of the eyes-to-chin distance, as drawn.
EYE_UP = 0.02


def lowstretch(dz, top=-0.25, old=-0.77, new=-0.82):
    """A height below the pupils (dz) on the old, shorter lower face, moved to the stretched one."""
    return dz if dz >= top else top + (dz - top) * (new - top) / (old - top)
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
R = lambda dz, y: (ZE + lowstretch(dz), y)
spline_fn('fpl', [R(-0.77, -0.44), R(-0.749, -0.478), R(-0.701, -0.515), R(-0.633, -0.54), R(-0.545, -0.56),
                 R(-0.47, -0.585), R(-0.38, -0.607), R(-0.265, -0.618), R(-0.15, -0.6),
                 R(-0.05, -0.583), R(0.03, -0.578), R(0.12, -0.592), R(0.22, -0.607), R(0.38, -0.613),
                 R(0.52, -0.596), R(0.66, -0.555), R(0.82, -0.48)])
T_ = lambda t, w: (ZE + lowstretch(t - 0.77), w)
# Seen from the front the outline swells softly below the eyes (full, rounded cheeks) and curves into a small, softly
# rounded chin: drawn by eye rather than traced (straight jaw lines traced from the portrait looked gaunt in 3D).
spline_fn('fws', [T_(0, 0.05), T_(0.03, 0.135), T_(0.07, 0.212), T_(0.12, 0.29), T_(0.17, 0.36), T_(0.22, 0.42), T_(0.27, 0.47),
                  T_(0.32, 0.512), T_(0.37, 0.547), T_(0.42, 0.578), T_(0.48, 0.607), T_(0.58, 0.636),
                  T_(0.68, 0.65), T_(0.77, 0.651), T_(0.96, 0.655), T_(1.26, 0.645), T_(1.55, 0.56), T_(1.62, 0.525)])
sc.func('fwd(h)', f"fws(h)*sqrt(max(0,h-{fmt(HC)})/(h-{fmt(HC - 0.035)}))")
sc.func('fd(h)', f"smn({fmt(Y0)}-fpl(h),1.18*fwd(h),0.08)")
# fq: boxy across the eyes (0.19) and, below them, flatter still across the cheeks than a real face (0.23), turning
# quickly at the outline as anime faces are drawn; Arch's light (from the camera) then stays even over the cheeks
# and falls off only near the edge (measured from the portrait's viewpoint: 0.93 of full at f 0.8, from 0.85)
sc.func('fq(h)', f"0.27-0.07*exp(-((h-{fmt(ZE)})/0.14)^2)+0.03/(1+exp((h-{fmt(ZE - 0.3)})/0.07))")
NT = ZE - 0.255     # height of the tip of the nose
sc.func('fno(h)', f"0.036*exp(-(max(0,h-{fmt(NT)})/0.07)^1.5-(max(0,{fmt(NT)}-h)/0.035)^2)")
sc.func('fr(p,h)',
        f"-fno(h)*exp(-(p/(0.034+0.024*exp(-((h-{fmt(NT - 0.01)})/0.05)^2)))^2)"
        f"-0.008*exp(-((h-{fmt(NT - 0.015)})/0.06)^2-(p/0.1)^2)"
        f"-0.008*exp(-((h-{fmt(ZE - 0.4)})/0.03)^2-(p/0.085)^2)"
        f"+0.006*exp(-((h-{fmt(ZE - 0.435)})/0.012)^2-(p/0.08)^2)"
        f"-0.008*exp(-((h-{fmt(ZE - 0.475)})/0.028)^2-(p/0.065)^2)"
        f"-0.006*exp(-((h-{fmt(ZE + lowstretch(-0.66))})/0.07)^2-(p/0.1)^2)"
        f"+0.02*exp(-((abs(p)-0.33)/0.17)^2-((h-{fmt(ZE + 0.02 + EYE_UP)})/0.1)^2)"
        f"-0.008*exp(-((abs(p)-0.3)/0.2)^2-((h-{fmt(ZE + 0.24 + EYE_UP)})/0.07)^2)"
        f"-0.058*exp(-((abs(p)-0.3)/0.21)^2-((h-{fmt(ZE - 0.24)})/0.18)^2)"
        f"-0.02*exp(-((abs(p)-0.26)/0.18)^2-((h-{fmt(ZE + lowstretch(-0.44))})/0.13)^2)")
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
    for k, (dn, f) in enumerate(((0.12, 0.4), (0.105, 0.65), (0.092, 0.88), (0.082, 1.0))):     # a crisp edge under the jaw
        rk = 0.2065 + 0.0012 * k
        Zs = f"{horner(list(_jc), f'({fmt(rk / 0.3)}*sin(a))')}-{fmt(dn)}+0.3*v"
        sc.surface((f"{fmt(rk)}*sin(a)", f"0.04-{fmt(0.09 / 1.32)}*({Zs}-10.2)-{fmt(rk)}*cos(a)", Zs), ('a', -1.9, 1.9), ('v', 0, 1),
                   mix(SKIN_SH, unlit('#cf9a98', 1.17), f), f'neckShadow{k + 1}', ('dir', (0, -1, 0)))


EAR_LOBES = {}
sc.comment("Ears: a dished shell with a rolled helix, an inner fold, the concha, tragus and lobe.")
EK = 0.88
for side, tag in ((1, 'L'), (-1, 'R')):
    n = unit((side, -0.75, 0))     # facing out and forward, as anime draws ears (the portrait shows her left one nearly
                                   # face-on), so the light from the front reaches it
    up0 = (0, 0.33, 1)
    b = unit(sub(up0, mul(n, dot(up0, n))))
    a = cross(b, n)
    if dot(a, (0, 1, 0)) < 0:
        a = mul(a, -1)          # a points toward the back of the head
    C = (side * 0.625, 0.17, ZE - 0.13)       # from the eyes' height to below the nose, just behind the jaw, as drawn
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
Layers, back to front: the eye white (faintly warm, shaded at both corners where the eyeball turns away, pink at
the inner corner); the iris (a dark rim that fades out at the bottom, then a smooth gradient in 20 steps from dark
mauve under the lid through red-brown to a warm amber crescent below, fine light streaks of uneven length, two white
glints on the rim, and, as anime eyes are painted, warm light gathered in the bottom of the iris and a faint ring of
light part way out); a soft upright pupil with a lighter ring; the upper lid's shadow over white and iris alike
(translucent, light, so the whites stay bright); the highlights composed as the drawing guides compose them: a crisp
rounded main catch-light at the top of the pupil, a small one opposite, a soft one low in the iris and a few sparkles in
the light there; the lash line, a dark core over a warm-brown layer, its feathered edge and its clumped lashes gathered
toward the outer corner, longest and most curved there; the lower lid line over the outer two-thirds only (drawn whole,
it boxed the eye in), with three faint lashes; and, for the smile, the lower lid raised and a faint fullness under it.
As on the hi-res natural-smile portrait. Local polynomial fits of the face (qe{L,R}) keep each decal cheap.""")
# The eye centres (x, z) were placed on the hi-res portrait's irises, seen from its viewpoint.
# The eyes' outlines, designed (not traced) on the usual anime pattern, keeping the drawing's proportions: the upper lid
# a strong arch peaking a little inside the iris and sloping gently to the outer corner, its lash line thin at the tear
# duct and heaviest over the outer half, flowing into a long wing; the lower lid one soft even curve, lowest just
# outside the iris, with no kinks; the crease parallel above. Traced, the lower lids kinked at the inner corner and
# sagged at the outer one, which read oddly in 3D. Her right eye, further from the viewer, is narrower on its inner
# side. In each eye's frame: s outward from the iris, heights from its centre; wing and stroke are tapering strokes
# (start, control, tip, half-width at the start).
EYES = {
    'L': dict(side=1, c=(0.324, ZE - 0.030 + EYE_UP), s_in=-0.17, s_out=0.175, tip=0.27,
              up=[(-0.17, -0.02), (-0.15, 0.005), (-0.12, 0.027), (-0.08, 0.043), (-0.03, 0.052), (0.02, 0.053), (0.07, 0.048),
                  (0.11, 0.037), (0.145, 0.02), (0.175, -0.002)],
              top=[(-0.17, -0.015), (-0.15, 0.022), (-0.12, 0.047), (-0.08, 0.067), (-0.03, 0.08), (0.02, 0.085), (0.07, 0.085),
                   (0.11, 0.08), (0.145, 0.07), (0.175, 0.058)],
              lid=[(-0.17, -0.02), (-0.155, -0.044), (-0.13, -0.061), (-0.09, -0.074), (-0.04, -0.082), (0.01, -0.086),
                   (0.06, -0.084), (0.1, -0.075), (0.13, -0.06), (0.155, -0.037), (0.175, -0.002)],
              crease=[(-0.15, 0.074), (-0.1, 0.098), (-0.04, 0.111), (0.02, 0.114), (0.08, 0.108), (0.13, 0.096), (0.16, 0.086)],
              wing=[((0.15, 0.035), (0.23, 0.028), (0.32, 0.012), 0.024), ((0.17, 0.06), (0.23, 0.07), (0.29, 0.075), 0.005),
                    ((0.16, 0.012), (0.22, 0.0), (0.265, -0.02), 0.006)],
              stroke=[((-0.14, 0.035), (-0.18, 0.022), (-0.22, 0.004), 0.0045), ((-0.15, 0.0), (-0.18, -0.02), (-0.205, -0.036), 0.0035)]),
    'R': dict(side=-1, c=(-0.330, ZE - 0.031 + EYE_UP), s_in=-0.127, s_out=0.2, tip=0.27,
              up=[(-0.127, -0.03), (-0.11, -0.004), (-0.085, 0.022), (-0.05, 0.04), (0.0, 0.05), (0.05, 0.051), (0.1, 0.043),
                  (0.14, 0.028), (0.175, 0.006), (0.2, -0.02)],
              top=[(-0.127, -0.025), (-0.11, 0.01), (-0.085, 0.046), (-0.05, 0.067), (0.0, 0.08), (0.05, 0.083), (0.1, 0.082),
                   (0.14, 0.074), (0.175, 0.062), (0.2, 0.05)],
              lid=[(-0.127, -0.03), (-0.11, -0.052), (-0.08, -0.068), (-0.04, -0.079), (0.01, -0.086), (0.06, -0.087), (0.1, -0.08),
                   (0.14, -0.066), (0.175, -0.045), (0.2, -0.02)],
              crease=[(-0.1, 0.085), (-0.05, 0.1), (0.0, 0.106), (0.05, 0.11), (0.1, 0.108), (0.14, 0.1), (0.17, 0.09)],
              wing=[((0.17, 0.03), (0.24, 0.006), (0.31, -0.022), 0.022), ((0.19, 0.054), (0.24, 0.056), (0.285, 0.05), 0.0045),
                    ((0.18, 0.005), (0.225, -0.012), (0.26, -0.035), 0.005)],
              stroke=[((-0.08, 0.045), (-0.115, 0.02), (-0.145, -0.002), 0.004)]),
}
# A smile lifts the lower lid: raised up to 0.012 in the middle and toward the outer corner (the corners stay where the
# lids meet), so the eye narrows into a soft crescent and the smile reaches the eyes.
SMILE_LID = 0.007
for _E in EYES.values():
    _a, _b = _E['lid'][0][0], _E['lid'][-1][0]
    _E['lid'] = [(s_, z_ + SMILE_LID * math.sin(math.pi * (s_ - _a) / (_b - _a)) ** 1.5 * (0.4 + 0.6 * (s_ - _a) / (_b - _a)))
                 for s_, z_ in _E['lid']]
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
    qfit(f'qi{tag}', cx - 0.11, cx + 0.12, cz - 0.11, cz + 0.13, deg=(5, 3))      # with room for the gaze (GAZE_DX, GAZE_DZ)
    curve_fn(f'eui{tag}', E['up'], deg=4, rng=(-0.1, 0.1))
    curve_fn(f'eli{tag}', E['lid'], deg=4, rng=(-0.1, 0.1))

# The iris from top to bottom (lit colours): a dark mauve-brown top third, red-brown at the centre line, amber in the
# lower third (as measured on the hi-res portrait), and a bright warm crescent at the bottom, where the light gathers in
# the hollow of the iris. Drawn as 20 fine steps, each covering the iris below its height, so the gradient is smooth
# (seven steps read as stripes); the lowest steps also cover the dark rim, which fades out at the bottom.
IRIS_STOPS = [(0.03, '#4d2c2b'), (0.012, '#5c3331'), (0.002, '#743326'), (-0.008, '#9a4831'), (-0.018, '#c06a45'),
              (-0.028, '#da8b5e'), (-0.038, '#e49e76'), (-0.05, '#e8a676'), (-0.068, '#ebac7a'), (-0.085, '#edb27f')]
IRIS_N = 20


def iris_colour(z):
    zs = [p[0] for p in IRIS_STOPS]
    if z >= zs[0]:
        return IRIS_STOPS[0][1]
    for (z0, c0), (z1, c1) in zip(IRIS_STOPS, IRIS_STOPS[1:]):
        if z1 <= z <= z0:
            return mix(c0, c1, (z0 - z) / (z0 - z1))
    return IRIS_STOPS[-1][1]


# Where she looks: painted eyes look straight along the face, which is level, so from the portrait's camera (12 degrees
# above her eyes and about 2 to her left) her gaze passed under the viewer and she seemed to look down and past them.
# The eyeballs are turned to the viewer instead: the iris, pupil and their lights move up and a little to her left inside
# the lids (about an eyeball's radius, 0.2, times the angle), and the lids stay where they are.
GAZE_DX, GAZE_DZ = 0.006, 0.016
LID_SHADOW = '#4e3440'        # the upper lid's shadow over the eye: translucent, so it shades the white and iris alike


def eye_layers(tag):
    E = EYES[tag]
    cx, cz, sd = E['c'][0], E['c'][1], E['side']
    QE = f'qe{tag}'
    eu, et, el, ec = f'eu{tag}', f'et{tag}', f'el{tag}', f'ec{tag}'
    F = EYE_F[tag]
    X = f"{fmt(cx)}+s" if sd > 0 else f"{fmt(cx)}-s"
    Zc = fmt(cz)
    si, so = E['s_in'], E['s_out']
    span = ('s', si, so)
    ends = f"min(1,(s-({fmt(si)}))/0.05)^0.5*min(1,({fmt(so)}-s)/0.05)^0.5"
    U = f"((s-({fmt(si)}))/{fmt(so - si)})"           # 0 at the inner corner, 1 at the outer
    # (7) the white: faintly warm, shaded where the eyeball turns away at both corners, pink at the inner corner
    fdec(X, f"{Zc}+{el}(s)+eta*({eu}(s)-{el}(s))", 0.003, span, ('eta', 0, 1), unlit('#f5eee9'), f'eyeWhite{tag}', q=QE)
    for k, (a_, b_, op) in enumerate(((so - 0.06, so, 0.16), (so - 0.03, so, 0.2), (si, si + 0.035, 0.12))):
        fdec(X, f"{Zc}+{el}(s)+eta*({eu}(s)-{el}(s))", 0.0033 + 0.0003 * k, ('s', a_, b_), ('eta', 0, 1), '#8a7078',
             f'eyeCorner{tag}{k + 1}', opacity=op, q=QE)
    fdec(X, f"{Zc}+{el}(s)+eta*({eu}(s)-{el}(s))", 0.0042, ('s', si, si + 0.016), ('eta', 0, 1), unlit('#eec3bb'), f'eyeInner{tag}', q=QE)

    QI = f'qi{tag}'
    gs = sd * GAZE_DX                                  # the gaze's shift in the eye's own s
    eui = lambda v: f"(eui{tag}({v}+{fmt(gs)})-{fmt(GAZE_DZ)})"     # the lids, seen from the iris's centre
    eli = lambda v: f"(eli{tag}({v}+{fmt(gs)})-{fmt(GAZE_DZ)})"
    cxi, czi = cx + GAZE_DX, cz + GAZE_DZ              # the iris's centre
    Xi = f"{fmt(cxi)}+s" if sd > 0 else f"{fmt(cxi)}-s"
    Zi = fmt(czi)

    def cap(zk, rs, d, color, ref, opacity=None):
        """The iris below height zk (scale rs), clipped by the lower lid and its own circle; zk None: the whole iris,
        clipped by both lids. Each step is drawn only as wide as its top lies inside the circle."""
        a, b = IRX * rs, IRZ * rs
        S = f"sqrt(abs(1-(s/{fmt(a)})^2))"
        lo = f"max({eli('s')},-{fmt(b)}*{S})"
        if zk is None:
            hi, w = f"min({eui('s')},{fmt(b)}*{S})", a
        elif zk >= 0:
            hi, w = f"min({fmt(zk)},{fmt(b)}*{S})", a
        else:
            hi, w = fmt(zk), a * math.sqrt(max(0.0, 1 - (zk / b) ** 2))
            # where the smiling lower lid rises above a low step's top, the band would turn over and show under the lid:
            # skip a step the lid covers entirely, and clamp one it covers in part
            lid = [float(F['lid'](v + gs)) - GAZE_DZ for v in np.linspace(-w, w, 41)]
            if sum(v < zk - 0.0015 for v in lid) < 6:        # (nearly) all under the lid
                return
            if max(lid) > zk:
                hi = f"max({fmt(zk)},{lo})"
        fdec(Xi, f"{Zi}+{lo}+eta*({hi}-{lo})", d, ('s', -w, w), ('eta', 0, 1), color, ref, opacity=opacity, q=QI)

    def glow(t, rs, d, color, ref, opacity):
        """A crescent of light along the bottom of the iris (scale rs), t deep in the middle and thinning toward the
        sides, clipped by both lids: where the light that enters the eye gathers, as anime eyes are painted."""
        a, b = IRX * rs, IRZ * rs
        S = f"sqrt(abs(1-(s/{fmt(a)})^2))"
        lo = f"max({eli('s')},-{fmt(b)}*{S})"
        hi = f"min({eui('s')},max({lo},-{fmt(b)}*{S}+{fmt(t)}*(1-0.6*(s/{fmt(a)})^2)))"
        fdec(Xi, f"{Zi}+{lo}+eta*({hi}-{lo})", d, ('s', -a, a), ('eta', 0, 1), color, ref, opacity=opacity, q=QI)

    def polar(s0, z0, ax, az, d, color, ref, opacity=None, shape='', r=(0, 1), w=(0, '2*pi')):
        Xp = f"{fmt(cxi + sd * s0)}+{fmt(ax)}*r*cos(w){shape}"
        fdec(Xp, f"{fmt(czi + z0)}+{fmt(az)}*r*sin(w){shape}", d, ('r',) + tuple(r), ('w',) + tuple(w), color, ref, opacity=opacity, q=QI)

    def stroke(P0, P1, P2, w0, d, color, ref, q=QE):
        """A stroke tapering from half-width w0 to a point, along a quadratic Bezier in the eye's (s, z) frame."""
        ns, nz = -(P2[1] - P0[1]), P2[0] - P0[0]
        n = math.hypot(ns, nz); ns, nz = ns / n, nz / n
        B = lambda i: f"(1-t)^2*{fmt(P0[i])}+2*t*(1-t)*{fmt(P1[i])}+t^2*{fmt(P2[i])}"
        W = f"{fmt(w0)}*(1-t)^0.9*q"
        fdec(f"{fmt(cx)}+{fmt(sd)}*({B(0)}+{fmt(ns)}*{W})", f"{Zc}+{B(1)}+{fmt(nz)}*{W}", d, ('t', 0, 1), ('q', -1, 1), color, ref, q=q)

    # (1) the iris: a dark rim, the mauve-brown body, then the 20 steps of the gradient
    cap(None, 1.0, 0.0046, unlit('#361614'), f'irisRim{tag}')
    cap(None, 0.94, 0.005, unlit(IRIS_STOPS[0][1]), f'iris{tag}')
    z_top, z_bot = 0.025, -0.085
    for k in range(IRIS_N):
        zk = z_top + (z_bot - z_top) * k / (IRIS_N - 1)
        rs = 0.94 + 0.035 * min(1.0, max(0.0, (-zk - 0.035) / 0.035))     # (4) the lowest steps cover most of the rim
        cap(zk, rs, 0.0054 + 0.0002 * k, unlit(iris_colour(zk - 0.003)), f'irisStep{tag}{k + 1}')
    # (4) fine radiating streaks of varied length through the lower iris, a few faint lavender ones in the shaded top
    def streak(th, r0, r1, wd, col, ref, d=0.0096):
        c, s_ = math.cos(th), math.sin(th)
        fdec(f"{fmt(cxi)}+{fmt(sd * IRX * c)}*rho-{fmt(sd * s_)}*{fmt(wd)}*q*(1.1-rho)", f"{Zi}+{fmt(IRZ * s_)}*rho+{fmt(c)}*{fmt(wd)}*q*(1.1-rho)",
             d, ('rho', r0, r1), ('q', -1, 1), unlit(col), ref, q=QI)
    for k in range(19):
        h1, h2 = (math.sin(2.39 * k + 0.7) + 1) / 2, (math.sin(3.71 * k + 2.1) + 1) / 2
        streak(-math.pi / 2 + (k - 9) * 0.145, 0.36 + 0.18 * h1, 0.7 + 0.22 * h2, 0.0013 + 0.0009 * h2,
               '#f6cb9b' if k % 2 else '#ebb482', f'irisStreak{tag}{k + 1}')
    for k, deg in enumerate((40, 62, 84, 106, 128)):
        streak(math.radians(deg), 0.5, 0.86, 0.0026, '#9a8894', f'irisSheen{tag}{k + 1}')
    # two white glints on the rim, low on each side: short level dashes, as drawn
    for nm, sg in (('Out', 1), ('In', -1)):
        fdec(f"{fmt(cxi)}+{fmt(sd * sg)}*(0.075+0.019*t)", f"{Zi}-0.028+q*0.0017*(0.4+0.6*t)", 0.0099, ('t', 0, 1), ('q', -1, 1),
             '#f4ecea', f'irisGlint{nm}{tag}', q=QI)
    # the light gathered in the bottom of the iris, as anime eyes are painted: a golden crescent in three translucent
    # steps, deepest in the middle, under the streaks so they stay crisp through it
    for k, (t, op) in enumerate(((0.06, 0.12), (0.04, 0.15), (0.022, 0.2))):        # warm amber-orange, as the
        glow(t, 0.97, 0.0093 + 0.0001 * k, '#ffa060', f'irisGlow{tag}{k + 1}', op)    # portrait's (gold read lemon)
    # a ring of light round the pupil, part way out, brightest below and fading up the sides (the guides' inner ring)
    for k, (w0, w1, op) in enumerate(((math.pi - 0.25, 2 * math.pi + 0.25, 0.2), (math.pi + 0.35, 2 * math.pi - 0.35, 0.25))):
        polar(0.0, -0.007, 0.05, 0.06, 0.00972 + 0.00001 * k, '#ffc890', f'irisRing{tag}{k + 1}', opacity=op, r=(0.88, 1), w=(w0, w1))
    # (4) the pupil: a soft upright oval with a lighter warm ring round it, no hard outline; a little larger than the
    # portrait's dot, so it reads as the eye's deep centre
    polar(0.0, -0.007, 0.031, 0.038, 0.0098, unlit('#9c4e33'), f'pupilRing{tag}')
    polar(0.0, -0.007, 0.0205, 0.0275, 0.0101, unlit('#4a2220'), f'pupil{tag}')
    # (2) the upper lid's shadow over the eye, white and iris alike: three translucent steps under the lash line
    for k, (depth, op) in enumerate(((0.05, 0.08), (0.034, 0.1), (0.018, 0.14))):      # light: the whites stay bright
        fdec(X, f"{Zc}+max({el}(s),{eu}(s)-eta*{fmt(depth)}*{ends})", 0.0104 + 0.0003 * k, span, ('eta', 0, 1), LID_SHADOW,
             f'lidShadow{tag}{k + 1}', opacity=op, q=QE)
    # a thin dark shade right under the lash line, inside the iris
    S = f"sqrt(abs(1-(s/{fmt(IRX * 0.94)})^2))"
    fdec(Xi, f"{Zi}+min({eui('s')},{fmt(IRZ * 0.94)}*{S})-eta*0.008", 0.0114, ('s', -IRX * 0.94, IRX * 0.94), ('eta', 0, 1),
         unlit('#1e0a08'), f'irisTop{tag}', q=QI)
    # (3) the catch-lights, both eyes alike as the light comes from one side (the viewer's upper left): a crisp rounded
    # highlight at the top of the pupil and a soft small one low in the iris, where the eye's wet surface shines
    # composed as the guides compose them: a main highlight, a small one opposite, and a few sparkles in the light below
    diamond = "/(abs(cos(w))^1.3+abs(sin(w))^1.3)^0.769"
    fdec(f"{fmt(cxi - 0.01)}+0.0145*r*cos(w){diamond}", f"{fmt(czi + 0.02)}+0.018*r*sin(w){diamond}", 0.0118, ('r', 0, 1), ('w', 0, '2*pi'),
         '#fffaf7', f'shine{tag}', q=QI)
    fdec(f"{fmt(cxi + 0.026)}+0.0062*r*cos(w)", f"{fmt(czi - 0.022)}+0.0062*r*sin(w)", 0.0118, ('r', 0, 1), ('w', 0, '2*pi'),
         '#fffaf5', f'shineSmall{tag}', opacity=0.9, q=QI)
    fdec(f"{fmt(cxi + 0.012)}+0.011*r*cos(w)", f"{fmt(czi - 0.058)}+0.0055*r*sin(w)", 0.0118, ('r', 0, 1), ('w', 0, '2*pi'),
         '#fff3e8', f'shineLow{tag}', opacity=0.65, q=QI)
    for k, (deg, rr, sz) in enumerate(((228, 0.74, 0.0032), (250, 0.8, 0.0024), (292, 0.78, 0.0028), (312, 0.68, 0.002))):
        a_ = math.radians(deg)
        fdec(f"{fmt(cxi + sd * IRX * rr * math.cos(a_))}+{fmt(sz)}*r*cos(w)", f"{fmt(czi + IRZ * rr * math.sin(a_))}+{fmt(sz)}*r*sin(w)",
             0.0117, ('r', 0, 1), ('w', 0, '2*pi'), '#fff6ea', f'sparkle{tag}{k + 1}', opacity=0.85, q=QI)
    # (5) the lash line: a dark core over a slightly larger warm-brown layer, so its edges are soft, with the lid's rim
    # showing as a warm line under it
    fdec(X, f"{Zc}+{eu}(s)-0.003*{ends}+q*(max(0,{et}(s)-{eu}(s))+0.0075*{ends})", 0.0136, span, ('q', 0, 1), '#6a3a2c', f'lashSoft{tag}', q=QE)
    fdec(X, f"{Zc}+{eu}(s)+q*max(0,{et}(s)-{eu}(s)-0.001)", 0.014, span, ('q', 0, 1), LASH, f'lash{tag}', q=QE)
    # its feathered top edge: short teeth leaning outward, of uneven length
    a_, b_ = si + 0.03, so - 0.01
    D = (b_ - a_) / 31
    L = (f"(0.003+0.007*((1+sin(41*s+{fmt(1.3 if tag == 'L' else 0.4)}))/2)^2)*(0.3+0.7*{U}^1.5)"
         f"*((1+cos({fmt(2 * math.pi / (4 * D))}*(s-({fmt(a_)}))))/2)^5")
    fdec(f"{fmt(cx)}+{fmt(sd)}*(s+0.6*{L}*q)", f"{Zc}+{et}(s)-0.002+{L}*q", 0.0141, ('s', a_, b_), ('q', 0, 1), LASH, f'lashFringe{tag}', q=QE)
    # the wing past the outer corner, a thin strand above it, and a stroke sweeping into the inner corner
    for k, (P0, P1, P2, w0) in enumerate(E['wing']):      # past the local fit (tip), onto the side of the face: the face itself
        stroke(P0, P1, P2, w0, 0.0141, LASH, f'lashWing{tag}{k + 1}', q='fy')
    for k, (P0, P1, P2, w0) in enumerate(E['stroke']):
        stroke(P0, P1, P2, w0, 0.0141, LASH, f'lashInner{tag}{k + 1}')
    # (5) separate lashes in clumps of uneven length and spacing, curving up and out, longer toward the outer corner
    clumps = {'L': ((-0.07, 0.018), (-0.005, 0.028), (0.05, 0.036), (0.092, 0.05), (0.128, 0.064), (0.158, 0.06), (0.184, 0.046)),
              'R': ((-0.05, 0.018), (0.015, 0.03), (0.07, 0.04), (0.115, 0.056), (0.152, 0.064), (0.186, 0.05))}[tag]
    for k, (s0, ln) in enumerate(clumps):
        u = (s0 - si) / (so - si)
        base = (s0, float(F['top'](s0)) - 0.008)        # rooted in the lash line, thick at the root like a clump
        lean = 0.3 + 0.7 * u
        stroke(base, (s0 + 0.25 * lean * ln, base[1] + 0.85 * ln), (s0 + lean * ln, base[1] + 1.05 * ln), 0.0062, 0.0142, LASH,
               f'lashUp{tag}{k + 1}')
    # (5) three faint short lashes under the outer third, leaning down and out
    for k, (u, ln) in enumerate(((0.62, 0.008), (0.74, 0.012), (0.86, 0.01))):
        s0 = si + u * (so - si)
        base = (s0, float(F['lid'](s0)) - 0.002)
        stroke(base, (s0 + 0.25 * ln, base[1] - 0.7 * ln), (s0 + 0.55 * ln, base[1] - ln), 0.0016, 0.013, unlit('#a8604f'), f'lashLow{tag}{k + 1}')
    # the lower lid: a thin warm line, strongest toward the outer corner, fading at both ends
    taper = f"min(1,max(0,{U}-0.22)/0.33)^1.3*min(1,({fmt(so)}-s)/0.035)^0.7*(0.55+0.45*{U})"
    fdec(X, f"{Zc}+{el}(s)+0.002-q*0.0058*{taper}", 0.013, span, ('q', 0, 1), unlit('#b86a58'), f'lidLow{tag}', q=QE)
    # (6) the smile under the eye: a faint lighter fullness just under the lower lid, with a soft line beneath it
    tp = f"sin(pi*{U})^0.9"
    fdec(X, f"{Zc}+{el}(s)-0.0035-q*0.017*{tp}", 0.0052, span, ('q', 0, 1), '#fff7f3', f'lidPuff{tag}', opacity=0.55, q=QE)
    fdec(X, f"{Zc}+{el}(s)-0.0035-0.017*{tp}-q*0.0045*{tp}", 0.0055, span, ('q', 0, 1), '#c99a90', f'lidPuffLine{tag}', opacity=0.3, q=QE)
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
BLUSH = unlit('#f0c2b8')       # the portrait's at its strongest: a faint flush
NB = 10
for side, tag in ((1, 'L'), (-1, 'R')):
    bx, bz = 0.34 * side, ZE - 0.2 + EYE_UP
    r1 = math.sqrt(-2 * math.log(0.5 / NB))                 # the outermost oval

    def keep(x, z, side=side, bx=bx, bz=bz, r1=r1):
        dx, dz = side * (x - bx), z - bz
        return (dx / ((0.075 if dx > 0 else 0.105) * r1)) ** 2 + (dz / ((0.036 if dz > 0 else 0.046) * r1)) ** 2 < 1.1
    xa, xb = sorted((side * 0.08, side * 0.58))
    qfit(f'qk{tag}', xa, xb, ZE - 0.38 + EYE_UP, ZE - 0.08 + EYE_UP, deg=(6, 4), keep=keep)
    for k in range(1, NB + 1):
        r = math.sqrt(-2 * math.log((k - 0.5) / NB))     # where the Gaussian falls to this level, in standard deviations
        out, inn, up, dn = 0.075 * r, 0.105 * r, 0.036 * r, 0.046 * r
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
BROW_C = unlit('#d0835c')      # the portrait's light orange-brown: a fine line, not a bar
for tag, B in BROWS.items():
    B['pts'] = [(s_, z_ + EYE_UP) for s_, z_ in B['pts']]
    curve_fn(f'bw{tag}', B['pts'], deg=4)
    B['f'] = nat_spline(B['pts'])
    s0, s1 = B['pts'][0][0], B['pts'][-1][0]
    xa, xb = sorted((B['side'] * (s0 - 0.03), B['side'] * (s1 + 0.03)))
    qfit(f'qb{tag}', xa, xb, ZE + 0.15 + EYE_UP, ZE + 0.3 + EYE_UP, deg=(7, 3))


def brow_th(s0, s1, u):
    """Brow thickness at u (an Arch expression in the brow's s): fuller at the inner end, thin at the outer."""
    t = f"(({u})-({fmt(s0)}))/{fmt(s1 - s0)}"
    return f"(0.012-0.005*{t})*min(1,{t}/0.1)^0.6*min(1,(1-{t})/0.3)^0.9"


def draw_brows(blades):
    """Draw both brows; blades are (ref, xyz) of the locks in front of the face (unturned), for the strips over them."""
    H = []
    for ref, xyz in blades:
        g = sc._f(*xyz, 't', 'w')
        for t in np.linspace(0, 1, 170):
            for w in np.linspace(0, 2 * math.pi, 36, endpoint=False):
                P = g(t, w)
                if P and abs(P[2] - (ZE + 0.215 + EYE_UP)) < 0.06:
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


HS_TOP, HS_CUT = ZE + 0.68, ZE + 0.12 + EYE_UP      # the bangs' shadows lie on the forehead, from the hairline to above the eyes
HS_LAYERS = ((1.7, 0.075, 0.45, 0.0021), (1.4, 0.05, 0.75, 0.0024), (1.15, 0.03, 1.0, 0.0027))    # widening, drop, strength, offset: a soft edge


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
                       ('u', -1, 1), ('q', -1, 1), mix(SKIN, unlit('#dfaaa4', 1.17), fk), f'{ref}Shadow{k + 1}', ('dir', (0, -1, 0)))
    print(f'  hair shadows: {n} locks, outline fit max error {worst[0]:.4f} ({worst[1]}), depth {wdepth[0]:.4f} ({wdepth[1]})'
          + (f'; over 0.001: {", ".join(deep)}' if deep else ''), file=sys.stderr)

sc.blank()
sc.note('Nose and mouth')
sc.comment("""As on the sheet: a thin shadow line down her right side of the nose that turns under its tip (an L), a
highlight above the tip, and a gentle closed smile whose corners turn up, her left one a little more.""")
fdec('-0.024-0.012*t^1.5+0.0028*q', f"{fmt(NT + 0.075)}-0.083*t", 0.0016, ('t', 0, 1), ('q', -1, 1), unlit('#dba79b'), 'noseLine')
fdec('-0.034+0.044*t', f"{fmt(NT - 0.012)}-0.008*t+0.006*t^2+0.0028*q*(1-0.6*t)", 0.0016, ('t', 0, 1), ('q', -1, 1), unlit('#d09a8e'), 'noseUnder')
fdec('0.004+0.009*r*cos(w)', f"{fmt(NT + 0.035)}+0.028*r*sin(w)", 0.0015, ('r', 0, 1), ('w', 0, '2*pi'), unlit('#fff6f0'), 'noseShine')
MZ = ZE - 0.435
sc.comment("""Lips, as on the hi-res portrait: a soft pink lower lip just under the smile (fading in three ovals), a
faint tint above it and a small gloss highlight; the smile itself is a warm red-brown.""")
LIP = unlit('#efb5ab')
for k, (ax, az, f) in enumerate(((0.068, 0.019, 0.35), (0.054, 0.0145, 0.7), (0.038, 0.01, 1.0))):
    fdec(f"-0.012+{fmt(ax)}*r*cos(w)", f"{fmt(MZ - 0.019)}+{fmt(az)}*r*sin(w)", 0.0014 + 0.0004 * k, ('r', 0, 1), ('w', 0, '2*pi'),
         mix(SKIN, LIP, f), f'lipLower{k + 1}')
fdec('-0.03+0.06*r*cos(w)', f"{fmt(MZ + 0.011)}+0.0075*r*sin(w)", 0.0014, ('r', 0, 1), ('w', 0, '2*pi'), mix(SKIN, LIP, 0.4), 'lipUpper')
fdec('0.008+0.009*r*cos(w)', f"{fmt(MZ - 0.017)}+0.0026*r*sin(w)", 0.0028, ('r', 0, 1), ('w', 0, '2*pi'), unlit('#fff3ef'), 'lipGloss')
U_ = "((p+0.03)/0.1)"
fdec('p', f"{fmt(MZ)}+0.004*{U_}^2+0.014*{U_}^4+0.005*max(0,{U_})^3+q*0.0075*(1-({U_}/1.02)^2)^0.6", 0.003, ('p', -0.13, 0.07), ('q', -0.45, 0.45),
     unlit('#a9544c'), 'smile')


# ---------------- painted shading ----------------
sc.blank()
sc.note('Shading')
sc.comment("""Arch has no textures and lights everything from the camera, so the toon shading is painted on, as anime 3D figures
paint theirs: never grey, but rose and mauve. The light comes from her left and above, as on the portrait: a soft shade
down her right temple, cheek and jaw, a little along her left cheek, a warm flush across the middle of the face (cheeks,
nose, ears), a touch of pink at the nose tip and chin, shade beside the nose and in the inner eye sockets, under the
lower lip and at the smile's corners, and a fine line along the jaw. Each is two or three translucent layers, so the
edges are soft and the shading tints whatever lies under it (blush, lips).""")
SHADE = unlit('#c98682', 1.17)      # the face's shade: a rose mauve
WARM = unlit('#f09484', 1.17)       # the flush across the middle of the face


def oval(x0, z0, ax, az, d, color, ref, op, tilt=0.0):
    c, s_ = math.cos(tilt), math.sin(tilt)
    fdec(f"{fmt(x0)}+{fmt(ax * c)}*r*cos(w)-{fmt(az * s_)}*r*sin(w)", f"{fmt(z0)}+{fmt(ax * s_)}*r*cos(w)+{fmt(az * c)}*r*sin(w)", d,
         ('r', 0, 1), ('w', 0, '2*pi'), color, ref, opacity=op)


# the far side, her right: a crescent hugging the outline, widest over the cheek, narrowing to the temple and the chin
for k, (wk, op) in enumerate(((1.0, 0.14), (0.68, 0.15), (0.4, 0.16))):
    W = f"{fmt(wk)}*(0.1+0.22*exp(-((h-{fmt(ZE - 0.3)})/0.3)^2))"
    fdec(f"-fwd(h)*(0.975-v*{W})", 'h', 0.0056 + 0.0003 * k, ('h', HC + 0.03, ZE + 0.42), ('v', 0, 1), SHADE, f'shadeFarR{k + 1}', opacity=op)
# the near side, her left, a narrow one under her hair
for k, (wk, op) in enumerate(((1.0, 0.11), (0.55, 0.13))):
    W = f"{fmt(wk)}*(0.06+0.1*exp(-((h-{fmt(ZE - 0.32)})/0.25)^2))"
    fdec(f"fwd(h)*(0.975-v*{W})", 'h', 0.0056 + 0.0003 * k, ('h', HC + 0.05, ZE + 0.3), ('v', 0, 1), SHADE, f'shadeNearL{k + 1}', opacity=op)
# the warm middle of the face, and pink at the nose tip and chin
oval(0.0, ZE - 0.21 + EYE_UP, 0.52, 0.17, 0.0052, WARM, 'flushWide', 0.09)
oval(0.0, ZE - 0.2 + EYE_UP, 0.38, 0.11, 0.0053, WARM, 'flushMid', 0.09)
oval(-0.004, NT + 0.012, 0.03, 0.022, 0.0058, WARM, 'flushNose', 0.28)
oval(-0.01, HC + 0.075, 0.055, 0.028, 0.0058, WARM, 'flushChin', 0.16)
# shade beside the nose (her right) and in the inner eye sockets
oval(-0.036, NT + 0.07, 0.018, 0.075, 0.0059, SHADE, 'shadeNose', 0.16, tilt=-0.08)
for sd, tag in ((1, 'L'), (-1, 'R')):
    oval(sd * 0.13, ZE + 0.03 + EYE_UP, 0.028, 0.055, 0.0059, SHADE, f'shadeSocket{tag}', 0.1, tilt=sd * 0.3)
# under the lower lip, and the smile's corners
oval(-0.012, MZ - 0.052, 0.046, 0.012, 0.006, SHADE, 'shadeLip', 0.18)
for nm_, (x0, z0) in (('R', (-0.128, MZ + 0.019)), ('L', (0.068, MZ + 0.024))):
    oval(x0, z0, 0.007, 0.006, 0.0062, unlit('#a9544c'), f'smileCorner{nm_}', 0.35)
# a fine warm-brown line along the jaw, from below the ears to the chin, as the portrait outlines it
HJL = ZE - 0.3
for sd, tag in ((1, 'L'), (-1, 'R')):
    H_ = f"({fmt(HC + 0.004)}+{fmt(HJL - HC)}*t)"
    X_ = f"{fmt(sd * 0.985)}*fwd({H_})"
    sc.curve((X_, f"fy({X_},{H_})-0.004", H_), ('t', 0, 1), '#8a4a3c', f'jawLine{tag}', thickness=1)
