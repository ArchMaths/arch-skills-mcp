# Run inside generate.py (shares its names): body, outfit and props.
import numpy as np
from motifs import fern_curves, sprig_xy, emblem_curves, emblem_radii, emblem_fills, facing_curves, _emb_len, EMB_CENTRE, EMB_DOT

# ---------------- extra helpers ----------------
sc.blank()
sc.comment("Rounded box (superquadric): exponent e near 0 is boxy, 1 is an ellipsoid.")
sc.func('sqc(t,pw)', 'sign(cos(t))*abs(cos(t))^pw')
sc.func('sqs(t,pw)', 'sign(sin(t))*abs(sin(t))^pw')
sc.comment("Puffy capsule profile: like cr, plus a bulge bu in the middle.")
sc.func('cpf(u,r0,r1,bu)', '(r0+(r1-r0)*cf(u)+bu*sin(u)^2)*sqrt(sqrt(1-cos(u)^4))')


def rbox(C, half, fr, color, ref, e=0.25, e2=None):
    e2 = e if e2 is None else e2
    L = (f"{fmt(half[0])}*sqs(u,{fmt(e)})*sqc(v,{fmt(e2)})", f"{fmt(half[1])}*sqs(u,{fmt(e)})*sqs(v,{fmt(e2)})", f"{fmt(half[2])}*sqc(u,{fmt(e)})")
    sc.surface(place(L, C, fr), ('u', 0, 'pi'), ('v', 0, '2*pi'), color, ref, 'closed')


def puff(A, B, r0, r1, bu, color, ref, gather=0.0, ng=9, up=(0, 0, 1)):
    d = sub(B, A)
    fr = frame(d, up)
    R = f"cpf(u,{fmt(r0)},{fmt(r1)},{fmt(bu)})*(1+{fmt(gather)}*sin({ng}*w))"
    L = (f"{R}*cos(w)", f"{R}*sin(w)", f"{fmt(norm(d))}*cf(u)")
    sc.surface(place(L, A, fr), ('u', 0, 'pi'), ('w', 0, '2*pi'), color, ref, ('tube', 'u'))
    return fr


def torus(C, fr, R, r, color, ref, arc=(0, '2*pi')):
    L = (f"({fmt(R)}+{fmt(r)}*cos(v))*cos(u)", f"({fmt(R)}+{fmt(r)}*cos(v))*sin(u)", f"{fmt(r)}*sin(v)")
    sc.surface(place(L, C, fr), ('u', arc[0], arc[1]), ('v', 0, '2*pi'), color, ref, ('tube', 'u'))


def bead_tube(pts_fn_xyz, n_beads, r, color, ref, cross_a=(0, 1, 0), cross_b=(0, 0, 1)):
    """Beaded tube (a chain) along a centre line given as three Arch expressions in t (0..1)."""
    X, Y, Z = pts_fn_xyz
    R = f"{fmt(r)}*(0.45+0.55*abs(cos({n_beads}*pi*t)))"
    xyz = tuple(f"{[X, Y, Z][i]}+{R}*(cos(v)*{fmt(cross_a[i])}+sin(v)*{fmt(cross_b[i])})" for i in range(3))
    sc.surface(xyz, ('t', 0, 1), ('v', 0, '2*pi'), color, ref, ('tube', 'u'))


def flat_flower(C, n_dir, R, ref, color=ORANGE, center=GOLD, n=14, up=(0, 0, 1)):
    flower_funcs()
    fr = frame(n_dir, up)
    L = (f"fa(r,w,{n},{fmt(R)},0)", f"fb(r,w,{n},{fmt(R)},0)", f"fc(r,w,{n},{fmt(R)},0.02,0)")
    sc.surface(place(L, C, fr), ('r', 0, 1), ('w', 0, '2*pi'), color, ref, ('dir', fr[2]))
    if center:
        L = (f"{fmt(0.25 * R)}*r*cos(w)", f"{fmt(0.25 * R)}*r*sin(w)", f"{fmt(0.06 * R)}*(1-r^2)+{fmt(0.02 * R)}")
        sc.surface(place(L, C, fr), ('r', 0, 1), ('w', 0, '2*pi'), center, ref + 'Mid', ('dir', fr[2]))


SK_TOP, SK_HEM = 8.66, 1.61      # the pleated skirt, from the waist (under the corset) to its hem; the boots show below
SK_L = SK_TOP - SK_HEM
SA_EXPR = '0.43+0.0579*ln(1+exp((s-0.057)/0.02))'
DK_EXPR = '(fk(s)+gk(s))/2-(gk(s)-fk(s))/2*cos(t)'
FK_EXPR = '0.57-0.04*stp((s-0.5)/0.15)+0.07*stp((s-0.78)/0.14)'     # front: measured on the side view
GK_EXPR = '0.75+0.25*s'                                             # back


def SA(s):
    """Half-width of the skirt, s from the waist (0) to the hem (1): narrow inside the corset, then one straight
    line from the corset's edge to the hem (0.41 out per unit down), which the frill carries on to 3.35 at
    z 1.14, the sheet's widest. The cloth hangs loose: it neither bulges nor draws in toward the hem."""
    return 0.43 + 0.0579 * math.log(1 + math.exp((s - 0.057) / 0.02))


def _stp(q):
    q = min(1.0, max(0.0, q))
    return q * q * (3 - 2 * q)


def DK(s, th):
    """Depth of the skirt over its half-width at angle th from the front. Measured on the sheet's side view:
    the front hangs nearly flat, 0.57 of the half-width at the waist, 0.53 at mid-thigh and 0.60 toward the
    hem; the back is full, 0.75 at the waist to 1 at the hem."""
    f = 0.57 - 0.04 * _stp((s - 0.5) / 0.15) + 0.07 * _stp((s - 0.78) / 0.14)
    g = 0.75 + 0.25 * s
    return (f + g) / 2 - (g - f) / 2 * math.cos(th)


def sk_s(h):
    return (SK_TOP - h) / SK_L


# =====================================================================
#  TORSO, COLLAR, JABOT, ASCOT
# =====================================================================
TORSO = [(7.4, 0.58, 0.41), (8.0, 0.42, 0.32), (8.5, 0.38, 0.3), (9.1, 0.6, 0.44), (9.75, 0.64, 0.44),
         (10.3, 0.92, 0.4), (10.52, 0.4, 0.25), (10.62, 0.18, 0.18)]
_tz = [bezier_eval(TORSO, s) for s in np.linspace(0, 1, 400)]


def torso_ab(z):
    zs = [p[0] for p in _tz]
    i = int(np.argmin([abs(zz - z) for zz in zs]))
    return _tz[i][1], _tz[i][2]


BUST_D, BUST_Z, BUST_X, BUST_W = 0.24, 9.47, 0.28, 0.25      # depth, height and half-spacing of the breasts, their width


def bust(z, x=BUST_X):
    """How far the blouse stands out over the bust: two soft mounds with the cloth bridging between them,
    falling away gently above and rounding in more quickly underneath."""
    sg = 0.3 + 0.07 * math.atan((z - BUST_Z) / 0.1)
    return BUST_D * math.exp(-((z - BUST_Z) / sg) ** 2) * (0.45 + 0.55 * math.exp(-((abs(x) - BUST_X) / BUST_W) ** 2))


BUST_EXPR = (f"{fmt(BUST_D)}*exp(-(((zq)-{fmt(BUST_Z)})/(0.3+0.07*atan(((zq)-{fmt(BUST_Z)})/0.1)))^2)"
             f"*(0.45+0.55*exp(-((abs(xq)-{fmt(BUST_X)})/{fmt(BUST_W)})^2))")


def chest_y(x, z):
    a, b = torso_ab(z)
    c = math.sqrt(max(0.0, 1 - (x / a) ** 2))
    return -(b * c + bust(z, x) * c)


# The hanging line of the chest's front (see chg below)
_hz = np.linspace(10.6, 8.9, 171)
_hf = np.maximum.accumulate([max(-chest_y(x, z) for x in np.linspace(-0.45, 0.45, 31)) for z in _hz])
CHG_FIT = np.polyfit(_hz - 9.7, _hf, 7)[::-1]


def CHG(z):
    return float(sum(c * (z - 9.7) ** i for i, c in enumerate(CHG_FIT)))


TUCK = '#d6c8b0'                # the blouse's tucks
JABOT_X = -0.06                 # the cascade hangs a little toward her right, as the ascot does
# jabot tiers: top, height, half-width, stand-off, ruffle depth, folds
JABOT_TIERS = ((9.84, 0.32, 0.42, 0.03, 0.035, 13), (9.58, 0.28, 0.38, 0.05, 0.05, 12), (9.36, 0.27, 0.31, 0.045, 0.05, 11))
# ascot tails: name, top (x left, x right, z), outer corner, point, inner corner (x, z), stand-off, pleats
ASCOT_TAILS = (('Front', (-0.155, 0.02, 10.38), (-0.52, 9.72), (-0.11, 9.43), (-0.03, 9.56), 0.18, 2.5),
               ('Back', (0.17, 0.0, 10.38), (0.3, 9.72), (0.0, 9.47), (-0.06, 9.55), 0.13, 2.0))

if want('body'):
    sc.blank()
    sc.note('Blouse: body, high ruffled collar and jabot')
    tz, ta, tb = bezier_poly(TORSO)
    Z = horner(tz, 's')
    sc.func('tsa(s)', horner(ta, 's'))
    sc.func('tsb(s)', horner(tb, 's'))
    sc.func('tsz(s)', Z)
    sc.func('bst(xq,zq)', BUST_EXPR)
    sc.comment("The body under the blouse, with the bust (bst): two soft mounds, the cloth bridging between them.")
    sc.surface(("tsa(s)*cos(w)", "tsb(s)*sin(w)-bst(tsa(s)*cos(w),tsz(s))*max(0,-sin(w))", "tsz(s)"),
               ('s', 0, 1), ('w', 0, '2*pi'), CREAM, 'blouse', ('tube', 'u'))
    sc.comment("High collar with a ruffle on top.")
    sc.surface(("(0.235+0.012*s)*cos(w)", "0.02+(0.235+0.012*s)*sin(w)", "10.42+0.46*s"), ('s', 0, 1), ('w', 0, '2*pi'), CREAM, 'collar', ('tube', 'u'))
    sc.surface(("(0.25+0.12*s+0.03*s*sin(16*w))*cos(w)", "0.02+(0.25+0.12*s+0.03*s*sin(16*w))*sin(w)", "10.86+0.11*s+0.035*s*cos(16*w)"),
               ('s', 0, 1), ('w', 0, '2*pi'), CREAM, 'collarRuffle', ('point', (0, 0.02, 10.5)))
    # Fine tucks down the blouse over the bust (the close-up's thin vertical lines)
    s_of = lambda z: float(np.linspace(0, 1, 400)[int(np.argmin([abs(p[0] - z) for p in _tz]))])
    for sd, tag in ((1, 'L'), (-1, 'R')):
        for i in range(5):
            x = 0.3 + 0.06 * i
            c = f"sqrt(max(0,1-({fmt(x)}/tsa(s))^2))"
            sc.curve((fmt(sd * x), f"-(tsb(s)+bst({fmt(x)},tsz(s)))*{c}-0.006", "tsz(s)"), ('s', s_of(9.14), s_of(9.95 - 0.05 * i)), TUCK, f'blouseTuck{tag}{i + 1}', thickness=1)

    # What hangs in front of the chest hangs from its fullest point: chg(z) is how far forward a cloth hanging down
    # the front lies at height z (the chest's front above the bust, straight down below it).
    sc.func('chg(zq)', horner(list(CHG_FIT), '((zq)-9.7)'))
    sc.comment("""Jabot (rebuilt from a close-up): three ruffled tiers cascading from under the ascot down to the corset. Each tier's folds deepen toward its lower edge.""")
    sc.func('jtr(q)', "(2/pi)*asin(0.96*sin(q))")       # a soft zigzag
    for i, (zt, h, w, off, dep, n) in enumerate(JABOT_TIERS):
        Zj = f"({fmt(zt)}-{fmt(h)}*s*(1-0.3*p*p)*(1+0.06*jtr({n}*p)))"
        Xj = f"({fmt(JABOT_X)}+{fmt(w)}*p*(0.6+0.4*s)+0.02*s*jtr({n}*p+1.57))"
        sc.surface((Xj, f"-(chg({Zj})+{fmt(off)}+s*({fmt(dep * 0.6)}+{fmt(dep)}*jtr({n}*p)))", Zj), ('p', -1, 1), ('s', 0, 1),
                   CREAM if i % 2 == 0 else '#f7f0e4', f'jabot{i + 1}', ('dir', (0, -1, 0.2)))

    sc.blank()
    sc.note('Black ascot with a chrysanthemum brooch')
    sc.comment("""Ascot (rebuilt from a close-up): two black tails hang from under the brooch, pleated in a fan from the
knot. The front one swings toward her right and ends in a point; the other shows beside it on her left.
Each has a double gold line inside its edges and the front one a small sprig at its point.""")
    K = (0, -(CHG(10.5) + 0.07), 10.5)
    ellipsoid(add(K, (0, 0.03, -0.08)), (0.1, 0.06, 0.09), BLACK, 'ascotKnot')
    flower(add(K, (0, -0.03, 0.0)), unit((0, -1, 0.12)), 0.19, 'brooch')
    for nm_, (xl, xr, zt), (xA, zA), (xT, zT), (xI, zI), off, npl in ASCOT_TAILS:
        uT = (xT - xA) / (xI - xA)

        def tail(u, v, lift='0', xl=xl, xr=xr, zt=zt, xA=xA, zA=zA, zT=zT, xI=xI, zI=zI, off=off, npl=npl, uT=uT):
            zb = f"({fmt(zT)}+{fmt((zA - zT) / uT)}*max(0,{fmt(uT)}-{u})+{fmt((zI - zT) / (1 - uT))}*max(0,{u}-{fmt(uT)}))"
            Zt = f"({fmt(zt)}+({zb}-{fmt(zt)})*{v})"
            Xt = f"({fmt(xl)}+{fmt(xr - xl)}*{u}+({fmt(xA - xl)}+{fmt(xI - xA - xr + xl)}*{u})*{v})"
            return (Xt, f"-(chg({Zt})+{fmt(off * 0.45)}+{fmt(off * 0.55)}*{v}+0.022*{v}*cos({fmt(2 * math.pi * npl)}*{u})+{lift})", Zt)
        sc.surface(tail('u', 'v'), ('u', 0, 1), ('v', 0, 1), BLACK, f'ascot{nm_}', ('dir', (0, -1, 0.2)))
        for j, vv in enumerate((0.93, 0.885)):
            sc.curve(tail('t', fmt(vv), '0.006'), ('t', 0.05 + 0.03 * j, 0.95 - 0.03 * j), GOLD, f'ascot{nm_}Hem{j + 1}', thickness=1)
            for e_, uu in (('A', 0.05 + 0.03 * j), ('B', 0.95 - 0.03 * j)):
                sc.curve(tail(fmt(uu), 't', '0.006'), ('t', 0.3, vv), GOLD, f'ascot{nm_}Side{e_}{j + 1}', thickness=1)
    nm_, _, _, (xT, zT), _, off, _ = ASCOT_TAILS[0]
    Xs, Hs = sprig_xy(fmt(xT), fmt(zT + 0.1))
    sc.curve((Xs, f"-(chg({Hs})+{fmt(off + 0.035)})", Hs), ('t', 0, 2.999), GOLD, 'ascotSprig', thickness=2)

# =====================================================================
#  CAPELET: measured on the sheet (cape.py)
# =====================================================================
run('cape.py')        # always: the hair keeps clear of it; draws it if want('cape')

# =====================================================================
#  ARMS: sleeves and hands, measured on the sheet (arms.py)
# =====================================================================
run('arms.py')        # always: the book and the hair use the arms' geometry; draws them if want('arms')

# =====================================================================
#  BOOK (held in her left hand)
# =====================================================================
def place_book():
    # The book hangs on the outside of her left hand, its cover facing out and forward and leaning out with
    # the skirt, pushed out until it clears the skirt.
    palm = HAND_PALM['L']
    o = unit((0.88, -0.48, 0))
    up = unit(add((0, 0, 1), mul(o, -0.42)))
    cover_n = unit(add(o, (0, 0, 0.42)))
    across = unit(cross(up, cover_n))       # along the book width, toward the spine side first
    Bc = add(add(palm, mul(cover_n, hand_reach('L', cover_n) + 0.095)), mul(up, -0.36))   # back cover against the hand

    def clear(P):
        s = sk_s(P[2])
        a = SA(s) + 0.1
        th = math.atan2(P[0], -P[1])
        return math.hypot(P[0] / a, P[1] / (DK(s, th) * a))
    box = [(i, j, k) for i in (-1, 0, 1) for j in (-1, -0.5, 0, 0.5, 1) for k in (-1, 1)]
    for _ in range(100):
        pts_ = [add(add(add(Bc, mul(across, 0.38 * i)), mul(up, 0.52 * j)), mul(cover_n, 0.09 * k)) for i, j, k in box]
        if min(clear(P) for P in pts_) >= 1.02:
            break
        Bc = add(Bc, mul(o, 0.01))
    fr = (across, up, cover_n)
    return Bc, fr, pts_


BOOK = place_book()      # also kept clear by the hair
if want('book'):
    sc.blank()
    sc.note('Leather book with gold corners and a chrysanthemum emblem')
    Bc, (across, up, cover_n), _ = BOOK
    fr = (across, up, cover_n)
    rbox(Bc, (0.38, 0.52, 0.09), fr, BROWN, 'bookCover', e=0.12)
    rbox(add(Bc, mul(across, -0.014)), (0.38, 0.51, 0.074), fr, '#efe3c8', 'bookPages', e=0.1)
    front = add(Bc, mul(cover_n, 0.092))
    L = ("(0.32*sqc(w,0.08)*r)", "(0.46*sqs(w,0.08)*r)", "0")
    sc.surface(place(L, front, fr), ('r', 0.94, 1), ('w', 0, '2*pi'), GOLD, 'bookFrame', ('dir', cover_n))
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1))):
        C = add(add(front, mul(across, 0.345 * sx)), mul(up, 0.49 * sy))
        rbox(C, (0.055, 0.055, 0.01), fr, GOLD, f'bookCorner{i + 1}', e=0.2)
    flower(add(front, mul(cover_n, 0.004)), cover_n, 0.19, 'bookEmblem', layers=2, flat=True)
    ribbon([add(Bc, add(mul(up, -0.5), mul(across, -0.05))), add(Bc, add(mul(up, -0.72), mul(across, -0.02))), add(Bc, add(mul(up, -0.9), mul(across, 0.02)))],
           0.03, ORANGE, 'bookmark', cover_n, vcut=0.3)

# =====================================================================
#  CORSET with gold buttons, chain, ring and tassel
# =====================================================================
CZB = 8.24          # the corset's lower edge at the sides of the points; the sheet's corset ends just above the hips
zz = np.linspace(7.55, 9.35, 60)
ca_fit = np.polyfit(zz - 8.5, [torso_ab(z)[0] + 0.045 for z in zz], 4)[::-1]
cb_fit = np.polyfit(zz - 8.5, [torso_ab(z)[1] + 0.04 for z in zz], 4)[::-1]


def corset_ab(z):
    fl = 0.32 * max(0, (CZB - z) / 0.3) ** 1.5
    a = sum(c * (z - 8.5) ** i for i, c in enumerate(ca_fit)) + fl
    b = sum(c * (z - 8.5) ** i for i, c in enumerate(cb_fit)) + 0.6 * fl
    return a, b


def corset_front(x, z, off=0.0):
    a, b = corset_ab(z)
    return -(b * math.sqrt(max(0.0, 1 - (x / a) ** 2))) - off


if want('corset'):
    sc.blank()
    sc.note('Black corset: gold buttons and piping, chain, ring and tassel')
    sc.func('ca(zq)', f"{horner(list(ca_fit), '(zq-8.5)')}+0.32*max(0,({fmt(CZB)}-zq)/0.3)^1.5")
    sc.func('cb(zq)', f"{horner(list(cb_fit), '(zq-8.5)')}+0.19*max(0,({fmt(CZB)}-zq)/0.3)^1.5")
    sc.comment("Bottom edge: points at the front, the sides and the back.")
    sc.func('czb(w)', f'{fmt(CZB)}-0.32*max(0,-sin(w))^10-0.2*cos(w)^10-0.14*max(0,sin(w))^10')
    Zx = "(czb(w)+(9.12-czb(w))*h)"
    sc.surface((f"ca({Zx})*cos(w)", f"cb({Zx})*sin(w)", Zx), ('h', 0, 1), ('w', 0, '2*pi'), BLACK, 'corset', ('tube', 'v'))
    for nm, Ze in (('corsetPipingTop', '9.12'), ('corsetPipingBottom', 'czb(w)')):
        sc.surface((f"(ca({Ze})+0.018*cos(v))*cos(w)", f"(cb({Ze})+0.018*cos(v))*sin(w)", f"{Ze}+0.018*sin(v)"),
                   ('w', 0, '2*pi'), ('v', 0, '2*pi'), GOLD, nm, ('tube', 'u'))
    for i, z in enumerate((8.45, 8.7, 8.95)):
        for side, tag in ((1, 'L'), (-1, 'R')):
            x = 0.13 * side
            ellipsoid((x, corset_front(x, z, 0.005), z), (0.036, 0.036, 0.022), GOLD, f'button{tag}{i + 1}', fr=((1, 0, 0), (0, 0, 1), (0, -1, 0)))
    # gold chain swag from her right side to a ring on her left
    ring = (0.36, corset_front(0.36, 8.36, 0.035), 8.36)
    xs = np.linspace(-0.42, 0.36, 40)
    zs = 8.84 - 0.36 * np.sin(np.pi * (xs + 0.42) / 0.78) ** 1.1 + (8.36 + 0.06 - 8.84) * (xs + 0.42) / 0.78 * 1.0
    ys = [corset_front(x, z, 0.045) for x, z in zip(xs, zs)]
    tt = (xs + 0.42) / 0.78
    px_ = np.polyfit(tt, xs, 3)[::-1]
    py_ = np.polyfit(tt, ys, 6)[::-1]
    pz_ = np.polyfit(tt, zs, 6)[::-1]
    bead_tube((horner(list(px_), 't'), horner(list(py_), 't'), horner(list(pz_), 't')), 26, 0.017, GOLD, 'corsetChain')
    torus(ring, ((1, 0, 0), (0, 0, 1), (0, -1, 0)), 0.065, 0.015, GOLD, 'corsetRing')
    tassel((ring[0], ring[1] - 0.005, ring[2] - 0.08), 0.55, GOLD, 'corsetTassel', r=0.045)
    flat_flower((0, corset_front(0, 8.1, 0.01), 8.1), unit((0, -1, 0.1)), 0.06, 'corsetFlower', n=10, center=GOLD_DK)

# =====================================================================
#  SKIRT, after the sheet's three views and its skirt panel (三片式长裙与褶边衬裙):
#  a pleated cream skirt with a frilled petticoat; black panels at the sides; over them a cream overskirt
#  in two side pieces, each caught up by a bow at the front and one at the back so its hem swags low at
#  the side; black bands down the overskirt's edges to the bows; an orange tab at the front and the back.
# =====================================================================
def skirt_point(h, th, off=0.0):
    """Point at height h and angle th (from the front, + toward her left) on the pleated skirt, off further out."""
    s = sk_s(h)
    a = SA(s) + off
    return (a * math.sin(th), -a * DK(s, th) * math.cos(th), h)


def skirt_normal(h, th):
    s = sk_s(h)
    k = DK(s, th)
    return unit((k * math.sin(th), -math.cos(th), 0.45 * math.hypot(k * math.sin(th), math.cos(th))))


# Arch lights a surface facing the viewer at about 1.2 times its colour, so anything brighter than 210 clips to
# flat white or flat orange and loses its shading. The pleated skirt and the orange cloth are given colours
# 1/1.2 of the sheet's (#d3633a for the cloth), so their folds shade.
RUST, RUST_DK, RUST_LT = '#b65632', '#863a21', '#c9713f'   # the dress's orange cloth, its creases, its turned-back side
SKIRT_CREAM = '#ddd3c2'
BORDER_CREAM, FERN = '#d9caa9', '#c45c36'     # the cream border woven into the orange cloth; the fern's thread


# The skirt's fern and the orange cloth's sprig are in motifs.py, measured on close-up photos (python3 motifs.py
# draws the fern in 2D over its photo).
PANEL_GOLD, PANEL_LINE, PANEL_PALE = '#a8855d', '#bfa982', '#bca377'     # the black panels' emblem and border lines (1/1.2 of the sheet's)
PLEAT_SHADE = '#9c8a72'         # the drawn line in each pleat
N_PLEAT = 40
OV_F, OV_B = 0.6, 2.47          # the overskirt pieces run from 34 degrees (front bows) to 141.5 (back bows)
OV_TOP = 8.2                    # their top, under the corset's lower edge
BOW_ZF, BOW_ZB = 5.55, 5.5      # heights of the front and back bows' knots, measured on the sheet
SIDE_EMBLEMS = ((2.14, 2.75, False), (1.84, 2.84, True))   # |x| and height of the circled emblems: front view, back view
# The black panels' big fold, measured on a close-up of the side view (see the panel's comment)
PF_TH, PF_LEAN = 0.788, 0.0917  # the fold line's angle at height 2, and how far it turns forward per unit up
PF_TOP = 4.6                    # the fold fades out by here, toward the bow
PF_SPAN, PF_A = 0.36, 0.3       # width of cloth taken up by the fold; how far it doubles back (q+A*sin(2 pi q))
PF_X0, PF_QE = 0.407, 0.661     # where the outer turn lies within the span, and its q
PF_D = 0.07                     # how far the lapping part stands off the strip
PF_MIN = 0.54                   # the fold stays behind the panel's front edge
PF_HEM = 2.07                   # the front strip's hem (the lapping part hangs to 1.98 at the fold)
PF_Q0 = 0.11                    # the main part's hem starts here on the old hem line bzh
PF_RIDGES, PF_RW, PF_RD = (0.2, 0.36, 0.52, 0.68), 0.03, 0.035   # soft folds across the main part: places, width, depth at the hem


def panel_off(h, th):
    """Stand-off of the black panel's main part at height h and angle th (for what is sewn on it)."""
    g = min(1.0, max(0.0, (PF_TOP - h) / 1.6)) ** 2
    span = PF_SPAN * g / (SA(sk_s(h)) + 0.09)
    tb = max(PF_MIN, PF_TH - PF_LEAN * (h - 2) - PF_X0 * span) + span
    q = min(1.0, max(0.0, (abs(th) - tb) / (2.66 - tb)))
    rid = sum(math.exp(-((q - v) / PF_RW) ** 2) for v in PF_RIDGES)
    return 0.09 + PF_D * g * math.exp(-4 * q) + PF_RD * min(1.0, max(0.0, (5 - h) / 3)) * rid


# The overskirt's hem, measured on its lower edge as traced over the front and back views (|x| -> angle on the
# overskirt): (degrees from the front, height). It hangs from the front bow down the piece's front edge to
# 4.3, stays high (3.9 at 60 degrees) before dropping steeply to the side, is lowest, 2.95, at the side, and
# rises up the back to 4.5 under the back bow; the lining shows where the edge curls.
OZH_PTS = [(34.4, 4.3), (42, 4.12), (48, 4.05), (52, 4.0), (57, 3.95), (61, 3.82), (64, 3.5), (67, 3.2), (75, 3.02), (95, 2.95),
           (110, 3.0), (118, 3.1), (122.5, 3.3), (127, 3.5), (131, 3.78), (134, 4.0), (137.5, 4.17), (139.5, 4.32), (141.5, 4.5)]
_OZK = [(math.radians(a) - OV_F) / (OV_B - OV_F) for a, _ in OZH_PTS]       # q of each point


def _ozh_fit():
    """Natural cubic spline through the points: a + b q + sum c_i max(0, q - q_i)^3, straight past the ends."""
    q = np.array(_OZK)
    z = np.array([h for _, h in OZH_PTS])
    n = len(q)
    A = np.zeros((n + 2, n + 2))
    A[:n, 0], A[:n, 1] = 1, q
    A[:n, 2:] = np.maximum(0, q[:, None] - q[None, :]) ** 3
    A[n, 2:], A[n + 1, 2:] = 1, q
    return np.linalg.solve(A, np.r_[z, 0, 0])


OZH_C = _ozh_fit()


def ozh(q):
    return OZH_C[0] + OZH_C[1] * q + sum(c * max(0.0, q - k) ** 3 for c, k in zip(OZH_C[2:], _OZK))


def OSR(h):
    """Radius of the overskirt at height h (before its folds)."""
    return SA(sk_s(h)) + 0.13 * min(1, max(0, OV_TOP - h) / 0.3)


if want('skirt'):
    sc.blank()
    sc.note('Skirt: cream pleated skirt with orange embroidery and a frilled petticoat')
    sc.comment(f"""Skirt: s from the waist (z={fmt(SK_TOP)}) to the hem (z={fmt(SK_HEM)}); t is the angle from the front, + toward her
left. sa(s) is its half-width: one straight line from the corset to the hem. Seen from the side the front hangs
nearly flat and the back is full, so the depth is dk(s,t) times the half-width: 0.57 at the front, 0.75
to 1 at the back from the waist to the hem. 40 knife pleats (spl), 9 degrees apart as traced: each rises slowly
to its edge and drops back sharply, cut into the skirt (the edges stay on sa) and deepening to 5% of the
radius at the hem. Arch meshes a surface with at most 60 steps a side, too few for 40 pleats, so the skirt is
built in ten sectors of 3.5 to 4.4 pleats each; a thin line is drawn in each pleat where the skirt shows.
sz(h) is s at height h.""")
    sc.func('sa(s)', SA_EXPR)
    sc.func('stp(q)', 'min(1,max(0,q))^2*(3-2*min(1,max(0,q)))')
    sc.func('fk(s)', FK_EXPR)
    sc.func('gk(s)', GK_EXPR)
    sc.func('dk(s,t)', DK_EXPR)
    sc.func('sz(h)', f"({fmt(SK_TOP)}-h)/{fmt(SK_L)}")
    sc.func('spl(s,t)', '1-0.05*(0.2+0.8*s)*(0.5+0.645*atan(0.7*sin(40*t)/(1-0.7*cos(40*t))))')
    Zs = f"{fmt(SK_TOP)}-{fmt(SK_L)}*s"
    def on_skirt(sx, tx, lift=1.012):
        return (f"sa({sx})*spl({sx},{tx})*{fmt(lift)}*sin({tx})", f"-sa({sx})*spl({sx},{tx})*dk({sx},{tx})*{fmt(lift)}*cos({tx})", f"{fmt(SK_TOP)}-{fmt(SK_L)}*({sx})")

    SEC_F, SEC_B = 0.56, 2.62      # the skirt shows inside +-SEC_F at the front and beyond +-SEC_B at the back
    _cuts = [-SEC_F, 0, SEC_F] + [SEC_F + (SEC_B - SEC_F) * k / 3 for k in (1, 2, 3)] + [math.pi, 2 * math.pi - SEC_B] \
        + [2 * math.pi - SEC_B + (SEC_B - SEC_F) * k / 3 for k in (1, 2, 3)]
    for k, (t0, t1) in enumerate(zip(_cuts[:-1], _cuts[1:])):
        sc.surface(("sa(s)*spl(s,t)*sin(t)", "-sa(s)*spl(s,t)*dk(s,t)*cos(t)", Zs), ('s', 0, 1), ('t', t0, t1), SKIRT_CREAM, f'underskirt{k + 1}', ('point', (0, 0, 4)))
    assert abs(_cuts[-1] - (2 * math.pi - SEC_F)) < 1e-9
    # a line in each pleat (at its deepest, 0.795/40 past the pleat's edge) across the front and the back
    for k in range(-N_PLEAT // 2, N_PLEAT // 2):
        tk = (math.acos(0.7) + 2 * math.pi * k) / N_PLEAT
        if abs(tk) < SEC_F or abs(tk) > SEC_B:
            sc.curve(on_skirt("t", fmt(tk), 1.01), ('t', 0.04, 1), PLEAT_SHADE, f'pleat{k + N_PLEAT // 2 + 1}', thickness=1)
    # The frill: two loose layers hanging from the hem in deep folds (36 and 48 round, 0.14 and 0.16 deep at the
    # bottom), their crests on the skirt's line carried on to 3.35 at z 1.14. Twelve sectors each, so that Arch's
    # mesh (60 steps a side at most) has 3 or 4 folds to resolve.
    for k, (z0, z1, r0, r1, amp, nw, col) in enumerate([(1.66, 1.32, 0.99, 0.993, 0.045, 36, CREAM), (1.4, 1.14, 0.985, 1.011, 0.05, 48, '#f8f2e8')]):
        Rk = f"sa(1)*({fmt(r0)}+{fmt(r1 - r0)}*s+{fmt(amp)}*s*sin({nw}*t))"
        for j in range(12):
            sc.surface((f"{Rk}*sin(t)", f"-{Rk}*dk(1,t)*cos(t)", f"{fmt(z0)}-{fmt(z0 - z1)}*s+0.03*s*cos({nw}*t)"),
                       ('s', 0, 1), ('t', -math.pi + j * math.pi / 6, -math.pi + (j + 1) * math.pi / 6), col, f'petticoat{k + 1}s{j + 1}', ('point', (0, 0, 4)))
    sc.comment("Embroidery where the pleated skirt shows, at the front and the back: hem lines and a row of ferns in orange thread.")
    sc.comment("The fern's parts as functions of the curve parameter: x across and h up from its foot (motifs.py).")
    FERN_PARTS = []
    for i_, (part, (X, H), T) in enumerate(fern_curves()):
        fx_, fh_ = f"fn{'abcdef'[i_]}x", f"fn{'abcdef'[i_]}h"
        sc.func(f'{fx_}(t)', X)
        sc.func(f'{fh_}(t)', H)
        FERN_PARTS.append((part, (f"{fx_}(t)", f"{fh_}(t)"), T))
    for side, nm, c0, hw in (('Front', 'F', 0.0, 0.36), ('Back', 'B', math.pi, 0.4)):
        sc.curve(on_skirt("0.955", f"({fmt(c0)}+t)"), ('t', -hw, hw), GOLD, f'hemLine{nm}1', thickness=2)
        sc.curve(on_skirt("0.94", f"({fmt(c0)}+t)"), ('t', -hw, hw), FERN, f'hemLine{nm}2', thickness=2)
        # ferns side by side along the hem, their feet on the upper hem line
        s_f = 0.935
        wid = SA(s_f - 0.06) * 2 * hw
        n_f = max(1, int(wid / 0.43))
        for i in range(n_f):
            tc = c0 + (hw - 0.215 / SA(s_f - 0.06)) * ((2 * i / (n_f - 1) - 1) if n_f > 1 else 0)
            for part, (X, H), T in FERN_PARTS:
                # (the pleats' depth and the skirt's depth-to-width ratio are taken at the fern's middle height: Arch drops
                # a curve whose expression grows too long)
                sx, tx = f"({fmt(s_f)}-{H}/{fmt(SK_L)})", f"({fmt(tc)}+{X}/{fmt(SA(s_f - 0.06))})"
                Rf = f"sa({sx})*spl({fmt(s_f - 0.06)},{tx})*1.018"
                sc.curve((f"{Rf}*sin({tx})", f"-{Rf}*dk({fmt(s_f - 0.06)},{tx})*cos({tx})", f"{fmt(SK_TOP)}-{fmt(SK_L)}*{sx}"), ('t',) + T, FERN, f'fern{nm}{i + 1}{part}', thickness=2)

    sc.blank()
    sc.note('Skirt: black side panels with orange facings, gold bands and chrysanthemum emblems')
    sc.comment("""Black side panels under the overskirt, from 29-31 degrees to 152 on each side (measured on the front
and back views; the pleated skirt shows between them at the front and the back), from just under the bows
down to a hem near 2, lying on the skirt (the cloth hangs: nothing holds it out). The front edge is faced in orange: a thin edge on
her right (as traced, one line at the panel's edge), 9 degrees on her left, a wide strip below the front tab. q runs across a panel from
its front edge, p down it.""")
    sc.func('bzh(q)', '1.95+0.1*sin(pi*q)')
    sc.func('eml(k)', _emb_len('k'))
    sc.func('emi(w)', emblem_radii('w')[0])
    sc.func('emo(w)', emblem_radii('w')[1])
    sc.func('bph(q,p)', '5.45-0.1*q-(5.45-0.1*q-bzh(q))*p')
    # The big fold (measured on a close-up of the side view): a line from under the front bow to the hem, 0.41 in
    # front of the emblem at the emblem's height and leaning 0.2 per unit of height. The cloth really folds there:
    # it runs back from the front edge, turns out and forward, then turns again and runs on to the back, so the
    # emblem's part laps over a narrow front strip whose hem ends higher. The fold fades out toward the bow.
    sc.func('pfg(h)', f"min(1,max(0,({fmt(PF_TOP)}-h)/1.6))^2")
    sc.func('pff(h)', f"{fmt(PF_TH)}-{fmt(PF_LEAN)}*(h-2)")
    sc.func('pfs(h)', f"{fmt(PF_SPAN)}*pfg(h)/(sa(sz(h))+0.09)")
    sc.func('pfa(h)', f"max({fmt(PF_MIN)},pff(h)-{fmt(PF_X0)}*pfs(h))")
    sc.func('pfb(h)', "pfa(h)+pfs(h)")
    sc.func('pfm(q)', "q*q*(3-2*q)")
    sc.func('pfr(q)', '+'.join(f"exp(-(((q)-{fmt(v)})/{fmt(PF_RW)})^2)" for v in PF_RIDGES))
    sc.func('pfd(h)', f"{fmt(PF_RD)}*min(1,max(0,(5-h)/3))")
    TB1 = 2.66
    for sd, tag in ((1, 'L'), (-1, 'R')):
        F0, TB0 = {1: (0.38, 0.54), -1: (0.48, 0.50)}[sd]     # facing from F0, black from TB0

        def ppt(th, ro, h, sd=sd):
            r = f"(sa(sz({h}))+{ro})"
            return (f"{fmt(sd)}*{r}*sin({th})", f"-{r}*dk(sz({h}),{th})*cos({th})", h)

        # the panel's pieces: name, q range, angle(q, h), stand-off(q, h), hem(q), top(q)
        qm = f"({fmt(PF_Q0)}+{fmt(1 - PF_Q0)}*q)"
        pieces = [('Strip', (0, 1), lambda h, TB0=TB0: f"({fmt(TB0)}+(pfa({h})-{fmt(TB0)})*q)", lambda h: "0.09", fmt(PF_HEM), "5.45"),
                  ('Fold', (0, 1), lambda h: f"(pfa({h})+pfs({h})*(q+{fmt(PF_A)}*sin(2*pi*q)))", lambda h: f"(0.09+{fmt(PF_D)}*pfg({h})*pfm(q))",
                   f"({fmt(PF_HEM)}+{fmt(1.95 + 0.1 * math.sin(math.pi * PF_Q0) - PF_HEM)}*pfm(q))", "5.45")]
        for j in range(3):
            pieces.append((f'Main{j + 1}', (j / 3, (j + 1) / 3), lambda h: f"(pfb({h})+({fmt(TB1)}-pfb({h}))*q)",
                           lambda h: f"(0.09+{fmt(PF_D)}*pfg({h})*exp(-4*q)+pfd({h})*pfr(q))", f"bzh({qm})", f"(5.45-0.1*{qm})"))
        for nm_, qr, th_, ro_, hem_, top_ in pieces:
            H = f"({top_}-({top_}-{hem_})*p)"
            sc.surface(ppt(th_(H), ro_(H), H), ('q',) + qr, ('p', 0, 1), BLACK, f'sidePanel{tag}{nm_}', ('point', (0, 0, 4)))
        # the lining shows as a thin dark-red line down the fold's outer edge
        Hl = f"({fmt(PF_TOP - 0.2)}-{fmt(PF_TOP - 0.2 - 2.0)}*t)"
        ql = fmt(PF_QE)
        sc.curve(ppt(pieces[1][2](Hl).replace('q', ql), pieces[1][3](Hl).replace('pfm(q)', f"pfm({ql})") + "+0.004", Hl), ('t', 0, 1), RUST_DK, f'sidePanel{tag}FoldEdge', thickness=1)
        # The front edge turned back: an orange facing. On her right a thin edge; on her left a flat strip whose foot
        # runs diagonally, 0.16 lower at its front edge than where it meets the panel (measured on a close-up, with
        # its border: a cream band 0.035 wide on the front edge and the foot, a thin line 0.07 inside them) and
        # carries a leafy cross and a sprig in cream thread.
        TF = f"({fmt(F0 - 0.01)}+{fmt(TB0 - F0 + 0.02)}*q)"
        wide = TB0 - F0 > 0.1
        FOOT0, RISE = 1.90, 0.16
        HF = f"(5.45-({fmt(5.45 - FOOT0)}-{fmt(RISE)}*q)*p)" if wide else "bph(0,p)"
        FB = "(0.5-0.5*cos(4*pi*{0}))*(0.2+0.8*{1})" if wide else "0*{0}*{1}"
        R = f"(sa(sz({HF}))+0.105+0.02*{FB.format('q', 'p')})"
        sc.surface((f"{fmt(sd)}*{R}*sin({TF})", f"-{R}*dk(sz({HF}),{TF})*cos({TF})", HF), ('q', 0, 1), ('p', 0, 1), RUST, f'sideFacing{tag}', ('point', (0, 0, 4)))
        if wide:
            arc = (TB0 - F0 + 0.02) * SA(sk_s(2.6))          # the strip's width, to turn widths into q
            def on_facing(q_, p_, lift):
                h_ = f"(5.45-({fmt(5.45 - FOOT0)}-{fmt(RISE)}*{q_})*{p_})"
                r_ = f"(sa(sz({h_}))+{fmt(0.125 + lift)})"
                t_ = f"({fmt(F0 - 0.01)}+{fmt(TB0 - F0 + 0.02)}*{q_})"
                return (f"{fmt(sd)}*{r_}*sin({t_})", f"-{r_}*dk(sz({h_}),{t_})*cos({t_})", h_)
            wq, wp = 0.035 / arc, 0.035 / (5.45 - FOOT0)
            sc.surface(on_facing(f"({fmt(wq)}*q)", 'p', 0.004), ('q', 0, 1), ('p', 0, 1), BORDER_CREAM, f'sideFacingBorder{tag}1', ('point', (0, 0, 4)))
            sc.surface(on_facing('q', f"(1-{fmt(wp)}*p)", 0.004), ('q', 0, 1), ('p', 0, 1), BORDER_CREAM, f'sideFacingBorder{tag}2', ('point', (0, 0, 4)))
            c1, c2 = "min(1,max(0,t))", "min(1,max(0,t-1))"
            sc.curve(on_facing(f"({fmt(0.085 / arc)}+{fmt(1 - 0.085 / arc)}*{c2})", f"({fmt(1 - 3 * wp)}*{c1})", 0.008), ('t', 0, 2), BORDER_CREAM, f'sideFacingLine{tag}', thickness=1)
            q0, hup = 0.36, 0.17                              # the cross: across the strip, above the foot
            tq = F0 - 0.01 + (TB0 - F0 + 0.02) * q0
            hb = FOOT0 + RISE * q0 + hup
            for part, (X, H), T in facing_curves():
                Hl = f"({fmt(hb)}+{H})"
                Rl = f"(sa(sz({Hl}))+0.135)"
                Tl = f"({fmt(tq)}+{X}/{fmt(SA(sk_s(hb)))})"
                sc.curve((f"{fmt(sd)}*{Rl}*sin({Tl})", f"-{Rl}*dk(sz({Hl}),{Tl})*cos({Tl})", Hl), ('t',) + T, BORDER_CREAM, f'sideFacingCross{tag}{part}', thickness=2)
            for i_, (qq, dh) in enumerate(((0.78, 0.11),)):
                tq = F0 + (TB0 - F0) * qq
                hb = FOOT0 + RISE * qq + dh
                Xl, Hl = sprig_xy('0', fmt(hb))
                Rl = f"(sa(sz({Hl}))+0.135)"
                Tl = f"({fmt(tq)}+{Xl}/{fmt(SA(sk_s(hb)))})"
                sc.curve((f"{fmt(sd)}*{Rl}*sin({Tl})", f"-{Rl}*dk(sz({Hl}),{Tl})*cos({Tl})", Hl), ('t', 0, 2.999), BORDER_CREAM, f'sideFacingSprig{tag}{i_ + 1}', thickness=2)
            tq = F0 - 0.01 + (TB0 - F0 + 0.02) * 0.97
            Hc = f"(5.45-{fmt(5.45 - FOOT0 - RISE * 0.97)}*t)"
            Rc = f"(sa(sz({Hc}))+0.128)"
            sc.curve((f"{fmt(sd)}*{Rc}*sin({fmt(tq)})", f"-{Rc}*dk(sz({Hc}),{fmt(tq)})*cos({fmt(tq)})", Hc), ('t', 0, 1), GOLD, f'sideFacingTrimB{tag}', thickness=2)
        sc.curve((f"{fmt(sd)}*(sa(sz(bph(0,t)))+0.11)*sin({fmt(F0)})", f"-(sa(sz(bph(0,t)))+0.11)*dk(sz(bph(0,t)),{fmt(F0)})*cos({fmt(F0)})", "bph(0,t)"),
                 ('t', 0, 1), GOLD, f'sideFacingTrim{tag}', thickness=2)
        # The hem's border, measured across the hem on a close-up: a pale gold line 0.05 wide on the edge, black
        # for 0.07, a second line 0.05 wide (centred 0.145 up) and a faint thin line 0.25 up.
        for nm_, qr, th_, ro_, hem_, top_ in pieces:
            for k, (h0, h1) in enumerate(((0.0, 0.05), (0.12, 0.17))):
                Hq = f"({hem_}+{fmt(h0)}+{fmt(h1 - h0)}*p)"
                sc.surface(ppt(th_(Hq), ro_(Hq) + "+0.008", Hq), ('q',) + qr, ('p', 0, 1), PANEL_LINE, f'sideBand{tag}{k + 1}{nm_}', ('point', (0, 0, 4)))
            Hq = f"({hem_}+0.25)"
            sc.curve(tuple(re.sub(r'\bq\b', 't', e_) for e_ in ppt(th_(Hq), ro_(Hq) + "+0.01", Hq)), ('t',) + qr, PANEL_GOLD, f'sideBand{tag}3{nm_}', thickness=1)
        # The border also runs up the panel's front and back edges (close-up): the pale band on the edge and the
        # thin line 0.11 inside it; a small sprig sits in each lower corner.
        for e_, (te, dirn, qe) in enumerate(((TB0, 1, 0), (TB1, -1, 1))):
            hem_e = PF_HEM if e_ == 0 else 1.95
            He = f"({fmt(5.45 - 0.1 * qe)}-{fmt(5.45 - 0.1 * qe - hem_e)}*p)"
            Re = f"(sa(sz({He}))+0.098)"
            TE = f"({fmt(te)}+{fmt(dirn * 0.05)}*v/{Re})"
            sc.surface((f"{fmt(sd)}*{Re}*sin({TE})", f"-{Re}*dk(sz({He}),{TE})*cos({TE})", He), ('p', 0, 1), ('v', 0, 1), PANEL_LINE, f'sideEdge{tag}{e_ + 1}', ('point', (0, 0, 4)))
            Ht = f"({fmt(5.45 - 0.1 * qe)}-{fmt(5.45 - 0.1 * qe - hem_e - 0.25)}*t)"
            Rt = f"(sa(sz({Ht}))+0.1)"
            TT = f"({fmt(te)}+{fmt(dirn * 0.11)}/{Rt})"
            sc.curve((f"{fmt(sd)}*{Rt}*sin({TT})", f"-{Rt}*dk(sz({Ht}),{TT})*cos({TT})", Ht), ('t', 0, 1), PANEL_GOLD, f'sideEdge{tag}{e_ + 1}Line', thickness=1)
            hb = hem_e + 0.30
            Xl, Hl = sprig_xy('0', fmt(hb))
            Rl = f"(sa(sz({Hl}))+0.1)"
            Tl = f"({fmt(te)}+({fmt(dirn * 0.2)}+{Xl})/{fmt(SA(sk_s(hb)))})"
            sc.curve((f"{fmt(sd)}*{Rl}*sin({Tl})", f"-{Rl}*dk(sz({Hl}),{Tl})*cos({Tl})", Hl), ('t', 0, 2.999), PANEL_GOLD, f'sideEdge{tag}{e_ + 1}Sprig', thickness=2)
        # The emblems where the sheet's circles are (the front pair on the front view, the back pair on the back):
        # a flower of pointed petals round a ringed centre, a bud above, a spray below and a twig and a dot to each
        # side (motifs.py, measured on the user's tracing of a close-up). The petals are filled in eight sectors.
        for k, (x_, h, back) in enumerate(SIDE_EMBLEMS):
            th = math.asin(min(1.0, x_ / (SA(sk_s(h)) + 0.105)))
            th = sd * (math.pi - th if back else th)
            C, n_ = skirt_point(h, th, panel_off(h, th) + 0.035), skirt_normal(h, th)      # flat, so 0.035 off the curved panel to clear it at its tips
            e_up = unit(sub((0, 0, 1), mul(n_, n_[2])))
            e_bk = cross(e_up, n_)
            if e_bk[1] < 0:
                e_bk = mul(e_bk, -1)
            fr = (e_bk, e_up, n_)
            for j in range(8):
                L = ("(emi(w)+(emo(w)-emi(w))*r)*cos(w)", "(emi(w)+(emo(w)-emi(w))*r)*sin(w)", "0")
                sc.surface(place(L, C, fr), ('r', 0, 1), ('w', math.pi / 2 + j * math.pi / 4, math.pi / 2 + (j + 1) * math.pi / 4), PANEL_GOLD, f'sideEmblem{tag}{k + 1}s{j + 1}', ('dir', n_))
            for nm_, rad, lift, col in (('Mid', EMB_CENTRE, 0.004, PANEL_GOLD), ('Dot', EMB_DOT, 0.008, PANEL_PALE)):
                sc.surface(place((f"{fmt(rad)}*r*cos(w)", f"{fmt(rad)}*r*sin(w)", fmt(lift)), C, fr), ('r', 0, 1), ('w', 0, '2*pi'), col, f'sideEmblem{tag}{k + 1}{nm_}', ('dir', n_))
            for nm_, (X, H) in emblem_fills():
                sc.surface(place((X, H, '0'), C, fr), ('u', 0, 1), ('v', -1, 1), PANEL_GOLD, f'sideEmblem{tag}{k + 1}{nm_}Fill', ('dir', n_))
            for part, (X, H), T in emblem_curves():
                sc.curve(place((X, H, '0.01'), C, fr), ('t',) + T, BLACK if part == 'Ring' else PANEL_GOLD, f'sideEmblem{tag}{k + 1}{part}', thickness=2)

    sc.blank()
    sc.note('Skirt: cream overskirt in two side pieces, caught up by bows at the front and the back')
    sc.comment(f"""Overskirt: on each side a cream piece from {fmt(round(math.degrees(OV_F)))} to {fmt(round(math.degrees(OV_B)))} degrees, just outside the skirt
(osr). Each end is caught up by a bow; the hem (ozh, a spline through points measured on the front and back
views) hangs straight down the front edge from the bow to 4.3, swags to 2.95 at the side and rises up the
back to 4.5 under the back bow; the cloth falls in folds that deepen toward the hem. q runs across a piece from
the front bow to the back bow, p down from the waist to the hem. Lined in orange, with the hem turned
back over the side (oft) so the lining shows there, as drawn on the sheet.""")
    sc.func('osr(h)', f"sa(sz(h))+0.13*min(1,max(0,{fmt(OV_TOP)}-h)/0.3)")
    sc.func('ozh(q)', '+'.join([fmt(OZH_C[0]), f"{fmt(OZH_C[1])}*q"] + [f"{fmt(c)}*max(0,q-{fmt(k)})^3" for c, k in zip(OZH_C[2:], _OZK)]))
    sc.func('oph(q,p)', f"{fmt(OV_TOP)}-({fmt(OV_TOP)}-ozh(q))*p")
    sc.func('ofd(q,p)', '0.07*p^1.2*(0.5+0.5*sin(10*pi*q))')
    sc.func('oft(q)', '0.26*sin(pi*q)^3')
    TH = f"({fmt(OV_F)}+{fmt(OV_B - OV_F)}*q)"
    for sd, tag in ((1, 'L'), (-1, 'R')):
        for nm, col, dr, p0 in (('overskirt', CREAM, 0.0, 0.0), ('overskirtLining', RUST, -0.015, 0.03)):
            H = "oph(q,p)"
            R = f"(osr({H})+ofd(q,p)+{fmt(dr)})"
            sc.surface((f"{fmt(sd)}*{R}*sin({TH})", f"-{R}*dk(sz({H}),{TH})*cos({TH})", H), ('q', 0, 1), ('p', p0, 1), col, f'{nm}{tag}', ('point', (0, 0, 4)))
        H = "(ozh(q)+oft(q)*v)"
        R = f"(osr(ozh(q))+ofd(q,1)+0.012+0.02*v)"
        sc.surface((f"{fmt(sd)}*{R}*sin({TH})", f"-{R}*dk(sz(ozh(q)),{TH})*cos({TH})", H), ('q', 0, 1), ('v', 0, 1), RUST, f'overskirtTurn{tag}', ('point', (0, 0, 4)))
        TT = TH.replace('q', 't')
        R = "(osr(ozh(t))+ofd(t,1)+0.012)"
        sc.curve((f"{fmt(sd)}*{R}*sin({TT})", f"-{R}*dk(sz(ozh(t)),{TT})*cos({TT})", "ozh(t)"), ('t', 0, 1), GOLD, f'overskirtHem{tag}', thickness=2)
        H = "(ozh(t)+oft(t)+0.14+0.05*sin(40*t))"
        R = f"(osr({H})+ofd(t,1)+0.012)"
        sc.curve((f"{fmt(sd)}*{R}*sin({TT})", f"-{R}*dk(sz({H}),{TT})*cos({TT})", H), ('t', 0, 1), ORANGE, f'overskirtVine{tag}', thickness=2)

        # black bands down the overskirt's front and back edges, from the waist to the bows
        for nm, te, dirn, zb_ in (('F', OV_F, 1, BOW_ZF), ('B', OV_B, -1, BOW_ZB)):
            H = f"({fmt(OV_TOP - 0.05)}-{fmt(OV_TOP - 0.05 - zb_)}*p)"
            R = f"(osr({H})+0.02)"
            TE = f"({fmt(te)}+{fmt(dirn)}*(v-0.25)*0.36/{R})"
            sc.surface((f"{fmt(sd)}*{R}*sin({TE})", f"-{R}*dk(sz({H}),{TE})*cos({TE})", H), ('p', 0, 1), ('v', 0, 1), BLACK, f'edgeBand{nm}{tag}', ('point', (0, 0, 4)))
            for k, vv in enumerate((-0.25, 1)):
                Ht = H.replace('p', 't')
                Rt = f"(osr({Ht})+0.026)"
                TE = f"({fmt(te)}+{fmt(dirn * (vv - 0.25) * 0.36)}/{Rt})"
                sc.curve((f"{fmt(sd)}*{Rt}*sin({TE})", f"-{Rt}*dk(sz({Ht}),{TE})*cos({TE})", Ht), ('t', 0, 1), GOLD, f'edgeBandTrim{nm}{tag}{k + 1}', thickness=2)

    sc.blank()
    sc.note('Skirt: orange tabs at the front (her left) and the back, folded at the foot, edged in gold')
    sc.comment("""Tabs, measured on outlines traced over the sheet: each lies on the pleated skirt (x is measured across, so
it follows the skirt's curve) and ends in a diagonal fold, the turned-back end hanging below it as a lighter
flap. Front tab, on her left of the centre line: inner edge ftx(0,h) from 0.31 in to 0.14, outer edge widening
from 0.5 to 0.87; folded from (0.14, 5.24) up to (0.87, 6.0), its point at (0.88, 4.5). Back tab, at the centre
back: 0.34 wide at the waist, widening to 0.9; folded from 4.76 on her right up to 5.48 on her left, the flap
narrowing to a point at 3.85 on her left. v runs across a tab, p down a part.""")
    sc.func('ftx(v,h)', '0.14+0.065*(h-5.24)+(min(0.87,0.5+0.24*(8.5-h))-0.14-0.065*(h-5.24))*v')
    sc.func('ffd(v)', '5.24+0.76*v')
    sc.func('ffb(v)', '5.24-0.74*v')
    sc.func('bhw(h)', '0.17+0.28*(1-exp(-(8.1-h)/0.8))')
    sc.func('btx(v,h)', '-bhw(h)-0.12*max(0,(5.5-h)/0.74)+(2*bhw(h)+0.12*max(0,(5.5-h)/0.74))*v')
    sc.func('btf(v,h)', '-bhw(h)-0.12*max(0,(5.5-h)/0.74)+(0.4-0.05*(5.48-h)+bhw(h)+0.12*max(0,(5.5-h)/0.74))*v')
    sc.func('bfd(v)', '4.76+0.72*v')
    sc.func('bfb(v)', '4.76-0.91*v')
    sc.comment("The tabs are cloth: each falls in two soft folds (tbf, how far the cloth stands off the skirt) that open from the waist to the foot, creased between them.")
    sc.func('tbf(v,p)', '0.075*(0.15+0.85*p)*(0.5-0.5*cos(4*pi*v))')

    def on_tab(X, H, off, back):
        A = f"(sa(sz({H}))+{off if isinstance(off, str) else fmt(off)})"
        T = f"(pi-asin(min(0.97,{X}/{A})))" if back else f"asin(min(0.97,{X}/{A}))"
        return (X, f"-{A}*dk(sz({H}),{T})*cos({T})", H)

    for nm, xf, xflap, top, fold, foot, back in (('frontTab', 'ftx', 'ftx', 8.0, 'ffd', 'ffb', False),
                                                  ('backTab', 'btx', 'btf', 8.05, 'bfd', 'bfb', True)):
        H = f"({fmt(top)}-({fmt(top)}-{fold}(v))*p)"
        sc.surface(on_tab(f"{xf}(v,{H})", H, "0.07+tbf(v,p)", back), ('v', 0, 1), ('p', 0, 1), RUST, nm, ('dir', (0, 1 if back else -1, 0.2)))
        sc.curve(on_tab(f"{xf}(0.5,{fmt(top)}-({fmt(top)}-{fold}(0.5))*t)", f"({fmt(top)}-({fmt(top)}-{fold}(0.5))*t)", 0.075, back),
                 ('t', 0.08, 1), RUST_DK, nm + 'Crease', thickness=2)
        H = f"({fold}(v)-({fold}(v)-{foot}(v))*p)"
        sc.surface(on_tab(f"{xflap}(v,{H})", H, "0.082+tbf(v,1-0.6*p)", back), ('v', 0, 1), ('p', 0, 1), RUST_LT, nm + 'Flap',
                   ('dir', (0, 1 if back else -1, 0.2)))
        # the woven border: a cream stripe above the fold and above the flap's foot, sprigs and dots above each
        for part, edge, xe, off_, v0 in (('', fold, xf, "0.078+tbf(v,1)", 0.0), ('Flap', foot, xflap, "0.09+tbf(v,0.4)", 0.12)):
            H = f"({edge}(v)+0.04+0.07*p)"
            sc.surface(on_tab(f"{xe}(v,{H})", H, off_, back), ('v', v0, 1), ('p', 0, 1), BORDER_CREAM, f'{nm}{part}Border', ('dir', (0, 1 if back else -1, 0.2)))
            vs = (0.2, 0.55, 0.87) if not part else (0.5, 0.82)
            for i_, vv in enumerate(vs):
                h0 = f"({edge}({vv})+0.19)"
                Xl, Hl = sprig_xy(f"{xe}({vv},{h0})", h0)
                sc.curve(on_tab(Xl, Hl, off_.replace('(v,', f'({vv},').replace('0.078', '0.083').replace('0.09', '0.095'), back), ('t', 0, 2.999), BORDER_CREAM, f'{nm}{part}Sprig{i_ + 1}', thickness=2)
            for i_, vv in enumerate([(a_ + b_) / 2 for a_, b_ in zip(vs[:-1], vs[1:])]):
                h0 = f"({edge}({fmt(vv)})+0.17)"
                sc.curve(on_tab(f"({xe}({fmt(vv)},{h0})+0.012*cos(t))", f"({h0}+0.012*sin(t))", off_.replace('(v,', f'({fmt(vv)},').replace('0.078', '0.083').replace('0.09', '0.095'), back),
                         ('t', 0, '2*pi'), BORDER_CREAM, f'{nm}{part}Dot{i_ + 1}', thickness=2)
        for k, (X, Hc) in enumerate(((f"{xf}(0,{fmt(top)}-({fmt(top)}-{fold}(0))*t)", f"({fmt(top)}-({fmt(top)}-{fold}(0))*t)"),
                                     (f"{xf}(1,{fmt(top)}-({fmt(top)}-{fold}(1))*t)", f"({fmt(top)}-({fmt(top)}-{fold}(1))*t)"),
                                     (f"{xflap}(t,{fold}(t))", f"{fold}(t)"),
                                     (f"{xflap}(t,{foot}(t))", f"{foot}(t)"),
                                     (f"{xflap}({0 if not back else 1},{fold}({0 if not back else 1})-({fold}({0 if not back else 1})-{foot}({0 if not back else 1}))*t)",
                                      f"({fold}({0 if not back else 1})-({fold}({0 if not back else 1})-{foot}({0 if not back else 1}))*t)"),
                                     (f"{xflap}({1 if not back else 0},{fold}({1 if not back else 0})-({fold}({1 if not back else 0})-{foot}({1 if not back else 0}))*t)",
                                      f"({fold}({1 if not back else 0})-({fold}({1 if not back else 0})-{foot}({1 if not back else 0}))*t)"))):
            sc.curve(on_tab(X, Hc, {2: "0.09+tbf(t,1)", 3: "0.09+tbf(t,0.4)"}.get(k, 0.09), back), ('t', 0, 1), GOLD, f'{nm}Trim{k + 1}', thickness=2)
    # the black strip under the corset's tassel, hanging over the front tab to 6.6 (traced on the front and side views)
    pts_ = [skirt_point(h, math.asin(0.505 / (SA(sk_s(h)) + 0.12)), 0.12) for h in (7.72, 7.15, 6.6)]
    ribbon(pts_, 0.085, BLACK, 'corsetStrip', unit((0, -1, 0.15)), vcut=0.0, rim=GOLD)

    sc.blank()
    sc.note('Skirt: four black bows with chrysanthemums and gold tassels where the overskirt is caught up')
    for sd, tag in ((1, 'L'), (-1, 'R')):
        for fb, th0, zb_, sz_ in (('F', OV_F, BOW_ZF, 1.0), ('B', OV_B, BOW_ZB, 0.92)):
            th = sd * th0
            s_ = sk_s(zb_)
            k_ = DK(s_, th)
            P = ((OSR(zb_) + 0.03) * math.sin(th), -(OSR(zb_) + 0.03) * k_ * math.cos(th), zb_)
            n = skirt_normal(zb_, th)
            tang = unit((math.cos(th), k_ * math.sin(th), 0))
            upv = cross(n, tang)
            C = add(P, mul(n, 0.07))
            bow(C, (tang, upv, n), (0.46 * sz_, 0.2 * sz_, 0.09), f'bow{fb}{tag}', loops=((1, -0.2), (-1, -0.2)))
            flower(add(add(C, mul(n, 0.08)), mul(upv, 0.5 * sz_)), n, 0.24 * sz_, f'bowFlower{fb}{tag}')     # the medallion above
            for k, (dx, ln) in enumerate(((0.09, 0.95), (-0.1, 0.8))):
                b0 = add(C, mul(tang, dx))
                ln *= sz_
                ribbon([b0, add(add(b0, mul(tang, dx * 2.2)), add(mul(n, 0.06), (0, 0, -ln * 0.5))), add(add(b0, mul(tang, dx * 3.6)), add(mul(n, 0.1), (0, 0, -ln)))],
                       0.14, BLACK, f'bowTail{fb}{tag}{k + 1}', n, rim=GOLD, vcut=0.3)
            for k, dx in enumerate((0.07, -0.08)):
                tassel(add(add(C, mul(n, 0.1)), add(mul(tang, dx * sz_), (0, 0, -0.12))), 0.95 * sz_, GOLD, f'bowTassel{fb}{tag}{k + 1}', r=0.05)

# =====================================================================
#  BOOTS
# =====================================================================
if want('boots'):
    run('boots.py')       # built from the three-view drawing measured on the sheet

# =====================================================================
#  DISPLAY BASE
# =====================================================================
if want('base'):
    sc.blank()
    sc.note('Display base')
    sc.surface(("3.4*sqs(u,0.3)*cos(v)", "3.4*sqs(u,0.3)*sin(v)", "-0.2+0.2*sqc(u,0.3)"), ('u', 0, 'pi'), ('v', 0, '2*pi'), '#2b2226', 'base', 'closed')
    torus((0, 0, -0.02), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 3.38, 0.035, GOLD, 'baseRim')
    torus((0, 0, 0.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 3.12, 0.012, GOLD, 'baseRing')
    sc.surface(("2.75*r*cos(w)", "2.45*r*sin(w)", "0.006"), ('r', 0, 1), ('w', 0, '2*pi'), 'black', 'shadow', ('dir', (0, 0, 1)), opacity=0.25)
    flower((0, -3.05, 0.012), (0, 0, 1), 0.2, 'baseEmblem', layers=2, flat=True, up=(0, 1, 0))
