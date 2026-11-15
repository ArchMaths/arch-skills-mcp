"""Embroidery motifs of Claude Chan's dress as Arch curve expressions, measured on close-up photos.

    python3 motifs.py --photo fern.jpg -o fern-demo.png     # 2D demo of the fern, over the photo it was measured on

A motif is a list of (name, (X, H), (t0, t1)): X across and H up, in model units, as Arch expressions in t.
Every leaf starts and ends where it joins its stem, so a curve that draws several leaves (index floor(t)) only
ever jumps along a stem it has already drawn and leaves no stray stroke.
"""
import math


def _f(v):
    return v if isinstance(v, str) else f"{v:.5g}"


def clamp01(e):
    return f"min(1,max(0,{e}))"


def quad3(a, b, c, m):
    """Expression equal to a, b, c at m = 0, 1, 2."""
    return f"({_f(a)}+{_f(-1.5 * a + 2 * b - 0.5 * c)}*{m}+{_f(0.5 * a - b + 0.5 * c)}*{m}^2)"


def leaf_off(tilt, L, W, w, taper=0.6):
    """Offsets (dx, dh) of a leaf's outline from its base as w runs 0..1: L long, about W wide, leaning tilt from
    straight up (+ toward +x). taper 0 is an ellipse; larger narrows the base into a teardrop."""
    L, W, tilt = _f(L), _f(W), _f(tilt)
    u = f"((1-cos(2*pi*({w})))/2)"
    a = f"({L})*{u}"
    c = f"({W})/2*sin(2*pi*({w}))" + (f"*{_f(1 / (0.94 * 0.67 ** taper))}*{u}^{_f(taper)}" if taper else '')
    return (f"({a}*sin({tilt})+{c}*cos({tilt}))", f"({a}*cos({tilt})-{c}*sin({tilt}))")


# ---- the skirt's fern (photo: 0.00232 units a pixel; stem at x 100, foot at y 460) ------------------------------
FERN_STEM = 0.79                       # the stem's top, where the top leaf starts
FERN_TOP = (0.145, 0.058)              # top leaf: length, width
FERN_NODE = (0.719, -0.1415)           # height of the first pair's node, step to the next (3 pairs)
FERN_PAIR_L = (0.125, 0.0155)          # pair leaf length: first pair, growth per pair
FERN_PAIR_W = (0.049, 0.004)
FERN_PAIR_TILT = (0.70, 0.015)          # radians off the stem
FERN_FORK = 0.195                      # where the twigs leave the stem
FERN_TWIG = ((0.12, 0.295), (0.16, 0.36))            # twig: its bend A and its end B (x, h), mirrored
FERN_TWIG_LEAVES = ((0.12, 0.100), (1.46, 0.092), (0.26, 0.094))   # (tilt, length): two at A, one at B
FERN_TWIG_W = 0.046
FERN_DIAMOND = (0.105, 0.062, 0.095)   # centre height, half width, half height
FERN_FOOT = (0.0, 0.125, 0.04, 0.75, 0.105, 0.048)  # foot twig: height at the stem, x and h of its ends; leaf tilt, length, width


def fern_curves():
    out = []
    dx, dh = leaf_off(0, *FERN_TOP, 'max(0,t-1)', taper=0.15)
    out.append(('Stem', (dx, f"({_f(FERN_STEM)}*min(1,t)+{dh})"), (0, 2)))
    k, j = 'floor(t)', 'floor(t/2)'
    sg = f"(1-2*({k}-2*{j}))"
    dx, dh = leaf_off(f"{sg}*({_f(FERN_PAIR_TILT[0])}+{_f(FERN_PAIR_TILT[1])}*{j})", f"{_f(FERN_PAIR_L[0])}+{_f(FERN_PAIR_L[1])}*{j}",
                      f"{_f(FERN_PAIR_W[0])}+{_f(FERN_PAIR_W[1])}*{j}", f"t-{k}", taper=0.9)
    out.append(('Pairs', (dx, f"({_f(FERN_NODE[0])}+{_f(FERN_NODE[1])}*{j}+{dh})"), (0, 5.999)))
    # a twig, one stroke: its end leaf, down the twig to its bend (two leaves there), on down to the stem
    (ax, ah), (bx, bh) = FERN_TWIG
    xl = f"({_f(bx)}-{_f(bx - ax)}*{clamp01('t-1')}-{_f(ax)}*{clamp01('t-4')})"
    s1, s2 = (ah - FERN_FORK) / ax, (bh - ah) / (bx - ax)
    hl = f"({_f(FERN_FORK)}+{_f(s1)}*{xl}+{_f(s2 - s1)}*max(0,{xl}-{_f(ax)}))"
    X, H = [xl], [hl]
    (t2, l2), (t3, l3), (t1, l1) = FERN_TWIG_LEAVES
    for seg, tilt, L in ((0, t1, l1), (2, t2, l2), (3, t3, l3)):
        dx, dh = leaf_off(tilt, L, FERN_TWIG_W, clamp01(f"t-{seg}"), taper=0.9)
        X.append(dx), H.append(dh)
    out.append(('TwigL', ('+'.join(X), '+'.join(H)), (0, 5)))
    out.append(('TwigR', ('-(' + '+'.join(X) + ')', '+'.join(H)), (0, 5)))
    hc, wx, wh = FERN_DIAMOND
    out.append(('Diamond', (f"{_f(wx)}*sqc(t,2)", f"({_f(hc)}+{_f(wh)}*sqs(t,2))"), (0, '2*pi')))
    h0, fx, fh, tilt, L, W = FERN_FOOT
    xl = f"({_f(-fx)}+{_f(2 * fx)}*{clamp01('t-1')})"
    a = leaf_off(-tilt, L, W, clamp01('t'), taper=0.9)
    b = leaf_off(tilt, L, W, clamp01('t-2'), taper=0.9)
    out.append(('Foot', (f"{xl}+{a[0]}+{b[0]}", f"{_f(h0)}+{_f((fh - h0) / fx)}*abs({xl})+{a[1]}+{b[1]}"), (0, 3)))
    return [(n, (f"({X})", f"({H})"), T) for n, (X, H), T in out]


# ---- the orange cloth's sprig (photo: 372 pixels a unit) ---------------------------------------------------------
def sprig_xy(x0, h0):
    """Three petals fanned 34 degrees from (x0, h0): 0.082 long between two of 0.062, 0.026 wide."""
    k = 'floor(t)'
    dx, dh = leaf_off(f"((1-{k})*0.6)", f"0.082-0.02*abs({k}-1)", '0.03', f"t-{k}", taper=0.35)
    return (f"({x0}+{dx})", f"({h0}+{dh})")


# ---- the black panels' emblem -------------------------------------------------------------------------------------
# Too small on the sheet to read (14 x 20 pixels: a dense core, about 8 spikes, a tail below, 0.33 from the centre
# to its top), so the design is Claude's proposal, which the user chose among the candidates: a star of 8 kite
# spikes with 8 shorter kite petals between them round a pale disc, and below it a stem with two kite leaves.
# A kite runs from its foot on the axis, widens to its full width 0.45 of the way along, and ends in a point.
EMB_FOOT = 0.05                        # where every kite starts
EMB_SPIKE = (0.32, 0.07)               # length from the centre, width
EMB_PETAL = (0.21, 0.075)
EMB_CENTRE = 0.06                      # the pale disc
EMB_STEM = (-0.34, -0.50)              # the sprig's stem, below the star
EMB_LEAF = (-0.40, 0.8, 0.11, 0.045)   # the leaves: height of their foot, angle off straight down, length, width


def _kite(phi, L, w):
    """Outer and inner radius of a kite at angle phi off its axis (0 and 9 where the kite does not reach)."""
    h, m = w / 2, EMB_FOOT + 0.45 * (L - EMB_FOOT)
    ps = math.atan2(h, m)
    on = f"(0.5+0.5*sign({_f(ps)}-{phi}))"
    q = f"min({phi},{_f(ps)})"
    front = f"{_f(L * h)}/({_f(h)}*cos({q})+{_f(L - m)}*sin({q}))*{on}"
    back = f"(9+({_f(EMB_FOOT * h)}/({_f(h)}*cos({q})-{_f(m - EMB_FOOT)}*sin({q}))-9)*{on})"
    return front, back


def emblem_radii(w='w'):
    """The star as a ring: (inner, outer) radius at angle w (from +x; a spike points up). Between the kites' feet
    the inner radius is beyond the disc, which leaves the dark teeth round it."""
    ps = f"(acos(cos(8*(({w})-pi/2)))/8)"            # angle to the nearest spike's axis
    pp = f"({_f(math.pi / 8)}-{ps})"                 # and to the nearest petal's
    sf, sb = _kite(ps, *EMB_SPIKE)
    pf, pb = _kite(pp, *EMB_PETAL)
    return f"min({sb},{pb})", f"max({sf},{pf})"


def emblem_curves():
    out = [('Star', ("emo(t)*cos(t)", "emo(t)*sin(t)"), (0, '2*pi')),
           ('Ring', (f"{_f(EMB_CENTRE + 0.003)}*cos(t)", f"{_f(EMB_CENTRE + 0.003)}*sin(t)"), (0, '2*pi')),
           ('Stem', ("0*t", f"{_f(EMB_STEM[0])}+{_f(EMB_STEM[1] - EMB_STEM[0])}*t"), (0, 1))]
    return [(n, (f"({x})", f"({h})"), T) for n, (x, h), T in out]


def emblem_leaf(side):
    """One leaf as a surface: (x, h) expressions in u (0..1 along it) and v (-1..1 across it)."""
    h0, ang, L, W = EMB_LEAF
    dx, dh = side * math.sin(ang), -math.cos(ang)
    hw = f"{_f(W / 2)}*min(u/0.45,(1-u)/0.55)*v"
    return (f"({_f(L * dx)}*u+{_f(-dh)}*{hw})", f"({_f(h0)}+{_f(L * dh)}*u+{_f(dx)}*{hw})")


# ---- evaluating a motif in Python --------------------------------------------------------------------------------
def evaluate(curves, n=1500):
    import numpy as np
    ns = dict(sin=np.sin, cos=np.cos, acos=lambda v: np.arccos(np.clip(v, -1, 1)), exp=np.exp, abs=np.abs, sign=np.sign, floor=np.floor, pi=np.pi, min=np.minimum, max=np.maximum,
              sqc=lambda t, p: np.sign(np.cos(t)) * np.abs(np.cos(t)) ** p, sqs=lambda t, p: np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p)
    out = []
    for name, (X, H), (t0, t1) in curves:
        t = np.linspace(float(eval(str(t0).replace('^', '**'), ns)), float(eval(str(t1).replace('^', '**'), ns)), n)
        ns['t'] = t
        out.append((name, eval(X.replace('^', '**'), ns) + 0 * t, eval(H.replace('^', '**'), ns) + 0 * t))
    return out


if __name__ == '__main__':
    import argparse
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--photo', help='the close-up the fern was measured on')
    ap.add_argument('-o', default='fern-demo.png')
    a = ap.parse_args()
    K, X0, Y0 = 0.00232, 100, 460
    cs = evaluate(fern_curves())
    fig, axs = plt.subplots(1, 3 if a.photo else 1, figsize=(13 if a.photo else 5, 9), squeeze=False)
    axs = axs[0]
    if a.photo:
        import cv2
        im = cv2.imread(a.photo)
        B, G, R = [im[..., i].astype(int) for i in range(3)]
        mask = (R - B > 70) & (R > 150) & (G < 170)
        mask[Y0:] = False
        model = np.zeros(mask.shape, np.uint8)
        for name, x, h in cs:
            P = np.c_[X0 + x / K, Y0 - h / K].round().astype(np.int32)
            cv2.polylines(model, [P], False, 1, 5)
            if name != 'Twigs' or True:
                pass
        # fill each closed leaf loop: flood the outline's inside
        ff = model.copy()
        cv2.floodFill(ff, None, (0, 0), 2)
        model = ((ff != 2) | (model > 0))
        iou = (model & mask)[:Y0].sum() / (model | mask)[:Y0].sum()
        ext = (-X0 * K, (im.shape[1] - X0) * K, (Y0 - im.shape[0]) * K, Y0 * K)
        axs[0].imshow(im[..., ::-1], extent=ext)
        axs[0].set_title('photo')
        axs[1].imshow(np.where(mask, 0.78, 1.0), extent=ext, cmap='gray', vmin=0, vmax=1)
        for name, x, h in cs:
            axs[1].plot(x, h, color='#c45c36', lw=1.4)
        axs[1].set_title(f'thread found in the photo (grey) and the motif; overlap {iou:.2f}')
        print(f'overlap (IoU of the filled motif and the photo\'s thread): {iou:.3f}')
    ax = axs[-1]
    for name, x, h in cs:
        ax.plot(x, h, color='#c45c36', lw=2.2, solid_joinstyle='round')
    ax.set_title('the motif as Arch draws it')
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-0.24, 0.28)
        ax.set_ylim(-0.03, 0.99)
    fig.tight_layout()
    fig.savefig(a.o, dpi=110)
