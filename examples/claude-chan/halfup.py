"""The half-up hairstyle's long hair: below the braid the hair is gathered into a tie at the nape, and long wavy
locks flare out of the tie over the capelet, over a looser layer that falls from behind the ears and the nape
(after the user's back-view reference: the gathered "V" under the band, a small tie, and a mass of wavy clumps that
widens to about two and a half head-widths, each ending in a point that flicks outward).

Pure geometry (numpy), so it can be designed and previewed outside Arch: hair.py calls strands() with the scene's own
clothes depth and tie, and writes each strand as the chest locks are written, C(t) + cos(w) U(t) + lcs(w) V(t), with
C, U, V polynomials in t from -1 to 1 (Horner form), in a few pieces so that every piece fits within FIT_TOL.

Depth follows the project's rule for long hair (CLAUDE.md): a strand hangs as a taut sheet, from its root straight
out over the fullest point of what it passes (the capelet) and then straight down, never bending in toward the waist;
it wraps round her a little as it fans out (WRAP), and the clumps on top lie just above the ones beneath.
"""
import math
import numpy as np

FIT_TOL = 0.003          # largest distance between a strand's samples and its polynomials
FIT_DEG = 14             # degree of each piece
GAP = 0.03               # between a strand's flat underside and the cloth or hair it lies on
WRAP = 0.07              # the hair comes forward by WRAP (x^2 - x0^2) as it fans out below its fullest point
THICK = 1 / 3.6          # a clump's thickness over its width (CLAUDE.md: broad long clumps about four times as wide)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def lcs(w):
    """The bangs' ridged crescent across a clump (hair.py): 1 at the crest (w = pi/2), -0.3 underneath."""
    w = np.asarray(w, float)
    return (0.65 * np.sin(w) + 0.35 * np.abs(np.sin(w))) * (0.78 + 0.22 * np.cos(2 * np.pi * np.cos(w)))


def width_profile(t, w0, wmax, t_full, t_taper):
    """Half-width along a clump: from w0 at the root to wmax by t_full, then to a sharp point at the tip."""
    w = w0 + (wmax - w0) * smoothstep(0, t_full, t)
    # to the point along a quarter sine: level where it starts, straight into the point (no power under 1 at the
    # tip, which the polynomials cannot follow: CLAUDE.md)
    tip = np.sin(0.5 * np.pi * np.clip((1 - t) / (1 - t_taper), 0, 1))
    return w * tip


def taut_depth(t, floor, y0):
    """How far back (y) a strand hangs along t: from y0 at its root, straight out over the fullest point of floor (the
    upper concave hull of the root and the floor, which only rises), then straight down; never under floor."""
    cm = np.maximum.accumulate(np.maximum(floor, -9))
    pts = [(0.0, y0)] + list(zip(t[1:], cm[1:]))
    hull = []
    for p in pts:                                   # upper hull, left to right
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    H = np.array(hull)
    y = np.interp(t, H[:, 0], H[:, 1])
    # round the hull's corners (a taut lock still bends smoothly), then lift it clear of the floor smoothly
    return lift_over(smooth(y, 8.0), cm)


def smooth(y, sigma):
    """y smoothed by a Gaussian of sigma samples (the ends held)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, r, mode='edge'), k, mode='valid')


def lift_over(y, floor, sigma=10.0):
    """y raised smoothly until it is nowhere under floor: each shortfall spread over its neighbours (a clamp would
    copy the floor's steps into the strand)."""
    r = int(2 * sigma)
    for _ in range(12):
        d = np.maximum(0.0, floor - y)
        if d.max() < 2e-4:
            break
        win = np.lib.stride_tricks.sliding_window_view(np.pad(d, r, mode='edge'), 2 * r + 1)
        y = y + smooth(win.max(1), sigma) + 2e-4
    return y


def frames(P, ref):
    """Unit tangent T, facing N (ref made perpendicular to T) and across B along a centre line P (n, 3)."""
    T = np.stack([smooth(c, 3.0) for c in np.gradient(P, axis=0).T], 1)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = ref - np.sum(ref * T, 1)[:, None] * T
    N /= np.linalg.norm(N, axis=1)[:, None]
    B = np.cross(N, T)
    return T, N, B


def fit_pieces(t, C, U, V, tol=FIT_TOL, deg=FIT_DEG, max_pieces=8):
    """Polynomials (in tau from -1 to 1 over each piece) for C, U, V sampled at t (0..1), in as few equal pieces as fit
    within tol. Returns [(Cs, Us, Vs)], each a list of three coefficient arrays, and the largest error."""
    for n in range(1, max_pieces + 1):
        pieces, worst = [], 0.0
        edges = np.linspace(0, 1, n + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a - 1e-9) & (t <= b + 1e-9)
            tau = 2 * (t[m] - a) / (b - a) - 1
            fit = []
            for S in (C, U, V):
                cs = [np.round(np.polynomial.polynomial.polyfit(tau, S[m, i], deg), 6) for i in range(3)]
                err = max(float(np.max(np.abs(np.polynomial.polynomial.polyval(tau, c) - S[m, i]))) for i, c in enumerate(cs))
                worst = max(worst, err)
                fit.append(cs)
            pieces.append(tuple(fit))
        if worst <= tol:
            break
    return pieces, worst


def depth_grid(strands, spec):
    """How far back (largest y) the strands reach at each (x, z) cell of spec (x0, z0, step, nx, nz); -9 where none."""
    x0, z0, st, nx, nz = spec
    G = np.full((nx, nz), -9.0)
    w = np.linspace(0, 2 * math.pi, 13)
    for s in strands:
        P = s['C'][:, None, :] + np.cos(w)[None, :, None] * s['U'][:, None, :] + lcs(w)[None, :, None] * s['V'][:, None, :]
        P = P.reshape(-1, 3)
        i, j = np.round((P[:, 0] - x0) / st).astype(int), np.round((P[:, 2] - z0) / st).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
        np.maximum.at(G, (i[ok], j[ok]), P[ok, 1])
    for _ in range(3):                      # close the gaps between samples
        Gp = np.pad(G, 1, constant_values=-9.0)
        G = np.max([Gp[a:a + nx, b:b + nz] for a in range(3) for b in range(3)], axis=0)
    return G


def grid_at(G, spec, x, z):
    x0, z0, st, nx, nz = spec
    i, j = np.round((np.asarray(x) - x0) / st).astype(int), np.round((np.asarray(z) - z0) / st).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
    return np.where(ok, G[np.clip(i, 0, nx - 1), np.clip(j, 0, nz - 1)], -9.0)


def build_strand(name, path_xz, y0, floor_fn, w0, wmax, ref_fn, t_full=0.28, t_taper=0.55, n=241, lift=0.0, wrap=WRAP,
                 thick=THICK):
    """A strand along path_xz(t) -> (x, z) in the back view, hung in depth as a taut sheet from y0 over floor_fn(x, z),
    with its section facing ref_fn(x, z, t) (made perpendicular to the strand)."""
    t = np.linspace(0, 1, n)
    x, z = path_xz(t)
    W = width_profile(t, w0, wmax, t_full, t_taper)
    TH = np.maximum(W * thick, 0.004)
    floor = floor_fn(x, z) + GAP + 0.3 * TH + lift
    y = taut_depth(t, floor, y0)
    # wrap round her below the fullest point, but never into what is under the strand
    k = int(np.argmax(np.maximum.accumulate(floor) >= floor.max() - 1e-9))
    xr = x[k]
    y = lift_over(y - wrap * smoothstep(t[k], min(1, t[k] + 0.25), t) * (x ** 2 - xr ** 2), floor)
    P = np.stack([x, y, z], 1)
    ref = ref_fn(x, z, t)
    T, N, Bv = frames(P, ref)
    U, V = W[:, None] * Bv, TH[:, None] * N
    pieces, err = fit_pieces(t, P, U, V)
    return dict(name=name, C=P, U=U, V=V, t=t, pieces=pieces, err=err)


def strands(tie, clothes_fn, under_root, head_fn, ear_roots=(), seed=5, n_fan=11, n_under=12, n_wisp=6, length=(7.35, 7.85)):
    """The long hair of the half-up style.
    tie: (x, y, z) of the tie's centre. clothes_fn(x, z): how far back the clothes reach (-9 where none).
    under_root(s): where the loose layer's strand s (-1 at her right to 1 at her left) starts, hidden under the braid,
    and which way it faces there: ((x, y, z), (nx, ny, nz)). head_fn(x, z): how far back the dark sheet under the hair
    on the head reaches, which the loose layer lies on. ear_roots: ((x, y, z), facing) of loose locks that fall from
    behind the ears, down the side of the head and over the shoulder (the ornament's ribbons lie on them). length: the
    range of heights the tips end at.
    Returns the strands, each a dict (name, layer, pieces, err, C, U, V, t), under-layer first."""
    rng = np.random.default_rng(seed)
    R = rng.random
    tx, ty, tz = tie
    zlo, zhi = length
    out = []

    def outward(x, z, t):        # facing back and a little out to the side, as the sheet wraps round her
        return np.stack([0.28 * x, np.ones_like(x), np.zeros_like(x)], 1)

    # 1) the loose under-layer, from under the braid round the back of the head and behind the ears: drops a little,
    #    then spreads over the shoulders
    def on_head(x, z):
        return np.maximum(clothes_fn(x, z), head_fn(x, z))
    for i in range(n_under):
        s = float(np.clip(-1 + 2 * i / (n_under - 1) + 0.04 * (R() - 0.5), -1, 1))
        r0, side = (np.array(v, float) for v in under_root(s))
        L = r0[2] - (zlo + (zhi - zlo) * (0.3 + 0.7 * abs(s)) - 0.15 * R())
        spread = s * (0.62 + 0.12 * R())
        amp, freq, ph = 0.13 + 0.05 * R(), 2.0, 1.3 * s + 0.7 * (R() - 0.5)
        cx, cz = math.copysign(0.45 * abs(s) ** 1.5, s), 0.3 * s * s

        def path(t, r0=r0, L=L, spread=spread, amp=amp, freq=freq, ph=ph, cx=cx, cz=cz):
            f = smoothstep(0.05, 0.75, t)
            k = smoothstep(0.72, 1.0, t)
            x = r0[0] + spread * f + amp * t ** 0.85 * np.sin(2 * np.pi * freq * t + ph) + cx * k * k
            z = r0[2] - L * t + cz * k * k
            return x, z

        def ref(x, z, t, side=side):     # facing out from the head at the root, back and out lower down
            b = smoothstep(0.0, 0.3, t)[:, None]
            return (1 - b) * side[None, :] + b * outward(x, z, t)
        st = build_strand(f'U{i + 1}', path, r0[1], on_head, 0.12, 0.34 + 0.08 * R(), ref, t_full=0.2, t_taper=0.6)
        st['layer'] = 'under'
        out.append(st)
    # behind the ears: straight down the side of the head first, then out over the shoulder and down beside the rest
    for i, (r0, side) in enumerate(ear_roots):
        r0, side = np.array(r0, float), np.array(side, float)
        sg = math.copysign(1, r0[0])
        L = r0[2] - (zlo + 0.15 + 0.45 * R())
        out_, amp, ph = sg * (0.25 + 0.25 * R()), 0.1 + 0.05 * R(), 1.9 * sg + 0.9 * (R() - 0.5)

        def path(t, r0=r0, L=L, out_=out_, amp=amp, ph=ph, sg=sg):
            f = smoothstep(0.08, 0.6, t)
            k = smoothstep(0.72, 1.0, t)
            x = r0[0] + out_ * f + amp * t ** 0.85 * np.sin(4 * np.pi * t + ph) + sg * 0.35 * k * k
            z = r0[2] - L * t + 0.25 * k * k
            return x, z

        def ref(x, z, t, side=side, sg=sg):  # facing out to her side, about 60 degrees round from facing back (as the
            b = smoothstep(0.15, 0.45, t)[:, None]     # earlier side hair, BK_SIDE_FACE: facing back, from the front
            out60 = np.array([0.87 * sg, 0.5, 0.0])   # the clumps showed as flat panels beside the neck)
            return (1 - b) * side[None, :] + b * out60[None, :]
        st = build_strand(f'E{i + 1}', path, r0[1], on_head, 0.03, 0.16 + 0.04 * R(), ref, t_full=0.3, t_taper=0.6, wrap=0.0,
                          thick=0.42)
        st['layer'] = 'under'
        out.append(st)
    under = list(out)
    spec = (-3.0, 6.0, 0.02, 301, 301)
    UG = depth_grid(under, spec)

    def on_under(x, z):
        return np.maximum(on_head(x, z), grid_at(UG, spec, x, z))

    # 2) the locks out of the tie: fan out at once, wave, and flick outward at the tips
    fan = []
    for i in range(n_fan):
        s = float(np.clip(-1 + 2 * i / (n_fan - 1) + 0.05 * (R() - 0.5), -1, 1))
        root = np.array([tx + 0.035 * s, ty + 0.015 * (1 - abs(s)), tz - 0.04])
        L = root[2] - (zlo + (zhi - zlo) * R() + 0.2 * abs(s))
        spread = s * 1.1 * (1 + 0.12 * (R() - 0.5))
        amp, freq, ph = 0.15 + 0.06 * R(), 2.2, 1.2 * s + 0.7 * (R() - 0.5)
        cx, cz = math.copysign(0.42 * abs(s) ** 1.5, s) + 0.06 * (R() - 0.5), 0.3 * s * s

        def path(t, root=root, L=L, spread=spread, amp=amp, freq=freq, ph=ph, cx=cx, cz=cz):
            f = 1 - (1 - np.minimum(t / 0.55, 1.0)) ** 2
            k = smoothstep(0.70, 1.0, t)
            x = root[0] + spread * f + amp * t ** 0.85 * np.sin(2 * np.pi * freq * t + ph) + cx * k * k
            z = root[2] - L * t + cz * k * k
            return x, z
        lift = 0.012 * (1 - abs(s)) + 0.01 * R()            # the middle locks lie a little over the outer ones
        st = build_strand(f'F{i + 1}', path, root[1], on_under, 0.05, 0.20 + 0.06 * R(), outward, t_full=0.12, lift=lift)
        st['layer'] = 'fan'
        st['path'] = path
        out.append(st)
        fan.append(st)

    # 3) thin wisps that peel off the top locks part way down (strand lines on the mass)
    FG = depth_grid(fan, spec)

    def on_fan(x, z):
        return np.maximum(on_under(x, z), grid_at(FG, spec, x, z))
    for i in range(n_wisp):
        P = fan[int(R() * len(fan))]
        t0 = 0.3 + 0.25 * R()
        sg = 1 if R() < 0.5 else -1
        drift, cx, cz = sg * (0.08 + 0.1 * R()), sg * (0.08 + 0.15 * R()), 0.1 * R()

        def path(u, P=P, t0=t0, drift=drift, cx=cx, cz=cz):
            x, z = P['path'](t0 + (1 - t0) * u * 0.95)
            k = smoothstep(0.55, 1.0, u)
            return x + drift * smoothstep(0, 1, u) + cx * k * k, z + cz * k * k
        x0_, z0_ = path(np.zeros(1))
        y0 = float(on_fan(x0_, z0_)[0]) + GAP
        st = build_strand(f'W{i + 1}', path, y0, on_fan, 0.015, 0.06 + 0.02 * R(), outward, t_full=0.25, t_taper=0.45,
                          lift=-0.01)
        st['layer'] = 'wisp'
        out.append(st)
    return out
