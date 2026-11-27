"""The face designed in Blender (/Users/tangziyan/clients/claude_chan_face/face.py), as pure numpy: the head's
design curves, the bumps on the face, and the drawn features as front-view shapes. face.py maps them onto the Arch
head (bf_* names: this module is imported, the others are exec'd into one namespace).

Blender's conventions: metres, z from the eye line, she faces -y, +x is her left. Features are designed for her left
side and mirrored. Colours are sRGB hex; Blender mixes them in linear light, so lmix() does too.
"""
import numpy as np

COL = dict(
    skin="#f5ebea", skin_shadow="#e4c9c8", blush="#f3c0bb", skin_line="#7c4a3d",
    sclera="#fefcfd", sclera_shadow="#c9c3d9",
    iris_dark="#4a3236", iris_mid="#8d6b5a", iris_light="#c8a47c", iris_glow="#eedab0", iris_ring="#3e2a2e",
    iris_shadow="#2c1e22", pupil="#563a3a", lid_shade="#f3e8ea",
    lash="#2b1b1a", lash_warm="#6a2e20", lash_lower="#8a5446", crease="#c79590",
    brow="#97553a", mouth="#8a625f", mouth_dark="#5c3e3d", mouth_light="#c6a5a2", nose_tick="#76585a",
)


def _lin(c):
    v = np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb(v):
    v = np.clip(v, 0, 1)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
    return '#' + ''.join(f'{int(round(x * 255)):02x}' for x in s)


def lmix(c0, c1, f):
    """f of the way from colour c0 to c1, mixed in linear light as Blender mixes."""
    return _srgb(_lin(c0) * (1 - f) + _lin(c1) * f)


def sstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (np.asarray(e1, float) - e0), 0, 1)
    return t * t * (3 - 2 * t)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def pchip(pts):
    """Monotone cubic through (x, y) points (no overshoot), extended linearly past the ends."""
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.empty_like(ys)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(xs) - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def f(x):
        x = np.asarray(x, float)
        xc = np.clip(x, xs[0], xs[-1])
        k = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, len(xs) - 2)
        t = (xc - xs[k]) / h[k]
        t2, t3 = t * t, t * t * t
        y = ((2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h[k] * m[k]
             + (-2 * t3 + 3 * t2) * ys[k + 1] + (t3 - t2) * h[k] * m[k + 1])
        return y + np.where(x < xs[0], (x - xs[0]) * m[0], 0.0) + np.where(x > xs[-1], (x - xs[-1]) * m[-1], 0.0)
    return f


# ---------------- the head ----------------
YS = 0.012                  # the side plane: the face's front curve meets the sides of the head here
LOWER = 0.90                # the face below the eyes is this much shorter than the design tables
ZC = -0.1005 * LOWER        # underside of the chin
Z1, ZT = 0.025, 0.121       # the skull's dome starts at Z1; top of the head at ZT
PROFILE = pchip([(-0.1005, -0.0520), (-0.0973, -0.0583), (-0.0908, -0.0630), (-0.0817, -0.0661),
                 (-0.0700, -0.0684), (-0.0599, -0.0712), (-0.0479, -0.0738), (-0.0325, -0.0750),
                 (-0.0183, -0.0732), (-0.0061, -0.0712), (0.0037, -0.0706), (0.0146, -0.0721), (0.025, -0.0738)])
WIDTH = pchip([(-0.1000, 0.0078), (-0.0960, 0.0210), (-0.0906, 0.0315), (-0.0840, 0.0410), (-0.0773, 0.0490),
               (-0.0706, 0.0560), (-0.0639, 0.0617), (-0.0572, 0.0662), (-0.0505, 0.0699), (-0.0439, 0.0730),
               (-0.0359, 0.0758), (-0.0232, 0.0782), (-0.0110, 0.0795), (0.0, 0.0796), (0.025, 0.0799)])
CHIN_ROUND = 0.006          # the width goes as sqrt(dz/(dz+CHIN_ROUND)) just above the chin


def dome(z, n):
    f = np.clip((np.asarray(z, float) - Z1) / (ZT - Z1), 0, 1)
    return (1 - f ** n) ** (1 / n)


def to_dome(z, low, high):
    s = sstep(Z1 - 0.012, Z1 + 0.022, z)
    return low * (1 - s) + high * s


def zdesign(z):
    z = np.asarray(z, float)
    return np.where(z < 0, z / LOWER, z)


def head_ws(z):
    """The face's half-width without the rounding into the chin (head_w = head_ws * sqrt(dz / (dz + CHIN_ROUND)))."""
    z = np.asarray(z, float)
    zd = zdesign(z)
    cheeks = 1 + 0.030 * np.exp(-((zd + 0.042) / 0.022) ** 2)
    lo = (0.95 * WIDTH(zd) * (1 - 0.05 * sstep(0, -0.1, zd)) * (1 + 0.025 * np.exp(-((zd - 0.004) / 0.03) ** 2)) * cheeks)
    temple = 1 + 0.035 * np.exp(-((z - 0.045) / 0.03) ** 2)
    dz = np.maximum(z - ZC, 0)
    r = np.sqrt(dz / (dz + CHIN_ROUND))
    hi = 0.95 * WIDTH(Z1) * temple * dome(z, 2.6)
    s = sstep(Z1 - 0.012, Z1 + 0.022, z)
    # to_dome blends lo * r with hi: dividing hi by r here makes head_ws * r exactly that blend
    return lo * (1 - s) + hi * s / np.maximum(r, 1e-9)


def head_w(z):
    dz = np.maximum(np.asarray(z, float) - ZC, 0)
    return head_ws(z) * np.sqrt(dz / (dz + CHIN_ROUND))


def head_pf(z):
    z = np.asarray(z, float)
    return to_dome(z, PROFILE(zdesign(z)), YS - (YS - PROFILE(Z1)) * dome(z, 2.6))


def head_q(z):
    z = np.asarray(z, float)
    zd = zdesign(z)
    q = 0.36 - 0.08 * np.exp(-(zd / 0.022) ** 2) + 0.06 / (1 + np.exp((zd + 0.05) / 0.01))
    return q + (0.5 - q) * sstep(Z1, ZT, z)


def head_fd(z):
    return smin(YS - head_pf(z), 1.18 * head_w(z), 0.004)


def front_y(x, z):
    """y of the bare face (no small features) at (x, z), on its front."""
    w, pf, q = head_w(z), head_pf(z), head_q(z)
    return pf + head_fd(z) * (1 - (1 - np.clip((x / np.maximum(w, 1e-9)) ** 2, 0, 1)) ** q)


NOSE_TIP = -0.0295
# the small features (nose, eye hollows, cheeks, lips, chin), each a Gaussian: (amplitude, |x| centre, z centre, sx, sz);
# the nose is written apart (its width and height change down it)
BUMPS = [(-0.0012, 0.0435, 0.002, 0.020, 0.013),       # eye hollows (|x|)
         (0.0030, 0.046, -0.031, 0.019, 0.018),        # cheeks (|x|)
         (0.0008, 0.0, -0.0472, 0.011, 0.0035),        # upper lip
         (0.0010, 0.0, -0.0553, 0.009, 0.004),         # lower lip
         (-0.0004, 0.0, -0.0510, 0.011, 0.0012),       # the line of the mouth
         (0.0012, 0.0, -0.081, 0.014, 0.007)]          # chin


def nose_a(z):
    z = np.asarray(z, float)
    c = np.clip((0.008 - z) / (0.008 - NOSE_TIP), 0, 1)
    return np.where(z > NOSE_TIP, 0.0064 * c ** 1.4, 0.0064 * np.exp(-((z - NOSE_TIP) / 0.0042) ** 2))


def nose_sig(z):
    c = np.clip((0.008 - np.asarray(z, float)) / (0.008 - NOSE_TIP), 0, 1)
    return 0.0030 + 0.0036 * c ** 1.5


def bumps(x, z):
    """How far the small features stand out from the bare face (Blender adds them along the rays)."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    b = nose_a(z) * np.exp(-(x / nose_sig(z)) ** 2)
    for a, cx, cz, sx, sz in BUMPS:
        b = b + a * np.exp(-((np.abs(x) - cx) / sx) ** 2 - ((z - cz) / sz) ** 2)
    return b


# ---------------- eyes (her left; mirrored) ----------------
EYE = dict(ix=0.0205, iz=-0.0036, ox=0.0665, oz=0.0040, hu=0.0088, hl=0.0086,
           rx=0.0116, rz=0.0126, cu=0.48, gaze=0.0012)
LID_UP_P = (0.62, 0.30)
LID_LO_P = (0.95, 0.75)


def lid_up(u):
    e = EYE
    u = np.clip(u, 0, 1)
    a, b = LID_UP_P
    return e["iz"] + (e["oz"] - e["iz"]) * u + e["hu"] * np.sin(np.pi * u ** a) ** b


def lid_lo(u):
    e = EYE
    u = np.clip(u, 0, 1)
    a, b = LID_LO_P
    return e["iz"] + (e["oz"] - e["iz"]) * u - e["hl"] * np.sin(np.pi * u ** a) ** b


def eye_x(u):
    return EYE["ix"] + (EYE["ox"] - EYE["ix"]) * np.asarray(u, float)


def eye_u(x):
    return (np.asarray(x, float) - EYE["ix"]) / (EYE["ox"] - EYE["ix"])


EYE["cx"] = float(eye_x(EYE["cu"]))
EYE["cz"] = float(lid_lo(EYE["cu"])) - 0.0010 + EYE["rz"]


def lid_normal(u, du=0.004):
    tx = eye_x(u + du) - eye_x(u - du)
    tz = lid_up(u + du) - lid_up(u - du)
    n = np.array([-tz, tx])
    return n / np.linalg.norm(n)


def band_t(u):
    return 0.0013 + 0.0049 * sstep(0.0, 0.55, u) + 0.0006 * sstep(0.6, 0.95, u)


IRIS_RAMP = [(0.08, 'iris_glow'), (0.30, 'iris_light'), (0.55, 'iris_mid'), (0.78, 'iris_dark')]


def iris_ramp(fac):
    """The iris's colour at fac = 0.5 + 0.5 dv (0 at its bottom, 1 at its top): Blender's ease ramp."""
    ps = [p for p, _ in IRIS_RAMP]
    if fac <= ps[0]:
        return COL[IRIS_RAMP[0][1]]
    if fac >= ps[-1]:
        return COL[IRIS_RAMP[-1][1]]
    for (p0, k0), (p1, k1) in zip(IRIS_RAMP, IRIS_RAMP[1:]):
        if p0 <= fac <= p1:
            t = (fac - p0) / (p1 - p0)
            return lmix(COL[k0], COL[k1], t * t * (3 - 2 * t))


# ---------------- drawn shapes: front-view strips (centre line, half-width, shift along the left normal) ----------------
def strip2d(pts, hw, offset=None):
    """As Blender's ribbon(): the centre line (moved by offset along its left normal) and the vector from it to the
    strip's left edge (normal x half-width), both (N, 2)."""
    pts = np.asarray(pts, float)
    hw = np.asarray(hw, float)
    T = np.gradient(pts, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    Nn = np.stack([-T[:, 1], T[:, 0]], 1)
    if offset is not None:
        pts = pts + Nn * np.asarray(offset, float)[:, None]
    return pts, Nn * hw[:, None]


def lash_band():
    """The upper lash band: a thin hook into the inner corner, heavy along the lid, a wing past the outer corner.
    Returns the strip and the indices where the hook ends and the wing starts."""
    e = EYE
    I, O = np.array([e["ix"], e["iz"]]), np.array([e["ox"], e["oz"]])
    u = np.linspace(0.0, 1.0, 90)
    lid = np.stack([eye_x(u), lid_up(u)], 1)
    t = band_t(u)
    hook = I + np.array([[-0.0030, -0.0036], [-0.0019, -0.0023], [-0.0009, -0.0011]])
    wing = O + np.array([[0.0020, 0.0011], [0.0039, 0.0020], [0.0056, 0.0030], [0.0070, 0.0040]])
    pts = np.vstack([hook, lid, wing])
    tt = np.r_[[0.0, 0.0004, 0.0008], t, [0.0060, 0.0045, 0.0026, 0.0]]
    off = np.r_[[0, 0, 0.1], 0.28 * np.ones_like(t), [0.25, 0.2, 0.15, 0.1]] * tt
    C, D = strip2d(pts, tt / 2, off)
    return C, D, 3, 3 + len(u)


LASH_SPIKES = ((0.46, 84, 0.0011, 0.00040), (0.56, 78, 0.0014, 0.00045), (0.66, 70, 0.0017, 0.00050),
               (0.76, 60, 0.0021, 0.00058), (0.86, 50, 0.0026, 0.00065), (0.95, 38, 0.0032, 0.00075))


def lash_spike(uu, ang, L, w0):
    n = lid_normal(uu)
    base = np.array([eye_x(uu), lid_up(uu)]) + n * (0.72 * float(band_t(uu)) - 0.0004)
    a = np.radians(ang)
    d = np.array([np.cos(a), np.sin(a)])
    s_ = np.linspace(0, 1, 10)[:, None]
    sp = base - d * 0.0006 + d * L * s_ + np.array([0.0006, -0.0008]) * (s_ ** 2)
    return strip2d(sp, w0 * (1 - s_[:, 0]) ** 1.1 + 0.00002)


def lower_lid():
    u = np.linspace(0.10, 1.0, 50)
    pts = np.stack([eye_x(u), lid_lo(u)], 1)
    w = 0.00014 + 0.00016 * sstep(0.45, 0.95, u)
    w = w * sstep(0.10, 0.22, u) + 0.00002
    return strip2d(pts, w, -w * 0.5)


def lid_shade():
    u = np.linspace(0.06, 1.0, 50)
    lo_e = lid_up(u) + 0.6 * band_t(u)
    hi_e = lid_up(u) + 0.0068 + 0.0008 * u
    mid = np.stack([eye_x(u), (lo_e + hi_e) / 2], 1)
    hw = (hi_e - lo_e) / 2 * np.sin(np.pi * (u - 0.06) / 0.94) ** 0.8
    return strip2d(mid, hw)


def lid_crease():
    u = np.linspace(0.22, 0.97, 40)
    pts = np.stack([eye_x(u) + 0.0005 * u, lid_up(u) + 0.0068 + 0.0008 * u], 1)
    w = 0.00020 * np.sin(np.pi * (u - 0.22) / 0.75) ** 0.6 + 0.00002
    return strip2d(pts, w)


def brow():
    u = np.linspace(0, 1, 50)
    x = 0.0190 + (0.0680 - 0.0190) * u
    z = 0.0262 + 0.0018 * np.sin(np.pi * u)
    hw = 0.00048 * (1 - 0.65 * u) * sstep(0, 0.08, u) + 0.00006
    return strip2d(np.stack([x, z], 1), hw)


MOUTH_Z, MOUTH_W, MOUTH_X = -0.0510, 0.0136, -0.0026


def mouth():
    """The mouth: a short, almost straight line, her right end (-x) darker, fuller and a little higher. Also returns
    u along it (0 at her right end), which its colour ramp follows."""
    s = np.linspace(-1, 1, 61)
    x = MOUTH_X + MOUTH_W * s
    z = MOUTH_Z + 0.00025 * s ** 2 - 0.0008 * s - 0.0003 * np.exp(-((s + 0.93) / 0.07) ** 2)
    hw = (0.00058 - 0.00036 * (s + 1) / 2) * np.clip(1 - np.abs(s) ** 6, 0, 1) ** 0.5 + 0.00003
    hw = hw + 0.00042 * np.exp(-((s + 0.90) / 0.07) ** 2)
    C, D = strip2d(np.stack([x, z], 1), hw)
    return C, D, (s + 1) / 2


MOUTH_RAMP = [(0.04, 'mouth_dark'), (0.45, 'mouth'), (1.0, 'mouth_light')]


def mouth_ramp(u):
    ps = [p for p, _ in MOUTH_RAMP]
    if u <= ps[0]:
        return COL[MOUTH_RAMP[0][1]]
    for (p0, k0), (p1, k1) in zip(MOUTH_RAMP, MOUTH_RAMP[1:]):
        if p0 <= u <= p1:
            return lmix(COL[k0], COL[k1], (u - p0) / (p1 - p0))
    return COL[MOUTH_RAMP[-1][1]]


def nose_tick():
    s = np.linspace(0, 1, 12)
    pts = np.stack([-0.0010 - 0.0010 * s - 0.0006 * np.sin(np.pi * s), -0.0278 - 0.0044 * s], 1)
    return strip2d(pts, 0.00036 * np.sin(np.pi * np.clip(s * 0.85 + 0.1, 0, 1)) ** 0.7 + 0.00002)


# the painted skin: the faint warmth under the eyes (|x|, z, sx, sz, strength) and the glint on the bridge of the nose
BLUSH = (0.048, -0.0190, 0.016, 0.008, 0.30)
NOSE_GLINT = ((-0.0008, -0.0195, 0.0026, 0.0060, 0.9), (-0.0009, -0.0207, 0.0012, 0.0027, 1.0))
