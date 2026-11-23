"""Long hair strands as drawn in 2D: a centre line in the back view with a half-width, built from a sweep, a wave and
a curl (hair.py's back strands are made from these). Pure numpy, so the design can be drawn over the user's picture
without the generator (see flow_ribbon).

A strand: flow(pts, widths, wave=None, curl=None)
  pts     the sweep: control points (x, z) from under the braid to where the curl starts (x + her left, z up)
  widths  half-width by height: [(z, hw), ...] from the top down (it eases between them)
  wave    (amp, wavelength, phase, z0, z1): a sideways wave of the sweep, 0 above z0 growing to amp at z1 and below,
          its phase along the strand (neighbours with the same phase flow together, as drawn)
  curl    (sense, radius, turn): the end curls round in the sheet of the hair: sense +1 turns anticlockwise in (x, z)
          (from straight down toward +x, her left), -1 clockwise; it turns `turn` radians along a spiral whose radius
          closes from `radius` to half of it, the strand narrowing through it to a point (its width where the curl
          starts is kept under 0.85 of the radius, so the inner edge never folds back)
Returns [(x, z, hw)] every 0.05 or so down the strand.
"""
import math
import numpy as np


def _nat_spline(d, P, s):
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


def flow(pts, widths, wave=None, curl=None, step=0.05):
    P = np.array(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = np.linspace(0, d[-1], int(math.ceil(d[-1] / step)) + 1)     # evenly (two points almost together would kink a spline)
    Q = _nat_spline(d, P, s)
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.c_[-T[:, 1], T[:, 0]]
    if wave:
        amp, lam, ph, z0, z1 = wave
        g = np.clip((z0 - Q[:, 1]) / (z0 - z1), 0, 1)
        g = g * g * (3 - 2 * g)
        Q = Q + N * (amp * g * np.sin(2 * math.pi * s / lam + ph))[:, None]
    zs, hs = zip(*widths)
    hw = np.interp(-Q[:, 1], -np.array(zs), np.array(hs))
    out = [(float(x), float(z), float(w)) for (x, z), w in zip(Q, hw)]
    if curl:
        sense, R, turn = curl
        t_end = Q[-1] - Q[-3]
        h0 = math.atan2(t_end[1], t_end[0])
        w0 = min(hw[-1], 0.85 * R)
        # ease the width down to w0 over the last 0.3 before the curl
        n_e = max(1, int(0.3 / step))
        for i in range(1, min(n_e, len(out)) + 1):
            x_, z_, w_ = out[-i]
            f = (i - 1) / n_e
            out[-i] = (x_, z_, w0 + (w_ - w0) * f if w_ > w0 else w_)
        x, z = Q[-1]
        n_c = max(8, int(turn * R / step))
        prev = 0.0
        for k in range(1, n_c + 1):
            th = turn * k / n_c
            r = R * (1 - 0.5 * th / turn)
            dth = th - prev
            hd = h0 + sense * (prev + 0.5 * dth)
            x += r * dth * math.cos(hd)
            z += r * dth * math.sin(hd)
            prev = th
            out.append((float(x), float(z), w0 * max(0.0, 1 - th / turn) ** 0.8))
    O = np.array(out)
    # the widths smoothed (the profile's corners and the ease into the curl would crease the surface), and closing to a
    # point over the last 0.3 without a corner
    k = np.exp(-0.5 * (np.arange(-6, 7) / 3) ** 2)
    O[:, 2] = np.convolve(np.pad(O[:, 2], 6, mode='edge'), k / k.sum(), 'valid')
    e = np.r_[np.cumsum(np.hypot(*np.diff(O[::-1, :2], axis=0).T))[::-1], 0]
    O[:, 2] *= np.minimum(1, e / 0.3)
    return [(round(float(x), 4), round(float(z), 4), round(max(float(w), 0.0), 4)) for x, z, w in O]


def flow_ribbon(strand):
    """The strand's two edges in (x, z), for drawing it in 2D."""
    Q = np.array(strand, float)
    T = np.gradient(Q[:, :2], axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
    N = np.c_[-T[:, 1], T[:, 0]]
    return Q[:, :2] + N * Q[:, 2:3], Q[:, :2] - N * Q[:, 2:3]
