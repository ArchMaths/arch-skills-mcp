"""Local, tangent-continuous roots for back locks; independent of the Arch generator."""
import numpy as np


def attach_root(points, anchor, tangent, end_height=10.95):
    """Replace only a lock's root by a cubic Hermite bridge, in height.

    The first three columns are XYZ, followed by half-width, thickness and twist.
    Width/thickness start with zero value and slope: the new lock grows out of the
    existing hair, rather than laying a full-thickness cut edge over it. Below the
    bridge the old samples are retained exactly. Both XYZ endpoint derivatives are
    constrained; the upper derivative is the actual crown lock's tangent.
    """
    points = np.asarray(points, float)
    anchor, tangent = np.asarray(anchor, float), np.asarray(tangent, float)
    if tangent[2] >= -1e-5:
        raise ValueError('The crown tangent must point downwards')
    # Leave at least 0.12 of unchanged path even for a short filler strand.
    end_height = max(end_height, points[-1, 2] + 0.12)
    end = int(np.nonzero(points[:, 2] <= end_height)[0][0])
    end = min(max(end, 2), len(points) - 2)
    q = points[end]
    height = anchor[2] - q[2]
    if height <= 0:
        raise ValueError('The attachment must be above the bridge endpoint')
    dz = points[end + 1, 2] - points[end - 1, 2]
    slope = (points[end + 1] - points[end - 1]) / dz
    a = np.r_[anchor, np.zeros(points.shape[1] - 3)]
    da = np.r_[tangent / (-tangent[2]), np.zeros(points.shape[1] - 3)] * height
    db = -slope * height
    u = np.linspace(0, 1, max(40, end * 3))
    h00, h10 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u
    h01, h11 = -2*u**3 + 3*u**2, u**3 - u**2
    bridge = h00[:, None]*a + h10[:, None]*da + h01[:, None]*q + h11[:, None]*db
    # Positive entry with exact end slope. A free Hermite width can go negative
    # even when all old widths are positive. Localising the exponential bounds
    # its factor by exp(1/e), including for a steep terminal slope.
    b = 6*u**5 - 15*u**4 + 10*u**3
    for j in range(3, points.shape[1]):
        if j in (3, 4):
            if q[j] <= 0:
                raise ValueError('The bridge must end before the width/thickness vanishes')
            k = db[j]/q[j] - 3
            bridge[:, j] = q[j]*u**3*np.exp(k*(u-1)*u**max(1.0, abs(k)))
        else:
            bridge[:, j] = b*q[j] + (u**3 * (u-1)) * db[j]
    bridge[0] = a
    bridge[-1] = q
    return np.r_[bridge[:-1], points[end:]], end, da[:3] / height


def fit_c1(t, values, degree, start_derivative=None):
    """Least squares in a Chebyshev null space, constrained at both endpoint jets.

    Using actual endpoint constraints (not a position-only post-fit offset) keeps
    a tangent root and all polynomial piece seams smooth after fitting for Arch.
    """
    t, values = np.asarray(t), np.asarray(values)
    from numpy.polynomial import chebyshev as ch, polynomial as po
    A = ch.chebvander(t, degree)
    basis = np.eye(degree + 1)
    dA = np.array([ch.chebval([-1, 1], ch.chebder(c)) for c in basis]).T
    C = np.r_[A[[0, -1]], dA]
    # A quadratic endpoint estimate is insensitive to one interpolation cell.
    left = (np.polynomial.polynomial.polyfit(t[:7]-t[0], values[:7], 2)[1]
            if start_derivative is None else np.asarray(start_derivative, float))
    right = np.polynomial.polynomial.polyfit(t[-7:]-t[-1], values[-7:], 2)[1]
    target = np.r_[values[[0, -1]], [left, right]]
    particular = np.linalg.lstsq(C, target, rcond=None)[0]
    _, _, vh = np.linalg.svd(C, full_matrices=True)
    free = vh[4:].T
    coef = particular + free @ np.linalg.lstsq(A @ free, values-A @ particular, rcond=None)[0]
    cs = [np.round(ch.Chebyshev(coef[:, i]).convert(kind=po.Polynomial).coef, 5) for i in range(values.shape[1])]
    fit = np.array([po.polyval(t, c) for c in cs]).T
    return cs, fit, float(np.max(np.linalg.norm(fit-values, axis=1)))


def legacy_end_jets(sig_total, series_at, degree=20):
    """Return the old raw final-piece coefficients and selected segment count.

    ``series_at(st)`` supplies ``[(C, U, V), ...]`` for the parent strand and
    its children; each vector array has shape ``(len(st), 3)``.  The returned
    coefficients have the same strand/vector/XYZ nesting, with ascending-power
    coefficient arrays at the leaves.  Their parameter is the final segment's
    local t in [-1, 1], not the global arc-length parameter.

    Reproduce the legacy unconstrained Chebyshev fits, including five-decimal
    rounding and the raw per-vector error threshold.  Old C0 seam corrections
    vanish with their first derivative at t=1, so they cannot affect these end
    jets.  No C1 refit or seam correction is applied.  As in the old search,
    return the eight-segment result if no candidate meets the strict threshold.
    """
    from numpy.polynomial import Chebyshev, Polynomial

    t = np.linspace(-1.0, 1.0, 1201)
    start = 1 if sig_total < 2.5 else 2
    for nseg in range(start, 9):
        errors = []
        for k in range(nseg):
            st = sig_total * (k + (t + 1.0)/2.0) / nseg
            last = []
            for series in series_at(st):
                strand = []
                for values in series:
                    values = np.asarray(values)
                    cs = [np.round(Chebyshev.fit(t, values[:, i], degree)
                                   .convert(kind=Polynomial).coef, 5)
                          for i in range(3)]
                    fit = np.column_stack([Polynomial(c)(t) for c in cs])
                    errors.append(float(np.max(np.linalg.norm(fit-values, axis=1))))
                    strand.append(cs)
                last.append(strand)
        if all(error < 0.006 for error in errors):
            break
    return last, nseg
