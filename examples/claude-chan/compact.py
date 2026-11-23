"""Surfaces rewritten as short polynomials in their own parameters, where that changes how they look by less than a
tolerance, so that the save file opens faster.

Arch meshes every surface when it opens a file: at implicitPrecision 10 on a 32 x 32 grid of its parameters,
evaluating its formula five times at each grid point (the point and four neighbours, for the normal) and once in
each cell, about 6,100 times a surface. A formula that pastes in long custom functions (a highlight that repeats its
whole lock, the head's turn) or calls ^ (in Arch a function call, several times slower than * and +) costs the open
time in proportion. Here such a surface is sampled with the generator's own formulas, fitted, and written back as
nested polynomials (Horner form, no ^, each parameter pasted once a term), its parameters running over -1..1 in the
same order and direction (so the same grid, the same side lit). The written line is evaluated again and kept only if
it matches the original within the tolerance at Arch's mesh points and between them; otherwise the original stays.

    compact_lines(lines, specs, log)   rewrites the matching surface lines of a scene (Scene.out) in place

specs: {ref: spec} or {regex: spec} (a regex matched against the whole ref), spec a dict:
  tol     the largest move allowed (default 0.001)
  sub     (for a surface G(p, S(p, q)) whose dependence on its other parameter q is all in S, as a highlight band
          across a lock whose edges zigzag): (S, centre, half, p) with S an Arch expression in the parameters and p
          the name of the parameter G runs along; G is fitted as a polynomial in p and (S - centre) / half, and S is
          written into the formula as it is
  deg     the largest degrees tried (default (24, 14); with sub, (24, 4))
  basis   (p, [f1, f2, ...]) for a surface that is, across its parameter p, a sum of these functions of p (Arch
          expressions, as a lock's section: 1, cos(w), cos(w)*cos(w), lcs(w), ...) each times a polynomial in its
          other parameter: those polynomials are fitted (far fewer terms than a polynomial in both)
  split   if it does not fit whole, the most pieces it may be cut into (default 1: none), each a polynomial
  gain    rewritten only if Arch then evaluates at most this fraction of what it did (default 0.6)
  cuts    {parameter: [values]}: where to cut it first, if it is cut (where its formula has a crease)
"""
import math
import re
import numpy as np
from numpy.polynomial import chebyshev as _cmp_ch

_CMP_NUM = dict(sin=np.sin, cos=np.cos, tan=np.tan, sqrt=np.sqrt, exp=np.exp, abs=np.abs, ln=np.log, log=np.log10,
                sign=np.sign, atan=np.arctan, asin=np.arcsin, acos=np.arccos, floor=np.floor, ceil=np.ceil,
                pi=math.pi, max=np.maximum, min=np.minimum, tanh=np.tanh, sinh=np.sinh, cosh=np.cosh)


def _cmp_split(s):
    """Split at the top-level commas."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append(''.join(cur))
    return out


def _cmp_py(expr):
    return expr.replace('^', '**')


def _cmp_namespace(lines):
    """The custom functions defined in the lines, as numpy functions (a ^ of a negative number to a fraction is NaN,
    as in Arch, where the point is left out)."""
    ns = dict(_CMP_NUM)
    for line in lines:
        m = re.match(r'^(\w+)\(([^)]*)\)=(.*)$', line)
        if m and not line.startswith('curve('):
            name, args, body = m.groups()
            ns[name] = eval(f"lambda {args}: {_cmp_py(body)}", ns)
    return ns


def _cmp_parse(line):
    """(X, Y, Z, u, u0, u1, v, v0, v1, options) of a surface line, or None."""
    if not (line.startswith('(') and ' | ' in line):
        return None
    tup, opts = line.rsplit(' | ', 1)
    parts = _cmp_split(tup.strip()[1:-1])
    if len(parts) != 9:
        return None
    return parts + [opts]


def _cmp_eval(ns, X, Y, Z, u, v, U, V):
    f = eval(f"lambda {u},{v}: ({_cmp_py(X)},{_cmp_py(Y)},{_cmp_py(Z)})", ns)
    with np.errstate(all='ignore'):
        P = f(U, V)
    return np.stack([np.broadcast_to(np.asarray(c, float), np.broadcast(U, V).shape) for c in P], axis=-1)


def _cmp_num(c):
    s = np.format_float_positional(float(c), precision=8, unique=False, trim='-')
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    if s in ('-0', ''):
        s = '0'
    return f'({s})' if s.startswith('-') else s


def _cmp_horner(coefs, var):
    """c0 + var*(c1 + var*(...)) for power-basis coefficients (trailing zeros dropped)."""
    c = list(coefs)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    out = _cmp_num(c[-1])
    for a in reversed(c[:-1]):
        out = f"{_cmp_num(a)}+{var}*({out})"
    return out


def _cmp_to_power(C):
    """2-D Chebyshev coefficients (rows: first variable) to power-basis coefficients, rounded to 1e-8."""
    m, k = C.shape

    def M(n):
        A = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            p = _cmp_ch.cheb2poly(e)
            A[:len(p), i] = p
        return A
    return np.round(M(m) @ C @ M(k).T, 8)


def _cmp_formula(Pw, outer, inner):
    """A 2-D power-basis polynomial, Horner in the outer variable with Horner polynomials in the inner one."""
    rows = [_cmp_horner(Pw[i], inner) for i in range(Pw.shape[0])]
    while len(rows) > 1 and re.fullmatch(r'\(?-?0\)?', rows[-1]):
        rows.pop()
    out = rows[-1]
    for r in reversed(rows[:-1]):
        out = f"{r}+{outer}*({out})"
    return out


def _cmp_grid(n):
    return np.linspace(-1, 1, n)


def _cmp_fit_grid(F, A, B, m, k):
    """Least-squares Chebyshev fit of degrees (m, k) to F (na, nb, 3) sampled at the grid A x B."""
    Va = _cmp_ch.chebvander(A, m)
    Vb = _cmp_ch.chebvander(B, k)
    Pa, Pb = np.linalg.pinv(Va), np.linalg.pinv(Vb)
    return [Pa @ F[:, :, c] @ Pb.T for c in range(3)]


def compact_lines(lines, specs, log=print):
    """Rewrite the matching surfaces of the scene lines in place (see the module's notes); returns a report. (The
    trial fits of high degree overflow on the way to being turned down; numpy's warnings are off meanwhile.)"""
    with np.errstate(all='ignore'):
        return _cmp_lines(lines, specs, log)


def _cmp_lines(lines, specs, log):
    ns = _cmp_namespace(lines)
    items = [(re.compile(k), v) for k, v in specs.items()]
    report = []
    import expansion
    ex = expansion.Expander(expansion.parse('\n'.join(lines))[0])

    def cost(exprs, params):            # what Arch evaluates: the formulas with the custom functions pasted in
        return sum(expansion.js_length(ex.expand(f), params) for f in exprs)
    for idx, line in enumerate(lines):
        p = _cmp_parse(line)
        if p is None:
            continue
        X, Y, Z, u, u0, u1, v, v0, v1, opts = p
        r = re.search(r'ref=(\S+)', opts)
        ref = r.group(1) if r else ''
        spec = next((s for rx, s in items if rx.fullmatch(ref)), None)
        if spec is None:
            continue
        lim = [float(eval(_cmp_py(t), ns)) for t in (u0, u1, v0, v1)]
        res = _cmp_one(ns, X, Y, Z, u, v, lim, spec)
        if res is not None:
            (Xn, Yn, Zn), err, note, rg = res
            new = f"({Xn},{Yn},{Zn},{u},{rg[0]},{rg[1]},{v},{rg[2]},{rg[3]}) | {opts}"
        else:
            # in pieces (each meshed on its own grid, so no coarser than before): cut at the spec's own cuts first
            # (where its formula has a crease), then evenly, fewest pieces first
            got = None
            cuts = spec.get('cuts', {})
            base = [[lim[0]] + [c for c in cuts.get(u, []) if lim[0] < c < lim[1]] + [lim[1]],
                    [lim[2]] + [c for c in cuts.get(v, []) if lim[2] < c < lim[3]] + [lim[3]]]
            combos = sorted(((a_, b_) for a_ in range(1, 5) for b_ in range(1, 5)
                             if (len(base[0]) - 1) * a_ * (len(base[1]) - 1) * b_ <= spec.get('split', 1)),
                            key=lambda ab: (ab[0] * ab[1], ab))
            for a_, b_ in combos:
                if (len(base[0]) - 1) * a_ * (len(base[1]) - 1) * b_ == 1:
                    continue
                edges = [np.concatenate([np.linspace(e0, e1, n_ + 1)[:-1] for e0, e1 in zip(bs[:-1], bs[1:])] + [[bs[-1]]])
                         for bs, n_ in zip(base, (a_, b_))]
                pieces = []
                for i in range(len(edges[0]) - 1):
                    for j in range(len(edges[1]) - 1):
                        r_ = _cmp_one(ns, X, Y, Z, u, v, [edges[0][i], edges[0][i + 1], edges[1][j], edges[1][j + 1]], spec)
                        if r_ is None:
                            break
                        pieces.append(r_)
                    else:
                        continue
                    break
                else:
                    got = (f'{len(edges[0]) - 1}x{len(edges[1]) - 1}', pieces)
                    break
            if got is None:
                report.append((ref, len(line), None, 'kept: no fit within tolerance'))
                continue
            shape, pieces = got
            new = '\n'.join(f"({Xn},{Yn},{Zn},{u},{rg[0]},{rg[1]},{v},{rg[2]},{rg[3]}) | {opts.replace(f'ref={ref}', f'ref={ref}Pc{i + 1}')}"
                            for i, ((Xn, Yn, Zn), _, _, rg) in enumerate(pieces))
            err = max(p_[1] for p_ in pieces)
            note = f'{shape} pieces, ' + ' '.join(p_[2] for p_ in pieces)
        # kept only if Arch then evaluates at most `gain` of what it did (pieces each count in full: each has its grid)
        before = cost((X, Y, Z), [u, v])
        after = sum(cost(_cmp_split(nl.rsplit(' | ', 1)[0].strip()[1:-1])[:3], [u, v]) for nl in new.split('\n'))
        if after > spec.get('gain', 0.6) * before:
            report.append((ref, len(line), None, f'kept: rewritten it costs {after / max(before, 1):.2f} of it'))
            continue
        note += f', expansion {before / 1e3:.0f} -> {after / 1e3:.0f} KB'
        report.append((ref, len(line), len(new), f'{note}, error {err:.5f}'))
        lines[idx] = new
    done = [r for r in report if r[2]]
    log(f"  compact: {len(done)} of {len(report)} surfaces rewritten, {sum(r[1] for r in done) / 1e3:.0f} KB -> "
        f"{sum(r[2] for r in done) / 1e3:.0f} KB written" + ''.join(f"; {r[0]} {r[3]}" for r in report if not r[2]))
    return report


def _cmp_one(ns, X, Y, Z, u, v, lim, spec):
    tol = spec.get('tol', 0.001)
    ua, ub = (lim[0] + lim[1]) / 2, (lim[1] - lim[0]) / 2
    va, vb = (lim[2] + lim[3]) / 2, (lim[3] - lim[2]) / 2
    n = 97
    A = _cmp_grid(n)
    UU, VV = np.meshgrid(ua + ub * A, va + vb * A, indexing='ij')
    F = _cmp_eval(ns, X, Y, Z, u, v, UU, VV)
    if not np.all(np.isfinite(F)):
        return None                 # holes in it (Arch leaves those cells out): a polynomial would fill them
    # where Arch samples it: its 32 x 32 grid, and halfway between, to check the written formula against
    G = np.unique(np.r_[np.linspace(-1, 1, 32), np.linspace(-1, 1, 63), np.linspace(-1, 1, 41)])
    GU, GV = np.meshgrid(G, G, indexing='ij')
    Ref = _cmp_eval(ns, X, Y, Z, u, v, ua + ub * GU, va + vb * GV)
    if not np.all(np.isfinite(Ref)):
        return None
    sub = spec.get('sub')
    if sub:
        return _cmp_sub(ns, X, Y, Z, u, v, (ua, ub, va, vb), spec, tol, GU, GV, Ref)
    if spec.get('basis'):
        return _cmp_basis(ns, X, Y, Z, u, v, (ua, ub, va, vb), spec, tol, F, A, GU, GV, Ref)
    mmax, kmax = spec.get('deg', (24, 14))
    best = None
    # the degrees each way that the other way's largest allows, then the cheapest pair that fits
    def err_of(m, k):
        C = _cmp_fit_grid(F, A, A, m, k)
        E = np.stack([_cmp_ch.chebgrid2d(G, G, C[c]) for c in range(3)], axis=-1)
        return float(np.max(np.linalg.norm(E - Ref, axis=-1))), C
    need_m = next((m for m in range(1, mmax + 1) if err_of(m, kmax)[0] < 0.7 * tol), None)
    need_k = next((k for k in range(1, kmax + 1) if err_of(mmax, k)[0] < 0.7 * tol), None)
    if need_m is None or need_k is None:
        return None
    for m, k in sorted(((m, k) for m in range(need_m, mmax + 1) for k in range(need_k, kmax + 1)),
                       key=lambda mk: (mk[0] + 1) * (mk[1] + 1)):
        e, C = err_of(m, k)
        if e < 0.8 * tol:
            best = (m, k, C)
            break
    if best is None:
        return None
    m, k, C = best
    exprs = tuple(_cmp_formula(_cmp_to_power(C[c]), u, v) for c in range(3))
    W = _cmp_eval(ns, *exprs, u, v, GU, GV)              # the formula as written, against the original
    err = float(np.max(np.linalg.norm(W - Ref, axis=-1)))
    if not err < tol:
        return None
    return exprs, err, f'degrees {m}x{k}', ('-1', '1', '-1', '1')


def _cmp_sub(ns, X, Y, Z, u, v, sc, spec, tol, GU, GV, Ref):
    """G(u, s) with s = (S - centre) / half: fitted on the samples' own (u, s), written with S pasted in (in the new
    parameters)."""
    ua, ub, va, vb = sc
    S, cen, half, along = spec['sub']
    # S in the new parameters (each running -1..1)
    S_new = S
    for name, c0, c1 in ((u, ua, ub), (v, va, vb)):
        if abs(c0) > 1e-12 or abs(c1 - 1) > 1e-12:
            S_new = re.sub(rf'\b{name}\b', f'({_cmp_num(c0)}+{_cmp_num(c1)}*{name})', S_new)
    s_expr = f"(({S_new})-{_cmp_num(cen)})*{_cmp_num(1 / half)}"
    n = 121
    A = _cmp_grid(n)
    UU, VV = np.meshgrid(A, A, indexing='ij')
    F = _cmp_eval(ns, X, Y, Z, u, v, ua + ub * UU, va + vb * VV).reshape(-1, 3)
    with np.errstate(all='ignore'):
        s = eval(f"lambda {u},{v}: {_cmp_py(s_expr)}", ns)(UU, VV)
        sg = eval(f"lambda {u},{v}: {_cmp_py(s_expr)}", ns)(GU, GV)
    s, sg = np.broadcast_to(s, UU.shape).ravel(), np.broadcast_to(sg, GU.shape).ravel()
    a, ag = (UU.ravel(), GU.ravel()) if along == u else (VV.ravel(), GV.ravel())
    mmax, kmax = spec.get('deg', (24, 4))
    for m in range(2, mmax + 1):
        for k in range(1, kmax + 1):
            Vm = _cmp_ch.chebvander2d(a, s, [m, k])
            C = np.linalg.lstsq(Vm, F, rcond=None)[0]
            Vg = _cmp_ch.chebvander2d(ag, sg, [m, k])
            with np.errstate(all='ignore'):
                e = float(np.max(np.linalg.norm(Vg @ C - Ref.reshape(-1, 3), axis=-1)))
            if not np.isfinite(e):
                continue
            if e < 0.8 * tol:
                exprs = []
                for c in range(3):
                    Cc = C[:, c].reshape(m + 1, k + 1)
                    Pw = _cmp_to_power(Cc.T)            # rows: s (the outer, low degree), columns: u
                    exprs.append(_cmp_formula(Pw, f"({s_expr})", along))
                W = _cmp_eval(ns, *exprs, u, v, GU, GV)
                err = float(np.max(np.linalg.norm(W - Ref, axis=-1)))
                if err < tol:
                    return tuple(exprs), err, f'degrees {m} by {k} in its band', ('-1', '1', '-1', '1')
                break                       # rounding lost it: more degree in u will not help
    return None


def _cmp_basis(ns, X, Y, Z, u, v, sc, spec, tol, F, A, GU, GV, Ref):
    """sum over a <= n1, b <= n2 of x1(p)^a x2(p)^b P_ab(other), the P_ab fitted as polynomials (Chebyshev, least
    squares) in the other parameter (running -1..1); p keeps its own range, so x1 and x2 are written as given; the
    whole written as Horner in x2 of Horner in x1 of Horner in the other parameter."""
    ua, ub, va, vb = sc
    p, x1, n1, x2, n2 = spec['basis']
    other = v if p == u else u
    c0, c1 = (ua, ub) if p == u else (va, vb)
    UU, VV = np.meshgrid(A, A, indexing='ij')
    PP, OO = (UU, VV) if p == u else (VV, UU)
    GP, GO = (GU, GV) if p == u else (GV, GU)
    with np.errstate(all='ignore'):
        f1, f2 = (eval(f"lambda {p}: {_cmp_py(x)}", ns) for x in (x1, x2))
        X1, X2 = (np.broadcast_to(f(c0 + c1 * PP), PP.shape).ravel() for f in (f1, f2))
        G1, G2 = (np.broadcast_to(f(c0 + c1 * GP), GP.shape).ravel() for f in (f1, f2))
    pairs = [(a_, b_) for b_ in range(n2 + 1) for a_ in range(n1 + 1)]
    Phi = np.stack([X1 ** a_ * X2 ** b_ for a_, b_ in pairs], axis=1)
    Phig = np.stack([G1 ** a_ * G2 ** b_ for a_, b_ in pairs], axis=1)
    Fr = F.reshape(-1, 3)
    for m in range(4, spec.get('deg', (24, 24))[0] + 1):
        To = _cmp_ch.chebvander(OO.ravel(), m)
        M = (Phi[:, :, None] * To[:, None, :]).reshape(len(Fr), -1)
        C = np.linalg.lstsq(M, Fr, rcond=None)[0]
        Tg = _cmp_ch.chebvander(GO.ravel(), m)
        Mg = (Phig[:, :, None] * Tg[:, None, :]).reshape(GO.size, -1)
        with np.errstate(all='ignore'):
            e = float(np.max(np.linalg.norm(Mg @ C - Ref.reshape(-1, 3), axis=-1)))
        if e < 0.8 * tol:
            exprs = []
            for c in range(3):
                Cb = C[:, c].reshape(len(pairs), m + 1)
                poly = {ab: _cmp_horner(np.round(_cmp_ch.cheb2poly(cb), 8), other) for ab, cb in zip(pairs, Cb)}

                def nest(xs, items):
                    out = items[-1]
                    for it in reversed(items[:-1]):
                        out = f"{it}+{xs}*({out})"
                    return out
                inner = [nest(x1, [poly[(a_, b_)] for a_ in range(n1 + 1)]) for b_ in range(n2 + 1)]
                exprs.append(nest(x2, inner))
            rg = (_cmp_num(c0 - c1), _cmp_num(c0 + c1))
            rg = (rg + ('-1', '1')) if p == u else (('-1', '1') + rg)
            W = _cmp_eval(ns, *exprs, u, v, GU if p != u else c0 + c1 * GU, GV if p != v else c0 + c1 * GV)
            err = float(np.max(np.linalg.norm(W - Ref, axis=-1)))
            if err < tol:
                return tuple(exprs), err, f'{len(pairs)} terms of degree {m}', rg
            return None
    return None
