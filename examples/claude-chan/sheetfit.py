#!/usr/bin/env python3
"""Measure the design sheet and check the model against it, in numbers.

The sheet's full-body views (front, side, back) are drawn straight on, so the outline of a part there is
the outline of its 3D shape projected along one axis. This tool reads those outlines row by row, with
sub-pixel edges, in model units; and it projects the generator's surfaces the same way (evaluated in
Python, no Arch run) and reports how far apart they are. Use it before and after modelling a part:

    # what the sheet shows: left and right edges, width and centre at each height
    python3 sheetfit.py measure --view side --box 985,380,1035,410
    # the model against the sheet (one boot; side view aligned at the toe tip)
    python3 sheetfit.py check --parts boots --ref 'boot.*R' --view front --box 798,384,820,407
    python3 sheetfit.py check --parts boots --ref 'boot.*R' --view side --box 985,385,1035,407 --align min
    # the same comparison drawn over the sheet, for review
    python3 sheetfit.py check ... --overlay review.png
    # layers: colour classes (light, black, orange) on the sheet and on the model drawn with hidden parts
    # removed; overlaps, and with --map n the two class maps as text, one character per n pixels
    python3 sheetfit.py layers --parts skirt,body --view front --box 735,300,905,372 --map 3

--box is x0,y0,x1,y1 in sheet pixels around the part in that view (keep other parts out of it, and stop a
row or two above the ground line, which is drawn thick); only the rows inside it are compared, so parts
hidden on the sheet (under the skirt) are not. The model is cut through the edges of a fine grid on each
surface, as Arch meshes it, so flat faces count too. The sheet is
--sheet or $CLAUDE_CHAN_SHEET. Needs numpy, scipy and opencv-python.
"""
import os, sys, io, re, math, argparse, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Calibration of the full-body views (f3937d17 sheet, 2048 x 1487): chin to sole on the front view is
# 10.96 model units; the soles stand on row 409; the views' centre lines.
U = 10.96 / (409 - 130.5)          # model units per sheet pixel
GROUND = 409.0
VIEWS = {
    # name: (sheet x of the centre line, camera azimuth as in Arch: -90 looks at her face)
    'front': (820.0, -90.0),      # her right on the left
    'side': (1010.0, 0.0),        # from her left, her front on the left (the dress there is drawn turned: --az -30)
    'back': (1225.0, 90.0),       # her left on the left
}
AZ = {}                           # per-view azimuth overrides (--az)


def view_axes(view):
    """(right, toward-camera) unit vectors of the orthographic view."""
    az = math.radians(AZ.get(view, VIEWS[view][1]))
    return np.array([-math.sin(az), math.cos(az), 0.0]), np.array([math.cos(az), math.sin(az), 0.0])


def h_of(view, P):
    return P @ view_axes(view)[0]


# ------------------------------------------------------------------------------------------- sheet
def load_sheet(path):
    import cv2
    path = path or os.environ.get('CLAUDE_CHAN_SHEET')
    if not path or not os.path.exists(path):
        sys.exit('sheetfit: give the sheet image with --sheet or $CLAUDE_CHAN_SHEET')
    return cv2.imread(path).astype(float).min(2)       # darkest channel: outlines and colours vs. white paper


def sheet_rows(g, box, view, thr=215.0):
    """Per row in the box: (z, left, right) in model units, edges where the row first/last gets darker than thr."""
    x0, y0, x1, y1 = box
    axis = VIEWS[view][0]
    out = []
    for y in range(y0, y1 + 1):
        r = g[y, x0:x1 + 1]
        dark = np.nonzero(r < thr)[0]
        if not len(dark):
            continue
        i, j = dark[0], dark[-1]
        l = x0 + i - (1 - (r[i - 1] - thr) / (r[i - 1] - r[i])) if i > 0 else float(x0)
        rr = x0 + j + (1 - (r[j + 1] - thr) / (r[j + 1] - r[j])) if j < len(r) - 1 else float(x1)
        out.append(((GROUND - y) * U, (l - axis) * U, (rr - axis) * U))
    return np.array(out)


# ------------------------------------------------------------------------------------------- model
def _split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        depth += ch in '(['
        depth -= ch in ')]'
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    return out + [cur]


def model_surfaces(parts, ref=None, skip=None):
    """Run generate.py for these parts; its surfaces (whose ref matches) as numpy functions:
    [(ref, colour, f(u, v) -> (X, Y, Z), (u0, u1, v0, v1))]."""
    sys.path.insert(0, HERE)
    argv = sys.argv
    sys.argv = ['generate.py', '-o', os.devnull, '--parts', parts, '--prec', '3']
    g = {'__name__': '__gen__', '__file__': os.path.join(HERE, 'generate.py')}
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            exec(compile(open(os.path.join(HERE, 'generate.py'), encoding='utf-8').read(), 'generate.py', 'exec'), g)
    finally:
        sys.argv = argv
    ns = dict(sin=np.sin, cos=np.cos, tan=np.tan, sqrt=np.sqrt, exp=np.exp, abs=np.abs, ln=np.log, log=np.log10,
              sign=np.sign, atan=np.arctan, asin=np.arcsin, acos=np.arccos, floor=np.floor, ceil=np.ceil,
              pi=math.pi, max=np.maximum, min=np.minimum)
    surfs = []
    np.seterr(all='ignore')
    for line in g['sc'].out:
        m = re.match(r'^(\w+)\(([^)]*)\)=(.*)$', line)
        if m and not line.startswith('curve('):
            name, args, body = m.groups()
            ns[name] = eval(f"lambda {args}: {body.replace('^', '**')}", ns)
            continue
        if not (line.startswith('(') and ' | ' in line):
            continue
        tup, opts = line.rsplit(' | ', 1)
        r = re.search(r'ref=(\S+)', opts)
        name = r.group(1) if r else ''
        if (ref and not re.fullmatch(ref, name)) or (skip and re.fullmatch(skip, name)):
            continue
        col = re.search(r'color=(\S+)', opts)
        X, Y, Z, u, u0, u1, v, v0, v1 = _split_top(tup.strip()[1:-1])
        lim = [float(eval(t.replace('^', '**'), ns)) for t in (u0, u1, v0, v1)]
        f = eval(f"lambda {u},{v}: ({X.replace('^', '**')},{Y.replace('^', '**')},{Z.replace('^', '**')})", ns)
        surfs.append((name, col.group(1) if col else '#888888', f, lim))
    if not surfs:
        sys.exit(f'sheetfit: no surfaces of parts {parts!r} match ref {ref!r}')
    return surfs


def _grid(f, lim, nu, nv):
    A, B = np.meshgrid(np.linspace(lim[0], lim[1], nu), np.linspace(lim[2], lim[3], nv), indexing='ij')
    return np.stack([np.broadcast_to(np.asarray(c, float), A.shape) for c in f(A, B)], -1)


def model_points(parts, ref=None, n=140):
    """The edges of a fine grid on each matching surface, as Arch's mesh draws them: (n, 2, 3)."""
    pts = []
    for _, _, f, lim in model_surfaces(parts, ref):
        P = _grid(f, lim, n, n)
        for E0, E1 in ((P[1:], P[:-1]), (P[:, 1:], P[:, :-1])):
            E = np.stack([E0.reshape(-1, 3), E1.reshape(-1, 3)], 1)
            pts.append(E[np.all(np.isfinite(E), (1, 2))])
    return np.concatenate(pts)


def model_rows(E, view, zs):
    """Per sheet row height: (left, right) where that level cuts the model's mesh edges; NaN if it misses."""
    h0, h1 = h_of(view, E[:, 0]), h_of(view, E[:, 1])
    z0, z1 = E[:, 0, 2], E[:, 1, 2]
    lo, hi = np.minimum(z0, z1), np.maximum(z0, z1)
    out = []
    for zr in zs:
        m = (lo <= zr) & (hi >= zr)
        if not m.any():
            out.append((np.nan, np.nan))
            continue
        dz = z1[m] - z0[m]
        t = np.where(np.abs(dz) > 1e-12, (zr - z0[m]) / np.where(np.abs(dz) > 1e-12, dz, 1), 0.5)
        h = h0[m] + t * (h1[m] - h0[m])
        out.append((h.min(), h.max()))
    return np.array(out)


# ------------------------------------------------------------------------------------------- layers
CLS = '.ckog-'    # background, light (cream or white), black, orange (fabric, and hair), gold, other (outlines, shading)


def classify(rgb):
    """Colour class of each pixel (rgb floats 0-255, last axis)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    lab = np.full(r.shape, 5, np.int8)
    lab[(mn >= 165) & ((mx - mn) < 0.25 * mx)] = 1
    lab[(mn >= 246) & (mx - mn <= 6)] = 0
    lab[mx < 75] = 2
    lab[(r > 120) & (r - b > 0.45 * r) & (g < 0.75 * r)] = 3
    lab[(r > 150) & (g >= 0.75 * r) & (b < 0.55 * r)] = 4
    return lab


def sheet_labels(path, box):
    import cv2
    x0, y0, x1, y1 = box
    return classify(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)[y0:y1 + 1, x0:x1 + 1].astype(float))


def _hex_rgb(h):
    h = {'black': '000000', 'white': 'ffffff'}.get(h, h).lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def model_labels(surfs, view, box, k=3, max_pts=3_000_000):
    """Colour class seen at each sheet pixel of the box, the model drawn with hidden parts removed (k x k samples
    per pixel, majority)."""
    x0, y0, x1, y1 = box
    axis = VIEWS[view][0]
    rv, dv = view_axes(view)
    W, H = (x1 - x0 + 1) * k, (y1 - y0 + 1) * k
    zbuf = np.full(W * H, -np.inf)
    lab = np.zeros(W * H, np.int8)

    def sub(P):
        return (axis + (P @ rv) / U - (x0 - 0.5)) * k, (GROUND - P[..., 2] / U - (y0 - 0.5)) * k

    for name, col, f, lim in surfs:
        c = classify(_hex_rgb(col)[None])[0]
        Pc = _grid(f, lim, 17, 17)
        px, py = sub(Pc)
        if not np.isfinite(px).any() or np.nanmax(px) < 0 or np.nanmin(px) > W or np.nanmax(py) < 0 or np.nanmin(py) > H:
            continue
        seg_u = np.nanmax(np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)))
        seg_v = np.nanmax(np.hypot(np.diff(px, axis=1), np.diff(py, axis=1)))
        nu, nv = int(np.clip(seg_u * 16 * 1.4, 4, 3000)), int(np.clip(seg_v * 16 * 1.4, 4, 3000))
        if nu * nv > max_pts:
            s_ = math.sqrt(max_pts / (nu * nv))
            nu, nv = max(4, int(nu * s_)), max(4, int(nv * s_))
        P = _grid(f, lim, nu, nv).reshape(-1, 3)
        P = P[np.all(np.isfinite(P), 1)]
        px, py = sub(P)
        ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
        m = (ix >= 0) & (iy >= 0) & (ix < W) & (iy < H)
        idx, dep = iy[m] * W + ix[m], P[m] @ dv
        if not len(idx):
            continue
        o = np.lexsort((-dep, idx))
        idx, dep = idx[o], dep[o]
        first = np.r_[True, idx[1:] != idx[:-1]]
        idx, dep = idx[first], dep[first]
        win = dep > zbuf[idx]
        zbuf[idx[win]] = dep[win]
        lab[idx[win]] = c
    lab = lab.reshape(H, W).reshape(H // k, k, W // k, k).transpose(0, 2, 1, 3).reshape(H // k, W // k, k * k)
    counts = np.stack([(lab == c).sum(-1) for c in range(len(CLS))], -1)
    return counts.argmax(-1).astype(np.int8)


def text_map(L, step):
    """One character per step x step block: the commonest class that is not 'other'."""
    rows = []
    for y in range(0, L.shape[0], step):
        s = ''
        for x in range(0, L.shape[1], step):
            b = L[y:y + step, x:x + step].ravel()
            b = b[b != 5]
            s += CLS[np.bincount(b, minlength=6).argmax()] if len(b) else '-'
        rows.append(s)
    return rows


def runs(L, y, box, view, cls=(1, 2, 3)):
    """Runs of the classes along row y of a label map, in model units."""
    x0 = box[0]
    axis = VIEWS[view][0]
    row = L[y]
    out = []
    i = 0
    while i < len(row):
        j = i
        while j < len(row) and row[j] == row[i]:
            j += 1
        if row[i] in cls and j - i >= 2:
            out.append(f"{CLS[row[i]]}[{(x0 + i - 0.5 - axis) * U:.2f},{(x0 + j - 0.5 - axis) * U:.2f}]")
        i = j
    return ' '.join(out)


def report_layers(view, box, S, M, step, show_map):
    valid = S != 5
    print(f"{view}: box {box}, z {(GROUND - box[3]) * U:.2f}..{(GROUND - box[1]) * U:.2f}")
    if M is not None:
        so, mo = (S != 0) & valid, (M != 0) & valid
        print(f"  outline overlap {np.sum(so & mo) / max(1, np.sum(so | mo)):.3f}   pixels agreeing {np.mean((S == M)[valid]):.3f}")
        for c in (1, 2, 3):
            a_, b_ = (S == c) & valid, (M == c) & valid
            print(f"  {CLS[c]}: sheet {a_.sum():5d} px  model {b_.sum():5d} px  overlap {np.sum(a_ & b_) / max(1, np.sum(a_ | b_)):.3f}")
    if show_map:
        ts = text_map(S, step)
        tm = text_map(M, step) if M is not None else [''] * len(ts)
        for i, (a_, b_) in enumerate(zip(ts, tm)):
            z = (GROUND - (box[1] + i * step + step / 2 - 0.5)) * U
            print(f"  {z:5.2f} {a_}  {b_}")


# ------------------------------------------------------------------------------------------- reports
def report_measure(S, every):
    print(f"{'z':>6} {'left':>7} {'right':>7} {'width':>6} {'centre':>7}")
    for z, l, r in S[::every]:
        print(f"{z:6.3f} {l:7.3f} {r:7.3f} {r - l:6.3f} {(l + r) / 2:7.3f}")


def report_check(view, S, M, shift, every):
    ok = np.isfinite(M[:, 0])
    if not ok.any():
        sys.exit('sheetfit: the model has nothing at these heights')
    z, sl, sr = S[ok, 0], S[ok, 1], S[ok, 2]
    ml, mr = M[ok, 0], M[ok, 1]
    dl, dr = ml - sl, mr - sr
    inter = np.clip(np.minimum(sr, mr) - np.maximum(sl, ml), 0, None)
    union = np.maximum(sr, mr) - np.minimum(sl, ml)
    iou = inter.sum() / union.sum()
    miss = int((~ok).sum())
    print(f"{view}: {ok.sum()} rows{f' ({miss} where the model has nothing)' if miss else ''}; model shifted {shift:+.3f}; "
          f"overlap {iou:.3f}")
    for name, d in (('left', dl), ('right', dr), ('width', dr - dl)):
        k = np.argmax(np.abs(d))
        print(f"  {name:5s} mean {np.mean(np.abs(d)):.3f}  max {d[k]:+.3f} at z {z[k]:.2f}")
    print(f"  {'z':>5} {'sheet':>15} {'model':>15} {'dl':>6} {'dr':>6}")
    for i in range(0, len(z), every):
        print(f"  {z[i]:5.2f} {sl[i]:7.3f}{sr[i]:8.3f} {ml[i]:7.3f}{mr[i]:8.3f} {dl[i]:+6.3f} {dr[i]:+6.3f}")


def overlay(g_path, box, view, S, M, out):
    import cv2
    x0, y0, x1, y1 = box
    axis = VIEWS[view][0]
    img = cv2.imread(g_path)[y0:y1 + 1, x0:x1 + 1]
    k = 12
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    for (z, sl, sr), (ml, mr) in zip(S, M):
        y = int(round((GROUND - z / U - y0 + 0.5) * k))
        for v, col in ((sl, (255, 120, 0)), (sr, (255, 120, 0)), (ml, (40, 40, 220)), (mr, (40, 40, 220))):
            if np.isfinite(v):
                cv2.circle(img, (int(round((axis + v / U - x0 + 0.5) * k)), y), 3, col, -1)
    cv2.putText(img, f'{view}: blue sheet, red model', (6, 18), 0, 0.5, (60, 60, 60), 1)
    cv2.imwrite(out, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('measure', 'check', 'layers'))
    ap.add_argument('--view', choices=tuple(VIEWS), required=True)
    ap.add_argument('--box', required=True, help='x0,y0,x1,y1 in sheet pixels')
    ap.add_argument('--sheet')
    ap.add_argument('--thr', type=float, default=215, help='darkness that counts as the outline (0-255)')
    ap.add_argument('--parts', default='all', help='generate.py parts to evaluate (check)')
    ap.add_argument('--ref', help='regex: only surfaces whose ref matches (check)')
    ap.add_argument('--align', choices=('none', 'min', 'max', 'centre'), default='none',
                    help='shift the model sideways to line up its left edge, right edge or centre with the sheet')
    ap.add_argument('--every', type=int, default=3, help='print every n-th row')
    ap.add_argument('--overlay', help='also draw the comparison over the sheet to this png')
    ap.add_argument('--az', type=float, help='camera azimuth for this view (the side view of the dress is drawn turned)')
    ap.add_argument('--skip', help='regex: leave out surfaces whose ref matches (layers)')
    ap.add_argument('--map', type=int, default=0, help='layers: print the class maps, one character per n pixels')
    ap.add_argument('--runs', type=int, default=0, help='layers: print the class runs on every n-th row')
    a = ap.parse_args()
    box = tuple(int(v) for v in a.box.split(','))
    if a.az is not None:
        AZ[a.view] = a.az
    if a.mode == 'layers':
        path = a.sheet or os.environ.get('CLAUDE_CHAN_SHEET')
        S = sheet_labels(path, box)
        M = model_labels(model_surfaces(a.parts, a.ref, a.skip), a.view, box) if a.parts != 'none' else None
        report_layers(a.view, box, S, M, a.map, a.map > 0)
        if a.runs:
            for y in range(0, S.shape[0], a.runs):
                z = (GROUND - box[1] - y) * U
                print(f"  {z:5.2f} sheet {runs(S, y, box, a.view)}")
                if M is not None:
                    print(f"        model {runs(M, y, box, a.view)}")
        return
    g = load_sheet(a.sheet)
    S = sheet_rows(g, box, a.view, a.thr)
    if a.mode == 'measure':
        report_measure(S, a.every)
        return
    P = model_points(a.parts, a.ref)
    M = model_rows(P, a.view, S[:, 0])
    ok = np.isfinite(M[:, 0])
    shift = {'none': 0.0,
             'min': np.min(S[ok, 1]) - np.min(M[ok, 0]),
             'max': np.max(S[ok, 2]) - np.max(M[ok, 1]),
             'centre': np.mean((S[ok, 1] + S[ok, 2]) / 2 - (M[ok, 0] + M[ok, 1]) / 2)}[a.align] if ok.any() else 0.0
    M = M + shift
    report_check(a.view, S, M, shift, a.every)
    if a.overlay:
        overlay(a.sheet or os.environ.get('CLAUDE_CHAN_SHEET'), box, a.view, S, M, a.overlay)


if __name__ == '__main__':
    main()
