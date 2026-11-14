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
    # name: (sheet x of the centre line, model coordinate along the sheet's x)
    'front': (820.0, lambda P: P[:, 0]),      # seen from the front: her right on the left
    'side': (1010.0, lambda P: P[:, 1]),      # her front on the left
    'back': (1225.0, lambda P: -P[:, 0]),     # seen from behind: her left on the left
}


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


def model_points(parts, ref=None, n=140):
    """Run generate.py for these parts and sample its surfaces (whose ref matches) with numpy."""
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
    pts = []
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
        if ref and not (r and re.fullmatch(ref, r.group(1))):
            continue
        X, Y, Z, u, u0, u1, v, v0, v1 = _split_top(tup.strip()[1:-1])
        lim = [float(eval(t.replace('^', '**'), ns)) for t in (u0, u1, v0, v1)]
        f = eval(f"lambda {u},{v}: ({X.replace('^', '**')},{Y.replace('^', '**')},{Z.replace('^', '**')})", ns)
        A, B = np.meshgrid(np.linspace(lim[0], lim[1], n), np.linspace(lim[2], lim[3], n), indexing='ij')
        P = np.stack([np.broadcast_to(np.asarray(c, float), A.shape) for c in f(A, B)], -1)
        for E0, E1 in ((P[1:], P[:-1]), (P[:, 1:], P[:, :-1])):    # the grid's edges, as Arch's mesh draws them
            E = np.stack([E0.reshape(-1, 3), E1.reshape(-1, 3)], 1)
            pts.append(E[np.all(np.isfinite(E), (1, 2))])
    if not pts:
        sys.exit(f'sheetfit: no surfaces of parts {parts!r} match ref {ref!r}')
    return np.concatenate(pts)


def model_rows(E, view, zs):
    """Per sheet row height: (left, right) where that level cuts the model's mesh edges; NaN if it misses."""
    h0, h1 = VIEWS[view][1](E[:, 0]), VIEWS[view][1](E[:, 1])
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
    ap.add_argument('mode', choices=('measure', 'check'))
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
    a = ap.parse_args()
    box = tuple(int(v) for v in a.box.split(','))
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
