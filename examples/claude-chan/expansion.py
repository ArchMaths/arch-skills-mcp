"""How big Arch's expanded formulas will be, estimated from a scene script without building it.

Arch evaluates each object through a copy of its formulas with every custom function's body pasted in at each
use (saved as processedExprX/Y/Z): an argument used five times in a body is pasted five times, and functions
that call functions multiply. max(a,b) and min(a,b) become 0.5*((a)+(b)+-abs((a)-(b))), each argument twice, so
nested ones double at each level; a^b becomes __advanced__.pow(a,b). That copy is what Arch evaluates at every point of the object, so its length is a
fair measure of the object's drawing cost, and it is what makes a save file built without --lean large.

    python3 expansion.py claude-chan-3d.arch [--top 25] [--txt claude-chan-3d.txt]

--txt compares the estimate with the lengths a built save file holds.
"""
import re
import sys
import argparse

MATH = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'exp', 'ln', 'log', 'sqrt', 'abs', 'floor', 'ceil',
        'min', 'max', 'sign', 'round', 'sinh', 'cosh', 'tanh', 'pow'}
_ID = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def split_top(s, sep=','):
    """Split s at top-level separators (outside parentheses)."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append(''.join(cur))
    return out


def _close(s, i):
    """Index of the parenthesis closing the one at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('unbalanced parentheses')


class Expander:
    def __init__(self, funcs):
        self.funcs = funcs            # name -> (params, body)
        self._tmpl = {}

    def _template(self, name):
        """The function's body, expanded, as pieces between parameter uses: [text, param index, text, ...]."""
        if name not in self._tmpl:
            params, body = self.funcs[name]
            text = self.expand(body)
            pieces, last = [], 0
            for m in _ID.finditer(text):
                if m.group(0) in params and not text[m.end():m.end() + 1] == '(':
                    pieces += [text[last:m.start()], params.index(m.group(0))]
                    last = m.end()
            pieces.append(text[last:])
            self._tmpl[name] = pieces
        return self._tmpl[name]

    def expand(self, s):
        out, i = [], 0
        while i < len(s):
            m = _ID.match(s, i)
            if not m:
                out.append(s[i])
                i += 1
                continue
            name, j = m.group(0), m.end()
            if name in ('max', 'min') and j < len(s) and s[j] == '(':
                # Arch writes max(a,b) as 0.5*((a)+(b)+abs((a)-(b))): each argument twice, so nesting doubles
                k = _close(s, j)
                args = [self.expand(a) for a in split_top(s[j + 1:k])]
                acc = args[0]
                for b in args[1:]:
                    acc = f"(0.5*(({acc})+({b}){'+' if name == 'max' else '-'}abs(({acc})-({b}))))"
                out.append(acc)
                i = k + 1
            elif name in self.funcs and j < len(s) and s[j] == '(':
                k = _close(s, j)
                args = ['(' + self.expand(a) + ')' for a in split_top(s[j + 1:k])]
                out.append('(' + ''.join(p if isinstance(p, str) else args[p] for p in self._template(name)) + ')')
                i = k + 1
            else:
                out.append(name)
                i = j
        return ''.join(out)


def js_length(expanded, params):
    """Length of the expanded text as Arch writes it: pi as a number, Math.sin(...), a^b as
    __advanced__.pow(a,b), parameters renamed."""
    n = len(expanded) + 18 * expanded.count('^')
    for m in _ID.finditer(expanded):
        w = m.group(0)
        if w == 'pi':
            n += 15
        elif w in MATH and expanded[m.end():m.end() + 1] == '(':
            n += 5
        elif w in params:
            n += 18 - len(w)
    return n


def parse(text):
    """(functions, objects) of a scene script; objects are (ref, [formulas], [parameter names])."""
    funcs, objs = {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in '#@/':
            continue
        body, _, opts = line.partition(' | ')
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\(([A-Za-z0-9_,]*)\)=(.*)$', body)
        if m and not body.startswith('curve(') and not body.startswith('surface('):
            funcs[m.group(1)] = (m.group(2).split(','), m.group(3))
            continue
        ref = (re.search(r'ref=(\S+)', opts) or [None, '?'])[1]
        inner = body[body.index('(') + 1:_close(body, body.index('('))]
        parts = split_top(inner)
        if body.startswith('curve(') or len(parts) == 6:
            objs.append((ref, parts[:3], [parts[3]]))
        elif len(parts) == 9:
            objs.append((ref, parts[:3], [parts[3], parts[6]]))
    return funcs, objs


def measure(text):
    funcs, objs = parse(text)
    ex = Expander(funcs)
    return [(ref, sum(js_length(ex.expand(f), ps) for f in fs), sum(len(f) for f in fs)) for ref, fs, ps in objs]


def report(text, top=20, out=print):
    rows = measure(text)
    total = sum(r[1] for r in rows)
    out(f"expanded formulas: {total / 1e6:.2f} MB over {len(rows)} objects (written: {sum(r[2] for r in rows) / 1e6:.2f} MB)")
    for ref, n, w in sorted(rows, key=lambda r: -r[1])[:top]:
        out(f"  {n:9d}  x{n / max(w, 1):5.0f}  {ref}")
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('arch')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--txt', help='a save file built from the script, to compare with')
    a = ap.parse_args()
    rows = report(open(a.arch, encoding='utf-8').read(), a.top)
    if a.txt:
        import json
        ents = [e for e in json.load(open(a.txt, encoding='utf-8'))['entries'] if e.get('type') == 'function']
        real = [sum(len(e.get(k, '')) for k in ('processedExprX', 'processedExprY', 'processedExprZ')) for e in ents]
        if len(real) != len(rows):
            sys.exit(f'{len(real)} objects in the save file, {len(rows)} in the script')
        ratio = [r[1] / max(x, 1) for r, x in zip(rows, real)]
        print(f"estimate / saved: total {sum(r[1] for r in rows) / sum(real):.3f}, objects {min(ratio):.2f} .. {max(ratio):.2f}")
