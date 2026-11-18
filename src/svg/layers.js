// Turns paint operations (in painting order) into layers that Arch draws bottom to top.
//
// Arch 12202+ can draw every object in creation order ("顺序绘图", ordered drawing, saved in the
// file), inequality fills included. So, unlike the flattening in regions.js, shapes are kept
// whole and simply stacked like the SVG paints them: nothing is cut out of what lies beneath,
// and no seam lines are needed between neighbouring pieces. A shape's outline is drawn once, by
// the shape itself, so there are no hairline gaps or notches where two pieces meet.
//
// Arch 2.0.05 uses standard source-over alpha blending. The generated scene disables smart
// colour inversion, so white and near-white paint keep their original RGB too. Preserve SVG
// colours and opacity directly; no background-dependent fitting or white cut-outs are needed.
// Legacy --flatten output is handled separately in regions.js.
// What still needs geometric compensation:
// - Arch interpolates each fill on its own sampling grid and cuts corners off within a cell. Where
//   a shape borders an earlier one (or a gradient band the next band), the lower one is extended
//   about a cell beneath the upper one, so no seam or notch shows along their common boundary.
//
// Layers that later opaque shapes hide completely are dropped (partly hidden ones are trimmed when
// that saves vertices), and a shape is merged into an earlier layer that Arch draws alike when
// nothing drawn in between overlaps it.
import * as P from './poly.js';
import {toHex} from './color.js';
import {gradientBands} from './regions.js';
import {joinedRuns, toIntPaths} from './outline.js';

const vertices = paths => paths.reduce((n, r) => n + r.length, 0);
const round3 = v => Math.round(v * 1000) / 1000;

// The transparent WebGL buffer stores premultiplied RGB. Arch 2.0.05 uses SRC_ALPHA /
// ONE_MINUS_SRC_ALPHA for RGB and ONE / ONE_MINUS_SRC_ALPHA for alpha. The browser then
// composites this buffer over the white canvas; unlike the old model, no extra white is added.
export const CLEAR = {r: 0, g: 0, b: 0, a: 0};
export const archOver = (c, s) => ({r: c.a * c.r + (1 - c.a) * s.r, g: c.a * c.g + (1 - c.a) * s.g,
    b: c.a * c.b + (1 - c.a) * s.b, a: c.a + (1 - c.a) * s.a});
export const archShown = s => ({r: s.r + 1 - s.a, g: s.g + 1 - s.a, b: s.b + 1 - s.a, a: 1});

// Kept as compatibility exports for callers of the converter helpers. With colour inversion
// disabled, every RGB colour is drawable and source-over works unchanged on any background.
export const nearestDrawable = color => ({...color});
export const fitLayer = target => ({...target});

// ops: from SvgRenderer.render(). Returns {layers, whitePieces, blended}; each layer is one of
//   {kind: 'fill', paths, hex, opacity, labels}
//   {kind: 'line', paths (open), hex, opacity, thickness, labels}
//   {kind: 'text', op}
// in drawing order. stepPx is Arch's sampling grid in pixels; seams: false skips the underlap.
// whitePieces and blended stay zero: colours no longer need legacy substitutions.
export function layerOps(ops, {q, tolerance, gradientSteps = 24, pxPerUnit, stepPx, minBandPx = 0, maxEdges = 400, seams = true}) {
    const pxInt = 1 / (pxPerUnit * q);                    // integer steps per screen pixel
    const margin = Math.max(2, Math.round(stepPx * pxInt));     // about one sampling cell
    const tiny = 0.25 * pxInt * pxInt;                    // a quarter of a square pixel
    let layers = [];
    let nextGroup = 1;

    const shapeOf = layer => layer.kind === 'fill' ? layer.paths : layer.footprint;
    const refresh = layer => {
        layer.bbox = P.bbox(layer.paths);
        if (layer.kind === 'line') layer.footprint = P.openOffset(layer.paths, layer.thickness / 2 * pxInt);
        layer.edges = vertices(layer.paths);
    };
    const prune = () => { layers = layers.filter(l => l.kind === 'text' || l.paths.length); };
    const intersects = (a, b) => P.overlaps(a.bbox, b.bbox) && P.area(P.intersect(shapeOf(a), shapeOf(b))) > tiny;
    const sameStyle = (a, b) => a.kind === b.kind && a.kind !== 'text' && a.hex === b.hex && a.opacity === b.opacity &&
        (a.kind !== 'line' || a.thickness === b.thickness);

    // Adds a layer on top, or merges it into an earlier layer of the same style when every layer
    // in between leaves it alone (so drawing it earlier changes nothing). Translucent shapes only
    // merge where they do not overlap: overlapping parts would otherwise blend once instead of twice.
    const place = layer => {
        refresh(layer);
        for (let i = layers.length - 1; i >= 0; i--) {
            const other = layers[i];
            if (sameStyle(other, layer) && other.edges + layer.edges <= maxEdges && (layer.opacity === 1 || !intersects(other, layer))) {
                if (layer.kind === 'fill') other.paths = P.union(other.paths, layer.paths);
                else other.paths = [...other.paths, ...layer.paths];
                other.labels.push(...layer.labels);
                refresh(other);
                return;
            }
            if (intersects(other, layer)) break;
        }
        layers.push(layer);
    };
    // draw: the colour and opacity Arch is given; shows: what the layer stands for (ordinary blending).
    const fillLayer = (paths, draw, shows, label, group = 0) => {
        const opacity = draw.a >= 0.9995 ? 1 : round3(draw.a);
        return {kind: 'fill', paths, draw: {...draw, a: opacity}, shows, hex: toHex(draw), opacity, labels: [label], group};
    };

    // Extends earlier opaque layers that border `paths` about a cell beneath it: only near its
    // outline (deeper inside nothing can show), only where the earlier layer shows, and only
    // along a shared boundary (not where the two outlines merely cross).
    const underlap = (paths, box, group) => {
        const inner = P.offset(paths, -margin);
        const zone = inner.length ? P.difference(paths, inner) : paths;
        if (!zone.length) return;
        const reach = P.grow(box, 2 * margin), area = P.rect(reach);
        layers.forEach((layer, index) => {
            if (layer.kind !== 'fill' || layer.opacity !== 1 || (group && layer.group === group) || !P.overlaps(layer.bbox, reach)) return;
            const near = P.intersect(layer.paths, area);
            if (!near.length) return;
            const extra = P.difference(P.intersect(P.offset(near, margin), zone), layer.paths);
            if (!extra.length) return;
            const candidates = P.components(extra).filter(rings => P.area(rings) >= 3 * margin * margin);
            if (!candidates.length) return;
            // Only where this layer shows: opaque layers above it may hide it there anyway.
            const local = P.grow(P.bbox(candidates.flat()), 2 * margin);
            let shown = P.intersect(near, P.rect(local));
            const above = layers.slice(index + 1).filter(l => l.kind === 'fill' && l.opacity === 1 && P.overlaps(l.bbox, local));
            if (shown.length && above.length) shown = P.difference(shown, P.normalize(above.flatMap(l => P.intersect(l.paths, P.rect(local)))));
            // A shared boundary: the growth touches the part of this layer outside `paths` that shows,
            // along at least two cells (a crossing point of the two outlines only touches it at a point).
            const outside = P.difference(shown, paths);
            if (!outside.length) return;
            const keep = candidates.filter(rings => P.area(P.intersect(P.offset(rings, 2), outside)) >= 8 * margin).flat();
            if (!keep.length) return;
            layer.paths = P.union(layer.paths, keep);
            refresh(layer);
        });
    };

    // What an opaque area hides: thin strokes beneath it are cut back to its outline, layers it
    // covers completely are dropped, and others lose what lies more than a cell inside it (they
    // keep reaching under its edge), when that does not cost vertices.
    // group: the gradient `paths` belongs to (its own bands are culled but never trimmed);
    // whole: `paths` is that whole gradient, which does not hide its own bands.
    const occlude = (paths, group = 0, whole = false) => {
        const box = P.bbox(paths);
        let inner = null;
        for (const layer of layers) {
            const own = group && layer.group === group;
            if (layer.kind === 'text' || (own && whole) || !P.overlaps(layer.bbox, box)) continue;
            if (layer.kind === 'line') { layer.paths = P.clipOpen(layer.paths, paths, false); refresh(layer); continue; }
            if (P.area(P.difference(layer.paths, paths)) <= tiny) { layer.paths = []; continue; }
            if (own) continue;
            inner ??= P.offset(paths, -margin);
            if (!inner.length) continue;
            const trimmed = P.difference(layer.paths, inner);
            if (vertices(trimmed) <= layer.edges && P.area(trimmed) < P.area(layer.paths) - tiny) { layer.paths = trimmed; refresh(layer); }
        }
        prune();
    };

    // An opaque colour Arch can draw: hides what it covers, its neighbours beneath reach under
    // its edge, then it goes on top.
    const addOpaque = (paths, color, label, group) => {
        occlude(paths, group);
        if (seams) underlap(paths, P.bbox(paths), group);
        place(fillLayer(paths, color, color, label, group));
    };

    const addFill = (paths, color, label, group = 0) => {
        if (!paths.length || color.a <= 0.002) return;
        if (color.a >= 0.998) return addOpaque(paths, {...color, a: 1}, label, group);
        place(fillLayer(paths, color, color, label, group));
    };

    const addLine = op => {
        const color = {...op.color, a: op.color.a * op.alpha};
        if (color.a <= 0.002 || !op.paths.length) return;
        const draw = color.a >= 0.998 ? {...color, a: 1} : color;
        const opacity = draw.a >= 0.9995 ? 1 : round3(draw.a);
        // Drawn as a point list: exact polylines, with their joins (and square or round caps) closed.
        const thickness = Math.min(20, Math.max(1, Math.round(op.widthPx * 10) / 10));
        const half = thickness / 2 * pxInt, caps = op.cap === 'square' || op.cap === 'round';
        const runs = [];
        for (const path of op.paths) {
            const closed = path.length > 3 && path[0].X === path[path.length - 1].X && path[0].Y === path[path.length - 1].Y;
            runs.push(...joinedRuns(path.map(p => [p.X, p.Y]), {half, closed, caps, minGap: 0.25 * pxInt}));
        }
        const paths = toIntPaths(runs);
        if (paths.length) place({kind: 'line', paths, hex: toHex(draw), opacity, thickness, labels: [op.label]});
    };

    for (const op of ops) {
        if (op.kind === 'line') addLine(op);
        else if (op.kind === 'text') {
            const width = 0.55 * op.size * op.content.length;
            const x = op.anchor === 'middle' ? op.x - width / 2 : op.anchor === 'end' ? op.x - width : op.x;
            const bbox = {minX: Math.floor(x / q), maxX: Math.ceil((x + width) / q), minY: Math.floor(op.y / q), maxY: Math.ceil((op.y + op.size) / q)};
            layers.push({kind: 'text', op, bbox, footprint: P.rect(bbox)});
        } else if (op.kind === 'fill') {
            if (op.paint.type === 'solid') addFill(op.paths, {...op.paint.color, a: op.paint.color.a * op.alpha}, op.label);
            else {
                const opaque = op.alpha >= 0.998 && op.paint.stops.every(s => s.color.a >= 0.998);
                const group = nextGroup++;
                const bands = gradientBands(op.paths, op.paint, {q, tolerance, steps: Math.max(1, gradientSteps), pxPerUnit, minBandPx,
                    layered: true, overlap: opaque && seams ? margin : 0, alpha: op.alpha});
                for (const band of bands) addFill(band.paths, band.color, op.label, group);
                // Together the bands hide what lies beneath the whole shape.
                if (opaque) occlude(op.paths, group, true);
            }
        }
    }
    return {layers: layers.map(({kind, paths, hex, opacity, thickness, labels, op}) =>
        kind === 'text' ? {kind, op} : kind === 'fill' ? {kind, paths, hex, opacity, labels} : {kind, paths, hex, opacity, thickness, labels}),
    whitePieces: 0, blended: 0};
}
