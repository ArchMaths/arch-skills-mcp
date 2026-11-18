// Sample declarative SVG animation in isolated Chromium, then reuse the static vector converter.
import {ArchEngine} from '../engine.js';
import {convertSvg} from './index.js';
import {parseXml} from './xml.js';

export function animationOptions(options = {}) {
    const {duration, fps = 12, start = 0, maxFrames = 120} = options;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('duration must be a positive, finite number of seconds.');
    if (!Number.isInteger(fps) || fps < 1 || fps > 60) throw new Error('fps must be an integer from 1 to 60.');
    if (!Number.isFinite(start) || start < 0) throw new Error('start must be a finite number of seconds >= 0.');
    if (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 600) throw new Error('maxFrames must be an integer from 1 to 600.');
    const frames = Math.ceil(duration * fps);
    if (frames > maxFrames) throw new Error(`Animation needs ${frames} frames, exceeding maxFrames=${maxFrames}; reduce duration/fps or explicitly raise maxFrames (up to 600).`);
    if (!Number.isFinite(start + duration)) throw new Error('Animation end time must be finite.');
    return {duration, fps, start, maxFrames, frames};
}

// Returns frozen SVG markup per frame. Never runs input scripts or loads network/local resources.
export async function sampleSvgAnimation(source, options = {}) {
    const timing = animationOptions(options);
    const parsed = parseXml(source); // Validate before starting a browser.
    if (!parsed.attrs.xmlns && !parsed.qname?.includes(':')) {
        // Match the static converter's support for namespace-less SVG snippets.
        const escape = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        const serialize = node => 'text' in node ? escape(node.text) :
            `<${node.qname || node.name}${Object.entries(node.attrs).map(([k, v]) => ` ${k}="${escape(v)}"`).join('')}>${node.children.map(serialize).join('')}</${node.qname || node.name}>`;
        parsed.attrs.xmlns = 'http://www.w3.org/2000/svg';
        source = serialize(parsed);
    }
    const owner = options.browser ? null : new ArchEngine();
    let context;
    try {
        const browser = options.browser || await owner.browser();
        context = await browser.newContext({viewport: {width: 870, height: 800}, serviceWorkers: 'block'});
        const url = 'https://svg-animation.invalid/input.svg';
        await context.route('**/*', route => route.request().url() === url
            ? route.fulfill({contentType: 'image/svg+xml', body: source, headers: {
                'Content-Security-Policy': "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; sandbox allow-same-origin"
            }}) : route.abort());
        const page = await context.newPage();
        await page.goto(url, {waitUntil: 'load', timeout: 30000});
        const metadata = await page.evaluate(() => {
            const root = document.documentElement;
            if (root.localName !== 'svg' || document.querySelector('parsererror')) throw new Error('Invalid SVG XML document.');
            root.pauseAnimations();
            root.setCurrentTime(0);
            const warnings = [];
            if (document.querySelector('script, foreignObject') || [...document.querySelectorAll('*')].some(el => [...el.attributes].some(a => /^on/i.test(a.name))))
                warnings.push('SVG scripts, event handlers and foreignObject content are disabled; only declarative SMIL/CSS animation is sampled.');
            if ([...document.querySelectorAll('animate, set')].some(el => /[a-z]+\.(?:click|begin|end)|click|mouseover|indefinite/.test(el.getAttribute('begin') || '')))
                warnings.push('Event-triggered SMIL animations are not triggered; use clock-based begin times.');
            // Freeze CSS timelines as well as the SVG SMIL timeline.
            for (const a of document.getAnimations()) { a.pause(); a.currentTime = 0; }
            return {warnings};
        });
        const snapshots = [];
        for (let i = 0; i < timing.frames; i++) {
            const time = timing.start + i / timing.fps;
            snapshots.push(await page.evaluate(time => {
                const root = document.documentElement;
                root.setCurrentTime(time);
                for (const animation of document.getAnimations()) { animation.pause(); animation.currentTime = time * 1000; }
                // Force style/SMIL resolution at this exact time (no wall-clock sleeps).
                root.getBoundingClientRect();
                const copy = root.cloneNode(true);
                const originals = [root, ...root.querySelectorAll('*')];
                const clones = [copy, ...copy.querySelectorAll('*')];
                const properties = ['fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-width', 'stroke-opacity',
                    'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit', 'stroke-dasharray', 'stroke-dashoffset',
                    'opacity', 'display', 'visibility', 'color', 'clip-path', 'clip-rule', 'mask', 'filter',
                    'stop-color', 'stop-opacity', 'font-size', 'font-family', 'font-weight', 'font-style',
                    'text-anchor', 'paint-order', 'overflow', 'marker-start', 'marker-mid', 'marker-end', 'vector-effect', 'mix-blend-mode'];
                const lengths = ['x', 'y', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height',
                    'dx', 'dy', 'fx', 'fy', 'fr', 'startOffset', 'offset', 'gradientTransform', 'patternTransform', 'viewBox', 'preserveAspectRatio'];
                const localUrl = value => value.replace(/url\(["']?([^)'"\s]+)["']?\)/g, (whole, ref) => {
                    try {
                        const target = new URL(ref, document.URL), here = new URL(document.URL);
                        const hash = target.hash; target.hash = ''; here.hash = '';
                        return hash && target.href === here.href ? `url(${hash})` : whole;
                    } catch { return whole; }
                });
                originals.forEach((el, index) => {
                    const out = clones[index];
                    const style = getComputedStyle(el);
                    out.removeAttribute('style');
                    for (const attr of [...out.attributes]) if (/^on/i.test(attr.name)) out.removeAttribute(attr.name);
                    for (const prop of properties) {
                        const value = style.getPropertyValue(prop);
                        if (value) out.setAttribute(prop, localUrl(value));
                    }
                    // SVGAnimatedLength/Number/Rect/List expose SMIL's interpolated geometry.
                    for (const name of lengths) {
                        const animated = el[name];
                        if (!animated || !('animVal' in animated)) continue;
                        const value = animated.animVal;
                        if (typeof value === 'number') out.setAttribute(name, String(value));
                        else if (value && typeof value.value === 'number') {
                            // Keep percentages in gradient/objectBoundingBox coordinates.
                            out.setAttribute(name, value.valueAsString ?? String(value.value));
                        }
                        else if (value && typeof value.numberOfItems === 'number') {
                            out.setAttribute(name, Array.from({length: value.numberOfItems}, (_, n) => {
                                const item = value.getItem(n);
                                if (item.matrix) { const m = item.matrix; return `matrix(${m.a},${m.b},${m.c},${m.d},${m.e},${m.f})`; }
                                return item.value;
                            }).join(' '));
                        } else if (name === 'viewBox' && el.hasAttribute(name)) out.setAttribute(name, `${value.x} ${value.y} ${value.width} ${value.height}`);
                    }
                    // CSS geometry wins over presentation attributes. Chromium exposes animated d as path(...).
                    const geometry = {rect: ['x', 'y', 'width', 'height', 'rx', 'ry'], circle: ['cx', 'cy', 'r'],
                        ellipse: ['cx', 'cy', 'rx', 'ry'], image: ['x', 'y', 'width', 'height'], use: ['x', 'y', 'width', 'height']};
                    for (const prop of geometry[el.localName] || []) {
                        const value = style.getPropertyValue(prop);
                        if (/^[+-]?(?:\d*\.)?\d+(?:e[+-]?\d+)?(?:px|%)?$/i.test(value)) out.setAttribute(prop, value.replace(/px$/, ''));
                        else if (value === 'auto' && (prop === 'rx' || prop === 'ry')) out.removeAttribute(prop);
                    }
                    const d = style.getPropertyValue('d');
                    const path = /^path\(["'](.*)["']\)$/.exec(d);
                    if (path) out.setAttribute('d', path[1]);
                    if (el.animatedPoints) out.setAttribute('points', Array.from({length: el.animatedPoints.numberOfItems}, (_, n) => {
                        const p = el.animatedPoints.getItem(n); return `${p.x},${p.y}`;
                    }).join(' '));
                    // Relative CTMs include animateMotion, animateTransform and CSS transform-origin.
                    // Nested svg/use have intrinsic viewport/x/y transforms, handled by the static renderer.
                    if (/^(g|a|path|rect|circle|ellipse|line|polyline|polygon|text|image)$/.test(el.localName)) {
                        const matrix = el.getCTM?.(), parent = el.parentElement?.getCTM?.();
                        if (matrix && parent) {
                            const m = parent.inverse().multiply(matrix);
                            if ([m.a, m.b, m.c, m.d, m.e, m.f].every(Number.isFinite))
                                out.setAttribute('transform', `matrix(${m.a},${m.b},${m.c},${m.d},${m.e},${m.f})`);
                        }
                    } else if (style.transform && style.transform !== 'none') out.setAttribute('transform', style.transform);
                });
                // All styles are baked before removing sheets and animation elements.
                copy.querySelectorAll('animate, animateTransform, animateMotion, set, mpath, style, script, foreignObject').forEach(el => el.remove());
                copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
                // A fixed root viewport prevents per-frame auto-fitting and visible camera jitter.
                if (!copy.hasAttribute('viewBox')) {
                    const w = root.width.baseVal.value || 300, h = root.height.baseVal.value || 150;
                    copy.setAttribute('viewBox', `0 0 ${w} ${h}`);
                }
                return new XMLSerializer().serializeToString(copy);
            }, time));
        }
        return {...timing, snapshots, warnings: metadata.warnings};
    } finally {
        if (context) await context.close();
        if (owner) await owner.close();
    }
}

export async function convertSvgAnimation(source, options = {}) {
    const sampled = await sampleSvgAnimation(source, options);
    const warnings = new Set(sampled.warnings), info = new Set();
    const body = [];
    let first, staticTexts, entries = 0, edges = 0, maxDrawMs = 0;
    for (const [index, svg] of sampled.snapshots.entries()) {
        const converted = convertSvg(svg, options);
        first ||= converted;
        for (const warning of converted.warnings) warnings.add(warning);
        for (const note of converted.info) info.add(note);
        const texts = converted.script.split('\n').filter(line => line.startsWith('@text '));
        if (staticTexts === undefined) staticTexts = texts;
        else if (JSON.stringify(texts) !== JSON.stringify(staticTexts))
            throw new Error('Animated SVG text is not supported by Arch frame playback. Convert text to paths before importing; unchanged text is supported as a shared overlay.');
        entries += converted.stats.entries;
        edges += converted.stats.edges;
        maxDrawMs = Math.max(maxDrawMs, converted.stats.estimatedDrawMs);
        body.push(`# Frame ${index + 1}: SVG time ${sampled.start + index / sampled.fps} s`);
        for (const line of converted.script.split('\n')) {
            if (!line.trim() || line.startsWith('#') || /^@(title|view|set)\b/.test(line)) continue;
            if (!line.startsWith('@text ')) body.push(`${line} frame=${index + 1}`);
        }
    }
    const header = first.script.split('\n').filter(line => /^@(title|view|set)\b/.test(line));
    const script = [...header, `@frames total=${sampled.frames} fps=${sampled.fps} current=1`,
        '# Sampled SVG animation: press Play in Arch to start. Each frame contains editable vector objects.', ...body, ...staticTexts, ''].join('\n');
    entries += staticTexts.length;
    if (staticTexts.length) warnings.add('Unchanged SVG text is a shared overlay above all frames; convert text to paths to preserve animated text or exact painting order.');
    info.add(`Samples [${sampled.start}, ${sampled.start + sampled.duration}) s at ${sampled.fps} fps; no duplicate endpoint. Playback lasts ${sampled.frames / sampled.fps} s (rounded up to whole frames).`);
    info.add('Press the frame Play control in Arch. Scripts and external resources are disabled; filters and other static-converter limitations still apply.');
    warnings.add('Frame sampling duplicates vector geometry: file size and build time grow with frame count. Animated nested viewports, use instances and text layout may be approximate.');
    return {script, stats: {frames: sampled.frames, fps: sampled.fps, start: sampled.start, duration: sampled.duration,
        playbackDuration: sampled.frames / sampled.fps, entries, edges, estimatedDrawMs: maxDrawMs}, warnings: [...warnings], info: [...info]};
}

export function formatAnimationConversion({stats: s, warnings, info}, {scriptPath} = {}) {
    return [`Converted SVG animation: ${s.frames} frames at ${s.fps} fps, ${s.entries} entries, ${s.edges} edges total.`,
        ...(scriptPath ? [`Scene script: ${scriptPath}`] : []),
        ...warnings.map(w => `Warning: ${w}`), ...info.map(note => `Note: ${note}`)].join('\n');
}
