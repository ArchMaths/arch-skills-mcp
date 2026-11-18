import {test, after} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, mkdtempSync, existsSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {ArchEngine} from '../src/engine.js';
import {animationOptions, sampleSvgAnimation, convertSvgAnimation} from '../src/svg/animation.js';
import {parseXml, walk} from '../src/svg/xml.js';
import {parseScript} from '../src/script.js';

const engine = new ArchEngine();
after(() => engine.close());
const wrap = body => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">${body}</svg>`;
const nodes = (svg, name) => [...walk(parseXml(svg))].filter(n => n.name === name);
const sample = async (svg, options = {}) => sampleSvgAnimation(svg, {duration: 1, fps: 2, browser: await engine.browser(), ...options});

test('animation validates timing and frame budget before launching a browser', async () => {
    for (const options of [{}, {duration: 0}, {duration: Infinity}, {duration: 1, fps: 2.5}, {duration: 1, fps: 61},
        {duration: 1, start: -1}, {duration: 1, maxFrames: 601}, {duration: 100}])
        assert.throws(() => animationOptions(options));
    assert.equal(animationOptions({duration: 0.3, fps: 12}).frames, 4);
    await assert.rejects(sampleSvgAnimation('not svg', {duration: 1}), /No <svg>/);
});

test('SMIL geometry, motion, color, path morph and delayed start are sampled exactly', async () => {
    const svg = wrap('<rect width="10" height="10" fill="red"><animate attributeName="x" from="0" to="80" dur="1s" fill="freeze"/>' +
        '<animate attributeName="fill" from="red" to="blue" dur="1s" fill="freeze"/></rect>' +
        '<circle r="2"><animateMotion path="M0 0L0 50" dur="1s" fill="freeze"/></circle>' +
        '<path d="M0 0L10 0L10 10Z"><animate attributeName="d" to="M0 0L20 0L20 20Z" dur="1s" fill="freeze"/></path>');
    const {snapshots} = await sample(svg, {start: 0.25});
    assert.equal(Number(nodes(snapshots[0], 'rect')[0].attrs.x), 20);
    assert.equal(Number(nodes(snapshots[1], 'rect')[0].attrs.x), 60);
    assert.match(nodes(snapshots[0], 'circle')[0].attrs.transform, /12\.5\)/);
    assert.match(nodes(snapshots[1], 'path')[0].attrs.d, /17\.5/);
    assert.notEqual(nodes(snapshots[0], 'rect')[0].attrs.fill, nodes(snapshots[1], 'rect')[0].attrs.fill);
    assert.equal(nodes(snapshots[0], 'animate').length, 0);
});

test('CSS keyframes and transform origins are baked with local gradient references', async () => {
    const {snapshots} = await sample(wrap('<style>@keyframes move{from{transform:translateX(0px);opacity:1}to{transform:translateX(40px);opacity:0}}rect{animation:move 1s linear infinite}</style>' +
        '<defs><linearGradient id="ink"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs>' +
        '<rect width="20" height="20" rx="5" fill="url(#ink)"/>'));
    const rect = nodes(snapshots[1], 'rect')[0];
    assert.match(rect.attrs.transform, /20,0\)/);
    assert.equal(rect.attrs.opacity, '0.5');
    assert.equal(rect.attrs.fill, 'url(#ink)');
    assert.equal(rect.attrs.rx, '5');
    assert.equal(rect.attrs.ry, undefined, 'auto radius must not be replaced by zero');
    assert.equal(nodes(snapshots[1], 'style').length, 0);
});

test('namespace-less snippets and CSS percentage geometry retain their animation', async () => {
    const {snapshots} = await sample('<svg viewBox="0 0 100 100"><style>@keyframes slide{from{x:0%}to{x:80%}}rect{animation:slide 1s linear infinite}</style><rect width="10" height="10"/></svg>');
    assert.equal(nodes(snapshots[1], 'rect')[0].attrs.x, '40%');
});

test('scripts, handlers and external resources cannot execute during sampling', async () => {
    const {snapshots, warnings} = await sample(wrap('<script>document.querySelector("rect").setAttribute("width","99")</script>' +
        '<rect width="10" height="10" onload="this.setAttribute(\'width\',\'99\')"/>' +
        '<image href="https://example.invalid/pixel.png" width="10" height="10"/>'));
    assert.equal(nodes(snapshots[0], 'rect')[0].attrs.width, '10');
    assert.equal(nodes(snapshots[0], 'rect')[0].attrs.onload, undefined);
    assert.equal(nodes(snapshots[0], 'script').length, 0);
    assert.ok(warnings.some(w => /disabled/.test(w)));
});

test('fixed view, frame tags and frame settings survive the real Arch exporter and reload', async () => {
    const conversion = await convertSvgAnimation(wrap('<rect width="10" height="10" fill="red"><animate attributeName="x" to="80" dur="1s"/></rect>'),
        {duration: 1, fps: 2, outlines: false, browser: await engine.browser()});
    assert.equal(conversion.stats.frames, 2);
    const steps = parseScript(conversion.script).steps;
    assert.equal(steps.filter(s => s.kind === 'view').length, 1);
    const result = await engine.run(conversion.script, {screenshot: false});
    assert.equal(result.ok, true, JSON.stringify(result.errors));
    const save = JSON.parse(result.json);
    assert.deepEqual(save.entries.map(e => e.displayFrame), [1, 2]);
    assert.equal(result.frames.total, 2);
    assert.equal(result.frames.current, 1);
    assert.equal(result.frames.fps, 2);
});

test('unchanged text is shared, animated text fails clearly rather than appearing on every frame', async () => {
    const options = {duration: 1, fps: 2, browser: await engine.browser()};
    const result = await convertSvgAnimation(wrap('<text x="10" y="20">Hello</text>'), options);
    assert.equal(result.script.match(/^@text /gm).length, 1);
    assert.doesNotThrow(() => parseScript(result.script));
    await assert.rejects(convertSvgAnimation(wrap('<text x="10" y="20">Hello<animate attributeName="x" to="60" dur="1s"/></text>'), options), /text.*paths/);
});

test('animation CLI exports editable script-only output and rejects excessive frame counts', () => {
    const dir = mkdtempSync(join(tmpdir(), 'arch-animation-test-'));
    const cli = fileURLToPath(new URL('../src/svg2arch-anim.js', import.meta.url));
    const svg = fileURLToPath(new URL('../examples/svg/animated-orbit.svg', import.meta.url));
    try {
        const script = join(dir, 'orbit.arch');
        const result = spawnSync(process.execPath, [cli, svg, '--duration', '1', '--fps', '2', '--script-only', '-s', script], {encoding: 'utf8'});
        assert.equal(result.status, 0, result.stderr);
        assert.match(readFileSync(script, 'utf8'), /@frames total=2 fps=2 current=1/);
        const bad = spawnSync(process.execPath, [cli, svg, '--duration', '1000', '-o', join(dir, 'bad.txt')], {encoding: 'utf8'});
        assert.equal(bad.status, 2);
        assert.equal(existsSync(join(dir, 'bad.txt')), false);
    } finally {
        // Only remove the exact temporary directory allocated by this test.
        assert.ok(dir.startsWith(join(tmpdir(), 'arch-animation-test-')));
        rmSync(dir, {recursive: true, force: true});
    }
});
