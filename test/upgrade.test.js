// Arch 2.0.05 regression coverage: parser, styles, and real-engine round trips.
import {test, after} from 'node:test';
import assert from 'node:assert/strict';
import {ArchEngine} from '../src/engine.js';
import {parseScript} from '../src/script.js';
import {decompile} from '../src/decompile.js';

const engine = new ArchEngine({keepWarm: true});
after(() => engine.close());

test('dynamic style parser preserves expressions, nested commas, spaces and logical OR', () => {
    const {steps} = parseScript('y=(x<0||x>1)?1:0 | color="hsb(max(a, 2),100,80)" opacity="a / 2" thickness="a + 1"\ny=x | thickness=0\n@set smartColorInversion=false');
    assert.equal(steps[0].expr, 'y=(x<0||x>1)?1:0');
    assert.deepEqual(steps[0].style, {color: {h: 'max(a, 2)', s: 100, b: 80, a: '(1-(a / 2))*100'}, thickness: 'a + 1'});
    assert.equal(steps[1].style.thickness, 0);
    assert.equal(steps[2].settings[0].name, 'smartColorInversion');
    for (const option of ['thickness=21', 'thickness=-1', 'opacity=2', 'opacity=Infinity', 'color=hsb(101,0,0)'])
        assert.throws(() => parseScript(`y=x | ${option}`), /must be between/);
});

test('dynamic styles and text survive export, async reload, decompile and rebuild', async () => {
    const source = [
        'a=0.5',
        'point(P,1,2) | color=hsb(100*a,100,80) opacity=a thickness="2 + 4*a"',
        'y=x | thickness=0',
        '@text x=-2 y=3 color=hsb(100*a,100,80) opacity=a | Dynamic',
        '@set smartColorInversion=false'
    ].join('\n');
    const first = await engine.run(source, {screenshot: false});
    assert.deepEqual(first.errors, []);
    const a = JSON.parse(first.json);
    assert.equal(a.version, 20005);
    assert.equal(a.smartColorInversionEnabled, false);
    const p = a.entries.find(e => e.name === 'P');
    assert.equal(p.thickness, 4);
    assert.equal(p.color.h, 50);
    assert.equal(p.color.a, 50);
    assert.equal(p.styleExpressions.thickness.expr, '2 + 4*a');
    assert.equal(a.entries.find(e => e.expr === 'y=x').thickness, 0);
    assert.equal(a.annotations[0].color.h, 50);
    assert.equal(a.annotations[0].styleExpressions.h.expr, '100*a');
    const second = await engine.run(decompile(first.json).script, {screenshot: false});
    assert.deepEqual(second.errors, []);
    const b = JSON.parse(second.json);
    assert.deepEqual(b.entries.map(e => [e.expr, e.color, e.thickness, e.styleExpressions]), a.entries.map(e => [e.expr, e.color, e.thickness, e.styleExpressions]));
    assert.deepEqual(b.annotations, a.annotations);
});

test('auto-created function plots decompile without duplicate creation', async () => {
    const first = await engine.run('f(x)=x^2\n@style func1 | thickness=5 color=blue\npoint(P,func1,2)', {screenshot: false});
    assert.deepEqual(first.errors, []);
    const decoded = decompile(first.json);
    assert.doesNotMatch(decoded.script, /^y=f\(x\)/m);
    const second = await engine.run(decoded.script, {screenshot: false});
    assert.deepEqual(second.errors, []);
    assert.deepEqual(second.objects.map(o => [o.expression, o.details, o.color, o.thickness]), first.objects.map(o => [o.expression, o.details, o.color, o.thickness]));
});

test('3D auto-created function surfaces restore dimension before their definitions', async () => {
    const first = await engine.run('@view 3d\nf(x,y)=x^2+y^2\n@style func1 | color=blue', {screenshot: false});
    assert.deepEqual(first.errors, []);
    const second = await engine.run(decompile(first.json).script, {screenshot: false});
    assert.deepEqual(second.errors, []);
    assert.deepEqual(second.objects.map(o => [o.expression, o.plotType]), first.objects.map(o => [o.expression, o.plotType]));
});

test('creation lines do not silently overwrite existing definitions', async () => {
    const result = await engine.run('a=1\na=2', {screenshot: false});
    assert.equal(result.ok, false);
    assert.match(result.errors[0].message, /already taken/);
    assert.equal(result.objects[0].value, 1);
});

test('new math operators reach Arch instead of being mistaken for options', async () => {
    const result = await engine.run('a=2\nb=(a==2||a<0)?a%3:0\ny=(x<0||x>1)?1:0 | color=blue', {screenshot: false});
    assert.deepEqual(result.errors, []);
    assert.equal(result.objects.find(o => o.name === 'b').value, 2);
});
