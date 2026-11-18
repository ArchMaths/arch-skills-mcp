// Talks to the MCP server over stdio exactly like Claude Code or Antigravity would.
import {test, before, after} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, readFileSync, writeFileSync, existsSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {Client} from '@modelcontextprotocol/sdk/client/index.js';
import {StdioClientTransport} from '@modelcontextprotocol/sdk/client/stdio.js';

const server = fileURLToPath(new URL('../src/mcp.js', import.meta.url));
const dir = mkdtempSync(join(tmpdir(), 'arch-mcp-test-'));
let client;

before(async () => {
    client = new Client({name: 'arch-mcp-test', version: '0.0.0'});
    await client.connect(new StdioClientTransport({command: process.execPath, args: [server], cwd: dir, stderr: 'ignore'}));
});
after(async () => {
    await client?.close();
    rmSync(dir, {recursive: true, force: true});
});

const textOf = result => result.content.filter(c => c.type === 'text').map(c => c.text).join('\n');

test('lists the five tools with instructions', async () => {
    const {tools} = await client.listTools();
    assert.deepEqual(tools.map(t => t.name).sort(), ['arch_build', 'arch_check', 'arch_decompile', 'arch_guide', 'arch_svg']);
    assert.match(client.getInstructions(), /arch_guide/);
    assert.match(client.getInstructions(), /arch_svg/);
    assert.match(client.getInstructions(), /timeout_seconds/);
    for (const name of ['arch_check', 'arch_build', 'arch_svg']) {
        const {properties} = tools.find(t => t.name === name).inputSchema;
        assert.equal(properties.timeout_seconds.type, 'number', name);
        assert.equal(properties.timeout_seconds.exclusiveMinimum, 0, name);
        assert.equal(properties.lean.type, 'boolean', name);
    }
    assert.equal(tools.find(t => t.name === 'arch_check').inputSchema.properties.brief.type, 'boolean');
});

test('arch_guide returns the guide', async () => {
    const result = await client.callTool({name: 'arch_guide', arguments: {}});
    assert.match(textOf(result), /What u means in `point\(P,host,u\)`/);
});

test('arch_guide can return only some sections', async () => {
    const result = await client.callTool({name: 'arch_guide', arguments: {section: 'geometry'}});
    assert.match(textOf(result), /What u means/);
    assert.ok(!textOf(result).includes('## Global program'));
    const bad = await client.callTool({name: 'arch_guide', arguments: {section: 'nope'}});
    assert.equal(bad.isError, true);
    assert.match(textOf(bad), /Unknown guide topic/);
});

test('arch_check reports errors with an image', async () => {
    const result = await client.callTool({name: 'arch_check', arguments: {script: 'y=x^2 | ref=p\nintersect(I,p,q,1,1)'}});
    assert.equal(result.isError, true);
    assert.match(textOf(result), /line 2: refers to "q"/);
    assert.ok(result.content.some(c => c.type === 'image' && c.mimeType === 'image/png'));
});

test('arch_check renders smaller screenshots and arch_build can report briefly', async () => {
    const check = await client.callTool({name: 'arch_check', arguments: {script: 'y=x^2', screenshot_scale: 0.5, brief: true}});
    assert.equal(check.isError, false);
    assert.equal(textOf(check), 'OK: script is valid (1 objects)');
    const png = Buffer.from(check.content.find(c => c.type === 'image').data, 'base64');
    assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [640, 400]);
    const build = await client.callTool({name: 'arch_build', arguments: {script: 'y=x^2', output_path: 'out/brief.txt', brief: true, screenshot: false}});
    assert.match(textOf(build), /^OK: wrote .*brief\.txt \(1 objects\)$/);
});

test('arch_build writes a loadable save file (relative to the server cwd)', async () => {
    const result = await client.callTool({name: 'arch_build', arguments: {
        script: '@title Unit circle\ncircle(c,O,1) | color=blue\npoint(P,c,0.125)\n@view x=-2..2 y=-2..2',
        output_path: 'out/unit.txt', screenshot_path: 'out/unit.png', screenshot: false}});
    assert.equal(result.isError, false);
    assert.match(textOf(result), /OK: wrote .*unit\.txt/);
    assert.match(textOf(result), /P \[point\] point\(P,c,0\.125\) {2}=> {2}\(0\.7071, 0\.7071\)/);
    const save = JSON.parse(readFileSync(join(dir, 'out/unit.txt'), 'utf8'));
    assert.equal(save.name, 'Unit circle');
    assert.ok(existsSync(join(dir, 'out/unit.png')));
    assert.ok(!result.content.some(c => c.type === 'image'));

    const dec = await client.callTool({name: 'arch_decompile', arguments: {save_path: 'out/unit.txt'}});
    assert.match(textOf(dec), /^point\(P,c,0\.125\)/m);
});

test('arch_check returns a second screenshot after running @script', async () => {
    const result = await client.callTool({name: 'arch_check', arguments: {
        script: 'a=0\ny=a*x\n@script\nsetInterval(() => arch.valuesetting(\'a\', 1), 50);\n@end'}});
    assert.match(textOf(result), /Preview: ran 2 s with no errors/);
    assert.equal(result.content.filter(c => c.type === 'image').length, 2);
    assert.match(textOf(result), /Screenshot 2: the same file after running its @script/);
});

test('arch_build refuses to write a broken script', async () => {
    const result = await client.callTool({name: 'arch_build', arguments: {script: 'sin(', output_path: 'out/broken.txt', screenshot: false}});
    assert.equal(result.isError, true);
    assert.ok(!existsSync(join(dir, 'out/broken.txt')));
});

test('arch_svg converts an SVG and writes the scene script and save file', async () => {
    const svg = '<svg viewBox="0 0 10 10"><title>Two squares</title><rect width="6" height="6" fill="#e63946"/>' +
        '<rect x="4" y="4" width="6" height="6" fill="#457b9d" filter="url(#none)"/></svg>';
    const result = await client.callTool({name: 'arch_svg', arguments: {svg, output_path: 'out/squares.txt', screenshot: false, timeout_seconds: 60, lean: true}});
    assert.equal(result.isError, false, textOf(result));
    // The upper square is stacked on the lower one, both whole: 4 edges each.
    assert.match(textOf(result), /Converted: 2 layer\(s\) drawn in order, 4 entries \(2 outline\(s\).* 8 edges/);
    assert.match(textOf(result), /filters are ignored/);
    assert.match(textOf(result), /OK: wrote .*squares\.txt/);
    assert.match(textOf(result), /Lean: left out the expanded formulas of 0 object\(s\)/);
    const save = JSON.parse(readFileSync(join(dir, 'out/squares.txt'), 'utf8'));
    assert.equal(save.name, 'Two squares');
    assert.ok(save.entries.some(e => e.sign === '<'));
    assert.equal(save.orderedDrawingEnabled, true);
    const flat = await client.callTool({name: 'arch_svg', arguments: {svg, check: true, flatten: true, screenshot: false}});
    assert.match(textOf(flat), /Converted: 2 colour region\(s\)/);
    assert.match(readFileSync(join(dir, 'out/squares.arch'), 'utf8'), /^cos\(pi\*/m);
    const missing = await client.callTool({name: 'arch_svg', arguments: {svg}});
    assert.equal(missing.isError, true);
});

test('reports input and filesystem errors as MCP tool errors', async () => {
    const cases = [
        [{}, /Pass the scene script/],
        [{script: 'y=x', script_path: 'missing.arch'}, /not both/],
        [{script: '', script_path: 'missing.arch'}, /not both/],
        [{script: '   '}, /Pass the scene script/],
        [{script_path: 'missing.arch'}, /ENOENT/]
    ];
    for (const [args, message] of cases) {
        const result = await client.callTool({name: 'arch_check', arguments: {...args, screenshot: false}});
        assert.equal(result.isError, true, JSON.stringify(args));
        assert.match(textOf(result), message);
    }
    const badSvg = await client.callTool({name: 'arch_svg', arguments: {svg: '', svg_path: 'missing.svg', check: true}});
    assert.equal(badSvg.isError, true);
    assert.match(textOf(badSvg), /not both/);
    const missingSave = await client.callTool({name: 'arch_decompile', arguments: {save_path: 'missing.txt'}});
    assert.equal(missingSave.isError, true);
    assert.match(textOf(missingSave), /ENOENT/);
});

test('validates timeout_seconds and lean on all engine tools', async () => {
    for (const name of ['arch_check', 'arch_build', 'arch_svg']) {
        const args = name === 'arch_svg' ? {svg: '<svg/>', check: true} : {script: 'y=x', output_path: 'out/invalid.txt'};
        for (const timeout_seconds of [0, -1, '30']) {
            const result = await client.callTool({name, arguments: {...args, timeout_seconds}});
            assert.equal(result.isError, true, `${name}: ${timeout_seconds}`);
            assert.match(textOf(result), /timeout_seconds/);
        }
        const result = await client.callTool({name, arguments: {...args, lean: 'true'}});
        assert.equal(result.isError, true, name);
        assert.match(textOf(result), /lean/);
    }
    assert.ok(!existsSync(join(dir, 'out/invalid.txt')));
});

test('timeout_seconds reaches each engine tool and timed-out requests do not poison the server', async () => {
    const svg = '<svg viewBox="0 0 10 10"><rect width="10" height="10"/></svg>';
    for (const name of ['arch_check', 'arch_build', 'arch_svg']) {
        const args = name === 'arch_svg' ? {svg, check: true} : {script: 'y=x', output_path: 'out/timeout.txt'};
        const result = await client.callTool({name, arguments: {...args, screenshot: false, timeout_seconds: 0.001}});
        assert.equal(result.isError, true, name);
        assert.match(textOf(result), /did not finish within 0\.001 s/, name);
    }
    assert.ok(!existsSync(join(dir, 'out/timeout.txt')));
    const recovered = await client.callTool({name: 'arch_check', arguments: {script: 'y=x', screenshot: false, timeout_seconds: 30}});
    assert.equal(recovered.isError, false, textOf(recovered));
});

test('lean builds omit expanded formulas and still pass the reload check', async () => {
    const script = 'f(t)=t^2+1\ny=f(x)';
    writeFileSync(join(dir, 'functions.arch'), script);
    const full = await client.callTool({name: 'arch_build', arguments: {script_path: 'functions.arch', output_path: 'out/full.txt', screenshot: false}});
    assert.equal(full.isError, false, textOf(full));
    const lean = await client.callTool({name: 'arch_build', arguments: {script_path: 'functions.arch', output_path: 'out/lean.txt', screenshot: false, lean: true, timeout_seconds: 30}});
    assert.equal(lean.isError, false, textOf(lean));
    assert.match(textOf(lean), /Lean: left out the expanded formulas of 1 object\(s\)/);
    const fullJson = readFileSync(join(dir, 'out/full.txt'), 'utf8');
    const leanJson = readFileSync(join(dir, 'out/lean.txt'), 'utf8');
    const entry = json => JSON.parse(json).entries.find(e => e.expr === 'y=f(x)');
    assert.ok(entry(fullJson).processedExpr);
    assert.equal(entry(leanJson).processedExpr, undefined);
    assert.ok(leanJson.length < fullJson.length);
    const check = await client.callTool({name: 'arch_check', arguments: {script, lean: true, brief: true, screenshot: false}});
    assert.equal(check.isError, false, textOf(check));
    assert.match(textOf(check), /Lean: left out the expanded formulas of 1 object\(s\)/);
    assert.ok(!textOf(check).includes('Objects ('));
});

test('script_preview false stores a global program without running it', async () => {
    const result = await client.callTool({name: 'arch_build', arguments: {
        script: 'y=x\n@script\nthrow new Error("must not run");\n@end',
        output_path: 'out/no-preview.txt', script_preview: false, screenshot: false
    }});
    assert.equal(result.isError, false, textOf(result));
    assert.match(textOf(result), /Preview: not run \(preview skipped\)/);
    assert.match(readFileSync(join(dir, 'out/no-preview.txt'), 'utf8'), /must not run/);
    assert.ok(!result.content.some(c => c.type === 'image'));
});

test('SVG check does not write an implicit script but can explicitly export one', async () => {
    const svg = '<svg viewBox="0 0 10 10"><rect width="10" height="10"/></svg>';
    const implicit = await client.callTool({name: 'arch_svg', arguments: {svg, check: true, output_path: 'out/check-only.txt', screenshot: false}});
    assert.equal(implicit.isError, false, textOf(implicit));
    assert.ok(!existsSync(join(dir, 'out/check-only.txt')));
    assert.ok(!existsSync(join(dir, 'out/check-only.arch')));
    const explicit = await client.callTool({name: 'arch_svg', arguments: {svg, check: true, output_path: 'out/export-only.txt', script_path: 'out/export-only.arch', screenshot: false}});
    assert.equal(explicit.isError, false, textOf(explicit));
    assert.ok(existsSync(join(dir, 'out/export-only.arch')));
    assert.ok(!existsSync(join(dir, 'out/export-only.txt')));
});
