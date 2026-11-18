// The skill, the guide topics and the CLI flags it relies on. No browser needed.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {execFileSync, spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {GUIDE_TOPICS, guideBlocks, readGuide, readGuideTopics, ROOT} from '../src/guide.js';
import {installSkill, renderSkill, SKILL_SOURCE} from '../src/install-skill.js';
import {formatReport} from '../src/report.js';

const cli = fileURLToPath(new URL('../src/cli.js', import.meta.url));
const runCli = args => spawnSync(process.execPath, [cli, ...args], {encoding: 'utf8'});

test('every guide heading belongs to a topic, and every topic resolves', () => {
    const covered = new Set();
    for (const [name, topic] of Object.entries(GUIDE_TOPICS)) {
        const text = readGuideTopics(name);
        assert.ok(text.length > 200, name);
        for (const start of topic.headings) assert.ok(text.includes(start), `${name}: ${start}`);
    }
    const blocks = guideBlocks();
    for (const [i, block] of blocks.entries()) {
        const owner = Object.values(GUIDE_TOPICS).flatMap(t => t.headings).find(h => block.heading.startsWith(h));
        if (owner) { covered.add(i); continue; }
        // A ### block is covered by its ## topic; a ## block only containing topics (Arch expression syntax) is a container.
        const parent = blocks.slice(0, i).findLast(b => b.level < block.level);
        const container = block.level === 2 && blocks[i + 1]?.level === 3;
        assert.ok(container || (parent && covered.has(blocks.indexOf(parent))), `"${block.heading}" is in no guide topic`);
        if (!container) covered.add(i);
    }
});

test('guide topics: prefixes, lists and unknown names', () => {
    assert.match(readGuideTopics('geo'), /^### Geometry commands/);
    const two = readGuideTopics('text, operators');
    assert.match(two, /## Canvas text/);
    assert.match(two, /## JavaScript operators/);
    assert.ok(!two.includes('## Global program'));
    assert.throws(() => readGuideTopics('bogus'), /Unknown guide topic "bogus"[\s\S]*geometry/);
    assert.throws(() => readGuideTopics(''), /Name a guide topic/);
    assert.ok(readGuideTopics('syntax').length < readGuide().length / 2);
});

test('the CLI prints guide topics and rejects bad options', () => {
    const geometry = runCli(['--guide', 'geometry']);
    assert.equal(geometry.status, 0);
    assert.match(geometry.stdout, /What u means/);
    assert.ok(!geometry.stdout.includes('## Script format'));
    assert.equal(runCli(['--guide']).stdout, readGuide());
    assert.equal(runCli(['--guide', 'nope']).status, 2);
    assert.equal(runCli(['--check', 'x.arch', '--png-scale', '0.5']).status, 2);
    assert.equal(runCli(['--check', 'x.arch', '--png', 'x.png', '--png-scale', '3']).status, 2);
});

test('brief reports keep errors and warnings but drop the object list', () => {
    const result = {ok: false, errors: [{line: 2, message: 'boom'}], warnings: [{line: 1, message: 'hmm'}],
        objects: [{line: 1, name: 'A', type: 'geometry', geometryType: 'point', expression: 'point(A,0,0)', details: '(0, 0)'}],
        refs: {p: 'func1'}, view: {dimension: '2d', center: {x: 0, y: 0}, unit: 100}};
    const brief = formatReport(result, {written: null, brief: true});
    assert.match(brief, /FAILED: 1 error/);
    assert.match(brief, /line 2: boom/);
    assert.match(brief, /line 1: hmm/);
    assert.ok(!/Objects|Refs|View/.test(brief));
    assert.match(formatReport(result, {written: null}), /Objects[\s\S]*Refs[\s\S]*View/);
});

test('the skill only mentions guide topics and CLI flags that exist', () => {
    const skill = readFileSync(SKILL_SOURCE, 'utf8');
    assert.match(skill, /^---\nname: arch-graph\ndescription: .+\n---\n/);
    const topics = [...skill.matchAll(/--guide ([a-z]+)/g)].map(m => m[1]);
    const tableTopics = [...skill.matchAll(/^\| `([a-z]+)` \|/gm)].map(m => m[1]);
    assert.ok(tableTopics.length >= 8);
    for (const t of [...topics, ...tableTopics]) assert.ok(GUIDE_TOPICS[t], `skill mentions unknown topic ${t}`);
    const svgHelp = spawnSync(process.execPath, [fileURLToPath(new URL('../src/svg2arch.js', import.meta.url)), '--help'], {encoding: 'utf8'}).stdout;
    const help = runCli(['--help']).stdout + svgHelp;
    for (const flag of new Set(skill.match(/(?<![\w-])--[a-z][a-z-]+/g))) assert.ok(help.includes(flag), `skill mentions unknown flag ${flag}`);
    // Keep the always-loaded part small; details live behind --guide.
    assert.ok(skill.length < 9000, `SKILL.md is ${skill.length} bytes`);
});

test('skill and guide track bundled 2.0.05 syntax and dynamic styles', () => {
    const skill = readFileSync(SKILL_SOURCE, 'utf8');
    const guide = readGuide();
    for (const text of [skill, guide]) {
        assert.match(text, /appVersion=20005/);
        assert.match(text, /18982f5/);
        assert.match(text, /smartColorInversion=false/);
        assert.match(text, /thickness="2\+3\*/);
        assert.match(text, /opacity="0\.3\+0\.5\*/);
        assert.match(text, /color=hsb\(100\*/);
        assert.match(text, /`f\(x\)=x\^2(?:\+1)?`/);
        assert.doesNotMatch(text, /cannot be `e x y z`|labels are not drawn/);
    }
    assert.match(readGuideTopics('format'), /zero allowed/);
    assert.match(readGuideTopics('syntax'), /auto-created/);
    assert.match(readGuideTopics('mistakes'), /label bug[\s\S]*fixed/);
    assert.match(readGuideTopics('script'), /expression strings/);
});

test('README records upstream provenance and MCP export controls', () => {
    const readme = readFileSync(join(ROOT, 'README.md'), 'utf8');
    assert.match(readme, /https:\/\/github\.com\/ArchMaths\/arch-skills-mcp\.git/);
    assert.match(readme, /cf94850a5f03fe074b66fab3af6c2501aa6fa241677bfb242ddbeb89f9065392/);
    assert.match(readme, /timeout_seconds/);
    assert.match(readme, /lean/);
    assert.match(readme, /standard alpha blending/);
    assert.doesNotMatch(readme, /bundled.*reports version 19999/);
});

test('install-skill fills in absolute paths and protects foreign skills', () => {
    const dir = mkdtempSync(join(tmpdir(), 'arch-skill-test-'));
    try {
        const rendered = renderSkill();
        assert.ok(!rendered.includes('{{'));
        assert.ok(rendered.includes(`node "${ROOT.replace(/\\/g, '/')}/src/cli.js" --check`));

        const first = installSkill({dir});
        assert.equal(first.updated, false);
        assert.equal(readFileSync(first.target, 'utf8'), rendered);
        assert.equal(installSkill({dir}).updated, true);

        const foreign = join(dir, 'other', 'arch-graph');
        mkdirSync(foreign, {recursive: true});
        writeFileSync(join(foreign, 'SKILL.md'), 'someone else\'s skill');
        assert.throws(() => installSkill({dir: join(dir, 'other')}), /not installed by arch-mcp/);
        installSkill({dir: join(dir, 'other'), force: true});

        const out = execFileSync(process.execPath, [fileURLToPath(new URL('../src/install-skill.js', import.meta.url)), '--dir', join(dir, 'cli')], {encoding: 'utf8'});
        assert.match(out, /Installed .*arch-graph[\\/]SKILL\.md/);
    } finally {
        rmSync(dir, {recursive: true, force: true});
    }
});
