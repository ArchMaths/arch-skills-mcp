#!/usr/bin/env node
// Sample an animated SVG in Chromium, then build frame-tagged Arch scene entries.
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {resolve, dirname} from 'node:path';
import {ArchEngine} from './engine.js';
import {formatReport} from './report.js';
import {DEFAULTS} from './svg/index.js';
import {convertSvgAnimation, formatAnimationConversion} from './svg/animation.js';
import {parseXml} from './svg/xml.js';

const USAGE = `Convert an animated SVG into frame-by-frame Arch playback (Chrome required, even with --script-only).

Usage:
  svg2arch-anim <drawing.svg> --duration <seconds> [-o drawing.txt] [options]
  svg2arch-anim <drawing.svg> --duration <seconds> --check [--png shot.png]
  svg2arch-anim <drawing.svg> --duration <seconds> --script-only [-s out.arch]

Options:
  --duration <s>          required positive duration in seconds
  --fps <n>               integer sampling/playback rate, 1 to 60 (default 12)
  --start <s>             nonnegative start time in seconds (default 0)
  --max-frames <n>        integer frame budget, 1 to 600 (default 120)
  -o, --out <path>        save file (default: SVG's name with .txt)
  -s, --script <path>     scene script (default: save file's name with .arch)
  --script-only          write only the scene script; still uses Chrome to sample SVG
  --check                convert and run in Arch, without writing a save or implicit script
  --png <path>           save a screenshot of Arch's initial frame
  --png-scale <s>         screenshot scale, 0.1 to 1
  --tolerance <px>        curve flattening tolerance (default ${DEFAULTS.tolerance})
  --precision <n>         Arch sampling precision, 0.5 to 20 (default ${DEFAULTS.precision})
  --gradient-steps <n>    maximum colour bands per gradient (default ${DEFAULTS.gradientSteps})
  --max-edges <n>         edges per inequality entry (default ${DEFAULTS.maxEdges})
  --title <text>         file name shown in Arch
  --axes                 keep Arch's axes and grid
  --flatten              compatibility output for Arch builds before 12202
  --no-seams             do not hide seams between neighbouring shapes
  --no-outlines          do not trace opaque shapes with exact point lists
  --timeout <s>          seconds each Arch build step may take (default 30)
  --html <path>          Arch index.html (default: ARCH_HTML or bundled index.html)
  --json                 print the full result as JSON
  -h, --help             show this help

Samples ceil(duration*fps) frames at start+i/fps: the interval excludes start+duration.
SMIL and CSS animations are sampled; SVG scripts and external resources are disabled.
Press Play manually in Arch. File size and conversion/build cost grow with frame count.
Static SVG limitations apply (including ignored filters). Exit codes: 0 OK, 1 Arch errors, 2 input/environment problem.`;

function parseArgs(argv) {
    const args = {options: {}};
    const number = (name, text) => {
        const v = Number(text);
        if (!Number.isFinite(v)) throw new Error(`${name} needs a finite number`);
        return v;
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const value = () => {
            if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
            return argv[++i];
        };
        switch (a) {
        case '-h': case '--help': args.help = true; break;
        case '-o': case '--out': args.out = value(); break;
        case '-s': case '--script': args.script = value(); break;
        case '--check': args.check = true; break;
        case '--script-only': args.scriptOnly = true; break;
        case '--png': args.png = value(); break;
        case '--png-scale': {
            const s = number(a, value());
            if (!(s >= 0.1 && s <= 1)) throw new Error('--png-scale must be from 0.1 to 1');
            args.pngScale = s;
            break;
        }
        case '--duration': args.options.duration = number(a, value()); break;
        case '--fps': args.options.fps = number(a, value()); break;
        case '--start': args.options.start = number(a, value()); break;
        case '--max-frames': args.options.maxFrames = number(a, value()); break;
        case '--tolerance': args.options.tolerance = number(a, value()); break;
        case '--precision': args.options.precision = number(a, value()); break;
        case '--gradient-steps': args.options.gradientSteps = Math.max(1, Math.round(number(a, value()))); break;
        case '--max-edges': args.options.maxEdges = Math.max(10, Math.round(number(a, value()))); break;
        case '--title': args.options.title = value(); break;
        case '--axes': args.options.axes = true; break;
        case '--flatten': args.options.flatten = true; break;
        case '--no-seams': args.options.seams = false; break;
        case '--no-outlines': args.options.outlines = false; break;
        case '--timeout': {
            const s = number(a, value());
            if (!(s > 0)) throw new Error('--timeout must be a positive number of seconds');
            args.timeoutMs = s * 1000;
            break;
        }
        case '--html': args.html = value(); break;
        case '--json': args.json = true; break;
        default:
            if (a.startsWith('-') && a !== '-') throw new Error(`Unknown option ${a}`);
            if (args.input !== undefined) throw new Error(`Unexpected argument ${a}`);
            args.input = a;
        }
    }
    if (args.help) return args;
    const {duration, fps = 12, start = 0, maxFrames = 120, tolerance, precision} = args.options;
    if (!(duration > 0)) throw new Error('--duration is required and must be a positive number of seconds');
    if (!Number.isInteger(fps) || fps < 1 || fps > 60) throw new Error('--fps must be an integer from 1 to 60');
    if (!(start >= 0) || !Number.isFinite(start + duration)) throw new Error('--start must be nonnegative and start+duration finite');
    if (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 600) throw new Error('--max-frames must be an integer from 1 to 600');
    if (Math.ceil(duration * fps) > maxFrames) throw new Error(`Sampling needs ${Math.ceil(duration * fps)} frames, exceeding --max-frames ${maxFrames}`);
    if (tolerance !== undefined && !(tolerance > 0)) throw new Error('--tolerance must be positive');
    if (precision !== undefined && !(precision >= 0.5 && precision <= 20)) throw new Error('--precision must be from 0.5 to 20');
    if (args.pngScale && !args.png) throw new Error('--png-scale needs --png');
    if (args.scriptOnly && (args.check || args.png)) throw new Error('--script-only cannot be combined with --check or --png');
    return args;
}

const write = (path, data) => {
    mkdirSync(dirname(resolve(path)), {recursive: true});
    writeFileSync(resolve(path), data);
};

async function main() {
    let args;
    try {
        args = parseArgs(process.argv.slice(2));
    } catch (error) {
        console.error(`${error.message}\n\n${USAGE}`);
        return 2;
    }
    if (args.help) { console.log(USAGE); return 0; }
    if (!args.input) { console.error(USAGE); return 2; }
    const svg = args.input === '-' ? readFileSync(0, 'utf8') : readFileSync(resolve(args.input), 'utf8');
    const stem = args.input === '-' ? 'drawing' : args.input.replace(/\.svgz?$/i, '');
    const out = args.out || `${stem}.txt`;
    const engine = new ArchEngine(args.html ? {htmlPath: args.html} : {});
    try {
        let conversion;
        try {
            parseXml(svg);
            conversion = await convertSvgAnimation(svg, {...args.options, browser: await engine.browser()});
        } catch (error) {
            console.error(`Could not convert ${args.input}: ${error.message}`);
            return 2;
        }
        const scriptPath = args.script || (args.check ? null : out.replace(/\.txt$/i, '') + '.arch');
        if (scriptPath) write(scriptPath, conversion.script);
        const summary = formatAnimationConversion(conversion, {scriptPath});
        const details = {stats: conversion.stats, warnings: conversion.warnings, info: conversion.info, script: scriptPath};
        if (args.scriptOnly) {
            console.log(args.json ? JSON.stringify(details, null, 2) : summary);
            return 0;
        }
        const result = await engine.run(conversion.script, {screenshot: Boolean(args.png), screenshotScale: args.pngScale ?? 1, timeoutMs: args.timeoutMs});
        if (args.png && result.png) write(args.png, result.png);
        if (result.ok && !args.check) write(out, result.json);
        if (args.json) {
            const {png, scriptPng, json, objects, ...rest} = result;
            console.log(JSON.stringify({conversion: details, build: rest}, null, 2));
        } else {
            console.log(summary);
            console.log(formatReport(result, {written: args.check ? undefined : (result.ok ? out : null), screenshot: args.png && result.png ? args.png : undefined, brief: true}));
        }
        return result.ok ? 0 : 1;
    } finally {
        await engine.close();
    }
}

main().then(code => { process.exitCode = code; }, error => {
    console.error(error.code === 'ENOENT' ? `File not found: ${error.path}` : (error.stack || error.message));
    process.exitCode = 2;
});
