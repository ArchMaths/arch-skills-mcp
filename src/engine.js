import {existsSync, readFileSync, statSync} from 'node:fs';
import {resolve, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';
import {parseScript, ScriptError} from './script.js';
import {runSteps, loadAndDescribe, previewScript, settle} from './page-runner.js';
import {ARCH_MESSAGES, translateArchMessage} from './messages.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const DEFAULT_HTML = process.env.ARCH_HTML ? resolve(process.env.ARCH_HTML) : resolve(ROOT, 'index.html');
const VIEWPORT = {width: 1280, height: 800};
const READY_TIMEOUT = 45000;
export const RUN_TIMEOUT = 30000;
export const PREVIEW_MS = 2000;
// Arch is served from a made-up HTTPS origin answered by Playwright itself. Unlike file://,
// the page cannot read local files, and every other request is aborted, so JavaScript in
// operators or @script runs isolated from both the disk and the network.
const PAGE_URL = 'https://arch.invalid/index.html';
// Arch's own server calls fail because the network is blocked on purpose; that is not a script problem.
const BLOCKED_NETWORK = /Failed to fetch|NetworkError|net::ERR_/;
// Code from operators and @script shows up in stack traces with one of these frames; everything
// else is Arch's own code. Arch 12201+ runs that code in its QuickJS sandbox under the file names
// arch-operator.js and arch-global-script.js; older builds compiled it with new Function.
const USER_CODE_FRAME = /eval at createSandboxedFunction|arch-operator\.js|arch-global-script\.js/;

// Uncaught errors on the build page, deduplicated (drawing errors repeat every frame).
function pageErrorWarnings(pageErrors) {
    const seen = new Set();
    const out = [];
    for (const {message, stack} of pageErrors) {
        if (BLOCKED_NETWORK.test(message)) continue;
        if (USER_CODE_FRAME.test(stack)) {
            if (!seen.has('op:' + message)) out.push({message: `JavaScript operator code raised an uncaught error: ${message}`});
            seen.add('op:' + message);
            continue;
        }
        const where = /\n\s+at (?:ArchEngine\.)?([A-Za-z_$][\w$]*) \(https:\/\/arch\.invalid/.exec(stack)?.[1];
        const key = `${where}:${message}`;
        if (seen.has(key)) continue;
        seen.add(key);
        let hint = '';
        if (where === 'drawMeasurements' && /toLowerCase/.test(message))
            hint = ' Known Arch bug: length(...) and angle(...) labels are not drawn when a measured point is created after any curve, equation or point list, or is the built-in O. Create the measured points first, and use point(O,0,0) instead of the built-in origin.';
        out.push({message: `Arch hit an internal error${where ? ` in ${where}` : ''}: ${message}. This is a bug inside Arch, not in the script; the file is still valid but part of the display may be missing.${hint}`});
    }
    return out;
}

// Browser choice: ARCH_BROWSER_PATH, then ARCH_BROWSER_CHANNEL, then Playwright's
// own Chromium, then installed Chrome / Edge.
async function launchBrowser() {
    const attempts = [];
    if (process.env.ARCH_BROWSER_PATH) attempts.push({executablePath: process.env.ARCH_BROWSER_PATH});
    if (process.env.ARCH_BROWSER_CHANNEL) attempts.push({channel: process.env.ARCH_BROWSER_CHANNEL});
    attempts.push({}, {channel: 'chrome'}, {channel: 'msedge'});
    const failures = [];
    for (const options of attempts) {
        try {
            return await chromium.launch({headless: true, ...options});
        } catch (error) {
            failures.push(`${JSON.stringify(options)}: ${String(error.message).split('\n')[0]}`);
        }
    }
    throw new Error('Could not start a Chromium browser. Install Google Chrome, or run "npx playwright-core install chromium", ' +
        'or set ARCH_BROWSER_PATH to a Chrome/Chromium executable.\n' + failures.join('\n'));
}

class RunTimeout extends Error {}
// The page's renderer process died (usually out of memory). Every call still waiting on it is abandoned.
class PageCrash extends Error {}
const CLOSE_TIMEOUT = 30000;

// scale < 1 asks Chrome to render the page smaller (sharp text, not a blurry downscale) without
// resizing the page: a 0.5 screenshot has a quarter of the pixels, so it costs an AI a quarter of the image tokens.
async function capture(page, scale = 1) {
    if (scale === 1) return page.screenshot({type: 'png'});
    const cdp = await page.context().newCDPSession(page);
    try {
        const {data} = await cdp.send('Page.captureScreenshot', {format: 'png', clip: {x: 0, y: 0, ...VIEWPORT, scale}});
        return Buffer.from(data, 'base64');
    } finally {
        await cdp.detach().catch(() => {});
    }
}

// Page code can loop forever (a bad operator or script); give up and let the caller close the page.
// A crashed page never answers, so its calls fail at once instead of waiting out the timeout.
function guard(promise, ms, what, slot) {
    promise.catch(() => {});
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new RunTimeout(
            `${what} did not finish within ${ms / 1000} s; a JavaScript operator or @script may loop forever, or the scene needs more time (CLI: --timeout).`)), ms);
    });
    const racers = [promise, timeout];
    if (slot) racers.push(slot.crashed.then(() => { throw new PageCrash(
        `${what} failed: the browser page crashed, most often because it ran out of memory (a very large scene, or several checks running at once). Run fewer checks at a time, or lower the scene's precision.`); }));
    return Promise.race(racers).finally(() => clearTimeout(timer));
}

// Closing a context or browser whose renderer is stuck can wait forever; give up after a while.
async function closeQuietly(target) {
    let timer;
    await Promise.race([target.close().catch(() => {}), new Promise(resolve => { timer = setTimeout(resolve, CLOSE_TIMEOUT); })]);
    clearTimeout(timer);
}

export class ArchEngine {
    constructor({htmlPath = DEFAULT_HTML, keepWarm = false, log = () => {}} = {}) {
        this.htmlPath = resolve(htmlPath);
        this.keepWarm = keepWarm;
        this.log = log;
        this._browser = null;
        this._html = null;
        this._spares = [];    // preloaded pages (promises); a run uses one to build and one to verify
    }

    async browser() {
        if (!this._browser) {
            this._browser = launchBrowser();
            this._browser.catch(() => { this._browser = null; });
        }
        return this._browser;
    }

    _htmlBody() {
        if (!existsSync(this.htmlPath))
            throw new Error(`Arch index.html not found at ${this.htmlPath}. Set ARCH_HTML or pass --html.`);
        const mtime = statSync(this.htmlPath).mtimeMs;
        if (!this._html || this._html.mtime !== mtime) this._html = {mtime, body: readFileSync(this.htmlPath)};
        return this._html.body;
    }

    // Opening a page has no natural time limit (a browser short of memory may never start its renderer),
    // so the whole load gets one, and a page that fails to load is closed again.
    async _loadPage() {
        const body = this._htmlBody();
        let context = null;
        const load = async () => {
            const browser = await this.browser();
            // (a browser short of memory has been seen to never answer this, so it is inside the time limit too)
            context = await browser.newContext({viewport: VIEWPORT, colorScheme: 'light', deviceScaleFactor: 1});
            // Offline: Arch itself is the only response. No login, cloud, AI or notice requests.
            await context.route('**/*', route => {
                if (route.request().url() === PAGE_URL)
                    return route.fulfill({status: 200, contentType: 'text/html; charset=utf-8', body});
                return route.abort();
            });
            const page = await context.newPage();
            const pageErrors = [];
            page.on('pageerror', error => pageErrors.push({message: String(error.message || error), stack: String(error.stack || '')}));
            page.on('dialog', dialog => dialog.dismiss().catch(() => {}));
            const crashed = new Promise(resolve => page.once('crash', resolve));
            await page.goto(PAGE_URL);
            await page.waitForFunction(() => window.archInstance && window.ArchAgentProtocol &&
                (window.archStartupFinished || !document.getElementById('arch-startup-splash')), null, {timeout: READY_TIMEOUT});
            // A first launch (empty storage) opens the About page over the canvas; use the normal plot layout.
            await page.evaluate(() => window.archInstance.createArchScriptExtensions().ui.sidebar('plot'));
            return {page, context, pageErrors, crashed};
        };
        try {
            return await guard(load(), 2 * READY_TIMEOUT, 'Opening Arch in the browser');
        } catch (error) {
            if (context) await closeQuietly(context);
            throw error instanceof RunTimeout ? new Error(`${error.message.split(';')[0]}; the computer may be short of memory (several checks running at once?).`) : error;
        }
    }

    // Each run gets pristine pages so builds never leak state into each other.
    // Returns a promise; with keepWarm the next pages start loading immediately.
    acquire() {
        const spare = this._spares.shift();
        if (this.keepWarm) this._topUp();
        if (!spare) return this._loadPage();
        return spare.catch(() => this._loadPage());
    }

    _topUp() {
        while (this._spares.length < 2) {
            const slot = this._loadPage();
            slot.catch(() => {});
            this._spares.push(slot);
        }
    }

    async close() {
        const spares = this._spares;
        this._spares = [];
        await Promise.all(spares.map(spare => spare.then(s => closeQuietly(s.context), () => {})));
        if (this._browser) {
            const browser = await this._browser.catch(() => null);
            this._browser = null;
            if (browser) await closeQuietly(browser);
        }
    }

    // Runs a scene script. Returns {ok, errors, warnings, objects, view, json, png, scriptPng, ...}.
    // ok is false when any line failed, the saved file does not reload identically, or the
    // @script preview raised an error.
    // lean: leave out the expanded formulas of objects that call custom functions (see stripExpanded); the reload
    // check then also confirms that Arch rebuilds each of them exactly.
    async run(source, {screenshot = true, screenshotScale = 1, verify = true, scriptPreview = true, timeoutMs = RUN_TIMEOUT, lean = false} = {}) {
        let parsed;
        try {
            parsed = parseScript(source);
        } catch (error) {
            if (error instanceof ScriptError)
                return {ok: false, errors: [{line: error.line, message: error.message}], warnings: [], objects: []};
            throw error;
        }
        if (!parsed.steps.some(s => ['create', 'operator', 'text'].includes(s.kind)) && !parsed.script)
            return {ok: false, errors: [{line: 0, message: 'The script creates nothing.'}], warnings: [], objects: []};

        // The verification page loads in parallel with the build, so verifying costs little extra time.
        const buildSlot = this.acquire();
        const verifySlot = verify ? this.acquire() : null;
        verifySlot?.catch(() => {});
        const dropVerify = () => verifySlot?.then(s => closeQuietly(s.context), () => {});
        const slot = await buildSlot.catch(error => { dropVerify(); throw error; });
        let result, png, scriptPng;
        // Drawing happens after the build call returns, so waiting for painted frames doubles as a
        // check that no operator hangs the page.
        const shoot = async (s, what) => {
            await guard(s.page.evaluate(settle, 500), timeoutMs, what, s);
            return guard(capture(s.page, screenshotScale), timeoutMs, what, s);
        };
        try {
            result = await guard(slot.page.evaluate(runSteps, {steps: parsed.steps, title: parsed.title, script: parsed.script, messages: ARCH_MESSAGES}),
                timeoutMs, 'Building the scene', slot);
            if (screenshot && !(verifySlot && result.errors.length === 0)) png = await shoot(slot, 'Drawing the scene');
            else await guard(slot.page.evaluate(settle, 100), timeoutMs, 'Drawing the scene', slot);
            const thrown = await guard(slot.page.evaluate(() => window.__archOperatorErrors || {}), timeoutMs, 'Drawing the scene', slot);
            for (const [name, item] of Object.entries(thrown))
                result.warnings.push({line: item.line, message: `operator ${name} threw "${item.message}" ${item.count} time(s) while Arch computed values; those values become NaN (gaps in curves)`});
            result.warnings.push(...pageErrorWarnings(slot.pageErrors));
        } catch (error) {
            dropVerify();
            if (error instanceof PageCrash) return {ok: false, errors: [{line: 0, message: error.message}], warnings: [], objects: []};
            if (error instanceof RunTimeout) return {ok: false, errors: [blameOperators(error, parsed)], warnings: [], objects: []};
            throw error;
        } finally {
            await closeQuietly(slot.context);
        }

        let fullJson = result.json;
        if (lean && result.errors.length === 0) {
            const stripped = stripExpanded(result.json);
            result.json = stripped.json;
            result.lean = {stripped: stripped.count, bytesBefore: fullJson.length, bytesAfter: stripped.json.length};
        }
        if (verifySlot && result.errors.length === 0) {
            // Reload the exported file exactly as a user would open it; the screenshot shows that view.
            const check = await verifySlot;
            try {
                const reloaded = await guard(check.page.evaluate(loadAndDescribe, result.json), timeoutMs, 'Reloading the saved file', check);
                for (const message of reloaded.errors) result.errors.push({message: `Reloading the saved file failed: ${message}`});
                for (const message of reloaded.warnings || []) result.warnings.push({message: `Arch used synchronous drawing while reloading: ${message}`});
                result.errors.push(...compareReload(result.objects, reloaded.objects));
                result.errors.push(...compareSaved(JSON.parse(fullJson), JSON.parse(reloaded.json)));
                if (result.lean) result.errors.push(...compareExpanded(JSON.parse(fullJson), JSON.parse(reloaded.json)));
                if (screenshot) png = await shoot(check, 'Drawing the reloaded file');
                else await guard(check.page.evaluate(settle, 100), timeoutMs, 'Drawing the reloaded file', check);
                if (parsed.script && scriptPreview && result.errors.length === 0) {
                    // Run the global program the way the Run button does, in this throwaway page.
                    const before = check.pageErrors.length;
                    const preview = await guard(check.page.evaluate(previewScript, PREVIEW_MS), timeoutMs, 'The @script preview', check);
                    if (screenshot) scriptPng = await guard(capture(check.page, screenshotScale), timeoutMs, 'The @script preview', check);
                    // Count page errors only when the stack passes through the script, which Arch compiles
                    // with new Function (shown as <anonymous>/eval). Arch's own blocked network calls are ignored.
                    const fromScript = check.pageErrors.slice(before).filter(e => USER_CODE_FRAME.test(e.stack));
                    const problems = [...preview.problems, ...fromScript.map(e => e.message)];
                    result.scriptPreview = {seconds: PREVIEW_MS / 1000, problems};
                    for (const problem of problems)
                        result.errors.push({line: parsed.script.line, message: `@script failed when run: ${translateArchMessage(problem)}`});
                }
            } catch (error) {
                if (error instanceof PageCrash) result.errors.push({line: 0, message: error.message});
                else if (!(error instanceof RunTimeout)) throw error;
                else result.errors.push(/preview/.test(error.message) ? {line: parsed.script.line, message: error.message} : blameOperators(error, parsed));
            } finally {
                await closeQuietly(check.context);
            }
        } else {
            dropVerify();
        }
        result.errors.sort((a, b) => (a.line ?? 1e9) - (b.line ?? 1e9));
        result.warnings.sort((a, b) => (a.line ?? 1e9) - (b.line ?? 1e9));
        return {ok: result.errors.length === 0, ...result, png, scriptPng};
    }
}

// A hang while drawing can only come from operator code; point at it.
function blameOperators(error, parsed) {
    const ops = parsed.steps.filter(s => s.kind === 'operator');
    if (!ops.length) return {line: 0, message: error.message};
    return {line: ops.length === 1 ? ops[0].line : 0,
        message: `${error.message} Operators in this script: ${ops.map(o => `${o.name} (line ${o.line})`).join(', ')}.`};
}

function compareReload(built, reloaded) {
    const problems = [];
    if (built.length !== reloaded.length) {
        problems.push({message: `Saved file reloads with ${reloaded.length} objects instead of ${built.length}.`});
        return problems;
    }
    const close = (a, b) => (a === b) || (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a)))
        || (Number.isNaN(a) && Number.isNaN(b)) || (a == null && b == null);
    built.forEach((b, i) => {
        const r = reloaded[i];
        const diffs = [];
        if (b.expression !== r.expression) diffs.push(`expression "${b.expression}" → "${r.expression}"`);
        if ((b.details || '') !== (r.details || '')) diffs.push(`value "${b.details}" → "${r.details}"`);
        if (!close(b.value, r.value)) diffs.push(`value ${b.value} → ${r.value}`);
        if (b.meaningful !== r.meaningful) diffs.push(`valid ${b.meaningful} → ${r.meaningful}`);
        if (diffs.length) problems.push({line: b.line, message: `Saved file reloads differently: ${diffs.join('; ')}`});
    });
    return problems;
}

// Document-level parts the object list does not cover: frames, text boxes, operators, the script.
// Arch saves each object's formulas a second time with every custom function's body pasted in at each use
// (processedExpr*), and rebuilds that copy from the object's own formula when it opens a file whenever the object
// calls a custom function. In a large scene built from functions the copies are most of the file (25 of 27 MB in
// examples/claude-chan). Objects that call no custom function keep theirs: Arch does not rebuild them.
const EXPANDED = ['processedExpr', 'processedExprX', 'processedExprY', 'processedExprZ'];

export function stripExpanded(json) {
    const state = JSON.parse(json);
    const names = (state.entries || []).filter(e => e && e.type === 'custom_function_definition').map(e => e.name).filter(Boolean);
    if (!names.length) return {json, count: 0};
    const calls = new RegExp(`(^|[^A-Za-z0-9_])(${names.map(n => n.replace(/[^A-Za-z0-9_]/g, '\\$&')).join('|')})\\s*\\(`);
    let count = 0;
    for (const entry of state.entries) {
        if (!entry || entry.type === 'custom_function_definition' || !EXPANDED.some(k => k in entry)) continue;
        if (!calls.test(entry.originalExprWithPoints || entry.expr || '')) continue;
        for (const k of EXPANDED) delete entry[k];
        count++;
    }
    return {json: JSON.stringify(state), count};
}

function compareExpanded(full, reloaded) {
    const diffs = [];
    (full.entries || []).forEach((entry, i) => {
        const other = (reloaded.entries || [])[i] || {};
        if (EXPANDED.some(k => (entry[k] || '') !== (other[k] || ''))) diffs.push(entry.expr);
    });
    return diffs.map(d => ({message: `Saved file reloads differently: ${d}: its expanded formula (built with --lean)`}));
}

function compareSaved(a, b) {
    // A fresh page leaves some fields undefined that a reload restores as "" (e.g. frameExpr).
    const norm = v => (v === undefined || v === null || v === '') ? null : v;
    const same = (x, y) => JSON.stringify(norm(x)) === JSON.stringify(norm(y));
    const diffs = [];
    if ((a.globalScript || '') !== (b.globalScript || '')) diffs.push('the global script');
    for (const key of ['showPlaybackControls', 'totalFrames', 'playbackFPS', 'currentFrame', 'frameExpr', 'offset3DZ', 'orderedDrawingEnabled', 'smartColorInversionEnabled'])
        if (!same(a[key], b[key])) diffs.push(`${key} ${JSON.stringify(a[key])} → ${JSON.stringify(b[key])}`);
    // With ordered drawing, objects are painted in creation order (Arch 12202+ saves it per object).
    const drawOrder = state => [...(state.entries || []).map((x, i) => ['entry', i, x.creationOrder]),
        ...(state.annotations || []).map((x, i) => ['annotation', i, x.creationOrder])]
        .filter(([, , n]) => Number.isFinite(n)).sort((x, y) => x[2] - y[2]).map(([kind, i]) => `${kind}${i}`);
    if (a.orderedDrawingEnabled && !same(drawOrder(a), drawOrder(b))) diffs.push('the drawing order');
    const styleFormulas = owner => Object.fromEntries(Object.entries(owner.styleExpressions || {}).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, v.expr]));
    const texts = state => (state.annotations || []).filter(t => t.type === 'text')
        .map(t => [t.content, t.x, t.y, t.width, t.height, t.fontSize, t.color, styleFormulas(t)]);
    if (!same(texts(a), texts(b))) diffs.push('text boxes');
    (a.entries || []).forEach((entry, i) => {
        const other = (b.entries || [])[i] || {};
        if (!same(styleFormulas(entry), styleFormulas(other))) diffs.push(`${entry.expr}: style expressions`);
        for (const key of ['displayFrame', 'bodyJs', 'params', 'parametricParameters'])
            if (!same(entry[key], other[key])) diffs.push(`${entry.expr}: ${key}`);
    });
    return diffs.map(d => ({message: `Saved file reloads differently: ${d}`}));
}
