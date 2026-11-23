// How long Arch takes to open a save file, as a user opens it: the bundled Arch reads the file (loadState), then its
// drawing worker meshes every object (until it is idle). With --drop, the surfaces whose ref matches are left out
// first, so the difference is what they cost to open (refs are read from the scene script the file was built from:
// its surfaces in order are the file's parametric surfaces in order).
//
//   node examples/claude-chan/opentime.mjs claude-chan-3d.txt
//   node examples/claude-chan/opentime.mjs claude-chan-3d.txt --arch claude-chan-3d.arch --drop 'lash\w*' --drop 'smile'
//
// Each --drop is timed on its own, after the whole file. Expect about 27 s for the whole figure (on an Apple M4).
import {readFileSync} from 'fs';
import {ArchEngine} from '../../src/engine.js';

const args = process.argv.slice(2);
const file = args.shift();
let arch = null;
const drops = [];
while (args.length) {
    const a = args.shift();
    if (a === '--arch') arch = args.shift();
    else if (a === '--drop') drops.push(args.shift());
}
const state = JSON.parse(readFileSync(file, 'utf8'));
const surfaces = state.entries.map((e, i) => [e, i]).filter(([e]) => e.parametricParameters && e.parametricParameters.length === 2).map(([, i]) => i);
let refs = null;
if (drops.length) {
    if (!arch) throw new Error('--drop needs --arch, the scene script the file was built from');
    // the scene's surface lines: "(X,Y,Z,u,u0,u1,v,v0,v1) | ref=..." (curves start with "curve(")
    refs = readFileSync(arch, 'utf8').split('\n').filter(l => l.startsWith('(') && l.includes(' | ')).map(l => (l.match(/ref=(\S+)/) || [])[1] || '');
    if (refs.length !== surfaces.length)
        throw new Error(`${arch} has ${refs.length} surfaces, ${file} ${surfaces.length}: not the script it was built from`);
}
const engine = new ArchEngine();
async function time(json) {
    const slot = await engine._loadPage();
    try {
        return await slot.page.evaluate(async json => {
            const e = window.archInstance;
            const t0 = performance.now();
            e.loadState(json);
            const t1 = performance.now();
            if (e._asyncDrawing?.waitForIdle) await e._asyncDrawing.waitForIdle();
            return {read: (t1 - t0) / 1000, draw: (performance.now() - t1) / 1000};
        }, json);
    } finally {
        await slot.context.close();
    }
}
try {
    const whole = await time(JSON.stringify(state));
    console.log(`whole file: ${(whole.read + whole.draw).toFixed(1)} s (reading ${whole.read.toFixed(1)} s, meshing ${whole.draw.toFixed(1)} s)`);
    for (const d of drops) {
        const rx = new RegExp(`^(${d})$`);
        const out = new Set(surfaces.filter((_, k) => rx.test(refs[k])));
        const t = await time(JSON.stringify({...state, entries: state.entries.filter((_, i) => !out.has(i))}));
        console.log(`without ${d} (${out.size} surfaces): ${(t.read + t.draw).toFixed(1)} s, so they cost ${(whole.read + whole.draw - t.read - t.draw).toFixed(1)} s`);
    }
} finally {
    await engine.close();
}
