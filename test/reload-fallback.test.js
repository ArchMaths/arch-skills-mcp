import {test} from 'node:test';
import assert from 'node:assert/strict';
import {loadAndDescribe} from '../src/page-runner.js';

const FALLBACK = '异步绘图暂不可用，已切回同步绘图';

async function withPage(messages, check, {alert, idleError} = {}) {
    const originalWindow = globalThis.window;
    const showTopPrompt = () => {};
    const originalAlert = () => {};
    const e = {
        entries: [{expr: 'y=x'}],
        asyncDrawingEnabled: false,
        showTopPrompt,
        loadState() {
            for (const [message, isError] of messages) this.showTopPrompt(message, isError);
            if (alert) window.alert(alert);
        },
        _asyncDrawing: {async waitForIdle() { if (idleError) throw idleError; }},
        exportStateObject() { return {entries: this.entries}; },
    };
    globalThis.window = {
        archInstance: e,
        alert: originalAlert,
        ArchAgentProtocol: {AgentAPI: class {describe(entry) { return {expression: entry.expr}; }}},
    };
    try {
        await check(e);
        assert.equal(e.showTopPrompt, showTopPrompt, 'prompt hook restored');
        assert.equal(window.alert, originalAlert, 'alert hook restored');
    } finally {
        if (originalWindow === undefined) delete globalThis.window;
        else globalThis.window = originalWindow;
    }
}

test('synchronous drawing fallback is a warning and still describes the reloaded file', async () => {
    await withPage([[FALLBACK, true]], async () => {
        const result = await loadAndDescribe('{"entries":[]}');
        assert.deepEqual(result.errors, []);
        assert.deepEqual(result.warnings, [FALLBACK]);
        assert.deepEqual(result.objects, [{expression: 'y=x'}]);
        assert.equal(JSON.parse(result.json).entries.length, 1);
    });
});

test('other reload error prompts and alerts remain fatal', async () => {
    await withPage([[FALLBACK, true], ['Invalid file', true], ['Ordinary notice', false]], async () => {
        const result = await loadAndDescribe('{"entries":[]}');
        assert.deepEqual(result.errors, ['Invalid file', 'Load alert']);
        assert.deepEqual(result.warnings, [FALLBACK]);
    }, {alert: 'Load alert'});
});

test('fallback without a confirmed switch to synchronous drawing remains fatal', async () => {
    await withPage([[FALLBACK, true]], async e => {
        e.asyncDrawingEnabled = true;
        const result = await loadAndDescribe('{"entries":[]}');
        assert.deepEqual(result.errors, [FALLBACK]);
        assert.deepEqual(result.warnings, []);
    });
});

test('fallback warnings retain the worker failure reason', async () => {
    await withPage([[FALLBACK, true]], async e => {
        e._asyncDrawing.error = 'Computation timed out';
        const result = await loadAndDescribe('{"entries":[]}');
        assert.deepEqual(result.errors, []);
        assert.deepEqual(result.warnings, [FALLBACK + ' (Computation timed out)']);
    });
});

test('reload exceptions are not swallowed and hooks are restored', async () => {
    const failure = new Error('worker failed');
    await withPage([], async () => {
        await assert.rejects(loadAndDescribe('{"entries":[]}'), failure);
    }, {idleError: failure});
});
