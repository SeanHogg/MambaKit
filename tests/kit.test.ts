/**
 * kit.test.ts – Unit tests for MambaKit (no GPU required).
 *
 * Uses Jest ESM-compatible mocking (jest.unstable_mockModule + dynamic import).
 */

/// <reference types="@webgpu/types" />

import { jest } from '@jest/globals';

// ── Shared mock objects ───────────────────────────────────────────────────────

const mockDevice = {
    destroy              : jest.fn(),
    createBuffer         : jest.fn(),
    queue                : { submit: jest.fn(), writeBuffer: jest.fn() },
    createShaderModule   : jest.fn(),
    createComputePipeline: jest.fn(),
    createBindGroup      : jest.fn(),
    createCommandEncoder : jest.fn(),
    lost                 : new Promise<never>(() => { /* never resolves */ }),
} as unknown as GPUDevice;

const mockTokenizer = {
    vocabSize      : 1000,
    encode         : jest.fn((text: string) => text.split(' ').map((_, i) => i + 1)),
    decode         : jest.fn((ids: number[]) => ids.map(id => `tok${id}`).join(' ')),
    load           : jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    loadFromObjects: jest.fn(),
    bosId          : null,
    eosId          : null,
    padId          : null,
    vocab          : new Map<string, number>(),
    idToToken      : new Map<number, string>(),
    merges         : new Map<string, number>(),
};

const mockModel = {
    config: { vocabSize: 1000, dModel: 512, numLayers: 8, dState: 16, dConv: 4, expand: 2, eosId: -1 },
    generate     : jest.fn<() => Promise<number[]>>().mockResolvedValue([1, 2, 3, 4, 5]),
    forward      : jest.fn<() => Promise<{ logits: Float32Array; gpuLogits: unknown; caches: unknown[] }>>()
        .mockResolvedValue({
            logits   : new Float32Array(1000).fill(0.1),
            gpuLogits: {},
            caches   : [],
        }),
    exportWeights: jest.fn<() => Promise<ArrayBuffer>>().mockResolvedValue(new ArrayBuffer(16)),
    loadWeights  : jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    parameters   : jest.fn().mockReturnValue([]),
    setWSLAMode  : jest.fn(),
    device       : mockDevice,
};

const mockTrainer = {
    train    : jest.fn<() => Promise<number[]>>().mockResolvedValue([1.5, 1.2, 0.9]),
    evaluate : jest.fn<() => Promise<number>>().mockResolvedValue(42.0),
    model    : mockModel,
    tokenizer: mockTokenizer,
    device   : mockDevice,
};

// ── Mock mambacode.js BEFORE any imports that use it ─────────────────────────

jest.unstable_mockModule('@seanhogg/mambacode.js', () => ({
    initWebGPU  : jest.fn<() => Promise<{ device: GPUDevice; adapter: unknown }>>()
        .mockResolvedValue({ device: mockDevice, adapter: {} }),
    BPETokenizer: jest.fn().mockImplementation(() => mockTokenizer),
    HybridMambaModel: jest.fn().mockImplementation(() => mockModel),
    MambaTrainer: jest.fn().mockImplementation(() => mockTrainer),
}));

// ── Dynamic imports (must come after unstable_mockModule) ─────────────────────

const { MambaKitError }                             = await import('../src/kit/errors.js');
const { MODEL_PRESETS, resolveModelConfig }          = await import('../src/kit/presets.js');
const { saveToIndexedDB, loadFromIndexedDB, triggerDownload } = await import('../src/kit/persistence.js');
const { MambaSession }                              = await import('../src/kit/session.js');

// ── IndexedDB mock ────────────────────────────────────────────────────────────

const idbStore = new Map<string, ArrayBuffer>();

type MockIDBReq = {
    result        : unknown;
    error         : null;
    onupgradeneeded: ((e: Event) => void) | null;
    onsuccess     : ((e: Event) => void) | null;
    onerror       : ((e: Event) => void) | null;
    readyState    : string;
};

const mockIDB = {
    open: jest.fn((_name: string, _version: number) => {
        const store = {
            put: jest.fn((value: ArrayBuffer, key: string) => {
                idbStore.set(key, value);
                const r: MockIDBReq = {
                    result: undefined, error: null,
                    onupgradeneeded: null, onsuccess: null, onerror: null,
                    readyState: 'done',
                };
                setTimeout(() => r.onsuccess?.(new Event('success')), 0);
                return r;
            }),
            get: jest.fn((key: string) => {
                const r: MockIDBReq = {
                    result: idbStore.get(key),
                    error: null,
                    onupgradeneeded: null, onsuccess: null, onerror: null,
                    readyState: 'done',
                };
                setTimeout(() => r.onsuccess?.(new Event('success')), 0);
                return r;
            }),
        };
        const tx = {
            objectStore: jest.fn(() => store),
            oncomplete : null as ((e: Event) => void) | null,
        };
        const db = {
            objectStoreNames: { contains: jest.fn(() => true) },
            transaction     : jest.fn(() => tx),
            close           : jest.fn(),
        } as unknown as IDBDatabase;
        const req: MockIDBReq = {
            result: db, error: null,
            onupgradeneeded: null, onsuccess: null, onerror: null,
            readyState: 'done',
        };
        setTimeout(() => req.onsuccess?.(new Event('success')), 0);
        return req;
    }),
};

Object.defineProperty(globalThis, 'indexedDB', { value: mockIDB, writable: true, configurable: true });

// ── URL / Blob / document mocks ───────────────────────────────────────────────

Object.defineProperty(globalThis, 'URL', {
    value   : { createObjectURL: jest.fn(() => 'blob:mock'), revokeObjectURL: jest.fn() },
    writable: true, configurable: true,
});

Object.defineProperty(globalThis, 'Blob', {
    value   : class MockBlob { constructor(public parts: unknown[], public opts?: unknown) {} },
    writable: true, configurable: true,
});

Object.defineProperty(globalThis, 'document', {
    value: {
        createElement: jest.fn(() => ({ style: {}, click: jest.fn(), href: '', download: '' })),
        body         : { appendChild: jest.fn(), removeChild: jest.fn() },
    },
    writable: true, configurable: true,
});

// ── Helpers ───────────────────────────────────────────────────────────────────

type MambaSessionOptions = Parameters<typeof MambaSession.create>[0];

function minimalOpts(overrides: Partial<MambaSessionOptions> = {}): MambaSessionOptions {
    return { vocabObject: { hello: 0, world: 1 }, mergesArray: [], ...overrides };
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

// ── MambaKitError ─────────────────────────────────────────────────────────────

describe('MambaKitError', () => {
    test('has correct code', () => {
        const err = new MambaKitError('GPU_UNAVAILABLE', 'no gpu');
        expect(err.code).toBe('GPU_UNAVAILABLE');
    });

    test('extends Error', () => {
        expect(new MambaKitError('UNKNOWN', 'oops')).toBeInstanceOf(Error);
    });

    test('name is MambaKitError', () => {
        expect(new MambaKitError('SESSION_DESTROYED', 'destroyed').name).toBe('MambaKitError');
    });

    test('message is set correctly', () => {
        expect(new MambaKitError('CHECKPOINT_INVALID', 'bad file').message).toBe('bad file');
    });

    test('cause is stored', () => {
        const original = new Error('original');
        expect(new MambaKitError('UNKNOWN', 'wrapped', original).cause).toBe(original);
    });
});

// ── resolveModelConfig ────────────────────────────────────────────────────────

describe('resolveModelConfig', () => {
    test('nano preset (default) fills all required fields', () => {
        const cfg = resolveModelConfig({}, 5000);
        expect(cfg).toMatchObject({ dModel: 128, numLayers: 4, dState: 16, dConv: 4, expand: 2, vocabSize: 5000 });
        expect(typeof cfg.eosId).toBe('number');
    });

    test('nano preset', () => {
        const cfg = resolveModelConfig({ modelSize: 'nano' }, 1000);
        expect(cfg.dModel).toBe(128);
        expect(cfg.numLayers).toBe(4);
    });

    test('small preset', () => {
        const cfg = resolveModelConfig({ modelSize: 'small' }, 1000);
        expect(cfg.dModel).toBe(256);
        expect(cfg.numLayers).toBe(6);
    });

    test('large preset', () => {
        const cfg = resolveModelConfig({ modelSize: 'large' }, 1000);
        expect(cfg.dModel).toBe(768);
        expect(cfg.numLayers).toBe(12);
    });

    test('custom overrides dModel and numLayers', () => {
        const cfg = resolveModelConfig({ modelSize: 'custom', modelConfig: { dModel: 256, numLayers: 7, nHeads: 4 } }, 2000);
        expect(cfg.dModel).toBe(256);
        expect(cfg.numLayers).toBe(7);
        expect(cfg.vocabSize).toBe(2000);
    });

    test('custom overrides ignored when modelSize is not "custom"', () => {
        const cfg = resolveModelConfig({ modelSize: 'nano', modelConfig: { dModel: 999 } }, 1000);
        expect(cfg.dModel).toBe(128);
    });

    test('vocabSize is always the supplied argument', () => {
        expect(resolveModelConfig({}, 42_000).vocabSize).toBe(42_000);
    });
});

// ── MODEL_PRESETS ─────────────────────────────────────────────────────────────

describe('MODEL_PRESETS', () => {
    test('contains nano, small, medium, large', () => {
        for (const key of ['nano', 'small', 'medium', 'large']) {
            expect(MODEL_PRESETS).toHaveProperty(key);
        }
    });
});

// ── IndexedDB persistence ─────────────────────────────────────────────────────

describe('saveToIndexedDB + loadFromIndexedDB round-trip', () => {
    beforeEach(() => idbStore.clear());

    test('saves and retrieves an ArrayBuffer', async () => {
        const buf = new Uint8Array([10, 20, 30, 40]).buffer;
        await saveToIndexedDB('key1', buf);
        const out = await loadFromIndexedDB('key1');
        expect(new Uint8Array(out!)).toEqual(new Uint8Array([10, 20, 30, 40]));
    });

    test('returns undefined for unknown key', async () => {
        expect(await loadFromIndexedDB('no-such-key')).toBeUndefined();
    });
});

// ── triggerDownload ───────────────────────────────────────────────────────────

describe('triggerDownload', () => {
    test('resolves without throwing', async () => {
        await expect(triggerDownload('out.bin', new ArrayBuffer(8))).resolves.not.toThrow();
    });

    test('calls URL.createObjectURL once', async () => {
        (URL.createObjectURL as ReturnType<typeof jest.fn>).mockClear();
        await triggerDownload('out.bin', new ArrayBuffer(4));
        expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    });
});

// ── MambaSession ──────────────────────────────────────────────────────────────

describe('MambaSession', () => {
    let session: Awaited<ReturnType<typeof MambaSession.create>>;

    beforeEach(async () => {
        jest.clearAllMocks();
        mockTokenizer.encode.mockImplementation((text: string) => text.split(' ').map((_, i) => i + 1));
        mockTokenizer.decode.mockImplementation((ids: number[]) => ids.map(id => `tok${id}`).join(' '));
        (mockTokenizer as { vocabSize: number }).vocabSize = 1000;
        mockModel.generate.mockResolvedValue([1, 2, 3, 4, 5]);
        session = await MambaSession.create(minimalOpts());
    });

    test('create() returns a MambaSession instance', () => {
        expect(session).toBeInstanceOf(MambaSession);
    });

    test('internals exposes device, model, trainer, tokenizer', () => {
        const { device, model, trainer, tokenizer } = session.internals;
        expect(device).toBeDefined();
        expect(model).toBeDefined();
        expect(trainer).toBeDefined();
        expect(tokenizer).toBeDefined();
    });

    test('complete() returns a string', async () => {
        expect(typeof await session.complete('hello world')).toBe('string');
    });

    test('complete() returns the continuation only (not prompt tokens)', async () => {
        mockTokenizer.encode.mockReturnValueOnce([1, 2]);
        mockModel.generate.mockResolvedValueOnce([1, 2, 3, 4, 5]);
        // continuation = outputIds.slice(promptLen=2) = [3,4,5]
        expect(await session.complete('hello world')).toBe('tok3 tok4 tok5');
    });

    test('adapt() returns AdaptResult shape', async () => {
        mockTrainer.train.mockResolvedValueOnce([1.5, 1.2, 0.9]);
        const r = await session.adapt('some long enough training text here');
        expect(r).toMatchObject({ epochCount: 3, losses: [1.5, 1.2, 0.9] });
        expect(typeof r.durationMs).toBe('number');
    });

    test('adapt() throws INPUT_TOO_SHORT for single-token input', async () => {
        mockTokenizer.encode.mockReturnValueOnce([1]);
        await expect(session.adapt('x')).rejects.toMatchObject({ code: 'INPUT_TOO_SHORT' });
    });

    test('evaluate() returns a number', async () => {
        expect(typeof await session.evaluate('code')).toBe('number');
    });

    test('save() calls model.exportWeights', async () => {
        await session.save();
        expect(mockModel.exportWeights).toHaveBeenCalled();
    });

    test('create() with checkpointUrl calls model.loadWeights', async () => {
        globalThis.fetch = jest.fn<typeof fetch>().mockResolvedValue({
            ok: true, status: 200, statusText: 'OK',
            arrayBuffer: jest.fn<() => Promise<ArrayBuffer>>().mockResolvedValue(new ArrayBuffer(32)),
        } as unknown as Response);

        const s = await MambaSession.create(minimalOpts({ checkpointUrl: 'http://example.com/model.bin' }));
        expect(mockModel.loadWeights).toHaveBeenCalled();
        s.destroy();
    });

    test('create() fires onProgress for gpu, tokenizer, model stages', async () => {
        const stages: string[] = [];
        await MambaSession.create(minimalOpts(), { onProgress: e => stages.push(e.stage) });
        expect(stages).toEqual(expect.arrayContaining(['gpu', 'tokenizer', 'model']));
    });

    test('adapt() sends wsla=true by default', async () => {
        mockTrainer.train.mockResolvedValueOnce([1.0]);
        mockTokenizer.encode.mockReturnValueOnce([1, 2, 3]);
        await session.adapt('a b c');
        expect(mockTrainer.train).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ wsla: true }));
    });

    test('adapt() sends wsla=false when fullTrain=true', async () => {
        mockTrainer.train.mockResolvedValueOnce([1.0]);
        mockTokenizer.encode.mockReturnValueOnce([1, 2, 3]);
        await session.adapt('a b c', { fullTrain: true });
        expect(mockTrainer.train).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ wsla: false }));
    });

    describe('after destroy()', () => {
        beforeEach(() => session.destroy());

        test('complete() throws SESSION_DESTROYED', async () => {
            await expect(session.complete('hi')).rejects.toMatchObject({ code: 'SESSION_DESTROYED' });
        });

        test('adapt() throws SESSION_DESTROYED', async () => {
            await expect(session.adapt('hi there')).rejects.toMatchObject({ code: 'SESSION_DESTROYED' });
        });

        test('evaluate() throws SESSION_DESTROYED', async () => {
            await expect(session.evaluate('hi')).rejects.toMatchObject({ code: 'SESSION_DESTROYED' });
        });

        test('save() throws SESSION_DESTROYED', async () => {
            await expect(session.save()).rejects.toMatchObject({ code: 'SESSION_DESTROYED' });
        });

        test('load() throws SESSION_DESTROYED', async () => {
            await expect(session.load()).rejects.toMatchObject({ code: 'SESSION_DESTROYED' });
        });
    });
});
