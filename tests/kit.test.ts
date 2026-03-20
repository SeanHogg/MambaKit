/**
 * kit.test.ts – Unit tests for MambaKit (no GPU required).
 *
 * All low-level MambaCode.js classes are mocked so tests run in Node.js.
 */

/// <reference types="@webgpu/types" />

// ── Mocks for mambacode.js ────────────────────────────────────────────────────

const mockDevice = {
    destroy       : jest.fn(),
    createBuffer  : jest.fn(),
    queue         : { submit: jest.fn(), writeBuffer: jest.fn() },
    createShaderModule   : jest.fn(),
    createComputePipeline: jest.fn(),
    createBindGroup      : jest.fn(),
    createCommandEncoder : jest.fn(),
    lost                 : new Promise(() => { /* never resolves */ }),
} as unknown as GPUDevice;

const mockTokenizer = {
    vocabSize       : 1000,
    encode          : jest.fn((text: string) => text.split(' ').map((_, i) => i + 1)),
    decode          : jest.fn((ids: number[]) => ids.map(id => `tok${id}`).join(' ')),
    load            : jest.fn().mockResolvedValue(undefined),
    loadFromObjects : jest.fn(),
    bosId           : null,
    eosId           : null,
    padId           : null,
    vocab           : new Map<string, number>(),
    idToToken       : new Map<number, string>(),
    merges          : new Map<string, number>(),
};

const mockModel = {
    config : { vocabSize: 1000, dModel: 512, numLayers: 8, dState: 16, dConv: 4, expand: 2, eosId: -1 },
    generate       : jest.fn().mockResolvedValue([1, 2, 3, 4, 5]),
    forward        : jest.fn().mockResolvedValue({
        logits   : new Float32Array(1000).fill(0.1),
        gpuLogits: {} as unknown as GPUBuffer,
        caches   : [],
    }),
    exportWeights  : jest.fn().mockResolvedValue(new ArrayBuffer(16)),
    loadWeights    : jest.fn().mockResolvedValue(undefined),
    parameters     : jest.fn().mockReturnValue([]),
    setWSLAMode    : jest.fn(),
    device         : mockDevice,
};

const mockTrainer = {
    train   : jest.fn().mockResolvedValue([1.5, 1.2, 0.9]),
    evaluate: jest.fn().mockResolvedValue(42.0),
    model   : mockModel,
    tokenizer: mockTokenizer,
    device  : mockDevice,
};

// Mock the entire mambacode.js module
jest.mock('mambacode.js', () => ({
    initWebGPU  : jest.fn().mockResolvedValue({ device: mockDevice, adapter: {} }),
    BPETokenizer: jest.fn().mockImplementation(() => mockTokenizer),
    MambaModel  : jest.fn().mockImplementation(() => mockModel),
    MambaTrainer: jest.fn().mockImplementation(() => mockTrainer),
}));

// ── Imports (after mocks are set up) ─────────────────────────────────────────

import { MambaKitError }                                              from '../src/kit/errors.js';
import { MODEL_PRESETS, resolveModelConfig }                          from '../src/kit/presets.js';
import { saveToIndexedDB, loadFromIndexedDB, triggerDownload }        from '../src/kit/persistence.js';
import { MambaSession }                                               from '../src/kit/session.js';
import type { MambaSessionOptions }                                   from '../src/kit/session.js';

// ── IndexedDB mock ────────────────────────────────────────────────────────────

const idbStore = new Map<string, ArrayBuffer>();

type MockIDBRequest = {
    result        : unknown;
    error         : DOMException | null;
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
                const r: MockIDBRequest = {
                    result: undefined, error: null,
                    onupgradeneeded: null, onsuccess: null, onerror: null,
                    readyState: 'done',
                };
                setTimeout(() => r.onsuccess?.(new Event('success')), 0);
                return r;
            }),
            get: jest.fn((key: string) => {
                const r: MockIDBRequest = {
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
            transaction      : jest.fn(() => tx),
            close            : jest.fn(),
        } as unknown as IDBDatabase;

        const req: MockIDBRequest = {
            result        : db,
            error         : null,
            onupgradeneeded: null,
            onsuccess     : null,
            onerror       : null,
            readyState    : 'done',
        };

        setTimeout(() => req.onsuccess?.(new Event('success')), 0);
        return req;
    }),
};

Object.defineProperty(globalThis, 'indexedDB', { value: mockIDB, writable: true });

// ── URL / Blob / document mocks ───────────────────────────────────────────────

Object.defineProperty(globalThis, 'URL', {
    value  : { createObjectURL: jest.fn(() => 'blob:mock'), revokeObjectURL: jest.fn() },
    writable: true,
});

Object.defineProperty(globalThis, 'Blob', {
    value  : class MockBlob { constructor(public parts: unknown[], public opts?: unknown) {} },
    writable: true,
});

Object.defineProperty(globalThis, 'document', {
    value: {
        createElement  : jest.fn(() => ({ style: {}, click: jest.fn(), href: '', download: '' })),
        body           : { appendChild: jest.fn(), removeChild: jest.fn() },
    },
    writable: true,
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeMinimalOptions(overrides: Partial<MambaSessionOptions> = {}): MambaSessionOptions {
    return {
        vocabObject: { hello: 0, world: 1 },
        mergesArray: [],
        ...overrides,
    };
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
        const err = new MambaKitError('UNKNOWN', 'oops');
        expect(err).toBeInstanceOf(Error);
    });

    test('name is MambaKitError', () => {
        const err = new MambaKitError('SESSION_DESTROYED', 'destroyed');
        expect(err.name).toBe('MambaKitError');
    });

    test('message is set correctly', () => {
        const err = new MambaKitError('CHECKPOINT_INVALID', 'bad file');
        expect(err.message).toBe('bad file');
    });

    test('cause is stored', () => {
        const original = new Error('original');
        const err      = new MambaKitError('UNKNOWN', 'wrapped', original);
        expect(err.cause).toBe(original);
    });
});

// ── resolveModelConfig ────────────────────────────────────────────────────────

describe('resolveModelConfig', () => {
    test('medium preset fills all required fields', () => {
        const config = resolveModelConfig({}, 5000);
        expect(config.dModel).toBe(512);
        expect(config.numLayers).toBe(8);
        expect(config.dState).toBe(16);
        expect(config.dConv).toBe(4);
        expect(config.expand).toBe(2);
        expect(config.vocabSize).toBe(5000);
        expect(typeof config.eosId).toBe('number');
    });

    test('nano preset uses correct dimensions', () => {
        const config = resolveModelConfig({ modelSize: 'nano' }, 1000);
        expect(config.dModel).toBe(128);
        expect(config.numLayers).toBe(4);
    });

    test('small preset uses correct dimensions', () => {
        const config = resolveModelConfig({ modelSize: 'small' }, 1000);
        expect(config.dModel).toBe(256);
        expect(config.numLayers).toBe(6);
    });

    test('large preset uses correct dimensions', () => {
        const config = resolveModelConfig({ modelSize: 'large' }, 1000);
        expect(config.dModel).toBe(768);
        expect(config.numLayers).toBe(12);
    });

    test('custom modelSize with overrides respects custom values', () => {
        const config = resolveModelConfig({
            modelSize  : 'custom',
            modelConfig: { dModel: 333, numLayers: 7 },
        }, 2000);
        expect(config.dModel).toBe(333);
        expect(config.numLayers).toBe(7);
        expect(config.vocabSize).toBe(2000);
    });

    test('custom overrides are NOT applied when modelSize is not "custom"', () => {
        const config = resolveModelConfig({
            modelSize  : 'nano',
            modelConfig: { dModel: 999 },
        }, 1000);
        expect(config.dModel).toBe(128); // nano value, not the override
    });

    test('vocabSize always reflects the argument', () => {
        const config = resolveModelConfig({}, 42_000);
        expect(config.vocabSize).toBe(42_000);
    });
});

// ── MODEL_PRESETS ─────────────────────────────────────────────────────────────

describe('MODEL_PRESETS', () => {
    test('contains nano, small, medium, large keys', () => {
        expect(MODEL_PRESETS).toHaveProperty('nano');
        expect(MODEL_PRESETS).toHaveProperty('small');
        expect(MODEL_PRESETS).toHaveProperty('medium');
        expect(MODEL_PRESETS).toHaveProperty('large');
    });
});

// ── IndexedDB persistence ─────────────────────────────────────────────────────

describe('saveToIndexedDB + loadFromIndexedDB round-trip', () => {
    beforeEach(() => idbStore.clear());

    test('saves and retrieves an ArrayBuffer', async () => {
        const original = new Uint8Array([1, 2, 3, 4]).buffer;
        await saveToIndexedDB('test-key', original);
        const loaded = await loadFromIndexedDB('test-key');
        expect(loaded).toBeDefined();
        expect(new Uint8Array(loaded!)).toEqual(new Uint8Array([1, 2, 3, 4]));
    });

    test('returns undefined for a missing key', async () => {
        const result = await loadFromIndexedDB('nonexistent-key');
        expect(result).toBeUndefined();
    });
});

// ── triggerDownload ───────────────────────────────────────────────────────────

describe('triggerDownload', () => {
    test('does not throw', async () => {
        const buffer = new ArrayBuffer(8);
        await expect(triggerDownload('test.bin', buffer)).resolves.not.toThrow();
    });

    test('calls URL.createObjectURL', async () => {
        (URL.createObjectURL as jest.Mock).mockClear();
        await triggerDownload('out.bin', new ArrayBuffer(4));
        expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    });
});

// ── MambaSession ──────────────────────────────────────────────────────────────

describe('MambaSession', () => {
    let session: MambaSession;

    beforeEach(async () => {
        jest.clearAllMocks();
        // Re-apply the mockTokenizer so encode/decode work predictably
        mockTokenizer.encode.mockImplementation(
            (text: string) => text.split(' ').map((_, i) => i + 1),
        );
        mockTokenizer.decode.mockImplementation(
            (ids: number[]) => ids.map(id => `tok${id}`).join(' '),
        );
        mockTokenizer.vocabSize = 1000;
        mockModel.generate.mockResolvedValue([1, 2, 3, 4, 5]);

        session = await MambaSession.create(makeMinimalOptions());
    });

    test('create() returns a MambaSession instance', () => {
        expect(session).toBeInstanceOf(MambaSession);
    });

    test('internals exposes model, trainer, tokenizer, device', () => {
        const { model, trainer, tokenizer, device } = session.internals;
        expect(model).toBeDefined();
        expect(trainer).toBeDefined();
        expect(tokenizer).toBeDefined();
        expect(device).toBeDefined();
    });

    test('complete() returns a string', async () => {
        const result = await session.complete('hello world');
        expect(typeof result).toBe('string');
    });

    test('complete() returns the continuation, not the prompt tokens', async () => {
        // prompt encodes to [1,2], generate returns [1,2,3,4,5], continuation = [3,4,5]
        mockTokenizer.encode.mockReturnValueOnce([1, 2]);
        mockModel.generate.mockResolvedValueOnce([1, 2, 3, 4, 5]);
        const result = await session.complete('hello world');
        // decode([3,4,5]) -> "tok3 tok4 tok5"
        expect(result).toBe('tok3 tok4 tok5');
    });

    test('adapt() returns an AdaptResult', async () => {
        mockTrainer.train.mockResolvedValueOnce([1.5, 1.2, 0.9]);
        const result = await session.adapt('some training text that is long enough');
        expect(result).toHaveProperty('losses');
        expect(result).toHaveProperty('epochCount');
        expect(result).toHaveProperty('durationMs');
        expect(result.losses).toEqual([1.5, 1.2, 0.9]);
        expect(result.epochCount).toBe(3);
    });

    test('adapt() throws INPUT_TOO_SHORT when input encodes to fewer than 2 tokens', async () => {
        mockTokenizer.encode.mockReturnValueOnce([1]); // single token
        await expect(session.adapt('x')).rejects.toMatchObject({
            code: 'INPUT_TOO_SHORT',
        });
    });

    test('evaluate() returns a number', async () => {
        const ppl = await session.evaluate('some code');
        expect(typeof ppl).toBe('number');
    });

    test('save() calls exportWeights', async () => {
        await session.save();
        expect(mockModel.exportWeights).toHaveBeenCalled();
    });

    // SESSION_DESTROYED guard tests
    describe('after destroy()', () => {
        beforeEach(() => session.destroy());

        test('complete() throws SESSION_DESTROYED', async () => {
            await expect(session.complete('hi')).rejects.toMatchObject({
                code: 'SESSION_DESTROYED',
            });
        });

        test('adapt() throws SESSION_DESTROYED', async () => {
            await expect(session.adapt('hi there')).rejects.toMatchObject({
                code: 'SESSION_DESTROYED',
            });
        });

        test('evaluate() throws SESSION_DESTROYED', async () => {
            await expect(session.evaluate('hi')).rejects.toMatchObject({
                code: 'SESSION_DESTROYED',
            });
        });

        test('save() throws SESSION_DESTROYED', async () => {
            await expect(session.save()).rejects.toMatchObject({
                code: 'SESSION_DESTROYED',
            });
        });

        test('load() throws SESSION_DESTROYED', async () => {
            await expect(session.load()).rejects.toMatchObject({
                code: 'SESSION_DESTROYED',
            });
        });
    });

    test('create() with checkpointUrl calls model.loadWeights', async () => {
        globalThis.fetch = jest.fn().mockResolvedValue({
            ok         : true,
            status     : 200,
            statusText : 'OK',
            arrayBuffer: jest.fn().mockResolvedValue(new ArrayBuffer(32)),
        }) as typeof fetch;

        const s = await MambaSession.create(makeMinimalOptions({
            checkpointUrl: 'http://example.com/model.bin',
        }));

        expect(mockModel.loadWeights).toHaveBeenCalled();
        s.destroy();
    });

    test('create() with onProgress callback fires for each stage', async () => {
        const events: string[] = [];
        await MambaSession.create(makeMinimalOptions(), {
            onProgress: (e) => events.push(e.stage),
        });
        expect(events).toContain('gpu');
        expect(events).toContain('tokenizer');
        expect(events).toContain('model');
    });

    test('adapt() passes wsla=true by default to trainer.train', async () => {
        mockTrainer.train.mockResolvedValueOnce([1.0]);
        mockTokenizer.encode.mockReturnValueOnce([1, 2, 3]);
        await session.adapt('hello world foo');
        expect(mockTrainer.train).toHaveBeenCalledWith(
            expect.any(String),
            expect.objectContaining({ wsla: true }),
        );
    });

    test('adapt() with fullTrain=true passes wsla=false', async () => {
        mockTrainer.train.mockResolvedValueOnce([1.0]);
        mockTokenizer.encode.mockReturnValueOnce([1, 2, 3]);
        await session.adapt('hello world foo', { fullTrain: true });
        expect(mockTrainer.train).toHaveBeenCalledWith(
            expect.any(String),
            expect.objectContaining({ wsla: false }),
        );
    });
});
