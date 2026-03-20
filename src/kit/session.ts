/**
 * session.ts – MambaSession: the single entry point for all MambaKit functionality.
 *
 * MambaSession is a facade over MambaCode.js that collapses the 8-step
 * async setup sequence into a single `MambaSession.create()` call.
 */

import {
    initWebGPU,
    BPETokenizer,
    MambaModel,
    MambaTrainer,
    type MambaModelConfig,
} from 'mambacode.js';

import { MambaKitError } from './errors.js';
import { resolveModelConfig } from './presets.js';
import {
    saveToIndexedDB,
    loadFromIndexedDB,
    triggerDownload,
    saveViaFileSystemAPI,
    loadViaFileSystemAPI,
} from './persistence.js';
import { tokenStream } from './streaming.js';

// ── Public type definitions ────────────────────────────────────────────────────

export interface MambaSessionOptions {
    /** URL to a .bin checkpoint file. Optional — model starts with random weights if omitted. */
    checkpointUrl?  : string;
    /** URL to vocab.json (Qwen3.5-Coder compatible). Required unless vocabObject is supplied. */
    vocabUrl?       : string;
    /** URL to merges.txt. Required unless mergesArray is supplied. */
    mergesUrl?      : string;
    /** In-memory vocabulary object — alternative to vocabUrl. */
    vocabObject?    : Record<string, number>;
    /** In-memory merges array — alternative to mergesUrl. */
    mergesArray?    : string[];
    /** Unique name for this session, used as the IndexedDB key. Default: 'default'. */
    name?           : string;
    /**
     * Model size preset. Overrides individual model config fields.
     * - 'nano'   : dModel=128,  numLayers=4
     * - 'small'  : dModel=256,  numLayers=6
     * - 'medium' : dModel=512,  numLayers=8  (default)
     * - 'large'  : dModel=768,  numLayers=12
     * - 'custom' : use modelConfig directly
     */
    modelSize?      : 'nano' | 'small' | 'medium' | 'large' | 'custom';
    /** Fine-grained model configuration. Only used when modelSize is 'custom'. */
    modelConfig?    : Partial<MambaModelConfig>;
    /** WebGPU power preference. Default: 'high-performance'. */
    powerPreference?: 'high-performance' | 'low-power';
    /** Number of times to retry a failed checkpoint fetch. Default: 2. */
    fetchRetries?   : number;
}

export interface CompleteOptions {
    maxNewTokens? : number;   // Default: 200
    temperature?  : number;   // Default: 0.8
    topK?         : number;   // Default: 50
    topP?         : number;   // Default: 0.9
}

export interface AdaptOptions {
    epochs?       : number;   // Default: 3
    learningRate? : number;   // Default: 1e-4
    seqLen?       : number;   // Default: 512
    wsla?         : boolean;  // Default: true  (WSLA fast-adapt mode)
    fullTrain?    : boolean;  // Convenience alias: sets wsla=false and epochs=5
    onProgress?   : (epoch: number, loss: number) => void;
}

export interface AdaptResult {
    losses     : number[];
    epochCount : number;
    durationMs : number;
}

export type StorageTarget = 'indexedDB' | 'download' | 'fileSystem';

export interface SaveOptions {
    storage?  : StorageTarget;  // Default: 'indexedDB'
    filename? : string;         // Used by 'download' and 'fileSystem'. Default: '<name>.bin'
    key?      : string;         // IndexedDB key override. Default: session name
}

export interface LoadOptions {
    storage?  : StorageTarget;  // Default: 'indexedDB'
    url?      : string;         // Used when storage is 'url'
    key?      : string;         // IndexedDB key override. Default: session name
}

export type CreateStage = 'gpu' | 'tokenizer' | 'model' | 'weights';

export interface CreateProgressEvent {
    stage    : CreateStage;
    progress : number;   // 0.0 – 1.0 within the current stage
    message  : string;
}

export interface SessionInternals {
    device    : GPUDevice;
    model     : MambaModel;
    trainer   : MambaTrainer;
    tokenizer : BPETokenizer;
}

export interface CreateCallbacks {
    onProgress?: (event: CreateProgressEvent) => void;
}

/** Base delay (ms) for the first checkpoint fetch retry. Subsequent retries double this. */
const RETRY_BASE_DELAY_MS  = 500;
/** Multiplier applied to delay on each successive retry. */
const RETRY_BACKOFF_FACTOR = 2;

// ── MambaSession ───────────────────────────────────────────────────────────────

export class MambaSession {
    private _device    : GPUDevice;
    private _tokenizer : BPETokenizer;
    private _model     : MambaModel;
    private _trainer   : MambaTrainer;
    private _name      : string;
    private _destroyed = false;

    private constructor(
        device    : GPUDevice,
        tokenizer : BPETokenizer,
        model     : MambaModel,
        trainer   : MambaTrainer,
        name      : string,
    ) {
        this._device    = device;
        this._tokenizer = tokenizer;
        this._model     = model;
        this._trainer   = trainer;
        this._name      = name;
    }

    // ── Static factory ─────────────────────────────────────────────────────────

    static async create(
        options  : MambaSessionOptions,
        callbacks: CreateCallbacks = {},
    ): Promise<MambaSession> {
        const { onProgress } = callbacks;
        const name        = options.name ?? 'default';
        const fetchRetries = options.fetchRetries ?? 2;

        const emit = (stage: CreateStage, progress: number, message: string) => {
            onProgress?.({ stage, progress, message });
        };

        // Step 1 — GPU
        emit('gpu', 0.0, 'Initialising WebGPU…');
        let device: GPUDevice;
        try {
            const result = await initWebGPU({
                powerPreference: options.powerPreference ?? 'high-performance',
            });
            device = result.device;
        } catch (err) {
            throw new MambaKitError(
                'GPU_UNAVAILABLE',
                `WebGPU initialisation failed: ${(err as Error).message}`,
                err,
            );
        }
        emit('gpu', 1.0, 'WebGPU ready');

        // Step 2 — Tokenizer
        emit('tokenizer', 0.0, 'Loading tokenizer…');
        const tokenizer = new BPETokenizer();
        try {
            if (options.vocabObject != null && options.mergesArray != null) {
                tokenizer.loadFromObjects(options.vocabObject, options.mergesArray);
            } else if (options.vocabUrl != null && options.mergesUrl != null) {
                await tokenizer.load(options.vocabUrl, options.mergesUrl);
            } else if (options.vocabUrl != null) {
                // Support passing vocab only; merges can be empty
                await tokenizer.load(options.vocabUrl, []);
            } else {
                // No vocab provided — use a minimal fallback so the session
                // is still usable for non-text workflows
                tokenizer.loadFromObjects({}, []);
            }
        } catch (err) {
            throw new MambaKitError(
                'TOKENIZER_LOAD_FAILED',
                `Tokenizer failed to load: ${(err as Error).message}`,
                err,
            );
        }
        emit('tokenizer', 1.0, 'Tokenizer ready');

        // Step 3 — Model & Trainer
        emit('model', 0.0, 'Building model…');
        const vocabSize = tokenizer.vocabSize > 0 ? tokenizer.vocabSize : 1;
        const config    = resolveModelConfig(options, vocabSize);
        const model     = new MambaModel(device, config);
        const trainer   = new MambaTrainer(model, tokenizer);
        emit('model', 1.0, 'Model ready');

        // Step 4 — Checkpoint (optional)
        if (options.checkpointUrl != null) {
            emit('weights', 0.0, 'Fetching checkpoint…');
            let buffer: ArrayBuffer | null = null;
            let lastErr: unknown;

            for (let attempt = 0; attempt <= fetchRetries; attempt++) {
                try {
                    const res = await fetch(options.checkpointUrl);
                    if (!res.ok) {
                        throw new Error(`HTTP ${res.status} ${res.statusText}`);
                    }
                    buffer = await res.arrayBuffer();
                    break;
                } catch (err) {
                    lastErr = err;
                    if (attempt < fetchRetries) {
                        await sleep(RETRY_BASE_DELAY_MS * Math.pow(RETRY_BACKOFF_FACTOR, attempt));
                    }
                }
            }

            if (buffer == null) {
                throw new MambaKitError(
                    'CHECKPOINT_FETCH_FAILED',
                    `Failed to fetch checkpoint from "${options.checkpointUrl}" after ${fetchRetries + 1} attempt(s): ${(lastErr as Error).message}`,
                    lastErr,
                );
            }

            try {
                await model.loadWeights(buffer);
            } catch (err) {
                throw new MambaKitError(
                    'CHECKPOINT_INVALID',
                    `Checkpoint file is invalid or incompatible: ${(err as Error).message}`,
                    err,
                );
            }
            emit('weights', 1.0, 'Checkpoint loaded');
        }

        return new MambaSession(device, tokenizer, model, trainer, name);
    }

    // ── Text generation ────────────────────────────────────────────────────────

    async complete(prompt: string, options: CompleteOptions = {}): Promise<string> {
        this._assertNotDestroyed();

        const {
            maxNewTokens = 200,
            temperature  = 0.8,
            topK         = 50,
            topP         = 0.9,
        } = options;

        const promptIds = this._tokenizer.encode(prompt);
        const outputIds = await this._model.generate(promptIds, maxNewTokens, {
            temperature,
            topK,
            topP,
        });

        // Return the continuation only (not the original prompt tokens)
        const continuationIds = outputIds.slice(promptIds.length);
        return this._tokenizer.decode(continuationIds);
    }

    async *completeStream(
        prompt : string,
        options: CompleteOptions = {},
    ): AsyncIterable<string> {
        this._assertNotDestroyed();

        const {
            maxNewTokens = 200,
            temperature  = 0.8,
            topK         = 50,
            topP         = 0.9,
        } = options;

        const promptIds = this._tokenizer.encode(prompt);

        for await (const tokenId of tokenStream(this._model, promptIds, maxNewTokens, {
            temperature,
            topK,
            topP,
        })) {
            yield this._tokenizer.decode([tokenId]);
        }
    }

    // ── Fine-tuning ────────────────────────────────────────────────────────────

    async adapt(text: string, options: AdaptOptions = {}): Promise<AdaptResult> {
        this._assertNotDestroyed();

        let {
            epochs       = 3,
            wsla         = true,
        } = options;
        const {
            learningRate = 1e-4,
            seqLen       = 512,
            fullTrain    = false,
            onProgress,
        } = options;

        // Convenience alias: fullTrain overrides wsla and epoch defaults
        if (fullTrain) {
            wsla   = false;
            epochs = options.epochs ?? 5;
        }

        const encoded = this._tokenizer.encode(text);
        if (encoded.length < 2) {
            throw new MambaKitError(
                'INPUT_TOO_SHORT',
                'The input text encodes to fewer than 2 tokens and cannot be used for training.',
            );
        }

        const startTime = Date.now();
        const losses    = await this._trainer.train(text, {
            epochs,
            learningRate,
            seqLen,
            wsla,
            onEpochEnd: onProgress ?? null,
        });

        return {
            losses,
            epochCount : losses.length,
            durationMs : Date.now() - startTime,
        };
    }

    // ── Evaluation ─────────────────────────────────────────────────────────────

    async evaluate(text: string): Promise<number> {
        this._assertNotDestroyed();
        return this._trainer.evaluate(text);
    }

    // ── Persistence ────────────────────────────────────────────────────────────

    async save(options: SaveOptions = {}): Promise<void> {
        this._assertNotDestroyed();

        const storage  = options.storage  ?? 'indexedDB';
        const key      = options.key      ?? this._name;
        const filename = options.filename ?? `${this._name}.bin`;

        const buffer = await this._model.exportWeights();

        switch (storage) {
            case 'indexedDB':
                await saveToIndexedDB(key, buffer);
                break;
            case 'download':
                await triggerDownload(filename, buffer);
                break;
            case 'fileSystem':
                await saveViaFileSystemAPI(filename, buffer);
                break;
            default:
                throw new MambaKitError('STORAGE_UNAVAILABLE', `Unknown storage target: "${storage as string}"`);
        }
    }

    async load(options: LoadOptions = {}): Promise<boolean> {
        this._assertNotDestroyed();

        const storage = options.storage ?? 'indexedDB';
        const key     = options.key     ?? this._name;

        let buffer: ArrayBuffer | undefined;

        switch (storage) {
            case 'indexedDB': {
                buffer = await loadFromIndexedDB(key);
                break;
            }
            case 'fileSystem': {
                buffer = await loadViaFileSystemAPI();
                break;
            }
            default: {
                // Treat any other string as a URL fetch (covers custom `url` option)
                const url = options.url;
                if (!url) {
                    throw new MambaKitError(
                        'STORAGE_UNAVAILABLE',
                        'load() with storage other than "indexedDB" or "fileSystem" requires a url option.',
                    );
                }
                const res = await fetch(url);
                if (!res.ok) {
                    throw new MambaKitError(
                        'CHECKPOINT_FETCH_FAILED',
                        `Failed to fetch checkpoint from "${url}": HTTP ${res.status}`,
                    );
                }
                buffer = await res.arrayBuffer();
            }
        }

        if (buffer == null) return false;

        try {
            await this._model.loadWeights(buffer);
        } catch (err) {
            throw new MambaKitError(
                'CHECKPOINT_INVALID',
                `Saved checkpoint is invalid or incompatible: ${(err as Error).message}`,
                err,
            );
        }

        return true;
    }

    // ── Resource cleanup ───────────────────────────────────────────────────────

    destroy(): void {
        if (this._destroyed) return;
        this._destroyed = true;
        this._device.destroy();
    }

    // ── Escape hatch ───────────────────────────────────────────────────────────

    get internals(): SessionInternals {
        return {
            device    : this._device,
            model     : this._model,
            trainer   : this._trainer,
            tokenizer : this._tokenizer,
        };
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private _assertNotDestroyed(): void {
        if (this._destroyed) {
            throw new MambaKitError(
                'SESSION_DESTROYED',
                'This MambaSession has been destroyed. Create a new session with MambaSession.create().',
            );
        }
    }
}

function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}
