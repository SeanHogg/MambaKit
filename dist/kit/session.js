/**
 * session.ts – MambaSession: the single entry point for all MambaKit functionality.
 *
 * MambaSession is a facade over MambaCode.js that collapses the 8-step
 * async setup sequence into a single `MambaSession.create()` call.
 */
import { initWebGPU, BPETokenizer, MambaModel, MambaTrainer, } from 'mambacode.js';
import { MambaKitError } from './errors.js';
import { resolveModelConfig } from './presets.js';
import { saveToIndexedDB, loadFromIndexedDB, triggerDownload, saveViaFileSystemAPI, loadViaFileSystemAPI, } from './persistence.js';
import { tokenStream } from './streaming.js';
// ── MambaSession ───────────────────────────────────────────────────────────────
export class MambaSession {
    _device;
    _tokenizer;
    _model;
    _trainer;
    _name;
    _destroyed = false;
    constructor(device, tokenizer, model, trainer, name) {
        this._device = device;
        this._tokenizer = tokenizer;
        this._model = model;
        this._trainer = trainer;
        this._name = name;
    }
    // ── Static factory ─────────────────────────────────────────────────────────
    static async create(options, callbacks = {}) {
        const { onProgress } = callbacks;
        const name = options.name ?? 'default';
        const fetchRetries = options.fetchRetries ?? 2;
        const emit = (stage, progress, message) => {
            onProgress?.({ stage, progress, message });
        };
        // Step 1 — GPU
        emit('gpu', 0.0, 'Initialising WebGPU…');
        let device;
        try {
            const result = await initWebGPU({
                powerPreference: options.powerPreference ?? 'high-performance',
            });
            device = result.device;
        }
        catch (err) {
            throw new MambaKitError('GPU_UNAVAILABLE', `WebGPU initialisation failed: ${err.message}`, err);
        }
        emit('gpu', 1.0, 'WebGPU ready');
        // Step 2 — Tokenizer
        emit('tokenizer', 0.0, 'Loading tokenizer…');
        const tokenizer = new BPETokenizer();
        try {
            if (options.vocabObject != null && options.mergesArray != null) {
                tokenizer.loadFromObjects(options.vocabObject, options.mergesArray);
            }
            else if (options.vocabUrl != null && options.mergesUrl != null) {
                await tokenizer.load(options.vocabUrl, options.mergesUrl);
            }
            else if (options.vocabUrl != null) {
                // Support passing vocab only; merges can be empty
                await tokenizer.load(options.vocabUrl, []);
            }
            else {
                // No vocab provided — use a minimal fallback so the session
                // is still usable for non-text workflows
                tokenizer.loadFromObjects({}, []);
            }
        }
        catch (err) {
            throw new MambaKitError('TOKENIZER_LOAD_FAILED', `Tokenizer failed to load: ${err.message}`, err);
        }
        emit('tokenizer', 1.0, 'Tokenizer ready');
        // Step 3 — Model & Trainer
        emit('model', 0.0, 'Building model…');
        const vocabSize = tokenizer.vocabSize > 0 ? tokenizer.vocabSize : 1;
        const config = resolveModelConfig(options, vocabSize);
        const model = new MambaModel(device, config);
        const trainer = new MambaTrainer(model, tokenizer);
        emit('model', 1.0, 'Model ready');
        // Step 4 — Checkpoint (optional)
        if (options.checkpointUrl != null) {
            emit('weights', 0.0, 'Fetching checkpoint…');
            let buffer = null;
            let lastErr;
            for (let attempt = 0; attempt <= fetchRetries; attempt++) {
                try {
                    const res = await fetch(options.checkpointUrl);
                    if (!res.ok) {
                        throw new Error(`HTTP ${res.status} ${res.statusText}`);
                    }
                    buffer = await res.arrayBuffer();
                    break;
                }
                catch (err) {
                    lastErr = err;
                    if (attempt < fetchRetries) {
                        await sleep(500 * Math.pow(2, attempt)); // 500ms, 1000ms
                    }
                }
            }
            if (buffer == null) {
                throw new MambaKitError('CHECKPOINT_FETCH_FAILED', `Failed to fetch checkpoint from "${options.checkpointUrl}" after ${fetchRetries + 1} attempt(s): ${lastErr.message}`, lastErr);
            }
            try {
                await model.loadWeights(buffer);
            }
            catch (err) {
                throw new MambaKitError('CHECKPOINT_INVALID', `Checkpoint file is invalid or incompatible: ${err.message}`, err);
            }
            emit('weights', 1.0, 'Checkpoint loaded');
        }
        return new MambaSession(device, tokenizer, model, trainer, name);
    }
    // ── Text generation ────────────────────────────────────────────────────────
    async complete(prompt, options = {}) {
        this._assertNotDestroyed();
        const { maxNewTokens = 200, temperature = 0.8, topK = 50, topP = 0.9, } = options;
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
    async *completeStream(prompt, options = {}) {
        this._assertNotDestroyed();
        const { maxNewTokens = 200, temperature = 0.8, topK = 50, topP = 0.9, } = options;
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
    async adapt(text, options = {}) {
        this._assertNotDestroyed();
        let { epochs = 3, learningRate = 1e-4, seqLen = 512, wsla = true, fullTrain = false, onProgress, } = options;
        // Convenience alias: fullTrain overrides wsla and epoch defaults
        if (fullTrain) {
            wsla = false;
            epochs = options.epochs ?? 5;
        }
        const encoded = this._tokenizer.encode(text);
        if (encoded.length < 2) {
            throw new MambaKitError('INPUT_TOO_SHORT', 'The input text encodes to fewer than 2 tokens and cannot be used for training.');
        }
        const startTime = Date.now();
        const losses = await this._trainer.train(text, {
            epochs,
            learningRate,
            seqLen,
            wsla,
            onEpochEnd: onProgress ?? null,
        });
        return {
            losses,
            epochCount: losses.length,
            durationMs: Date.now() - startTime,
        };
    }
    // ── Evaluation ─────────────────────────────────────────────────────────────
    async evaluate(text) {
        this._assertNotDestroyed();
        return this._trainer.evaluate(text);
    }
    // ── Persistence ────────────────────────────────────────────────────────────
    async save(options = {}) {
        this._assertNotDestroyed();
        const storage = options.storage ?? 'indexedDB';
        const key = options.key ?? this._name;
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
                throw new MambaKitError('STORAGE_UNAVAILABLE', `Unknown storage target: "${storage}"`);
        }
    }
    async load(options = {}) {
        this._assertNotDestroyed();
        const storage = options.storage ?? 'indexedDB';
        const key = options.key ?? this._name;
        let buffer;
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
                    throw new MambaKitError('STORAGE_UNAVAILABLE', 'load() with storage other than "indexedDB" or "fileSystem" requires a url option.');
                }
                const res = await fetch(url);
                if (!res.ok) {
                    throw new MambaKitError('CHECKPOINT_FETCH_FAILED', `Failed to fetch checkpoint from "${url}": HTTP ${res.status}`);
                }
                buffer = await res.arrayBuffer();
            }
        }
        if (buffer == null)
            return false;
        try {
            await this._model.loadWeights(buffer);
        }
        catch (err) {
            throw new MambaKitError('CHECKPOINT_INVALID', `Saved checkpoint is invalid or incompatible: ${err.message}`, err);
        }
        return true;
    }
    // ── Resource cleanup ───────────────────────────────────────────────────────
    destroy() {
        if (this._destroyed)
            return;
        this._destroyed = true;
        this._device.destroy();
    }
    // ── Escape hatch ───────────────────────────────────────────────────────────
    get internals() {
        return {
            device: this._device,
            model: this._model,
            trainer: this._trainer,
            tokenizer: this._tokenizer,
        };
    }
    // ── Private helpers ────────────────────────────────────────────────────────
    _assertNotDestroyed() {
        if (this._destroyed) {
            throw new MambaKitError('SESSION_DESTROYED', 'This MambaSession has been destroyed. Create a new session with MambaSession.create().');
        }
    }
}
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
//# sourceMappingURL=session.js.map