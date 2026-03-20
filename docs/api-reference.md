# MambaKit API Reference

## Table of Contents

- [MambaSession](#mambasession)
  - [MambaSession.create()](#mambasessioncreateoptions-callbacks)
  - [session.complete()](#sessioncompleteprompt-options)
  - [session.completeStream()](#sessioncompletestreamprompt-options)
  - [session.adapt()](#sessionadapttext-options)
  - [session.evaluate()](#sessionevaluatetext)
  - [session.save()](#sessionsaveoptions)
  - [session.load()](#sessionloadoptions)
  - [session.destroy()](#sessiondestroy)
  - [session.internals](#sessioninternals)
- [MambaKitError](#mambakiterror)
- [MODEL\_PRESETS](#model_presets)
- [Type Definitions](#type-definitions)

---

## MambaSession

The single entry point for all MambaKit functionality. Wraps GPU initialisation, tokenisation, model construction, and weight management behind a clean async factory.

### `MambaSession.create(options, callbacks?)`

Static async factory method. Initialises the GPU, tokenizer, model, and (optionally) loads a checkpoint — all in a single call.

```ts
const session = await MambaSession.create(options, callbacks?);
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `options` | `MambaSessionOptions` | Session configuration (see below) |
| `callbacks` | `CreateCallbacks` | Optional lifecycle callbacks (e.g. `onProgress`) |

**`MambaSessionOptions`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpointUrl` | `string` | — | URL to a `.bin` checkpoint file. If omitted, the model starts with random weights. |
| `vocabUrl` | `string` | — | URL to `vocab.json` (Qwen3.5-Coder compatible BPE vocabulary). Required unless `vocabObject` is supplied. |
| `mergesUrl` | `string` | — | URL to `merges.txt`. Required unless `mergesArray` is supplied. |
| `vocabObject` | `Record<string, number>` | — | In-memory vocabulary — alternative to `vocabUrl`. |
| `mergesArray` | `string[]` | — | In-memory BPE merge rules — alternative to `mergesUrl`. |
| `name` | `string` | `'default'` | Unique session name used as the IndexedDB key. |
| `modelSize` | `'nano' \| 'small' \| 'medium' \| 'large' \| 'custom'` | `'medium'` | Model size preset. Use `'custom'` with `modelConfig` for fine-grained control. |
| `modelConfig` | `Partial<MambaModelConfig>` | — | Custom model config. Only applied when `modelSize === 'custom'`. |
| `powerPreference` | `'high-performance' \| 'low-power'` | `'high-performance'` | WebGPU power preference hint. |
| `fetchRetries` | `number` | `2` | Number of times to retry a failed checkpoint fetch. Uses exponential backoff starting at 500 ms. |

**`CreateCallbacks`**

| Field | Type | Description |
|-------|------|-------------|
| `onProgress` | `(event: CreateProgressEvent) => void` | Called at the start and end of each initialisation stage. |

**Returns** `Promise<MambaSession>`

**Throws** `MambaKitError` with codes:
- `GPU_UNAVAILABLE` — WebGPU adapter could not be acquired
- `TOKENIZER_LOAD_FAILED` — vocab or merges file could not be fetched or parsed
- `CHECKPOINT_FETCH_FAILED` — checkpoint URL returned a non-OK response after all retries
- `CHECKPOINT_INVALID` — checkpoint file is corrupt or incompatible with the current model config

---

### `session.complete(prompt, options?)`

Generates a text continuation for the given prompt and returns it as a plain string.

```ts
const result = await session.complete(prompt, options?);
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | — | The input text |
| `options.maxNewTokens` | `number` | `200` | Maximum number of tokens to generate |
| `options.temperature` | `number` | `0.8` | Sampling temperature. Higher = more random. |
| `options.topK` | `number` | `50` | Top-K sampling cutoff |
| `options.topP` | `number` | `0.9` | Nucleus (Top-P) sampling cutoff |

**Returns** `Promise<string>` — The generated continuation only (prompt tokens are excluded from the output).

**Throws** `MambaKitError` with code `SESSION_DESTROYED` if called after `destroy()`.

---

### `session.completeStream(prompt, options?)`

Streaming variant of `complete()`. Returns an `AsyncIterable<string>` that yields one decoded token string at a time.

```ts
for await (const chunk of session.completeStream(prompt, options?)) {
  process.stdout.write(chunk);
}
```

**Parameters** — same as `complete()`.

**Returns** `AsyncIterable<string>` — each iteration yields one token's decoded text.

**Throws** `MambaKitError` with code `SESSION_DESTROYED` if called after `destroy()`.

---

### `session.adapt(text, options?)`

Fine-tunes the model on the provided text using WSLA fast-adapt mode by default.

```ts
const result = await session.adapt(text, options?);
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string` | — | Training text (must encode to at least 2 tokens) |
| `options.epochs` | `number` | `3` | Number of training epochs |
| `options.learningRate` | `number` | `1e-4` | AdamW learning rate |
| `options.seqLen` | `number` | `512` | Training sequence length |
| `options.wsla` | `boolean` | `true` | Enable WSLA fast-adapt mode (recommended for quick personalisation) |
| `options.fullTrain` | `boolean` | `false` | Convenience alias: sets `wsla=false` and `epochs=5`. Overrides `wsla` and `epochs`. |
| `options.onProgress` | `(epoch: number, loss: number) => void` | — | Called after each epoch with the current epoch index and loss value |

**Returns** `Promise<AdaptResult>`

```ts
interface AdaptResult {
  losses     : number[];  // Per-epoch loss values
  epochCount : number;    // Total number of epochs run
  durationMs : number;    // Wall-clock training time in milliseconds
}
```

**Throws** `MambaKitError` with codes:
- `INPUT_TOO_SHORT` — input text encodes to fewer than 2 tokens
- `SESSION_DESTROYED` — called after `destroy()`

---

### `session.evaluate(text)`

Computes the model's perplexity on the given text. Lower values indicate the model is more confident on this input.

```ts
const perplexity = await session.evaluate(text);
```

**Returns** `Promise<number>` — perplexity score (lower is better).

**Throws** `MambaKitError` with code `SESSION_DESTROYED` if called after `destroy()`.

---

### `session.save(options?)`

Exports the current model weights and persists them to the chosen storage backend.

```ts
await session.save(options?);
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `options.storage` | `'indexedDB' \| 'download' \| 'fileSystem'` | `'indexedDB'` | Where to persist the weights |
| `options.key` | `string` | session name | IndexedDB key override |
| `options.filename` | `string` | `'<name>.bin'` | File name used for `'download'` and `'fileSystem'` targets |

**Throws** `MambaKitError` with codes:
- `STORAGE_UNAVAILABLE` — IndexedDB or File System Access API is not available
- `SESSION_DESTROYED` — called after `destroy()`

---

### `session.load(options?)`

Restores model weights from the chosen storage backend.

```ts
const found = await session.load(options?);
```

**Parameters** — same storage/key fields as `save()`, plus:

| Parameter | Type | Description |
|-----------|------|-------------|
| `options.url` | `string` | URL to fetch weights from (used when `storage` is not `'indexedDB'` or `'fileSystem'`) |

**Returns** `Promise<boolean>` — `true` if weights were found and loaded; `false` if the key was not found in IndexedDB (no error is thrown).

**Throws** `MambaKitError` with codes:
- `CHECKPOINT_INVALID` — saved file is corrupt or incompatible
- `STORAGE_UNAVAILABLE` — URL mode requires `options.url`
- `SESSION_DESTROYED` — called after `destroy()`

---

### `session.destroy()`

Releases the underlying `GPUDevice` and marks the session as destroyed. All subsequent method calls will throw `SESSION_DESTROYED`. Safe to call multiple times.

```ts
session.destroy();
```

---

### `session.internals`

Escape hatch that exposes the underlying MambaCode.js objects for advanced use cases.

```ts
const { device, model, trainer, tokenizer } = session.internals;
```

**Returns** `SessionInternals`

| Field | Type | Description |
|-------|------|-------------|
| `device` | `GPUDevice` | The WebGPU device |
| `model` | `MambaModel` | The raw MambaCode.js model |
| `trainer` | `MambaTrainer` | The raw MambaCode.js trainer |
| `tokenizer` | `BPETokenizer` | The raw BPE tokenizer |

---

## MambaKitError

All errors thrown by MambaKit are instances of `MambaKitError`, which extends the built-in `Error` class.

```ts
import { MambaKitError, type MambaKitErrorCode } from 'mambakit';
```

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `code` | `MambaKitErrorCode` | Machine-readable error code |
| `message` | `string` | Human-readable error description |
| `cause` | `unknown` | Original error that caused this one (if applicable) |
| `name` | `string` | Always `'MambaKitError'` |

**Error Codes**

| Code | When thrown |
|------|-------------|
| `GPU_UNAVAILABLE` | `navigator.gpu` is not present or the GPU adapter request failed |
| `TOKENIZER_LOAD_FAILED` | `vocab.json` or `merges.txt` could not be fetched or parsed |
| `CHECKPOINT_FETCH_FAILED` | Checkpoint URL returned a non-OK response after all retries |
| `CHECKPOINT_INVALID` | `loadWeights()` threw — bad magic bytes, version mismatch, or size mismatch |
| `INPUT_TOO_SHORT` | `adapt()` input encodes to fewer than 2 tokens |
| `STORAGE_UNAVAILABLE` | IndexedDB or File System Access API is not available |
| `SESSION_DESTROYED` | A method was called after `destroy()` |
| `UNKNOWN` | An unexpected error not covered by the above codes |

---

## MODEL_PRESETS

A read-only record of the four built-in model size presets.

```ts
import { MODEL_PRESETS } from 'mambakit';
```

| Preset | `dModel` | `numLayers` | `dState` | `dConv` | `expand` | Approx params |
|--------|----------|-------------|---------|---------|---------|---------------|
| `nano` | 128 | 4 | 16 | 4 | 2 | ~6 M |
| `small` | 256 | 6 | 16 | 4 | 2 | ~20 M |
| `medium` *(default)* | 512 | 8 | 16 | 4 | 2 | ~50 M |
| `large` | 768 | 12 | 16 | 4 | 2 | ~120 M |

---

## Type Definitions

```ts
// Session creation
interface MambaSessionOptions { ... }   // see MambaSession.create() above
interface CreateCallbacks      { onProgress?: (e: CreateProgressEvent) => void; }
interface CreateProgressEvent  { stage: CreateStage; progress: number; message: string; }
type CreateStage = 'gpu' | 'tokenizer' | 'model' | 'weights';

// Text generation
interface CompleteOptions {
  maxNewTokens? : number;   // Default: 200
  temperature?  : number;   // Default: 0.8
  topK?         : number;   // Default: 50
  topP?         : number;   // Default: 0.9
}

// Fine-tuning
interface AdaptOptions {
  epochs?       : number;                            // Default: 3
  learningRate? : number;                            // Default: 1e-4
  seqLen?       : number;                            // Default: 512
  wsla?         : boolean;                           // Default: true
  fullTrain?    : boolean;                           // Default: false
  onProgress?   : (epoch: number, loss: number) => void;
}
interface AdaptResult {
  losses     : number[];
  epochCount : number;
  durationMs : number;
}

// Persistence
type StorageTarget = 'indexedDB' | 'download' | 'fileSystem';
interface SaveOptions { storage?: StorageTarget; filename?: string; key?: string; }
interface LoadOptions { storage?: StorageTarget; url?: string; key?: string; }

// Internals
interface SessionInternals {
  device    : GPUDevice;
  model     : MambaModel;
  trainer   : MambaTrainer;
  tokenizer : BPETokenizer;
}
```
