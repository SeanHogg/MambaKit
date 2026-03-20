# MambaKit

> **Opinionated facade over [MambaCode.js](https://www.npmjs.com/package/mambacode.js)** — on-device AI in a single import.

[![npm](https://img.shields.io/npm/v/mambakit)](https://www.npmjs.com/package/mambakit)
[![CI](https://github.com/SeanHogg/MambaKit/actions/workflows/ci.yml/badge.svg)](https://github.com/SeanHogg/MambaKit/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

MambaKit collapses the 8-step MambaCode.js setup into **a single `MambaSession.create()` call**.  
All GPU management, tokenisation, model construction, and persistence are handled for you.

---

## Installation

```bash
npm install mambakit
```

---

## Quick Start

```ts
import { MambaSession } from 'mambakit';

// 1 — Create a session (handles GPU init, tokenizer, model, checkpoint)
const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-coder-base.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});

// 2 — Generate a code completion
const completion = await session.complete('function fibonacci(n: number)');
console.log(completion);

// 3 — Fine-tune on private code (WSLA fast-adapt by default)
await session.adapt(myPrivateCodebase);

// 4 — Persist weights to IndexedDB
await session.save();

// 5 — Clean up GPU resources
session.destroy();
```

---

## Key API

| Method | Description |
|---|---|
| `MambaSession.create(options)` | Static async factory — initialises GPU, tokenizer, model, and checkpoint |
| `session.complete(prompt, opts?)` | Returns a generated text continuation as a plain string |
| `session.completeStream(prompt, opts?)` | `AsyncIterable<string>` — yields one token at a time for streaming UIs |
| `session.adapt(text, opts?)` | Fine-tunes on text (WSLA by default); returns `AdaptResult` |
| `session.evaluate(text)` | Returns perplexity (lower = better) |
| `session.save(opts?)` | Persists weights to IndexedDB (default), download, or File System API |
| `session.load(opts?)` | Restores weights; returns `false` if no checkpoint found |
| `session.destroy()` | Releases GPU device and resources |
| `session.internals` | Escape hatch — exposes underlying `MambaModel`, `MambaTrainer`, `BPETokenizer`, `GPUDevice` |

### Model size presets

| Preset | `dModel` | `numLayers` | Approx params |
|---|---|---|---|
| `nano` | 128 | 4 | ~6 M |
| `small` | 256 | 6 | ~20 M |
| **`medium`** *(default)* | 512 | 8 | ~50 M |
| `large` | 768 | 12 | ~120 M |

---

## TypeScript

Full declaration files (`.d.ts`) are published alongside the compiled output:

```ts
import {
  MambaSession,
  MambaKitError,
  type MambaSessionOptions,
  type CompleteOptions,
  type AdaptOptions,
  type AdaptResult,
} from 'mambakit';
```

---

## License

MIT
