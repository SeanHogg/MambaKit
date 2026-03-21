# MambaKit

> **Opinionated facade over [MambaCode.js](https://www.npmjs.com/package/mambacode.js)** — on-device AI in a single import.

[![npm](https://img.shields.io/npm/v/mambakit)](https://www.npmjs.com/package/mambakit)
[![CI](https://github.com/SeanHogg/MambaKit/actions/workflows/ci.yml/badge.svg)](https://github.com/SeanHogg/MambaKit/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

MambaKit collapses the multi-step MambaCode.js setup into **a single `MambaSession.create()` call**.
GPU management, tokenisation (Qwen2.5-Coder built in), model construction, Mamba variant selection, WSLA fine-tuning, and weight persistence are all handled for you.

> **Looking for model-building tools?** Pretraining, checkpoint generation, and HuggingFace conversion utilities live in [MambaCode.js `tools/`](https://github.com/SeanHogg/Mamba/tree/main/tools). MambaKit assumes you already have a checkpoint.

---

## What's New in v2.0.0

| Feature | Detail |
|---|---|
| **`mambaVersion`** | Set `'mamba1'` (default), `'mamba2'` (SSD), or `'mamba3'` (complex states) |
| **`layerSchedule`** | Per-layer type array, or `'jamba'` / `'zamba'` preset strings |
| **`nHeads`** | Head count for Mamba-2/3 and Attention layers |
| **MBJS v2** | Checkpoint format updated automatically; v1 files still load |

---

## Model Weights

MambaKit's architecture is incompatible with Transformer model weights (Qwen, LLaMA, etc.). The Qwen2.5-Coder **tokenizer vocabulary** is built in — only SSM-architecture weights are needed.

To obtain weights:

| Path | How |
|---|---|
| **Default checkpoint** | Once hosted, `MambaSession.create({})` downloads it automatically. See `DEFAULT_CHECKPOINT_URL` in `session.ts`. |
| **Pretrain** | Use `tools/pretrain.html` in [MambaCode.js](https://github.com/SeanHogg/Mamba) to train on TinyStories or your own corpus. |
| **Convert** | Use `tools/convert.html` in MambaCode.js to convert `state-spaces/mamba` HuggingFace checkpoints. |
| **Generate blank** | Use `node tools/generate-bin.js` in MambaCode.js to create a randomly-initialised MBJS file. |

---

## Getting Started

### Prerequisites

- Node.js 18+ (for the dev server)
- A browser with WebGPU support: Chrome 113+, Edge 113+

### Run the examples

```bash
npm install
npm run build
npm run serve
```

Open the example links printed in the terminal in a WebGPU-capable browser.

---

## Quick Start

```ts
import { MambaSession } from 'mambakit';

// All initialisation handled automatically
const session = await MambaSession.create({
  modelSize : 'nano',          // 'nano' | 'small' | 'medium' | 'large' | 'custom'
  mambaVersion: 'mamba1',     // default — use 'mamba2' or 'mamba3' for newer variants
});

// Fine-tune on your content (runs in the browser, no data leaves)
await session.adapt(myDocumentText);

// Generate a completion
const answer = await session.complete('What does MambaKit do?');

// Stream tokens in real time
for await (const token of session.completeStream('function add(a, b)')) {
  outputEl.innerText += token;
}

// Save to IndexedDB
await session.save();

// Load back in a future session
await session.load();

session.destroy();
```

---

## Choosing a Mamba Variant

```ts
// Mamba-1 — original S6, maximum checkpoint compatibility (default)
await MambaSession.create({ mambaVersion: 'mamba1' });

// Mamba-2 — SSD chunked scan, better GPU throughput
await MambaSession.create({ mambaVersion: 'mamba2' });

// Mamba-3 — complex states + ET discretisation, 2× smaller state, faster inference
await MambaSession.create({ mambaVersion: 'mamba3' });

// Jamba-style hybrid: every 4th layer is attention, rest Mamba-2
await MambaSession.create({ layerSchedule: 'jamba', modelSize: 'small' });

// Zamba-style hybrid: every 6th layer is attention, rest Mamba-3
await MambaSession.create({ layerSchedule: 'zamba', modelSize: 'medium' });

// Fully custom per-layer schedule
await MambaSession.create({
  modelSize : 'custom',
  modelConfig: { dModel: 256, numLayers: 6, nHeads: 8 },
  layerSchedule: [
    { type: 'mamba3' },
    { type: 'mamba3' },
    { type: 'attention', config: { hasFfn: true } },
    { type: 'mamba3' },
    { type: 'mamba3' },
    { type: 'attention' },
  ],
});
```

---

## API Reference

### `MambaSession.create(options, callbacks?)`

Static async factory. Initialises WebGPU, loads the Qwen2.5-Coder tokenizer, builds the model, and optionally fetches a checkpoint.

```ts
const session = await MambaSession.create(
  {
    // Checkpoint (optional — random weights if omitted)
    checkpointUrl  : '/models/my-model.bin',
    fetchRetries   : 2,

    // Tokenizer (optional — Qwen2.5-Coder default)
    vocabUrl       : '/vocab.json',
    mergesUrl      : '/merges.txt',
    vocabObject    : { ... },   // alternative to URLs
    mergesArray    : [ ... ],

    // Model size preset (default: 'nano')
    modelSize      : 'nano',    // 'nano' | 'small' | 'medium' | 'large' | 'custom'
    modelConfig    : { ... },   // only used when modelSize is 'custom'

    // SSM variant (default: 'mamba1')
    mambaVersion   : 'mamba1',  // 'mamba1' | 'mamba2' | 'mamba3'

    // Per-layer schedule (overrides mambaVersion)
    layerSchedule  : 'jamba',   // 'jamba' | 'zamba' | LayerSpec[]

    // Session name (used as IndexedDB key)
    name           : 'my-session',

    powerPreference: 'high-performance',
  },
  {
    onProgress: ({ stage, progress, message }) => {
      console.log(`[${stage}] ${Math.round(progress * 100)}% — ${message}`);
    },
  },
);
```

**Progress stages:** `'gpu'` → `'tokenizer'` → `'model'` → `'weights'`

---

### `session.complete(prompt, options?)`

```ts
const text = await session.complete('function add(a, b)', {
  maxNewTokens : 200,   // default 200
  temperature  : 0.8,   // default 0.8
  topK         : 50,    // default 50
  topP         : 0.9,   // default 0.9
});
```

### `session.completeStream(prompt, options?)`

`AsyncIterable<string>` — yields one decoded token string at a time.

```ts
for await (const chunk of session.completeStream('const x =', { maxNewTokens: 64 })) {
  editor.append(chunk);
}
```

### `session.adapt(text, options?)`

Fine-tunes on `text` and returns an `AdaptResult`.

```ts
const result = await session.adapt(myCode, {
  wsla        : true,    // default true — WSLA fast-adapt
  epochs      : 3,
  learningRate: 1e-4,
  seqLen      : 512,
  fullTrain   : false,   // convenience: sets wsla=false, epochs=5
  onProgress  : (epoch, loss) => console.log(`epoch ${epoch}: loss=${loss.toFixed(4)}`),
});
// result: { losses: number[], epochCount: number, durationMs: number }
```

### `session.evaluate(text)`

Returns the model's perplexity on `text`. Lower is better.

### `session.save(options?)` / `session.load(options?)`

```ts
await session.save();                                       // IndexedDB (default)
await session.save({ storage: 'download', filename: 'my.bin' });
await session.save({ storage: 'fileSystem' });

const loaded = await session.load();                        // returns false if not found
await session.load({ url: '/models/checkpoint.bin' });
```

### `session.destroy()`

Releases the GPU device and all GPU buffers.

### `session.internals`

Escape hatch to the underlying MambaCode.js objects:

```ts
const { device, model, trainer, tokenizer } = session.internals;
// model is typed as HybridMambaModel — call model.layers, model.exportWeights(), etc.
```

---

## Model Size Presets

| Preset | `dModel` | `numLayers` | `nHeads` | Approx params |
|---|---|---|---|---|
| **`nano`** *(default)* | 128 | 4 | 4 | ~6 M |
| `small` | 256 | 6 | 8 | ~20 M |
| `medium` | 512 | 8 | 8 | ~50 M |
| `large` | 768 | 12 | 12 | ~120 M |
| `custom` | — | — | — | set via `modelConfig` |

`nHeads` is used by Mamba-2/3 and Attention layers. For `mamba1` it is ignored.

---

## Examples

The `src/examples/` directory contains higher-level classes that demonstrate real MambaKit usage patterns. These are the classes powering the HTML demos.

### `MambaChatbot` — [`src/examples/chatbot.ts`](src/examples/chatbot.ts)

```ts
import { MambaChatbot } from 'mambakit/examples/chatbot';

const bot = new MambaChatbot(session, 'You are a helpful coding assistant.');

const reply = await bot.chat('How do I reverse a string in JavaScript?');
for await (const token of bot.chatStream('Explain recursion briefly')) {
  outputEl.innerText += token;
}

bot.clearHistory();
console.log(bot.turnCount);
```

### `MambaCodeCompleter` — [`src/examples/code-completion.ts`](src/examples/code-completion.ts)

```ts
import { MambaCodeCompleter } from 'mambakit/examples/code-completion';

const completer = new MambaCodeCompleter(session);

const result = await completer.complete('function add(a: number, b:');
const line   = await completer.completeLine('const sum = a +');
for await (const token of completer.completeStream('class Foo {')) {
  editorEl.innerText += token;
}
```

### `MambaKnowledgeBase` — [`src/examples/knowledge-base.ts`](src/examples/knowledge-base.ts)

```ts
import { MambaKnowledgeBase } from 'mambakit/examples/knowledge-base';

const kb = new MambaKnowledgeBase(session);

const result = await kb.ingest({ id: 'readme', content: myDocsText });
console.log(`${result.perplexityBefore.toFixed(2)} → ${result.perplexityAfter.toFixed(2)}`);

await kb.ingestAll([
  { id: 'api',    content: apiDocs },
  { id: 'guides', content: guides  },
]);

const answer = await kb.query('What does MambaKit do?');
console.log(`${kb.documentCount} documents ingested`);
```

---

## How the Examples Work Together

```
┌────────────────────────────────────────────────────────┐
│  1. Knowledge Base  →  ingest  →  save to browser      │
│                                        │               │
│                         ┌─────────────┘                │
│                         ▼                              │
│  2. Chatbot         ←  load from browser               │
│  3. Code Completion ←  load from browser               │
└────────────────────────────────────────────────────────┘
```

**Start with Knowledge Base** — it works immediately with no configuration.

---

## File Structure

```
src/kit/
├── session.ts          ← MambaSession (main public API)
├── presets.ts          ← MODEL_PRESETS, resolveLayerSchedule, resolveModelConfig
├── errors.ts           ← MambaKitError with typed error codes
├── persistence.ts      ← IndexedDB, download, File System Access API
├── streaming.ts        ← AsyncIterable token stream adapter
└── index.ts            ← package entry point

src/examples/
├── chatbot.ts          ← MambaChatbot — multi-turn conversation
├── code-completion.ts  ← MambaCodeCompleter — editor-style completions
└── knowledge-base.ts   ← MambaKnowledgeBase — ingest + query

tests/
├── jest.setup.ts
├── kit.test.ts
└── examples.test.ts
```

---

## Error Handling

```ts
import { MambaKitError } from 'mambakit';

try {
  await MambaSession.create({ ... });
} catch (err) {
  if (err instanceof MambaKitError) {
    switch (err.code) {
      case 'GPU_UNAVAILABLE':
      case 'TOKENIZER_LOAD_FAILED':
      case 'CHECKPOINT_FETCH_FAILED':
      case 'CHECKPOINT_INVALID':
      case 'INPUT_TOO_SHORT':
      case 'STORAGE_UNAVAILABLE':
      case 'SESSION_DESTROYED':
    }
  }
}
```

---

## TypeScript

```ts
import {
  MambaSession,
  MambaKitError,
  type MambaSessionOptions,
  type CompleteOptions,
  type AdaptOptions,
  type AdaptResult,
  type LayerSchedulePreset,
} from 'mambakit';
```

---

## Professional Platform

**MambaKit powers the Hybrid Local Brain in [Builderforce.ai](https://builderforce.ai).**

Builderforce.ai is the professional enterprise platform that uses MambaKit as its in-browser AI engine:

- **Hybrid Local Brain** — `MambaSession` runs alongside Transformers.js in the Builderforce IDE; agent state persists to IndexedDB as a compact `MambaStateSnapshot` embedded in each exported `AgentPackage`
- **CPU fallback included** — `allowCpuFallback: true` enables the `forceFallbackAdapter` software path for non-WebGPU browsers; the `gpuMode` getter tells you which path is active
- **Managed training UI** — Builderforce provides a full LoRA training panel on top of MambaKit, with dataset generation, loss curve visualization, and AI evaluation scoring — no custom UI required
- **Workforce Registry** — publish your fine-tuned session as a specialist AI agent; others discover and hire it via skill tags and evaluation scores
- **Agent mesh** — hired agents install directly into [CoderClaw](https://coderclaw.ai) as self-hosted coding agents, orchestrated from Builderforce

| Layer | Role |
|---|---|
| **MambaCode.js** | WebGPU WGSL kernels (Mamba-1/2/3, SSD, attention) |
| **MambaKit** | Session abstraction (`MambaSession.create()`) |
| **Builderforce.ai** | Enterprise IDE + training UI + Workforce Registry |
| **CoderClaw** | Self-hosted agent runtime + claw-to-claw mesh |

Start with MambaKit for single-import simplicity. Upgrade to Builderforce.ai for the full enterprise experience — cloud storage, dataset management, agent publishing, and team orchestration.

---

## License

MIT
