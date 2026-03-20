# Quick Start

Get from zero to your first generated completion in five lines of application code.

## Prerequisites

- A browser or runtime with [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) support
- A served tokenizer (`vocab.json` + `merges.txt`) and, optionally, a trained checkpoint (`.bin`)

## Install

```bash
npm install mambakit
```

## Minimal example

```ts
import { MambaSession } from 'mambakit';

// 1. Create a session — handles GPU init, tokenizer, model, and checkpoint in one call
const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-coder-base.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});

// 2. Generate a completion
const completion = await session.complete('function fibonacci(n: number)');
console.log(completion);
// → ": number {\n  if (n <= 1) return n;\n  return fibonacci(n - 1) + fibonacci(n - 2);\n}"

// 3. Always release GPU resources when done
session.destroy();
```

## Tuning generation quality

Pass a `CompleteOptions` object to control sampling behaviour:

```ts
const completion = await session.complete('// Sort an array of numbers', {
  maxNewTokens : 150,    // stop after 150 new tokens
  temperature  : 0.6,    // lower = more deterministic
  topK         : 40,
  topP         : 0.85,
});
```

## Using in-memory tokenizer data

If you have the vocabulary and merge rules bundled in your application, skip the network round-trip:

```ts
import vocab  from './vocab.json'   assert { type: 'json' };
import merges from './merges.json'  assert { type: 'json' };  // array of "A B" strings

const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-coder-base.bin',
  vocabObject   : vocab,
  mergesArray   : merges,
});
```

## Starting with random weights (no checkpoint)

Useful for training from scratch or integration testing:

```ts
const session = await MambaSession.create({
  vocabObject : { hello: 0, world: 1 },
  mergesArray : [],
  modelSize   : 'nano',   // smallest preset — fast to initialise
});
```

## Named sessions

Give each session a unique name so that `save()` and `load()` use the right IndexedDB key:

```ts
const session = await MambaSession.create({
  vocabObject : vocab,
  mergesArray : merges,
  name        : 'my-project-model',
});

await session.save();               // stores under key "my-project-model"
await session.load();               // restores from key "my-project-model"
```

## Next steps

- [Streaming completions →](./02-streaming.md)
- [Fine-tuning your model →](./03-fine-tuning.md)
- [Saving and loading weights →](./04-persistence.md)
