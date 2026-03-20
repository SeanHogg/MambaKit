# Fine-tuning

`session.adapt()` fine-tunes the model on a string of text and returns an `AdaptResult` with per-epoch loss values.

## Default fine-tuning (WSLA fast-adapt)

The default mode uses **WSLA** (Weight-Space Learning Adaptation) — a fast-adapt algorithm that converges in fewer steps than standard SGD while staying on the device:

```ts
import { MambaSession } from 'mambakit';

const session = await MambaSession.create({ /* ... */ });

const result = await session.adapt(`
  export function add(a: number, b: number): number {
    return a + b;
  }
  export function subtract(a: number, b: number): number {
    return a - b;
  }
`);

console.log('epochs:   ', result.epochCount);
console.log('losses:   ', result.losses);
console.log('duration: ', result.durationMs, 'ms');
```

## Full training mode

Disable WSLA and run a standard multi-epoch training pass:

```ts
const result = await session.adapt(myCodebase, {
  fullTrain : true,   // sets wsla=false and epochs=5 automatically
});
```

## Custom hyperparameters

```ts
const result = await session.adapt(trainingText, {
  epochs       : 10,
  learningRate : 5e-5,   // smaller LR for fine-tuning a pre-trained checkpoint
  seqLen       : 256,    // reduce if VRAM is constrained
});
```

## Tracking progress per epoch

Pass an `onProgress` callback to receive live feedback during training:

```ts
const result = await session.adapt(trainingText, {
  epochs     : 5,
  onProgress : (epoch, loss) => {
    console.log(`Epoch ${epoch + 1}: loss = ${loss.toFixed(4)}`);
  },
});
// Epoch 1: loss = 1.4821
// Epoch 2: loss = 1.2034
// Epoch 3: loss = 0.9877
// Epoch 4: loss = 0.8213
// Epoch 5: loss = 0.7015
```

## Evaluating model quality

`session.evaluate()` returns the perplexity of the model on a piece of text.  
Lower perplexity means the model is more confident on that text:

```ts
const before = await session.evaluate(testText);
await session.adapt(trainingText);
const after = await session.evaluate(testText);

console.log(`Perplexity before: ${before.toFixed(2)}`);
console.log(`Perplexity after:  ${after.toFixed(2)}`);
// Perplexity before: 45.30
// Perplexity after:  12.07
```

## Adapt → evaluate → save workflow

```ts
const session = await MambaSession.create({ /* ... */ });

// Measure baseline perplexity
const baseline = await session.evaluate(holdoutText);

// Fine-tune
const result = await session.adapt(trainingCorpus, {
  epochs     : 3,
  onProgress : (epoch, loss) => console.log(epoch, loss),
});

// Only save if perplexity improved
const perplexity = await session.evaluate(holdoutText);
if (perplexity < baseline) {
  await session.save();
  console.log('Checkpoint saved — perplexity improved from', baseline, 'to', perplexity);
}

session.destroy();
```
