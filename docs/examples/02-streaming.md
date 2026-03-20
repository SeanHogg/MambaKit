# Streaming Completions

`session.completeStream()` returns an `AsyncIterable<string>` that yields one decoded token at a time. This lets you display text as it is generated rather than waiting for the full output.

## Basic streaming

```ts
import { MambaSession } from 'mambakit';

const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-coder-base.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});

let output = '';
for await (const chunk of session.completeStream('function greet(name: string)')) {
  output += chunk;
  process.stdout.write(chunk);   // or update a DOM element
}

session.destroy();
```

## Updating a DOM element in real time

```ts
const outputEl = document.getElementById('output')!;
outputEl.textContent = '';

for await (const chunk of session.completeStream(prompt, { maxNewTokens: 300 })) {
  outputEl.textContent += chunk;
}
```

## Streaming with React state

```tsx
const [output, setOutput] = useState('');

async function run() {
  setOutput('');
  for await (const chunk of session.completeStream(prompt)) {
    setOutput(prev => prev + chunk);
  }
}
```

## Controlling sampling during streaming

The same `CompleteOptions` that `complete()` accepts also apply to `completeStream()`:

```ts
for await (const chunk of session.completeStream('interface User {', {
  maxNewTokens : 100,
  temperature  : 0.4,   // lower temperature → more focused, deterministic output
  topK         : 30,
  topP         : 0.9,
})) {
  process.stdout.write(chunk);
}
```

## Stopping early

Break out of the `for await` loop at any time to stop generation:

```ts
let tokenCount = 0;
for await (const chunk of session.completeStream(prompt)) {
  process.stdout.write(chunk);
  if (++tokenCount >= 50) break;   // stop after 50 tokens regardless of maxNewTokens
}
```

## Collecting the full string

If you want streaming delivery but also need the complete result, accumulate chunks:

```ts
async function streamToString(
  session: MambaSession,
  prompt: string,
): Promise<string> {
  const chunks: string[] = [];
  for await (const chunk of session.completeStream(prompt)) {
    chunks.push(chunk);
  }
  return chunks.join('');
}
```
