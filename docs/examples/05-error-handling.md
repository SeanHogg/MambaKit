# Error Handling

All errors thrown by MambaKit are instances of `MambaKitError`. Each error carries a machine-readable `code` property so you can branch on specific failure modes.

## Basic pattern

```ts
import { MambaSession, MambaKitError } from 'mambakit';

try {
  const session = await MambaSession.create({
    checkpointUrl : '/models/checkpoint.bin',
    vocabUrl      : '/vocab.json',
    mergesUrl     : '/merges.txt',
  });
  const result = await session.complete('hello');
  session.destroy();
} catch (err) {
  if (err instanceof MambaKitError) {
    console.error(`[${err.code}] ${err.message}`);
    // optionally inspect err.cause for the original underlying error
  } else {
    throw err;   // re-throw unexpected errors
  }
}
```

## Handling specific error codes

```ts
import { MambaKitError, type MambaKitErrorCode } from 'mambakit';

function handleKitError(err: MambaKitError): void {
  switch (err.code) {
    case 'GPU_UNAVAILABLE':
      showMessage('WebGPU is not supported in this browser. Please try Chrome 113+.');
      break;

    case 'TOKENIZER_LOAD_FAILED':
      showMessage('Could not load the tokenizer files. Check your network connection.');
      break;

    case 'CHECKPOINT_FETCH_FAILED':
      showMessage('Could not download the model checkpoint. Check your network connection.');
      break;

    case 'CHECKPOINT_INVALID':
      showMessage('The model file appears to be corrupt or from an incompatible version.');
      break;

    case 'INPUT_TOO_SHORT':
      showMessage('Please provide more text for fine-tuning (at least 2 tokens).');
      break;

    case 'STORAGE_UNAVAILABLE':
      showMessage('Local storage is not available. Try enabling cookies / site data.');
      break;

    case 'SESSION_DESTROYED':
      console.warn('Attempted to use a destroyed session — create a new one.');
      break;

    default:
      console.error('Unexpected error:', err);
  }
}
```

## GPU not available

```ts
try {
  const session = await MambaSession.create({ /* ... */ });
} catch (err) {
  if (err instanceof MambaKitError && err.code === 'GPU_UNAVAILABLE') {
    // Show a browser-compatibility message
  }
}
```

## Checkpoint load failure with retries

`create()` automatically retries a failed checkpoint fetch (default: 2 retries with exponential backoff). If all retries fail, it throws `CHECKPOINT_FETCH_FAILED`:

```ts
try {
  const session = await MambaSession.create({
    checkpointUrl : 'https://cdn.example.com/model.bin',
    fetchRetries  : 3,   // try up to 4 times total (1 initial + 3 retries)
    vocabObject   : vocab,
    mergesArray   : merges,
  });
} catch (err) {
  if (err instanceof MambaKitError && err.code === 'CHECKPOINT_FETCH_FAILED') {
    console.error('Model download failed after all retries:', err.message);
  }
}
```

## Input too short

`adapt()` requires text that encodes to at least 2 tokens:

```ts
try {
  await session.adapt('hi');
} catch (err) {
  if (err instanceof MambaKitError && err.code === 'INPUT_TOO_SHORT') {
    console.warn('Training text is too short — provide a longer sample.');
  }
}
```

## Using a destroyed session

```ts
session.destroy();

try {
  await session.complete('hello');
} catch (err) {
  if (err instanceof MambaKitError && err.code === 'SESSION_DESTROYED') {
    // Create a new session if needed
    const fresh = await MambaSession.create({ /* ... */ });
  }
}
```

## Error code reference

| Code | Thrown by | Cause |
|------|-----------|-------|
| `GPU_UNAVAILABLE` | `create()` | WebGPU adapter unavailable |
| `TOKENIZER_LOAD_FAILED` | `create()` | Vocab/merges load failed |
| `CHECKPOINT_FETCH_FAILED` | `create()` | Checkpoint URL non-OK after retries |
| `CHECKPOINT_INVALID` | `create()`, `load()` | Weight file corrupt or incompatible |
| `INPUT_TOO_SHORT` | `adapt()` | Input encodes to fewer than 2 tokens |
| `STORAGE_UNAVAILABLE` | `save()`, `load()` | IndexedDB or File System API unavailable |
| `SESSION_DESTROYED` | all methods | Called after `destroy()` |
| `UNKNOWN` | any | Unexpected underlying error |
