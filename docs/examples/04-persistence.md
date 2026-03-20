# Persistence

MambaKit supports three storage backends for saving and restoring model weights.

## IndexedDB (default)

IndexedDB is the recommended backend for browser applications. Weights persist across page reloads and are stored locally on the user's device.

```ts
import { MambaSession } from 'mambakit';

const session = await MambaSession.create({ /* ... */ });

// Save (defaults to IndexedDB, key = session name)
await session.save();

// Later — restore into a fresh session
const session2 = await MambaSession.create({ /* ... */ });
const found = await session2.load();
console.log(found ? 'weights restored' : 'no saved weights found');
```

## Custom IndexedDB key

Override the default key (session name) with any string:

```ts
await session.save({ key: 'my-fine-tuned-v2' });

const found = await session2.load({ key: 'my-fine-tuned-v2' });
```

## Browser download

Trigger a file download in the browser so the user can save the checkpoint locally:

```ts
await session.save({
  storage  : 'download',
  filename : 'mamba-coder-finetuned.bin',
});
```

## File System Access API

Use the browser's native file picker for full control over where the file is stored:

```ts
// Save — opens a system "Save File" dialog
await session.save({
  storage  : 'fileSystem',
  filename : 'checkpoint.bin',
});

// Load — opens a system "Open File" dialog
const found = await session2.load({ storage: 'fileSystem' });
```

> **Note:** `showSaveFilePicker` and `showOpenFilePicker` are only available in Chromium-based browsers over HTTPS. Use `try/catch` with the `STORAGE_UNAVAILABLE` error code to handle gracefully.

## Checking whether a checkpoint exists

`load()` returns `false` (without throwing) when no checkpoint is found for the given key:

```ts
const found = await session.load({ key: 'my-model' });
if (!found) {
  console.log('No checkpoint — starting from scratch or loading from URL');
}
```

## Loading from a URL

Pass a URL directly to load weights without a file picker:

```ts
const found = await session.load({
  storage : 'url' as never,   // not a StorageTarget — treated as URL fetch
  url     : 'https://cdn.example.com/checkpoints/mamba-small.bin',
});
```

## Full save / load lifecycle

```ts
const session = await MambaSession.create({
  name          : 'code-assistant',
  checkpointUrl : '/models/base.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});

// Fine-tune on private code
await session.adapt(privateCodeCorpus);

// Persist the adapted weights
await session.save();   // stored in IndexedDB under key "code-assistant"
session.destroy();

// --- Next page load ---

const restored = await MambaSession.create({
  name     : 'code-assistant',
  vocabUrl : '/vocab.json',
  mergesUrl: '/merges.txt',
  // no checkpointUrl — load() will supply the weights
});

const found = await restored.load();
console.log(found ? 'Fine-tuned weights restored ✓' : 'Cold start');
```
