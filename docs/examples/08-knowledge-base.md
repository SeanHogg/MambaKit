# Knowledge Base

`MambaKnowledgeBase` fine-tunes MambaSession on a set of documents and lets you query the adapted model with natural-language questions. It measures perplexity before and after each ingestion so you can see concretely how much the model has learned.

## Setup

```ts
import { MambaSession }       from 'mambakit';
import { MambaKnowledgeBase } from 'mambakit/examples/knowledge-base';

const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-base.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
  name          : 'my-knowledge-base',
});

const kb = new MambaKnowledgeBase(session);
```

## Ingesting a single document

```ts
const result = await kb.ingest({
  id      : 'typescript-handbook',
  content : `
    TypeScript is a strongly typed programming language that builds on JavaScript.
    It adds optional static types, classes, and modules to JavaScript.
    The TypeScript compiler can catch errors at compile time rather than runtime.
  `,
});

console.log(`Perplexity before: ${result.perplexityBefore.toFixed(2)}`);
console.log(`Perplexity after:  ${result.perplexityAfter.toFixed(2)}`);
console.log(`Improved:          ${result.improved}`);
// Perplexity before: 48.30
// Perplexity after:  11.72
// Improved:          true
```

## Ingesting multiple documents

```ts
const docs = [
  { id: 'readme',    content: readFileSync('README.md', 'utf8')    },
  { id: 'api-docs',  content: readFileSync('API.md',    'utf8')    },
  { id: 'changelog', content: readFileSync('CHANGELOG.md', 'utf8') },
];

const results = await kb.ingestAll(docs);

for (const r of results) {
  console.log(`${r.id}: ${r.perplexityBefore.toFixed(1)} → ${r.perplexityAfter.toFixed(1)}`);
}
// readme:    52.1 → 9.8
// api-docs:  61.3 → 12.4
// changelog: 44.7 → 8.2
```

## Querying the knowledge base

After ingestion, use `query()` to ask the model questions about what it has learned:

```ts
const answer = await kb.query('What is TypeScript?');
console.log(answer);
// → "TypeScript is a strongly typed programming language that builds on
//    JavaScript, adding optional static types, classes, and modules…"
```

## Checking the knowledge base state

```ts
console.log(kb.documentCount);  // 3
console.log(kb.documentIds);    // ['readme', 'api-docs', 'changelog']
```

## Persisting the adapted weights

```ts
await kb.save();   // delegates to session.save() — stores in IndexedDB by default
```

## Custom ingestion options

Control fine-tuning hyperparameters per document:

```ts
const result = await kb.ingest(largeDocument, {
  epochs       : 5,
  learningRate : 5e-5,
  wsla         : false,   // full fine-tune instead of fast-adapt
  onProgress   : (epoch, loss) => console.log(`  epoch ${epoch + 1}: ${loss.toFixed(4)}`),
});
```

## Full example: documentation Q&A bot

```ts
import { readFileSync, readdirSync } from 'fs';
import { join }                      from 'path';
import { MambaSession }              from 'mambakit';
import { MambaKnowledgeBase }        from 'mambakit/examples/knowledge-base';
import { MambaChatbot }              from 'mambakit/examples/chatbot';

async function buildDocsBot(docsDir: string) {
  // 1. Create a base session
  const session = await MambaSession.create({
    checkpointUrl : '/models/mamba-base.bin',
    vocabUrl      : '/vocab.json',
    mergesUrl     : '/merges.txt',
    name          : 'docs-bot',
  });

  // 2. Try to restore a previously adapted checkpoint
  const kb     = new MambaKnowledgeBase(session);
  const loaded = await session.load();

  if (!loaded) {
    // 3. Ingest all markdown files from the docs directory
    const files = readdirSync(docsDir).filter(f => f.endsWith('.md'));
    const docs  = files.map(f => ({
      id      : f,
      content : readFileSync(join(docsDir, f), 'utf8'),
    }));

    console.log(`Ingesting ${docs.length} documents…`);
    const results = await kb.ingestAll(docs);
    console.log(`Done. Average perplexity reduction: ${averageReduction(results).toFixed(1)}%`);

    // 4. Save the adapted checkpoint for next time
    await kb.save();
  }

  // 5. Wrap in a chatbot for conversational Q&A
  const chatbot = new MambaChatbot(session,
    'You are a documentation assistant. Answer questions based on the provided docs.',
  );

  return chatbot;
}

function averageReduction(results: { perplexityBefore: number; perplexityAfter: number }[]) {
  const reductions = results.map(r => (r.perplexityBefore - r.perplexityAfter) / r.perplexityBefore * 100);
  return reductions.reduce((a, b) => a + b, 0) / reductions.length;
}

// Usage
const bot = await buildDocsBot('./docs');
const answer = await bot.chat('How do I stream completions?');
console.log(answer);
```
