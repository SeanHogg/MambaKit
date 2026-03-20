# Code Completion

`MambaCodeCompleter` wraps `MambaSession` with code-oriented sampling defaults and a `completeLine()` method that clips the output at the first natural code boundary.

## Setup

```ts
import { MambaSession }      from 'mambakit';
import { MambaCodeCompleter } from 'mambakit/examples/code-completion';

const session   = await MambaSession.create({
  checkpointUrl : '/models/mamba-coder.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});

const completer = new MambaCodeCompleter(session);
```

## Full completion

`complete()` returns a `CompletionResult` with `prefix`, `completion`, and the concatenated `full` string:

```ts
const result = await completer.complete('function add(a: number, b: number)');
console.log(result.completion);
// → ": number {\n  return a + b;\n}"

console.log(result.full);
// → "function add(a: number, b: number): number {\n  return a + b;\n}"
```

## Single-line suggestion

`completeLine()` stops at the first `;`, `}`, or newline — perfect for inline suggestions in an editor:

```ts
const line = await completer.completeLine('const greeting = ');
console.log(line);
// → "const greeting = 'Hello, world!';"
```

## Controlling length and creativity

```ts
const result = await completer.complete('// Bubble sort\nfunction bubbleSort(arr: number[])', {
  maxNewTokens : 200,
  temperature  : 0.3,   // lower = more predictable code
});
```

## Streaming completion

Display code as it is generated — useful for large function bodies:

```ts
process.stdout.write(prefix);
for await (const chunk of completer.completeStream(prefix)) {
  process.stdout.write(chunk);
}
```

## Debounced editor integration

Debounce requests so the model is only queried when the user pauses typing:

```ts
import { MambaCodeCompleter } from 'mambakit/examples/code-completion';

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

function onEditorChange(currentLine: string): void {
  if (debounceTimer) clearTimeout(debounceTimer);

  debounceTimer = setTimeout(async () => {
    const suggestion = await completer.completeLine(currentLine);
    showGhostText(suggestion.slice(currentLine.length));
  }, 300);
}
```

## VS Code-style inline completion provider

```ts
import * as vscode from 'vscode';
import { MambaCodeCompleter } from 'mambakit/examples/code-completion';

export class MambaInlineProvider implements vscode.InlineCompletionItemProvider {
  constructor(private completer: MambaCodeCompleter) {}

  async provideInlineCompletionItems(
    document : vscode.TextDocument,
    position : vscode.Position,
  ): Promise<vscode.InlineCompletionList> {
    const prefix = document.getText(
      new vscode.Range(new vscode.Position(0, 0), position),
    );

    const result = await this.completer.complete(prefix, { maxNewTokens: 60 });

    return {
      items: [
        new vscode.InlineCompletionItem(
          result.completion,
          new vscode.Range(position, position),
        ),
      ],
    };
  }
}
```

## Defaults applied by `MambaCodeCompleter`

| Option | Default | Reason |
|--------|---------|--------|
| `maxNewTokens` | `128` | Code completions are usually short |
| `temperature` | `0.4` | Lower temperature → more deterministic, syntactically correct code |
| `topK` | `40` | Focused vocabulary typical for code tokens |
| `topP` | `0.85` | Nucleus filtering for code quality |

These can be overridden per call.
