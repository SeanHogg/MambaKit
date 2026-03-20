# Chatbot

`MambaChatbot` is a stateful multi-turn conversation wrapper that formats a growing message history into the text prompt the model needs, then returns only the new assistant reply.

## Setup

```ts
import { MambaSession }  from 'mambakit';
import { MambaChatbot }  from 'mambakit/examples/chatbot';

const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-chat.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});

const chatbot = new MambaChatbot(session, 'You are a helpful coding assistant.');
```

## Single-turn chat

```ts
const reply = await chatbot.chat('What does the ?? operator do in TypeScript?');
console.log(reply);
// → "The ?? (nullish coalescing) operator returns its right-hand operand
//    when the left-hand operand is null or undefined, and the left otherwise."
```

## Multi-turn conversation

Every call to `chat()` appends to the internal history, so the model sees the full conversation context:

```ts
const chatbot = new MambaChatbot(session);

const r1 = await chatbot.chat('What is a closure in JavaScript?');
console.log(r1);
// → "A closure is a function that retains access to variables…"

const r2 = await chatbot.chat('Can you give me a short example?');
console.log(r2);
// → "Sure: const counter = () => { let n = 0; return () => ++n; }; …"

console.log(chatbot.turnCount);  // 2
console.log(chatbot.history);
// [
//   { role: 'user',      content: 'What is a closure in JavaScript?' },
//   { role: 'assistant', content: 'A closure is a function…' },
//   { role: 'user',      content: 'Can you give me a short example?' },
//   { role: 'assistant', content: 'Sure: const counter = …' },
// ]
```

## Streaming chat

Use `chatStream()` to display tokens as they arrive:

```ts
process.stdout.write('Assistant: ');
for await (const chunk of chatbot.chatStream('Explain async/await in one sentence')) {
  process.stdout.write(chunk);
}
console.log();
```

## Custom generation options

```ts
const reply = await chatbot.chat('Refactor this for me', {
  maxNewTokens : 400,
  temperature  : 0.5,   // more focused / deterministic
  topP         : 0.85,
});
```

## Per-turn system prompt

Override the system prompt for a single turn without changing the default:

```ts
const reply = await chatbot.chat('Summarise this code', {
  systemPrompt : 'You are a senior TypeScript code reviewer. Be concise.',
});
```

## Resetting the conversation

```ts
chatbot.clearHistory();
console.log(chatbot.turnCount);  // 0
```

## Prompt format

The prompt fed to the model looks like this:

```
System: You are a helpful coding assistant.
User: What is a closure in JavaScript?
Assistant: A closure is a function that…
User: Can you give me a short example?
Assistant:
```

The model continues from `Assistant:`, and the generated text up to the first `User:` marker is returned as the reply.

## Full example: browser chat widget

```ts
import { MambaSession } from 'mambakit';
import { MambaChatbot } from 'mambakit/examples/chatbot';

async function main() {
  const session = await MambaSession.create({ /* ... */ });
  const chatbot = new MambaChatbot(session);

  const input    = document.querySelector<HTMLInputElement>('#input')!;
  const messages = document.querySelector<HTMLDivElement>('#messages')!;
  const send     = document.querySelector<HTMLButtonElement>('#send')!;

  send.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    // Show user message
    messages.innerHTML += `<p><b>You:</b> ${text}</p>`;

    // Stream the assistant reply
    const replyEl = document.createElement('p');
    replyEl.innerHTML = '<b>Assistant:</b> ';
    messages.appendChild(replyEl);

    for await (const chunk of chatbot.chatStream(text)) {
      replyEl.textContent += chunk;
    }
  });
}

main().catch(console.error);
```
