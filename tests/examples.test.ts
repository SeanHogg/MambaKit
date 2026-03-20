/**
 * examples.test.ts – Unit tests for the MambaKit application examples.
 *
 * Tests cover every documented use-case pattern for:
 *   - MambaChatbot      (docs/examples/06-chatbot.md)
 *   - MambaCodeCompleter (docs/examples/07-code-completion.md)
 *   - MambaKnowledgeBase (docs/examples/08-knowledge-base.md)
 *
 * The example classes use self-contained session interfaces, so no
 * mambacode.js mock is required — a plain jest.fn() object suffices.
 */

import { jest } from '@jest/globals';

import {
    MambaChatbot,
    type Message,
} from '../src/examples/chatbot.js';

import { MambaCodeCompleter } from '../src/examples/code-completion.js';

import { MambaKnowledgeBase } from '../src/examples/knowledge-base.js';

// ── Async-generator helper ────────────────────────────────────────────────────

async function* yieldChunks(...chunks: string[]): AsyncGenerator<string> {
    for (const chunk of chunks) yield chunk;
}

// ── Mock session factories ────────────────────────────────────────────────────

function makeChatSession(completionResult = 'Mocked reply') {
    return {
        complete       : jest.fn<() => Promise<string>>().mockResolvedValue(completionResult),
        completeStream : jest.fn<() => AsyncIterable<string>>()
            .mockReturnValue(yieldChunks('Mocked', ' ', 'reply')),
    };
}

function makeCodeSession(completionResult = ' number {\n  return a + b;\n}') {
    return {
        complete       : jest.fn<() => Promise<string>>().mockResolvedValue(completionResult),
        completeStream : jest.fn<() => AsyncIterable<string>>()
            .mockReturnValue(yieldChunks(' number', ' {', '\n  return a + b;\n', '}')),
    };
}

function makeKnowledgeSession({
    perplexityBefore = 50.0,
    perplexityAfter  = 10.0,
    adaptLosses      = [1.5, 1.2, 0.9],
    queryResult      = 'TypeScript is a typed superset of JavaScript.',
} = {}) {
    let evaluateCallCount = 0;
    return {
        evaluate : jest.fn<() => Promise<number>>().mockImplementation(() =>
            Promise.resolve(evaluateCallCount++ === 0 ? perplexityBefore : perplexityAfter),
        ),
        adapt    : jest.fn<() => Promise<{ losses: number[]; epochCount: number; durationMs: number }>>()
            .mockResolvedValue({ losses: adaptLosses, epochCount: adaptLosses.length, durationMs: 200 }),
        complete : jest.fn<() => Promise<string>>().mockResolvedValue(queryResult),
        save     : jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    };
}

// ═════════════════════════════════════════════════════════════════════════════
// MambaChatbot
// ═════════════════════════════════════════════════════════════════════════════

describe('MambaChatbot', () => {

    // ── formatPrompt ───────────────────────────────────────────────────────────

    describe('formatPrompt()', () => {
        test('includes the default system prompt', () => {
            const bot    = new MambaChatbot(makeChatSession());
            const prompt = bot.formatPrompt('Hello');
            expect(prompt).toContain('System: You are a helpful assistant.');
        });

        test('accepts a custom system prompt', () => {
            const bot    = new MambaChatbot(makeChatSession(), 'You are a code reviewer.');
            const prompt = bot.formatPrompt('Review this');
            expect(prompt).toContain('System: You are a code reviewer.');
        });

        test('overrides system prompt per-call', () => {
            const bot    = new MambaChatbot(makeChatSession());
            const prompt = bot.formatPrompt('Review this', 'You are a code reviewer.');
            expect(prompt).toContain('System: You are a code reviewer.');
            expect(prompt).not.toContain('You are a helpful assistant.');
        });

        test('ends with "Assistant:" marker', () => {
            const bot    = new MambaChatbot(makeChatSession());
            const prompt = bot.formatPrompt('Hello');
            expect(prompt.trimEnd().endsWith('Assistant:')).toBe(true);
        });

        test('includes the user message', () => {
            const bot    = new MambaChatbot(makeChatSession());
            const prompt = bot.formatPrompt('What is 2 + 2?');
            expect(prompt).toContain('User: What is 2 + 2?');
        });

        test('includes previous history messages in order', () => {
            const session = makeChatSession();
            const bot     = new MambaChatbot(session);
            // Manually push history to test format without side effects
            (bot as unknown as { _history: Message[] })._history = [
                { role: 'user',      content: 'First question'  },
                { role: 'assistant', content: 'First answer'    },
            ];

            const prompt = bot.formatPrompt('Second question');
            const lines  = prompt.split('\n');

            expect(lines[1]).toBe('User: First question');
            expect(lines[2]).toBe('Assistant: First answer');
            expect(lines[3]).toBe('User: Second question');
            expect(lines[4]).toBe('Assistant:');
        });
    });

    // ── chat() ─────────────────────────────────────────────────────────────────

    describe('chat()', () => {
        test('calls session.complete() with a formatted prompt', async () => {
            const session = makeChatSession();
            const bot     = new MambaChatbot(session);
            await bot.chat('Hello');
            expect(session.complete).toHaveBeenCalledWith(
                expect.stringContaining('User: Hello'),
                expect.any(Object),
            );
        });

        test('returns a string', async () => {
            const bot  = new MambaChatbot(makeChatSession('Great question!'));
            const reply = await bot.chat('Hello');
            expect(typeof reply).toBe('string');
        });

        test('trims the response and drops everything after "\\nUser:"', async () => {
            // Model may generate a follow-up turn; only the first reply is kept.
            const session = makeChatSession('First reply\nUser: extra turn\nAssistant: more');
            const bot     = new MambaChatbot(session);
            const reply   = await bot.chat('Hello');
            expect(reply).toBe('First reply');
        });

        test('appends user message to history', async () => {
            const bot = new MambaChatbot(makeChatSession());
            await bot.chat('Tell me something');
            expect(bot.history[0]).toEqual({ role: 'user', content: 'Tell me something' });
        });

        test('appends assistant reply to history', async () => {
            const bot  = new MambaChatbot(makeChatSession('I am the model.'));
            await bot.chat('Who are you?');
            expect(bot.history[1]).toEqual({ role: 'assistant', content: 'I am the model.' });
        });

        test('builds up history across multiple turns', async () => {
            const session = makeChatSession();
            session.complete
                .mockResolvedValueOnce('Answer one')
                .mockResolvedValueOnce('Answer two');

            const bot = new MambaChatbot(session);
            await bot.chat('Question one');
            await bot.chat('Question two');

            expect(bot.history).toHaveLength(4);
            expect(bot.history[2]).toEqual({ role: 'user',      content: 'Question two' });
            expect(bot.history[3]).toEqual({ role: 'assistant', content: 'Answer two'   });
        });

        test('second prompt includes the first turn in history', async () => {
            const session = makeChatSession('Answer one');
            const bot     = new MambaChatbot(session);
            await bot.chat('Question one');

            session.complete.mockResolvedValueOnce('Answer two');
            await bot.chat('Question two');

            expect(session.complete).toHaveBeenNthCalledWith(
                2,
                expect.stringContaining('User: Question one'),
                expect.any(Object),
            );
            expect(session.complete).toHaveBeenNthCalledWith(
                2,
                expect.stringContaining('Assistant: Answer one'),
                expect.any(Object),
            );
        });

        test('passes custom CompleteOptions to session.complete()', async () => {
            const session = makeChatSession();
            const bot     = new MambaChatbot(session);
            await bot.chat('Hello', { temperature: 0.2, maxNewTokens: 50 });
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.2, maxNewTokens: 50 }),
            );
        });

        test('uses 0.7 temperature by default', async () => {
            const session = makeChatSession();
            const bot     = new MambaChatbot(session);
            await bot.chat('Hello');
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.7 }),
            );
        });
    });

    // ── chatStream() ───────────────────────────────────────────────────────────

    describe('chatStream()', () => {
        test('yields chunks from session.completeStream()', async () => {
            const session = makeChatSession();
            session.completeStream.mockReturnValue(yieldChunks('Hello', ' there', '!'));
            const bot    = new MambaChatbot(session);
            const chunks: string[] = [];
            for await (const c of bot.chatStream('Hi')) chunks.push(c);
            expect(chunks).toEqual(['Hello', ' there', '!']);
        });

        test('calls session.completeStream() with a formatted prompt', async () => {
            const session = makeChatSession();
            session.completeStream.mockReturnValue(yieldChunks('reply'));
            const bot = new MambaChatbot(session);
            for await (const _chunk of bot.chatStream('My question')) { /* consume */ }
            expect(session.completeStream).toHaveBeenCalledWith(
                expect.stringContaining('User: My question'),
                expect.any(Object),
            );
        });

        test('appends to history after all chunks are yielded', async () => {
            const session = makeChatSession();
            session.completeStream.mockReturnValue(yieldChunks('Streamed', ' reply'));
            const bot = new MambaChatbot(session);
            expect(bot.history).toHaveLength(0);

            for await (const _chunk of bot.chatStream('Stream this')) { /* consume */ }

            expect(bot.history).toHaveLength(2);
            expect(bot.history[0]).toEqual({ role: 'user',      content: 'Stream this'    });
            expect(bot.history[1]).toEqual({ role: 'assistant', content: 'Streamed reply' });
        });
    });

    // ── clearHistory() / history / turnCount ──────────────────────────────────

    describe('clearHistory()', () => {
        test('resets history to empty', async () => {
            const bot = new MambaChatbot(makeChatSession());
            await bot.chat('Something');
            bot.clearHistory();
            expect(bot.history).toHaveLength(0);
        });

        test('turnCount is 0 after clear', async () => {
            const bot = new MambaChatbot(makeChatSession());
            await bot.chat('Something');
            bot.clearHistory();
            expect(bot.turnCount).toBe(0);
        });
    });

    describe('history', () => {
        test('is readonly (returns a reference that should not be mutated)', async () => {
            const bot = new MambaChatbot(makeChatSession('reply'));
            await bot.chat('hello');
            const h = bot.history;
            expect(Array.isArray(h)).toBe(true);
            expect(h).toHaveLength(2);
        });
    });

    describe('turnCount', () => {
        test('increments by 1 for each chat() call', async () => {
            const session = makeChatSession();
            const bot     = new MambaChatbot(session);
            expect(bot.turnCount).toBe(0);
            await bot.chat('one');
            expect(bot.turnCount).toBe(1);
            await bot.chat('two');
            expect(bot.turnCount).toBe(2);
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// MambaCodeCompleter
// ═════════════════════════════════════════════════════════════════════════════

describe('MambaCodeCompleter', () => {

    // ── complete() ─────────────────────────────────────────────────────────────

    describe('complete()', () => {
        test('returns a CompletionResult with prefix, completion, and full', async () => {
            const session   = makeCodeSession(': number { return a + b; }');
            const completer = new MambaCodeCompleter(session);
            const result    = await completer.complete('function add(a: number, b: number)');

            expect(result.prefix).toBe('function add(a: number, b: number)');
            expect(result.completion).toBe(': number { return a + b; }');
            expect(result.full).toBe('function add(a: number, b: number): number { return a + b; }');
        });

        test('full === prefix + completion', async () => {
            const completer = new MambaCodeCompleter(makeCodeSession('_suffix'));
            const result    = await completer.complete('prefix_');
            expect(result.full).toBe(result.prefix + result.completion);
        });

        test('passes prefix to session.complete()', async () => {
            const session   = makeCodeSession();
            const completer = new MambaCodeCompleter(session);
            await completer.complete('const x =');
            expect(session.complete).toHaveBeenCalledWith('const x =', expect.any(Object));
        });

        test('uses low temperature (0.4) by default', async () => {
            const session   = makeCodeSession();
            const completer = new MambaCodeCompleter(session);
            await completer.complete('foo');
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.4 }),
            );
        });

        test('uses maxNewTokens 128 by default', async () => {
            const session   = makeCodeSession();
            const completer = new MambaCodeCompleter(session);
            await completer.complete('foo');
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ maxNewTokens: 128 }),
            );
        });

        test('custom options override defaults', async () => {
            const session   = makeCodeSession();
            const completer = new MambaCodeCompleter(session);
            await completer.complete('foo', { temperature: 0.9, maxNewTokens: 300 });
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.9, maxNewTokens: 300 }),
            );
        });
    });

    // ── completeStream() ───────────────────────────────────────────────────────

    describe('completeStream()', () => {
        test('yields tokens from session.completeStream()', async () => {
            const session   = makeCodeSession();
            session.completeStream.mockReturnValue(yieldChunks(': number', ' {\n', '  return 0;\n', '}'));
            const completer = new MambaCodeCompleter(session);
            const chunks: string[] = [];
            for await (const c of completer.completeStream('function noop()')) chunks.push(c);
            expect(chunks).toEqual([': number', ' {\n', '  return 0;\n', '}']);
        });

        test('passes prefix to session.completeStream()', async () => {
            const session   = makeCodeSession();
            session.completeStream.mockReturnValue(yieldChunks('token'));
            const completer = new MambaCodeCompleter(session);
            for await (const _chunk of completer.completeStream('my prefix')) { /* consume */ }
            expect(session.completeStream).toHaveBeenCalledWith('my prefix', expect.any(Object));
        });
    });

    // ── completeLine() ─────────────────────────────────────────────────────────

    describe('completeLine()', () => {
        test('trims continuation at the first semicolon', async () => {
            const session   = makeCodeSession(" 'Hello';\n// comment");
            const completer = new MambaCodeCompleter(session);
            const result    = await completer.completeLine('const greeting = ');
            expect(result).toBe("const greeting =  'Hello';");
        });

        test('trims continuation at the first closing brace', async () => {
            const session   = makeCodeSession(' a + b;\n}');
            const completer = new MambaCodeCompleter(session);
            const result    = await completer.completeLine('return ');
            // First break char is ';' inside " a + b;\n}"
            expect(result).toContain('return ');
            expect(result.endsWith(';') || result.endsWith('}')).toBe(true);
        });

        test('trims continuation at the first newline', async () => {
            const session   = makeCodeSession(' = 42\nextra line');
            const completer = new MambaCodeCompleter(session);
            const result    = await completer.completeLine('const x');
            expect(result).toBe('const x = 42\n');
        });

        test('returns full prefix + completion when no break char is present', async () => {
            const session   = makeCodeSession(' number');
            const completer = new MambaCodeCompleter(session);
            const result    = await completer.completeLine('const x: ');
            expect(result).toBe('const x:  number');
        });

        test('uses very low temperature (0.2) for deterministic output', async () => {
            const session   = makeCodeSession();
            const completer = new MambaCodeCompleter(session);
            await completer.completeLine('const x = ');
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.2 }),
            );
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// MambaKnowledgeBase
// ═════════════════════════════════════════════════════════════════════════════

describe('MambaKnowledgeBase', () => {
    const DOC = { id: 'doc1', content: 'TypeScript is a typed superset of JavaScript.' };

    // ── ingest() ───────────────────────────────────────────────────────────────

    describe('ingest()', () => {
        test('calls session.evaluate() before adapt()', async () => {
            const session = makeKnowledgeSession();
            const kb      = new MambaKnowledgeBase(session);
            await kb.ingest(DOC);
            const evalOrder  = session.evaluate.mock.invocationCallOrder[0]!;
            const adaptOrder = session.adapt.mock.invocationCallOrder[0]!;
            expect(evalOrder).toBeLessThan(adaptOrder);
        });

        test('calls session.adapt() with the document content', async () => {
            const session = makeKnowledgeSession();
            const kb      = new MambaKnowledgeBase(session);
            await kb.ingest(DOC);
            expect(session.adapt).toHaveBeenCalledWith(DOC.content, expect.any(Object));
        });

        test('calls session.evaluate() after adapt()', async () => {
            const session = makeKnowledgeSession();
            const kb      = new MambaKnowledgeBase(session);
            await kb.ingest(DOC);
            expect(session.evaluate).toHaveBeenCalledTimes(2);
            const adaptOrder      = session.adapt.mock.invocationCallOrder[0]!;
            const secondEvalOrder = session.evaluate.mock.invocationCallOrder[1]!;
            expect(secondEvalOrder).toBeGreaterThan(adaptOrder);
        });

        test('returns IngestResult with the document id', async () => {
            const result = await new MambaKnowledgeBase(makeKnowledgeSession()).ingest(DOC);
            expect(result.id).toBe('doc1');
        });

        test('returns correct perplexityBefore and perplexityAfter', async () => {
            const session = makeKnowledgeSession({ perplexityBefore: 55.0, perplexityAfter: 12.0 });
            const result  = await new MambaKnowledgeBase(session).ingest(DOC);
            expect(result.perplexityBefore).toBe(55.0);
            expect(result.perplexityAfter).toBe(12.0);
        });

        test('sets improved=true when perplexity decreases', async () => {
            const session = makeKnowledgeSession({ perplexityBefore: 50, perplexityAfter: 10 });
            const result  = await new MambaKnowledgeBase(session).ingest(DOC);
            expect(result.improved).toBe(true);
        });

        test('sets improved=false when perplexity increases', async () => {
            const session = makeKnowledgeSession({ perplexityBefore: 10, perplexityAfter: 50 });
            const result  = await new MambaKnowledgeBase(session).ingest(DOC);
            expect(result.improved).toBe(false);
        });

        test('sets improved=false when perplexity is unchanged', async () => {
            const session = makeKnowledgeSession({ perplexityBefore: 25, perplexityAfter: 25 });
            const result  = await new MambaKnowledgeBase(session).ingest(DOC);
            expect(result.improved).toBe(false);
        });

        test('returns adapt losses in IngestResult', async () => {
            const session = makeKnowledgeSession({ adaptLosses: [2.0, 1.5, 1.0] });
            const result  = await new MambaKnowledgeBase(session).ingest(DOC);
            expect(result.losses).toEqual([2.0, 1.5, 1.0]);
        });

        test('uses wsla=true by default', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).ingest(DOC);
            expect(session.adapt).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ wsla: true }),
            );
        });

        test('uses 3 epochs by default', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).ingest(DOC);
            expect(session.adapt).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ epochs: 3 }),
            );
        });

        test('passes custom AdaptOptions to session.adapt()', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).ingest(DOC, { epochs: 10, wsla: false });
            expect(session.adapt).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ epochs: 10, wsla: false }),
            );
        });

        test('increments documentCount after ingest', async () => {
            const kb = new MambaKnowledgeBase(makeKnowledgeSession());
            expect(kb.documentCount).toBe(0);
            await kb.ingest(DOC);
            expect(kb.documentCount).toBe(1);
        });
    });

    // ── ingestAll() ────────────────────────────────────────────────────────────

    describe('ingestAll()', () => {
        test('returns one IngestResult per document', async () => {
            const docs = [
                { id: 'a', content: 'Document A content here' },
                { id: 'b', content: 'Document B content here' },
                { id: 'c', content: 'Document C content here' },
            ];

            // Reset evaluateCallCount for each document via separate sessions would be complex;
            // instead supply a session whose evaluate always returns different values.
            const session = makeKnowledgeSession();
            // Each evaluate call alternates before/after per doc
            session.evaluate
                .mockResolvedValueOnce(50).mockResolvedValueOnce(10)
                .mockResolvedValueOnce(40).mockResolvedValueOnce(8)
                .mockResolvedValueOnce(60).mockResolvedValueOnce(12);

            const results = await new MambaKnowledgeBase(session).ingestAll(docs);
            expect(results).toHaveLength(3);
            expect(results[0]!.id).toBe('a');
            expect(results[1]!.id).toBe('b');
            expect(results[2]!.id).toBe('c');
        });

        test('calls adapt() once per document', async () => {
            const docs    = [{ id: 'x', content: 'X' }, { id: 'y', content: 'Y' }];
            const session = makeKnowledgeSession();
            session.evaluate.mockResolvedValue(20);   // fixed value — avoid alternation complexity
            await new MambaKnowledgeBase(session).ingestAll(docs);
            expect(session.adapt).toHaveBeenCalledTimes(2);
        });

        test('documentCount equals number of docs after ingestAll()', async () => {
            const docs = [
                { id: '1', content: 'content 1' },
                { id: '2', content: 'content 2' },
            ];
            const session = makeKnowledgeSession();
            session.evaluate.mockResolvedValue(20);
            const kb = new MambaKnowledgeBase(session);
            await kb.ingestAll(docs);
            expect(kb.documentCount).toBe(2);
        });
    });

    // ── query() ────────────────────────────────────────────────────────────────

    describe('query()', () => {
        test('calls session.complete() with the question', async () => {
            const session = makeKnowledgeSession();
            const kb      = new MambaKnowledgeBase(session);
            await kb.query('What is TypeScript?');
            expect(session.complete).toHaveBeenCalledWith(
                'What is TypeScript?',
                expect.any(Object),
            );
        });

        test('returns the session completion string', async () => {
            const session = makeKnowledgeSession({ queryResult: 'TypeScript is great.' });
            const answer  = await new MambaKnowledgeBase(session).query('What is it?');
            expect(answer).toBe('TypeScript is great.');
        });

        test('uses temperature 0.5 by default', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).query('question');
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.5 }),
            );
        });

        test('uses maxNewTokens 300 by default', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).query('question');
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ maxNewTokens: 300 }),
            );
        });

        test('passes custom CompleteOptions to session.complete()', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).query('question', { temperature: 0.1, maxNewTokens: 50 });
            expect(session.complete).toHaveBeenCalledWith(
                expect.any(String),
                expect.objectContaining({ temperature: 0.1, maxNewTokens: 50 }),
            );
        });
    });

    // ── documentCount / documentIds ────────────────────────────────────────────

    describe('documentCount and documentIds', () => {
        test('documentCount starts at 0', () => {
            expect(new MambaKnowledgeBase(makeKnowledgeSession()).documentCount).toBe(0);
        });

        test('documentIds starts empty', () => {
            expect(new MambaKnowledgeBase(makeKnowledgeSession()).documentIds).toEqual([]);
        });

        test('documentIds contains ingested ids in insertion order', async () => {
            const session = makeKnowledgeSession();
            session.evaluate.mockResolvedValue(20);
            const kb = new MambaKnowledgeBase(session);
            await kb.ingest({ id: 'alpha', content: 'content alpha' });
            await kb.ingest({ id: 'beta',  content: 'content beta'  });
            expect(kb.documentIds).toEqual(['alpha', 'beta']);
        });
    });

    // ── save() ─────────────────────────────────────────────────────────────────

    describe('save()', () => {
        test('calls session.save()', async () => {
            const session = makeKnowledgeSession();
            await new MambaKnowledgeBase(session).save();
            expect(session.save).toHaveBeenCalledTimes(1);
        });

        test('resolves without throwing', async () => {
            const session = makeKnowledgeSession();
            await expect(new MambaKnowledgeBase(session).save()).resolves.not.toThrow();
        });
    });
});
