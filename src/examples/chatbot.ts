/**
 * chatbot.ts – Multi-turn conversational wrapper for MambaKit.
 *
 * Demonstrates how to build a stateful chatbot on top of MambaSession by
 * formatting a message history into a single prompt string, streaming or
 * batching the model response, and maintaining conversation state.
 *
 * Usage:
 *   const chatbot = new MambaChatbot(session, 'You are a helpful assistant.');
 *   const reply = await chatbot.chat('What is 2 + 2?');
 */

// ── Minimal session interface ─────────────────────────────────────────────────
// Defined locally so this example has no dependency on the mambacode.js build.

export interface CompleteOptions {
    maxNewTokens? : number;
    temperature?  : number;
    topK?         : number;
    topP?         : number;
}

export interface ChatSession {
    complete(prompt: string, options?: CompleteOptions): Promise<string>;
    completeStream(prompt: string, options?: CompleteOptions): AsyncIterable<string>;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant';

export interface Message {
    role    : MessageRole;
    content : string;
}

export interface ChatOptions extends CompleteOptions {
    /** Override the system prompt for a single turn. */
    systemPrompt?: string;
}

// ── MambaChatbot ──────────────────────────────────────────────────────────────

export class MambaChatbot {
    private _history: Message[] = [];

    constructor(
        private readonly _session: ChatSession,
        private readonly _defaultSystemPrompt = 'You are a helpful assistant.',
    ) {}

    /**
     * Formats the current history plus a new user message into a single
     * prompt string that the model can continue.
     */
    formatPrompt(userMessage: string, systemPrompt?: string): string {
        const sys   = systemPrompt ?? this._defaultSystemPrompt;
        const lines = [`System: ${sys}`];

        for (const msg of this._history) {
            const speaker = msg.role === 'user' ? 'User' : 'Assistant';
            lines.push(`${speaker}: ${msg.content}`);
        }

        lines.push(`User: ${userMessage}`);
        lines.push('Assistant:');
        return lines.join('\n');
    }

    /**
     * Sends a user message, waits for the full response, updates history,
     * and returns the assistant reply as a string.
     */
    async chat(userMessage: string, options: ChatOptions = {}): Promise<string> {
        const { systemPrompt, ...completeOpts } = options;
        const prompt = this.formatPrompt(userMessage, systemPrompt);

        const raw = await this._session.complete(prompt, {
            maxNewTokens : 200,
            temperature  : 0.7,
            topK         : 50,
            topP         : 0.9,
            ...completeOpts,
        });

        // The model may generate several turns; keep only the first assistant reply.
        const response = raw.split('\nUser:')[0].trim();

        this._history.push({ role: 'user',      content: userMessage });
        this._history.push({ role: 'assistant', content: response    });

        return response;
    }

    /**
     * Streaming variant of `chat()`.  Yields tokens as they arrive and
     * updates history once the full response has been assembled.
     */
    async *chatStream(userMessage: string, options: ChatOptions = {}): AsyncIterable<string> {
        const { systemPrompt, ...completeOpts } = options;
        const prompt = this.formatPrompt(userMessage, systemPrompt);

        let fullResponse = '';
        for await (const chunk of this._session.completeStream(prompt, {
            maxNewTokens : 200,
            temperature  : 0.7,
            topK         : 50,
            topP         : 0.9,
            ...completeOpts,
        })) {
            fullResponse += chunk;
            yield chunk;
        }

        const response = fullResponse.split('\nUser:')[0].trim();
        this._history.push({ role: 'user',      content: userMessage });
        this._history.push({ role: 'assistant', content: response    });
    }

    /** Clears all conversation history. */
    clearHistory(): void {
        this._history = [];
    }

    /** Returns a read-only snapshot of the current conversation history. */
    get history(): readonly Message[] {
        return this._history;
    }

    /** Number of messages in the current conversation (user + assistant turns). */
    get turnCount(): number {
        return Math.floor(this._history.length / 2);
    }
}
