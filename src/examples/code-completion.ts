/**
 * code-completion.ts – IDE-style code completion helper for MambaKit.
 *
 * Wraps MambaSession with code-oriented defaults (lower temperature, shorter
 * bursts) and provides a `completeLine()` convenience method that stops
 * at the first natural code boundary.
 *
 * Usage:
 *   const completer = new MambaCodeCompleter(session);
 *   const result    = await completer.complete('function add(a: number, b:');
 *   console.log(result.full);
 */

// ── Minimal session interface ─────────────────────────────────────────────────

export interface CompleteOptions {
    maxNewTokens? : number;
    temperature?  : number;
    topK?         : number;
    topP?         : number;
}

export interface CodeSession {
    complete(prefix: string, options?: CompleteOptions): Promise<string>;
    completeStream(prefix: string, options?: CompleteOptions): AsyncIterable<string>;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CompletionResult {
    /** The original input prefix. */
    prefix     : string;
    /** The generated continuation (excluding the prefix). */
    completion : string;
    /** Concatenation of prefix + completion. */
    full       : string;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

const CODE_DEFAULTS: CompleteOptions = {
    maxNewTokens : 128,
    temperature  : 0.4,
    topK         : 40,
    topP         : 0.85,
};

/** Characters that mark the end of a logical code statement or block. */
const LINE_BREAK_RE = /[;\n}]/;

// ── MambaCodeCompleter ────────────────────────────────────────────────────────

export class MambaCodeCompleter {
    constructor(private readonly _session: CodeSession) {}

    /**
     * Generates a code completion for the given prefix and returns the result
     * as a `CompletionResult` containing the prefix, continuation, and full text.
     *
     * Uses lower-temperature sampling by default to produce more deterministic
     * (less creative) code output.
     */
    async complete(
        prefix  : string,
        options : CompleteOptions = {},
    ): Promise<CompletionResult> {
        const completion = await this._session.complete(prefix, {
            ...CODE_DEFAULTS,
            ...options,
        });
        return { prefix, completion, full: prefix + completion };
    }

    /**
     * Streaming variant of `complete()`.  Yields one token string at a time
     * so the editor can display characters as they arrive.
     */
    async *completeStream(
        prefix  : string,
        options : CompleteOptions = {},
    ): AsyncIterable<string> {
        yield* this._session.completeStream(prefix, {
            ...CODE_DEFAULTS,
            ...options,
        });
    }

    /**
     * Single-line completion: generates a continuation and trims it at the
     * first semicolon, closing brace, or newline — whichever comes first.
     *
     * Ideal for inline code suggestions where only one statement is needed.
     */
    async completeLine(prefix: string): Promise<string> {
        const result = await this.complete(prefix, {
            maxNewTokens : 80,
            temperature  : 0.2,   // very deterministic for single-line suggestions
        });

        const breakIdx = result.completion.search(LINE_BREAK_RE);
        const trimmed  = breakIdx >= 0
            ? result.completion.slice(0, breakIdx + 1)
            : result.completion;

        return prefix + trimmed;
    }
}
