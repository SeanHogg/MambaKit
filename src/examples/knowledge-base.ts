/**
 * knowledge-base.ts – Domain knowledge ingestion and querying for MambaKit.
 *
 * Demonstrates how to build a private knowledge base by fine-tuning
 * MambaSession on a corpus of documents, measuring the perplexity improvement
 * for each one, and then querying the adapted model with natural-language
 * questions.
 *
 * Usage:
 *   const kb = new MambaKnowledgeBase(session);
 *   await kb.ingest({ id: 'intro', content: 'MambaKit is…' });
 *   const answer = await kb.query('What is MambaKit?');
 */

// ── Minimal session interface ─────────────────────────────────────────────────

export interface CompleteOptions {
    maxNewTokens? : number;
    temperature?  : number;
    topK?         : number;
    topP?         : number;
}

export interface AdaptOptions {
    epochs?       : number;
    learningRate? : number;
    seqLen?       : number;
    wsla?         : boolean;
    fullTrain?    : boolean;
    onProgress?   : (epoch: number, loss: number) => void;
}

export interface AdaptResult {
    losses     : number[];
    epochCount : number;
    durationMs : number;
}

export interface KnowledgeSession {
    evaluate(text: string): Promise<number>;
    adapt(text: string, options?: AdaptOptions): Promise<AdaptResult>;
    complete(question: string, options?: CompleteOptions): Promise<string>;
    save(options?: { storage?: string; key?: string }): Promise<void>;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Document {
    /** Unique identifier for the document. */
    id      : string;
    /** Raw text content to ingest. */
    content : string;
}

export interface IngestResult {
    id               : string;
    /** Model perplexity on this document before fine-tuning. */
    perplexityBefore : number;
    /** Model perplexity on this document after fine-tuning. */
    perplexityAfter  : number;
    /** True when perplexity decreased (model learned the content). */
    improved         : boolean;
    /** Per-epoch loss values from the adapt() call. */
    losses           : number[];
}

// ── MambaKnowledgeBase ────────────────────────────────────────────────────────

export class MambaKnowledgeBase {
    private _docs: Document[] = [];

    constructor(private readonly _session: KnowledgeSession) {}

    /**
     * Ingests a single document: measures perplexity before and after
     * fine-tuning, then returns an `IngestResult` describing the improvement.
     */
    async ingest(doc: Document, options: AdaptOptions = {}): Promise<IngestResult> {
        const perplexityBefore = await this._session.evaluate(doc.content);

        const result = await this._session.adapt(doc.content, {
            wsla   : true,
            epochs : 3,
            ...options,
        });

        const perplexityAfter = await this._session.evaluate(doc.content);

        this._docs.push(doc);

        return {
            id               : doc.id,
            perplexityBefore,
            perplexityAfter,
            improved         : perplexityAfter < perplexityBefore,
            losses           : result.losses,
        };
    }

    /**
     * Ingests multiple documents in sequence.
     * Returns one `IngestResult` per document in the same order.
     */
    async ingestAll(
        docs    : Document[],
        options : AdaptOptions = {},
    ): Promise<IngestResult[]> {
        const results: IngestResult[] = [];
        for (const doc of docs) {
            results.push(await this.ingest(doc, options));
        }
        return results;
    }

    /**
     * Queries the adapted model with a natural-language question and returns
     * the generated answer as a string.
     */
    async query(question: string, options: CompleteOptions = {}): Promise<string> {
        return this._session.complete(question, {
            maxNewTokens : 300,
            temperature  : 0.5,
            topK         : 40,
            topP         : 0.9,
            ...options,
        });
    }

    /** Persists the adapted model weights via the underlying session. */
    async save(): Promise<void> {
        await this._session.save();
    }

    /** Total number of documents ingested so far. */
    get documentCount(): number {
        return this._docs.length;
    }

    /** IDs of all ingested documents in insertion order. */
    get documentIds(): string[] {
        return this._docs.map(d => d.id);
    }
}
