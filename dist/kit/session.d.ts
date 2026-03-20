/**
 * session.ts – MambaSession: the single entry point for all MambaKit functionality.
 *
 * MambaSession is a facade over MambaCode.js that collapses the 8-step
 * async setup sequence into a single `MambaSession.create()` call.
 */
import { BPETokenizer, MambaModel, MambaTrainer, type MambaModelConfig } from 'mambacode.js';
export interface MambaSessionOptions {
    /** URL to a .bin checkpoint file. Optional — model starts with random weights if omitted. */
    checkpointUrl?: string;
    /** URL to vocab.json (Qwen3.5-Coder compatible). Required unless vocabObject is supplied. */
    vocabUrl?: string;
    /** URL to merges.txt. Required unless mergesArray is supplied. */
    mergesUrl?: string;
    /** In-memory vocabulary object — alternative to vocabUrl. */
    vocabObject?: Record<string, number>;
    /** In-memory merges array — alternative to mergesUrl. */
    mergesArray?: string[];
    /** Unique name for this session, used as the IndexedDB key. Default: 'default'. */
    name?: string;
    /**
     * Model size preset. Overrides individual model config fields.
     * - 'nano'   : dModel=128,  numLayers=4
     * - 'small'  : dModel=256,  numLayers=6
     * - 'medium' : dModel=512,  numLayers=8  (default)
     * - 'large'  : dModel=768,  numLayers=12
     * - 'custom' : use modelConfig directly
     */
    modelSize?: 'nano' | 'small' | 'medium' | 'large' | 'custom';
    /** Fine-grained model configuration. Only used when modelSize is 'custom'. */
    modelConfig?: Partial<MambaModelConfig>;
    /** WebGPU power preference. Default: 'high-performance'. */
    powerPreference?: 'high-performance' | 'low-power';
    /** Number of times to retry a failed checkpoint fetch. Default: 2. */
    fetchRetries?: number;
}
export interface CompleteOptions {
    maxNewTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
}
export interface AdaptOptions {
    epochs?: number;
    learningRate?: number;
    seqLen?: number;
    wsla?: boolean;
    fullTrain?: boolean;
    onProgress?: (epoch: number, loss: number) => void;
}
export interface AdaptResult {
    losses: number[];
    epochCount: number;
    durationMs: number;
}
export type StorageTarget = 'indexedDB' | 'download' | 'fileSystem';
export interface SaveOptions {
    storage?: StorageTarget;
    filename?: string;
    key?: string;
}
export interface LoadOptions {
    storage?: StorageTarget;
    url?: string;
    key?: string;
}
export type CreateStage = 'gpu' | 'tokenizer' | 'model' | 'weights';
export interface CreateProgressEvent {
    stage: CreateStage;
    progress: number;
    message: string;
}
export interface SessionInternals {
    device: GPUDevice;
    model: MambaModel;
    trainer: MambaTrainer;
    tokenizer: BPETokenizer;
}
export interface CreateCallbacks {
    onProgress?: (event: CreateProgressEvent) => void;
}
export declare class MambaSession {
    private _device;
    private _tokenizer;
    private _model;
    private _trainer;
    private _name;
    private _destroyed;
    private constructor();
    static create(options: MambaSessionOptions, callbacks?: CreateCallbacks): Promise<MambaSession>;
    complete(prompt: string, options?: CompleteOptions): Promise<string>;
    completeStream(prompt: string, options?: CompleteOptions): AsyncIterable<string>;
    adapt(text: string, options?: AdaptOptions): Promise<AdaptResult>;
    evaluate(text: string): Promise<number>;
    save(options?: SaveOptions): Promise<void>;
    load(options?: LoadOptions): Promise<boolean>;
    destroy(): void;
    get internals(): SessionInternals;
    private _assertNotDestroyed;
}
//# sourceMappingURL=session.d.ts.map