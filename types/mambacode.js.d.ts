/**
 * Type declarations for the mambacode.js npm package.
 * The package ships as plain JavaScript; these declarations provide type
 * safety when consumed from TypeScript.
 */
declare module 'mambacode.js' {
    // ── GPU utilities ──────────────────────────────────────────────────────────

    export interface InitWebGPUOptions {
        powerPreference?: 'high-performance' | 'low-power';
    }

    export interface InitWebGPUResult {
        device: GPUDevice;
        adapter: GPUAdapter;
    }

    export function initWebGPU(opts?: InitWebGPUOptions): Promise<InitWebGPUResult>;

    export function createStorageBuffer(
        device: GPUDevice,
        data: Float32Array | Uint32Array | number[],
        readable?: boolean
    ): GPUBuffer;

    export function createEmptyStorageBuffer(
        device: GPUDevice,
        byteSize: number,
        readable?: boolean
    ): GPUBuffer;

    export function createUniformBuffer(
        device: GPUDevice,
        data: ArrayBuffer | ArrayBufferView
    ): GPUBuffer;

    export function createComputePipeline(
        device: GPUDevice,
        wgslSource: string,
        entryPoint: string
    ): GPUComputePipeline;

    export function createBindGroup(
        device: GPUDevice,
        pipeline: GPUComputePipeline,
        buffers: GPUBuffer[],
        groupIndex?: number
    ): GPUBindGroup;

    export function dispatchKernel(
        device: GPUDevice,
        pipeline: GPUComputePipeline,
        bindGroup: GPUBindGroup,
        workgroups: [number, number, number]
    ): void;

    export function readBuffer(
        device: GPUDevice,
        srcBuffer: GPUBuffer,
        byteSize: number
    ): Promise<Float32Array>;

    export function uploadBuffer(
        device: GPUDevice,
        buffer: GPUBuffer,
        data: Float32Array,
        byteOffset?: number
    ): void;

    export function cdiv(a: number, b: number): number;

    // ── Model ──────────────────────────────────────────────────────────────────

    export interface MambaModelConfig {
        vocabSize: number;
        dModel: number;
        numLayers: number;
        dState?: number;
        dConv?: number;
        expand?: number;
        eosId?: number;
    }

    export interface SamplingOptions {
        temperature?: number;
        topK?: number;
        topP?: number;
    }

    export interface ModelForwardResult {
        logits: Float32Array;
        gpuLogits: GPUBuffer;
        caches: unknown[];
    }

    export class MambaModel {
        device: GPUDevice;
        config: Required<MambaModelConfig>;

        constructor(device: GPUDevice, config: MambaModelConfig);

        forward(
            tokenIds: number[] | Uint32Array,
            batch: number,
            seqLen: number
        ): Promise<ModelForwardResult>;

        generate(
            promptIds: number[],
            maxNewTokens?: number,
            samplingOpts?: SamplingOptions
        ): Promise<number[]>;

        parameters(): Array<{ buf: GPUBuffer; numel: number; name: string }>;

        setWSLAMode(enabled: boolean): void;

        exportWeights(): Promise<ArrayBuffer>;

        loadWeights(buffer: ArrayBuffer): Promise<void>;
    }

    export class MambaBlock {
        constructor(device: GPUDevice, config: {
            dModel: number;
            dState?: number;
            dConv?: number;
            expand?: number;
        });
    }

    // ── Trainer ────────────────────────────────────────────────────────────────

    export interface TrainOptions {
        learningRate?: number;
        epochs?: number;
        batchSize?: number;
        seqLen?: number;
        maxGradNorm?: number;
        weightDecay?: number;
        beta1?: number;
        beta2?: number;
        eps?: number;
        wsla?: boolean;
        onEpochEnd?: ((epoch: number, loss: number) => void) | null;
    }

    export class MambaTrainer {
        model: MambaModel;
        tokenizer: BPETokenizer | null;
        device: GPUDevice;

        constructor(model: MambaModel, tokenizer?: BPETokenizer | null);

        train(input: string | number[], opts?: TrainOptions): Promise<number[]>;
        evaluate(input: string | number[]): Promise<number>;
    }

    // ── Tokenizer ──────────────────────────────────────────────────────────────

    export interface BPEEncodeOptions {
        addBos?: boolean;
        addEos?: boolean;
    }

    export class BPETokenizer {
        vocab: Map<string, number>;
        idToToken: Map<number, string>;
        merges: Map<string, number>;
        bosId: number | null;
        eosId: number | null;
        padId: number | null;

        constructor();

        load(
            vocab: string | Record<string, number>,
            merges: string | string[]
        ): Promise<void>;

        loadFromObjects(
            vocabObj: Record<string, number>,
            mergeArr: string[]
        ): void;

        encode(text: string, opts?: BPEEncodeOptions): number[];
        decode(ids: number[]): string;

        get vocabSize(): number;
    }

    // ── Autograd ───────────────────────────────────────────────────────────────

    export function crossEntropyLoss(logits: Float32Array, targetId: number): number;
    export function crossEntropyGrad(logits: Float32Array, targetId: number): Float32Array;

    // ── Quantization ───────────────────────────────────────────────────────────

    export function quantizeFp16(data: Float32Array): Uint16Array;
    export function dequantizeFp16(data: Uint16Array): Float32Array;
    export function quantizeInt8(data: Float32Array): { data: Int8Array; scale: number };
    export function dequantizeInt8(data: Int8Array, scale: number): Float32Array;
    export function estimateMemory(config: MambaModelConfig): number;

    // ── WGSL Kernel sources ────────────────────────────────────────────────────

    export const SELECTIVE_SCAN_FORWARD_WGSL: string;
    export const SELECTIVE_SCAN_BACKWARD_WGSL: string;
    export const CONV1D_FORWARD_WGSL: string;
    export const CONV1D_BACKWARD_WGSL: string;
    export const LINEAR_FORWARD_WGSL: string;
    export const LINEAR_BACKWARD_WGSL: string;
    export const WEIGHT_UPDATE_WGSL: string;
    export const GRAD_CLIP_WGSL: string;
    export const ACTIVATIONS_WGSL: string;
    export const ACTIVATIONS_BACKWARD_WGSL: string;

    // ── Version ────────────────────────────────────────────────────────────────

    export const VERSION: string;
    export const DESCRIPTION: string;
}
