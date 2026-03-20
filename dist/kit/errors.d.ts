/**
 * errors.ts – Typed error class for MambaKit.
 */
export type MambaKitErrorCode = 'GPU_UNAVAILABLE' | 'TOKENIZER_LOAD_FAILED' | 'CHECKPOINT_FETCH_FAILED' | 'CHECKPOINT_INVALID' | 'INPUT_TOO_SHORT' | 'STORAGE_UNAVAILABLE' | 'SESSION_DESTROYED' | 'UNKNOWN';
export declare class MambaKitError extends Error {
    readonly code: MambaKitErrorCode;
    readonly cause?: unknown | undefined;
    constructor(code: MambaKitErrorCode, message: string, cause?: unknown | undefined);
}
//# sourceMappingURL=errors.d.ts.map