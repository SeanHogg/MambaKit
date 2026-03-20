/**
 * errors.ts – Typed error class for MambaKit.
 */
export class MambaKitError extends Error {
    code;
    cause;
    constructor(code, message, cause) {
        super(message);
        this.code = code;
        this.cause = cause;
        this.name = 'MambaKitError';
    }
}
//# sourceMappingURL=errors.js.map