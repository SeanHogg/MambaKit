/**
 * streaming.ts – AsyncIterable token streaming adapter for MambaKit.
 *
 * Wraps the step-by-step generation loop from MambaModel so that each
 * token is yielded immediately after sampling, enabling real-time streaming UIs.
 */
import type { MambaModel, SamplingOptions } from 'mambacode.js';
/**
 * Yields one token ID at a time, applying the same sampling logic as
 * `MambaModel.generate()` but yielding each step incrementally.
 */
export declare function tokenStream(model: MambaModel, promptIds: number[], maxNewTokens: number, samplingOpts?: SamplingOptions): AsyncGenerator<number>;
//# sourceMappingURL=streaming.d.ts.map