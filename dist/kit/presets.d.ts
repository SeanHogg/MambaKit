/**
 * presets.ts – Model size presets and config resolver for MambaKit.
 */
import type { MambaModelConfig } from 'mambacode.js';
import type { MambaSessionOptions } from './session.js';
/** Pre-defined model size presets matching the PRD specification. */
export declare const MODEL_PRESETS: Record<string, Partial<MambaModelConfig>>;
/**
 * Resolves a fully-populated `MambaModelConfig` from session options and the
 * actual tokenizer vocab size.
 *
 * Resolution order:
 *  1. Preset fields (default: 'medium')
 *  2. `modelConfig` overrides (only applied when `modelSize === 'custom'`)
 *  3. Required `vocabSize` from the tokenizer
 */
export declare function resolveModelConfig(options: MambaSessionOptions, vocabSize: number): Required<MambaModelConfig>;
//# sourceMappingURL=presets.d.ts.map