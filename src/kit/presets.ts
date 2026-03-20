/**
 * presets.ts – Model size presets and config resolver for MambaKit.
 */

import type { MambaModelConfig } from 'mambacode.js';
import type { MambaSessionOptions } from './session.js';

/** Pre-defined model size presets matching the PRD specification. */
export const MODEL_PRESETS: Record<string, Partial<MambaModelConfig>> = {
    nano  : { dModel: 128, numLayers:  4, dState: 16, dConv: 4, expand: 2 },
    small : { dModel: 256, numLayers:  6, dState: 16, dConv: 4, expand: 2 },
    medium: { dModel: 512, numLayers:  8, dState: 16, dConv: 4, expand: 2 },
    large : { dModel: 768, numLayers: 12, dState: 16, dConv: 4, expand: 2 },
};

const DEFAULT_PRESET = 'medium';

/**
 * Resolves a fully-populated `MambaModelConfig` from session options and the
 * actual tokenizer vocab size.
 *
 * Resolution order:
 *  1. Preset fields (default: 'medium')
 *  2. `modelConfig` overrides (only applied when `modelSize === 'custom'`)
 *  3. Required `vocabSize` from the tokenizer
 */
export function resolveModelConfig(
    options: MambaSessionOptions,
    vocabSize: number,
): Required<MambaModelConfig> {
    const presetName = options.modelSize === 'custom' || options.modelSize == null
        ? DEFAULT_PRESET
        : options.modelSize;

    const preset = MODEL_PRESETS[presetName] ?? MODEL_PRESETS[DEFAULT_PRESET]!;

    const overrides: Partial<MambaModelConfig> =
        options.modelSize === 'custom' && options.modelConfig
            ? options.modelConfig
            : {};

    return {
        vocabSize,
        dModel    : overrides.dModel     ?? preset.dModel     ?? 512,
        numLayers : overrides.numLayers  ?? preset.numLayers  ?? 8,
        dState    : overrides.dState     ?? preset.dState     ?? 16,
        dConv     : overrides.dConv      ?? preset.dConv      ?? 4,
        expand    : overrides.expand     ?? preset.expand     ?? 2,
        eosId     : overrides.eosId      ?? preset.eosId      ?? -1,
    };
}
