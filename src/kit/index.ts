/**
 * MambaKit – Opinionated facade over MambaCode.js.
 *
 * Import from this entry point:
 *   import { MambaSession } from 'mambakit';
 *
 * Or, when consuming from the mambacode.js sub-path:
 *   import { MambaSession } from 'mambacode.js/kit';
 */

export { MambaSession }        from './session.js';
export { MambaKitError }       from './errors.js';
export { MODEL_PRESETS, resolveLayerSchedule } from './presets.js';

export type { MambaKitErrorCode }  from './errors.js';
export type { LayerSchedulePreset } from './presets.js';
export type {
    MambaSessionOptions,
    CompleteOptions,
    AdaptOptions,
    AdaptResult,
    SaveOptions,
    LoadOptions,
    StorageTarget,
    CreateProgressEvent,
    CreateStage,
    CreateCallbacks,
    SessionInternals,
} from './session.js';
