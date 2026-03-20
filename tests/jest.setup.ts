/**
 * jest.setup.ts – Polyfills for WebGPU globals required by mambacode.js.
 *
 * The published mambacode.js package references GPUBufferUsage and
 * GPUMapMode at module-evaluation time. Since Node.js has no WebGPU
 * these constants must be defined before any test file imports the module.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
const g = globalThis as any;

g.GPUBufferUsage = {
    MAP_READ  : 0x0001,
    MAP_WRITE : 0x0002,
    COPY_SRC  : 0x0004,
    COPY_DST  : 0x0008,
    INDEX     : 0x0010,
    VERTEX    : 0x0020,
    UNIFORM   : 0x0040,
    STORAGE   : 0x0080,
    INDIRECT  : 0x0100,
    QUERY_RESOLVE: 0x0200,
};

g.GPUMapMode = {
    READ  : 0x0001,
    WRITE : 0x0002,
};

g.GPUShaderStage = {
    VERTEX  : 0x1,
    FRAGMENT: 0x2,
    COMPUTE : 0x4,
};
