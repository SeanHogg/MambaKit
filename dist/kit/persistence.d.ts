/**
 * persistence.ts – Storage helpers for MambaKit.
 *
 * Supports three storage targets:
 *  - IndexedDB  (browser default)
 *  - Download   (Blob URL trigger)
 *  - File System Access API
 */
export declare function saveToIndexedDB(key: string, buffer: ArrayBuffer): Promise<void>;
export declare function loadFromIndexedDB(key: string): Promise<ArrayBuffer | undefined>;
export declare function triggerDownload(filename: string, buffer: ArrayBuffer): Promise<void>;
export declare function saveViaFileSystemAPI(filename: string, buffer: ArrayBuffer): Promise<void>;
export declare function loadViaFileSystemAPI(): Promise<ArrayBuffer>;
//# sourceMappingURL=persistence.d.ts.map