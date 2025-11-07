// TEAM-409: marketplace-node implementation
// Created by: TEAM-409
//
// Wrapper around marketplace-sdk WASM bindings for Node.js/TypeScript usage

import type {
  WorkerCatalogEntry,
  WorkerFilter,
  WorkerType,
  Platform,
  Architecture,
} from '@rbee/marketplace-sdk'

// Lazy-load SDK to avoid initialization issues
let sdk: typeof import('@rbee/marketplace-sdk') | null = null

async function getSDK() {
  if (!sdk) {
    sdk = await import('@rbee/marketplace-sdk')
    sdk.init()
  }
  return sdk
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WORKER CATALOG FUNCTIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * List all available worker binaries from the catalog
 * 
 * @returns Array of worker catalog entries
 * @throws Error if network request fails
 * 
 * @example
 * ```typescript
 * const workers = await listWorkerBinaries()
 * console.log(`Found ${workers.length} workers`)
 * ```
 */
export async function listWorkerBinaries(): Promise<WorkerCatalogEntry[]> {
  const sdk = await getSDK()
  
  try {
    const workers = await sdk.list_workers()
    return workers as WorkerCatalogEntry[]
  } catch (error) {
    console.error('[marketplace-node] Failed to list workers:', error)
    throw new Error(`Failed to list workers: ${error}`)
  }
}

/**
 * Get a specific worker by ID
 * 
 * @param id - Worker ID (e.g., "llm-worker-rbee-cuda")
 * @returns Worker catalog entry or null if not found
 * @throws Error if network request fails
 * 
 * @example
 * ```typescript
 * const worker = await getWorkerById("llm-worker-rbee-cuda")
 * if (worker) {
 *   console.log(`Found worker: ${worker.name}`)
 * }
 * ```
 */
export async function getWorkerById(id: string): Promise<WorkerCatalogEntry | null> {
  const sdk = await getSDK()
  
  try {
    const worker = await sdk.get_worker(id)
    return worker as WorkerCatalogEntry | null
  } catch (error) {
    console.error(`[marketplace-node] Failed to get worker ${id}:`, error)
    throw new Error(`Failed to get worker: ${error}`)
  }
}

/**
 * Filter workers by criteria
 * 
 * @param filter - Filter options
 * @returns Array of matching workers
 * @throws Error if network request fails
 * 
 * @example
 * ```typescript
 * const cudaWorkers = await filterWorkers({
 *   workerType: "cuda",
 *   platform: "linux",
 *   architecture: "x86_64"
 * })
 * ```
 */
export async function filterWorkers(filter: WorkerFilter): Promise<WorkerCatalogEntry[]> {
  const sdk = await getSDK()
  
  try {
    const workers = await sdk.filter_workers(filter)
    return workers as WorkerCatalogEntry[]
  } catch (error) {
    console.error('[marketplace-node] Failed to filter workers:', error)
    throw new Error(`Failed to filter workers: ${error}`)
  }
}

/**
 * Filter workers by type
 * 
 * @param type - Worker type (cpu, cuda, metal)
 * @returns Array of workers matching the type
 * 
 * @example
 * ```typescript
 * const cudaWorkers = await filterWorkersByType("cuda")
 * ```
 */
export async function filterWorkersByType(type: WorkerType): Promise<WorkerCatalogEntry[]> {
  return filterWorkers({ workerType: type })
}

/**
 * Filter workers by platform
 * 
 * @param platform - Platform (linux, macos, windows)
 * @returns Array of workers supporting the platform
 * 
 * @example
 * ```typescript
 * const linuxWorkers = await filterWorkersByPlatform("linux")
 * ```
 */
export async function filterWorkersByPlatform(platform: Platform): Promise<WorkerCatalogEntry[]> {
  return filterWorkers({ platform })
}

/**
 * Filter workers by architecture
 * 
 * @param architecture - CPU architecture (x86_64, aarch64)
 * @returns Array of workers supporting the architecture
 * 
 * @example
 * ```typescript
 * const arm64Workers = await filterWorkersByArchitecture("aarch64")
 * ```
 */
export async function filterWorkersByArchitecture(architecture: Architecture): Promise<WorkerCatalogEntry[]> {
  return filterWorkers({ architecture })
}

/**
 * Find workers compatible with a specific model
 * 
 * @param architecture - Model architecture (e.g., "llama", "mistral")
 * @param format - Model format (e.g., "safetensors", "gguf")
 * @returns Array of compatible workers
 * @throws Error if network request fails
 * 
 * @example
 * ```typescript
 * const compatibleWorkers = await findCompatibleWorkers("llama", "safetensors")
 * console.log(`Found ${compatibleWorkers.length} compatible workers`)
 * ```
 */
export async function findCompatibleWorkers(
  architecture: string,
  format: string
): Promise<WorkerCatalogEntry[]> {
  const sdk = await getSDK()
  
  try {
    const workers = await sdk.find_compatible_workers(architecture, format)
    return workers as WorkerCatalogEntry[]
  } catch (error) {
    console.error('[marketplace-node] Failed to find compatible workers:', error)
    throw new Error(`Failed to find compatible workers: ${error}`)
  }
}

/**
 * Get workers compatible with a model's requirements
 * 
 * @param requirements - Model requirements
 * @returns Array of compatible workers
 * 
 * @example
 * ```typescript
 * const workers = await getCompatibleWorkers({
 *   architecture: "llama",
 *   format: "safetensors",
 *   minContextLength: 8192
 * })
 * ```
 */
export async function getCompatibleWorkers(requirements: {
  architecture?: string
  format?: string
  minContextLength?: number
  platform?: Platform
  workerType?: WorkerType
}): Promise<WorkerCatalogEntry[]> {
  const filter: WorkerFilter = {
    modelArchitecture: requirements.architecture,
    modelFormat: requirements.format,
    minContextLength: requirements.minContextLength,
    platform: requirements.platform,
    workerType: requirements.workerType,
  }
  
  return filterWorkers(filter)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-409: COMPATIBILITY CHECKING FUNCTIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * Check if a model is compatible with ANY of our workers
 *
 * @param metadata - Model metadata from HuggingFace
 * @returns Compatibility result with detailed information
 * @throws Error if WASM call fails
 *
 * @example
 * ```typescript
 * const metadata = {
 *   architecture: "llama",
 *   format: "safetensors",
 *   quantization: null,
 *   parameters: "7B",
 *   size_bytes: 14000000000,
 *   max_context_length: 8192
 * }
 * const result = await isModelCompatible(metadata)
 * console.log(result.compatible) // true/false
 * ```
 */
export async function isModelCompatible(metadata: any): Promise<any> {
  const sdk = await getSDK()
  
  try {
    return await sdk.is_model_compatible_wasm(metadata)
  } catch (error) {
    console.error('[marketplace-node] Failed to check compatibility:', error)
    throw new Error(`Failed to check compatibility: ${error}`)
  }
}

/**
 * Filter a list of models to only include compatible ones
 *
 * PRIMARY FUNCTION for HuggingFace filtering!
 *
 * @param models - Array of model metadata from HuggingFace
 * @returns Array of compatible models only
 * @throws Error if WASM call fails
 *
 * @example
 * ```typescript
 * const allModels = await fetchHuggingFaceModels()
 * const compatible = await filterCompatibleModels(allModels)
 * // Only shows models that work with our workers
 * ```
 */
export async function filterCompatibleModels(models: any[]): Promise<any[]> {
  const sdk = await getSDK()
  
  try {
    return await sdk.filter_compatible_models_wasm(models)
  } catch (error) {
    console.error('[marketplace-node] Failed to filter models:', error)
    throw new Error(`Failed to filter models: ${error}`)
  }
}

/**
 * Check if a specific model is compatible with a specific worker
 *
 * @param metadata - Model metadata
 * @param workerArchitectures - Architectures supported by worker
 * @param workerFormats - Formats supported by worker
 * @param workerMaxContext - Max context length of worker
 * @returns Compatibility result with detailed information
 * @throws Error if WASM call fails
 *
 * @example
 * ```typescript
 * const result = await checkModelWorkerCompatibility(
 *   modelMetadata,
 *   ["llama", "mistral"],
 *   ["safetensors", "gguf"],
 *   32768
 * )
 * console.log(result.compatible) // true/false
 * console.log(result.warnings)   // Any warnings
 * ```
 */
export async function checkModelWorkerCompatibility(
  metadata: any,
  workerArchitectures: string[],
  workerFormats: string[],
  workerMaxContext: number
): Promise<any> {
  const sdk = await getSDK()
  
  try {
    return await sdk.check_model_worker_compatibility_wasm(
      metadata,
      workerArchitectures,
      workerFormats,
      workerMaxContext
    )
  } catch (error) {
    console.error('[marketplace-node] Failed to check worker compatibility:', error)
    throw new Error(`Failed to check worker compatibility: ${error}`)
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CIVITAI API FUNCTIONS (TEAM-460)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export {
  listCivitaiModels,
  getCivitaiModel,
  getCompatibleCivitaiModels,
} from './civitai'

export type {
  CivitaiModel,
  CivitaiModelVersion,
  CivitaiFile,
  CivitaiImage,
  CivitaiListResponse,
  ListCivitaiModelsOptions,
} from './civitai'

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RE-EXPORTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export type {
  WorkerCatalogEntry,
  WorkerFilter,
  WorkerType,
  Platform,
  Architecture,
  WorkerImplementation,
  BuildSystem,
  SourceInfo,
  BuildConfig,
} from '@rbee/marketplace-sdk'
