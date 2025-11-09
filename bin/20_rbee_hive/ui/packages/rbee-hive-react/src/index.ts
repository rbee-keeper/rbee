// TEAM-353: Migrated to use TanStack Query (no manual state management)
// TEAM-353: Uses WASM SDK (job-based architecture)
// TEAM-377: React Query removed - use @rbee/ui/providers instead

// TEAM-377: React Query REMOVED
// DO NOT re-export React Query - import from @rbee/ui/providers:
//   import { QueryProvider } from '@rbee/ui/providers'
// This ensures consistent configuration across all apps

// TEAM-381: Import WASM SDK dynamically to avoid module load issues
import type { HeartbeatMonitor, HiveClient, ModelInfo, OperationBuilder } from '@rbee/rbee-hive-sdk'
import { useMutation, useQuery } from '@tanstack/react-query'

// TEAM-381: Re-export types only (NOT classes - they must be imported directly from SDK)
export type {
  HeartbeatMonitor,
  // WASM classes (import these from @rbee/rbee-hive-sdk when needed)
  HiveClient,
  HiveHeartbeatEvent,
  HiveInfo,
  ModelInfo,
  OperationBuilder,
  // TEAM-381: Auto-generated types from Rust contract crates
  ProcessStats,
} from '@rbee/rbee-hive-sdk'

// TEAM-381: HuggingFace model types (for search - UI only, not from backend)
// Defined here since it's not part of the Rust backend contract
export interface HFModel {
  id: string
  modelId: string
  author: string
  downloads: number
  likes: number
  tags: string[]
  private: boolean
  gated: boolean | string
}

// TEAM-381: Lazy WASM SDK initialization (avoid module load issues)
let sdkModule: typeof import('@rbee/rbee-hive-sdk') | null = null
async function ensureWasmInit() {
  if (!sdkModule) {
    sdkModule = await import('@rbee/rbee-hive-sdk')
    sdkModule.init() // Initialize WASM module
  }
  return sdkModule
}

// TEAM-381: Lazy client initialization (avoid window access at module load time)
let client: HiveClient | null = null
async function getClient(): Promise<HiveClient> {
  if (!client) {
    const sdk = await ensureWasmInit()
    // Get hive address from window.location (Hive UI is served BY the Hive)
    const hiveAddress = window.location.hostname
    const hivePort = '7835' // TODO: Get from config
    client = new sdk.HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)
  }
  return client
}

// TEAM-353: Worker type (local to this package)
export interface Worker {
  pid: number
  model: string
  device: string
}

/**
 * Hook for fetching model list from Hive
 *
 * TEAM-353: Migrated to TanStack Query + WASM SDK (job-based architecture)
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 * - Stale data management
 */
export function useModels() {
  const {
    data: models,
    isLoading: loading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['hive-models'],
    queryFn: async () => {
      const sdk = await ensureWasmInit()
      const client = await getClient() // TEAM-381: Lazy client initialization
      const hiveId = client.hiveId // TEAM-353: Get hive_id from client
      const op = sdk.OperationBuilder.modelList(hiveId)

      // TEAM-384: Capture data events from dual-channel SSE
      let modelsData: any = null

      // TEAM-381: Add timeout to prevent infinite hanging if backend doesn't respond
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(
          () => reject(new Error('Request timeout: Backend did not respond within 10 seconds. Is rbee-hive running?')),
          10000,
        )
      })

      const streamPromise = client.submitAndStream(op, (line: string) => {
        if (line === '[DONE]') return

        // TEAM-384: Check if this is a data event (starts with "event: data")
        // SSE format: "event: data\n{json}"
        if (line.startsWith('event: data')) {
          // Next line will be the JSON payload - we'll get it on next callback
          return
        }

        // TEAM-384: Parse data event payload
        if (line.startsWith('{') && line.includes('"action"')) {
          try {
            const dataEvent = JSON.parse(line)
            if (dataEvent.action === 'model_list' && dataEvent.payload) {
              modelsData = dataEvent.payload.models
            }
          } catch (e) {
            // Not a data event, ignore
          }
        }
      })

      await Promise.race([streamPromise, timeoutPromise])

      // TEAM-384: Return models from data event, or empty array
      return modelsData || []
    },
    staleTime: 30000, // Models change less frequently (30 seconds)
    retry: 2, // TEAM-381: Reduced from 3 to fail faster
    retryDelay: 1000, // TEAM-381: Fixed 1s delay instead of exponential
  })

  return {
    models: models || [],
    loading,
    error: error as Error | null,
    refetch,
  }
}

/**
 * Hook for fetching worker list from Hive
 *
 * TEAM-353: Migrated to TanStack Query + WASM SDK (job-based architecture)
 * - Automatic polling (refetchInterval)
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 */
export function useWorkers() {
  const {
    data: workers,
    isLoading: loading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['hive-workers'],
    queryFn: async () => {
      const sdk = await ensureWasmInit()
      const client = await getClient() // TEAM-381: Lazy client initialization
      const hiveId = client.hiveId // TEAM-353: Get hive_id from client
      const op = sdk.OperationBuilder.workerList(hiveId)
      const lines: string[] = []

      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
        }
      })

      // Parse JSON response from last line
      const lastLine = lines[lines.length - 1]
      return lastLine ? JSON.parse(lastLine) : []
    },
    staleTime: 5000, // Workers change frequently (5 seconds)
    refetchInterval: 2000, // Auto-refetch every 2 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return {
    workers: workers || [],
    loading,
    error: error as Error | null,
    refetch,
  }
}

export type {
  SpawnWorkerParams,
  UseHiveOperationsResult,
  WorkerType,
  WorkerTypeOption,
} from './hooks/useHiveOperations'
// Export operation hooks
export { useHiveOperations, WORKER_TYPE_OPTIONS, WORKER_TYPES } from './hooks/useHiveOperations'
export type { InstalledWorker } from './hooks/useInstalledWorkers'
// TEAM-382: Installed workers listing
export { useInstalledWorkers } from './hooks/useInstalledWorkers'
export type {
  DeleteModelParams,
  LoadModelParams,
  UnloadModelParams,
  UseModelOperationsResult,
} from './hooks/useModelOperations'
export { useModelOperations } from './hooks/useModelOperations'
export type { UseWorkerOperationsResult } from './hooks/useWorkerOperations'
// TEAM-378: Worker operations (install + spawn)
export { useWorkerOperations } from './hooks/useWorkerOperations'
