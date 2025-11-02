// TEAM-378: Worker operations using TanStack Query mutations
// Handles worker installation and spawning
// TEAM-384: Returns raw SSE lines for app to handle (UI-agnostic)

'use client'

import { useMutation } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

// TEAM-378: Initialize WASM (same pattern as other hooks)
let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init()
    wasmInitialized = true
  }
}

// TEAM-378: Create client instance
const hiveAddress = window.location.hostname
const hivePort = '7835'
const client = new HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)

// TEAM-377: Worker types (matches Rust enum in worker-catalog/src/types.rs)
export type WorkerType = 'cpu' | 'cuda' | 'metal'

export const WORKER_TYPES: readonly WorkerType[] = ['cpu', 'cuda', 'metal'] as const

export interface WorkerTypeOption {
  value: WorkerType
  label: string
  description: string
}

export const WORKER_TYPE_OPTIONS: readonly WorkerTypeOption[] = [
  { value: 'cpu', label: 'CPU', description: 'CPU-based LLM worker' },
  { value: 'cuda', label: 'CUDA', description: 'NVIDIA GPU-based LLM worker' },
  { value: 'metal', label: 'Metal', description: 'Apple Metal GPU-based LLM worker (macOS)' },
] as const

export interface SpawnWorkerParams {
  modelId: string
  workerType?: WorkerType
  deviceId?: number
}

export interface UseWorkerOperationsOptions {
  /** Optional callback for SSE messages (for narration UI) */
  onSSEMessage?: (line: string) => void
}

export interface UseWorkerOperationsResult {
  installWorker: (workerId: string) => void
  spawnWorker: (params: SpawnWorkerParams) => void
  terminateWorker: (pid: number) => void
  isPending: boolean
  isSuccess: boolean
  isError: boolean
  error: Error | null
  reset: () => void
}

/**
 * Hook for Worker operations using TanStack Query mutations
 * 
 * TEAM-378: Handles worker installation and spawning
 * TEAM-384: Accepts optional onSSEMessage callback for narration UI
 * 
 * - installWorker: Download PKGBUILD, build, and install worker binary
 * - spawnWorker: Start a worker process with a model
 * - terminateWorker: Terminate a running worker process by PID
 * 
 * @param options - Optional configuration
 * @param options.onSSEMessage - Callback for SSE messages (for narration UI)
 * @returns Mutation functions and state
 * 
 * @example
 * ```tsx
 * const { installWorker, spawnWorker, isPending } = useWorkerOperations({
 *   onSSEMessage: (line) => {
 *     // Handle narration in app layer
 *     const parsed = parseNarrationLine(line)
 *     addToNarrationStore(parsed)
 *   }
 * })
 * 
 * // Install a worker binary
 * <button onClick={() => installWorker('llm-worker-rbee-cpu')}>
 *   Install Worker
 * </button>
 * ```
 */
export function useWorkerOperations(options?: UseWorkerOperationsOptions): UseWorkerOperationsResult {
  const { onSSEMessage } = options || {}
  
  // TEAM-378: Worker installation mutation
  const installMutation = useMutation<any, Error, string>({
    mutationFn: async (workerId: string) => {
      console.log('[useWorkerOperations] 🎬 Starting installation mutation for:', workerId)
      
      console.log('[useWorkerOperations] 🔧 Initializing WASM...')
      await ensureWasmInit()
      console.log('[useWorkerOperations] ✓ WASM initialized')
      
      const hiveId = client.hiveId
      console.log('[useWorkerOperations] 🏠 Hive ID:', hiveId)
      
      // TEAM-378: workerInstall(hive_id, worker_id)
      console.log('[useWorkerOperations] 🔨 Building WorkerInstall operation...')
      const op = OperationBuilder.workerInstall(hiveId, workerId)
      console.log('[useWorkerOperations] ✓ Operation built:', op)
      
      const lines: string[] = []
      
      console.log('[useWorkerOperations] 📡 Submitting operation and streaming SSE...')
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[useWorkerOperations] 📨 SSE message:', line)
          // TEAM-384: Call callback if provided (for narration UI in app layer)
          onSSEMessage?.(line)
        } else {
          console.log('[useWorkerOperations] 🏁 SSE stream complete ([DONE] received)')
        }
      })
      
      // TEAM-384: Check for errors in SSE stream before returning success
      // Error markers: ❌, ✗, "failed:", "Error details:", "ERROR:"
      // BUT ignore normal cargo output like "ERROR: Compiling X"
      const hasErrors = lines.some(line => {
        // Ignore normal cargo compilation progress
        if (line.includes('Compiling ') || line.includes('Downloading ') || line.includes('Building ')) {
          return false
        }
        
        return (
          line.includes('❌') ||
          line.includes('✗') ||
          line.includes('failed:') ||
          line.includes('Error details:') ||
          line.toLowerCase().includes('error:')
        )
      })
      
      if (hasErrors) {
        console.error('[useWorkerOperations] ❌ Errors detected in installation stream')
        
        // Extract the most relevant error message
        const errorLine = lines.find(line => 
          line.includes('failed:') || 
          line.includes('Error details:')
        )
        
        // Extract just the error message (remove ANSI codes and narration metadata)
        let errorMessage = errorLine || 'Installation failed - check logs'
        errorMessage = errorMessage
          .replace(/\x1b\[[0-9;]*m/g, '') // Remove ANSI escape codes
          .replace(/^.*?\s{2,}/, '')      // Remove narration prefix (module name + spaces)
          .trim()
        
        console.error('[useWorkerOperations] Error message:', errorMessage)
        throw new Error(errorMessage)
      }
      
      console.log('[useWorkerOperations] ✅ Installation complete! Total messages:', lines.length)
      return { success: true, workerId }
    },
    retry: 1,
    retryDelay: 1000,
  })

  // TEAM-377: Worker spawn mutation
  const spawnMutation = useMutation<any, Error, SpawnWorkerParams>({
    mutationFn: async ({ modelId, workerType = 'cuda', deviceId = 0 }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      // TEAM-377: workerSpawn(hive_id, model, worker_type, device_id)
      const op = OperationBuilder.workerSpawn(hiveId, modelId, workerType, deviceId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Worker spawn:', line)
        }
      })
      
      // Parse response from last line
      const lastLine = lines[lines.length - 1]
      return lastLine ? JSON.parse(lastLine) : null
    },
    retry: 1,
    retryDelay: 1000,
  })

  // TEAM-384: Worker terminate mutation
  const terminateMutation = useMutation<any, Error, number>({
    mutationFn: async (pid: number) => {
      console.log('[useWorkerOperations] 🛑 Terminating worker PID:', pid)
      
      await ensureWasmInit()
      const hiveId = client.hiveId
      
      // TEAM-384: workerDelete(hive_id, pid)
      const op = OperationBuilder.workerDelete(hiveId, pid)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[useWorkerOperations] Worker terminate:', line)
        }
      })
      
      console.log('[useWorkerOperations] ✓ Worker terminated')
      return { success: true, pid }
    },
    retry: 0, // Don't retry termination
    retryDelay: 0,
  })

  return {
    installWorker: installMutation.mutate,
    spawnWorker: spawnMutation.mutate,
    terminateWorker: terminateMutation.mutate,
    isPending: installMutation.isPending || spawnMutation.isPending || terminateMutation.isPending,
    isSuccess: installMutation.isSuccess || spawnMutation.isSuccess || terminateMutation.isSuccess,
    isError: installMutation.isError || spawnMutation.isError || terminateMutation.isError,
    error: installMutation.error || spawnMutation.error || terminateMutation.error,
    reset: () => {
      installMutation.reset()
      spawnMutation.reset()
      terminateMutation.reset()
    },
  }
}
