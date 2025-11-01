// TEAM-382: React hook for listing installed worker binaries
// Pattern: Same as useModelOperations (TEAM-268)

import { useQuery } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init()
    wasmInitialized = true
  }
}

const hiveAddress = window.location.hostname
const hivePort = '7835'
const client = new HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)

export interface InstalledWorker {
  id: string
  name: string
  worker_type: string
  platform: string
  version: string
  size: number
  path: string
  added_at: string
}

/**
 * Hook to list installed worker binaries from catalog
 * 
 * @returns Query result with array of installed workers
 * 
 * @example
 * ```tsx
 * const { data: workers, isLoading, error } = useInstalledWorkers()
 * ```
 */
export function useInstalledWorkers() {
  return useQuery<InstalledWorker[]>({
    queryKey: ['installed-workers'],
    queryFn: async () => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.workerListInstalled(hiveId)
      
      const lines: string[] = []
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
        }
      })
      
      // Find JSON response line (starts with "{")
      const jsonLine = lines.find(line => line.trim().startsWith('{'))
      if (!jsonLine) {
        throw new Error('No JSON response received from server')
      }
      
      const response = JSON.parse(jsonLine)
      return response.workers
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  })
}
