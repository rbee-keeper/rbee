// TEAM-382: React hook for listing installed worker binaries
// Pattern: Same as useModelOperations (TEAM-268)

import { HiveClient, init, OperationBuilder } from '@rbee/rbee-hive-sdk'
import { useQuery } from '@tanstack/react-query'

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
      console.log('[useInstalledWorkers] 🎬 Starting query...')

      try {
        console.log('[useInstalledWorkers] 🔧 Initializing WASM...')
        await ensureWasmInit()
        console.log('[useInstalledWorkers] ✓ WASM initialized')

        const hiveId = client.hiveId
        console.log('[useInstalledWorkers] 🏠 Hive ID:', hiveId)

        const op = OperationBuilder.workerListInstalled(hiveId)
        console.log('[useInstalledWorkers] 🔨 Operation built:', op)

        const lines: string[] = []

        console.log('[useInstalledWorkers] 📡 Submitting operation...')
        await client.submitAndStream(op, (line: string) => {
          if (line !== '[DONE]') {
            console.log('[useInstalledWorkers] 📨 SSE line:', line)
            lines.push(line)
          } else {
            console.log('[useInstalledWorkers] 🏁 SSE stream complete ([DONE] received)')
          }
        })

        console.log('[useInstalledWorkers] ✅ Stream complete! Total lines:', lines.length)
        console.log('[useInstalledWorkers] 📋 All lines:', lines)

        // TEAM-384: Narration format is:
        // Line 1: "job_router::handle_job worker_list_installed_json"
        // Line 2: "{\"workers\": [...]}"
        // So we need to find a line that contains JSON (has both { and })
        const jsonLine = lines.find((line) => {
          const trimmed = line.trim()
          return trimmed.includes('{') && trimmed.includes('}')
        })

        if (!jsonLine) {
          console.error('[useInstalledWorkers] ❌ No JSON found in lines:', lines)
          throw new Error(`No JSON response received from server. Got ${lines.length} lines but none contained JSON.`)
        }

        console.log('[useInstalledWorkers] 🔍 Found JSON line:', jsonLine)

        // Extract JSON from the line (it might have narration prefix)
        const jsonMatch = jsonLine.match(/(\{.*\})/)
        if (!jsonMatch) {
          console.error('[useInstalledWorkers] ❌ Could not extract JSON from line:', jsonLine)
          throw new Error('Could not parse JSON response - line does not contain valid JSON')
        }

        console.log('[useInstalledWorkers] 🔍 Extracted JSON:', jsonMatch[1])

        const response = JSON.parse(jsonMatch[1])
        console.log('[useInstalledWorkers] 📦 Parsed response:', response)

        const workers = response.workers || []
        console.log('[useInstalledWorkers] ✅ Returning', workers.length, 'workers:', workers)

        return workers
      } catch (error) {
        console.error('[useInstalledWorkers] ❌ Error:', error)
        console.error('[useInstalledWorkers] ❌ Error stack:', error instanceof Error ? error.stack : 'No stack')
        throw error
      }
    },
    refetchInterval: (query) => {
      // TEAM-384: Only auto-refetch if the last query was successful
      // Don't spam the backend if there's an error!
      return query.state.status === 'success' ? 10000 : false
    },
    retry: false, // TEAM-384: NO RETRIES - don't spam the backend!
    staleTime: 5000, // TEAM-384: Consider data fresh for 5 seconds
    refetchOnWindowFocus: false, // TEAM-384: Don't refetch when window regains focus
    refetchOnMount: true, // TEAM-384: Only refetch on mount
  })
}
