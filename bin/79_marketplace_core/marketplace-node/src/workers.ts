// TEAM-453: Worker catalog integration
// Fetches workers from gwc.rbee.dev (Global Worker Catalog)

const WORKER_CATALOG_URL = process.env.MARKETPLACE_API_URL || 'https://gwc.rbee.dev'

export interface WorkerCatalogEntry {
  id: string
  implementation: string
  workerType: string
  version: string
  platforms: string[]
  architectures: string[]
  name: string
  description: string
  license: string
  pkgbuildUrl: string
  buildSystem: string
  source: {
    type: string
    url: string
    branch: string
    path: string
  }
  build: {
    features: string[]
    profile: string
  }
  depends: string[]
  makedepends: string[]
  binaryName: string
  installPath: string
  supportedFormats: string[]
  maxContextLength: number
  supportsStreaming: boolean
  supportsBatching: boolean
}

export interface WorkersListResponse {
  workers: WorkerCatalogEntry[]
}

/**
 * List all available workers from the global worker catalog
 * 
 * Fetches from gwc.rbee.dev/workers
 * 
 * @returns Array of worker catalog entries
 * 
 * @example
 * ```typescript
 * const workers = await listWorkers()
 * console.log(workers) // [{ id: 'llm-worker-rbee-cpu', ... }]
 * ```
 */
export async function listWorkers(): Promise<WorkerCatalogEntry[]> {
  try {
    console.log(`[workers] Fetching from ${WORKER_CATALOG_URL}/workers`)
    
    const response = await fetch(`${WORKER_CATALOG_URL}/workers`, {
      headers: {
        'Accept': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Worker catalog API returned ${response.status}: ${response.statusText}`)
    }

    const data: WorkersListResponse = await response.json()
    
    console.log(`[workers] Fetched ${data.workers.length} workers`)
    
    return data.workers
  } catch (error) {
    console.error('[workers] Failed to fetch workers:', error)
    throw error
  }
}

/**
 * Get a specific worker by ID
 * 
 * @param workerId - Worker ID (e.g., 'llm-worker-rbee-cpu')
 * @returns Worker catalog entry or null if not found
 * 
 * @example
 * ```typescript
 * const worker = await getWorker('llm-worker-rbee-cpu')
 * console.log(worker.name) // 'LLM Worker (CPU)'
 * ```
 */
export async function getWorker(workerId: string): Promise<WorkerCatalogEntry | null> {
  try {
    console.log(`[workers] Fetching worker ${workerId}`)
    
    const response = await fetch(`${WORKER_CATALOG_URL}/workers/${workerId}`, {
      headers: {
        'Accept': 'application/json',
      },
    })

    if (response.status === 404) {
      return null
    }

    if (!response.ok) {
      throw new Error(`Worker catalog API returned ${response.status}: ${response.statusText}`)
    }

    const worker: WorkerCatalogEntry = await response.json()
    
    return worker
  } catch (error) {
    console.error(`[workers] Failed to fetch worker ${workerId}:`, error)
    throw error
  }
}
