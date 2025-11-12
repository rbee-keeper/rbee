// TEAM-482: Global Worker Catalog (GWC) details API - Fetch single worker by ID

import type { MarketplaceModel } from '../common'
import type { GWCWorker } from './types'
import { convertGWCWorker } from './list'

/**
 * GWC API base URL
 */
const GWC_API_BASE = process.env.NEXT_PUBLIC_GWC_API_URL || 'https://gwc.rbee.dev'

/**
 * Fetch a single worker by ID (DETAILS API)
 *
 * @param workerId - Worker ID (string, e.g., "llm-worker-rbee")
 * @returns Normalized MarketplaceModel
 */
export async function fetchGWCWorker(workerId: string): Promise<MarketplaceModel> {
  const url = `${GWC_API_BASE}/workers/${workerId}`

  console.log('[GWC API] Fetching worker:', url)

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`GWC API error: ${response.status} ${response.statusText}`)
    }

    const worker: GWCWorker = await response.json()

    console.log(`[GWC API] Fetched worker: ${worker.name}`)

    // Convert to MarketplaceModel format
    return convertGWCWorker(worker)
  } catch (error) {
    console.error(`[GWC API] Error fetching worker ${workerId}:`, error)
    throw error
  }
}
