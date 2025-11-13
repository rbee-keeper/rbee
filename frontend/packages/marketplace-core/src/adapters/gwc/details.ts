// TEAM-482: Global Worker Catalog (GWC) details API - Fetch single worker by ID
// TEAM-501: Added fetchGWCWorkerReadme for README markdown

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

/**
 * Fetch worker README markdown
 * TEAM-501: Fetches raw README from worker's readmeUrl
 *
 * @param workerId - Worker ID (string, e.g., "llm-worker-rbee")
 * @returns README markdown string or null if not available
 */
export async function fetchGWCWorkerReadme(workerId: string): Promise<string | null> {
  try {
    // First fetch the worker to get the readmeUrl
    const url = `${GWC_API_BASE}/workers/${workerId}`
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      return null
    }

    const worker: GWCWorker = await response.json()

    if (!worker.readmeUrl) {
      return null
    }

    // Fetch the README from the URL
    console.log('[GWC API] Fetching README:', worker.readmeUrl)
    const readmeResponse = await fetch(worker.readmeUrl, {
      headers: {
        'Accept': 'text/plain, text/markdown, text/x-markdown, */*',
      },
    })

    if (!readmeResponse.ok) {
      return null
    }

    const readme = await readmeResponse.text()
    console.log(`[GWC API] Fetched README (${readme.length} chars)`)

    // TEAM-501: Check if we accidentally got HTML instead of markdown
    if (readme.trim().startsWith('<!DOCTYPE') || readme.trim().startsWith('<html')) {
      console.warn('[GWC API] Received HTML instead of markdown, returning null')
      return null
    }

    return readme
  } catch (error) {
    console.error(`[GWC API] Error fetching README for ${workerId}:`, error)
    return null
  }
}
